"""Build local JSONL manifests from a locally available SuperNI release."""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Union


class SuperNIError(ValueError):
    """Raised for an unexpected local SuperNI directory layout or task file."""


def _task_id(value: str) -> str:
    """Normalize an optional .json suffix without truncating dots in task ids."""
    name = Path(value.strip()).name
    return name[:-5] if name.endswith(".json") else name


def load_split_tasks(root: Union[str, Path], split: str) -> List[str]:
    """Load task ids from current or legacy SuperNI split layouts."""
    source_root = Path(root)
    filename = "{}_tasks.txt".format(split)
    candidates = [
        source_root / "splits" / filename,
        source_root / "splits" / "default" / filename,
        source_root / "spilts" / filename,
        source_root / "task_splits" / "default" / filename,
    ]
    split_path = next((path for path in candidates if path.is_file()), None)
    if split_path is None:
        searched = ", ".join(str(path) for path in candidates)
        raise SuperNIError("SuperNI split file not found. Searched: {}".format(searched))
    task_ids = [_task_id(line.split()[0]) for line in split_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not task_ids:
        raise SuperNIError("SuperNI split file is empty: {}".format(split_path))
    return task_ids


def _as_text_list(value: Any) -> List[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return value
    raise SuperNIError("Each SuperNI instance Output must be a string or list of strings.")


def _records_for_task_ids(
    root: Path,
    task_ids: Sequence[str],
    split_label: str,
    max_instances: Optional[int],
) -> Iterable[Dict[str, Any]]:
    for task_id in task_ids:
        task_path = root / "tasks" / "{}.json".format(task_id)
        if not task_path.is_file():
            raise SuperNIError("Task listed in split is missing: {}".format(task_path))
        with task_path.open("r", encoding="utf-8") as handle:
            task = json.load(handle)
        if not isinstance(task, Mapping) or not isinstance(task.get("Instances"), list):
            raise SuperNIError("Invalid SuperNI task file: {}".format(task_path))
        definitions = task.get("Definition", [])
        if isinstance(definitions, str):
            definitions = [definitions]
        if not isinstance(definitions, list) or not all(isinstance(item, str) for item in definitions):
            raise SuperNIError("Invalid Definition in {}".format(task_path))
        categories = task.get("Categories", [])
        if not isinstance(categories, list):
            categories = []
        instances = task["Instances"] if max_instances is None else task["Instances"][:max_instances]
        for index, instance in enumerate(instances):
            if not isinstance(instance, Mapping) or not isinstance(instance.get("input"), str):
                raise SuperNIError("Invalid instance {} in {}".format(index, task_path))
            yield {
                "id": "{}:{}".format(task_id, index),
                "task_id": task_id,
                "split": split_label,
                "categories": categories,
                "definition": definitions,
                "input": instance["input"],
                "references": _as_text_list(instance.get("output")),
            }


def _write_manifest(
    root: Path,
    output_path: Union[str, Path],
    partitions: Mapping[str, Sequence[str]],
    max_instances_per_task: Optional[int],
) -> Dict[str, Any]:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    counts: Counter = Counter()
    categories: Counter = Counter()
    task_ids = set()
    with target.open("w", encoding="utf-8") as handle:
        for split_label, selected_task_ids in partitions.items():
            for record in _records_for_task_ids(root, selected_task_ids, split_label, max_instances_per_task):
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                counts[split_label] += 1
                task_ids.add(record["task_id"])
                categories.update(record["categories"])
    return {
        "manifest_path": str(target),
        "examples": sum(counts.values()),
        "tasks": len(task_ids),
        "examples_by_split": dict(sorted(counts.items())),
        "categories": dict(sorted(categories.items())),
    }


def build_manifest(
    root: Union[str, Path],
    output_path: Union[str, Path],
    splits: Sequence[str],
    max_instances_per_task: Optional[int] = None,
) -> Dict[str, Any]:
    """Write an example-level JSONL manifest and return a compact summary.

    The output is derived from raw examples and must remain outside version control.
    """
    if not splits:
        raise SuperNIError("At least one split is required.")
    if max_instances_per_task is not None and max_instances_per_task < 1:
        raise SuperNIError("max_instances_per_task must be positive when set.")
    source_root = Path(root)
    if not source_root.is_dir():
        raise SuperNIError("SuperNI root does not exist: {}".format(source_root))
    partitions = {split: load_split_tasks(source_root, split) for split in splits}
    return _write_manifest(source_root, output_path, partitions, max_instances_per_task)


def build_train_dev_manifests(
    root: Union[str, Path],
    train_output_path: Union[str, Path],
    dev_output_path: Union[str, Path],
    dev_task_count: int = 50,
    seed: int = 42,
    max_instances_per_task: Optional[int] = None,
) -> Dict[str, Any]:
    """Create task-disjoint train/dev manifests from the official train split.

    SuperNI releases without a dev_tasks.txt file should use this function. The
    official test split is never read, avoiding test-set leakage during tuning.
    """
    source_root = Path(root)
    if not source_root.is_dir():
        raise SuperNIError("SuperNI root does not exist: {}".format(source_root))
    if dev_task_count < 1:
        raise SuperNIError("dev_task_count must be positive.")
    train_tasks = load_split_tasks(source_root, "train")
    if len(set(train_tasks)) != len(train_tasks):
        raise SuperNIError("The official train split contains duplicate task ids.")
    if dev_task_count >= len(train_tasks):
        raise SuperNIError("dev_task_count must be smaller than the number of train tasks ({}).".format(len(train_tasks)))
    if Path(train_output_path).resolve() == Path(dev_output_path).resolve():
        raise SuperNIError("Train and dev output paths must differ.")
    held_out = set(random.Random(seed).sample(train_tasks, dev_task_count))
    train_partition = [task_id for task_id in train_tasks if task_id not in held_out]
    dev_partition = [task_id for task_id in train_tasks if task_id in held_out]
    train_summary = _write_manifest(source_root, train_output_path, {"train": train_partition}, max_instances_per_task)
    dev_summary = _write_manifest(source_root, dev_output_path, {"dev": dev_partition}, max_instances_per_task)
    return {
        "source_split": "train",
        "seed": seed,
        "dev_task_count": dev_task_count,
        "train": train_summary,
        "dev": dev_summary,
        "test_split_used": False,
    }
