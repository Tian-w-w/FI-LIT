"""Build local JSONL manifests from a locally available SuperNI release."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Union


class SuperNIError(ValueError):
    """Raised for an unexpected local SuperNI directory layout or task file."""


def _task_id(value: str) -> str:
    return Path(value.strip()).stem


def load_split_tasks(root: Union[str, Path], split: str) -> List[str]:
    """Load task ids from task_splits/default/<split>_tasks.txt."""
    split_path = Path(root) / "task_splits" / "default" / "{}_tasks.txt".format(split)
    if not split_path.is_file():
        raise SuperNIError("SuperNI split file not found: {}".format(split_path))
    task_ids = [_task_id(line) for line in split_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not task_ids:
        raise SuperNIError("SuperNI split file is empty: {}".format(split_path))
    return task_ids


def _as_text_list(value: Any) -> List[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return value
    raise SuperNIError("Each SuperNI instance Output must be a string or list of strings.")


def _task_records(root: Path, split: str, max_instances: Optional[int]) -> Iterable[Dict[str, Any]]:
    for task_id in load_split_tasks(root, split):
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
                "split": split,
                "categories": categories,
                "definition": definitions,
                "input": instance["input"],
                "references": _as_text_list(instance.get("output")),
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
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    counts: Counter = Counter()
    categories: Counter = Counter()
    task_ids = set()
    with target.open("w", encoding="utf-8") as handle:
        for split in splits:
            for record in _task_records(source_root, split, max_instances_per_task):
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                counts[split] += 1
                task_ids.add(record["task_id"])
                categories.update(record["categories"])
    return {
        "manifest_path": str(target),
        "examples": sum(counts.values()),
        "tasks": len(task_ids),
        "examples_by_split": dict(sorted(counts.items())),
        "categories": dict(sorted(categories.items())),
    }

