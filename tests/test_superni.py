from __future__ import annotations

import json

from fi_lit.superni import build_manifest


def _write_task(root, task_id: str, category: str, inputs: list) -> None:
    task = {
        "Categories": [category],
        "Definition": ["Answer the input."],
        "Instances": [
            {"input": item, "output": [item.upper(), item.title()]}
            for item in inputs
        ],
    }
    (root / "tasks" / (task_id + ".json")).write_text(json.dumps(task), encoding="utf-8")


def test_build_manifest_from_official_style_layout(tmp_path) -> None:
    root = tmp_path / "superni"
    (root / "tasks").mkdir(parents=True)
    split_dir = root / "task_splits" / "default"
    split_dir.mkdir(parents=True)
    (split_dir / "train_tasks.txt").write_text("task001.json\n", encoding="utf-8")
    (split_dir / "dev_tasks.txt").write_text("task002\n", encoding="utf-8")
    _write_task(root, "task001", "classification", ["one", "two"])
    _write_task(root, "task002", "generation", ["three"])

    output = tmp_path / "manifests" / "superni.jsonl"
    summary = build_manifest(root, output, ["train", "dev"])
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

    assert summary["examples"] == 3
    assert summary["examples_by_split"] == {"dev": 1, "train": 2}
    assert rows[0]["id"] == "task001:0"
    assert rows[0]["references"] == ["ONE", "One"]
    assert rows[-1]["split"] == "dev"


def test_max_instances_limits_each_task(tmp_path) -> None:
    root = tmp_path / "superni"
    (root / "tasks").mkdir(parents=True)
    split_dir = root / "task_splits" / "default"
    split_dir.mkdir(parents=True)
    (split_dir / "train_tasks.txt").write_text("task001\n", encoding="utf-8")
    _write_task(root, "task001", "classification", ["one", "two"])
    summary = build_manifest(root, tmp_path / "out.jsonl", ["train"], max_instances_per_task=1)
    assert summary["examples"] == 1

