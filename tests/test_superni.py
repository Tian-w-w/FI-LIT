from __future__ import annotations

import json

from fi_lit.superni import build_manifest, build_train_dev_manifests


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


def test_build_train_dev_from_current_superni_layout_without_dev_file(tmp_path) -> None:
    root = tmp_path / "superni"
    (root / "tasks").mkdir(parents=True)
    split_dir = root / "splits"
    split_dir.mkdir(parents=True)
    (split_dir / "train_tasks.txt").write_text("task001.json\ntask075_squad1.1_answer_generation\ntask003\n", encoding="utf-8")
    (split_dir / "test_tasks.txt").write_text("task004\n", encoding="utf-8")
    (split_dir / "excluded_tasks.txt").write_text("task005\n", encoding="utf-8")
    _write_task(root, "task001", "classification", ["one", "two"])
    _write_task(root, "task075_squad1.1_answer_generation", "generation", ["three"])
    _write_task(root, "task003", "generation", ["four"])
    _write_task(root, "task004", "test-only", ["five"])

    train_output = tmp_path / "manifests" / "superni-train.jsonl"
    dev_output = tmp_path / "manifests" / "superni-dev.jsonl"
    summary = build_train_dev_manifests(root, train_output, dev_output, dev_task_count=1, seed=7)
    train_rows = [json.loads(line) for line in train_output.read_text(encoding="utf-8").splitlines()]
    dev_rows = [json.loads(line) for line in dev_output.read_text(encoding="utf-8").splitlines()]

    assert summary["test_split_used"] is False
    assert summary["train"]["tasks"] == 2
    assert summary["dev"]["tasks"] == 1
    assert {row["task_id"] for row in train_rows}.isdisjoint({row["task_id"] for row in dev_rows})
    assert {row["task_id"] for row in train_rows + dev_rows} == {"task001", "task075_squad1.1_answer_generation", "task003"}
    assert all(row["task_id"] != "task004" for row in train_rows + dev_rows)


def test_build_manifest_supports_legacy_split_layout(tmp_path) -> None:
    root = tmp_path / "superni"
    (root / "tasks").mkdir(parents=True)
    split_dir = root / "task_splits" / "default"
    split_dir.mkdir(parents=True)
    (split_dir / "train_tasks.txt").write_text("task001\n", encoding="utf-8")
    _write_task(root, "task001", "classification", ["one", "two"])

    output = tmp_path / "superni.jsonl"
    summary = build_manifest(root, output, ["train"])
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

    assert summary["examples"] == 2
    assert rows[0]["id"] == "task001:0"
    assert rows[0]["references"] == ["ONE", "One"]


def test_max_instances_limits_each_task_with_spilts_typo_layout(tmp_path) -> None:
    root = tmp_path / "superni"
    (root / "tasks").mkdir(parents=True)
    split_dir = root / "spilts"
    split_dir.mkdir(parents=True)
    (split_dir / "train_tasks.txt").write_text("task001\n", encoding="utf-8")
    _write_task(root, "task001", "classification", ["one", "two"])
    summary = build_manifest(root, tmp_path / "out.jsonl", ["train"], max_instances_per_task=1)
    assert summary["examples"] == 1
