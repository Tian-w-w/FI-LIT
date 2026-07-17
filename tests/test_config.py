from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from fi_lit.config import ConfigError, distributed_runtime, dry_run_plan, load_config, validate_config


CONFIG_PATH = Path(__file__).parents[1] / "configs" / "qlora_ddp_superni.yaml"


def test_default_config_validates_without_training_dependencies() -> None:
    config = load_config(CONFIG_PATH)
    validate_config(config)
    plan = dry_run_plan(config, {"WORLD_SIZE": "1", "RANK": "0", "LOCAL_RANK": "0"})
    assert plan["runtime"]["device_map"] == "auto"
    assert plan["quantization"]["load_in_4bit"] is True
    assert plan["max_steps"] == -1


def test_distributed_runtime_uses_local_rank() -> None:
    config = load_config(CONFIG_PATH)
    runtime = distributed_runtime(config, {"WORLD_SIZE": "2", "RANK": "1", "LOCAL_RANK": "1"})
    assert runtime["is_distributed"] is True
    assert runtime["device_map"] == {"": 1}


def test_invalid_qlora_flag_is_rejected() -> None:
    config = deepcopy(load_config(CONFIG_PATH))
    config["model"]["quantization"]["load_in_4bit"] = False
    with pytest.raises(ConfigError, match="load_in_4bit"):
        validate_config(config)


def test_unexpected_world_size_is_rejected() -> None:
    config = load_config(CONFIG_PATH)
    with pytest.raises(ConfigError, match="expected_world_size"):
        distributed_runtime(config, {"WORLD_SIZE": "3", "RANK": "0", "LOCAL_RANK": "0"})


def test_invalid_max_steps_is_rejected() -> None:
    config = deepcopy(load_config(CONFIG_PATH))
    config["training"]["max_steps"] = 0
    with pytest.raises(ConfigError, match="max_steps"):
        validate_config(config)
