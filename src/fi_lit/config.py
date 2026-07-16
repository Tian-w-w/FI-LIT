"""Configuration loading and CPU-only validation for QLoRA/DDP jobs."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

import yaml


class ConfigError(ValueError):
    """Raised when a FI-LIT experiment configuration is inconsistent."""


def load_config(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a YAML mapping without importing training or GPU dependencies."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ConfigError("Top-level configuration must be a YAML mapping.")
    return config


def _mapping(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key)
    if not isinstance(value, Mapping):
        raise ConfigError("'{}' must be a mapping.".format(key))
    return value


def _positive_int(value: Any, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ConfigError("'{}' must be a positive integer.".format(name))


def validate_config(config: Mapping[str, Any]) -> None:
    """Validate contracts that can fail before a costly distributed launch."""
    model = _mapping(config, "model")
    data = _mapping(config, "data")
    training = _mapping(config, "training")
    ddp = _mapping(config, "ddp")
    quantization = _mapping(model, "quantization")
    lora = _mapping(model, "lora")

    if not isinstance(model.get("name_or_path"), str) or not model["name_or_path"].strip():
        raise ConfigError("'model.name_or_path' must be a non-empty string.")
    if quantization.get("load_in_4bit") is not True:
        raise ConfigError("FI-LIT QLoRA runs require 'model.quantization.load_in_4bit: true'.")
    if quantization.get("quant_type") not in {"nf4", "fp4"}:
        raise ConfigError("'model.quantization.quant_type' must be 'nf4' or 'fp4'.")
    if quantization.get("compute_dtype") not in {"float16", "bfloat16"}:
        raise ConfigError("QLoRA compute_dtype must be float16 or bfloat16.")
    _positive_int(lora.get("r"), "model.lora.r")
    _positive_int(lora.get("alpha"), "model.lora.alpha")
    targets = lora.get("target_modules")
    if not isinstance(targets, list) or not targets or not all(isinstance(item, str) and item for item in targets):
        raise ConfigError("'model.lora.target_modules' must be a non-empty list of module names.")

    for key in ("train_manifest", "eval_manifest"):
        if not isinstance(data.get(key), str) or not data[key].strip():
            raise ConfigError("'data.{}' must be a non-empty path string.".format(key))
    _positive_int(data.get("max_seq_length"), "data.max_seq_length")
    for key in ("per_device_train_batch_size", "per_device_eval_batch_size", "gradient_accumulation_steps"):
        _positive_int(training.get(key), "training.{}".format(key))
    if not isinstance(training.get("learning_rate"), (int, float)) or training["learning_rate"] <= 0:
        raise ConfigError("'training.learning_rate' must be positive.")
    _positive_int(training.get("num_train_epochs"), "training.num_train_epochs")
    if not isinstance(training.get("output_dir"), str) or not training["output_dir"].strip():
        raise ConfigError("'training.output_dir' must be a non-empty path string.")

    if ddp.get("enabled") is not True:
        raise ConfigError("This baseline is defined for DDP; set 'ddp.enabled: true'.")
    if ddp.get("launcher") != "torchrun":
        raise ConfigError("'ddp.launcher' must be 'torchrun'.")
    _positive_int(ddp.get("expected_world_size"), "ddp.expected_world_size")


def distributed_runtime(config: Mapping[str, Any], env: Optional[Mapping[str, str]] = None) -> Dict[str, Any]:
    """Return DDP runtime metadata using environment values, without torch imports."""
    validate_config(config)
    environment = os.environ if env is None else env
    try:
        world_size = int(environment.get("WORLD_SIZE", "1"))
        rank = int(environment.get("RANK", "0"))
        local_rank = int(environment.get("LOCAL_RANK", "0"))
    except ValueError as exc:
        raise ConfigError("WORLD_SIZE, RANK, and LOCAL_RANK must be integers.") from exc
    if world_size < 1 or not 0 <= rank < world_size or local_rank < 0:
        raise ConfigError("Invalid distributed rank environment.")
    expected = config["ddp"]["expected_world_size"]
    if world_size not in {1, expected}:
        raise ConfigError("WORLD_SIZE={} does not match expected_world_size={}.".format(world_size, expected))
    return {
        "world_size": world_size,
        "rank": rank,
        "local_rank": local_rank,
        "is_distributed": world_size > 1,
        "device_map": {"": local_rank} if world_size > 1 else "auto",
    }


def dry_run_plan(config: Mapping[str, Any], env: Optional[Mapping[str, str]] = None) -> Dict[str, Any]:
    """Produce a serializable launch plan suitable for a CPU-only preflight."""
    runtime = distributed_runtime(config, env)
    return {
        "run_name": config.get("run_name", "unnamed"),
        "model_path": config["model"]["name_or_path"],
        "train_manifest": config["data"]["train_manifest"],
        "eval_manifest": config["data"]["eval_manifest"],
        "output_dir": config["training"]["output_dir"],
        "quantization": config["model"]["quantization"],
        "lora_rank": config["model"]["lora"]["r"],
        "runtime": runtime,
    }

