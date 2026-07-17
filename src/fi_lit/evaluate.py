"""Offline generation and lexical scoring for SuperNI manifests."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from fi_lit.config import ConfigError, load_config, validate_config


class EvaluationError(ValueError):
    """Raised when evaluation inputs or generation settings are invalid."""


def format_generation_prompt(record: Mapping[str, Any]) -> str:
    """Use the training serialization without appending the reference output."""
    definition = "\n".join(record.get("definition", []))
    return "Definition:\n{}\n\nInput:\n{}\n\nOutput:\n".format(definition, record["input"])


def normalize_text(value: str) -> str:
    """Case-fold and normalize whitespace without deleting meaningful symbols."""
    return " ".join(value.casefold().split())


def _lcs_length(left: Sequence[str], right: Sequence[str]) -> int:
    previous = [0] * (len(right) + 1)
    for left_item in left:
        current = [0]
        for index, right_item in enumerate(right, start=1):
            current.append(previous[index - 1] + 1 if left_item == right_item else max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def rouge_l_f1(prediction: str, reference: str) -> float:
    """Compute whitespace-token ROUGE-L F1 without an external metric package."""
    predicted_tokens = normalize_text(prediction).split()
    reference_tokens = normalize_text(reference).split()
    if not predicted_tokens or not reference_tokens:
        return float(predicted_tokens == reference_tokens)
    overlap = _lcs_length(predicted_tokens, reference_tokens)
    precision = overlap / len(predicted_tokens)
    recall = overlap / len(reference_tokens)
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def score_prediction(prediction: str, references: Sequence[str]) -> Dict[str, float]:
    """Score against the best of a task instance's accepted references."""
    if not references:
        raise EvaluationError("Each evaluation row requires at least one reference output.")
    normalized_prediction = normalize_text(prediction)
    normalized_references = [normalize_text(reference) for reference in references]
    return {
        "exact_match": float(normalized_prediction in normalized_references),
        "rouge_l": max(rouge_l_f1(prediction, reference) for reference in references),
    }


def aggregate_scores(scored_rows: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    """Return instance-micro and task-macro metrics from scored prediction rows."""
    rows = list(scored_rows)
    if not rows:
        raise EvaluationError("No scored predictions were provided.")
    task_metrics: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        task_metrics[str(row["task_id"])].append(row)

    def average(items: Sequence[Mapping[str, Any]], metric: str) -> float:
        return sum(float(item[metric]) for item in items) / len(items)

    per_task = {
        task_id: {
            "examples": len(items),
            "exact_match": average(items, "exact_match"),
            "rouge_l": average(items, "rouge_l"),
        }
        for task_id, items in sorted(task_metrics.items())
    }
    return {
        "examples": len(rows),
        "tasks": len(per_task),
        "micro": {
            "exact_match": average(rows, "exact_match"),
            "rouge_l": average(rows, "rouge_l"),
        },
        "macro_by_task": {
            "exact_match": sum(item["exact_match"] for item in per_task.values()) / len(per_task),
            "rouge_l": sum(item["rouge_l"] for item in per_task.values()) / len(per_task),
        },
        "per_task": per_task,
    }


def _read_manifest(path: Path, limit: Optional[int]) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if limit is not None and index >= limit:
                break
            record = json.loads(line)
            if not isinstance(record, dict) or not isinstance(record.get("references"), list):
                raise EvaluationError("Invalid manifest row {} in {}".format(index + 1, path))
            yield record


def _batches(items: Iterable[Dict[str, Any]], batch_size: int) -> Iterable[List[Dict[str, Any]]]:
    batch: List[Dict[str, Any]] = []
    for item in items:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _load_model(config: Mapping[str, Any], adapter_path: Path) -> Tuple[Any, Any]:
    """Load a local 4-bit base model and a local PEFT adapter for inference."""
    if not adapter_path.is_dir():
        raise EvaluationError("Adapter checkpoint does not exist: {}".format(adapter_path))
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as exc:
        raise RuntimeError("Evaluation requires the private offline [train] dependencies.") from exc

    model_config = config["model"]
    quant = model_config["quantization"]
    model_path = Path(model_config["name_or_path"])
    if not model_path.is_dir():
        raise EvaluationError("Base model directory does not exist: {}".format(model_path))
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True, trust_remote_code=model_config["trust_remote_code"])
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=quant["quant_type"],
        bnb_4bit_use_double_quant=quant["double_quant"],
        bnb_4bit_compute_dtype=getattr(torch, quant["compute_dtype"]),
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        local_files_only=True,
        trust_remote_code=model_config["trust_remote_code"],
        quantization_config=quantization,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(model, str(adapter_path), local_files_only=True)
    model.config.use_cache = True
    model.eval()
    return tokenizer, model


def evaluate_superni(
    config_path: Union[str, Path],
    adapter_path: Union[str, Path],
    manifest_path: Union[str, Path],
    predictions_path: Union[str, Path],
    metrics_path: Union[str, Path],
    batch_size: int = 4,
    max_new_tokens: int = 128,
    max_input_tokens: int = 2048,
    max_examples: Optional[int] = None,
    log_every: int = 100,
) -> Dict[str, Any]:
    """Generate deterministic outputs and write auditable predictions and metrics."""
    if batch_size < 1 or max_new_tokens < 1 or max_input_tokens < 1 or log_every < 1:
        raise EvaluationError("Batch size, token limits, and log interval must be positive.")
    if max_examples is not None and max_examples < 1:
        raise EvaluationError("max_examples must be positive when set.")
    config = load_config(config_path)
    validate_config(config)
    source = Path(manifest_path)
    if not source.is_file():
        raise EvaluationError("Evaluation manifest does not exist: {}".format(source))
    predictions = Path(predictions_path)
    metrics = Path(metrics_path)
    predictions.parent.mkdir(parents=True, exist_ok=True)
    metrics.parent.mkdir(parents=True, exist_ok=True)
    tokenizer, model = _load_model(config, Path(adapter_path))

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Evaluation requires torch.") from exc

    scored_rows: List[Dict[str, Any]] = []
    generated = 0
    with predictions.open("w", encoding="utf-8") as handle:
        for batch in _batches(_read_manifest(source, max_examples), batch_size):
            prompts = [format_generation_prompt(record) for record in batch]
            encoded = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_input_tokens,
            )
            device = next(model.parameters()).device
            encoded = {key: value.to(device) for key, value in encoded.items()}
            input_length = encoded["input_ids"].shape[1]
            with torch.inference_mode():
                output_ids = model.generate(
                    **encoded,
                    do_sample=False,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            completions = tokenizer.batch_decode(output_ids[:, input_length:], skip_special_tokens=True)
            for record, prediction in zip(batch, completions):
                scores = score_prediction(prediction, record["references"])
                row = {
                    "id": record["id"],
                    "task_id": record["task_id"],
                    "prediction": prediction,
                    "references": record["references"],
                    **scores,
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                scored_rows.append(row)
            generated += len(batch)
            if generated % log_every < len(batch):
                print("generated {} examples".format(generated), flush=True)

    summary = aggregate_scores(scored_rows)
    summary.update(
        {
            "config_path": str(config_path),
            "adapter_path": str(adapter_path),
            "manifest_path": str(source),
            "predictions_path": str(predictions),
            "generation": {
                "decoding": "greedy",
                "batch_size": batch_size,
                "max_new_tokens": max_new_tokens,
                "max_input_tokens": max_input_tokens,
            },
            "metric_note": "Exact Match is case/whitespace-normalized. ROUGE-L is whitespace-token F1 against the best reference.",
        }
    )
    metrics.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return summary
