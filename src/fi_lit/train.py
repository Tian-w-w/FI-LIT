"""Opt-in QLoRA training entry point; dry-run mode has no GPU/model dependency."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from fi_lit.config import ConfigError, distributed_runtime, dry_run_plan, load_config, validate_config


def _format_user_content(record: Mapping[str, Any]) -> str:
    definition = "\n".join(record.get("definition", []))
    return "Definition:\n{}\n\nInput:\n{}".format(definition, record["input"])


def _tokenize_completion_only(record: Mapping[str, Any], tokenizer: Any, max_seq_length: int) -> Dict[str, List[int]]:
    """Render Qwen chat messages and mask the user prompt from causal-LM loss."""
    user_content = _format_user_content(record)
    target = record.get("references", [""])[0]
    if not isinstance(target, str):
        raise ConfigError("SuperNI reference outputs must be strings.")
    prompt_ids = list(tokenizer.apply_chat_template(
        [{"role": "user", "content": user_content}],
        tokenize=True,
        add_generation_prompt=True,
    ))
    full_ids = list(tokenizer.apply_chat_template(
        [{"role": "user", "content": user_content}, {"role": "assistant", "content": target}],
        tokenize=True,
        add_generation_prompt=False,
    ))
    if full_ids[:len(prompt_ids)] != prompt_ids:
        raise ConfigError("Tokenizer chat template does not preserve the generation prompt as a prefix.")
    completion_ids = full_ids[len(prompt_ids):]
    if not completion_ids:
        raise ConfigError("Tokenizer chat template produced an empty assistant completion.")
    if len(completion_ids) >= max_seq_length:
        completion_ids = completion_ids[:max_seq_length]
        prompt_ids = []
    else:
        prompt_ids = prompt_ids[:max_seq_length - len(completion_ids)]
    input_ids = prompt_ids + completion_ids
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": [-100] * len(prompt_ids) + completion_ids,
    }


def _cleanup_distributed_process_group(torch_module: Any) -> None:
    """Release the NCCL process group on every rank after a distributed run."""
    distributed = getattr(torch_module, "distributed", None)
    if distributed is not None and distributed.is_available() and distributed.is_initialized():
        distributed.destroy_process_group()


def run_training(config: Dict[str, Any]) -> None:
    """Launch a job after local assets and dependencies are available.

    Imports are intentionally local so config checks remain usable on a laptop.
    """
    validate_config(config)
    runtime = distributed_runtime(config)
    train_path = Path(config["data"]["train_manifest"])
    eval_path = Path(config["data"]["eval_manifest"])
    if not train_path.is_file() or not eval_path.is_file():
        raise ConfigError("Training and evaluation manifests must exist before launching a job.")
    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DataCollatorForSeq2Seq, Trainer, TrainingArguments
    except ImportError as exc:
        raise RuntimeError("Install the private offline wheelhouse with the [train] extra before a real run.") from exc

    quant = config["model"]["quantization"]
    dtype = getattr(torch, quant["compute_dtype"])
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=quant["quant_type"],
        bnb_4bit_use_double_quant=quant["double_quant"],
        bnb_4bit_compute_dtype=dtype,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["name_or_path"],
        local_files_only=True,
        trust_remote_code=config["model"]["trust_remote_code"],
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        config["model"]["name_or_path"],
        local_files_only=True,
        trust_remote_code=config["model"]["trust_remote_code"],
        quantization_config=bnb,
        device_map=runtime["device_map"],
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=config["training"]["gradient_checkpointing"],
    )
    lora = config["model"]["lora"]
    model = get_peft_model(
        model,
        LoraConfig(
            r=lora["r"],
            lora_alpha=lora["alpha"],
            lora_dropout=lora["dropout"],
            bias=lora["bias"],
            target_modules=lora["target_modules"],
            task_type="CAUSAL_LM",
        ),
    )
    dataset = load_dataset("json", data_files={"train": str(train_path), "eval": str(eval_path)})

    def tokenize(record: Dict[str, Any]) -> Dict[str, Any]:
        return _tokenize_completion_only(record, tokenizer, config["data"]["max_seq_length"])

    tokenized = dataset.map(tokenize, remove_columns=dataset["train"].column_names)
    train_args = TrainingArguments(
        output_dir=config["training"]["output_dir"],
        per_device_train_batch_size=config["training"]["per_device_train_batch_size"],
        per_device_eval_batch_size=config["training"]["per_device_eval_batch_size"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        learning_rate=config["training"]["learning_rate"],
        num_train_epochs=config["training"]["num_train_epochs"],
        max_steps=config["training"].get("max_steps", -1),
        warmup_ratio=config["training"]["warmup_ratio"],
        logging_steps=config["training"]["logging_steps"],
        save_steps=config["training"]["save_steps"],
        eval_strategy="steps",
        eval_steps=config["training"]["eval_steps"],
        bf16=config["training"]["bf16"],
        tf32=config["training"]["tf32"],
        ddp_find_unused_parameters=config["ddp"]["find_unused_parameters"],
        report_to=config["training"]["report_to"],
        seed=config.get("seed", 42),
    )
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["eval"],
        data_collator=DataCollatorForSeq2Seq(tokenizer, label_pad_token_id=-100, pad_to_multiple_of=8),
    )
    try:
        trainer.train()
    finally:
        _cleanup_distributed_process_group(torch)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="FI-LIT QLoRA/DDP training")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true", help="Validate and print plan; never load models or torch")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        if args.dry_run:
            print(json.dumps(dry_run_plan(config), ensure_ascii=False, indent=2, sort_keys=True))
            return 0
        run_training(config)
        return 0
    except (ConfigError, OSError, ValueError, RuntimeError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
