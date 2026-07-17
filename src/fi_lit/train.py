"""Opt-in QLoRA training entry point; dry-run mode has no GPU/model dependency."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from fi_lit.config import ConfigError, distributed_runtime, dry_run_plan, load_config, validate_config


def _format_record(record: Dict[str, Any]) -> str:
    definition = "\n".join(record.get("definition", []))
    output = record.get("references", [""])[0]
    return "Definition:\n{}\n\nInput:\n{}\n\nOutput:\n{}".format(definition, record["input"], output)


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
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DataCollatorForLanguageModeling, Trainer, TrainingArguments
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
        return tokenizer(
            _format_record(record),
            truncation=True,
            max_length=config["data"]["max_seq_length"],
        )

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
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    trainer.train()


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
