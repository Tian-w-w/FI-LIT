"""Command line interface for CPU-only preparation and validation tasks."""

from __future__ import annotations

import argparse
import json
from typing import Optional, Sequence

from fi_lit.config import ConfigError, dry_run_plan, load_config, validate_config
from fi_lit.evaluate import EvaluationError, evaluate_superni
from fi_lit.offline import AssetManifestError, check_assets
from fi_lit.superni import SuperNIError, build_manifest, build_train_dev_manifests


def _emit(payload: object) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fi-lit", description="FI-LIT offline experiment utilities")
    commands = parser.add_subparsers(dest="command", required=True)

    manifest = commands.add_parser("build-superni-manifest", help="Build a local JSONL manifest from SuperNI")
    manifest.add_argument("--superni-root", required=True)
    manifest.add_argument("--output", required=True)
    manifest.add_argument("--splits", nargs="+", default=["train"])
    manifest.add_argument("--max-instances-per-task", type=int)
    manifest.add_argument("--instances-per-task", type=int)
    manifest.add_argument("--instance-seed", type=int, default=42)

    train_dev = commands.add_parser("build-superni-train-dev", help="Make task-disjoint train/dev manifests from SuperNI train tasks")
    train_dev.add_argument("--superni-root", required=True)
    train_dev.add_argument("--train-output", required=True)
    train_dev.add_argument("--dev-output", required=True)
    train_dev.add_argument("--dev-task-count", type=int, default=50)
    train_dev.add_argument("--seed", type=int, default=42)
    train_dev.add_argument("--max-instances-per-task", type=int)
    train_dev.add_argument("--instances-per-task", type=int, help="Seeded random sample per task for a reported experiment")
    train_dev.add_argument("--instance-seed", type=int, default=42)

    config = commands.add_parser("validate-config", help="Validate QLoRA/DDP YAML without loading a model")
    config.add_argument("--config", required=True)
    config.add_argument("--show-plan", action="store_true")

    assets = commands.add_parser("check-assets", help="Validate local offline asset inventory")
    assets.add_argument("--manifest", required=True)

    evaluate = commands.add_parser("evaluate-superni", help="Generate and score a local SuperNI manifest with a PEFT adapter")
    evaluate.add_argument("--config", required=True)
    evaluate.add_argument("--adapter", required=True)
    evaluate.add_argument("--manifest", required=True)
    evaluate.add_argument("--predictions", required=True)
    evaluate.add_argument("--metrics", required=True)
    evaluate.add_argument("--batch-size", type=int, default=4)
    evaluate.add_argument("--max-new-tokens", type=int, default=128)
    evaluate.add_argument("--max-input-tokens", type=int, default=2048)
    evaluate.add_argument("--max-examples", type=int)
    evaluate.add_argument("--log-every", type=int, default=100)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "build-superni-manifest":
            _emit(build_manifest(args.superni_root, args.output, args.splits, args.max_instances_per_task, args.instances_per_task, args.instance_seed))
        elif args.command == "build-superni-train-dev":
            _emit(build_train_dev_manifests(args.superni_root, args.train_output, args.dev_output, args.dev_task_count, args.seed, args.max_instances_per_task, args.instances_per_task, args.instance_seed))
        elif args.command == "validate-config":
            config = load_config(args.config)
            validate_config(config)
            _emit(dry_run_plan(config) if args.show_plan else {"valid": True, "config": args.config})
        elif args.command == "check-assets":
            result = check_assets(args.manifest)
            _emit(result)
            return 0 if result["passed"] else 2
        elif args.command == "evaluate-superni":
            _emit(evaluate_superni(args.config, args.adapter, args.manifest, args.predictions, args.metrics, args.batch_size, args.max_new_tokens, args.max_input_tokens, args.max_examples, args.log_every))
    except (ConfigError, SuperNIError, AssetManifestError, EvaluationError, OSError, RuntimeError, json.JSONDecodeError) as exc:
        _emit({"valid": False, "error": str(exc)})
        return 2
    return 0
