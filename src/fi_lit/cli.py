"""Command line interface for CPU-only preparation and validation tasks."""

from __future__ import annotations

import argparse
import json
from typing import Optional, Sequence

from fi_lit.config import ConfigError, dry_run_plan, load_config, validate_config
from fi_lit.offline import AssetManifestError, check_assets
from fi_lit.superni import SuperNIError, build_manifest


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

    config = commands.add_parser("validate-config", help="Validate QLoRA/DDP YAML without loading a model")
    config.add_argument("--config", required=True)
    config.add_argument("--show-plan", action="store_true")

    assets = commands.add_parser("check-assets", help="Validate local offline asset inventory")
    assets.add_argument("--manifest", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "build-superni-manifest":
            _emit(build_manifest(args.superni_root, args.output, args.splits, args.max_instances_per_task))
        elif args.command == "validate-config":
            config = load_config(args.config)
            validate_config(config)
            _emit(dry_run_plan(config) if args.show_plan else {"valid": True, "config": args.config})
        elif args.command == "check-assets":
            result = check_assets(args.manifest)
            _emit(result)
            return 0 if result["passed"] else 2
    except (ConfigError, SuperNIError, AssetManifestError, OSError, json.JSONDecodeError) as exc:
        _emit({"valid": False, "error": str(exc)})
        return 2
    return 0

