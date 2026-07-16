# FI-LIT

FI-LIT is an offline-first research scaffold for reproducible instruction-tuning experiments. This first milestone provides a SuperNI manifest builder, a QLoRA + DDP training configuration, validation utilities, CI, and server hand-off documentation.

It deliberately contains no model weights, raw datasets, generated manifests, experiment outputs, or GPU runs.

## Quick validation (no model required)

    python -m pip install -e ".[dev]"
    pytest
    python -m fi_lit validate-config --config configs/qlora_ddp_superni.yaml
    python -m fi_lit check-assets --manifest assets/offline-assets.example.json

The last command is expected to report missing placeholder paths until it is filled in on the offline server. See docs/offline_execution.md for the complete transfer and execution procedure, and docs/superni_manifest.md for the manifest format.

## Layout

    configs/       Reproducible QLoRA/DDP configuration
    src/fi_lit/    Manifest builder, configuration validation, and train entry point
    assets/        Versioned asset-manifest templates only
    docs/          Offline transfer and execution instructions
    tests/         CPU-only unit tests using synthetic fixtures

## Current scope

The training entry point supports a --dry-run planning mode without importing model libraries or accessing a GPU. A real run is intentionally opt-in and requires local model, data, and wheelhouse paths supplied by the operator.
