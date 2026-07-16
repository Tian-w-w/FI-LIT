# Offline Linux server execution

These instructions assume a disconnected Linux server with two GPUs. They do not download models, datasets, or packages at any stage.

## 1. Prepare and transfer assets

On a controlled staging machine, collect the already-approved model snapshot, SuperNI release, and Python wheelhouse. Record checksums in a private copy of assets/offline-assets.example.json; do not commit that copy. Transfer the Git checkout and those assets through the approved channel into a layout such as:

    /opt/fi-lit/
      repo/                 # this Git checkout
      models/base-model/    # existing local Hugging Face-format snapshot
      datasets/natural-instructions/
      wheelhouse/           # prebuilt compatible wheels
      manifests/            # generated locally on the server
      outputs/              # ignored experiment outputs

Update configs/qlora_ddp_superni.yaml with those absolute paths. Confirm that the exact torch, CUDA-compatible wheels, transformers, peft, datasets, accelerate, and bitsandbytes packages are already present in the wheelhouse.

## 2. Preflight without a GPU run

    cd /opt/fi-lit/repo
    python -m venv .venv
    . .venv/bin/activate
    pip install --no-index --find-links /opt/fi-lit/wheelhouse -e '.[dev,train]'

    python -m fi_lit check-assets --manifest /opt/fi-lit/private-offline-assets.json
    python -m fi_lit validate-config --config configs/qlora_ddp_superni.yaml --show-plan
    python -m fi_lit.train --config configs/qlora_ddp_superni.yaml --dry-run
    pytest

The template asset manifest intentionally fails until its placeholder paths are replaced. --dry-run neither imports torch nor opens model/data files.

## 3. Build manifests and launch deliberately

    python -m fi_lit build-superni-manifest \
      --superni-root /opt/fi-lit/datasets/natural-instructions \
      --output /opt/fi-lit/manifests/superni-train.jsonl --splits train
    python -m fi_lit build-superni-manifest \
      --superni-root /opt/fi-lit/datasets/natural-instructions \
      --output /opt/fi-lit/manifests/superni-dev.jsonl --splits dev

    CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
      -m fi_lit.train --config configs/qlora_ddp_superni.yaml

Before the last command, inspect nvidia-smi, Python/Torch/CUDA compatibility, free disk space, the model format, and generated manifest counts. The launch command is documented only; it has not been run by this repository setup.

## Reproducibility record

For every server run, retain the Git commit SHA, copied configuration, private asset-manifest checksums, nvidia-smi output, package versions, command line, and resulting metrics. Do not add raw prompts, model checkpoints, or private filesystem paths to Git.

