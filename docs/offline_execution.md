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

    python -m fi_lit build-superni-train-dev \
      --superni-root /opt/fi-lit/datasets/natural-instructions \
      --train-output /opt/fi-lit/manifests/superni-train.jsonl \
      --dev-output /opt/fi-lit/manifests/superni-dev.jsonl \
      --dev-task-count 50 --seed 42 \
      --instances-per-task 100 --instance-seed 42

    CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
      -m fi_lit.train --config configs/qlora_ddp_superni.yaml

Before the last command, inspect nvidia-smi, Python/Torch/CUDA compatibility, free disk space, the model format, and generated manifest counts. The launch command is documented only; it has not been run by this repository setup.

For a one-GPU smoke test, copy the configuration to a private server-only file, set training.max_steps to 5, and launch with CUDA_VISIBLE_DEVICES=0. Do not use this five-step result as an experiment metric; restore max_steps to -1 for a deliberate full run.

## 4. Select and evaluate an adapter

Select the checkpoint with the lowest development metric, never the test metric. Build the official test manifest with the same deterministic per-task sampling policy, then generate greedily from the base model plus the selected adapter. The evaluator reports case/whitespace-normalized Exact Match and whitespace-token ROUGE-L F1, both as instance micro averages and task macro averages. These lexical scores are reproducible baseline diagnostics; task-family-specific metrics may be added for a final paper.

    PYTHONPATH=src python -m fi_lit evaluate-superni \
      --config configs/server_qlora.yaml \
      --adapter /opt/fi-lit/outputs/qlora-ddp-superni/checkpoint-200 \
      --manifest /opt/fi-lit/manifests/superni-100-test.jsonl \
      --predictions /opt/fi-lit/outputs/eval/test-predictions.jsonl \
      --metrics /opt/fi-lit/outputs/eval/test-metrics.json \
      --batch-size 4 --max-new-tokens 128

## Reproducibility record

For every server run, retain the Git commit SHA, copied configuration, private asset-manifest checksums, nvidia-smi output, package versions, command line, and resulting metrics. Do not add raw prompts, model checkpoints, or private filesystem paths to Git.
