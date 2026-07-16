# SuperNI manifest build

The builder expects a locally available SuperNI tree containing:

    SUPERNI_ROOT/
      tasks/task*.json
      task_splits/default/train_tasks.txt
      task_splits/default/dev_tasks.txt
      task_splits/default/test_tasks.txt

Create manifests only in an ignored directory because each JSONL row contains task instructions, inputs, and reference outputs derived from the raw release:

    python -m fi_lit build-superni-manifest \
      --superni-root /opt/fi-lit/datasets/natural-instructions \
      --output /opt/fi-lit/manifests/superni-train.jsonl \
      --splits train

To make a small smoke-test manifest on the server, append --max-instances-per-task 2. This option truncates each task independently and should never be used for a reported experiment.

Each manifest row has stable id, task_id, split, categories, definition, input, and references fields. The training script uses the first reference output as the supervised target. Keep train/dev/test manifests separate; use the official test split only for final evaluation.

