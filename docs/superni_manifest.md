# SuperNI manifest build

The builder expects a locally available SuperNI tree containing:

    SUPERNI_ROOT/
      tasks/task*.json
      splits/train_tasks.txt
      splits/test_tasks.txt
      splits/excluded_tasks.txt

Create manifests only in an ignored directory because each JSONL row contains task instructions, inputs, and reference outputs derived from the raw release:

    python -m fi_lit build-superni-train-dev \
      --superni-root /opt/fi-lit/datasets/natural-instructions \
      --train-output /opt/fi-lit/manifests/superni-train.jsonl \
      --dev-output /opt/fi-lit/manifests/superni-dev.jsonl \
      --dev-task-count 50 --seed 42

To make a small smoke-test manifest on the server, append --max-instances-per-task 2. This option truncates each task independently and should never be used for a reported experiment.

Each manifest row has stable id, task_id, split, categories, definition, input, and references fields. The training script uses the first reference output as the supervised target. The train/dev command selects 50 complete tasks from the official train split using its recorded seed; it never reads the official test split. Use test_tasks.txt only for final evaluation.
