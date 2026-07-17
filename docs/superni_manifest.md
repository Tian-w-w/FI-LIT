# SuperNI manifest build

The builder expects a locally available SuperNI tree containing:

    SUPERNI_ROOT/
      tasks/task*.json
      splits/default/train_tasks.txt
      splits/default/test_tasks.txt
      splits/default/excluded_tasks.txt

Create manifests only in an ignored directory because each JSONL row contains task instructions, inputs, and reference outputs derived from the raw release:

    python -m fi_lit build-superni-train-dev \
      --superni-root /opt/fi-lit/datasets/natural-instructions \
      --train-output /opt/fi-lit/manifests/superni-train.jsonl \
      --dev-output /opt/fi-lit/manifests/superni-dev.jsonl \
      --dev-task-count 50 --seed 42 \
      --instances-per-task 100 --instance-seed 42

For a reported experiment, use --instances-per-task N. It deterministically samples N instances per task from a SHA-256-derived task seed, preserving the original instance IDs and recording the seed in the manifest summary. The recommended first setting is 100 with instance seed 42. To make a small smoke-test manifest only, use --max-instances-per-task 2 instead; this takes the first N instances and must not be used for reported results.

Each manifest row has stable id, task_id, split, categories, definition, input, and references fields. The training script uses the first reference output as the supervised target. The train/dev command selects 50 complete tasks from the official train split using its recorded seed; it never reads the official test split. Use test_tasks.txt only for final evaluation.
