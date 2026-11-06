# N01/A01 anomaly-detection supporting materials

This repository contains the N01 normal recording, A01 attack recording, the ML
and LLM implementations, and their saved scientific results. The recordings
contain 239,152 observations and 71,348 five-second topic windows. The LLM
comparison uses 1,078 matched windows and 38,808 planned requests.

```text
Datasets/
  N01/                     Normal telemetry and collection evidence
  A01/                     Attack telemetry and execution evidence
  Processed/               Saved windows, labels, events and partitions
ML/
  Code/ml_pipeline.py      Preparation, IF/LOF training and scoring
  Results/                 Saved models, predictions, metrics and evidence tables
LLM/
  Code/llm_pipeline.py     Preparation, inference and evaluation
  Results/                 Prepared inputs, native responses and result tables
requirements.txt           Pinned scientific dependencies
```

Each workflow is implemented in one Python script. Settings and the executed
LLM prompt are embedded in those scripts. The saved datasets and result files
retain their original contents. Raw logs and attack receipts are collection
evidence; paths inside those records identify their original collection system.
The scripts resolve their runtime inputs relative to this repository.

## Setup

Scientific preparation and ML execution were tested with Python 3.14 and the
pinned dependencies. From this directory in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Use that environment's Python for the commands below. On Linux/macOS, its path
is `.venv/bin/python`. Offline LLM request previews and evaluation require only
Python's standard library.

## ML

```powershell
python -B ML/Code/ml_pipeline.py run --output runs/ml
```

This rebuilds N01/A01 features and labels, trains all six Isolation Forest/LOF
models across the network, system and combined views, and compares their scores
and decisions with the saved results. To score the saved models instead:

```powershell
python -B ML/Code/ml_pipeline.py score-ml --output runs/ml-scored
```

## LLM

The saved inputs in `LLM/Results/Prepared` are ready for inference. Rebuild them
into a new directory, or preview one request without contacting a model server:

```powershell
python -B LLM/Code/llm_pipeline.py prepare-llm --output runs/llm-prepared
python -B LLM/Code/llm_pipeline.py run-llm --task-id 0 --dry-run --output runs/preview
```

Actual inference requires Ollama 0.12.9, the exact model digests printed by
`python -B LLM/Code/llm_pipeline.py show-config`, and GPU allocation reported by
Ollama. Model weights are external dependencies and are not downloaded by the
script. Remove `--dry-run` to execute a task; use a distinct output directory for
each task ID, from 0 through 1007. Evaluate task outputs with:

```powershell
python -B LLM/Code/llm_pipeline.py evaluate-llm --runs runs/task-0 --output runs/evaluated
```

Use `--release runs/llm-prepared` on both inference and evaluation commands to
select newly prepared inputs. Missing and invalid responses remain in the planned
denominators. Source identity, input identity and numerical claims are checked
without changing the model's original decision.

## Outputs and interpretation

Output directories must be new. Inside the repository, outputs are restricted
to `runs/`, which is ignored by Git. Existing datasets and saved results cannot
be overwritten through the command-line interfaces. Either script's `verify`
command checks the fixed inputs needed by that workflow; `--help` lists commands.

Exposure labels denote temporal association with eligible attack execution.
Unknown-window alerts measure review workload. Seed repeats are dependent
measurements of the same windows, and the comparisons are retrospective.

Some included files exceed GitHub's 25 MiB browser-upload limit. Use Git or
GitHub Desktop for these files; see [GitHub's file-size documentation](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github).
