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
Analysis/
  Code/                    Saved-score PR/ROC, seed, and Qwen claim analyses
  Results/                 Saved analytical tables, curves, cases and provenance
  requirements.txt         Separate analysis and plotting environment
requirements.txt           Pinned scientific dependencies
```

The ML and LLM workflows are each implemented in one Python script. Settings
and the executed LLM prompt are embedded in those scripts. The saved datasets and result files
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

## Analysis

These analyses reproduce the paper's PR/ROC curves, matched alert-budget
comparisons, and Qwen seed-disagreement results from the saved N01/A01 files.
The complete Qwen claim audit compares flipped and stable windows and thinking
on/off, using every complete saved seed group within each view and label.
All analyses use saved files without fitting models, contacting a model server,
or collecting new data.

### Environment

Use Python 3.14 and a separate environment for the plotting analysis. Its tested
NumPy, pandas, and scikit-learn versions differ from the root requirements used
to reproduce the saved ML models. Do not replace that environment's packages.
From the repository root in PowerShell:

```powershell
python -m venv .venv-analysis
.\.venv-analysis\Scripts\python.exe -m pip install -r Analysis/requirements.txt
.\.venv-analysis\Scripts\Activate.ps1
```

On Linux/macOS, activate with `source .venv-analysis/bin/activate`. The seed
analysis and complete Qwen audit use only Python's standard library; the
threshold analysis also needs the pinned scientific and plotting packages.
The complete Qwen audit can run separately with Python 3.11 or later, without
installing the plotting dependencies.

### Reproduce the analyses

Run from the repository root with the analysis environment active:

```powershell
python -B Analysis/Code/run_threshold_analysis.py --output runs/threshold
python -B Analysis/Code/analyze_seeds.py --output runs/seed
python -B Analysis/Code/audit_saved_qwen.py --output runs/qwen-audit
```

Output directories must be new. Inside this repository, generated files are
restricted to `runs/`, which Git ignores. Existing datasets and
`Analysis/Results/` are never overwritten by these commands. A new directory
outside the repository is also allowed. Inputs resolve from the script location,
so moving or renaming the repository does not require editing code.

The threshold and Qwen claim-audit scripts accept `--inputs` for another
supporting-materials root.
The seed script accepts that root as its optional first positional argument.
Input hashes identify the files used; paths in the published provenance files
are relative to this repository.

### Saved outputs

- `Analysis/Results/Threshold/score_metrics.csv` contains ROC area and average precision
  for each detector, feature view, and seed. `threshold_curves.csv` contains
  every attainable score threshold and its confusion counts.
- `Analysis/Results/Threshold/matched_alert_budgets.csv` contains the selected thresholds
  for the specified percentage ceilings and saved IF normal-alert ceilings.
  `pr_roc_curves.pdf` and `.png` contain the paper's score curves.
- `Analysis/Results/Seed/seed_stability_by_label.csv` separates normal, exposure, and
  unknown windows. `paper_seed_table.csv` and `.tex` contain the paper table.
  `selected_cases.json` preserves saved responses and independent claim checks.
- `Analysis/Results/QwenAudit/per_window.csv` records complete mode/view/window groups;
  `window_summary.csv` separates all, flipped, and stable groups by mode, view,
  and label. `per_response.csv` retains every valid response and flags whether
  it belongs to a complete group.
- `Analysis/Results/QwenAudit/pairwise.csv` and `pairwise_summary.csv` compare seed pairs.
  `response_summary.csv` reports claim-occurrence totals and Boolean-stratified
  comparisons. `matched_mode_per_window.csv` and `matched_mode_summary.csv`
  compare thinking on/off on their common complete windows.
- The Qwen audit's `excluded_groups.csv` and `invalid_responses.json` preserve
  incomplete-group accounting. `selected_cases.json` records illustrative
  examples with their selection rules. `methods.json` defines every field and
  interpretation limit; `provenance.json` identifies the saved inputs and code.
- The threshold provenance records software versions and source hashes. Seed
  input paths in `input_hashes.json` are relative to the selected supporting-
  materials root; `validation.json` records generated cohort counts, model
  settings, run integrity, and limitations.



### Interpretation

PR/ROC calculations use the matched known-label windows; unknown windows are
excluded from detection metrics. Unknown-window disagreements describe output
instability, not detection errors. Cohort sizes, valid-response denominators,
and IF alert ceilings are calculated from the saved files.

The 1%, 5%, and 10% budgets and the diagnostic score threshold of 0.5 are
explicit analysis choices. The original LLM Boolean decisions remain distinct
from score-threshold decisions. The legacy CSV name `complete_three_seeds`
is retained for compatibility; completeness uses the saved protocol's seed set.

Seed repeats are dependent observations. These retrospective score sweeps do
not independently calibrate a deployment threshold, and the selected normal
windows overlap the ML calibration population. The proposed two-tier
architecture was not evaluated as a combined detector.

For the complete Qwen audit, a stable window has an unchanged Boolean across
the planned seeds; stability does not establish detection correctness. Claim
sets are unordered unique `(feature, reference, position)` triples, while
claim-rate denominators retain every original occurrence. Supported-only
responses must contain at least one claim. A window with an identical supported
claim set and opposite decisions needs at least one such seed pair; that does
not require all seed responses to contain identical claims.

Claim-set changes, contradictions, and identical-claim disagreements can overlap.
The selected structured claims need not exhaust the evidence considered by the
model. These comparisons describe saved output variation and do not identify
an internal cause or isolate a temperature effect. Complete-group summaries
exclude incomplete seed groups; thinking-mode comparisons use the common
complete windows in both modes. All counts and denominators are derived from
the saved files. No new inference or model fitting is required.

## Outputs and interpretation

Output directories must be new. Inside the repository, outputs are restricted
to `runs/`, which is ignored by Git. Existing datasets and saved results cannot
be overwritten through the command-line interfaces. The ML and LLM pipeline
`verify` commands check their fixed inputs; `--help` lists available commands.
Git attributes preserve file bytes across checkouts so scientific integrity
hashes are not changed by line-ending conversion.

Exposure labels denote temporal association with eligible attack execution.
Unknown-window alerts measure review workload. Seed repeats are dependent
measurements of the same windows, and the comparisons are retrospective.

Some included files exceed GitHub's 25 MiB browser-upload limit. Use Git or
GitHub Desktop for these files; see [GitHub's file-size documentation](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github).
