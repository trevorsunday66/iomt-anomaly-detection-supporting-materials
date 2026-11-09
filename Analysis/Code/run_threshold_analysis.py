from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from pathlib import Path

import analysis_paths
from analysis_paths import PACKAGE_ROOT, new_output, package_relative

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

OUT = None
DEFAULT_INPUT = PACKAGE_ROOT
VIEWS = ["network", "system", "combined"]
SEEDS = [42, 43, 44]
LLMS = ["llama", "mistral", "qwen_off", "qwen_on"]
MODELS = ["IF", "LOF"] + LLMS
# These are analysis choices, not empirical counts or fitted thresholds.
ALERT_BUDGETS = [("1_percent", .01), ("5_percent", .05), ("10_percent", .10)]
NAMES = {"IF": "Isolation Forest", "LOF": "LOF", "llama": "Llama 3.1", "mistral": "Mistral", "qwen_off": "Qwen, thinking off", "qwen_on": "Qwen, thinking on"}
COLORS = {"IF": "#0072B2", "LOF": "#E69F00", "llama": "#009E73", "mistral": "#CC79A7", "qwen_off": "#777777", "qwen_on": "#D55E00"}
STYLES = {"IF": "-", "LOF": "--", "llama": "-.", "mistral": ":", "qwen_off": "--", "qwen_on": "-"}
INPUT_FILES = {
    "cohort": "LLM/Results/Prepared/private/cohort.csv",
    "decisions": "LLM/Results/Evaluation/decisions.csv",
    "raw": "LLM/Results/Raw_Responses/native_responses.jsonl",
    "IF": "ML/Results/Matched_Comparison/if_predictions.csv",
    "LOF": "ML/Results/LOF/matched_lof_predictions.csv",
}


def check(condition, message, checks):
    if not bool(condition):
        raise AssertionError(message)
    checks.append(message)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def as_bool(series):
    mapped = series.astype(str).str.lower().map({"true": True, "false": False})
    if mapped.isna().any():
        raise ValueError("Invalid or missing saved Boolean")
    return mapped.astype(bool)


def cohort_counts(cohort):
    """Derive population sizes from canonical labels, including excluded rows."""
    if not cohort.label.isin([-1, 0, 1]).all():
        raise ValueError("Cohort labels must be -1 (unknown), 0 (normal), or 1 (exposure)")
    normal = int(cohort.label.eq(0).sum())
    exposure = int(cohort.label.eq(1).sum())
    if not normal or not exposure:
        raise ValueError("ROC/PR evaluation requires both normal and exposure windows")
    return {"normal": normal, "exposure": exposure,
            "excluded_unknown": int(cohort.label.eq(-1).sum()),
            "total": len(cohort), "known": normal + exposure}


def derive_if_caps(joined):
    """Count saved IF alerts on known-normal windows separately for each view."""
    normal = joined[joined.detector.eq("IF") & joined.label.eq(0)]
    if set(normal.feature_set) != set(VIEWS):
        raise ValueError("Each IF feature view needs known-normal predictions")
    if normal.duplicated(["feature_set", "request_id"]).any():
        raise ValueError("Duplicate IF normal-window predictions")
    if not normal.saved_decision.isin([True, False]).all():
        raise ValueError("IF decisions must be valid Booleans")
    return {view: int(group.saved_decision.sum())
            for view, group in normal.groupby("feature_set")}


def budget_counts(normal_count, if_cap):
    """Convert chosen percentage ceilings; retain the saved IF count exactly."""
    if normal_count <= 0 or int(normal_count) != normal_count:
        raise ValueError("The normal-window count must be a positive integer")
    if if_cap < 0 or if_cap > normal_count or int(if_cap) != if_cap:
        raise ValueError("The IF alert ceiling must be an integer within the normal population")
    result = [(name, rate, int(math.floor(normal_count * rate + 1e-10)))
              for name, rate in ALERT_BUDGETS]
    result.append(("IF_saved_rate", if_cap / normal_count, int(if_cap)))
    return result


def metrics(y, flag):
    tp = int(np.sum((y == 1) & flag))
    fp = int(np.sum((y == 0) & flag))
    return {"tp": tp, "fp": fp, "tn": int(np.sum(y == 0)) - fp,
            "fn": int(np.sum(y == 1)) - tp,
            "recall": tp / np.sum(y == 1), "fpr": fp / np.sum(y == 0),
            "precision": tp / (tp + fp) if tp + fp else np.nan,
            "flagged": tp + fp}


def sweep(y, scores):
    # Descending distinct thresholds preserve tied blocks. Infinity is the
    # no-alert endpoint; minimum observed score is the all-alert endpoint.
    thresholds = np.r_[np.inf, np.unique(scores)[::-1]]
    rows = []
    for t in thresholds:
        row = {"threshold": t, **metrics(y, scores >= t)}
        row["precision_for_plot"] = row["precision"] if row["flagged"] else 1.0
        rows.append(row)
    return pd.DataFrame(rows)


def load(input_root, checks):
    cohort = pd.read_csv(input_root / INPUT_FILES["cohort"], float_precision="round_trip")
    population = cohort_counts(cohort)
    check(cohort.request_id.notna().all() and cohort.request_id.is_unique,
          f"Canonical cohort contains {population['total']:,} unique non-null window IDs", checks)
    known = cohort[cohort.label.isin([0, 1])].copy()
    ids = set(known.request_id)
    all_ids = set(cohort.request_id)
    d = pd.read_csv(input_root / INPUT_FILES["decisions"], low_memory=False, float_precision="round_trip")
    d = d[d.detector.isin(LLMS)].copy()
    expected_groups = {(detector, view, seed) for detector in LLMS for view in VIEWS for seed in SEEDS}
    check(set(d.groupby(["detector", "feature_set", "seed"]).groups) == expected_groups,
          "Every configured LLM/view/seed group is present", checks)
    check(len(d) == population["total"] * len(expected_groups),
          f"Retained {len(d):,} LLM records for the canonical cohort and configured groups", checks)
    check(not d.duplicated(["detector", "feature_set", "seed", "request_id"]).any(), "LLM request keys unique across configuration, view, seed, window", checks)
    for key, group in d.groupby(["detector", "feature_set", "seed"]):
        check(set(group.request_id) == all_ids, f"{key}: planned IDs match the full canonical cohort", checks)
    d = d[d.request_id.isin(ids)].copy()
    check(len(d) == population["known"] * len(expected_groups) and d.status.eq("valid").all(),
          f"All {len(d):,} known-label LLM responses valid", checks)
    d["saved_decision"] = as_bool(d.prediction)
    d["score"] = pd.to_numeric(d.score, errors="raise")
    check(np.isfinite(d.score).all() and d.score.between(0, 1).all(), "All known-label LLM scores finite and within [0,1]", checks)
    # Confirm every score and Boolean directly against the archived API output.
    lookup = d.set_index(["detector", "feature_set", "seed", "request_id"])
    raw_seen = set()
    for line in (input_root / INPUT_FILES["raw"]).open(encoding="utf-8"):
        r = json.loads(line)
        key = (r["profile"], r["view"], int(r["seed"]), r["request_id"])
        if key not in lookup.index:
            continue
        if key in raw_seen:
            raise AssertionError("Duplicate archived response")
        raw_seen.add(key)
        answer = json.loads(json.loads(r["raw_response"])["message"]["content"])
        saved = lookup.loc[key]
        if float(answer["anomaly_score"]) != saved.score or bool(answer["is_anomaly"]) != saved.saved_decision:
            raise AssertionError(f"Archived response differs from evaluation CSV: {key}")
    check(len(raw_seen) == len(d), f"All {len(d):,} LLM scores and Booleans match archived native responses", checks)
    frames = [d[["detector", "feature_set", "seed", "request_id", "label", "score", "saved_decision"]]]
    for detector in ["IF", "LOF"]:
        m = pd.read_csv(input_root / INPUT_FILES[detector], float_precision="round_trip")
        check(len(m) == population["total"] * len(VIEWS) and
              not m.duplicated(["feature_set", "request_id"]).any(),
              f"{detector}: one canonical record per view/window", checks)
        check(set(m.feature_set) == set(VIEWS), f"{detector}: all configured feature views present", checks)
        for view, group in m.groupby("feature_set"):
            check(set(group.request_id) == all_ids, f"{detector}/{view}: IDs match the full canonical cohort", checks)
        m = m[m.request_id.isin(ids)].copy()
        if detector == "IF":
            m = m.rename(columns={"ml_score": "score", "ml_alert": "saved_decision"})
        else:
            check(m.status.eq("valid").all(), "LOF: all known-label matched responses valid", checks)
            m = m.rename(columns={"alert": "saved_decision"})
        m["saved_decision"] = as_bool(m.saved_decision)
        m["detector"] = detector
        m["seed"] = -1  # A single fitted model per view; not a repeated prediction.
        frames.append(m[["detector", "feature_set", "seed", "request_id", "label", "score", "saved_decision"]])
    joined = pd.concat(frames, ignore_index=True)
    canonical = known.set_index("request_id")
    for (detector, view, seed), group in joined.groupby(["detector", "feature_set", "seed"]):
        check(len(group) == population["known"] and set(group.request_id) == ids,
              f"{detector}/{view}/{seed}: exactly the same {population['known']:,} known-label IDs", checks)
        check(np.array_equal(group.label.to_numpy(), canonical.loc[group.request_id, "label"].to_numpy()),
              f"{detector}/{view}/{seed}: labels match canonical cohort", checks)
        check(np.isfinite(group.score).all(), f"{detector}/{view}/{seed}: scores finite", checks)
    joined = joined.merge(known.drop(columns="label"), on="request_id", validate="many_to_one")
    return joined, cohort


def make_plots(joined, curves, baselines):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                         "axes.titlesize": 9, "axes.labelsize": 8,
                         "xtick.labelsize": 7, "ytick.labelsize": 7,
                         "pdf.fonttype": 42, "ps.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.7), sharex="row", sharey="row")
    for j, view in enumerate(VIEWS):
        for detector in MODELS:
            seeds = [-1] if detector in ["IF", "LOF"] else list(reversed(SEEDS))
            for seed in seeds:
                g = joined[(joined.detector == detector) & (joined.feature_set == view) & (joined.seed == seed)]
                y, s = g.label.to_numpy(), g.score.to_numpy()
                p, r, _ = precision_recall_curve(y, s)
                fpr, tpr, _ = roc_curve(y, s, drop_intermediate=False)
                main = seed in [-1, SEEDS[0]]
                kw = {"color": COLORS[detector], "linestyle": STYLES[detector],
                      "linewidth": 1.5 if main else 0.75, "alpha": 1 if main else 0.38}
                axes[0, j].step(r, p, where="post", **kw)
                axes[1, j].plot(fpr, tpr, **kw)
                b = baselines[(baselines.detector == detector) & (baselines.feature_set == view) & (baselines.seed == seed)].iloc[0]
                # Hollow circles are saved decisions, not score-derived points.
                if main and b.flagged:
                    axes[0, j].plot(b.recall, b.precision, "o", color=COLORS[detector], markerfacecolor="white", markersize=3.6, markeredgewidth=0.8)
                if main:
                    axes[1, j].plot(b.fpr, b.recall, "o", color=COLORS[detector], markerfacecolor="white", markersize=3.6, markeredgewidth=0.8)
        labels = joined.loc[joined.detector.eq("IF") & joined.feature_set.eq(view), "label"]
        axes[0, j].axhline(float(labels.eq(1).mean()), color="#bbbbbb", linewidth=0.7, linestyle=":")
        axes[1, j].plot([0, 1], [0, 1], color="#bbbbbb", linewidth=0.7, linestyle=":")
        axes[0, j].set_title(view.title())
        axes[0, j].set_xlabel("Recall")
        axes[1, j].set_xlabel("Normal-alert rate (FPR)")
        for i in range(2):
            axes[i, j].set_xlim(-0.02, 1.02)
            axes[i, j].set_ylim(-0.02, 1.04)
            axes[i, j].set_xticks([0, .25, .5, .75, 1])
            axes[i, j].set_yticks([0, .25, .5, .75, 1])
            axes[i, j].grid(alpha=.15, linewidth=.5)
    axes[0, 0].set_ylabel("Precision")
    axes[1, 0].set_ylabel("Exposure detection (TPR)")
    handles = [Line2D([0], [0], color=COLORS[d], linestyle=STYLES[d], linewidth=1.5, label=NAMES[d]) for d in MODELS]
    handles.append(Line2D([0], [0], marker="o", color="#333333", markerfacecolor="white", markersize=4, linestyle="None", label=f"Saved decision (seed {SEEDS[0]} / ML)"))
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=7.5, columnspacing=1.5, handlelength=2.4, bbox_to_anchor=(.5, .005))
    fig.subplots_adjust(left=.085, right=.985, top=.95, bottom=.18, wspace=.18, hspace=.34)
    fig.savefig(OUT / "pr_roc_curves.pdf", metadata={"Title": "Matched N01/A01 score-threshold curves"})
    fig.savefig(OUT / "pr_roc_curves.png", dpi=300)
    plt.close(fig)


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, required=True,
                        help="New run directory under package/runs or outside the package")
    args = parser.parse_args()
    OUT = new_output(args.output)
    checks = []
    joined, cohort = load(args.inputs, checks)
    population = cohort_counts(cohort)
    joined.to_csv(OUT / "matched_known_label_scores.csv", index=False, float_format="%.17g")
    all_curves, summary, baseline_rows, budgets = [], [], [], []
    if_caps = derive_if_caps(joined)
    for (detector, view, seed), g in joined.groupby(["detector", "feature_set", "seed"], sort=True):
        keys = {"detector": detector, "feature_set": view, "seed": int(seed)}
        y, s = g.label.to_numpy(), g.score.to_numpy()
        normal_count, exposure_count = int(np.sum(y == 0)), int(np.sum(y == 1))
        c = sweep(y, s)
        c = c.assign(**keys)
        fpr, tpr, thresholds = roc_curve(y, s, drop_intermediate=False)
        check(np.array_equal(c.threshold.to_numpy(), thresholds) and np.allclose(c.fpr, fpr) and np.allclose(c.recall, tpr),
              f"{detector}/{view}/{seed}: direct tied-block sweep equals scikit-learn ROC", checks)
        roc_auc = float(roc_auc_score(y, s))
        ap = float(average_precision_score(y, s))
        direct_ap = float(np.sum(np.diff(c.recall) * c.precision_for_plot.to_numpy()[1:]))
        check(abs(ap-direct_ap) < 1e-12, f"{detector}/{view}/{seed}: direct recall-increment AP equals scikit-learn AP", checks)
        check(c.iloc[0].flagged == 0 and c.iloc[-1].tp == exposure_count and c.iloc[-1].fp == normal_count,
              f"{detector}/{view}/{seed}: no-alert and all-alert endpoints correct", checks)
        b = {**keys, **metrics(y, g.saved_decision.to_numpy())}
        baseline_rows.append(b)
        summary.append({**keys, "normal_windows": normal_count, "exposure_windows": exposure_count, "unique_scores": int(g.score.nunique()),
                        "roc_auc": roc_auc, "average_precision": ap, "saved_tp": b["tp"], "saved_fp": b["fp"]})
        for name, rate, cap in budget_counts(normal_count, if_caps[view]):
            eligible = c[c.fp <= cap]
            # Last row is the lowest observed inclusive threshold satisfying
            # the normal-alert cap, including the entire tied score block.
            chosen = eligible.iloc[-1]
            check(int(chosen.fp) <= cap, f"{detector}/{view}/{seed}: {name} cap respected with whole ties", checks)
            budgets.append({**keys, "budget": name, "nominal_fpr_budget": rate, "normal_alert_cap": cap,
                            **{k: chosen[k] for k in ["threshold", "tp", "fp", "tn", "fn", "recall", "fpr", "precision", "flagged"]}})
        all_curves.append(c)
    curves = pd.concat(all_curves, ignore_index=True)
    sm = pd.DataFrame(summary)
    baseline = pd.DataFrame(baseline_rows)
    bd = pd.DataFrame(budgets)
    for name, frame in [("score_metrics.csv", sm), ("threshold_curves.csv", curves), ("saved_decision_operating_points.csv", baseline), ("matched_alert_budgets.csv", bd)]:
        frame.to_csv(OUT/name, index=False, float_format="%.17g")
    ranges = sm.groupby(["feature_set", "detector"]).agg(roc_auc_min=("roc_auc","min"), roc_auc_max=("roc_auc","max"), average_precision_min=("average_precision","min"), average_precision_max=("average_precision","max"), unique_scores_min=("unique_scores","min"), unique_scores_max=("unique_scores","max")).reset_index()
    ranges.to_csv(OUT/"score_metric_ranges.csv",index=False,float_format="%.17g")
    five_percent = bd[bd.budget.eq("5_percent")].groupby(["feature_set", "detector"]).agg(tp_min=("tp", "min"), tp_max=("tp", "max"), fp_min=("fp", "min"), fp_max=("fp", "max")).reset_index()
    five_percent.to_csv(OUT/"five_percent_budget_ranges.csv",index=False)
    make_plots(joined, curves, baseline)
    manifest = {"analysis": "Frozen-score threshold sweep, no fitting or new inference", "inputs_root": package_relative(args.inputs),
                "inputs": {k:{"path":rel,"sha256":sha256(args.inputs/rel)} for k,rel in INPUT_FILES.items()},
                "script_sha256":sha256(Path(__file__)), "path_helper_sha256":sha256(Path(analysis_paths.__file__)), "versions":{"python":platform.python_version(),"numpy":np.__version__,"pandas":pd.__version__,"matplotlib":matplotlib.__version__,"scikit_learn":sklearn.__version__},
                "population": population, "if_saved_normal_alert_caps": if_caps,
                "method":{"positive_label":1,"score_direction":"higher means more anomalous for every detector","decision":"score >= threshold","thresholds":"infinity plus descending unique observed scores","pr_summary":"non-interpolated average precision (AP), not trapezoidal PR area","roc_summary":"trapezoidal ROC AUC with tied-score handling","budget_selection":"lowest observed threshold with false-positive count <= floor(normal_count*budget); IF_saved_rate uses the directly counted IF normal alerts; whole tied blocks, no interpolation or randomization","seeds":f"LLM seeds {SEEDS} analyzed separately; seed -1 identifies one fixed fitted ML model","plot":f"seed {SEEDS[0]} and ML emphasized; other seeds thin/translucent; hollow markers saved Boolean/calibrated decisions; no-positive-prediction precision undefined and marker omitted"},
                "limitations":[f"{population['normal']} normal windows overlap ML calibration population","Qwen thinking mode was selected after this comparison","adjacent windows and repeated seed predictions are dependent","budget thresholds selected on the reported cohort are descriptive, not independently validated deployment thresholds",f"precision depends on selected {population['exposure']}/{population['known']} positive fraction","scores are ranking signals, not calibrated probabilities","score threshold decisions are distinct from saved LLM Boolean decisions","matched alert caps need not yield equal attained normal-alert counts when scores tie","curves do not control differences in representation, model configuration, or hardware"],
                "validation":{"passed":len(checks),"checks":checks}}
    (OUT/"analysis_manifest.json").write_text(json.dumps(manifest,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    (OUT/"summary.json").write_text(json.dumps({"score_metrics":summary,"validation_checks_passed":len(checks)},indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(ranges.to_string(index=False))
    print("\nIF-saved normal-alert budget results:")
    print(bd[bd.budget.eq("IF_saved_rate")][["feature_set","detector","seed","normal_alert_cap","tp","fp","recall"]].to_string(index=False))
    print(f"\nPassed {len(checks)} checks; outputs: {OUT}")


if __name__ == "__main__":
    main()
