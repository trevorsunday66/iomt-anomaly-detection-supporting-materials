from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

from analysis_paths import PACKAGE_ROOT, new_output

OUT = None
LABEL_NAMES = {-1: "unknown", 0: "normal", 1: "exposure"}


def save_json(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def save_csv(name, rows, fieldnames=None):
    if fieldnames is None:
        fieldnames = list(rows[0])
    with (OUT / name).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_csv(path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def bool_value(value):
    assert value in ("True", "False"), value
    return value == "True"


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def count_word(count):
    """Spell small counts for prose; these are formatting rules, not study data."""
    words = ("zero", "one", "two", "three", "four", "five", "six", "seven",
             "eight", "nine", "ten", "eleven", "twelve")
    return words[count] if 0 <= count < len(words) else f"{count:,}"


def fraction(numerator, denominator):
    return numerator / denominator if denominator else None


def incomplete_table_note(invalid):
    if not invalid:
        return "All planned responses were valid."
    if len(invalid) == 1:
        row = invalid[0]
        mode = {"qwen_on": "thinking", "qwen_off": "non-thinking"}.get(row["detector"], row["detector"])
        label = LABEL_NAMES[row["label"]]
        article = "an" if label[0] in "aeiou" else "a"
        return f"The incomplete {row['feature_set'].title()} {mode} response was on {article} {label} window."
    counts = Counter(LABEL_NAMES[row["label"]] for row in invalid)
    details = ", ".join(f"{label}={count}" for label, count in sorted(counts.items()))
    return f"{len(invalid)} invalid responses were excluded from complete-seed counts ({details})."


def invalid_findings_note(invalid, seeds):
    counts = Counter((row["feature_set"], row["detector"]) for row in invalid
                     if row["detector"] in ("qwen_on", "qwen_off"))
    parts = []
    names = {"qwen_on": "Qwen-thinking", "qwen_off": "Qwen-nonthinking"}
    for (view, profile), count in sorted(counts.items()):
        noun = "response" if count == 1 else "responses"
        parts.append(f"{count_word(count)} invalid {view.title()} {names[profile]} {noun}")
    if not parts:
        return "no invalid Qwen responses."
    return ("; ".join(parts) + f" excluded from complete-{count_word(len(seeds))}-seed counts.")


def sampling_findings(profiles, summary):
    qwen = [name for name in ("qwen_off", "qwen_on") if name in profiles]
    options = [profiles[name]["options"] for name in qwen]
    fields = ("temperature", "top_p", "top_k")
    if len(qwen) > 1 and all(tuple(o.get(k) for k in fields) == tuple(options[0].get(k) for k in fields) for o in options):
        o = options[0]
        text = f"Qwen used temperature {o.get('temperature')}, top_p {o.get('top_p')} and top_k {o.get('top_k')} in both thinking modes"
    else:
        text = "; ".join(f"{name} used " + ", ".join(f"{key} {profiles[name]['options'].get(key)}" for key in fields) for name in qwen)
    controls = [name for name in ("llama", "mistral") if name in profiles]
    if controls:
        temps = [profiles[name]["options"].get("temperature") for name in controls]
        names = " and ".join(name.title() for name in controls)
        if len(set(temps)) == 1:
            text += f", while {names} used temperature {temps[0]}"
        else:
            text += "; " + ", ".join(f"{name.title()} used temperature {temp}" for name, temp in zip(controls, temps))
    text += ". Nonzero-temperature sampling permits seed-dependent output; the observed final responses identify changed decisions and evidence statements, but do not isolate an internal cause or the causal effect of temperature/thinking."
    if controls:
        flips = {name: sum(row["flip_windows"] for row in summary if row["profile"] == name and row["label"] == "all") for name in controls}
        if len(set(flips.values())) == 1:
            text += f" {names} had {count_word(next(iter(flips.values())))} Boolean disagreements in this archive."
        else:
            text += " " + "; ".join(f"{name.title()} had {count_word(count)} Boolean disagreements" for name, count in flips.items()) + " in this archive."
    return "Interpretation: " + text


def illustrative_findings(by_key, inputs, refs, seeds):
    # These identifiers select the published illustrations, not expected results.
    # Values, decisions, claims, and exact reasons are read from the saved files.
    examples = (("system", "w01746fbf5eba1b29ed232e06", "wifi_rssi_dbm__std", "RSSI standard deviation", " dBm"),
                ("network", "w01dfadf44e653faab2adeae3", "payload_entropy__mean", "payload entropy mean", ""))
    notes = []
    for view, rid, feature, name, unit in examples:
        group = [by_key.get(("qwen_on", view, seed, rid)) for seed in seeds]
        if not all(row and row["status"] == "valid" for row in group):
            continue
        value = inputs[view][rid][feature]
        low, high = refs[view][feature]["p01"], refs[view][feature]["p99"]
        position = "below" if value < low else "above" if value > high else "within"
        details = []
        for row in group:
            claims = row["claims"]
            verdicts = Counter(c["verdict"] for c in claims)
            comparisons = [c["position"] for c in claims if c["feature"] == feature]
            details.append(f"Seed {row['seed']}: {str(row['prediction']).lower()}/{row['score']}, "
                           f"{verdicts['supported']} supported and {verdicts['contradicted']} contradicted claims, "
                           f"{verdicts['unverifiable']} unverifiable claims, supplied-feature claim positions={comparisons}; "
                           f"reason={json.dumps(row['reason'], ensure_ascii=False)}")
        label = LABEL_NAMES[group[0]["label"]]
        notes.append(f"Illustrative {view.title()} {label} case {rid}: {name} was {value}{unit}, {position} "
                     f"the inclusive p01-p99 interval [{low},{high}]{unit}. " + " | ".join(details) + ". "
                     "The saved final outputs describe the differing decisions and evidence claims; they do not establish an internal cause of the change.")
    return notes


def score_boolean_findings(score_boolean, summary):
    configs = [r for r in score_boolean if r["profile"] == "qwen_on" and r["valid_responses"]]
    failures = sum(not r["single_monotone_score_threshold_reproduces_all_booleans"] for r in configs)
    if not configs:
        return "No valid Qwen-thinking view/seed configurations were available for score/Boolean diagnostics."
    if failures == len(configs):
        text = (f"All {count_word(len(configs))} Qwen-thinking view/seed configurations have overlapping score ranges "
                "for positive and negative Booleans, so no single monotone score threshold reproduces every direct Boolean "
                "within any of those valid-response sets.")
    else:
        text = (f"{failures}/{len(configs)} Qwen-thinking view/seed configurations have overlapping score ranges "
                "for positive and negative Booleans and cannot reproduce every direct Boolean with a single monotone score threshold.")
    equal = [(r["view"], r["flips_same_score_opposite_boolean"]) for r in summary
             if r["profile"] == "qwen_on" and r["label"] == "all" and r["flips_same_score_opposite_boolean"]]
    if equal:
        parts = [f"{count_word(count)} {view.title()}" for view, count in equal]
        noun = "window also returns" if sum(count for _, count in equal) == 1 else "window also return" if all(count == 1 for _, count in equal) else "windows also return"
        joined = " and ".join(parts)
        text += " " + joined[0].upper() + joined[1:] + f" {noun} the same score with opposite Booleans across seeds."
    return text + " The threshold sweep is therefore a separate decision rule, not reconstruction of the original Boolean rule."


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="?", type=Path, default=PACKAGE_ROOT)
    parser.add_argument("--output", "--output-dir", dest="output", type=Path, required=True,
                        help="New output directory, under package runs/ or outside the package")
    args = parser.parse_args()
    args.inputs = args.inputs.expanduser().resolve()
    OUT = new_output(args.output)
    ROOT = args.inputs / "LLM" / "Results"
    profiles = json.loads((ROOT / "Prepared" / "profiles.json").read_text(encoding="utf-8"))
    protocol = json.loads((ROOT / "Prepared" / "protocol.json").read_text(encoding="utf-8"))
    SEEDS = tuple(sorted(protocol["config"]["seeds"]))
    VIEWS = tuple(protocol["config"]["views"])
    profile_order = ("llama", "mistral", "qwen_off", "qwen_on")
    PROFILES = tuple(sorted(profiles, key=lambda name: (profile_order.index(name) if name in profile_order else len(profile_order), name)))
    assert SEEDS and len(SEEDS) == len(set(SEEDS)), "Saved seed list must be nonempty and unique"
    assert VIEWS and len(VIEWS) == len(set(VIEWS)), "Saved view list must be nonempty and unique"
    cohort_rows = load_csv(ROOT / "Prepared" / "private" / "cohort.csv")
    cohort = {r["request_id"]: r for r in cohort_rows}
    assert cohort and len(cohort) == len(cohort_rows), "Cohort request IDs must be nonempty and unique"
    label_counts = Counter(int(r["label"]) for r in cohort_rows)
    assert set(label_counts) <= set(LABEL_NAMES), "Unknown cohort label code"
    tasks = json.loads((ROOT / "Prepared" / "tasks.json").read_text(encoding="utf-8"))
    planned_keys = set()
    for task in tasks:
        for rid in task["request_ids"]:
            key = task["profile"], task["view"], int(task["seed"]), rid
            assert key not in planned_keys, f"Duplicate planned key: {key}"
            planned_keys.add(key)
    expected_keys = set(itertools.product(PROFILES, VIEWS, SEEDS, cohort))
    assert planned_keys == expected_keys, "Saved tasks must cover the cohort, profiles, views, and seeds"
    input_paths = [ROOT / "Evaluation" / "decisions.csv", ROOT / "Raw_Responses" / "native_responses.jsonl", ROOT / "Prepared" / "profiles.json", ROOT / "Prepared" / "inference" / "references.json"] + [ROOT / "Prepared" / "inference" / f"{v}.jsonl" for v in VIEWS] + [ROOT / "Prepared" / "private" / "cohort.csv", ROOT / "Prepared" / "protocol.json", ROOT / "Prepared" / "tasks.json"]
    refs = json.loads((ROOT / "Prepared" / "inference" / "references.json").read_text(encoding="utf-8"))
    inputs = {}
    for view in VIEWS:
        with (ROOT / "Prepared" / "inference" / f"{view}.jsonl").open(encoding="utf-8") as f:
            objects = list(map(json.loads, f))
            inputs[view] = {obj["request_id"]: obj["features"] for obj in objects}
            assert len(inputs[view]) == len(objects) and set(inputs[view]) == set(cohort), f"{view}: inference/cohort request IDs differ"

    rows = load_csv(ROOT / "Evaluation" / "decisions.csv")
    rows = [r for r in rows if r["detector"] in PROFILES]
    assert len(rows) == len(planned_keys), "Evaluation row count differs from saved task plan"
    by_key = {}
    groups = defaultdict(dict)
    invalid = []
    for r in rows:
        profile, view, seed, rid = r["detector"], r["feature_set"], int(r["seed"]), r["request_id"]
        key = profile, view, seed, rid
        assert key not in by_key
        assert seed in SEEDS and view in VIEWS
        r["label"] = int(r["label"])
        assert key in planned_keys, f"Unplanned evaluation key: {key}"
        assert r["label"] == int(cohort[rid]["label"]), f"Cohort/evaluation label mismatch: {key}"
        assert all(r[field] == cohort[rid][field] for field in ("campaign", "topic", "window_start")), f"Cohort/evaluation identity mismatch: {key}"
        r["seed"] = seed
        if r["status"] == "valid":
            r["prediction"] = bool_value(r["prediction"])
            r["score"] = float(r["score"])
            assert math.isfinite(r["score"]) and 0 <= r["score"] <= 1
            r["claims"] = json.loads(r["claims"])
        else:
            invalid.append({k: r[k] for k in ("detector", "feature_set", "seed", "request_id", "status", "label", "error")})
        by_key[key] = r
        groups[(profile, view, rid)][seed] = r
    assert set(by_key) == planned_keys, "Evaluation request keys differ from saved task plan"

    # Independently compare every valid Qwen final response to the evaluation CSV.
    # The optional internal thinking text is not used to infer a causal mechanism.
    raw_checked = 0
    raw_keys = set()
    with (ROOT / "Raw_Responses" / "native_responses.jsonl").open(encoding="utf-8") as f:
        for raw in map(json.loads, f):
            if raw["profile"] not in ("qwen_on", "qwen_off"):
                continue
            key = raw["profile"], raw["view"], raw["seed"], raw["request_id"]
            assert key not in raw_keys
            raw_keys.add(key)
            r = by_key[key]
            assert raw["status"] == r["status"]
            if r["status"] != "valid":
                continue
            payload = json.loads(raw["raw_response"])
            response = json.loads(payload["message"]["content"])
            assert response["is_anomaly"] is r["prediction"]
            assert float(response["anomaly_score"]) == r["score"]
            assert response["reason"] == r["reason"]
            saved_claims = [{k: c[k] for k in ("feature", "reference", "position")} for c in r["claims"]]
            assert response["evidence_claims"] == saved_claims
            r["request_sha256"] = raw["request_sha256"]
            raw_checked += 1
    qwen_keys = {key for key in planned_keys if key[0] in ("qwen_on", "qwen_off")}
    assert raw_keys == qwen_keys, "Native Qwen keys differ from saved task plan"
    assert raw_checked == sum(by_key[key]["status"] == "valid" for key in qwen_keys), "Valid native/evaluation Qwen counts differ"

    window_rows, flipped_groups = [], []
    for (profile, view, rid), seed_rows in sorted(groups.items()):
        assert set(seed_rows) == set(SEEDS)
        assert len({r["label"] for r in seed_rows.values()}) == 1
        assert len({(r["campaign"], r["topic"], r["window_start"]) for r in seed_rows.values()}) == 1
        first = seed_rows[SEEDS[0]]
        valid = [r for r in seed_rows.values() if r["status"] == "valid"]
        complete = len(valid) == len(SEEDS)
        flip = complete and len({r["prediction"] for r in valid}) > 1
        scores = [r["score"] for r in valid]
        equal_score_flip = complete and any(a["score"] == b["score"] and a["prediction"] != b["prediction"] for a,b in itertools.combinations(valid,2))
        reversed_score_flip = complete and any(a["prediction"] and not b["prediction"] and a["score"] < b["score"] for a,b in itertools.permutations(valid,2))
        rec = {"profile":profile,"view":view,"request_id":rid,"label":first["label"],"label_name":LABEL_NAMES[first["label"]],"campaign":first["campaign"],"topic":first["topic"],"window_start":first["window_start"],"complete_three_seeds":complete,"decision_flip":flip,"valid_responses":len(valid),"positive_decisions":sum(r["prediction"] for r in valid),"score_min":min(scores) if scores else None,"score_max":max(scores) if scores else None,"score_span":max(scores)-min(scores) if scores else None,"same_score_opposite_boolean":equal_score_flip,"higher_score_normal_lower_score_anomaly":reversed_score_flip,"distinct_reasons":len({r["reason"] for r in valid}),"distinct_claim_sets":len({json.dumps(r["claims"],sort_keys=True) for r in valid})}
        for seed in SEEDS:
            r = seed_rows[seed]
            for field in ("prediction","score","reason"):
                rec[f"{field}_{seed}"] = r[field] if r["status"] == "valid" else None
        window_rows.append(rec)
        if flip:
            flipped_groups.append((rec,seed_rows))

    summary = []
    for profile in PROFILES:
        for view in VIEWS:
            for label in ("all",-1,0,1):
                subset = [r for r in window_rows if r["profile"] == profile and r["view"] == view and (label == "all" or r["label"] == label)]
                complete = [r for r in subset if r["complete_three_seeds"]]
                flips = [r for r in complete if r["decision_flip"]]
                stable = [r for r in complete if not r["decision_flip"]]
                summary.append({"profile":profile,"view":view,"label":label,"label_name":"all" if label == "all" else LABEL_NAMES[label],"planned_windows":len(subset),"complete_windows":len(complete),"flip_windows":len(flips),"flip_fraction":fraction(len(flips),len(complete)),"flipped_median_score_span":median([r["score_span"] for r in flips]) if flips else None,"stable_median_score_span":median([r["score_span"] for r in stable]) if stable else None,"flips_same_score_opposite_boolean":sum(r["same_score_opposite_boolean"] for r in flips),"flips_score_order_reversal":sum(r["higher_score_normal_lower_score_anomaly"] for r in flips)})

    pairs=[]
    for profile in PROFILES:
        for view in VIEWS:
            for label in ("all",-1,0,1):
                for a,b in itertools.combinations(SEEDS,2):
                    paired=[]
                    for (p,v,rid), sr in groups.items():
                        if p!=profile or v!=view or (label!="all" and sr[a]["label"]!=label):
                            continue
                        if sr[a]["status"]==sr[b]["status"]=="valid":
                            paired.append((sr[a],sr[b]))
                    discord=sum(x["prediction"]!=y["prediction"] for x,y in paired)
                    pairs.append({"profile":profile,"view":view,"label":label,"seed_a":a,"seed_b":b,"paired_valid_windows":len(paired),"disagreements":discord,"disagreement_fraction":fraction(discord,len(paired))})

    score_boolean=[]
    for profile in PROFILES:
        for view in VIEWS:
            for seed in SEEDS:
                group=[r for r in rows if r["detector"]==profile and r["feature_set"]==view and r["seed"]==seed and r["status"]=="valid"]
                positive=[r["score"] for r in group if r["prediction"]]
                negative=[r["score"] for r in group if not r["prediction"]]
                # A >= threshold represents all Booleans iff max negative < min positive.
                representable = not positive or not negative or max(negative) < min(positive)
                score_boolean.append({"profile":profile,"view":view,"seed":seed,"valid_responses":len(group),"boolean_positive":len(positive),"boolean_negative":len(negative),"min_positive_score":min(positive) if positive else None,"max_negative_score":max(negative) if negative else None,"single_monotone_score_threshold_reproduces_all_booleans":representable,"mismatch_boolean_vs_score_ge_0_5":sum(r["prediction"] != (r["score"]>=0.5) for r in group),"diagnostic_only":"0.5 was not the original Boolean decision rule"})

    # Deterministic illustrative selection: one flipped case per Qwen mode/view/label;
    # prefer same-score opposite decisions, then the largest score span, then ID.
    selected=[]
    claim_checks=0
    for profile in ("qwen_off","qwen_on"):
        for view in VIEWS:
            for label in (0,1,-1):
                candidates=[(rec,sr) for rec,sr in flipped_groups if rec["profile"]==profile and rec["view"]==view and rec["label"]==label]
                if not candidates:
                    continue
                rec,sr = sorted(candidates,key=lambda x:(not x[0]["same_score_opposite_boolean"],-x[0]["score_span"],x[0]["request_id"]))[0]
                responses=[]
                for seed in SEEDS:
                    r=sr[seed]
                    claims=[]
                    for c in r["claims"]:
                        feature=c["feature"]
                        value=inputs[view][rec["request_id"]].get(feature)
                        ref=refs[view].get(feature,{})
                        low,high=(ref.get("min"),ref.get("max")) if c["reference"]=="min_max" else (ref.get("p01"),ref.get("p99"))
                        if any(x is None for x in (value,low,high)):
                            position=None
                            verdict="unverifiable"
                        else:
                            position="below" if value<low else "above" if value>high else "within"
                            verdict="supported" if position==c["position"] else "contradicted"
                        assert verdict==c["verdict"]
                        claims.append({**c,"supplied_value":value,"lower_bound":low,"upper_bound":high,"independently_computed_position":position,"independently_computed_verdict":verdict})
                        claim_checks+=1
                    responses.append({"seed":seed,"is_anomaly":r["prediction"],"anomaly_score":r["score"],"reason":r["reason"],"claims":claims,"request_sha256":r["request_sha256"]})
                selected.append({"profile":profile,"view":view,"request_id":rec["request_id"],"label":label,"label_name":LABEL_NAMES[label],"topic":rec["topic"],"window_start":rec["window_start"],"score_span":rec["score_span"],"same_score_opposite_boolean":rec["same_score_opposite_boolean"],"selection_rule":"within mode/view/label, prefer same-score opposite Booleans, then maximum score span, then lexicographically smallest request ID; illustrative, not a representative random sample","responses":responses})

    save_csv("window_seed_details.csv",window_rows)
    save_csv("flipped_window_details.csv",[r for r in window_rows if r["decision_flip"]], fieldnames=list(window_rows[0]))
    save_csv("seed_stability_by_label.csv",summary)
    save_csv("pairwise_seed_disagreement.csv",pairs, fieldnames=["profile", "view", "label", "seed_a", "seed_b", "paired_valid_windows", "disagreements", "disagreement_fraction"])
    save_csv("score_boolean_diagnostics.csv",score_boolean)
    paper_rows=[]
    for profile in ("qwen_off","qwen_on"):
        for view in VIEWS:
            sub={r["label"]:r for r in summary if r["profile"]==profile and r["view"]==view}
            paper_rows.append({"profile":profile,"view":view,"normal_flips":sub[0]["flip_windows"],"normal_complete":sub[0]["complete_windows"],"exposure_flips":sub[1]["flip_windows"],"exposure_complete":sub[1]["complete_windows"],"unknown_flips":sub[-1]["flip_windows"],"unknown_complete":sub[-1]["complete_windows"],"all_flips":sub["all"]["flip_windows"],"all_complete":sub["all"]["complete_windows"],"flip_percent":100*sub["all"]["flip_fraction"] if sub["all"]["flip_fraction"] is not None else None,"median_score_span_flipped":sub["all"]["flipped_median_score_span"],"median_score_span_stable":sub["all"]["stable_median_score_span"]})
    save_csv("paper_seed_table.csv",paper_rows)
    seed_range = (f"{SEEDS[0]}--{SEEDS[-1]}" if len(SEEDS) > 1 and SEEDS == tuple(range(SEEDS[0], SEEDS[-1] + 1)) else ", ".join(map(str, SEEDS)))
    caption = (r"\caption{Qwen Boolean disagreement across seeds " + seed_range + ". Each cell reports windows with at least one changed decision divided by windows with valid responses for all " + count_word(len(SEEDS)) + " seeds.}")
    tex=[r"\begin{table*}[t]",r"\centering\small",caption,r"\label{tab:seed_breakdown}",r"\begin{tabular}{@{}llrrrr@{}}",r"\toprule",r"Thinking & View & Normal & Exposure & Unknown & Total \\",r"\midrule"]
    for r in paper_rows:
        mode="Off" if r["profile"]=="qwen_off" else "On"
        cells=[mode,r["view"].title()]+[f"{r[k+'_flips']}/{r[k+'_complete']}" for k in ("normal","exposure","unknown","all")]
        tex.append(" & ".join(cells)+r" \\")
    tex.extend([r"\bottomrule",r"\end{tabular}",r"\par\smallskip",r"\parbox{\textwidth}{\footnotesize " + incomplete_table_note(invalid) + " Disagreements on unknown windows measure output instability, not detection errors. Views and seeds reuse observations and are not independent trials.}",r"\end{table*}"])
    (OUT/"paper_seed_table.tex").write_text("\n".join(tex)+"\n",encoding="utf-8")
    save_json("selected_cases.json",selected)
    save_json("invalid_responses.json",invalid)
    save_json("input_hashes.json",[{"path":p.relative_to(args.inputs).as_posix(),"bytes":p.stat().st_size,"sha256":sha256(p)} for p in input_paths])
    validation={"saved_llm_rows":len(rows),"unique_request_profile_view_seed_keys":len(by_key),"qwen_native_valid_responses_reconciled":raw_checked,"qwen_native_total_records":len(raw_keys),"invalid_responses":len(invalid),"complete_response_groups":sum(r["complete_three_seeds"] for r in window_rows),"cohort_windows":len(cohort),"cohort_label_counts":{LABEL_NAMES[label]:label_counts[label] for label in LABEL_NAMES},"planned_llm_rows":len(planned_keys),"planned_response_groups":len(groups),"planned_qwen_rows":len(qwen_keys),"planned_seed_count":len(SEEDS),"input_integrity_checks_passed":True,"selected_case_claims_independently_rechecked":claim_checks,"software":"Python standard library","seeds":SEEDS,"profiles":profiles,"method_limitations":["Descriptive repeated predictions on dependent windows, not independent trials.","Flips are disagreement, not necessarily detection errors; unknown labels are not false positives.","The score was an uncalibrated ranking signal and the Boolean was generated directly; a 0.5 diagnostic is not an assumed original decision threshold.","Final responses and sampling settings do not establish internal causes of seed dependence.","Illustrative cases are selected by an explicit diagnostic rule, not a random sample.","No new inference, temperature ablation, or hardware test was performed."]}
    save_json("validation.json",validation)

    lines=["Qwen saved-result seed analysis", "", f"Reconciled {raw_checked:,} valid Qwen responses against native final response JSON; " + invalid_findings_note(invalid, SEEDS), ""]
    for profile in ("qwen_on","qwen_off"):
        lines.append(profile)
        for view in VIEWS:
            sub=[r for r in summary if r["profile"]==profile and r["view"]==view]
            total=next(r for r in sub if r["label"]=="all")
            counts={r["label_name"]:r["flip_windows"] for r in sub if r["label"]!="all"}
            percent = f"{100*total['flip_fraction']:.2f}%" if total["flip_fraction"] is not None else "undefined: no complete windows"
            lines.append(f"  {view}: {total['flip_windows']}/{total['complete_windows']} flips ({percent}); normal={counts['normal']}, exposure={counts['exposure']}, unknown={counts['unknown']}; median score span in flipped windows={total['flipped_median_score_span']}; stable={total['stable_median_score_span']}; same-score opposite-Boolean windows={total['flips_same_score_opposite_boolean']}; score-order reversals={total['flips_score_order_reversal']}.")
    lines.extend(["", sampling_findings(profiles, summary), "",
                  "Each selected case preserves exact final reason text and independently checks structured numerical claims against the supplied rounded inputs and inclusive reference intervals. A changed explanation does not itself establish that reasoning caused the decision change.", "",
                  "The score/Boolean diagnostics are descriptive: the prompt requested a direct Boolean and an uncalibrated ranking score, so inconsistency with an arbitrary 0.5 threshold is not a protocol violation."])
    for note in illustrative_findings(by_key, inputs, refs, SEEDS):
        lines.extend(["", note])
    lines.extend(["", score_boolean_findings(score_boolean, summary)])
    (OUT/"findings.txt").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print("\n".join(lines[:12]))
    print(f"Wrote audit to {OUT}; independently checked {claim_checks} selected-case claims.")


if __name__ == "__main__":
    main()
