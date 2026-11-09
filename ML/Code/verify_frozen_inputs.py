"""Verify frozen ML inputs and saved decision metrics without training or inference.

Historical row counts belong here, not in the analysis loader. Classification
metrics and unknown-alert workload are recalculated from saved predictions and
compared with the saved metric table. This does not recompute scores, ranking
metrics, event coverage, or latency. No analysis module or model is loaded.
"""

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path


DEFAULT_INPUTS = Path(__file__).resolve().parents[2]
KEYS = ('campaign', 'topic', 'window_start', 'window_end')
VIEWS = ('network', 'system', 'combined')
FROZEN_WINDOWS = 71348
FROZEN_COHORT = 1078


def rows(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        yield from csv.DictReader(stream)


def key(row):
    return tuple(row[name] for name in KEYS)


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def ratio(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def decision_metrics(counts):
    tn, fp, fn, tp = (counts[(0, False)], counts[(0, True)],
                      counts[(1, False)], counts[(1, True)])
    n = tn + fp + fn + tp
    return dict(n=n, positives=fn + tp, tn=tn, fp=fp, fn=fn, tp=tp,
                accuracy=ratio(tn + tp, n), sample_precision=ratio(tp, tp + fp),
                recall=ratio(tp, tp + fn), sample_f1=ratio(2 * tp, 2 * tp + fp + fn),
                normal_window_fpr=ratio(fp, fp + tn),
                unknown_windows=counts[(-1, False)] + counts[(-1, True)],
                unknown_alerts=counts[(-1, True)])


def verify(inputs):
    checks, files, calculated = [], {}, []

    def checked_path(relative):
        path = inputs / relative
        files[relative] = sha256(path)
        return path

    def check(name, actual, expected):
        if isinstance(expected, float):
            passed = math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12)
        else:
            passed = actual == expected
        checks.append(dict(name=name, passed=bool(passed), actual=actual, expected=expected))

    windows, window_rows = {}, 0
    for row in rows(checked_path('Datasets/Processed/windows_N01_A01.csv')):
        window_rows += 1
        windows[key(row)] = (int(row['label']), row['split'])
    check('frozen_window_rows', window_rows, FROZEN_WINDOWS)
    check('unique_window_keys', len(windows), window_rows)

    cohort, cohort_rows, cohort_mismatch = set(), 0, 0
    for row in rows(checked_path('LLM/Results/Prepared/private/cohort.csv')):
        cohort_rows += 1
        identity = key(row)
        cohort.add(identity)
        cohort_mismatch += windows.get(identity) != (int(row['label']), row['split'])
    check('frozen_cohort_rows', cohort_rows, FROZEN_COHORT)
    check('unique_cohort_keys', len(cohort), cohort_rows)
    check('cohort_metadata_mismatches', cohort_mismatch, 0)

    metric_rows = list(rows(checked_path('ML/Results/LOF/ml_metrics_N01_A01.csv')))
    saved_metrics = {(row['detector'], row['feature_set'], row['scope']): row
                     for row in metric_rows}
    check('unique_saved_metric_keys', len(saved_metrics), len(metric_rows))
    expected_metric_keys = {(detector, view, scope)
                            for detector in ('Isolation Forest', 'LOF')
                            for view in VIEWS
                            for scope in ('validation', 'matched_development')}
    check('saved_metric_keys_match_protocol', sorted(saved_metrics), sorted(expected_metric_keys))

    for folder, detector, score_column, alert_column in (
            ('Isolation_Forest', 'Isolation Forest', 'ml_score', 'ml_alert'),
            ('LOF', 'LOF', 'score', 'alert')):
        selection_path = checked_path(f'ML/Results/{folder}/selection.json')
        selection = json.loads(selection_path.read_text(encoding='utf-8-sig'))['models']
        prediction_path = checked_path(f'ML/Results/{folder}/predictions_N01_A01.csv')
        seen = {view: set() for view in VIEWS}
        counters = {(view, scope): Counter() for view in VIEWS
                    for scope in ('validation', 'matched_development')}
        errors = Counter()
        for row in rows(prediction_path):
            view, identity = row['feature_set'], key(row)
            if view not in seen:
                errors['unexpected_views'] += 1
                continue
            errors['duplicate_prediction_keys'] += identity in seen[view]
            seen[view].add(identity)
            metadata = (int(row['label']), row['split'])
            if windows.get(identity) != metadata:
                errors['prediction_metadata_mismatches'] += 1
            fitting = metadata == (0, 'train')
            if detector == 'LOF' and fitting:
                errors['invalid_fitting_rows'] += not (
                    row['status'] == 'fitting_row_not_scored'
                    and row[score_column] == '' and row[alert_column] == '')
                continue
            if row[alert_column] not in ('True', 'False'):
                errors['invalid_decisions'] += 1
                continue
            alert = row[alert_column] == 'True'
            score = float(row[score_column])
            errors['invalid_scores'] += not math.isfinite(score)
            errors['threshold_decision_mismatches'] += alert != (score >= selection[view]['threshold'])
            if detector == 'LOF':
                errors['invalid_scored_status'] += row['status'] != 'valid'
            label, split = metadata
            errors['invalid_labels'] += label not in (-1, 0, 1)
            if split == 'validation':
                counters[(view, 'validation')][(label, alert)] += 1
            if identity in cohort:
                counters[(view, 'matched_development')][(label, alert)] += 1
        for name in ('unexpected_views', 'duplicate_prediction_keys',
                     'prediction_metadata_mismatches', 'invalid_fitting_rows',
                     'invalid_decisions', 'invalid_scores', 'threshold_decision_mismatches',
                     'invalid_scored_status', 'invalid_labels'):
            check(f'{detector}:{name}', errors[name], 0)
        for view in VIEWS:
            check(f'{detector}:{view}:window_key_mismatches',
                  len(seen[view].symmetric_difference(windows)), 0)
            for scope in ('validation', 'matched_development'):
                actual = decision_metrics(counters[(view, scope)])
                calculated.append(dict(detector=detector, feature_set=view, scope=scope, **actual))
                reference = saved_metrics.get((detector, view, scope))
                if reference is None:
                    continue  # The protocol-key check above records the missing row.
                for name, value in actual.items():
                    expected = int(reference[name]) if isinstance(value, int) else float(reference[name])
                    check(f'{detector}:{view}:{scope}:{name}', value, expected)

    return dict(passed=all(item['passed'] for item in checks), check_count=len(checks),
                checks=checks, input_sha256=files, calculated_decision_metrics=calculated,
                scope='Frozen input counts, window identities, saved thresholds and decisions, '
                      'classification metrics, and unknown-alert workload; no model execution, '
                      'score reproduction, ranking metrics, event coverage, or latency checks.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, default=DEFAULT_INPUTS)
    parser.add_argument('--output', type=Path, help='New JSON report file; otherwise print to stdout')
    args = parser.parse_args()
    if args.output and args.output.exists():
        parser.error('--output must name a new file')
    report = verify(args.inputs.resolve())
    rendered = json.dumps(report, indent=2, allow_nan=False) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x', encoding='utf-8') as stream:
            stream.write(rendered)
        print(json.dumps(dict(passed=report['passed'], check_count=report['check_count'],
                              output=str(args.output.resolve()))))
    else:
        print(rendered, end='')
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
