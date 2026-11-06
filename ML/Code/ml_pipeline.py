import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]

REQUIRED_ARTIFACTS = {'Datasets/N01/telemetry_observations.csv': 'b31baac556d61cec66da21481015ddcae16617c1fb0ae3565bcc951a89a459ab',
 'Datasets/N01/campaign_metadata.json': '5b85495537101a4cab88f275cf27e7360fe0accdc88eeb58c463a38f3f94fe4c',
 'Datasets/A01/telemetry_observations.csv': '9347a8d86109721dd3fd5dbacede2b7518f45d9cae2fce70b72981984347cebd',
 'Datasets/A01/campaign_metadata.json': 'efc17b6ae768920b2c5eede81083b3ab3d694856e2fa4c4115c3e408c7810dee',
 'Datasets/Processed/windows_N01_A01.csv': 'c3c31cd3926da681aa0c25701914e71bc0614af3f963473978de7e3199fdde8e',
 'Datasets/Processed/events_A01.json': 'f3c34b3b4c11accad394cf0ee0f418bfbb678106212112a1711d464ae5ccba80',
 'ML/Results/Isolation_Forest/predictions_N01_A01.csv': 'bf0224d1580c5ac0d450ed8ac9e5bd0da5fc4c8ad7463d76dfc41af123063408',
 'ML/Results/LOF/predictions_N01_A01.csv': 'c5138651c9199a7e6d529a406085f51c7d272e65ee9200e7bfd5ad426897ca8c',
 'ML/Results/Isolation_Forest/models/network.joblib': 'c16564c7b960517f8dae851d2ce229768a6d9378067a35066c6c545a6f7f1e4b',
 'ML/Results/Isolation_Forest/models/system.joblib': '560b6fd2fc4b51620c4a6d2b4c70fc9068360ba4c6186f63c60101ddf8772ec5',
 'ML/Results/Isolation_Forest/models/combined.joblib': '9e44b839b06b6c08d9809b55c86dfd7d4b374e260160b13a443084c38393e07d',
 'ML/Results/LOF/models/network.joblib': '6f5c13a7247ba078620728298feef2c1802cae62891af728a6530da9e966a390',
 'ML/Results/LOF/models/system.joblib': '1ec405ad585f3dc5c23f2689b60eb0eae992394f80fbb00683ace6d57d60e58b',
 'ML/Results/LOF/models/combined.joblib': '9bbfb6227a62c898c00efecf8afee2200eeaa6bcff4d2ad7d87ab7bffe79cab5'}
CAMPAIGNS = {'normal_20260923T161519Z', 'vuln_20260924T150444Z'}
VIEWS = ['network', 'system', 'combined']
KEYS = ['campaign', 'topic', 'window_start', 'window_end']


# Fixed study settings
IF_CONFIG = {'schema': 'september_legacy_v1',
 'pipeline_version': 2,
 'feature_sets': ['network', 'system', 'combined'],
 'prediction_reporting': {'version': 1, 'voting': False, 'agreement_is_diagnostic_only': True},
 'campaigns': ['normal_20260923T161519Z', 'vuln_20260924T150444Z'],
 'window_seconds': 5,
 'guard_seconds': 120,
 'validation_fraction': 0.3,
 'normal_validation_fpr': 0.01,
 'model': {'n_estimators': 200, 'max_samples': 1024, 'max_features': 0.7, 'random_state': 42},
 'design': 'One workflow, three separate feature views with the same fixed IF settings and 1% '
           'normal-validation calibration target. Settings originated in historical combined-model '
           'development; this is not a new zero-shot experiment. No search, model voting or use of '
           'adjudicated unknowns for training/calibration. This network calibration is not the '
           'historical V2 0.5% policy.',
 'evidence_audit': {'policy_version': 'unknown_exposure_v1',
                    'coverage_tests': ['vitals_integrity_attack', 'mqtt_protocol_metrics'],
                    'receipt_time_precision_seconds': 1,
                    'automatic_normal_adjudication': False,
                    'benchmark_labels_immutable': True},
 'evidence_review': {'version': 'background_review_v1',
                     'episode_max_seconds': 900,
                     'diagnostic_gap_seconds': 10,
                     'automatic_normal_adjudication': False,
                     'original_labels_immutable': True,
                     'predictions_used': False}}

LOF_CONFIG = {'experiment': 'LOF across the three N01/A01 feature views',
 'status': 'Original numerical settings retained; standalone execution uses N01/A01',
 'views': ['network', 'system', 'combined'],
 'model': {'n_neighbors': 20,
           'algorithm': 'brute',
           'leaf_size': 30,
           'metric': 'minkowski',
           'p': 2,
           'metric_params': None,
           'contamination': 'auto',
           'novelty': True,
           'n_jobs': 2},
 'thread_limit': 4,
 'score_batch_rows': 4096,
 'training': 'Exactly the canonical label=0, split=train windows; no subsampling or deduplication',
 'preprocessing': 'Reuse the fitted canonical IF preprocessing and variance selection without '
                  'refitting, including the existing System topic-class treatment',
 'calibration': 'Exactly the canonical label=0, split=validation windows',
 'normal_validation_fpr': 0.01,
 'threshold': 'nextafter(quantile(-score_samples, 0.99, method=higher), +inf); alert when score >= '
              'threshold',
 'default_offset': 'The LOF contamination offset is not used for calibration or decisions',
 'model_choice': 'Fixed 20-neighbor Euclidean novelty LOF; no parameter search or result-dependent '
                 'revision; brute force is used consistently across views',
 'fitting_row_scores': 'Not evaluated with novelty score_samples; retain rows with null '
                       'score/alert and fitting_row_not_scored status',
 'evaluations': ['All N01/A01 non-fitting windows; calibration and matched development reported '
                 'separately',
                 'Exactly the saved 1,078-window development cohort'],
 'event_policy': 'An event is detected when any eligible exposure window in the matched cohort is '
                 'flagged',
 'unknown_policy': 'Unknown labels are excluded from classification metrics and reported as alert '
                   'workload separately',
 'claim_scope': 'Retrospective baseline extension on previously examined recordings; not new '
                'prospective confirmation',
 'preservation': 'Saved scientific artifacts remain unchanged; all new outputs use a separate new '
                 'directory',
 'llm_policy': 'Existing LLM results are preserved; new inference runs require explicit task '
               'commands',
 'reference': 'https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html'}


# Package integrity and output handling
def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False,default=str)+'\n',encoding='utf-8')


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def require(condition,message):
    if not condition:raise ValueError(message)


def new_output(path,create=True):
    path=Path(path).expanduser().resolve()
    require(not path.exists(),'Output already exists; choose a new directory')
    if path.is_relative_to(ROOT):
        require(path.is_relative_to(ROOT/'runs') and path!=ROOT/'runs',
                'Inside this package, outputs must be new subdirectories of runs/')
    require(not ROOT.is_relative_to(path),'Output may not contain the package')
    if create:path.mkdir(parents=True,exist_ok=False)
    return path


def scientific():
    import importlib.metadata
    expected={'numpy':'2.5.2','pandas':'3.0.5','scikit-learn':'1.9.0','joblib':'1.6.0','scipy':'1.18.1','threadpoolctl':'3.6.0'}
    mismatch={k:importlib.metadata.version(k) for k,v in expected.items() if importlib.metadata.version(k)!=v}
    require(not mismatch,'Install requirements.txt into an isolated environment. Version mismatch: '+str(mismatch))


def seal(folder):
    folder=Path(folder)
    write(folder/'manifest.json',{p.relative_to(folder).as_posix():sha(p) for p in sorted(folder.rglob('*')) if p.is_file() and p!=folder/'manifest.json'})


def verify():
    """Check the fixed scientific inputs without packaging or audit dependencies."""
    for relative, checksum in REQUIRED_ARTIFACTS.items():
        path = (ROOT / relative).resolve()
        require(path.is_relative_to(ROOT) and path.is_file(), 'Required artifact missing: ' + relative)
        require(sha(path) == checksum, 'Scientific artifact changed: ' + relative)
    return {'status': 'PASS', 'required_artifacts_verified': len(REQUIRED_ARTIFACTS), 'recordings': ['N01', 'A01']}


def scientific_imports():
    """Load the pinned libraries only when a scientific command is requested."""
    scientific()
    global np, pd, sklearn, joblib, threadpool_limits
    global ColumnTransformer, SimpleImputer, Pipeline, OneHotEncoder, StandardScaler
    global IsolationForest, VarianceThreshold, LocalOutlierFactor
    import numpy as np
    import pandas as pd
    import sklearn
    import joblib
    from threadpoolctl import threadpool_limits
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.ensemble import IsolationForest
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.neighbors import LocalOutlierFactor



# Window features, evidence labels and chronological partitions
PRIMARY_KINDS = {"replay", "spoofed_device", "retained_abuse", "topic_misuse",
                 "topic_device_mismatch", "payload_size", "malformed_payload", "timing_jitter"}


RAW_FEATURES = ["mqtt_msg_rate_per_min", "avg_msg_size_bytes", "payload_entropy",
                "inter_arrival_ms", "flow_bytes_per_sec", "topic_device_mismatch",
                "event_timestamp_skew_seconds"]


SYSTEM_FEATURES = ["wifi_rssi_dbm", "free_heap_bytes", "cpu_temp_f"]


def feature_columns(feature_set):
    network = [f'{c}__{stat}' for c in RAW_FEATURES + ['malformed_increment']
               for stat in ('mean', 'std', 'min', 'max')] + ['message_count', 'observed_messages_per_second']
    system = [f'{c}__{stat}' for c in SYSTEM_FEATURES for stat in ('mean', 'std', 'min', 'max')]
    if feature_set not in {'network', 'system', 'combined'}:
        raise ValueError(f'Unknown feature set: {feature_set}')
    return network if feature_set == 'network' else system if feature_set == 'system' else network + system


def stamp(value):
    ts = pd.to_datetime(value, utc=True, errors="raise")
    if pd.isna(ts):
        raise ValueError(f"Missing timestamp: {value!r}")
    return ts


def overlaps(windows, start, end):
    """Half-open windows; evidence end is exclusive."""
    return (windows.window_start < end) & (windows.window_end > start)


def load_events(campaign):
    """Read ALL manifest tests, validate hashes, retain excluded checks too."""
    events = []
    manifests = sorted(campaign.glob("kali_results/*/manifest.json"))
    marker_path = campaign / "kali_results/.kali_complete.json"
    if not marker_path.exists():
        raise ValueError(f"Missing completion marker: {campaign.name}")
    marker = read(marker_path)
    if (marker.get("status") != "complete" or marker.get("returncode") != 0
            or marker.get("dry_run") is not False or marker.get("missing_by_burst")
            or marker.get("manifest_count") != len(manifests)):
        raise ValueError(f"Incomplete/inconsistent campaign marker: {campaign.name}")
    for path in manifests:
        manifest = read(path)
        for test in manifest.get("tests", []):
            start, end = stamp(test["started_at"]), stamp(test["finished_at"])
            if end < start:
                raise ValueError(f"Reversed event timestamps: {path}")
            # Coverage manifests retain absolute Pi/Kali paths after copying.
            # Resolve only their basename inside this burst.
            result_name = str(test.get("json_output", "missing")).replace("\\", "/").rsplit("/", 1)[-1]
            result_path = (path.parent / result_name).resolve()
            if not result_path.is_relative_to(path.parent.resolve()):
                raise ValueError(f"Result path escapes manifest directory: {path}")
            result = read(result_path) if result_path.is_file() else {}
            expected_hash = test.get("json_sha256")
            if expected_hash and (not result_path.is_file() or sha(result_path) != expected_hash):
                raise ValueError(f"Evidence SHA-256 mismatch: {result_path}")
            kind = str(result.get("kind", test.get("attack_label", test.get("name"))))
            primary = manifest.get("attack_mode") == "randomized" and manifest.get("evaluation_role") != "coverage_only"
            messages = result.get("messages", [])
            acked = [m for m in messages if m.get("publish_ack") is True]
            cleanup = result.get("cleanup", {})
            reason = "eligible_exposure"
            if not primary or kind not in PRIMARY_KINDS:
                reason = "coverage_only_or_nonprimary"
            elif manifest.get("dry_run") is not False:
                reason = "dry_run_or_unknown_execution"
            elif not expected_hash:
                reason = "missing_evidence_hash"
            elif cleanup.get("required") and cleanup.get("success") is not True:
                reason = "cleanup_failed_or_unconfirmed"
            elif test.get("returncode") != 0 or result.get("success") is not True or not acked:
                reason = "failed_or_unconfirmed_execution"
            topics = sorted({str(m["topic"]) for m in acked})
            # The legacy collector subscription is phms/+/data, including shadow IDs.
            visible = [t for t in topics if len(t.split("/")) == 3
                       and t.split("/")[0] == "phms" and t.split("/")[2] == "data"]
            if reason == "eligible_exposure" and not visible:
                reason = "outside_collector_subscription"
            if acked:
                first_publish = min(stamp(m["timestamp"]) for m in acked)
                last_publish = max(stamp(m["timestamp"]) for m in acked)
                if first_publish < start or last_publish > end:
                    raise ValueError(f"Publish outside execution interval: {result_path}")
            else:
                first_publish = start
            # Legacy receipts have only whole-second precision. Expand outward
            # to that resolution, without pretending exact payload attribution.
            exposure_start = first_publish.floor("s")
            exposure_end = end.ceil("s")
            if exposure_end <= exposure_start:
                exposure_end = exposure_start + pd.Timedelta(seconds=1)
            events.append(dict(campaign=campaign.name, event_id=f"{path.parent.name}/{test['name']}",
                               burst=path.parent.name, attack_type=kind, reason=reason,
                               start=start, end=end, first_publish=first_publish,
                               exposure_start=exposure_start, exposure_end=exposure_end,
                               topics=visible, attempted_topics=topics, result=str(result_path),
                               evidence_sha256=sha(result_path) if result_path.is_file() else None))
    if not events:
        raise ValueError(f"No manifest events: {campaign}")
    return events


def engineer_windows(raw, campaign_name, seconds, include_system=False):
    """Only current/past window observations; no interpolation across windows."""
    raw = raw.sort_values("timestamp", kind="stable").copy()
    raw["window_start"] = raw.timestamp.dt.floor(f"{seconds}s")
    raw["topic"] = raw.mqtt_topic
    signals = RAW_FEATURES + (SYSTEM_FEATURES if include_system else [])
    for col in signals:
        raw[col] = pd.to_numeric(raw[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    # Cumulative malformed counters leak history; model only observed increments.
    counter = pd.to_numeric(raw.malformed_payload_count, errors="coerce")
    raw["malformed_increment"] = counter.groupby(raw.topic).diff().clip(lower=0)
    features = signals + ["malformed_increment"]
    g = raw.groupby(["topic", "window_start"], sort=True)
    windows = g[features].agg(["mean", "std", "min", "max"])
    windows.columns = [f"{col}__{stat}" for col, stat in windows.columns]
    windows["message_count"] = g.size()
    windows["observed_messages_per_second"] = windows.message_count / seconds
    windows = windows.reset_index()
    windows["topic_class"] = np.where(windows.topic.isin(["phms/patient_a/data", "phms/patient_b/data"]),
                                      "canonical_data", "other_data")
    windows["campaign"] = campaign_name
    windows["window_end"] = windows.window_start + pd.Timedelta(seconds=seconds)
    return windows


def label_windows(w, events, mode, guard_seconds):
    w = w.copy()
    w["label"] = 0 if mode == "normal" else -1
    w["state"] = "normal" if mode == "normal" else "attack_campaign_background_unknown"
    w["event_ids"], w["attack_types"] = "", ""
    guard = pd.Timedelta(seconds=guard_seconds)
    # Background near ANY check is not clean background, even on another topic.
    for event in events:
        mask = overlaps(w, event["start"] - guard, event["end"] + guard)
        w.loc[mask, "state"] = "event_guard_unknown"
    for event in events:
        if event["reason"] != "eligible_exposure":
            continue
        mask = overlaps(w, event["exposure_start"], event["exposure_end"]) & w.topic.isin(event["topics"])
        w.loc[mask, "label"] = 1
        w.loc[mask, "state"] = "attack_exposure"
        for col, value in [("event_ids", event["event_id"]), ("attack_types", event["attack_type"])]:
            w.loc[mask, col] = w.loc[mask, col].map(lambda old: "|".join(sorted(set(filter(None, old.split("|") + [value])))))
    # Coverage/failed tests override positives if intervals overlap.
    for event in events:
        if event["reason"] == "eligible_exposure":
            continue
        start = event["start"] if event["reason"] == "outside_collector_subscription" else event["start"] - guard
        end = w.window_end.max() if event["reason"] == "cleanup_failed_or_unconfirmed" else event["end"] + guard
        mask = overlaps(w, start, end)
        w.loc[mask, ["label", "state", "event_ids", "attack_types"]] = [-1, "excluded_check_guard", "", ""]
    return w


def split_data(w, audit, events, validation_fraction, guard_seconds):
    expected = {'normal_20260923T161519Z': 'normal', 'vuln_20260924T150444Z': 'attack'}
    if {a['campaign']: a['mode'] for a in audit} != expected:
        raise ValueError('Exactly the packaged normal and attack recordings are required')
    w = w.copy()
    w['split'] = 'purged'
    cuts = {}
    for campaign in audit:
        name = campaign['campaign']
        cut = campaign['start'] + (campaign['end'] - campaign['start']) * (1 - validation_fraction)
        if campaign['mode'] == 'attack':
            bounds = {}
            for e in [e for e in events if e['campaign'] == name]:
                lo, hi = bounds.get(e['burst'], (e['start'], e['end']))
                bounds[e['burst']] = (min(lo, e['start']), max(hi, e['end']))
            values = sorted(bounds.values())
            gaps = [a[1] + (b[0]-a[1])/2 for a,b in zip(values,values[1:]) if a[1]<b[0]]
            if not gaps: raise ValueError('No complete burst boundary for the split')
            cut = min(gaps, key=lambda t: abs(t-cut))
        gap = pd.Timedelta(seconds=guard_seconds)
        belongs = w.campaign.eq(name)
        w.loc[belongs & (w.window_end <= cut-gap),'split'] = 'train'
        w.loc[belongs & (w.window_start >= cut+gap),'split'] = 'validation'
        cuts[name] = cut
    for event in events:
        cut=cuts[event['campaign']]; gap=pd.Timedelta(seconds=guard_seconds)
        event['split']='train' if event['end']<=cut-gap else 'validation' if event['start']>=cut+gap else 'purged'
    for identity, group in pd.DataFrame(events).groupby(['campaign','burst']):
        if len(set(group.split)-{'purged'})>1: raise ValueError(f'Burst crosses split: {identity}')
    for split in ['train','validation']:
        if set(w.loc[w.split.eq(split)&w.label.ge(0),'label'])!={0,1}:
            raise ValueError(f'{split} must contain both observed classes')
    return w,cuts


def make_preprocessor(numeric):
    return ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
                              ("scale", StandardScaler())]), numeric),
        ("category", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["topic_class"]),
    ], sparse_threshold=0)


def normal_threshold(scores, fpr):
    """Strictly above selected normal quantile; ties do not inflate FPR."""
    return float(np.nextafter(np.quantile(scores, 1 - fpr, method="higher"), np.inf))


# Independent consistency observations
RULE_ID = 'topic_device_consistency_v1'


MISMATCH_REASON = 'payload_device_id_differs_from_topic_device_id'


@dataclass(frozen=True)
class MessageResult:
    alert: bool
    status: str
    reason: str


def check_message(topic, payload_device_id):
    """Case-sensitive equality; no registry, attack label or ML score is used."""
    if not isinstance(topic, str):
        return MessageResult(False, 'not_evaluable', 'missing_topic')
    parts = topic.split('/')
    if len(parts) != 3 or parts[0] != 'phms' or not parts[1] or parts[2] != 'data' or any(c in parts[1] for c in ('+', '#')):
        return MessageResult(False, 'not_evaluable', 'unexpected_topic_shape')
    if not isinstance(payload_device_id, str) or not payload_device_id.strip():
        return MessageResult(False, 'not_evaluable', 'missing_payload_device_id')
    if parts[1] != payload_device_id:
        return MessageResult(True, 'mismatch_observed', MISMATCH_REASON)
    return MessageResult(False, 'no_mismatch_observed', '')


def aggregate_legacy_flags(raw, campaign, source_path, seconds=5):
    """One alert per topic window if any recorded mismatch flag equals one."""
    data = raw[['timestamp', 'mqtt_topic', 'device_id', 'topic_device_mismatch']].copy()
    timestamps = pd.to_datetime(data.timestamp, utc=True, errors='raise', format='mixed')
    if timestamps.isna().any() or data.mqtt_topic.isna().any():
        raise ValueError('Missing timestamp/topic in rule input')
    flag = pd.to_numeric(data.topic_device_mismatch, errors='raise')
    if not flag.dropna().isin([0, 1]).all():
        raise ValueError('Mismatch flags must be 0, 1 or missing')
    data['campaign'] = campaign
    data['topic'] = data.mqtt_topic
    data['window_start'] = timestamps.dt.floor(f'{seconds}s')
    data['line'] = np.arange(len(data)) + 2
    data['observed'] = flag.notna().astype(int)
    data['mismatch'] = flag.eq(1).astype(int)
    for row in data[data.mismatch.eq(1)].itertuples():
        if not check_message(row.topic, row.device_id).alert:
            raise ValueError(f'Recorded mismatch conflicts with observed identifiers: {source_path}:{row.line}')
    keys = ['campaign', 'topic', 'window_start']
    result = data.groupby(keys, sort=True).agg(
        rule_raw_rows=('line', 'size'), rule_observed_rows=('observed', 'sum'),
        rule_mismatch_rows=('mismatch', 'sum')).reset_index()
    result['window_end'] = result.window_start + pd.Timedelta(seconds=seconds)
    result['rule_missing_rows'] = result.rule_raw_rows - result.rule_observed_rows
    result['consistency_alert'] = result.rule_mismatch_rows.gt(0)
    result['consistency_status'] = np.where(result.consistency_alert, 'mismatch_observed',
        np.where(result.rule_observed_rows.eq(0), 'not_evaluable',
                 np.where(result.rule_missing_rows.gt(0), 'partially_observed_no_mismatch', 'no_mismatch_observed')))
    result['consistency_reason'] = np.where(result.consistency_alert, MISMATCH_REASON,
        np.where(result.rule_missing_rows.gt(0), 'missing_mismatch_observations', ''))
    refs = data[data.mismatch.eq(1)].groupby(keys).line.agg(list).to_dict()
    import json
    result['rule_supporting_references'] = [json.dumps([f'{source_path}:{line}' for line in refs.get((r.campaign, r.topic, r.window_start), [])]) for r in result.itertuples()]
    result['consistency_rule_id'] = RULE_ID
    return result


# Raw recording preparation
def read_recording(path):
    raw = pd.read_csv(path, low_memory=False)
    required = set(RAW_FEATURES + SYSTEM_FEATURES + [
        'timestamp', 'mqtt_topic', 'device_id', 'malformed_payload_count', 'collection_session_id'])
    if not required.issubset(raw):
        raise ValueError(f'Missing September columns in {path}: {sorted(required - set(raw))}')
    if {'observation_schema_version', 'collector_run_id'} & set(raw):
        raise ValueError('Receipt-aware schema needs a new evidence adapter')
    if raw.empty or raw.collection_session_id.isna().any() or raw.collection_session_id.nunique() != 1:
        raise ValueError('Expected one nonempty collection session per CSV')
    if not raw.timestamp.astype(str).str.contains(r'(?:Z|[+-]\d{2}:\d{2})$', regex=True).all():
        raise ValueError('Explicit UTC/offset timestamps required')
    duplicates = int(raw.duplicated().sum())
    raw['timestamp'] = pd.to_datetime(raw.timestamp, utc=True, format='mixed', errors='raise')
    if raw.timestamp.isna().any() or raw.mqtt_topic.isna().any() or raw.mqtt_topic.eq('').any():
        raise ValueError('Invalid timestamp/topic')
    return raw, duplicates


def prepare(dataset_dir, config):
    frames, rules, audits, events, digests = [], [], [], [], set()
    for name in config['campaigns']:
        if Path(name).name != name or '202609' not in name:
            raise ValueError(f'Only explicit September campaign names are supported: {name}')
        folder = dataset_dir / {'normal_20260923T161519Z':'N01','vuln_20260924T150444Z':'A01'}[name]
        path = folder / 'telemetry_observations.csv'
        digest = sha(path)
        if digest in digests:
            raise ValueError('Duplicate recording copy')
        digests.add(digest)
        mode = read(folder / 'campaign_metadata.json').get('mode')
        if mode not in {'normal', 'attack'}:
            raise ValueError(f'Invalid campaign mode: {name}')
        raw, duplicates = read_recording(path)
        campaign_events = load_events(folder) if mode == 'attack' else []
        for event in campaign_events:
            event['campaign'] = name
        events.extend(campaign_events)
        windows = engineer_windows(raw, name, config['window_seconds'], include_system=True)
        windows = label_windows(windows, campaign_events, mode, config['guard_seconds'])
        start, end = raw.timestamp.min(), raw.timestamp.max()
        edge = (windows.window_start < start + pd.Timedelta(seconds=config['guard_seconds'])) | (windows.window_end > end)
        windows.loc[edge, ['label', 'state', 'event_ids', 'attack_types']] = [-1, 'recording_edge', '', '']
        frames.append(windows)
        rules.append(aggregate_legacy_flags(raw, name, str(path.resolve()), config['window_seconds']))
        audits.append(dict(campaign=name, mode=mode, rows=len(raw), windows=len(windows),
                           start=start, end=end, sha256=digest, identical_rows_preserved=duplicates))
        print(f'Prepared {name}: {len(raw):,} rows, {len(windows):,} windows', flush=True)
    windows = pd.concat(frames, ignore_index=True)
    windows, cuts = split_data(windows, audits, events, config['validation_fraction'], config['guard_seconds'])
    keys = ['campaign', 'topic', 'window_start', 'window_end']
    windows = windows.merge(pd.concat(rules, ignore_index=True), on=keys, validate='one_to_one')
    if len(windows) != sum(a['windows'] for a in audits) or windows.consistency_alert.isna().any():
        raise ValueError('Rule and feature windows differ')
    return windows, audits, events, cuts


# Isolation Forest and novelty LOF
def model_input(windows, numeric, feature_set):
    frame = windows[numeric + ['topic_class']].copy()
    if feature_set == 'system':
        frame['topic_class'] = 'unused_constant'
    return frame


def fit_model(windows, config, jobs=2, feature_set='combined'):
    train = windows[windows.split.eq('train') & windows.label.eq(0)]
    validation = windows[windows.split.eq('validation') & windows.label.eq(0)]
    if train.empty or validation.empty:
        raise ValueError('Normal training and validation windows are required')
    numeric = [c for c in feature_columns(feature_set) if train[c].notna().any()]
    model = Pipeline([('preprocess', make_preprocessor(numeric)), ('select', VarianceThreshold()),
                      ('model', IsolationForest(**config['model'], n_jobs=jobs))])
    model.fit(model_input(train, numeric, feature_set))
    threshold = normal_threshold(-model.score_samples(model_input(validation, numeric, feature_set)), config['normal_validation_fpr'])
    names = model.named_steps['preprocess'].get_feature_names_out()
    retained = names[model.named_steps['select'].get_support()].tolist()
    families = [RAW_FEATURES, SYSTEM_FEATURES] if feature_set == 'combined' else [SYSTEM_FEATURES if feature_set == 'system' else RAW_FEATURES]
    for family in families:
        if not any(any(f'__{base}__' in c for base in family) for c in retained):
            raise ValueError('Variance filtering removed a required feature family')
    return dict(pipeline=model, threshold=threshold, numeric_features=numeric, retained_features=retained,
                fitting_rows=len(train), calibration_rows=len(validation), config=config,
                training_campaigns=sorted(train.campaign.unique()), sklearn_version=sklearn.__version__,
                feature_set=feature_set, score_direction='higher_is_more_anomalous')


def score_lof(bundle,windows,batch_rows=4096):
    scores=[]
    for first in range(0,len(windows),batch_rows):
        part=windows.iloc[first:first+batch_rows]
        x=bundle['preprocessing'].transform(model_input(part,bundle['numeric_features'],bundle['feature_set']))
        scores.append(-bundle['model'].score_samples(x))
    return np.concatenate(scores) if scores else np.array([],dtype=float)


def fit_lof(windows,frozen,config,threshold_function):
    train=windows[windows.split.eq('train')&windows.label.eq(0)]
    cal=windows[windows.split.eq('validation')&windows.label.eq(0)]
    view=frozen['feature_set'];preprocessing=frozen['pipeline'][:2]
    model=LocalOutlierFactor(**config['model'])
    model.fit(preprocessing.transform(model_input(train,frozen['numeric_features'],view)))
    bundle={'preprocessing':preprocessing,'model':model,'numeric_features':frozen['numeric_features'],
        'retained_features':frozen['retained_features'],'feature_set':view,'score_direction':'higher_is_more_anomalous',
        'fitting_rows':len(train),'calibration_rows':len(cal),'sklearn_version':sklearn.__version__}
    bundle['threshold']=threshold_function(score_lof(bundle,cal,config['score_batch_rows']),config['normal_validation_fpr'])
    return bundle


# Rebuild, train and score against the frozen study artifacts
def frame(path):
    f=pd.read_csv(path,float_precision='round_trip',low_memory=False)
    for col in f.select_dtypes(include=['object','str']):f[col]=f[col].fillna('')
    require(set(f.campaign)==CAMPAIGNS,'Input must contain only the two packaged recordings')
    require(not f.duplicated(KEYS).any(),'Duplicate physical windows')
    require(len(f)==71348,'Expected 71,348 N01/A01 windows')
    return f


def prepare_data(output):
    scientific_imports()
    w,audit,events,cuts=prepare(ROOT/'Datasets',IF_CONFIG)
    saved=frame(ROOT/'Datasets/Processed/windows_N01_A01.csv')
    # Path-bearing receipt references are provenance, not detector inputs.
    saved=saved.set_index(KEYS).sort_index()
    w['window_start']=w.window_start.astype(str);w['window_end']=w.window_end.astype(str)
    current=w.set_index(KEYS).sort_index()
    require(current.index.equals(saved.index),'Regenerated window identities differ')
    for col in saved.columns:
        if col=='rule_supporting_references':continue
        if pd.api.types.is_numeric_dtype(saved[col]):
            require(np.allclose(current[col].astype(float),saved[col].astype(float),rtol=1e-12,atol=1e-12,equal_nan=True),'Regenerated values differ: '+col)
        else:require(current[col].fillna('').astype(str).equals(saved[col].fillna('').astype(str)),'Regenerated metadata differs: '+col)
    w.to_csv(output/'windows.csv',index=False)
    write(output/'dataset_audit.json',audit);write(output/'events.json',events);write(output/'split_cuts.json',cuts)
    result={'status':'PASS','windows':len(w),'observations':sum(a['rows'] for a in audit),
            'saved_features_labels_and_splits_reproduced':True,'path_provenance_excluded_from_numeric_comparison':True}
    write(output/'VALIDATION.json',result);seal(output);return result


def models(output,retrain,windows_path=None,jobs=2):
    scientific_imports()
    require(jobs>0,'--jobs must be positive')
    w=frame(windows_path or ROOT/'Datasets/Processed/windows_N01_A01.csv')
    cfg=IF_CONFIG
    lc=LOF_CONFIG
    fitting=w.split.eq('train')&w.label.eq(0)
    bundles={};comparison=[];outputs=[]
    (output/'models').mkdir()
    # Freeze all six models before evaluation when refitting.
    with threadpool_limits(limits=4):
        for view in VIEWS:
            print(('Fitting ' if retrain else 'Loading ')+view+' models',flush=True)
            ib=fit_model(w,cfg,jobs,view) if retrain else joblib.load(ROOT/f'ML/Results/Isolation_Forest/models/{view}.joblib')
            lb=fit_lof(w,ib,lc,normal_threshold) if retrain else joblib.load(ROOT/f'ML/Results/LOF/models/{view}.joblib')
            bundles[('IF',view)]=ib;bundles[('LOF',view)]=lb
            for family,b in [('IF',ib),('LOF',lb)]:
                folder=output/'models'/family;folder.mkdir(exist_ok=True)
                joblib.dump(b,folder/f'{view}.joblib')
        for (family,view),b in bundles.items():
            print('Scoring '+family+' '+view,flush=True)
            selected=w if family=='IF' else w.loc[~fitting]
            scores=(-b['pipeline'].score_samples(model_input(selected,b['numeric_features'],view))
                    if family=='IF' else score_lof(b,selected,lc['score_batch_rows']))
            result=selected[KEYS+['label','state','event_ids','attack_types','split']].copy()
            result['feature_set']=view;result['detector']=family
            result['score']=scores;result['alert']=scores>=b['threshold'];result['status']='valid'
            source='Isolation_Forest' if family=='IF' else 'LOF'
            old=pd.read_csv(ROOT/f'ML/Results/{source}/predictions_N01_A01.csv',float_precision='round_trip',low_memory=False)
            old=old[old.feature_set.eq(view)].set_index(KEYS).loc[pd.MultiIndex.from_frame(selected[KEYS])]
            sc='ml_score' if family=='IF' else 'score';al='ml_alert' if family=='IF' else 'alert'
            score_match=np.allclose(scores,old[sc].to_numpy(dtype=float),rtol=1e-12,atol=1e-12)
            alert_match=np.array_equal(result.alert.to_numpy(),old[al].to_numpy(dtype=bool))
            require(score_match and alert_match,f'{family}/{view} differs from saved scores or alerts')
            comparison.append({'detector':family,'view':view,'scored_windows':len(result),
                'score_max_absolute_difference':float(np.max(np.abs(scores-old[sc].to_numpy(dtype=float)))),
                'decisions_match':bool(alert_match),'threshold':b['threshold']})
            if family=='LOF':
                omitted=w.loc[fitting,KEYS+['label','state','event_ids','attack_types','split']].copy()
                omitted['feature_set']=view;omitted['detector']=family;omitted['score']=np.nan
                omitted['alert']=None;omitted['status']='fitting_row_not_scored'
                result=pd.concat([result,omitted],ignore_index=True)
            outputs.append(result)
    pd.concat(outputs,ignore_index=True).to_csv(output/'predictions.csv',index=False)
    result={'status':'PASS','operation':'refit_and_score' if retrain else 'score_saved_models','comparisons':comparison}
    write(output/'VALIDATION.json',result);seal(output);return result


def run_all(output, jobs=2):
    """Rebuild raw windows, then fit and score all six models from those windows."""
    require(jobs > 0, '--jobs must be positive')
    data_output = new_output(output / 'data')
    prepared = prepare_data(data_output)
    model_output = new_output(output / 'ml')
    fitted = models(model_output, True, data_output / 'windows.csv', jobs)
    result = {'status': 'PASS', 'operation': 'prepare_refit_and_score',
              'data': prepared, 'ml': fitted}
    write(output / 'VALIDATION.json', result)
    seal(output)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('verify', help='Verify required scientific inputs')
    commands.add_parser('check-environment', help='Check pinned scientific libraries')
    for command, description in [
        ('run', 'Rebuild raw data, fit and score all six models, and compare saved results'),
        ('prepare-data', 'Rebuild N01/A01 windows, evidence labels and partitions'),
        ('train-ml', 'Fit and score all six models and compare saved results'),
        ('score-ml', 'Score the six saved models and compare saved results'),
    ]:
        sub = commands.add_parser(command, help=description)
        sub.add_argument('--output', type=Path, required=True, help='New output directory')
        if command != 'prepare-data':
            sub.add_argument('--jobs', type=int, default=2,
                             help='Isolation Forest workers (default: 2); LOF retains its fixed settings')
        if command in {'train-ml', 'score-ml'}:
            sub.add_argument('--windows', type=Path, help='Optional regenerated N01/A01 windows.csv')
    args = parser.parse_args(argv)
    if args.command == 'verify':
        result = verify()
    elif args.command == 'check-environment':
        scientific()
        result = {'status': 'PASS'}
    else:
        if hasattr(args, 'jobs'):
            require(args.jobs > 0, '--jobs must be positive')
        verify()
        output = new_output(args.output)
        try:
            if args.command == 'run':
                result = run_all(output, args.jobs)
            elif args.command == 'prepare-data':
                result = prepare_data(output)
            else:
                result = models(output, args.command == 'train-ml', args.windows, args.jobs)
        except Exception as error:
            write(output / 'FAILED.json', {'error': str(error), 'saved_results_modified': False})
            raise
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

