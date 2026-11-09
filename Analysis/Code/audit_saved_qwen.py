from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
from statistics import median
import sys

sys.dont_write_bytecode = True
from analysis_paths import PACKAGE_ROOT, new_output

DEFAULT_INPUT = PACKAGE_ROOT
MODES = ('qwen_off', 'qwen_on')
LABELS = {-1:'unknown',0:'normal',1:'exposure'}
WINDOW_FLAGS = ['flipped','claim_set_changed','feature_reference_set_changed',
    'same_claim_set_opposite_boolean_pair','same_nonempty_claim_set_opposite_boolean_pair',
    'same_supported_claim_set_opposite_boolean_pair','any_contradicted','any_unverifiable',
    'all_claims_supported_nonempty','mixed_supported_only_and_contradicted',
    'shared_reference_positions_changed','claim_correctness_status_changed',
    'any_empty_claim_response','any_duplicate_claim_occurrences']
WINDOW_COLUMNS = ['profile','view','request_id','label','label_name','campaign','topic','window_start',
    'seed_count','positive_decisions',*WINDOW_FLAGS,'opposite_boolean_pairs','same_claim_set_opposite_boolean_pairs',
    'same_feature_reference_set_opposite_boolean_pairs','claim_count','unique_claim_occurrences',
    'duplicate_claim_occurrences','supported_claims','contradicted_claims','unverifiable_claims',
    'supported_only_responses','contradicted_responses','unverifiable_responses','empty_claim_responses',
    'reason_distinct_count','score_min','score_max','score_span']
RESPONSE_COLUMNS = ['profile','view','seed','request_id','label','label_name','complete_group','window_flipped',
    'is_anomaly','score','reason','claims_json','claim_count','unique_claim_count','duplicate_claim_occurrences',
    'supported','contradicted','unverifiable','claim_correctness_status','claim_correctness_category']
PAIR_COLUMNS = ['profile','view','request_id','label','label_name','window_flipped','seed_a','seed_b',
    'is_anomaly_a','is_anomaly_b','boolean_disagreement','claim_set_equal','claim_sets_nonempty',
    'feature_reference_set_equal','shared_feature_reference_count','shared_reference_positions_changed',
    'claim_correctness_status_a','claim_correctness_status_b','claim_correctness_status_changed',
    'claim_correctness_category_a','claim_correctness_category_b','contradicted_a','contradicted_b',
    'any_contradicted','both_supported_only_nonempty','claim_count_a','claim_count_b','score_a','score_b','absolute_score_difference']


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def read_csv(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def write_csv(path,rows,fields):
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def write_json(path,value):
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')


def frac(a,b):return a/b if b else None


def bool_value(value):
    if value not in ('True','False'):raise ValueError('Invalid saved Boolean')
    return value=='True'


def audit_claims(claims,features,reference):
    checked=[]
    for c in claims:
        if c['reference'] not in ('min_max','p01_p99') or c['position'] not in ('below','within','above'):
            raise ValueError('Unsupported claim schema')
        value=features.get(c['feature'])
        bounds=reference.get(c['feature'],{})
        lo,hi=(bounds.get('min'),bounds.get('max')) if c['reference']=='min_max' else (bounds.get('p01'),bounds.get('p99'))
        if any(v is None for v in (value,lo,hi)):
            verdict='unverifiable'
        else:
            if not all(math.isfinite(v) for v in (value,lo,hi)) or lo>hi:raise ValueError('Invalid supplied numerical interval')
            position='below' if value<lo else 'above' if value>hi else 'within'
            verdict='supported' if c['position']==position else 'contradicted'
        checked.append({k:c[k] for k in ('feature','reference','position')}|{'verdict':verdict})
    return checked


def response_features(row,claims):
    counts=Counter(c['verdict'] for c in claims)
    claimset=frozenset((c['feature'],c['reference'],c['position']) for c in claims)
    feature_refs=frozenset((c['feature'],c['reference']) for c in claims)
    positions={fr:frozenset(p for f,r,p in claimset if (f,r)==fr) for fr in feature_refs}
    status='|'.join(name for name in ('supported','contradicted','unverifiable') if counts[name]) or 'no_claims'
    supported_only=bool(claims) and counts['supported']==len(claims)
    category=('no_claims' if not claims else 'contains_contradiction' if counts['contradicted']
              else 'unverifiable_without_contradiction' if counts['unverifiable'] else 'supported_only')
    return dict(row,claims=claims,claim_set=claimset,feature_reference_set=feature_refs,positions=positions,
        claim_count=len(claims),unique_claim_count=len(claimset),duplicate_claim_occurrences=len(claims)-len(claimset),
        supported=counts['supported'],contradicted=counts['contradicted'],unverifiable=counts['unverifiable'],
        claim_correctness_status=status,claim_correctness_category=category,supported_only=supported_only)


def make_pair(a,b,window_flipped):
    shared=a['feature_reference_set']&b['feature_reference_set']
    return {'profile':a['profile'],'view':a['view'],'request_id':a['request_id'],'label':a['label'],
        'label_name':LABELS[a['label']],'window_flipped':window_flipped,'seed_a':a['seed'],'seed_b':b['seed'],
        'is_anomaly_a':a['is_anomaly'],'is_anomaly_b':b['is_anomaly'],
        'boolean_disagreement':a['is_anomaly']!=b['is_anomaly'],
        'claim_set_equal':a['claim_set']==b['claim_set'],
        'claim_sets_nonempty':bool(a['claim_set']) and bool(b['claim_set']),
        'feature_reference_set_equal':a['feature_reference_set']==b['feature_reference_set'],
        'shared_feature_reference_count':len(shared),
        'shared_reference_positions_changed':any(a['positions'][fr]!=b['positions'][fr] for fr in shared),
        'claim_correctness_status_a':a['claim_correctness_status'],'claim_correctness_status_b':b['claim_correctness_status'],
        'claim_correctness_status_changed':a['claim_correctness_status']!=b['claim_correctness_status'],
        'claim_correctness_category_a':a['claim_correctness_category'],'claim_correctness_category_b':b['claim_correctness_category'],
        'contradicted_a':a['contradicted'],'contradicted_b':b['contradicted'],
        'any_contradicted':bool(a['contradicted'] or b['contradicted']),
        'both_supported_only_nonempty':a['supported_only'] and b['supported_only'],
        'claim_count_a':a['claim_count'],'claim_count_b':b['claim_count'],
        'score_a':a['score'],'score_b':b['score'],'absolute_score_difference':abs(a['score']-b['score'])}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',type=Path,default=DEFAULT_INPUT)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();package=args.inputs.expanduser().resolve()
    destination=args.output.expanduser().resolve()
    if destination.is_relative_to(package):
        runs=package/'runs'
        if not destination.is_relative_to(runs) or destination==runs:
            raise ValueError('Inside the selected input root, outputs must be new paths under runs/')
    if package.is_relative_to(destination):
        raise ValueError('Output may not contain the selected input root')
    out=new_output(destination)
    root=package/'LLM/Results'
    protocol=json.loads((root/'Prepared/protocol.json').read_text(encoding='utf-8'))
    profiles=json.loads((root/'Prepared/profiles.json').read_text(encoding='utf-8'))
    seeds=tuple(sorted(protocol['config']['seeds']));views=tuple(protocol['config']['views'])
    assert len(seeds)>=2 and len(set(seeds))==len(seeds) and all(p in profiles for p in MODES)
    cohort_rows=read_csv(root/'Prepared/private/cohort.csv');cohort={r['request_id']:r for r in cohort_rows}
    assert len(cohort)==len(cohort_rows) and set(int(r['label']) for r in cohort_rows)<=set(LABELS)
    refs=json.loads((root/'Prepared/inference/references.json').read_text(encoding='utf-8'))
    features={}
    for view in views:
        with (root/f'Prepared/inference/{view}.jsonl').open(encoding='utf-8') as f:objects=[json.loads(line) for line in f]
        features[view]={r['request_id']:r['features'] for r in objects}
        assert len(objects)==len(features[view]) and set(features[view])==set(cohort)
    paths=[root/'Prepared/protocol.json',root/'Prepared/profiles.json',root/'Prepared/private/cohort.csv',
        root/'Prepared/tasks.json',root/'Prepared/inference/references.json',
        *[root/f'Prepared/inference/{v}.jsonl' for v in views],root/'Evaluation/decisions.csv',root/'Raw_Responses/native_responses.jsonl']
    before={p.relative_to(package).as_posix():{'sha256':sha(p),'bytes':p.stat().st_size} for p in paths}
    tasks=json.loads((root/'Prepared/tasks.json').read_text(encoding='utf-8'))
    planned=[(t['profile'],t['view'],int(t['seed']),rid) for t in tasks if t['profile'] in MODES for rid in t['request_ids']]
    planned_set=set(planned)
    assert len(planned)==len(planned_set)
    assert planned_set==set(itertools.product(MODES,views,seeds,cohort))
    saved={}
    for row in read_csv(root/'Evaluation/decisions.csv'):
        if row['detector'] not in MODES:continue
        k=(row['detector'],row['feature_set'],int(row['seed']),row['request_id'])
        assert k not in saved and k in planned_set
        assert row['label']==cohort[k[3]]['label']
        saved[k]=row
    assert set(saved)==set(planned)
    records={};invalid=[];seen=set();checked_claims=0;verdicts=Counter()
    with (root/'Raw_Responses/native_responses.jsonl').open(encoding='utf-8') as f:
        for native in map(json.loads,f):
            if native['profile'] not in MODES:continue
            k=(native['profile'],native['view'],int(native['seed']),native['request_id'])
            assert k in saved and k not in seen;seen.add(k);prior=saved[k]
            assert native['status']==prior['status']
            if prior['status']!='valid':
                invalid.append({'profile':k[0],'view':k[1],'seed':k[2],'request_id':k[3],'label':int(prior['label']),'status':prior['status'],'error':prior['error']})
                continue
            payload=json.loads(native['raw_response']);answer=json.loads(payload['message']['content'])
            assert type(answer['is_anomaly']) is bool and answer['is_anomaly']==bool_value(prior['prediction'])
            assert float(answer['anomaly_score'])==float(prior['score']) and answer['reason']==prior['reason']
            archived=json.loads(prior['claims'])
            assert answer['evidence_claims']==[{f:c[f] for f in ('feature','reference','position')} for c in archived]
            checked=audit_claims(answer['evidence_claims'],features[k[1]][k[3]],refs[k[1]])
            assert checked==archived
            checked_claims+=len(checked);verdicts.update(c['verdict'] for c in checked)
            r={'profile':k[0],'view':k[1],'seed':k[2],'request_id':k[3],'label':int(prior['label']),
               'is_anomaly':answer['is_anomaly'],'score':float(answer['anomaly_score']),'reason':answer['reason']}
            assert math.isfinite(r['score']) and 0<=r['score']<=1
            records[k]=response_features(r,checked)
    assert seen==set(planned)
    groups=defaultdict(dict)
    for k in planned:groups[(k[0],k[1],k[3])][k[2]]=records.get(k)
    windows=[];pairs=[];excluded=[];complete_responses=[];window_index={}
    for (profile,view,rid),group in sorted(groups.items()):
        valid=[group[s] for s in seeds if group[s] is not None]
        if len(valid)!=len(seeds):
            excluded.append({'profile':profile,'view':view,'request_id':rid,'label':int(cohort[rid]['label']),
                'planned_seeds':json.dumps(seeds),'valid_seeds':json.dumps([r['seed'] for r in valid]),
                'planned_responses':len(seeds),'valid_responses':len(valid)})
            continue
        flipped=len({r['is_anomaly'] for r in valid})>1
        ps=[make_pair(a,b,flipped) for a,b in itertools.combinations(valid,2)]
        differing=[p for p in ps if p['boolean_disagreement']]
        row={'profile':profile,'view':view,'request_id':rid,'label':valid[0]['label'],'label_name':LABELS[valid[0]['label']],
            **{f:cohort[rid][f] for f in ('campaign','topic','window_start')},'seed_count':len(seeds),
            'positive_decisions':sum(r['is_anomaly'] for r in valid),'flipped':flipped,
            'claim_set_changed':len({r['claim_set'] for r in valid})>1,
            'feature_reference_set_changed':len({r['feature_reference_set'] for r in valid})>1,
            'same_claim_set_opposite_boolean_pair':any(p['claim_set_equal'] for p in differing),
            'same_nonempty_claim_set_opposite_boolean_pair':any(p['claim_set_equal'] and p['claim_sets_nonempty'] for p in differing),
            'same_supported_claim_set_opposite_boolean_pair':any(p['claim_set_equal'] and p['both_supported_only_nonempty'] for p in differing),
            'any_contradicted':any(r['contradicted'] for r in valid),'any_unverifiable':any(r['unverifiable'] for r in valid),
            'all_claims_supported_nonempty':all(r['supported_only'] for r in valid),
            'mixed_supported_only_and_contradicted':any(r['supported_only'] for r in valid) and any(r['contradicted'] for r in valid),
            'shared_reference_positions_changed':any(p['shared_reference_positions_changed'] for p in ps),
            'claim_correctness_status_changed':len({r['claim_correctness_status'] for r in valid})>1,
            'any_empty_claim_response':any(not r['claim_count'] for r in valid),
            'any_duplicate_claim_occurrences':any(r['duplicate_claim_occurrences'] for r in valid),
            'opposite_boolean_pairs':len(differing),'same_claim_set_opposite_boolean_pairs':sum(p['claim_set_equal'] for p in differing),
            'same_feature_reference_set_opposite_boolean_pairs':sum(p['feature_reference_set_equal'] for p in differing),
            'claim_count':sum(r['claim_count'] for r in valid),'unique_claim_occurrences':sum(r['unique_claim_count'] for r in valid),
            'duplicate_claim_occurrences':sum(r['duplicate_claim_occurrences'] for r in valid),
            'supported_claims':sum(r['supported'] for r in valid),'contradicted_claims':sum(r['contradicted'] for r in valid),
            'unverifiable_claims':sum(r['unverifiable'] for r in valid),'supported_only_responses':sum(r['supported_only'] for r in valid),
            'contradicted_responses':sum(bool(r['contradicted']) for r in valid),'unverifiable_responses':sum(bool(r['unverifiable']) for r in valid),
            'empty_claim_responses':sum(not r['claim_count'] for r in valid),'reason_distinct_count':len({r['reason'] for r in valid}),
            'score_min':min(r['score'] for r in valid),'score_max':max(r['score'] for r in valid),
            'score_span':max(r['score'] for r in valid)-min(r['score'] for r in valid)}
        windows.append(row);window_index[(profile,view,rid)]=row;pairs.extend(ps);complete_responses.extend(valid)
    response_rows=[]
    for k,r in sorted(records.items()):
        window=window_index.get((r['profile'],r['view'],r['request_id']))
        response_rows.append({f:r[f] for f in RESPONSE_COLUMNS if f in r}|{'label_name':LABELS[r['label']],
            'complete_group':window is not None,'window_flipped':window['flipped'] if window else None,
            'claims_json':json.dumps(r['claims'],sort_keys=True)})
    summary=[];response_summary=[];pair_summary=[]
    for profile,view,label,stratum in itertools.product(MODES,views,('all',-1,0,1),('all','flipped','stable')):
        ws=[w for w in windows if w['profile']==profile and w['view']==view and (label=='all' or w['label']==label)
            and (stratum=='all' or w['flipped']==(stratum=='flipped'))]
        ids={w['request_id'] for w in ws};n=len(ws)
        s={'profile':profile,'view':view,'label':label,'stratum':stratum,'windows':n,'responses':n*len(seeds)}
        for flag in WINDOW_FLAGS:
            count=sum(w[flag] for w in ws);s[flag+'_windows']=count;s[flag+'_fraction']=frac(count,n)
        for name in ('claim_count','supported_claims','contradicted_claims','unverifiable_claims','duplicate_claim_occurrences'):
            s[name]=sum(w[name] for w in ws)
        s['contradicted_claim_fraction']=frac(s['contradicted_claims'],s['claim_count'])
        s['score_span_median']=median(w['score_span'] for w in ws) if ws else None
        s['reason_distinct_count_median']=median(w['reason_distinct_count'] for w in ws) if ws else None
        summary.append(s)
        rs=[r for r in complete_responses if r['profile']==profile and r['view']==view and r['request_id'] in ids]
        for boolean in ('all',False,True):
            br=[r for r in rs if boolean=='all' or r['is_anomaly']==boolean]
            total=sum(r['claim_count'] for r in br);bad=sum(r['contradicted'] for r in br)
            response_summary.append({'profile':profile,'view':view,'label':label,'stratum':stratum,'is_anomaly':boolean,
                'responses':len(br),'claim_occurrences':total,'unique_claim_occurrences':sum(r['unique_claim_count'] for r in br),
                'duplicate_claim_occurrences':sum(r['duplicate_claim_occurrences'] for r in br),
                'supported_claims':sum(r['supported'] for r in br),'contradicted_claims':bad,
                'unverifiable_claims':sum(r['unverifiable'] for r in br),'contradicted_claim_fraction':frac(bad,total),
                'responses_with_contradiction':sum(bool(r['contradicted']) for r in br),
                'response_contradiction_fraction':frac(sum(bool(r['contradicted']) for r in br),len(br)),
                'supported_only_nonempty_responses':sum(r['supported_only'] for r in br),
                'empty_claim_responses':sum(not r['claim_count'] for r in br)})
        ps=[p for p in pairs if p['profile']==profile and p['view']==view and p['request_id'] in ids]
        for outcome in ('all','agree','disagree'):
            bs=[p for p in ps if outcome=='all' or p['boolean_disagreement']==(outcome=='disagree')]
            row={'profile':profile,'view':view,'label':label,'window_stratum':stratum,'pair_outcome':outcome,'pairs':len(bs)}
            for flag in ('claim_set_equal','claim_sets_nonempty','feature_reference_set_equal','shared_reference_positions_changed',
                         'claim_correctness_status_changed','any_contradicted','both_supported_only_nonempty'):
                count=sum(p[flag] for p in bs);row[flag+'_pairs']=count;row[flag+'_fraction']=frac(count,len(bs))
            pair_summary.append(row)
    matched=[];mode_summary=[]
    for view in views:
        off={w['request_id']:w for w in windows if w['profile']=='qwen_off' and w['view']==view}
        on={w['request_id']:w for w in windows if w['profile']=='qwen_on' and w['view']==view}
        shared=sorted(set(off)&set(on))
        for rid in shared:
            a,b=off[rid],on[rid];assert a['label']==b['label']
            row={'view':view,'request_id':rid,'label':a['label'],'label_name':a['label_name']}
            for flag in WINDOW_FLAGS:
                row['off_'+flag]=a[flag];row['on_'+flag]=b[flag]
            row.update(off_score_span=a['score_span'],on_score_span=b['score_span'])
            matched.append(row)
        for label in ('all',-1,0,1):
            selected=[rid for rid in shared if label=='all' or off[rid]['label']==label];n=len(selected)
            row={'view':view,'label':label,'common_complete_windows':n,
                'off_complete_windows':sum(label=='all' or w['label']==label for w in off.values()),
                'on_complete_windows':sum(label=='all' or w['label']==label for w in on.values())}
            for fa,fb in itertools.product((False,True),repeat=2):
                row[f'off_flip_{int(fa)}_on_flip_{int(fb)}']=sum(off[rid]['flipped']==fa and on[rid]['flipped']==fb for rid in selected)
            for flag in WINDOW_FLAGS:
                a=sum(off[rid][flag] for rid in selected);b=sum(on[rid][flag] for rid in selected)
                row['off_'+flag+'_windows']=a;row['on_'+flag+'_windows']=b
                row['off_'+flag+'_fraction']=frac(a,n);row['on_'+flag+'_fraction']=frac(b,n)
                row[flag+'_change_percentage_points']=100*(b-a)/n if n else None
            mode_summary.append(row)
    selected_cases=[]
    for profile,view in itertools.product(MODES,views):
        pool=[w for w in windows if w['profile']==profile and w['view']==view]
        rules={'flip_same_supported_claim_set':lambda w:w['same_supported_claim_set_opposite_boolean_pair'],
               'flip_with_contradiction':lambda w:w['flipped'] and w['any_contradicted'],
               'stable_with_changed_claim_set':lambda w:not w['flipped'] and w['claim_set_changed']}
        for name,predicate in rules.items():
            candidates=sorted((w for w in pool if predicate(w)),key=lambda w:w['request_id'])
            if not candidates:continue
            w=candidates[0]
            selected_cases.append({'selection_rule':name+'; lexicographically first eligible request ID; illustrative, not random',
                'window':w,'responses':[{k:r[k] for k in ('seed','is_anomaly','score','reason','claims')} for seed in seeds
                    for r in [records[(profile,view,seed,w['request_id'])]]]})
    files={'per_window.csv':(windows,WINDOW_COLUMNS),'per_response.csv':(response_rows,RESPONSE_COLUMNS),
        'pairwise.csv':(pairs,PAIR_COLUMNS),'excluded_groups.csv':(excluded,['profile','view','request_id','label','planned_seeds','valid_seeds','planned_responses','valid_responses']),
        'window_summary.csv':(summary,list(summary[0])),'response_summary.csv':(response_summary,list(response_summary[0])),
        'pairwise_summary.csv':(pair_summary,list(pair_summary[0])),
        'matched_mode_per_window.csv':(matched,list(matched[0])),'matched_mode_summary.csv':(mode_summary,list(mode_summary[0]))}
    for name,(rows,columns) in files.items():write_csv(out/name,rows,columns)
    write_json(out/'selected_cases.json',selected_cases);write_json(out/'invalid_responses.json',invalid)
    methods={'unit':'A complete group is one physical window in one Qwen mode and feature view, with a valid response at every planned seed.',
        'complete_groups_only':'Window, pairwise, matched-mode and response-summary analyses exclude incomplete groups. per_response.csv retains all valid responses and explicitly flags incomplete-group membership.',
        'claim_set':'Unordered unique (feature, reference, position) triples; verdict is excluded. Original claim order and duplicates do not affect set equality.',
        'occurrence_counts':'Every original claim occurrence contributes to claim correctness fractions; duplicate occurrences are separately counted and not silently removed.',
        'feature_reference_set':'Unordered unique (feature, reference) pairs, ignoring the asserted position.',
        'supported_only':'A response must contain at least one claim and every claim occurrence must be supported. all_claims_supported_nonempty requires this for every response in the window group.',
        'mixed_supported_only_and_contradicted':'At least one supported-only nonempty response and at least one response containing a contradicted claim within the same complete group.',
        'shared_reference_positions_changed':'At least one seed pair shares a feature/reference pair but has different sets of asserted positions for it.',
        'claim_correctness_status':'Presence set of supported, contradicted, unverifiable, serialized in that order; an empty response is no_claims. Status changes ignore occurrence counts.',
        'claim_correctness_category':'no_claims; contains_contradiction if any contradicted claim; otherwise unverifiable_without_contradiction if any unverifiable claim; otherwise supported_only.',
        'same_claim_set_opposite_boolean_pair':'At least one seed pair has opposite Booleans and equal unique structured-claim sets. The base indicator allows equal empty sets; nonempty and supported-only variants are separate.',
        'reason_strings':'Distinct exact final reason strings are a diagnostic of wording variation, not semantic difference or internal reasoning.',
        'matched_modes':'Compare thinking off/on only on their common complete physical-window IDs within each view and label; record the paired 2x2 flip table and on-minus-off percentage-point differences.',
        'no_causal_inference':'These are descriptive associations in saved outputs. Equal recorded claims do not establish equal full evidence use. Claim selection changes, contradictions, and Boolean disagreement do not identify an internal cause or the causal effect of thinking mode.',
        'dependence':'Views, labels, seeds and nested strata reuse observations. Do not sum view totals as unique physical windows or treat repeated responses/claims as independent trials.',
        'unknown_labels':'Unknown windows contribute output-stability and review-workload descriptions, not false-positive or detection-error counts.',
        'interval_check':'Use saved rounded feature values and min/max or p01/p99 reference bounds; equality is within. Any missing value/bound makes that claim unverifiable.',
        'schema':{name:columns for name,(_,columns) in files.items()},'seeds':seeds,'views':views,'profiles':{p:profiles[p] for p in MODES}}
    write_json(out/'methods.json',methods)
    after={p.relative_to(package).as_posix():{'sha256':sha(p),'bytes':p.stat().st_size} for p in paths};assert before==after
    assert len(pairs)==len(windows)*math.comb(len(seeds),2)
    assert len(complete_responses)==len(windows)*len(seeds)
    for w in windows:
        assert w['claim_count']==w['supported_claims']+w['contradicted_claims']+w['unverifiable_claims']
        assert w['claim_count']==w['unique_claim_occurrences']+w['duplicate_claim_occurrences']
    provenance={'passed':True,'inputs':before,'script_sha256':sha(Path(__file__)),'python_version':sys.version,
        'entrypoint':Path(__file__).resolve().relative_to(PACKAGE_ROOT).as_posix(),
        'input_root':'.' if package==PACKAGE_ROOT else 'external input root supplied with --inputs',
        'input_path_basis':'Input hash keys are relative to the selected input root.',
        'path_policy_sha256':sha(Path(__file__).with_name('analysis_paths.py')),
        'planned_response_records':len(planned),'native_qwen_records_reconciled':len(seen),
        'valid_native_qwen_records_reconciled':len(records),'invalid_responses':len(invalid),
        'structured_claim_occurrences_independently_checked':checked_claims,'all_valid_response_claim_verdict_counts':dict(verdicts),
        'planned_mode_view_window_groups':len(groups),'complete_mode_view_window_groups':len(windows),'excluded_mode_view_window_groups':len(excluded),
        'valid_responses_in_complete_groups':len(complete_responses),'valid_responses_excluded_from_summaries':len(records)-len(complete_responses),
        'seed_pairs_in_complete_groups':len(pairs),'cohort_windows':len(cohort),'cohort_label_counts':dict(Counter(int(r['label']) for r in cohort_rows)),
        'frozen_inputs_unchanged':True,'new_inference':False,'new_model_fit':False,'paper_changed':False,
        'output_sha256':{p.name:sha(p) for p in out.iterdir() if p.is_file()}}
    write_json(out/'provenance.json',provenance)
    print(json.dumps({k:v for k,v in provenance.items() if k not in ('inputs','output_sha256','command','python_version')},indent=2))
    for s in summary:
        if s['profile']=='qwen_on' and s['label']=='all' and s['stratum']!='all':
            print(s['view'],s['stratum'],'N=',s['windows'],'contradicted=',s['any_contradicted_windows'],
                'claim_set_changed=',s['claim_set_changed_windows'],'same_supported_claim_set_opposite=',s['same_supported_claim_set_opposite_boolean_pair_windows'])


if __name__=='__main__':main()
