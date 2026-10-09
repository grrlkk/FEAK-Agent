"""Local v4 preparation report, with pending GPU results explicitly absent."""
from datetime import datetime,timezone
from pathlib import Path
import math
import shutil
import time

from .common import REPO,ROOT,read_json,write_json,file_sha,collection_lock


def optional(path):
    return read_json(path) if path.exists() else None


def budget_record(path,cap):
    value=optional(path)
    if value is None:
        return None
    for key in ('confirmed_usd','reserved_usd'):
        if not math.isfinite(value[key]) or value[key]<0:
            raise ValueError('Invalid budget record: '+str(path))
    if value['confirmed_usd']+value['reserved_usd']>cap+1e-8:
        raise ValueError('Preparation cap exceeded: '+str(path))
    return value


def render():
    design=read_json(ROOT/'design.json')
    train={design['source_metadata'][s]['question_hash'] for s in design['train300']}
    test={design['source_metadata'][s]['question_hash'] for s in design['test100']}
    if train & test or test & set(design['training_question_hashes']):
        raise ValueError('Question-disjoint evaluation contract changed')
    root_a=REPO/'verak/v3/outputs/data_boost/global'
    a_ready=optional(root_a/'cpu_ready.json')
    a_final=optional(root_a/'complete.json')
    b=optional(ROOT/'B/complete.json')
    c=optional(ROOT/'C/final_complete.json')
    d=optional(ROOT/'D/complete.json')
    # C's extended audit may carry a second marker; only its final report's
    # explicit coverage can establish completion of the fixed 60-map review.
    a_path=next((p for p in (root_a/'v4/component_report.md',root_a/'v4/component_pre_gpu.md',
                              root_a/'v4/preparation_report.md') if p.exists()),None)
    components={name:optional(ROOT/name/'metrics.json') for name in ('B','D')}
    components['C']=optional(ROOT/'C/component_metrics.json')
    accounts={name:budget_record(ROOT/name/'api/accounting.json',cap)
              for name,cap in (('B',5),('C',4),('D',15))}
    accounts['A']=budget_record(root_a/'api/accounting.json',40)
    now=datetime.now(timezone.utc).isoformat()
    lines=['# FEAK-Agent v4 preparation','',f'Updated {now}.','',
        'The reference is `imple/FEAK_AGENT_METHOD.md`. No v4 training or map scale-up was started. '
        'Existing RFT1 rollout/environment/rewards and its two GPUs remain assigned to RFT1. '
        'A can join RFT1 only after the previously scheduled GPU reference pass; '
        'this report never substitutes provisional CPU role rewards for final selections.','',
        '## Approved decisions and frozen data','',
        '- Corruption teacher selection uses role R >= 0.80, terminal STOP, and <=1 rejected action '
        '(with the existing no-GLOBAL exception), not recovery >=0.80.',
        f'- Evaluation uses 100 unique test sources from the 62 eligible unseen questions '
        f'({len(test)} questions represented in this sample). Overlap with training questions: 0.',
        '- Feedback: 300 unique train sources (100 per genre) and 100 test sources (34/33/33). '
        'Maps: a fixed subset of 150 train sources (50 per genre). D uses only B training sources.',
        '- Deduplication normalizes whitespace and excludes test sources matching any train/valid source. '
        'Question/source IDs and input hashes were frozen before teacher calls.',
        '- The JSONL has one stored rubric-feedback field and two human-grader score arrays; '
        'it does not separately identify two feedback authors. No scorer rewards are computed for B/C/D.',
        '- The adjacent-boundary G_DEL_LINK rule exists only in v4 evaluation. Historical rewards/training are unchanged.',
        '- A is capped at four practices per source/operator, including archived concentrated data; '
        'active train sources require verified new positions. Missing CPU comparisons remain unknown; '
        'all final role rewards and selections require the GPU reference.','',
        '## Cost and completion','',
        '|Component|Confirmed USD|Reserved USD|Cap USD|Status|','|---|---:|---:|---:|---|']
    statuses={'A':('final GPU report complete' if a_final else 'teachers frozen; GPU reference pending' if a_ready else 'teacher collection in progress'),
        'B':'complete' if b else 'in progress','C':'pilot/report review pending','D':'complete' if d else 'in progress'}
    if c:
        statuses['C']='fixed pilot and audit complete; map scale-up awaiting user review'
    for name,cap in (('A',40),('B',5),('C',4),('D',15)):
        account=accounts[name]
        amounts=[f'{account[k]:.6f}' for k in ('confirmed_usd','reserved_usd')] if account else ['pending','pending']
        lines.append(f'|{name}|{amounts[0]}|{amounts[1]}|{cap}|{statuses[name]}|')
    lines+=['','A includes spending before this continuation; the $40 cap is cumulative. '
        'Costs are usage-ledger estimates under the frozen experiment rates, not a billing invoice.','',
        '## A. Extra GLOBAL teachers','',a_path.read_text() if a_path else 'Collection artifacts are pending.','',
        '## B. Teacher-feedback items','']
    bp=ROOT/'B/report.md'
    lines.append(bp.read_text() if bp.exists() else 'Item collection is in progress.')
    if components['B']:
        counts=components['B']['counts']
        lines+=['', '|Split|Items|Revision|Korean|Writer|Search needed|Essay-wide/unspecified location|',
            '|---|---:|---:|---:|---:|---:|---:|']
        for split in ('train','test'):
            g=counts[split]
            lines.append(f'|{split}|{g["items"]}|{g["owner_counts"].get("revision",0)}|{g["owner_counts"].get("korean",0)}|'
                f'{g["owner_counts"].get("writer",0)}|{g["needs_search"].get("yes",0)}|{g["unlocated_or_essay_wide"]}|')
        lines+=['','An essay-wide location is not a claim of sentence-level localization accuracy. '
            'Item labels are Sol-derived; semantic correctness has not been established by a human audit.']
    lines+=['','## C. Genre-neutral maps','']
    cp=ROOT/'C/component_report.md'
    lines.append(cp.read_text() if cp.exists() else 'Two extractions per fixed essay and the Sol audit are in progress.')
    lines+=['','Examples: [V4_MAP_EXAMPLES.md](V4_MAP_EXAMPLES.md). '
        'The fixed 150-essay pilot is the extraction limit until the user reviews those examples.','',
        '## D. Revision content/expression','']
    dp=ROOT/'D/report.md'
    lines.append(dp.read_text() if dp.exists() else 'The bounded 100-essay, two-attempt pilot starts after B.')
    manual=ROOT/'D/manual_review_50.md'
    if manual.exists():
        target=REPO/'imple/reports/V4_CONTENT_REVIEW_50.md'
        shutil.copyfile(manual,target)
        lines+=['','Manual review: [V4_CONTENT_REVIEW_50.md](V4_CONTENT_REVIEW_50.md). '
            'The filename is fixed; actual available case count is disclosed above until collection completes.']
    lines+=['','## Validation and reproducibility','',
        '- The archived 92 pilot-2 Luna trajectories reproduced their historical v1 outcomes: '
        '91 full reward matches plus the preserved historical failure/completed GLOBAL result. '
        'Replay made no new API, scorer, Bareun, or GPU calls.',
        '- All 466 new structural practice source/position proofs passed the offline ingestion check. '
        'The new-position guard and source cap apply at future data selection; v1 runtime files are unchanged.',
        '- Tests cover atomic edits/undo, role restrictions, private-feedback exclusion, '
        'actual pinned-tokenizer action-only masks, item-judgment coverage, and the existing GPU allocation boundary.',
        f'- Frozen cohort manifest: `{ROOT/"design.json"}` (SHA256 `{file_sha(ROOT/"design.json")}`).',
        f'- Raw requests/results, failures and ledgers: `{ROOT}`. No raw essays, reports, model files or credentials were uploaded.',
        '- No new training is started by this preparation pipeline. The previously authorized RFT1 pipeline remains independent.','']
    path=REPO/'imple/reports/V4_PREP.md'
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.md.tmp')
    temporary.write_text('\n'.join(lines))
    temporary.replace(path)
    result={'at':now,'report':str(path),'report_sha256':file_sha(path),'statuses':statuses,
            'final':bool(a_final and b and c and d),
            'costs':accounts,'gpu_used':False,'paid_calls':0,'training':False}
    write_json(ROOT/'report_status.json',result)
    return result


def watch():
    # This writer cannot dispatch an API, GPU, or training call. It only updates
    # the requested report as the independently owned components finish.
    while True:
        with collection_lock(ROOT/'report'):
            result=render()
        if result['final']:
            write_json(ROOT/'report_complete.json',result)
            return result
        time.sleep(30)


def launch():
    import os
    import subprocess
    import sys
    directory=ROOT/'report_worker'
    with collection_lock(directory/'launcher'):
        marker=directory/'worker.json'
        if marker.exists():
            prior=read_json(marker)
            try:
                cmdline=Path(f'/proc/{prior["pid"]}/cmdline').read_bytes()
            except FileNotFoundError:
                cmdline=b''
            if b'verak.v4.report' in cmdline and b'--watch' in cmdline:
                return prior
        directory.mkdir(parents=True,exist_ok=True)
        with (directory/'worker.log').open('ab',buffering=0) as log:
            process=subprocess.Popen([sys.executable,'-m','verak.v4.report','--watch'],
                cwd=Path(__file__).resolve().parents[2],stdin=subprocess.DEVNULL,stdout=log,stderr=log,
                env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1',
                     'OPENBLAS_NUM_THREADS':'1'},start_new_session=True)
        result={'pid':process.pid,'at':time.time(),'scope':'local report updates only','gpu_used':False,
                'paid_calls':0,'training':False}
        write_json(marker,result)
        return result


if __name__=='__main__':
    import argparse
    from .common import constrain_cpu
    constrain_cpu()
    parser=argparse.ArgumentParser()
    group=parser.add_mutually_exclusive_group()
    group.add_argument('--watch',action='store_true')
    group.add_argument('--launch',action='store_true')
    args=parser.parse_args()
    if args.launch:
        print(launch())
    elif args.watch:
        watch()
    else:
        with collection_lock(ROOT/'report'):
            print(render()['report'])
