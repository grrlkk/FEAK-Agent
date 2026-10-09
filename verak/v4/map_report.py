"""Readable C pilot reports from immutable extraction/judgment artifacts only."""
from collections import Counter, defaultdict
import json

from .common import GENRES, GENRE_KO, REPO, atomic_new, file_sha, read_json, safe_id, write_json
from .maps import SENTENCE_TYPES, PARAGRAPH_TYPES

RELATION_KO = {'main':'중심 내용→문항', 'support':'근거·설명→대상', 'example':'사례·장면→대상',
    'contrast':'대조→기준', 'sequence':'앞→뒤', 'cause_effect':'원인→결과', 'summary':'요약→대상'}


def cell(value):
    return str(value).replace('|','&#124;').replace('\n','<br>')


def percent(numerator, denominator):
    return f'{numerator/denominator:.2%}' if denominator else 'NA'


def table(headers, rows):
    return '\n'.join(['| '+' | '.join(map(cell,headers))+' |',
        '| '+' | '.join('---' for _ in headers)+' |', *['| '+' | '.join(map(cell,row))+' |' for row in rows]])


def aggregate(rows, output, frozen, accounting):
    attempts, joined, judgments = {}, {}, {}
    for row in rows:
        sid, name = row['source_id'], safe_id(row['source_id'])+'.json'
        attempts[sid] = [read_json(output/f'attempt_{i}'/name) for i in (1,2)]
        joined[sid] = read_json(output/'consensus'/name)
    for sid in frozen['sol_maps60']:
        judgments[sid] = read_json(output/'sol'/(safe_id(sid)+'.json'))
    groups = {'all': rows, **{g:[r for r in rows if r['genre']==g] for g in GENRES}}
    totals = {}
    for genre, members in groups.items():
        usable = [joined[r['source_id']] for r in members if joined[r['source_id']]['status']=='valid']
        relations = {}
        for field, labels in [('sentence_relations',SENTENCE_TYPES),('paragraph_relations',PARAGRAPH_TYPES)]:
            for label in labels:
                key = field+':'+label
                states = [j['diagnostics']['relations'][key] for j in usable]
                sums = {k:sum(s[k] for s in states) for k in ('left','right','intersection','union')}
                observed = [s['jaccard'] for s in states if s['jaccard'] is not None]
                relations[key] = {**sums, 'micro_jaccard': sums['intersection']/sums['union'] if sums['union'] else None,
                    'macro_jaccard_nonempty':sum(observed)/len(observed) if observed else None,
                    'essays_with_nonempty_union':len(observed), 'valid_pairs':len(usable)}
        para = [p for j in usable for p in j['diagnostics']['paragraph_roles']]
        flags = [j['diagnostics']['off_topic'] for j in usable]
        totals[genre] = {'planned_essays':len(members), 'valid_pairs':len(usable),
            'attempt_statuses':dict(Counter(a['status'] for r in members for a in attempts[r['source_id']])),
            'relations':relations, 'paragraph_roles':{'total':len(para),
                'agree':sum(p['role_agrees'] for p in para),
                'both_unknown':sum(p['left']['role'] is None and p['right']['role'] is None for p in para),
                'key_agree':sum(p['key_agrees'] for p in para)},
            'off_topic':{'explicit_intersection':sum(len(f['explicit_intersection']) for f in flags),
                'protected_override_sentences':sum(len(f['protected_override']) for f in flags),
                'protected_override_essays':sum(bool(f['protected_override']) for f in flags),
                'final_sentences':sum(len(f['final']) for f in flags)}}
    sol = {}
    for genre in groups:
        selected = [j for j in judgments.values() if genre=='all' or j['genre']==genre]
        bins = defaultdict(Counter)
        for item in selected:
            checks = {c['id']:c for c in item['checks']}
            if item['status'] in {'valid','empty_map'}:
                for verdict in item['value']['judgments']:
                    check = checks[verdict['id']]
                    bins[check['kind']+':'+check['type']][verdict['verdict']] += 1
            else:
                for check in checks.values():
                    bins[check['kind']+':'+check['type']]['unmeasured'] += 1
        sol[genre] = {'planned_maps':len(selected), 'map_statuses':dict(Counter(j['status'] for j in selected)),
                     'by_check_type':{k:{v:c[v] for v in ('plausible','wrong','unknown','unmeasured')} for k,c in sorted(bins.items())}}
    return {'schema_version':1, 'component':'C', 'source_manifest_sha256':frozen['source_manifest_sha256'],
        'design_sha256':file_sha(output/'design.json'), 'extraction':totals, 'sol':sol,
        'luna_model':frozen.get('model'), 'sol_model':frozen.get('sol_model'),
        'sol_fixed_ids':frozen['sol_maps60'], 'example_ids':frozen['examples6'],
        'accounting':accounting, 'cap_usd':4., 'gpu_used':False, 'training':False,
        'feedback_or_scores_transmitted':False,
        'limitations':['Agreement is extractor consistency, not correctness.',
            'Sol judgments are model judgments, not human validation.',
            'Invalid extraction pairs are excluded from edge agreement and retain a separate fixed-source denominator.',
            'The 60 Sol map IDs and 6 shortest examples were frozen before extraction; unavailable IDs are never replaced.']}, joined, judgments


def relation_rows(metrics):
    rows=[]
    for genre in ('all',*GENRES):
        for key, stats in metrics['extraction'][genre]['relations'].items():
            rows.append([genre,key,stats['valid_pairs'],stats['left'],stats['right'],stats['intersection'],stats['union'],
                         percent(stats['intersection'],stats['union'])])
    return rows


def sol_rows(metrics):
    rows=[]
    for genre in ('all',*GENRES):
        values=metrics['sol'][genre]['by_check_type']
        keys=sorted(set(values)|{'sentence_relation:'+k for k in SENTENCE_TYPES}|
                    {'paragraph_relation:'+k for k in PARAGRAPH_TYPES})
        for key in keys:
            counts=values.get(key,{})
            rows.append([genre,key,*[counts.get(k,0) for k in ('plausible','wrong','unknown','unmeasured')]])
    return rows


def render_component(metrics):
    totals=metrics['extraction']['all']; costs=metrics['accounting']; lines=[
        '## C. 장르 중립 지도 파일럿', '',
        f"고정 train 150편(장르별 50편)에 Luna low 독립 추출 두 번을 시행했다. 유효한 추출 쌍은 {totals['valid_pairs']}/150편이다. "
        '문장·문단 관계는 방향과 유형이 모두 같은 교집합만 유지한다. 지지/예시의 문장당 외향 관계는 합쳐 최대 하나다.', '',
        table(['장르','계획 편수','유효한 두 추출','추출 상태','Sol map 상태'], [
            [g,metrics['extraction'][g]['planned_essays'],metrics['extraction'][g]['valid_pairs'],
             json.dumps(metrics['extraction'][g]['attempt_statuses'],ensure_ascii=False),
             json.dumps(metrics['sol'][g]['map_statuses'],ensure_ascii=False)] for g in GENRES]), '',
        '아래 일치율은 유효한 쌍의 관계 수를 합친 intersection/union이다. 두 추출 모두 해당 유형을 쓰지 않은 경우는 NA이며 정답률이 아니다.', '',
        table(['장르','관계','유효 쌍','추출1','추출2','교집합','합집합','일치율'],relation_rows(metrics)), '',
        'Sol 대상 60편(장르별 20편, seed 131)은 추출 전에 고정했다. 유지된 모든 문장·문단 관계를 점검하고 문단 역할과 off_topic도 별도 집계했다. '
        'invalid/API 실패는 wrong 또는 unknown에 합산하지 않는다. 유효한 두 추출이 없는 사전 대상은 교체하지 않았다.', '',
        table(['장르','검사 유형','plausible','wrong','unknown','미측정'],sol_rows(metrics)), '',
        table(['장르','문단 역할 일치/전체','양쪽 unknown','핵심 문장 일치/전체','명시적 off_topic 교집합','보호 override 문장/편','최종 off_topic'], [
            [g,f"{m['paragraph_roles']['agree']}/{m['paragraph_roles']['total']}",m['paragraph_roles']['both_unknown'],
             f"{m['paragraph_roles']['key_agree']}/{m['paragraph_roles']['total']}",m['off_topic']['explicit_intersection'],
             f"{m['off_topic']['protected_override_sentences']}/{m['off_topic']['protected_override_essays']}",m['off_topic']['final_sentences']]
            for g,m in metrics['extraction'].items()]), '',
        '관련성 보호는 어느 추출에서든 main→Q인 문장을 시작점으로 삼고, 어느 추출에든 있는 support/example/contrast를 역방향으로 최대 2홉 내려간다. '
        '보호는 off_topic 제외에만 쓰며 관계의 정답이나 관련성의 증명이 아니다. Q 경로 부재로 off_topic을 만들지 않는다.', '',
        f"비용: 확정 ${costs['confirmed_usd']:.6f}, 보류 예약 ${costs['reserved_usd']:.6f}, 합계 ${costs['confirmed_usd']+costs['reserved_usd']:.6f} / $4. "
        f"유료 요청 {costs['calls']}건, 미해결 pending {costs['pending']}건. Luna 모델은 `{metrics['luna_model']}`, Sol은 `{metrics['sol_model']}`이다. 모델 샘플링 seed는 지원되지 않아 각 추출은 독립 요청이다.", '',
        '입력은 문항·장르·문장/문단 ID와 텍스트만 포함한다. feedback/점수 입력, GPU 사용, 학습은 없다. '
        '6개 예시는 장르별 len(text), source_id 오름차순의 최단 두 편으로 고정했으며 좋은 결과를 골라 교체하지 않았다.', '',
        'v4 G_DEL_LINK 평가는 삭제로 원문 문단이 비었을 때 바로 앞 원문 문단의 끝 또는 바로 뒤 원문 문단의 시작에 삽입한 문장을 후보로 인정한다. '
        '보통 경우는 원래 문단의 ±1 위치 조건을 유지한다. 이 함수는 v4에서만 호출할 수 있으며 v1/v2 보상과 과거 결과는 바꾸지 않는다.', '',
        '모델 간 일치와 Sol 타당성 판정은 사람 평가 또는 수정 성능의 증거로 해석하지 않는다. 150편 이후 확장 수집은 하지 않았다.', '']
    return '\n'.join(lines)


def _relations(edges, texts):
    if not edges:
        return '없음.'
    return table(['관계','출발','도착'], [[e['type']+' ('+RELATION_KO[e['type']]+')',
        e['source']+': '+texts.get(e['source'],e['source']), e['target']+': '+texts.get(e['target'],e['target'])] for e in edges])


def _mermaid(row, value):
    # Use IDs only in Mermaid labels: all exact text is printed above, never interpreted as Mermaid code.
    lines=['```mermaid','flowchart TD','  Q["Q: 문항"]']
    for paragraph in row['paragraphs']:
        lines.append('  subgraph '+paragraph['id']+'["'+paragraph['id']+'"]')
        for sentence in paragraph['sentences']:
            lines.append('    '+sentence['id']+'["'+sentence['id']+'"]')
        lines.append('  end')
    for edge in value['sentence_relations']:
        lines.append('  '+edge['source']+' -->|'+edge['type']+'| '+edge['target'])
    for sid in value['off_topic']:
        lines.append('  style '+sid+' fill:#ffe0e0,stroke:#c00')
    lines.extend(['```','','```mermaid','flowchart LR'])
    for p in value['paragraph_roles']:
        lines.append('  '+p['paragraph']+'["'+p['paragraph']+': '+(p['role'] or 'unknown')+'"]')
    for edge in value['paragraph_relations']:
        lines.append('  '+edge['source']+' -->|'+edge['type']+'| '+edge['target'])
    lines.append('```')
    return '\n'.join(lines)


def example(row, joined, judged=None):
    text={s['id']:s['text'] for p in row['paragraphs'] for s in p['sentences']}
    text['Q']=row['question']
    paragraphs={p['id']:' '.join(s['text'] for s in p['sentences']) for p in row['paragraphs']}
    lines=[f"## {row['source_id']} — {GENRE_KO[row['genre']]} ({row['genre']})",'',
        f"문항: {row['question']}", '', f"원문 길이: {len(row['text'])}자. 추출 상태: {', '.join(joined['attempt_statuses'])}.", '',
        table(['문단','문장','원문'],[[p['id'],s['id'],s['text']] for p in row['paragraphs'] for s in p['sentences']]), '']
    if joined['status']!='valid':
        lines += ['두 유효 추출이 없어 교집합 지도와 관계 평가는 미측정이다. 사전 예시를 대체하지 않았다.', '']
        return '\n'.join(lines)
    value,diag=joined['map'],joined['diagnostics']
    lines += ['문단 역할과 핵심 문장(불일치는 unknown):','',table(['문단','역할','핵심 문장','추출1 역할/핵심','추출2 역할/핵심'], [
        [r['paragraph'],r['role'] or 'unknown',r['key_sentence'] or 'unknown',
         f"{d['left']['role']}/{d['left']['key_sentence']}",f"{d['right']['role']}/{d['right']['key_sentence']}"]
        for r,d in zip(value['paragraph_roles'],diag['paragraph_roles'])]), '',
        '유지된 문장 관계:','',_relations(value['sentence_relations'],text),'',
        '유지된 문단 관계:','',_relations(value['paragraph_relations'],paragraphs),'',
        '명시적 off_topic: '+(', '.join(value['off_topic']) or '없음')+'.',
        '두 추출의 off_topic 교집합: '+(', '.join(diag['off_topic']['explicit_intersection']) or '없음')+'.',
        '2홉 보호로 제외된 flag: '+(', '.join(diag['off_topic']['protected_override']) or '없음')+'.','']
    for field,label,lookup in [('sentence_relations','문장',text),('paragraph_relations','문단',paragraphs)]:
        for side,title in [('left_only','추출1에만 있음'),('right_only','추출2에만 있음')]:
            lines += [f'교집합에서 탈락한 {label} 관계 — {title}:','',_relations(diag['dropped'][field][side],lookup),'']
    lines += ['최종 지도(문장 텍스트는 위 ID 표와 대응):','',_mermaid(row,value),'']
    if judged is not None:
        lines += ['사전 Sol 대상 여부: 포함. 검사 상태: '+judged['status']+'.','']
        if judged['status']=='valid':
            checks={c['id']:c for c in judged['checks']}
            lines += [table(['check','종류/관계','출발→도착','판정','이유'],[
                [v['id'],checks[v['id']]['kind']+':'+checks[v['id']]['type'],
                 checks[v['id']].get('source','')+'→'+checks[v['id']].get('target',''),v['verdict'],v['reason']]
                for v in judged['value']['judgments']]),'']
    else:
        lines += ['사전 Sol 대상에 포함되지 않은 예시다. 예시를 위해 추가 유료 판정을 호출하지 않았다.','']
    return '\n'.join(lines)


def publish(rows, output, frozen, accounting):
    if (output/'complete.json').exists():
        completed=read_json(output/'complete.json')
        for label in ('metrics','report','examples'):
            if file_sha(completed[label+'_path'])!=completed[label+'_sha256']:
                raise ValueError('Completed map report artifact changed')
        return completed
    metrics,joined,judgments=aggregate(rows,output,frozen,accounting)
    metrics_path,report_path=output/'component_metrics.json',output/'component_report.md'
    write_json(metrics_path,metrics)
    report_path.write_text(render_component(metrics),encoding='utf-8')
    sources={r['source_id']:r for r in rows}
    examples=['# V4 장르 중립 지도: 고정된 6개 예시','',
        '고정 150편에서 장르별 원문 길이(len(text))와 source_id 순으로 최단 두 편을 추출 전에 정했다. '
        '이 6편은 전체 성능을 대표하도록 무작위 선정한 표본이 아니다. 원문 ID와 문장을 그대로 제시한다.', '',
        f"source manifest SHA-256: `{frozen['source_manifest_sha256']}`; C design SHA-256: `{metrics['design_sha256']}`.", '']
    examples += [example(sources[sid],joined[sid],judgments.get(sid)) for sid in frozen['examples6']]
    example_path=REPO/'imple/reports/V4_MAP_EXAMPLES.md'
    example_path.parent.mkdir(parents=True,exist_ok=True)
    example_path.write_text('\n'.join(examples),encoding='utf-8')
    finished={'status':'budget_stop' if metrics['extraction']['all']['attempt_statuses'].get('budget_not_dispatched') or
        metrics['sol']['all']['map_statuses'].get('budget_not_dispatched') else 'complete',
        'component':'C','metrics_path':str(metrics_path),'metrics_sha256':file_sha(metrics_path),
        'report_path':str(report_path),'report_sha256':file_sha(report_path),
        'examples_path':str(example_path),'examples_sha256':file_sha(example_path),
        'accounting':accounting,'no_live_paid_calls':accounting['pending']==0,'gpu_used':False,'training':False,'stopped':True}
    if (output/'complete.json').exists():
        if read_json(output/'complete.json')!=finished:
            raise ValueError('Completed map report changed')
    else:
        atomic_new(output/'complete.json',finished)
    write_json(output/'status.json',finished)
    return finished
