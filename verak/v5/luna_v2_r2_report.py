"""Complete human annotation tables and both preparation arms, never eval50."""
from collections import Counter
import csv
import json
from pathlib import Path
import sqlite3

from verak.v4.common import REPO,read_json,write_json,file_sha
from .luna_v2_data import ROOT
from .luna_v2_env import PHI

REPORTS=REPO/'reports'
PHASES={'검사 끔':'prep_r2_discovery','검사 켬':'prep_r2_checked'}


def table_escape(value):return str(value).replace('|','\\|').replace('\n','<br>')


def run():
    cfg=read_json(ROOT/'prep_r2_config.json')
    complete=read_json(ROOT/'prep_r2_complete.json') if (ROOT/'prep_r2_complete.json').exists() else None
    if complete:
        measured={}
        for phase in PHASES.values():
            files=list((ROOT/phase/'B').glob('*/result.json'))
            beginnings=[p.stat().st_mtime-read_json(p)['wall_seconds'] for p in files]
            endings=[p.stat().st_mtime for p in files]
            if phase=='prep_r2_discovery':endings += [p.stat().st_mtime for p in (ROOT/'calibration_r2').glob('*.json')]
            measured[phase]={'start':min(beginnings),'end':max(endings),'seconds':max(endings)-min(beginnings)}
        complete={**complete,'wall_seconds_including_resume':measured['prep_r2_checked']['end']-measured['prep_r2_discovery']['start'],
                  'parts_seconds':{'checks_off_and_noise_calibration':measured['prep_r2_discovery']['seconds'],
                                   'checks_on':measured['prep_r2_checked']['seconds']},
                  'timing_basis':'Saved completion file timestamps minus measured per-episode monotonic durations; includes completed work reused on resume.'}
        write_json(ROOT/'prep_r2_wall_clock.json',complete)
    selection=read_json(ROOT/'selection.json')
    result_rows={};notices=[];summaries={};fingerprints=set();all_events={}
    for label,phase in PHASES.items():
        records=[];reasons=Counter();actions=Counter();phi=Counter();parse=Counter();structure_rejections=[]
        for meta in selection['prep10']:
            folder=ROOT/phase/'B'/meta['source_id'].replace(':','_')
            if not (folder/'result.json').exists():continue
            record=read_json(folder/'result.json');records.append(record)
            fingerprints.update(c['fingerprint'] for c in record['calls'])
            events=[json.loads(line) for line in (folder/'events.jsonl').read_text().splitlines()]
            all_events[label,record['essay_id']]=events
            pending=[]
            for number,event in enumerate(events,1):
                if event['kind']=='plan_parse':parse['성공' if event['passed'] else '실패']+=1
                if event['kind']=='structure_protocol':structure_rejections.append({'essay_id':record['essay_id'],'reason':event['error']})
                if event['kind']=='attempt':reasons.update(k for k,_ in event['reasons'])
                if event['kind']=='notices':
                    pending=[]
                    for n in event['notices']:
                        phi[n['phi']]+=1
                        row={'번호':len(notices)+1,'설정':label,'글':record['essay_id'],'이벤트':number,'문장':n['sid'],
                             '유형':n['phi'],'변화':n.get('event',''),'원인 문장':', '.join(n.get('source_sids',[])),
                             '메시지':n['message'],'한국어 결정':'','결정 이유':'','수리 전':'','수리 후':'',
                             '요청 행동':'','요청 출력':'','판정(유효/잡음/형식 오류)':''}
                        notices.append(row);pending.append(row)
                if event['kind']=='korean_decision':
                    actions[event['action']]+=1
                    for row in pending:
                        if row['문장']==event['sid']:
                            row.update({'한국어 결정':event['action'],'결정 이유':event['content'],'수리 전':event['before'],
                                        '수리 후':event['after'],'요청 행동':event.get('requested_action',event['action']),
                                        '요청 출력':event.get('requested_content',event['content'])})
        result_rows[label]=records
        items=[item for r in records for item in r['items']]
        summaries[label]={'essays':len(records),'notices':dict(phi),'actions':dict(actions),'reasons':dict(reasons),'plan_parse':dict(parse),
                          'kept_paragraphs':sum(i['kept_original'] for i in items),'paragraphs':len(items),
                          'retries':sum(i['attempts']-1 for i in items),'unchanged_essays':sum(r['original']==r['final'] for r in records),
                          'mean_character_edit_ratio':sum(r['modification']['ratio'] for r in records)/len(records) if records else None,
                          'structure_protocol_rejections':structure_rejections,'essay_wall_seconds_sum':sum(r['wall_seconds'] for r in records)}
    fields=['번호','설정','글','이벤트','문장','유형','변화','원인 문장','메시지','한국어 결정','결정 이유','수리 전','수리 후','요청 행동','요청 출력','판정(유효/잡음/형식 오류)']
    with (REPORTS/'luna_upper_bound_prep_r2_notices.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(notices)
    write_json(ROOT/'prep_r2_notice_table.json',notices)
    columns=['번호','설정','글','문장','유형','메시지','한국어 결정','판정(유효/잡음/형식 오류)']
    notice_md=['# 준비 2회차 — 전체 알림 판정표','','판정 칸은 실제 사람의 검토를 위해 모두 비워 두었다. CSV에는 원인 문장, 수리 전후, 원래 요청 행동과 환경이 정규화한 행동도 들어 있다.','','| '+' | '.join(columns)+' |','|'+'|'.join(['---']*len(columns))+'|']
    for row in notices:notice_md.append('| '+' | '.join(table_escape(row[k]) for k in columns)+' |')
    (REPORTS/'luna_upper_bound_prep_r2_notices.md').write_text('\n'.join(notice_md)+'\n')
    essays=['# 준비 2회차 — 원문과 검사 끔/켬 전체 결과','']
    for meta in selection['prep10']:
        source=read_json(ROOT/'sources'/(meta['source_id'].replace(':','_')+'.json'))
        essays+=['## '+meta['source_id'],'',source['question'],'','### 원문','',source['text']]
        for label in PHASES:
            record=next((r for r in result_rows[label] if r['essay_id']==meta['source_id']),None)
            if record:
                essays+=['','### '+label,'',f"주장 추출: {record['plan']['thesis']}",'',f"글자 편집 비율: {record['modification']['ratio']:.1%}",'',record['final']]
    (REPORTS/'luna_upper_bound_prep_r2_essays.md').write_text('\n'.join(essays)+'\n')
    # Preserve a full machine log and a readable event log for every essay/arm.
    for (label,sid),events in all_events.items():
        lines=[f'# {sid} / {label} — 전체 로그','']
        for i,e in enumerate(events,1):
            lines+=['',f"## {i}. {e['kind']} ({e['elapsed_s']:.2f}초)",'']
            if e['kind']=='call':lines += ['역할: '+e['role'],'','관찰:','',e['observation'],'','행동:','',e['action']]
            else:lines+=['```json',json.dumps(e,ensure_ascii=False,indent=2),'```']
        target=REPORTS/'luna_upper_bound_prep_r2_logs'/PHASES[label]/(sid.replace(':','_')+'.md')
        target.parent.mkdir(parents=True,exist_ok=True);target.write_text('\n'.join(lines)+'\n')
    db=sqlite3.connect(ROOT/'api/ledger.sqlite')
    ledger=db.execute('SELECT fingerprint,kind,status,confirmed,reserved FROM calls').fetchall();db.close()
    money={'confirmed':sum(x[3] for x in ledger),'reserved':sum(x[4] for x in ledger),
           'r2_episode_unique_confirmed':sum(x[3] for x in ledger if x[0] in fingerprints)}
    write_json(ROOT/'prep_r2_summary.json',{'arms':summaries,'notice_rows':len(notices),'cost':money,'complete':complete is not None})
    body=['# Luna 준비 실험 2회차 — 사람 검토 반영','','**평가 50편은 실행하지 않았다.** 같은 train 준비 글 10편을 검사 끔과 검사 켬으로 각각 실행했다. 최종 테스트 원문은 열지 않았고 8B 학습도 실행하지 않았다.','',
          '## 결과','','| 항목 | 검사 끔 | 검사 켬 |','|---|---:|---:|']
    keys=[('완료 글','essays'),('원문 그대로인 글','unchanged_essays'),('계획 문단','paragraphs'),('원래 문단 유지','kept_paragraphs'),('재시도','retries')]
    for name,key in keys:body.append('| '+name+' | '+' | '.join(str(summaries[a][key]) for a in PHASES)+' |')
    body.append('| 평균 글자 편집 비율 | '+' | '.join(f"{summaries[a]['mean_character_edit_ratio']:.1%}" if summaries[a]['mean_character_edit_ratio'] is not None else '미완료' for a in PHASES)+' |')
    body+=['','| 알림 유형 | 검사 끔 | 검사 켬 |','|---|---:|---:|']
    for phi in PHI:body.append('| '+phi+' | '+' | '.join(str(summaries[a]['notices'].get(phi,0))+(' **(0건)**' if not summaries[a]['notices'].get(phi,0) else '') for a in PHASES)+' |')
    body+=['','| 한국어 행동 | 검사 끔 | 검사 켬 |','|---|---:|---:|']
    for action in ['고침','유지','막힘']:body.append('| '+action+' | '+' | '.join(str(summaries[a]['actions'].get(action,0)) for a in PHASES)+' |')
    body+=['','## 고정한 기준','',f"δ={cfg['delta']:.6f}: 새 정책으로 의미 보존을 재검사한 표현 변경 {cfg['delta_sample_count']}개의 전체 글 점수 변화 절댓값 90백분위수. 수정량 상한={cfg['modification_limit']:.4f} ({cfg['modification_limit']:.1%}): 검사 끈 탐색에서 의미 검사를 통과한 {cfg['modification_calibration_samples']}개 문단의 글자 편집 비율 90백분위수. 검사 켠 실행 전에 고정했다.",'',
          '글자 편집 비율은 Levenshtein 최소 편집 글자 수 / max(수정 전 글자 수, 수정 후 글자 수)이다. Unicode 글자와 공백을 포함한다. 문단 재작성은 해당 문단, 구성은 글 전체에서 계산한다. 문장 비율은 정렬 진단에만 남기고 수정량 검사에는 사용하지 않는다.','',
          '## 반영한 정책과 환경','',
          '- thesis는 원문 문장의 직접 추출만 허용한다. 맞지 않으면 추출값을 한 번 교정하고, 여전히 원문에 없으면 실행 실패로 기록한다. 명시된 주장이 없으면 새 주장을 만들지 않는다.',
          '- P2/P4에서 새 주장·의견·해결책·대안을 금지하고 P6 new_content에 포함했다. 구성은 원문 문장 ID의 순서 변경·삭제만 허용하며 삭제 30% 상한은 검사 끔에서도 행동 유효성 제약으로 지킨다.',
          '- 구성 의미 검사는 P6와 분리했다. 삭제 때문에 원래 주장이 사라졌는지만 검사한다.',
          '- 알림 대상은 옮겨진 문장과 바뀐 문단 바로 바깥의 그대로 둔 문장이다. 재작성 문단 안의 문장에는 알림을 내지 않는다.',
          '- 지시 표현과 접속 표현은 바른 형태소·품사로 감지한다. 접속어 추가/삭제/교체와 앞 문장 변경은 별도 event로 기록한다.',
          '- 앞 문장 유사도 0.9 이상이고 내용어 multiset이 같으면 지시어 알림을 억제한다. 당위문 일반 주어도 제외한다.',
          '- 주제어는 주절 화제 표지 은/는/만/도를 먼저 찾고, 종속절 이/가는 제외한다. 절과 주절 구분은 바른 형태소에 근거한 보수적 표면 규칙이며 완전한 구문 분석이라고 주장하지 않는다.',
          '- 알림 관계 이름을 한국어로 표시하고 받침에 따라 조사를 고른다. source_sids는 대상 문장의 조상이 아니라 변화를 일으킨 문장 ID이다.',
          '- 고침 출력이 현재 문장과 같으면 유지로 기록한다. 허용 범위 밖 형태소가 바뀌면 막힘으로 기록하며 그 수리는 적용하지 않는다. 원래 요청과 실제 결정, 수리 전후를 모두 저장한다.',
          '- 정렬에는 1↔2, 1↔3 나눔·합침을 포함했다. 모호한 정렬 블록과 유사도도 보존한다.','',
          '검사 끔은 형식·행동 범위 제약을 지키면서 품질 때문에 문단 후보를 탈락시키지 않는다. 의미·점수 검사는 상한 보정과 비교를 위해 기록하지만 채택에는 쓰지 않는다. 검사 켬은 형식→보존→한국어→의미→점수 순으로 검사하고 실패하면 한 번 재시도한다.','',
          '문장 내부 연결어미는 직접 재작성한 문장을 알림에서 제외하면 관측되지 않을 수 있다. 0건은 그대로 표시했으며 가짜 알림으로 채우지 않았다. 문체 알림은 이음새에서 앞·뒤 문장 문체의 실제 변화가 새로운 불일치를 만들 때 낸다.','',
          '## 검사 기록','']
    for label in PHASES:
        s=summaries[label];body += [f"- {label}: 사유 {json.dumps(s['reasons'],ensure_ascii=False)}, 계획 파싱 {json.dumps(s['plan_parse'],ensure_ascii=False)}. 구성 유효성 거절 {len(s['structure_protocol_rejections'])}건."]
    body+=['','## 시간과 비용','']
    if complete:
        body += [f"검사 끈 탐색·잡음 재검사 {complete['parts_seconds']['checks_off_and_noise_calibration']:.1f}초, 검사 켠 실행 {complete['parts_seconds']['checks_on']:.1f}초, 재개 구간을 포함한 전체 {complete['wall_seconds_including_resume']:.1f}초. 글은 최대 4편 동시 처리하고 GPU 채점은 직렬화했다. 중간 집계 키 오류를 수정한 뒤 이미 완료된 탐색을 재사용했다. 단계 시간은 저장 시각과 에피소드 실측 시간으로 재구성해 재개 전에 끝난 작업도 포함했다."]
    body += ['',f"실험 장부 확정 비용 ${money['confirmed']:.6f}, 미확정 예약 ${money['reserved']:.6f}, 총 상한 $5. 두 재실행의 고유 호출 확정액 ${money['r2_episode_unique_confirmed']:.6f}; 잡음 재검사와 앞서 제출한 1회차 Claude 비용 측정은 전체 장부에 별도 포함된다.",'',
             '## 사람이 판정할 자료','',f"- [전체 알림 {len(notices)}행](luna_upper_bound_prep_r2_notices.md)",
             '- [전체 알림 CSV — 수리 전후 포함](luna_upper_bound_prep_r2_notices.csv)',
             '- [10편 원문·검사 끔·검사 켬 전체 글](luna_upper_bound_prep_r2_essays.md)',
             '- 전체 관찰·행동·검사 로그: `reports/luna_upper_bound_prep_r2_logs/`',
             '- 설정·원시 JSON·API 장부: `verak/v5/outputs/luna_upper_bound_v2/`','',
             '**판정(유효/잡음/형식 오류) 칸은 모두 비워 두었다.** 이 보고서는 재실행과 로그의 결과이며 사람의 유효성 판정이나 평가 50편의 성능을 대신하지 않는다.']
    report=REPORTS/'luna_upper_bound.md'
    archive=ROOT/'reports/before_r2_report.md'
    if report.exists() and not archive.exists():archive.parent.mkdir(parents=True,exist_ok=True);archive.write_text(report.read_text())
    report.write_text('\n'.join(body)+'\n')
    print(json.dumps({'report':str(report),'summary':summaries,'notice_rows':len(notices),'cost':money},ensure_ascii=False))


if __name__=='__main__':run()
