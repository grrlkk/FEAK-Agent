# Corruption rule v5 검색형 OFFTOPIC 결과

## 결론

rule v4의 `INSERT_OFFTOPIC`은 자동 점수와 2-LLM 선호 방향은 통과했지만, 생성문 93.2%에
일상·시간 cue가 들어가고 의미가 비슷한 여담 템플릿이 반복되는 문제가 있었다. rule v5는
LLM 자유 생성을 중단하고 다른 질문의 실제 학생 글 문장을 검색하는 방식으로 교체했다.

최종 TVM용 데이터는 오래된 OFFTOPIC과 그 뒤의 downstream transition을 버리고, 결함이
들어가기 전 clean prefix와 새 검색형 OFFTOPIC만 결합한다. 따라서 같은 `chain_id`가 stage별로
반복되는 정상적인 구조는 유지하지만, 자연 키 `(chain_id, stage_k)`는 중복되지 않는다.

## rule v5 생성 규칙

- distractor는 `data/corruption/main_1000.jsonl`의 다른 essay·다른 질문에서 가져온다.
- 1,000개 source 전체에 후보를 먼저 고유 배정하므로 shard가 달라도 삽입문이 재사용되지 않는다.
- 15~80자, 완결된 한 문장, 숫자·메타데이터·문맥 의존 접속어·인용 artifact를 제외한다.
- source와 같은 종결체를 사용하고, 앞부분에 명시적 주어/화제가 있는 문장만 허용한다.
- BGE-M3로 distractor와 질문·source의 cosine을 각각 0.50 이하로 제한한다.
- edit마다 `distractor_record_id`, `distractor_question`, 두 BGE 유사도를 저장한다.
- anchor/source 문장 복사, source 8-gram 겹침, `핵심 키워드:` anchor, 다문장 삽입은 hard reject한다.
- `이로 인한`, `그러면`, `두번째로`, `…뜻이 된다`처럼 donor 앞 문맥을 요구하는 문장도
  생성과 최종 조립 단계에서 hard reject한다.

## 30개 파일럿

- 생성: 30/30 성공, 60개 삽입문 모두 고유
- 사전 감사: exact/n-gram violation 0, 의미 군집 violation 0, provenance 오류 0
- 질문/source BGE 0.50 초과 0
- 일상·시간 cue: 2/60(3.3%); rule v4의 136/146(93.2%)에서 크게 감소
- G1: 18/30 수용(60.0%), 수용 target drop 중앙값 0.698, 평균 0.790
- 두 독립 API 모델(`gpt-5-mini-2025-08-07`, `gpt-4.1-2025-04-14`):
  - 각 18/18 clean 방향 선택, 모델 간 일치 18/18
  - corrupted 글의 국소 유창성 3점 이상: 두 모델 모두 18/18
  - 평균 국소 유창성: 3.39 / 3.50

pair-level `canned_artifact` 응답은 각각 83.3%, 66.7%였지만, 근거를 확인하면 거의 모두
“질문과 무관한 문장이 눈에 띈다”였다. 이는 의도한 OFFTOPIC 결함과 템플릿 artifact를 구분하지
못하는 construct이므로 진단값으로만 보존했다. 템플릿 게이트는 corpus 전체를 보는 exact/n-gram,
BGE 군집, 첫 토큰, cue 분포로 수행한다.

## 200개 본셋 처리

- 기존 chain 중 OFFTOPIC 포함: 147개
- 검색형 OFFTOPIC prefix 재생성 성공: 144개
- 안전하게 제외: 3개
  - BGE 질문 유사도 상한을 만족하는 후보 부족 2개
  - 앞선 corruption 때문에 두 문장 삽입 시 source 길이 상한 초과 1개
- 새 삽입문: 288개, 문자열 중복 0, 의미 violation 군집 0, 관련성 초과 0
- 일상·시간 cue: 21/288(7.3%)
- 오래된 OFFTOPIC 현재/후속 transition은 모두 최종 pool에서 제외한다.
- G1: 144개 replacement 중 77개 수용(53.5%)
  - 수용 target drop 중앙값 0.607, 평균 0.650, 최솟값 0.227
- 2-LLM 50쌍 본셋 검수에서 방향성은 50/50, 49/50이었지만, 낮은 국소 유창성 근거를
  확인해 실제 donor-context 의존 문장 6개를 추가 제외했다.
- 최종 데이터에 남은 검수 표본 44쌍:
  - clean 방향 선택 44/44, 43/44; 모델 간 일치 43/44
  - 독립 재판정 후 44/44
  - corrupted 국소 유창성 3점 이상 41/44(93.2%), 36/44(81.8%)
  - 두 모델 모두 사전 기준 70% 방향성, 80% 국소 유창성을 통과

## 최종 clean-prefix pool

- 최종 accepted: 193행, 140개 essay
- 기존 accepted 256행 중 clean prefix 122행 유지
- 오래된 OFFTOPIC 현재/후속 134행 제거
- 새 검색형 OFFTOPIC: G1 수용 77행 중 donor-context 감사 통과 71행
- 연산자: DELETE 33, SHUFFLE 33, LEX 56, OFFTOPIC 71
- 최대 연산자 비율: OFFTOPIC 36.8%로 40% 상한 통과
- 자연 키 `(chain_id, stage_k)` 중복 0, 완전 동일 row 중복 0
- OFFTOPIC 삽입문 142개 모두 고유, provenance 누락 0
- 질문/source BGE 최대 유사도 0.475/0.494로 0.50 상한 통과
- exact/n-gram, BGE 의미 군집, 출처, 관련성, cue 분포 감사 모두 통과

## 산출물

- 파일럿 chain: `experiments/results/corruption_g1_rulev5_retrieval_offtopic_30_chains.jsonl`
- 파일럿 accepted: `experiments/results/corruption_g1_rulev5_retrieval_offtopic_30_accepted.jsonl`
- 파일럿 2-LLM 보고서: `experiments/results/corruption_g1_rulev5_retrieval_offtopic_30_two_llm_api_report.json`
- 본셋 replacement chain: `experiments/results/corruption_g1_rulev5_retrieval_offtopic_200_prefix_ok.jsonl`
- 본셋 생성 감사: `experiments/results/corruption_g1_rulev5_retrieval_offtopic_200_generated_quality.json`
- 본셋 replacement G1 요약: `experiments/results/corruption_g1_rulev5_retrieval_offtopic_200_summary.json`
- 본셋 2-LLM 최종 보고서: `experiments/results/corruption_g1_rulev5_retrieval_offtopic_200_two_llm_api_curated_report.json`
- 최종 accepted: `experiments/results/corruption_g1_rulev5_200_accepted.jsonl`
- 최종 보고서: `experiments/results/corruption_g1_rulev5_200_report.json`
