# FEAK-Agent

한국어 글의 상태를 **기존 Kanana 채점기**로 측정하고, **수정 전후 직접 비교(RV)**로
수정본의 채택을 결정하는 학습 없는 연구 파일럿입니다.

현재 구현 기준은 사용자가 지정한 **FEAK_RV_PILOT_IMPLEMENTATION.md**입니다.
[파일럿 안내](docs/RV_PILOT.md)에 실행·점수 척도·로그 계약이 있습니다.

```text
Current draft → Kanana → Planner: 문제 하나 → Reviser: 후보 하나
       ↑                                      ↓
       └──────── ACCEPT ← Controller ← RV: 네 기준
```

RV는 목표 달성, 수정 필요성, 의미 보존, 문서 전체의 이득을 PASS/FAIL/UNCERTAIN으로
판정합니다. 모두 PASS인 후보만 채택합니다. 점수 상승은 채택 조건이 아닙니다.
거절 후보의 재수정 1회, 불확실 판정의 독립 재검증 1회, 반복 최대 3회입니다.

## 실행

기존 feak_agent 환경과 형제 저장소 essay_scoring_llm의 채점기·보정 모델을 사용합니다.
GPT 기본 모델은 gpt-5-mini, reasoning low입니다. .env의 OPENAI_API_KEY 또는
FEAK_ENV_FILE을 사용합니다. 모델 가중치·키·결과는 커밋하지 않습니다.

```bash
conda activate feak_agent
python scripts/check_env.py

# 모델/API 없는 전체 제어 흐름 확인
python scripts/run_pilot.py --input examples/pilot_samples.jsonl --offline-smoke --output-dir experiments/results/rv_pilot_offline

# 실제 Kanana + GPT: 합성 예제 5편, 최대 3 iterations
python scripts/run_pilot.py --input examples/pilot_samples.jsonl --output-dir experiments/results/rv_pilot_real

python -m pytest -q
```

출력은 매번 새 경로를 지정합니다. 예제 5편은 직접 작성한 실행 확인용 글이며 논문 성능 평가
데이터가 아닙니다. API 실행은 문항과 글을 OpenAI로 전송합니다.

## 사람 블라인드 평가

저장된 파일럿 후보를 사람이 평가하려면 [블라인드 평가 화면](docs/HUMAN_REVIEW.md)을 사용합니다.
네 기준의 판정·근거를 평가자별로 저장하며 모델/API 호출 없이 실행됩니다.

```bash
python scripts/run_pilot_review.py prepare --run-dir experiments/results/rv_pilot_real --study-dir experiments/results/human_review_study --raters 2
python scripts/run_pilot_review.py serve --study-dir experiments/results/human_review_study
python scripts/run_pilot_review.py links --study-dir experiments/results/human_review_study
```

## 구조

```text
feak_tc/agent/          현재 4기준 RV: 계획·수정·검증·제어·점수 어댑터
feak_tc/review/         현재 파일럿의 사람 블라인드 평가 화면·저장
feak_tc/runtime/        공용 Kanana 프로세스·GPT 구조화 출력 통신
feak_tc/diagnose/       기존 채점기 연결
configs/pilot_gpt.yaml  현재 설정과 호출 예산
scripts/run_pilot.py    현재 CLI 진입점
examples/              실행 확인용 예제
feak_tc/legacy/agent/   과거 2축 RV·점수 선택·경로 가드 구현
```

mvp/, rv/, corruption/, 데이터 도구와 기존 에이전트 웹은 이전 실험 재현용입니다.
scripts/run_agent.py와 scripts/run_agent_web.py는 과거 루프를 실행합니다.
이전 설명은 [보관된 README](docs/LEGACY_AGENT_README.md), 이번 변경은
[정리 기록](docs/CLEANUP_2026-09-28.md)을 참고하세요.

## 연구 배경

[FEAK](https://github.com/grrlkk/FEAK)의 채점·자질 계산을 재사용합니다.

> From Evaluation to Feedback: A Feature-Based and LLM-Constrained Tool for Korean Writing Assessment.
> Chanwoo Jang et al., ACM SAC 2026. [DOI](https://doi.org/10.1145/3748522.3780021)

실행 성공은 품질 향상이나 수정 판정 정확도를 입증하지 않습니다.
라이선스: MIT. 서드파티 고지: [NOTICE](NOTICE).
