# FEAK-Agent

총괄·글수정·한국어 에이전트와 corruption 데이터 생성 코드를 모은 저장소입니다.
작업 폴더는 `/home/chanwoo/FEAK-Agent` 하나를 사용합니다.

현재 중심 구현은 **총괄·글수정·한국어의 3에이전트**입니다.
총괄이 진단·계획·위임을 하고, 글수정 에이전트가 내용·구성을 수정하며,
한국어 에이전트가 표현·문장 연결·문체를 점검합니다.
이 경로에는 수정본마다 채택 여부를 판단하는 별도 RV 에이전트가 없습니다.
도구 실행 제약 검사와 저장된 결과에 대한 품질 평가는 별개입니다.

과거 **Planner → Reviser → RV → Controller** 코드는 `feak_tc/agent/`와
`verak/src/`에 남아 있습니다. `scripts/run_pilot.py`와
`python -m verak.src.run_single --mode loop`는 그 RV 경로를 실행합니다.
RV 경로와 현재 3에이전트 경로를 같은 방법으로 소개하지 않습니다.

## 구현 위치

| 목적 | 코드 | 설명 |
|---|---|---|
| 3에이전트 실행·협업 | `verak/v5/cooperate.py`, `verak/v5/cooperate_env.py` | 총괄이 글수정/한국어 역할에 위임하고 결과를 검토합니다. |
| 총괄 에이전트 | `verak/v4/orchestrator_v44.py` | 진단·계획·위임·종료 판단입니다. |
| 글수정·한국어 실행 환경 | `verak/v4/environment_v46.py`, `verak/v4/policy_prompts_v43.py` | 역할별 편집 도구, 한국어 관찰, 프롬프트를 제공합니다. |
| API 모델로 구성한 단순 3역할 비교 | `verak/v5/luna_simple.py`, `verak/v5/luna_simple_prompts.json` | 총괄 → 글수정 → 한국어 순서입니다. 기존 train 10편에 한정된 실험 실행기입니다. |
| Corruption 생성 | `verak/v3/corrupt/` | 연산자·후보 생성·품질 검사·최종 데이터 구성을 담당합니다. |
| 이전 corruption 생성 도구 | `feak_tc/corruption/`, `scripts/build_corruption_rulev5_dataset.py` | 이전 방식의 데이터 생성 코드입니다. |
| 초기 피드백 실험 | `staged_feedback/src/staged_feedback/` | LIFT 이전의 총괄·피드백·자동 수정 구현입니다. |
| **LIFT 논문 구현** | [grrlkk/LIFT](https://github.com/grrlkk/LIFT) | 별도 저장소의 총괄·피드백 구조입니다. |

`verak/v3`, `v4`, `v5`는 서로 가져다 쓰는 모듈입니다. 별도 작업 폴더 복제본이 아니며,
버전 이름만 보고 삭제하면 3에이전트와 데이터 생성 코드의 import가 깨집니다.

## LIFT 논문과의 관계

논문 PDF: `LIFT) Learning-Oriented Iterative Writing Feedback Agents.pdf`.
LIFT는 **Coordinator + Feedback + Shared profile** 구조입니다.
학생 대신 글을 고치는 Student Simulator는 실험용이며, 세 번째 지도 에이전트가 아닙니다.
FEAK의 총괄·글수정·한국어 3에이전트와 구분합니다.

실제 구현은 별도 저장소 `grrlkk/LIFT`, 로컬 `/home/chanwoo/LIFT`에 있습니다.

- `writing_feedback/roles.py`: 총괄 초기 계획·검토, 피드백 생성.
- `writing_feedback/engine.py`: 공유 상태 갱신, 피드백·학생 수정·재계획 반복.
- `writing_feedback/schemas.py`: 요구사항·이슈·전략·수정 결과 스키마.
- `writing_feedback/simulator.py`: 실험용 학생 수정.
- `writing_feedback/paper_scenarios.py`: 논문의 초기 계획 및 재계획 사례.
- `writing_feedback/resources/prompts/`: 실행에 필요한 역할별 Markdown 프롬프트.

논문의 두 초안 사례(6643·7644행)와 재계획 사례(6252행)가 이 프로젝트와 연결됩니다.
다만 PDF의 100편 결과와 로컬 `runs/random30_20261017`의 30편 결과는 다릅니다.
제출본 수치 전체의 재현 버전으로 확정하려면 100편 실행 기록을 추가로 확인해야 합니다.

## 실행 안내

기존 환경은 `/home/chanwoo/anaconda3/envs/feak_agent/bin/python`입니다.
프로젝트 루트에서 실행합니다. 다음 명령은 사용법만 표시합니다.

```bash
conda activate feak_agent
python -m verak.v5.cooperate --help
python -m verak.v5.luna_simple --help
python -m verak.v3.cli.prepare_corruption_sources --help
python -m verak.v3.cli.build_corruption_candidates --help
python -m verak.v3.cli.finalize_corruptions --help
```

Corruption 파이프라인은 원본 선정 → 후보 생성 → QC → 최종 데이터 구성 순서입니다.
설정은 `verak/v3/config.yaml`, 핵심 연산자는 `verak/v3/corrupt/operators.py`,
후보 생성은 `instance_policy.py`, 최종 구성은 `final_corpus.py`입니다.
이미 생성된 데이터는 `verak/v3/data/`에 보존합니다.

3에이전트 실행기는 기존 체크포인트·코호트·예산에 연결된 연구용 실행기입니다.
실제 실행에는 해당 설정, 모델, Bareun 등의 의존성이 필요합니다.

```bash
python -m pytest -q
python -m pytest -q --import-mode=importlib verak/v3/tests verak/v4/tests verak/v5/tests
PYTHONPATH=staged_feedback/src python -m pytest -q -c staged_feedback/pyproject.toml staged_feedback/tests

# LIFT 구현의 단위 테스트
cd /home/chanwoo/LIFT
python -m pytest -q
```

## 정리 및 보존

2026-11-02: 최신 통합 소스를 본 폴더에 복원하고 중복 worktree와 과거 로컬 브랜치,
실험 Markdown 문서를 정리했습니다. 설명은 이 README로 통합했습니다.
`AGENTS.md`는 저장소 작업 지침입니다.
원본 데이터·모델·JSON 실행 기록·PDF·로컬 설정은 보존합니다.

정리 전 Git 이력, 미커밋 작업, 제거한 문서의 복구 자료는
`/home/chanwoo/.local/share/feak-cleanup/20261102-134040/`에 있습니다.
`all-refs.bundle`은 전체 브랜치 이력, `worktrees/*.tar.gz`는 제거한 작업 폴더,
`removed-documents.tar.gz`는 문서 사본입니다. 복구 안내는 같은 폴더의 `RESTORE.txt`입니다.
원격 실험 브랜치는 정리하고 `main`을 유지합니다. 별도 커밋이 있던 브랜치는
`archive/2026-11-02/` 태그에 보존합니다. 로컬 전체 Git 백업도 보관합니다.

라이선스: MIT. 서드파티 고지: [NOTICE](NOTICE).
