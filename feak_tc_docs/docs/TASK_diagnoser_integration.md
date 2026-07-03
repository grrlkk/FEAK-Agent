# TASK — 채점기(Diagnoser) 통합 작업

> Claude Code 실행용 지시서. 목표: 동료의 kanana 기반 채점기(essay_scoring_llm)를
> MVP의 diagnose 단계에 연결하되, 기존 FEAK(KoBERT) 버전도 레거시로 남긴다.
> 두 채점기를 공통 인터페이스 뒤에 두어 config로 갈아끼울 수 있게 한다.

## 컨텍스트

- 프로젝트 루트: `FEAK-Agent` (기존 FEAK KoBERT 채점기가 이미 여기 있음)
- 새 채점기: https://github.com/ghko99/essay_scoring_llm (kanana-8B + LoRA)
  - rubric 8개 점수 + FEAK 29 언어자질(`extract_feak_features`)을 **둘 다** 제공
- 설치 방식: **방법 A (editable 설치)** — 별도 위치에 clone 후 `pip install -e`
- 전체 배경은 `docs/PROJECT_CONTEXT.md` 참조 (특히 "채점기에 자질 안 넣음", rubric/자질/transition feature 용어 구분)

## 절대 규칙

- 기존 FEAK(KoBERT) 채점 코드는 **삭제·수정하지 않는다.** 어댑터로 감싸기만.
- 동료 패키지(essay_scoring_llm) 소스는 수정하지 않는다. import해서 호출만.
- 두 채점기 모두 **글 → rubric(+자질)** 만 한다. 자질을 채점기 입력으로 넣지 않는다.
- 확실하지 않은 함수 시그니처는 추측하지 말고, 소스를 열어 실제 시그니처를 확인한다.

---

## STEP 0 — 동료 패키지 설치 및 소스 파악

**설치 위치 (중요)**: `/home/chanwoo/essay_scoring_llm`
즉 프로젝트 폴더 `/home/chanwoo/FEAK-Agent` 의 **밖, 홈 바로 밑 형제 폴더**로 clone한다.
`FEAK-Agent` 안에 clone하지 말 것 (중첩 git 충돌 / 남의 코드와 내 코드 혼동 방지).
editable 설치이므로 위치가 프로젝트 밖이어도 `import essay_scoring_llm` 이 정상 동작한다.

```bash
cd /home/chanwoo                     # 프로젝트 밖, 홈 바로 밑
git lfs install
git clone https://github.com/ghko99/essay_scoring_llm.git
cd essay_scoring_llm
git lfs pull
pip install -e ".[live]"     # live extra = 자질 추출 포함
```

결과 구조:
```
/home/chanwoo/
├── FEAK-Agent/          ← 이 프로젝트 (여기서 코드 작성)
└── essay_scoring_llm/   ← 동료 채점기 (형제 폴더, editable 설치)
```

설치 후 **패키지 소스를 직접 열어 다음을 확인**하고 요약할 것:
1. "글 하나 → rubric 점수 dict"를 리턴하는 **파이썬 함수**가 무엇인가?
   (single CLI 커맨드가 내부적으로 부르는 함수. `cli.py`, `pipeline.py`, `scorer.py` 등 탐색)
2. 그 함수의 정확한 **입력/출력 시그니처** (인자, 반환 타입, rubric 키 이름 8개)
3. `extract_feak_features(text)` 의 반환형 (자질 29개의 키 이름)
4. 모델/LoRA 가중치 로딩 방식 (한 번 로드 후 재사용 가능한가 — 매 호출마다 로드하면 안 됨)

> 이 4가지를 확인하기 전에는 어댑터 코드를 확정하지 말 것. 확인 결과를 사용자에게 먼저 요약.

---

## STEP 1 — Diagnoser 인터페이스 정의

`FEAK-Agent/src/feak_tc/diagnose/base.py`:

```python
from dataclasses import dataclass
from typing import Protocol

@dataclass
class Diagnosis:
    text: str
    rubrics: dict[str, float]      # 8개 rubric 점수
    features: dict[str, float]     # 29개 언어자질
    weak_rubrics: list[str]        # 하위 N개 (규칙 선별)

class Diagnoser(Protocol):
    def diagnose(self, text: str) -> Diagnosis: ...
```

- rubric 키 이름, feature 키 이름은 STEP 0에서 확인한 **실제 이름**으로 상수화 (`schemas.py`).

---

## STEP 2 — kanana 어댑터 (기본 구현)

`FEAK-Agent/src/feak_tc/diagnose/kanana.py`:

```python
from essay_scoring_llm...  import <채점함수>          # STEP 0에서 확인한 실제 경로
from essay_scoring_llm.feak import extract_feak_features
from .base import Diagnosis, Diagnoser

class KananaDiagnoser:
    def __init__(self, ...):
        # 모델/LoRA를 여기서 1회 로드하고 재사용 (매 diagnose 호출마다 로드 금지)
        ...
    def diagnose(self, text: str) -> Diagnosis:
        rubrics = <채점함수>(text)              # dict[str,float]
        features = extract_feak_features(text)  # dict[str,float]
        weak = self._select_weak(rubrics)
        return Diagnosis(text, rubrics, features, weak)
    def _select_weak(self, rubrics, top_n=3):
        return [k for k,_ in sorted(rubrics.items(), key=lambda kv: kv[1])[:top_n]]
```

- 모델 로딩은 무겁다. `__init__`에서 1회, 이후 재사용. 필요하면 lazy singleton.

---

## STEP 3 — FEAK(KoBERT) 레거시 어댑터

`FEAK-Agent/src/feak_tc/diagnose/feak_kobert.py`:

- FEAK-Agent에 이미 있는 KoBERT 채점 코드를 **감싸기만** 한다 (원본 수정 금지).
- 같은 `Diagnosis`를 반환하도록 어댑터 작성.
- rubric/feature 키 이름이 kanana 버전과 다르면, 공통 스키마로 **매핑 테이블**을 둔다
  (두 채점기가 같은 인터페이스로 보이도록). 매핑이 불명확하면 사용자에게 확인.

---

## STEP 4 — 선택 스위치

`FEAK-Agent/src/feak_tc/diagnose/__init__.py`:

```python
def get_diagnoser(kind: str = "kanana", **kw) -> Diagnoser:
    if kind == "kanana":
        from .kanana import KananaDiagnoser
        return KananaDiagnoser(**kw)
    if kind == "feak_kobert":
        from .feak_kobert import FeakKobertDiagnoser
        return FeakKobertDiagnoser(**kw)
    raise ValueError(kind)
```

- `configs/diagnose.yaml` 에 `diagnoser: kanana` (기본). FEAK로 되돌리려면 이 한 줄만 변경.
- 이렇게 두면 나중에 baseline 비교(FEAK vs kanana 진단)도 config로 전환 가능.

---

## STEP 5 — MVP 루프 연결 및 스모크 테스트

- MVP 루프(`loop.py`)는 `Diagnoser` 인터페이스만 의존하게 한다 (구체 구현 직접 import 금지).
- `scripts/smoke_diagnose.py`: 샘플 글 1개를 `get_diagnoser("kanana")`로 진단하여
  rubrics/features/weak_rubrics 출력. 정상 동작 확인.
- 같은 글을 `get_diagnoser("feak_kobert")`로도 돌려 두 채점기 출력 비교 로그.

---

## STEP 6 — 문서화

- `docs/DIAGNOSER_INTEGRATION.md` 생성: STEP 0에서 확인한 실제 함수 시그니처,
  rubric/feature 키 이름 목록, 두 채점기 키 매핑 테이블, 설치 방법 기록.
- 이 문서는 이후 transition feature 계산(`transition.py`)의 입력 명세가 된다.

---

## 완료 기준

- [ ] essay_scoring_llm editable 설치 완료, import 성공
- [ ] 실제 채점 함수 시그니처 확인 및 문서화
- [ ] `get_diagnoser("kanana")` 로 샘플 글 진단 성공 (rubric+feature 반환)
- [ ] `get_diagnoser("feak_kobert")` 레거시 경로 동작 (원본 코드 미수정)
- [ ] MVP 루프가 Diagnoser 인터페이스로만 채점기에 접근
- [ ] 두 채점기 출력 비교 로그 확인

## 주의 (실패 방지)

- LFS: 모델 가중치가 Git LFS일 수 있음. `git lfs pull` 안 하면 빈 포인터 파일 → 로딩 실패.
- GPU 메모리: kanana-8B 로딩은 VRAM 사용. 채점기(어댑터A)와 이후 TVM(어댑터B)이
  동시 로드되지 않도록 lazy 로딩/언로드 고려 (MVP 단계에선 채점기만 로드).
- 시그니처 불명확 시: 코드를 짜기 전에 소스를 열어 확인하고 사용자에게 보고.
