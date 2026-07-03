# Diagnoser Integration

## Scope

FEAK-TC keeps the legacy FEAK/UKTA KoBERT scorer untouched and adds a common
Diagnoser interface for the new Kanana scorer.

The MVP loop depends only on:

```python
diagnoser.diagnose(text) -> Diagnosis
```

## Kanana Scorer Package

Sibling repository:

```text
/home/chanwoo/essay_scoring_llm
```

Editable install was verified with:

```bash
python -m pip install -e /home/chanwoo/essay_scoring_llm
```

Import verification:

```python
import essay_scoring_llm
from essay_scoring_llm.rubrics import RUBRIC_KEYS, FEAK_NAMES
```

## Confirmed Kanana Functions

The live single-essay pipeline is implemented in:

```python
essay_scoring_llm.single.score_essay(
    question: str,
    essay: str,
    config: essay_scoring_llm.config.ScoringConfig,
    keywords: Optional[str] = None,
    rater1=None,
    rater2=None,
) -> dict
```

Internally it calls:

```python
essay_scoring_llm.soft_sc.load_model_and_tokenizer(config)
essay_scoring_llm.soft_sc.build_user_prompt(question, essay, keywords)
essay_scoring_llm.soft_sc.score_example(model, tokenizer, helper, system, user, config)
essay_scoring_llm.feak.extract_feak_features(essay)
essay_scoring_llm.correction.CorrectionModule.predict(means, stds, feak_values)
```

The FEAK-TC adapter uses these lower-level functions so the Kanana model is
loaded once in `KananaDiagnoser.__init__`/lazy load and reused across calls.

## Rubric Keys

The Kanana scorer uses this fixed 8-rubric order:

```text
task_1
content_1
content_2
content_3
organization_1
organization_2
expression_1
expression_2
```

Korean display names:

```text
task_1          과제충실성
content_1       설명명료성
content_2       설명구체성
content_3       설명적절성
organization_1  문장연결성
organization_2  글통일성
expression_1    어휘적절성
expression_2    어법적절성
```

Score scale: `1..9`.

FEAK-TC local constants in `feak_tc.diagnose.constants` were checked against
`essay_scoring_llm.rubrics` and match exactly.

## FEAK Feature Keys

The Kanana scorer exposes 29 FEAK features through:

```python
essay_scoring_llm.feak.extract_feak_features(text) -> dict[str, float]
```

Feature order:

```text
C_Cnt
E_Cnt
F_Cnt
J_Cnt
X_Cnt
char_Cnt
word_Cnt
E_NDW
morph_NDW
morph_LenAvg
morph_LenStd
word_LenStd
grade_2_ratio
grade_3_ratio
grade_4_ratio
grade_m1_ratio
N_MSTTR
V_HDD
lemma_MATTR
2-gram_NDW
NN_repRatio
word_sentLenAvg
2-gram_RTTR
adjacent_sentence_overlap_function_lemmas
char_paraLenAvg
avgSentSimilarity
topicConsistency
text_dalechall
text_oridx
```

## Current Environment Status

Working in the shared `feak_agent` conda environment:

- `import essay_scoring_llm`
- `RUBRIC_KEYS` and `FEAK_NAMES` import
- `essay_scoring_llm.feak.extract_feak_features(...)` returns 29 FEAK features
- Bareun server on `localhost:5656`
- local Kanana base model at
  `/home/chanwoo/FEAK-Agent/.hf_cache/kanana-1.5-8b-instruct-2505`
- live Kanana smoke diagnosis on GPU 3

## FEAK-TC Adapters

Implemented:

```text
feak_tc/diagnose/base.py        Diagnosis + Diagnoser protocol
feak_tc/diagnose/kanana.py      KananaDiagnoser lazy adapter
feak_tc/diagnose/feak_kobert.py Legacy wrapper, source code untouched
feak_tc/diagnose/stub.py        Offline deterministic diagnoser for MVP tests
```

Default local smoke tests use `StubDiagnoser`. True Kanana MVP execution should
use:

```bash
python scripts/run_mvp.py \
  --diagnoser kanana \
  --device-id 3 \
  --question "인권의 뜻과 특징에 대해 서술하세요" \
  --keywords "인간(사람), 당연, 권리, 존중(침해)" \
  --text-file sample_essay.txt
```

The Kanana adapter extracts FEAK features in an isolated subprocess before
loading the 8B scorer. This avoids keeping feature-extractor GPU allocations
alive during Kanana generation.
