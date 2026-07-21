# 중간정리 — soft threshold 후보 직접 질적 검토

> 작성일: 2026-07-13
> 기준 로그: `experiments/results/mvp_stage_a_20_v3.jsonl`
> 검토 대상: `target_gain_min=0.3`, `non_target_drop_max=1.0`에서 accept된 7개 후보 + 0.3 근처 near-miss 후보

## 1. 결론 요약

숫자만 보면 `target_gain_min=0.3`이 적당해 보였지만, 직접 읽어보면 그대로 쓰기 어렵다.

가장 중요한 이유는 두 가지다.

1. **잘못된 patch가 threshold를 통과했다.**
   - 문장이 중복되거나
   - 실제 삭제된 문장이 target_span과 다르거나
   - 글의 첫 문장이 사라져 문맥이 깨지는 경우가 있었다.

2. **반대로 안전한 작은 수정은 0.3 아래에서 막혔다.**
   - 띄어쓰기 수정처럼 실제로 좋은 style fix는 `target_gain=0.244`라 reject된다.

따라서 현재 상태에서는 threshold만 정하면 안 된다.
먼저 patch validity와 factuality guard를 강화한 뒤 threshold를 다시 봐야 한다.

## 2. target_gain_min = 0.3 accept 후보 7개 검토

| record | action | gain | 사람이 본 판정 | 이유 |
|---|---|---:|---|---|
| train_45 | ADD_DETAIL | 0.760 | 보류 | 투표권법 예시는 구체적이지만, 원래 발문인 인권의 뜻/특징과는 초점이 어긋난 글이다. 내용 구체성은 늘지만 과제 충실성을 근본적으로 고치지는 못한다. |
| train_59 | RESTRUCTURE | 0.438 | reject | 첫 문장이 깨지고 같은 히잡 문장이 중복된다. patch가 target_span을 제대로 바꾸지 못했다. |
| train_70 | ADD_DETAIL | 0.390 | reject 또는 보류 | “외국인 노동자의 60%가 임금 차별”이라는 수치를 새로 만든다. 근거 없는 통계 삽입 위험이 있다. |
| train_85 | DELETE_OR_FOCUS | 0.780 | reject | target_span과 실제 삭제된 문장이 다르다. 글의 첫 문장이 사라져 글이 “얼굴 인식 기능이...”로 시작한다. 문맥이 깨졌다. |
| train_94 | ADD_DETAIL | 0.676 | accept | 스페인의 아메리카 금·은 착취 사례를 추가해 구체성이 좋아진다. 약간 반복적이지만 큰 훼손은 없다. |
| train_97 | ADD_DETAIL | 0.625 | weak accept | 외국인 노동자 차별의 일반 사례를 추가한다. 원문에 이미 구체 예시가 있어 효과는 크지 않지만, 문맥 훼손은 작다. |
| train_114 | RESTRUCTURE | 0.534 | weak accept | “권리 따윈”을 “권리는”으로 고쳐 표현이 명확해진다. 다만 실제 action은 RESTRUCTURE보다 STYLE_REFINE에 가깝다. |

## 3. 0.3 accept 후보 중 실제로 쓸 만한 것

엄격하게 보면 강한 accept는 많지 않다.

```text
확실히 쓸 만함: train_94
약하게 쓸 만함: train_97, train_114
보류: train_45
reject: train_59, train_70, train_85
```

즉 숫자상 7개 accept였지만, 사람이 보면 실제 유효 후보는 약 2~3개 정도다.

## 4. near-miss 후보 검토

`target_gain_min=0.3` 바로 아래에서 막힌 후보도 읽어봤다.

| record | action | gain | 사람이 본 판정 | 이유 |
|---|---|---:|---|---|
| train_40 | ADD_DETAIL | 0.214 | weak accept | “장애인, 여성, 빈민층도 인권을 가지고 있다”를 조금 더 구체화한다. 큰 개선은 아니지만 자연스럽다. |
| train_90 | ADD_DETAIL | 0.297 | weak accept | “인권 침해에 해당한다” 뒤에 흑인의 고통을 보충한다. 거의 0.3에 가까운 합리적 수정이다. |
| train_91 | STYLE_REFINE | 0.244 | accept | “이주노동자는기본적인” 띄어쓰기 오류를 고친다. 작은 수정이지만 명백히 좋다. |
| train_99 | ADD_DETAIL | 0.234 | 보류 | 히잡 설명을 조금 명확히 하지만, 마지막 문장 반복 같은 더 큰 문제는 그대로 남는다. |
| train_117 | ADD_DETAIL | 0.290 | 보류 | “많은 사람들이 고통받았다”는 너무 일반적이다. 원문에 이미 삼각무역 설명이 있어 정보 추가 효과가 약하다. |

near-miss 중에는 오히려 `train_91`처럼 명확하게 좋은 작은 수정이 있다.
따라서 단일 `target_gain_min`만으로 모든 action을 처리하면 style fix가 과도하게 불리해진다.

## 5. 발견된 구조적 문제

### 문제 1 — patch consistency 검증 부족

`train_59`, `train_85`에서 명확히 드러났다.

- candidate의 `target_span`과 patch의 `before`가 다르다.
- 실제 수정된 위치가 target_span이 아니다.
- 삭제 결과가 문맥을 깨뜨린다.

필요한 guard:

```text
patch.before가 원문에 존재해야 함
patch.before와 target_span이 크게 어긋나면 reject
operation 대상과 candidate target_span 불일치 시 reject
수정 후 문장 중복 증가 시 reject
삭제 후 첫 문장/도입부가 깨지면 reject
```

### 문제 2 — ADD_DETAIL의 근거 없는 사실 삽입

`train_70`에서 “최근 조사 60%” 같은 근거 없는 수치가 추가됐다.

필요한 guard:

```text
원문에 없는 정확한 수치, 연도, 기관, 조사 결과를 새로 만들면 reject 또는 penalty
예시는 가능하지만 통계/고유사실은 원문 기반일 때만 허용
```

### 문제 3 — action label과 실제 수정 불일치

`train_114`는 RESTRUCTURE로 들어왔지만 실제로는 표현 수정이다.

필요한 guard:

```text
RESTRUCTURE는 문장 순서, 연결어, 문장 관계 변화가 있어야 함
STYLE_REFINE은 작은 어휘/띄어쓰기/표현 수정 허용
action_type과 실제 patch 유형이 어긋나면 penalty 또는 relabel
```

### 문제 4 — action별 threshold가 필요할 가능성

ADD_DETAIL은 큰 gain을 요구해도 되지만, STYLE_REFINE은 작은 gain이어도 좋은 수정일 수 있다.

예:

```text
train_91 STYLE_REFINE
gain = 0.244
사람 판정 = accept
```

따라서 모든 action에 같은 `target_gain_min`을 적용하면 style/grammar 계열 수정이 불리하다.

## 6. threshold에 대한 수정된 판단

이전 sweep만 보면 다음이 좋아 보였다.

```text
target_gain_min = 0.3
non_target_drop_max = 1.0
```

하지만 직접 읽은 뒤에는 이렇게 보는 것이 더 맞다.

```text
0.3은 noise 기준으로는 합리적인 시작점이다.
그러나 현재 validity/patch guard가 약해서 그대로 쓰면 안 된다.
```

즉 다음 순서는 threshold 확정이 아니라 guard 보강이다.

## 7. 바로 다음 작업 제안

### 1순위 — patch validity 강화

먼저 아래 reject rule을 추가한다.

```text
1. patch.before가 원문에 없으면 reject
2. target_span과 patch.before가 너무 다르면 reject
3. replace/delete가 target_span 밖을 건드리면 reject
4. 수정 후 같은 문장 또는 긴 n-gram 반복이 증가하면 reject
5. 삭제 후 글이 도입 없이 시작하거나 문맥 주어가 사라지면 reject
```

### 2순위 — ADD_DETAIL factuality guard

```text
원문에 없는 숫자/연도/조사/통계/기관명이 새로 생기면 reject 또는 penalty
```

### 3순위 — action별 threshold 검토

예시:

```text
ADD_DETAIL: target_gain_min 0.3 이상
RESTRUCTURE: target_gain_min 0.3 이상 + repetition guard
STYLE_REFINE: target_gain_min 0.15~0.25도 검토
DELETE_OR_FOCUS: 삭제 후 문맥 보존 guard 필수
```

## 8. 최종 결론

직접 읽어본 결과, 현재 문제는 threshold 하나로 해결되지 않는다.

현재 시스템은 soft score로 전환되면서 정수 점수 노이즈 문제는 줄였지만, 이제 다음 병목은
**patch 품질 검증**이다.

따라서 다음 개발 순서는 다음이 맞다.

```text
1. patch validity 강화
2. ADD_DETAIL factuality guard 추가
3. action별 threshold 후보 재검토
4. 같은 v3 로그로 다시 virtual sweep
5. 그 다음 50건 확장
```
