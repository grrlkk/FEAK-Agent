# 중간정리 — reject_all 7건 검토 자료

> `experiments/results/mvp_stage_a_20_v2.jsonl`에서 decision=`reject_all`인 7건을 추출했다.
> gain_min이 과보수인지 판단하기 위한 육안 검토 자료이며, 판단은 보류한다.

- source: `experiments/results/mvp_stage_a_20_v2.jsonl`
- reject_all count: 7

## train_46

- weak_rubrics: `['content_1', 'content_3', 'organization_2']`
- decision reason: All non-STOP candidates violated hard constraints.

**Essay 앞부분 300자**

```text
인권이란 인간이 인간으로서 당연히 갖는 기본적 권리이다. 사회에는 다양한 차별이 존재한다. 최근 우리 사회에서 벌어지는 여러 차별 중 하나는 장애인 차별이다.  이러한 차별을 해결하기 위해서는 장애인이 우리와 다르다고 차별하지 않고 동등하게 대해야 한다. 차별이 존재하는 사회는 혼란스러워지다. 인권 침해 문제를 해결하기 위해서는 국가에선 미국 재활법 504조와 같이 장애인들을 위한 법을 만드는 노력을 해야 한다. 개인은 일상생활에서 노력한다면 더 나은 사회가 될 것이다.
```

**Before rubrics**

| rubric | score |
|---|---:|
| task_1 | 8.0 |
| content_1 | 7.0 |
| content_2 | 8.0 |
| content_3 | 7.0 |
| organization_1 | 8.0 |
| organization_2 | 7.0 |
| expression_1 | 8.0 |
| expression_2 | 9.0 |

**Candidates**

| idx | action | target_rubric | gain | drop | score | rejected | reject_reasons |
|---:|---|---|---:|---:|---:|---|---|
| 0 | ADD_DETAIL | content_2 | -1.0 | 1.0 | 0.134 | True | target_gain |
| 1 | DELETE_OR_FOCUS | organization_2 | 0.0 | 1.0 | 0.608 | True | target_gain |
| 2 | COMPRESS | content_3 | 0.0 | 1.0 | 0.619 | True | target_gain |
| 3 | RESTRUCTURE | organization_1 | -1.0 | 1.0 | 0.118 | True | target_gain |
| 4 | STYLE_REFINE | expression_1 | -1.0 | 2.0 | -0.136 | True | target_gain, non_target_drop |

## train_79

- weak_rubrics: `['content_3', 'task_1', 'content_1']`
- decision reason: All non-STOP candidates violated hard constraints.

**Essay 앞부분 300자**

```text
인권은 태어나자마자 모두가 가지고 있는 기본적인 인간의 권리를 뜻한다. 예를 들어 사람들은 모두가 인권이 있기 때문에 차별을 받으면 안 된다. 사회적 차별이 생겨나는 이유는 아이들이 어른들로부터 잘못된 생각이나 가치관이 심어져 차별하게 되는 경우도 있다. 예를 들어 장애인이어서 같이 놀지 말라든지, 취업을 못한다든지, 백인은 흑인과 놀 수 없다든지, 지역 차별, 힘쓰는 직업은 남자만 할 수 있고 여자는 할 수 없다며 차별하는 것 등이 있다. 차별 없는 세상, 인권이 보장되는 사회가 되기 위해서는 이러한 관념들을 타파해야 한다.
```

**Before rubrics**

| rubric | score |
|---|---:|
| task_1 | 9.0 |
| content_1 | 9.0 |
| content_2 | 9.0 |
| content_3 | 8.0 |
| organization_1 | 9.0 |
| organization_2 | 9.0 |
| expression_1 | 9.0 |
| expression_2 | 9.0 |

**Candidates**

| idx | action | target_rubric | gain | drop | score | rejected | reject_reasons |
|---:|---|---|---:|---:|---:|---|---|
| 0 | ADD_DETAIL | content_2 | 0.0 | 1.0 | 0.346 | True | target_gain |
| 1 | DELETE_OR_FOCUS | task_1 | 0.0 | 1.0 | 0.485 | True | target_gain |
| 2 | COMPRESS | content_3 | -1.0 | 2.0 | -0.214 | True | target_gain, non_target_drop, goal_preservation |
| 3 | RESTRUCTURE | organization_1 | -2.0 | 1.0 | -0.186 | True | target_gain |
| 4 | STYLE_REFINE | expression_1 | 0.0 | 1.0 | 0.314 | True | target_gain |

## train_81

- weak_rubrics: `['task_1', 'content_3', 'content_1']`
- decision reason: All non-STOP candidates violated hard constraints.

**Essay 앞부분 300자**

```text
공정함이란, ‘똑같은 출발선에서 출발하는 것’이다. 하지만 사회에서는 공정함의 뜻을 잘못 이해하고 있는 사람들이 많다. 장애인 시설을 만들지 않는 것과 몸이 불편한 학생을 다른 학생들과 동일한 기준을 적용하는 것은 공정함이 될 수 없다. 공정함은 그에 맞는 결과를 가져야 한다. 이런 차별을 없애기 위해서 장애인이나 노약자들을 위해서 대중교통을 늘리고, 계단이나 장애물 등을 줄여야 한다. 국가의 노력만이 아니라 우리 모두가 고정관념을 버려야 한다.
```

**Before rubrics**

| rubric | score |
|---|---:|
| task_1 | 1.0 |
| content_1 | 2.0 |
| content_2 | 2.0 |
| content_3 | 1.0 |
| organization_1 | 6.0 |
| organization_2 | 2.0 |
| expression_1 | 8.0 |
| expression_2 | 8.0 |

**Candidates**

| idx | action | target_rubric | gain | drop | score | rejected | reject_reasons |
|---:|---|---|---:|---:|---:|---|---|
| 0 | ADD_DETAIL | content_2 | 0.0 | 2.0 | 0.150 | True | target_gain, non_target_drop |
| 1 | DELETE_OR_FOCUS | task_1 | 0.0 | 2.0 | -0.072 | True | target_gain, non_target_drop |
| 2 | COMPRESS | content_1 | -1.0 | 1.0 | 0.369 | True | target_gain |
| 3 | RESTRUCTURE | organization_1 | 0.0 | 2.0 | 0.031 | True | target_gain, non_target_drop |
| 4 | STYLE_REFINE | expression_1 | -1.0 | 1.0 | 0.088 | True | target_gain |

## train_85

- weak_rubrics: `['content_3', 'task_1', 'content_1']`
- decision reason: All non-STOP candidates violated hard constraints.

**Essay 앞부분 300자**

```text
스마트 교복은 중국에서 만들어진 여러 기능이 있는 교복이다. 얼굴 인식 기능이 있어 교복이 바뀔 위험이 적고, 수업 시간에 태도를 인지해 알람이 울린다. 또 시간과 위치 기록 기능이 있어 허락 없이 학교 밖으로 나갈 때 부모님이나 선생님에게 알려준다. 교내 이동 경로도 파악할 수 있고 등교할 때는 등교 날짜와 시간이 짧은 영상과 함께 부모님에게 전송된다. 이 스마트교복에 대해 찬성과 반대로 나뉘어 논란이 되었다. 인권에 대한 침해가 우려된다는 목소리가 나오고 있다.
```

**Before rubrics**

| rubric | score |
|---|---:|
| task_1 | 3.0 |
| content_1 | 3.0 |
| content_2 | 5.0 |
| content_3 | 1.0 |
| organization_1 | 5.0 |
| organization_2 | 3.0 |
| expression_1 | 5.0 |
| expression_2 | 7.0 |

**Candidates**

| idx | action | target_rubric | gain | drop | score | rejected | reject_reasons |
|---:|---|---|---:|---:|---:|---|---|
| 0 | ADD_DETAIL | content_3 | 0.0 | 3.0 | 0.156 | True | target_gain, non_target_drop |
| 1 | DELETE_OR_FOCUS | task_1 | 0.0 | 0.0 | 0.852 | True | target_gain |
| 2 | COMPRESS | content_1 | 0.0 | 2.0 | 0.322 | True | target_gain, non_target_drop |
| 3 | RESTRUCTURE | organization_1 | -1.0 | 3.0 | -0.406 | True | target_gain, non_target_drop |
| 4 | STYLE_REFINE | expression_1 | 0.0 | 3.0 | -0.225 | True | target_gain, non_target_drop |

## train_91

- weak_rubrics: `['task_1', 'content_1', 'content_3']`
- decision reason: All non-STOP candidates violated hard constraints.

**Essay 앞부분 300자**

```text
현재 이주노동자가 많아지고 있는 상황이지만, 그들에 대한 차별이 여전히 문제가 되고 있다. 이주노동자는기본적인 권리를 누리지 못하는 경우가 많다. 깨끗한 작업 환경, 다쳤을 때의 보상, 적절한 월급을 받지 못하였다. 만약 이러한 차별이 개선되지 않는다면 우리나라 노동자들도 외국에 가서 적당한 환경을 제공받지 못할 것이다. 또한 외국인에 대한 고정관념과 차별의 문제가 더욱 심해질 테고, 이들의 생활 여건은 나아지지 못할 것이다.  모두가 소중하고 동등한 인권을 가지고 있다는 사실을 인지해야 한다.
```

**Before rubrics**

| rubric | score |
|---|---:|
| task_1 | 7.0 |
| content_1 | 7.0 |
| content_2 | 8.0 |
| content_3 | 7.0 |
| organization_1 | 9.0 |
| organization_2 | 7.0 |
| expression_1 | 9.0 |
| expression_2 | 8.0 |

**Candidates**

| idx | action | target_rubric | gain | drop | score | rejected | reject_reasons |
|---:|---|---|---:|---:|---:|---|---|
| 0 | ADD_DETAIL | content_2 | 0.0 | 1.0 | 0.382 | True | target_gain |
| 1 | DELETE_OR_FOCUS | task_1 | 0.0 | 1.0 | 0.426 | True | target_gain |
| 2 | COMPRESS | content_3 | 0.0 | 1.0 | 0.584 | True | target_gain |
| 3 | RESTRUCTURE | organization_1 | -2.0 | 1.0 | -0.118 | True | target_gain |
| 4 | STYLE_REFINE | expression_1 | 0.0 | 0.0 | 0.627 | True | target_gain |

## train_119

- weak_rubrics: `['content_3', 'task_1', 'content_1']`
- decision reason: All non-STOP candidates violated hard constraints.

**Essay 앞부분 300자**

```text
인체실험은 타당하지 않다. 많은 사람들의 희생이 있었기에 의료 기술이 지금처럼 발전할 수 있었던 것이다. 하지만 의료 기술은 생명은 살리는 기술인데, 많은 사람들을 희생 시키면서 발전했다는 것은 생명 윤리에 어긋난다. 인체 실험 지원자들은 대부분 가난한 사람들이다. 그 사람들은 돈을 벌기 위해서 목숨까지 걸고 실험에 참여한 사람들이다. 그렇기 때문에 인권 윤리에 어긋나는 것이다. 소수의 희생으로 많은 사람들의 목숨을 살릴 수 있지만 그 소수의 사람들도 모두 생명이라는 사실을 잊어서는 안 된다.
```

**Before rubrics**

| rubric | score |
|---|---:|
| task_1 | 5.0 |
| content_1 | 5.0 |
| content_2 | 5.0 |
| content_3 | 3.0 |
| organization_1 | 5.0 |
| organization_2 | 5.0 |
| expression_1 | 5.0 |
| expression_2 | 6.0 |

**Candidates**

| idx | action | target_rubric | gain | drop | score | rejected | reject_reasons |
|---:|---|---|---:|---:|---:|---|---|
| 0 | ADD_DETAIL | content_3 | -1.0 | 1.0 | 0.400 | True | target_gain |
| 1 | DELETE_OR_FOCUS | task_1 | -2.0 | 2.0 | -0.096 | True | target_gain, non_target_drop |
| 2 | COMPRESS | content_1 | -1.0 | 0.0 | 0.663 | True | target_gain |
| 3 | RESTRUCTURE | organization_1 | 0.0 | 1.0 | 0.433 | True | target_gain |
| 4 | STYLE_REFINE | expression_1 | 0.0 | 1.0 | 0.290 | True | target_gain |

## train_120

- weak_rubrics: `['task_1', 'content_3', 'organization_2']`
- decision reason: All non-STOP candidates violated hard constraints.

**Essay 앞부분 300자**

```text
오늘날 전세계에서 생산되는 모든 식량을 합하면, 전세계의 모든 인구를 충분히 먹여 살릴만큼 풍족하다. 그러나 개발도상국의 상황은 다르다. 돈이 많지 않은 개도국은 영양실조와 같은 문제가 여전히 많이 남아 있다. 식량은 상품으로 여겨져서는 안 된다. 인권이란 모든 인간이 인간의 존엄성을 갖고 인간답게 살 수 있는 권리이다. 먹을 식량이 없어 굶주린다면 이는 인간답게 살지 못하는 것이므로 최소한의 인권을 보장받지 못하는 것이다. 식량은 상품화 되어서는 안 되고 권리로 여겨져야 한다.
```

**Before rubrics**

| rubric | score |
|---|---:|
| task_1 | 2.0 |
| content_1 | 4.0 |
| content_2 | 4.0 |
| content_3 | 2.0 |
| organization_1 | 4.0 |
| organization_2 | 3.0 |
| expression_1 | 5.0 |
| expression_2 | 6.0 |

**Candidates**

| idx | action | target_rubric | gain | drop | score | rejected | reject_reasons |
|---:|---|---|---:|---:|---:|---|---|
| 0 | ADD_DETAIL | content_2 | 0.0 | 1.0 | 0.389 | True | target_gain |
| 1 | DELETE_OR_FOCUS | task_1 | 0.0 | 2.0 | 0.036 | True | target_gain, non_target_drop |
| 2 | COMPRESS | content_3 | 0.0 | 1.0 | 0.649 | True | target_gain |
| 3 | RESTRUCTURE | organization_2 | 0.0 | 0.0 | 0.950 | True | target_gain, no_effect |
| 4 | STYLE_REFINE | expression_1 | 0.0 | 0.0 | 0.564 | True | target_gain |

