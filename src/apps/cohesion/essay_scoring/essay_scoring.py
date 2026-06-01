# -*- coding: utf-8 -*-

import json
import pandas as pd
from kobert_transformers import get_kobert_model, get_tokenizer
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import collections
import torch.nn.functional as F

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ------------------------ 경로/스케일러 공통 ------------------------
BASE = Path(__file__).parent
SCALER_CSV = BASE / "features" / "scaler_notlabel.csv"

def _read_scaler(path: Path) -> pd.DataFrame:
    """scaler.csv (feature,mean,scale) 로드 + 안전장치"""
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="cp949")
    # 컬럼 검증
    required = {"feature", "mean", "scale"}
    if not required.issubset(set(df.columns)):
        raise RuntimeError(f"스케일러 파일 포맷 오류: {path} (필수 컬럼: {required})")
    # 숫자화 + 0-scale 방지
    df["mean"] = pd.to_numeric(df["mean"], errors="coerce").fillna(0.0)
    df["scale"] = pd.to_numeric(df["scale"], errors="coerce").replace(0, 1.0).fillna(1.0)
    return df

# ------------------- 개선 GRU + LN + UKTA Attention -------------------
class GRUScoreModuleWithLNUKTAAttention(nn.Module):
    def __init__(self, output_dim, hidden_dim, ukt_a_dim=29, dropout=0.5):
        super().__init__()
        self.gru = nn.GRU(
            input_size=768,
            hidden_size=hidden_dim,
            num_layers=2,
            dropout=dropout,
            batch_first=True,
            bidirectional=True,
        )
        self.ukt_a_fc = nn.Linear(ukt_a_dim, hidden_dim)
        self.attention_weights = nn.Linear(hidden_dim * 2, ukt_a_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 3, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, ukt_a):
        # x: [B, T, 768], ukt_a: [B, ukt_a_dim]
        x, _ = self.gru(x)                   # [B, T, 2H]
        x = torch.mean(x, dim=1)             # [B, 2H]
        x = self.layer_norm(x)

        attn_scores = self.attention_weights(x)      # [B, ukt_a_dim]
        attn = F.softmax(attn_scores, dim=-1)        # [B, ukt_a_dim]

        weighted_ukt_a = ukt_a * attn                # [B, ukt_a_dim]
        ukt_a_feat = self.ukt_a_fc(weighted_ukt_a)   # [B, H]

        combined = torch.cat((x, ukt_a_feat), dim=1) # [B, 3H]
        combined = self.dropout(combined)
        out = self.fc(combined)
        out = self.sigmoid(out)
        return out, attn

# ------------------------- voc_grades → ratio --------------------------
def extract_grade_ratios(voc_grades):
    """
    voc_grades: process()가 만드는 리스트 형태
      [("2", [ {..., "cnt": int}, ... ]), ("3", [...]), ("4", [...]), ("-1", [...]), ...]
    반환: {"grade_2_ratio": float, "grade_3_ratio": float, "grade_4_ratio": float, "grade_m1_ratio": float}
    """
    ratios = dict(grade_2_ratio=0.0, grade_3_ratio=0.0, grade_4_ratio=0.0, grade_m1_ratio=0.0)
    if not isinstance(voc_grades, list):
        return ratios

    buckets = {"2": 0, "3": 0, "4": 0, "-1": 0}
    total = 0

    for entry in voc_grades:
        # entry는 ("등급", [아이템들]) 혹은 ["등급", [아이템들]]
        if not (isinstance(entry, (list, tuple)) and len(entry) == 2):
            continue
        grade_key, items = entry
        if not isinstance(items, list):
            continue

        c = 0
        for it in items:
            if isinstance(it, dict):
                c += int(it.get("cnt", 1))
            else:
                c += 1
        total += c
        if grade_key in buckets:
            buckets[grade_key] += c

    if total > 0:
        ratios["grade_2_ratio"] = buckets["2"]  / total
        ratios["grade_3_ratio"] = buckets["3"]  / total
        ratios["grade_4_ratio"] = buckets["4"]  / total
        ratios["grade_m1_ratio"] = buckets["-1"] / total

    return ratios

# ----------------------------- 스코어링 ------------------------------
def scoring(bert_model, gru_model, extracted_features, tokenizer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) 문장 리스트
    sentences = [sent["text"]["content"] for sent in extracted_features["morpheme"]["sentences"]]

    # 2) 자질 모으기(원천 dict → flat)  —— keys 유지
    #    * dict 섹션만 펼치고, 숫자형만 채택
    sample_essay_features, feature_list = [], []
    keys = [
        "ttr", "similarity", "adjacency", "basic_count", "basic_density",
        "NDW", "readability", "sentenceLvl", "sentenceLvlRep", "basic_level",
        # 'voc_grades'는 dict가 아니라 여기선 제외(아래에서 따로 ratio 계산하여 병합)
    ]
    for key in keys:
        payload = extracted_features.get(key, {})
        if isinstance(payload, dict):
            for k, v in payload.items():
                if isinstance(v, (int, float)):
                    feature_list.append(k)
                    sample_essay_features.append(v)

    # voc_grades에서 ratio 4개를 계산해 합치기
    vg_ratios = extract_grade_ratios(extracted_features.get("voc_grades"))
    for k, v in vg_ratios.items():
        feature_list.append(k)
        sample_essay_features.append(v)

    # 3) 스케일러 로드 & 스케일러 기준 정렬(고정 차원 보장)
    scaler = _read_scaler(SCALER_CSV)
    scaler_feats = scaler["feature"].tolist()
    src_map = {k: v for k, v in zip(feature_list, sample_essay_features)}

    # === 추가: 스케일러에 있지만 입력에 없는 자질 찾기 ===
    missing = [feat for feat in scaler_feats if feat not in src_map]
    if missing:
        logger.warning(
            "[essay_score] 누락 자질 %d개(스케일러에는 있으나 입력에서 없음). 예시: %s",
            len(missing), missing[:10]
        )
        logger.debug("[essay_score] 전체 누락 자질: %s", missing)
    else:
        logger.info("[essay_score] 누락 자질 없음. (입력 자질이 스케일러를 모두 커버)")
    # === 추가 끝 ===

    # 4) 표준화 — 스케일러 순서대로 값 채우기(없으면 0.0 임퓨트)
    ordered_scaled = []
    for feat in scaler_feats:
        row = scaler.loc[scaler["feature"] == feat].iloc[0]
        mean, scale = float(row["mean"]), float(row["scale"] or 1.0)
        val = src_map.get(feat, row["mean"])
        ordered_scaled.append((val - mean) / (scale if scale != 0 else 1.0))

    scaled_features = torch.tensor([ordered_scaled], dtype=torch.float32, device=device)

    assert len(scaler_feats) == scaled_features.shape[1], \
        f"ukt_a_dim mismatch: scaler={len(scaler_feats)} vs tensor={scaled_features.shape[1]}"

    feature_list_for_attn = np.array(scaler_feats)  # 주의: attn과 동일 차원

    # 5) KoBERT 임베딩 ([CLS])
    max_length = 256  # 400→메모리 이슈로 256
    inputs = tokenizer.batch_encode_plus(
        sentences, max_length=max_length, padding="max_length", truncation=True
    )
    input_ids = torch.tensor(inputs["input_ids"]).to(bert_model.device)
    attention_mask = torch.tensor(inputs["attention_mask"]).to(bert_model.device)
    out = bert_model(input_ids=input_ids, attention_mask=attention_mask)
    embedded_features = out[0].detach().cpu()[:, 0, :]  # [num_sents, 768]
    embedded_features = torch.tensor(np.array(embedded_features), dtype=torch.float32)

    # 6) 시퀀스 패딩(선패딩) 길이=128
    max_len = 128
    if embedded_features.size(0) > max_len:
        embedded_features = embedded_features[-max_len:]  # 뒤 128개 유지
    else:
        pad = torch.zeros((max_len - embedded_features.size(0), embedded_features.size(1)))
        embedded_features = torch.cat((pad, embedded_features), dim=0)
    embedded_features = embedded_features.unsqueeze(0).to(device)  # [1, T, 768]

    # 7) 모델 추론
    gru_model.eval()
    with torch.no_grad():
        logits, attn = gru_model(embedded_features, scaled_features)
    output = np.rint(logits[0].cpu().numpy() * 3).astype(int)  # 0~1 → 0~3 라운딩
    attn = attn[0].cpu().numpy()
    top_k_features = get_topK_features(attn, feature_list_for_attn)
    return output, top_k_features

def get_topK_features(attention, feature_list, k=10):
    idx = np.argsort(attention)[::-1]
    return feature_list[idx[:k]]

# --------------------------- 로딩/외부 API ---------------------------
def load_essay_model(device):
    bert_model = get_kobert_model().to(device)
    tokenizer = get_tokenizer()

    # 스케일러에서 차원 동기화
    scaler = _read_scaler(SCALER_CSV)
    ukt_a_dim = len(scaler["feature"])  # 스케일러 feature 개수와 일치

    gru_model = GRUScoreModuleWithLNUKTAAttention(
        output_dim=11,         # 루브릭 11개
        hidden_dim=256,        # 학습 시 사용한 값에 맞춤
        ukt_a_dim=ukt_a_dim,   # 스케일러 feature 개수와 일치
        dropout=0.5,
    ).to(device)

    weight_path = BASE / "model" / "not_topic_model.pth"  # 프롬프트 사용 안한 최신 모델
    state = torch.load(weight_path, map_location=device)
    gru_model.load_state_dict(state)
    gru_model.eval()
    return bert_model, gru_model, tokenizer

# ------------------------- JSON 직렬화 유틸 --------------------------
def _to_jsonable(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.generic,)):
        return x.item()
    return x

# ===================== [추가] 원문 + 29자질 추출 헬퍼 =====================
def _collect_feat29_and_text(extracted_features):
    """
    반환:
      - feat29_raw: {feature_name: float}  # 스케일러(feature/scaler_notlabel.csv) 순서 기준으로 채움
      - essay_text: str                    # 문장들을 합쳐 원문
    """
    # 1) 원문 텍스트
    sentences = [s["text"]["content"] for s in extracted_features["morpheme"]["sentences"]]
    essay_text = "\n".join(sentences)

    # 2) 자질 수집(숫자형만)
    keys = [
        "ttr", "similarity", "adjacency", "basic_count", "basic_density",
        "NDW", "readability", "sentenceLvl", "sentenceLvlRep", "basic_level",
    ]
    raw_map = {}
    for key in keys:
        payload = extracted_features.get(key, {})
        if isinstance(payload, dict):
            for k, v in payload.items():
                if isinstance(v, (int, float)):
                    raw_map[k] = float(v)

    # 3) voc_grades → 4개 비율 추가
    vg = extract_grade_ratios(extracted_features.get("voc_grades"))
    for k, v in vg.items():
        raw_map[k] = float(v)

    # 4) 스케일러 기준으로 순서/누락 처리(없으면 스케일러 평균으로 임퓨트)
    scaler = _read_scaler(SCALER_CSV)
    feat29_raw = {}
    for feat in scaler["feature"].tolist():
        if feat in raw_map:
            feat29_raw[feat] = raw_map[feat]
        else:
            mean_val = float(scaler.loc[scaler["feature"] == feat, "mean"].iloc[0])
            feat29_raw[feat] = mean_val

    return feat29_raw, essay_text

def score_results_with_feats(extracted_features, bert_model, gru_model, tokenizer):
    """
    기존 score_results와 동일 + feat29, text까지 함께 반환
    """
    output, top_k_features = scoring(bert_model, gru_model, extracted_features, tokenizer)

    rubric = [
        "grammar","vocabulary","sentence_expression","intra_paragraph_structure",
        "inter_paragraph_structure","structural_consistency","length",
        "topic_clarity","originality","prompt_comprehension","narrative",
    ]
    result = {rubric[i]: int(output[i]) for i in range(11)}
    result["top_k_features"] = _to_jsonable(top_k_features)

    # 추가: 29자질 원시값 + 원문
    feat29_raw, essay_text = _collect_feat29_and_text(extracted_features)
    result["feat29"] = feat29_raw
    result["text"] = essay_text
    return result

# ----------------------------- 외부 호출 -----------------------------
def score_results(extracted_features, bert_model, gru_model, tokenizer):
    """
    기준 코드 스키마와 동일하게 직렬화:
      - 루브릭 11개: int
      - top_k_features: list (np.ndarray → list 변환)
    """
    output, top_k_features = scoring(bert_model, gru_model, extracted_features, tokenizer)
    rubric = [
        "grammar","vocabulary","sentence_expression","intra_paragraph_structure",
        "inter_paragraph_structure","structural_consistency","length",
        "topic_clarity","originality","prompt_comprehension","narrative",
    ]
    result = {}
    for i in range(11):
        result[rubric[i]] = int(output[i])

    # 핵심: ndarray -> list (기준 코드 호환)
    result["top_k_features"] = _to_jsonable(top_k_features)

    # 혹시 다른 값에 numpy 타입 섞였을 경우 대비(선택적)
    for k, v in list(result.items()):
        result[k] = _to_jsonable(v)

    return result

# ----------------------------- 단독 테스트 -----------------------------
if __name__ == "__main__":
    # 예제 입력 로드
    with open("./example/ESSAY_51802.json", "r", encoding="utf-8-sig") as f:
        extracted_features = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert_model, gru_model, tokenizer = load_essay_model(device)

    # 기존 결과
    result = score_results(extracted_features, bert_model, gru_model, tokenizer)
    rubric_ko = [
        "문법","단어","문장 표현","문단 내 구조의 적절성","문단 간 구조의 적절성",
        "구조의 일관성","분량","주제의 명료성","참신성","프롬프트 독해력","서술력",
    ]
    print({rk: result[k] for rk, k in zip(rubric_ko, [
        "grammar","vocabulary","sentence_expression","intra_paragraph_structure",
        "inter_paragraph_structure","structural_consistency","length",
        "topic_clarity","originality","prompt_comprehension","narrative",
    ])})
    print("top_k_features:", result["top_k_features"])

    # 추가된 결과(원문+feat29 포함)
    result2 = score_results_with_feats(extracted_features, bert_model, gru_model, tokenizer)
    print("text (head):", result2["text"][:120].replace("\n"," ") + "...")
    print("feat29 keys:", list(result2["feat29"].keys())[:10], " ... (total:", len(result2["feat29"]), ")")
