import os
import sys
from pathlib import Path
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm

# 1. 2번 GPU 사용 설정
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")

# 2. 프로젝트 경로 설정
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# 제공된 essay_scoring.py 모듈 로드
try:
    from apps.cohesion.essay_scoring.essay_scoring import load_essay_model, score_results
except ImportError as e:
    print(f"❌ 모듈 로드 에러: {e}")
    print("현재 경로에서 'src/apps' 폴더가 있는지 확인해주세요.")
    sys.exit()

# --- [설정] 배점 및 그룹 설정 ---
# 15점 만점 항목 (AI 원본 점수 0~3점 * 5)
SCORE_5X = ["topic_clarity", "narrative", "originality", "intra_paragraph_structure", "inter_paragraph_structure"]
# 6점 만점 항목 (AI 원본 점수 0~3점 * 2)
SCORE_2X = ["grammar", "vocabulary", "sentence_expression"]

# 최종 결과에 담을 순서 (내용 -> 조직 -> 표현)
ORDERED_KEYS = [
    "topic_clarity", "narrative", "originality",              # 내용
    "intra_paragraph_structure", "inter_paragraph_structure", # 조직
    "grammar", "vocabulary", "sentence_expression"             # 표현
]

# 파일 경로
INPUT_FILE = os.getenv("FEAK_INPUT_FILE", str(PROJECT_ROOT / "data" / "UKTA_1128_total_result.xlsx"))
OUTPUT_FILE = os.getenv("FEAK_OUTPUT_FILE", str(PROJECT_ROOT / "experiments" / "results" / "UKTA_1128_rubric.xlsx"))

# --- 1. 모델 로딩 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 AI 모델 로딩 중... (사용 장치: {device})")
bert_model, gru_model, tokenizer = load_essay_model(device)
print("✅ 모델 로딩 완료!")

def map_row_to_features(row):
    """536개 열에서 접두사를 제거하고 모든 자질 그룹을 안전하게 매핑"""
    raw_text = str(row.get('essay', ""))
    sentences = [{"text": {"content": s.strip()}} for s in raw_text.split('\n') if s.strip()]
    
    input_data = {
        "morpheme": {"sentences": sentences},
        "voc_grades": [
            ("2", [{"cnt": row.get('grade_2', 0)}]),
            ("3", [{"cnt": row.get('grade_3', 0)}]),
            ("4", [{"cnt": row.get('grade_4', 0)}]),
            ("-1", [{"cnt": row.get('grade_m1', 0)}])
        ]
    }
    
    # 누락되었던 basic_level을 포함한 모든 접두사 리스트
    target_groups = ["ttr", "similarity", "adjacency", "basic_count", "basic_level", "NDW", "readability", "sentenceLvl"]
    
    for group in target_groups:
        group_dict = {}
        prefix = f"{group}_"
        for col in row.index:
            if str(col).startswith(prefix):
                clean_name = str(col)[len(prefix):]
                val = row[col]
                # NaN 값인 경우 0으로 치환하여 에러 방지
                group_dict[clean_name] = 0 if pd.isna(val) else val
        input_data[group] = group_dict
        
    return input_data

# --- 2. 데이터 로드 및 채점 시작 ---
if not os.path.exists(INPUT_FILE):
    print(f"❌ 파일을 찾을 수 없습니다: {INPUT_FILE}")
    sys.exit()

df = pd.read_excel(INPUT_FILE)
final_summary = []

print(f"📊 총 {len(df)}건 채점 시작 (기본 점수 7점 반영)")

for i, row in tqdm(df.iterrows(), total=len(df)):
    input_features = map_row_to_features(row)
    summary_row = {
        "essay": row.get('essay', ""),
        "source_file": row.get('source_file', "")
    }
    
    # 모든 글에 부여되는 기본 점수
    BASE_SCORE = 7
    
    try:
        # 모델 추론
        raw_scores = score_results(input_features, bert_model, gru_model, tokenizer)
        ai_sum = 0
        
        for key in ORDERED_KEYS:
            val = raw_scores.get(key, 0)
            
            # 모델 결과가 NaN인 경우 안전하게 0점 처리
            if pd.isna(val) or np.isnan(val):
                clean_val = 0
            else:
                clean_val = int(val)
                
            # 배점 환산 (x5 또는 x2)
            final_val = clean_val * (5 if key in SCORE_5X else 2)
            summary_row[f"AI_{key}"] = final_val
            ai_sum += final_val
            
        # 최종 총점 = AI 항목 합계 + 기본 점수 7점
        summary_row["AI_Total_Score"] = ai_sum + BASE_SCORE
        
    except Exception as e:
        # 에세이가 너무 짧거나 모델 에러 발생 시
        for key in ORDERED_KEYS:
            summary_row[f"AI_{key}"] = 0
        summary_row["AI_Total_Score"] = BASE_SCORE  # 에러 시에도 기본점수 7점 부여
        summary_row["Note"] = "Too short or Error (Default 7pts applied)"

    final_summary.append(summary_row)

# --- 3. 엑셀 저장 ---
df_final = pd.DataFrame(final_summary)
df_final.to_excel(OUTPUT_FILE, index=False)
print(f"\n✨ 채점 완료! 결과 파일: {OUTPUT_FILE}")
