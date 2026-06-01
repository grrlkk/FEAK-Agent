import os
import sys
from pathlib import Path
import pandas as pd
from openai import OpenAI  # 클래스 임포트 방식 변경
from tqdm import tqdm
import json
import time
from dotenv import load_dotenv

# 1. .env 파일 로드
PROJECT_ROOT = Path(__file__).resolve().parents[1]
env_path = os.getenv("FEAK_ENV_FILE", str(PROJECT_ROOT / ".env"))
load_dotenv(dotenv_path=env_path)

# 2. 클라이언트 객체 생성 (최신 방식)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

if not client.api_key:
    print(f"❌ API Key not found. Please check your .env file at: {env_path}")
    sys.exit()

# --- [Configuration] ---
INPUT_FILE = os.getenv("FEAK_INPUT_FILE", str(PROJECT_ROOT / "data" / "UKTA_1128_total_result.xlsx"))
OUTPUT_FILE = os.getenv("FEAK_OUTPUT_FILE", str(PROJECT_ROOT / "experiments" / "results" / "UKTA_1128_GPT_rubric.xlsx"))
BASE_SCORE = 7

SCORE_5X = ["topic_clarity", "narrative", "originality", "intra_paragraph_structure", "inter_paragraph_structure"]
SCORE_2X = ["grammar", "vocabulary", "sentence_expression"]

def get_gpt_score(essay_text):
    prompt = f"""
    You are an expert Korean essay evaluator. Grade the following [Essay] based strictly on the [Rubric] provided.
    Assign a score of 3(Excellent), 2(Good), 1(Fair), or 0(Poor) for each criterion.

    [Rubric Criteria]
    1. topic_clarity: Clarity of the main argument and relevance to the overall theme.
    2. narrative: Validity and diversity of supporting evidence for the argument.
    3. originality: Novelty of ideas/perspectives and logical consistency of the content.
    4. intra_paragraph_structure: Connectivity between the topic sentence and supporting sentences.
    5. inter_paragraph_structure: Clear distinction between Intro/Body/Conclusion and balance of paragraph length.
    6. grammar: Accuracy of grammar usage and frequency of errors.
    7. vocabulary: Diversity of vocabulary and appropriateness for the context.
    8. sentence_expression: Diversity of sentence structures and appropriateness of sentence lengths.

    [Essay]
    {essay_text}

    [Output Format]
    Respond ONLY in the following JSON format. Do not include any conversational text.
    {{
        "topic_clarity": 0, 
        "narrative": 0,
        "originality": 0,
        "intra_paragraph_structure": 0,
        "inter_paragraph_structure": 0,
        "grammar": 0,
        "vocabulary": 0,
        "sentence_expression": 0
    }}
    """
    
    try:
        # 최신 버전 호출 방식: client.chat.completions.create
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a professional essay grader specializing in Korean academic writing."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            response_format={ "type": "json_object" } # JSON 출력 강제 설정 (최신 기능)
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"\nAPI Error: {e}")
        return None

# --- [Execution] ---
if not os.path.exists(INPUT_FILE):
    print(f"❌ Input file not found: {INPUT_FILE}")
    sys.exit()

df = pd.read_excel(INPUT_FILE)
gpt_results = []

print(f"🚀 Starting GPT Grading (Total: {len(df)} essays) via GPT-4o")

for i, row in tqdm(df.iterrows(), total=len(df)):
    essay = str(row.get('essay', ""))
    summary_row = {"Essay": essay}
    
    if len(essay.strip()) < 10:
        for key in SCORE_5X + SCORE_2X:
            summary_row[key] = 0
        summary_row["Total_Score"] = BASE_SCORE
        summary_row["Note"] = "Too short"
    else:
        scores = get_gpt_score(essay)
        
        if scores:
            current_sum = 0
            for key in SCORE_5X + SCORE_2X:
                # API 응답에서 해당 키의 값을 가져와 정수로 변환 (기본값 0)
                raw_val = int(scores.get(key, 0))
                multiplier = 5 if key in SCORE_5X else 2
                final_val = raw_val * multiplier
                summary_row[key] = final_val
                current_sum += final_val
            
            summary_row["Total_Score"] = current_sum + BASE_SCORE
        else:
            for key in SCORE_5X + SCORE_2X:
                summary_row[key] = 0
            summary_row["Total_Score"] = BASE_SCORE
            summary_row["Note"] = "API Error"
    
    gpt_results.append(summary_row)
    time.sleep(0.1)

# 결과 저장
output_columns = ["Essay"] + SCORE_5X + SCORE_2X + ["Total_Score"]
pd.DataFrame(gpt_results)[output_columns].to_excel(OUTPUT_FILE, index=False)

print(f"\n✨ Success! Grading saved to: {OUTPUT_FILE}")
