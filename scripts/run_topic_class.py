import os
from pathlib import Path
import pandas as pd
import torch
import warnings
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
import transformers

# 1. 환경 설정 및 경고 차단
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")
warnings.filterwarnings("ignore")
transformers.logging.set_verbosity_error()

# 2. 지시문 전문
REFERENCE_PROMPT = """
카피레프트(Copyleft)는 창작물을 누구나 자유롭게 사용하고 공유할 수 있도록 하자는 개방형 저작권 운동입니다. 
사회적 자산을 바탕으로 만들어진 창작물에 대해 저작권 보호를 제한하자는 주장에 대해 찬성 또는 반대 입장을 선택하고, 그 이유를 논리적으로 서술하시오.
"""

# 3. 모델 로드
model_id = "LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct"
print(f"🚀 EXAONE-3.0 '단순 답변 모드'로 재설정 중...")

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16),
    device_map="auto",
    trust_remote_code=True
)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

# 4. 초간결 판별 함수
def check_topic_relevance(ref_prompt, student_essay):
    # 설명 생략, 예시 생략, 오직 본질만 질문
    prompt = (
        f"[|system|]제시된 지시문과 학생 에세이의 소재가 같으면 PASS, 다르면 FAIL로만 답하세요. "
        f"찬성/반대 입장은 전혀 상관없습니다. 소재가 '저작권'이면 무조건 PASS입니다.[|user|]"
        f"지시문: {ref_prompt}\n\n학생 에세이: {student_essay}\n\n결과:[|assistant|]"
    )

    outputs = pipe(
        prompt,
        max_new_tokens=5, # 아주 짧게 제한하여 딴소리 원천 차단
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # 모델의 순수 출력값 추출
    ans = outputs[0]["generated_text"].split("[|assistant|]")[-1].strip().upper()
    return ans

# 5. 데이터 로드 및 분석
PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = os.getenv("FEAK_INPUT_FILE", str(PROJECT_ROOT / "data" / "UKTA_1128_total_result.xlsx"))
df = pd.read_excel(INPUT_FILE)

print(f"\n📊 분석 시작 및 디버깅 (처음 5개 결과 노출)...")
results = []

for i, row in tqdm(df.iterrows(), total=len(df), desc="Analyzing"):
    essay = str(row.get('essay', row.get('Essay', "")))
    
    # 판정 실행
    raw_decision = check_topic_relevance(REFERENCE_PROMPT, essay)
    
    # 처음 5개에 대해서만 모델이 실제로 뭐라고 했는지 출력 (확인용)
    if i < 5:
        print(f"\n[Index {i} 실시간 결과]: {raw_decision}")

    # PASS가 포함되어 있으면 무조건 NORMAL
    decision = "PASS" if "PASS" in raw_decision else "FAIL"
    
    results.append({
        "Note": "NORMAL" if decision == "PASS" else "OFF-TOPIC",
        "Raw_AI": raw_decision
    })

# 6. 결과 병합 및 통계
df['Topic_Note'] = [r['Note'] for r in results]
df['Raw_AI_Response'] = [r['Raw_AI'] for r in results]

total = len(df)
normal = len(df[df['Topic_Note'] == 'NORMAL'])
print("\n" + "="*80)
print(f"🎯 최종 분석 결과 요약")
print(f"✅ NORMAL: {normal}건 ({(normal/total)*100:.1f}%)")
print(f"⚠️ OFF-TOPIC: {total-normal}건 ({(1 - normal/total)*100:.1f}%)")
print("="*80)

# 7. 오답 샘플 확인
print("\n🔍 상위 10개 상세 판별 결과 (에세이 앞부분 포함):")
for idx, row in df.head(10).iterrows():
    print(f"[{idx}] 결과: {row['Topic_Note']} | AI응답: {row['Raw_AI_Response']} | 내용: {row['essay'][:40]}...")
