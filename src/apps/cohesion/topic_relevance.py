import os
import re
import logging
from typing import Optional
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path

# 로거 설정
logger = logging.getLogger(__name__)

# 1. 경로 정의 및 .env 로드
ROOT_DIR = Path(__file__).resolve().parents[3]
env_path = ROOT_DIR / ".env"  

# 터미널 디버깅용 출력
print(f"\n--- [DEBUG] .env 경로: {env_path}")
print(f"--- [DEBUG] .env 존재 여부: {env_path.exists()}")

if env_path.exists():
    load_dotenv(env_path)
    print(f"--- [DEBUG] API 키 로드 상태: {'성공' if os.getenv('OPENAI_API_KEY') else '실패'}")
else:
    print("--- [ERROR] .env 파일을 찾을 수 없습니다.")

def get_topic_relevance_score(essay_text: str, topic: str) -> Optional[int]:
    """
    제공된 전문 프롬프트 규격에 따라 주제 적합성 점수 산출
    """
    if not topic or not essay_text:
        logger.warning("주제(topic) 또는 에세이 본문(essay_text)이 누락되었습니다.")
        return None

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    client = OpenAI(api_key=api_key)

    # 평가 프롬프트
    prompt = f"""
# Roles and Purpose
- Guide users on the given assignment and writing standards, and clearly provide the answer format that fits the evaluation criteria.

# Instructions
1. Evaluate the answers for the two prompts below (question/answer pair).
2. For each answer (essay, writing), assign a score based on the criteria provided.

## Evaluation Criteria
- 3 points: Clearly understands and addresses the essay prompt with appropriate writing.
- 2 points: Understands most of the content requested by the prompt, but some minor parts are incorrect or unclear.
- 1 point: Understands only a portion of the content requested by the prompt and many parts are incorrectly explained.
- 0 points: The answer is completely unrelated to the prompt or interprets it entirely incorrectly.

## Additional Guidelines
- Focus your evaluation mainly on how well the answer demonstrates understanding of the essay prompt.
- The answer should be output as a single digit integer according to the Output Format below. Do not include any extra explanations or other characters.
- If there is ambiguity in scoring, select the most conservative (lowest) score.

# Input Data
- Topic (Prompt): {topic}
- Essay (Answer): {essay_text}

# Output Format
- Output only the evaluation score as a single digit integer. (Example: 2)
- Do not include any additional explanation or characters.

# Completion Conditions
- Once you have accurately output the score in the requested format (a single digit integer), the task is complete.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o", 
            messages=[
                {"role": "system", "content": "You are a professional essay evaluator."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=5
        )
        
        score_str = response.choices[0].message.content.strip()
        score_match = re.search(r'[0-3]', score_str)
        
        if score_match:
            return int(score_match.group())
        return None
        
    except Exception as e:
        print(f"--- [API ERROR] GPT 호출 실패: {e}")
        return None
