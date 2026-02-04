from openai import AsyncOpenAI, APITimeoutError
import asyncio
import logging
import os

# # 配置日志，便于调试
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

openai_key = "sk-0xGYTmwLliZxud3UlCOPpyW4V7CV1dQ8SHb1wiNMV0uZJCCM"
openai_base = 'https://xiaoai.plus/v1'
openai_model = "gpt-4o-mini"


# Note: It's more efficient to create the client once and pass it,
# but we'll stick to the original code structure for this modification.
# client = AsyncOpenAI(api_key=openai_key, base_url=openai_base)


anyone_check_prompt = """
You are a mathematics expert with deep understanding of mathematical expressions and notation.

Your task is to determine whether **any** of the predicted answers is mathematically equivalent to the ground-truth answer. Equivalence can be exact (e.g., "1/2" vs. "0.5") or approximate (e.g., "√2" vs. "1.414").

### Instructions:
- Consider semantic equivalence, not just string similarity.
- Accept common variations in notation: fractions, decimals, LaTeX-style formats (e.g., '\\frac{{5}}{{13}}'), radicals (e.g., 'sqrt(2)'), etc.
- If **any** prediction matches, respond with: `Match`
- If **none** match, respond with: `Mismatch`
- Your response **must be only** "Match" or "Mismatch".

### Examples:
Predicted: ['1/2', '0.5', '0.25']  
Ground-truth: '0.5'  
→ Match

Predicted: ['2/3', '0.66', '3/5']  
Ground-truth: '0.666...'  
→ Match

Predicted: ['117', '84', '90']  
Ground-truth: '084'
→ Match

Predicted: ['3/4', '0.8', '0.9']  
Ground-truth: '3/7'  
→ Mismatch

### Now evaluate:
Predicted: {pred_list}  
Ground-truth: {gt_label}  
→
"""


async def anyone_check(right_answer, answers):
    """
    Checks if the correct_answer can be extracted from any of the outputs
    with a retry mechanism for API calls that time out.
    """
    client = AsyncOpenAI(
        api_key=openai_key,
        base_url=openai_base,
    )
    
    temperature = 0.0
    max_tokens = 10

    prompt_text = anyone_check_prompt.format(
        pred_list=repr(answers),
        gt_label=repr(right_answer)
    )

    # 预定义重试参数
    MAX_RETRIES = 3 # 最大重试次数
    initial_delay = 1 # 初始等待秒数

    for i in range(MAX_RETRIES):
        try:
            logging.info(f"Attempt {i+1}/{MAX_RETRIES} to call OpenAI API for model: {openai_model}")
            
            response = await client.chat.completions.create(
                model=openai_model,
                messages=[
                    {"role": "system", "content": "You are a mathematics expert."},
                    {"role": "user", "content": prompt_text},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            
            # 如果 API 调用成功，获取结果并返回
            result = response.choices[0].message.content.strip()
            logging.info(f"API call successful after {i+1} attempt(s).")
            return result

        except APITimeoutError as e:
            # 捕获超时异常
            if i < MAX_RETRIES - 1:
                delay = initial_delay * (2 ** i) # 指数退避，例如 1s, 2s, 4s
                logging.warning(f"Request timed out. Retrying in {delay} seconds...")
                await asyncio.sleep(delay)
            else:
                logging.error(f"Request timed out after {MAX_RETRIES} attempts. Giving up.")
                raise e # 最后一次重试失败，重新抛出异常
        
        except Exception as e:
            # 捕获其他可能的异常，例如网络错误
            logging.error(f"An unexpected error occurred during API call: {e}")
            raise e
    
    # 在所有重试都失败的情况下，确保函数不会意外返回
    return ""
