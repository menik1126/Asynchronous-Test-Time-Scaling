from openai import AsyncOpenAI, APITimeoutError
import asyncio
import logging
import os

openai_key = os.environ.get("OPENAI_API_KEY", "")
openai_base = os.environ.get("OPENAI_BASE_URL") or None
openai_model = os.environ.get("OPENAI_MODEL", "gpt-5.2")

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
        pred_list=repr(answers), gt_label=repr(right_answer)
    )

    MAX_RETRIES = 3
    initial_delay = 1

    for i in range(MAX_RETRIES):
        try:
            logging.info(
                f"Attempt {i+1}/{MAX_RETRIES} to call OpenAI API for model: {openai_model}"
            )

            response = await client.chat.completions.create(
                model=openai_model,
                messages=[
                    {"role": "system", "content": "You are a mathematics expert."},
                    {"role": "user", "content": prompt_text},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )

            result = response.choices[0].message.content.strip()
            logging.info(f"API call successful after {i+1} attempt(s).")
            return result

        except APITimeoutError as e:
            if i < MAX_RETRIES - 1:
                delay = initial_delay * (2**i)
                logging.warning(f"Request timed out. Retrying in {delay} seconds...")
                await asyncio.sleep(delay)
            else:
                logging.error(
                    f"Request timed out after {MAX_RETRIES} attempts. Giving up."
                )
                raise e

        except Exception as e:
            logging.error(f"An unexpected error occurred during API call: {e}")
            raise e

    return ""
