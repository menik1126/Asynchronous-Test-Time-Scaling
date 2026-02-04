import sglang as sgl
import openai
import os

openai_key = ""
openai_base = 'https://xiaoai.plus/v1'
openai_model = "gpt-4o-mini"

client = openai.OpenAI(api_key=openai_key, base_url=openai_base)

class agent():
    def __init__(self, name: str, gpu: int, tp: int = 1, dp: int = 1):
        self.name = name
        self.model_args = {
            "model_path": name,
            "tokenizer_path": name,
            "tokenizer_mode": "auto",
            "load_format": "auto",
            "trust_remote_code": True,
            "dtype": "bfloat16",
            "kv_cache_dtype": "auto",
            "device": "cuda",
            "mem_fraction_static": 0.9,
            "tp_size": tp,
            "dp_size": dp,
            "chunked_prefill_size": -1,
            "base_gpu_id": gpu,
        }
        self.model = sgl.Engine(**self.model_args)

system_prompt = (
    "You are a highly accurate math reasoning assistant.\n\n"
    "Please carefully read the following long response and extract **only the final explicit numerical answer** "
    "(a number like 3.14, 42, or 0.875). The sentence in the beginning is the problem and the model may generate useless output after it generates correct answer. You should carefully analysis the output and find the answer. If there are multiple numbers, choose the one that is "
    "**clearly marked as the final answer or result** (e.g., after 'therefore', 'we find', 'thus', "
    "'the final answer is', etc.).\n\n"
    "Only return the number itself. If there is no such number, return 'invalid'."
)

my_prompt = "You are an assistant skilled at information extraction. Please extract the explicit numerical answer from the response. Output only the number—no units, punctuation, or explanations. If no answers, return 'invalid'"

def extract_answer(answers):
    key = openai_key
    base = openai_base
    model = openai_model
    
    temperature = 0.0
    max_tokens = 50

    result = []
    for answer in answers:
        # print("============")
        # print(answer["text"])
        response= client.chat.completions.create(
            model=openai_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": answer["text"]},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        result.append(response.choices[0].message.content)
    return result

majority_check_prompt = """
You are a mathematics expert with deep understanding of mathematical expressions and notation. Your task is to determine whether the result of a majority vote from a list of predicted answers semantically matches the ground-truth answer, even if they are written in different formats (e.g., '5/13' vs '\\\\frac{{5}}{{13}}'; '0.5' vs '1/2').

Instructions:
1. Identify the label(s) with the highest frequency (majority). There may be a tie.
2. If **any** of the tied majority labels is mathematically equivalent to the ground-truth answer, output "Match".
3. Otherwise, output "Mismatch".

Only consider semantic mathematical equivalence. For example, '1/2' and '0.5' are equal, and '\\\\sqrt{{2}}' and '1.414' are approximately equal. Use mathematical reasoning, not string comparison.

Respond **only** with "Match" or "Mismatch".

Here are some examples:

Example 1:
Predicted: ['1/2', '0.5', '1/2']
Ground-truth: '0.5'
→ Match

Example 2:
Predicted: ['5/13', '5/13', '\\\\frac{{5}}{{13}}', '5/13']
Ground-truth: '\\\\frac{{5}}{{13}}'
→ Match

Example 3:
Predicted: ['2/3', '0.66', '2/3']
Ground-truth: '0.666...'
→ Match

Example 4:
Predicted: ['3/4', '0.8', '0.8']
Ground-truth: '3/4'
→ Mismatch

Example 5:
Predicted: ['sqrt(2)', '\\\\sqrt{{2}}', '1.414']
Ground-truth: '\\\\sqrt{{2}}'
→ Match

Example 6:
Predicted: ['2', '3', '1']
Ground-truth: '1'
→ Match

Now evaluate the following:

Predicted: {pred_list}
Ground-truth: {gt_label}
→
"""

def majority_check(right_answer, answers):    
    key = openai_key
    base = openai_base
    model = openai_model
    
    temperature = 0.0
    max_tokens = 50

    response= client.chat.completions.create(
        model=openai_model,
        messages=[
            {"role": "system", "content": "You are a mathematics expert."},
            {"role": "user", "content": majority_check_prompt.format(
                                    pred_list=repr(answers),
                                    gt_label=repr(right_answer)
                                )
            },             
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    result = response.choices[0].message.content

    return result

anyone_check_prompt = """
You are a mathematics expert with deep understanding of mathematical expressions and notation. Your task is to determine whether **any** of the predicted answers semantically matches the ground-truth answer, even if they are written in different formats (e.g., '5/13' vs '\\\\frac{{5}}{{13}}'; '0.5' vs '1/2').

Instructions:
1. Examine all items in the predicted list.
2. If **any** predicted answer is mathematically equivalent to the ground-truth answer, output "Match".
3. If none of them are equivalent, output "Mismatch".

Only consider semantic mathematical equivalence. For example, '1/2' and '0.5' are equal, and '\\\\sqrt{{2}}' and '1.414' are approximately equal. Use mathematical reasoning, not string comparison.

Respond **only** with "Match" or "Mismatch".

Here are some examples:

Example 1:
Predicted: ['1/2', '0.5', '0.25']
Ground-truth: '0.5'
→ Match

Example 2:
Predicted: ['5/13', '0.3', '0.25']
Ground-truth: '\\\\frac{{5}}{{13}}'
→ Match

Example 3:
Predicted: ['2/3', '0.66', '3/5']
Ground-truth: '0.666...'
→ Match

Example 4:
Predicted: ['3/4', '0.8', '0.9']
Ground-truth: '3/7'
→ Mismatch

Example 5:
Predicted: ['sqrt(2)', '\\\\sqrt{{2}}', '1.414']
Ground-truth: '\\\\sqrt{{2}}'
→ Match

Now evaluate the following:

Predicted: {pred_list}
Ground-truth: {gt_label}
→
"""

def anyone_check(right_answer, answers):    
    try:
        key = openai_key
        base = openai_base
        model = openai_model
        
        temperature = 0.0
        max_tokens = 50
        response= client.chat.completions.create(
            model=openai_model,
            messages=[
                {"role": "system", "content": "You are a mathematics expert."},
                {"role": "user", "content": anyone_check_prompt.format(
                                        pred_list=repr(answers),
                                        gt_label=repr(right_answer)
                                    )
                },             
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        result = response.choices[0].message.content
        return result
    except Exception as e:
        print(f"OpenAI API 调用失败: {e}")
        print("跳过答案验证，返回默认结果")
        return "Mismatch"  # 默认返回不匹配
