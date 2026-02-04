from dataset import (
    load_my_dataset,
  
)

from transformers import AutoTokenizer
import math
import time
import random
import copy
import os
import numpy as np
import torch
import nvtx
import asyncio
import openai
import requests
from tqdm.asyncio import tqdm

small_model = openai.Client(base_url=f"http://127.0.0.1:40001/v1", api_key="None")
small_model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
eval_model = openai.Client(base_url=f"http://127.0.0.1:40000/v1", api_key="None")
tokenizer = AutoTokenizer.from_pretrained("Qwen/QwQ-32B")
tokenizer.use_default_system_prompt = True 

def build_question(question):
    if type(question) == str:
        return f"""
    Please answer the following problem using step-by-step reasoning.
    Please separate your reasoning steps with two newline characters (\\n\\n).
    Please must put your final answer within \\boxed{{}}.

    Question: {question}
    """
    elif type(question) == tuple:
        return f"""
    This is a multiple-choice question.
    Please answer the following problem using step-by-step reasoning.
    Separate each reasoning step with **two newline characters** (`\n\n`).
    You must put your final answer within \\boxed{{}}, such as \\boxed{{A}}, \\boxed{{B}}, \\boxed{{C}}, or \\boxed{{D}}. No other formats are allowed.

    Question: {question[0]}

    Choices:
    A. {question[1]}
    B. {question[2]}
    C. {question[3]}
    D. {question[4]}

    """

def build_cot(history):
    history = "\n\n".join([
        f"\n{history[i]}\n"
        for i in range(len(history))
    ])
    
    return " <think> " + history

def build_small_init_prompt(question):
    message = [
        {"role": "system", "content": "You are a math expert."},
        {"role": "user", "content": build_question(question)}
    ]
    return message

async def call_small_model(prompt, small_model_max_tokens):
    message = build_small_init_prompt(prompt[0])
    # print(f"Small model prompt: {message}")
    global small_model, small_model_name
    response = await asyncio.to_thread(
        small_model.chat.completions.create,
        model=small_model_name,
        messages=message,
        temperature=0.7,
        max_tokens=small_model_max_tokens,
        stop=["\\boxed{"],
    )

    return response.choices[0].message.content

def build_eval_prompt(question, history):
    prompts = "\n\n".join([
        f"{history[i]}"
        for i in range(len(history))
    ])
    message = build_question(question) + "\n" + prompts
    return message

async def call_eval_model_ppl(prompt):
    global tokenizer
    message = build_eval_prompt(prompt[0], prompt[1])
    prompt[1][-1] = prompt[1][-1].strip('\n')

    position = message.find(prompt[1][-1])
    sub_message = message[:position]
    logprob_start_len = len(tokenizer.tokenize(sub_message))

    if logprob_start_len == -1:
        print(message)
        print("---------------------------")
        print(prompt[1][-1])
        raise ValueError("Prompt tokens not found in full tokens.")

    response = requests.post(
        "http://localhost:40000/generate",
        json={
            "text": message,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": 1,
            },
            "return_logprob": True,
            "logprob_start_len": logprob_start_len,
            "top_logprobs_num": 1,
        },
    )

    input_token_logprobs = response.json()['meta_info']['input_token_logprobs'][1:]
    logprobs = [entry[0] for entry in input_token_logprobs if entry[0] is not None]
    avg_neg_logprob = -sum(logprobs) / len(logprobs)
    ppl = math.exp(avg_neg_logprob)
    return ppl

async def process_single_problem(problem, small_model_max_tokens):
    prompt = [problem, []]
    small_out = await call_small_model(prompt, small_model_max_tokens)
    prompt[1].append(small_out)
    # print(f"Small model output: {small_out}")
    ppl = await call_eval_model_ppl(prompt)
    return ppl

async def compute_ppl(
        problems,
        small_model_max_tokens, 
        ppl_array=None,
        random_enable=True,
        turns=1,
    ):

    all_tasks = [
        asyncio.create_task(
            process_single_problem(
                problem,
                small_model_max_tokens,
            )
        )
        for problem in problems
    ]

    # 使用 tqdm 显示进度条
    ppls = await tqdm.gather(*all_tasks, desc="Processing problems")
    print(ppls)
    np.save(ppl_array, np.array(ppls))
    return None

async def main():
    repeats = 16
    confidence = 0
    conformal_hyperparam = 1
    random_enable = True
    self_eval = False
    turns = 1
    context, answer = load_my_dataset("gpqa", repeats)
    ppl_array = '/zju_0038/xj/sglang-parallel-test-time-scaling/ppls_gpqa_64_qwq_32_r1_1.npy'
    dictionary = dict()
    for item in zip(context, answer):
        dictionary[item[0]] = item[1]

    small_model_max_tokens = 2000
    start_time = time.time()
    # torch.cuda.cudart().cudaProfilerStart()
    # with nvtx.annotate("description", color="blue"):
    # print(context)
    print(f"repeats: {repeats}")
    cont = await compute_ppl(
        problems=context,
        small_model_max_tokens=small_model_max_tokens,
        ppl_array=ppl_array,
        random_enable=random_enable,
        turns=turns,
    )
    # torch.cuda.cudart().cudaProfilerStop()
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"Elapsed time: {elapsed:.6f} seconds")

if __name__ == "__main__":
    asyncio.run(main())

