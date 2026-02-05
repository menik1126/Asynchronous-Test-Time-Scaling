import argparse
from dataset import (
    load_my_dataset,
)

from transformers import AutoTokenizer
import math
import time
import numpy as np
import asyncio
import openai
import requests
from tqdm.asyncio import tqdm

small_model = None
eval_model = None
small_model_name = ""
eval_model_name = ""
tokenizer = None

def build_question(question):
    if isinstance(question, str):
        return f"""
    Please answer the following problem using step-by-step reasoning.
    Please separate your reasoning steps with two newline characters (\\n\\n).
    Please must put your final answer within \\boxed{{}}.

    Question: {question}
    """
    elif isinstance(question, tuple):
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
    return ""

def build_small_init_prompt(question):
    message = [
        {"role": "system", "content": "You are a math expert."},
        {"role": "user", "content": build_question(question)}
    ]
    return message

async def call_small_model(prompt, small_model_max_tokens, small_model_temperature):
    message = build_small_init_prompt(prompt[0])
    global small_model, small_model_name
    response = await asyncio.to_thread(
        small_model.chat.completions.create,
        model=small_model_name,
        messages=message,
        temperature=small_model_temperature,
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

async def call_eval_model_ppl(prompt, eval_model_port):
    global tokenizer
    message = build_eval_prompt(prompt[0], prompt[1])

    last_history_item = prompt[1][-1].strip('\n')

    position = message.find(last_history_item)
    if position == -1:
        print(message)
        print("---------------------------")
        print(last_history_item)
        raise ValueError("Prompt tokens not found in full tokens.")

    sub_message = message[:position]
    logprob_start_len = len(tokenizer.tokenize(sub_message))

    response = requests.post(
        f"http://localhost:{eval_model_port}/generate",
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

    try:
        input_token_logprobs = response.json()['meta_info']['input_token_logprobs'][1:]
        logprobs = [entry[0] for entry in input_token_logprobs if entry[0] is not None]
        avg_neg_logprob = -sum(logprobs) / len(logprobs)
        ppl = math.exp(avg_neg_logprob)
    except (KeyError, IndexError, ValueError) as e:
        print(f"Error parsing response from eval model: {e}")
        ppl = float('inf')
    return ppl

async def process_single_problem(problem, small_model_max_tokens, eval_model_port, small_model_temperature):
    prompt = [problem, []]
    small_out = await call_small_model(prompt, small_model_max_tokens, small_model_temperature)
    prompt[1].append(small_out)
    ppl = await call_eval_model_ppl(prompt, eval_model_port)
    return ppl

async def compute_ppl(problems, small_model_max_tokens, ppl_array_path, eval_model_port, small_model_temperature, max_concurrent=16, sample_size=16):
    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_with_semaphore(original_problem_idx, problem):
        async with semaphore:
            ppl = await process_single_problem(
                problem,
                small_model_max_tokens,
                eval_model_port,
                small_model_temperature,
            )
            return original_problem_idx, ppl

    all_tasks = [
        asyncio.create_task(process_with_semaphore(i // sample_size, problem))
        for i, problem in enumerate(problems)
    ]

    results = await tqdm.gather(*all_tasks, desc=f"Processing problems (max {max_concurrent} concurrent)")

    ppls_by_problem = {}
    for problem_idx, ppl in results:
        if problem_idx not in ppls_by_problem:
            ppls_by_problem[problem_idx] = []
        ppls_by_problem[problem_idx].append(ppl)

    np.save(ppl_array_path, ppls_by_problem)

    for original_problem_idx, ppls in ppls_by_problem.items():
        print(f"problem {original_problem_idx}: {len(ppls)} samples, PPL range: [{min(ppls):.4f}, {max(ppls):.4f}]")

    return ppls_by_problem

async def main():
    parser = argparse.ArgumentParser(description="Run baseline evaluation with customizable models and dataset.")
    parser.add_argument("--small_model_name", type=str, required=True,
                        help="Name of the small model for generating responses.")
    parser.add_argument("--eval_model_name", type=str, required=True,
                        help="Name of the model to use for PPL evaluation.")
    parser.add_argument("--dataset_name", type=str, required=True,
                        help="Name of the dataset to use (e.g., gpqa, math500).")
    parser.add_argument("--small_model_max_tokens", type=int, default=100,
                        help="Maximum tokens for the small model's response.")
    parser.add_argument("--ppl_array_path", type=str, required=True,
                        help="Path to save the PPL results array (.npy file).")
    parser.add_argument("--small_model_port", type=int, default=40001,
                        help="Port for the small model server.")
    parser.add_argument("--eval_model_port", type=int, default=40000,
                        help="Port for the evaluation model server.")
    parser.add_argument("--sample_size", type=int, default=16,
                        help="Number of samples per problem.")
    parser.add_argument("--max_concurrent", type=int, default=16,
                        help="Maximum number of concurrent requests.")
    parser.add_argument("--small_model_temperature", type=float, default=0.8,
                        help="Temperature for the small model sampling.")

    args = parser.parse_args()

    global small_model, eval_model, small_model_name, eval_model_name, tokenizer
    small_model_name = args.small_model_name
    eval_model_name = args.eval_model_name

    small_model = openai.Client(base_url=f"http://127.0.0.1:{args.small_model_port}/v1", api_key="None")
    eval_model = openai.Client(base_url=f"http://127.0.0.1:{args.eval_model_port}/v1", api_key="None")

    tokenizer = AutoTokenizer.from_pretrained(args.eval_model_name)
    tokenizer.use_default_system_prompt = True

    context, answer = load_my_dataset(args.dataset_name, args.sample_size)

    start_time = time.time()

    print(f"Starting evaluation on dataset: {args.dataset_name}")
    print(f"Saving results to: {args.ppl_array_path}")

    await compute_ppl(
        problems=context,
        small_model_max_tokens=args.small_model_max_tokens,
        ppl_array_path=args.ppl_array_path,
        eval_model_port=args.eval_model_port,
        small_model_temperature=args.small_model_temperature,
        max_concurrent=args.max_concurrent,
        sample_size=args.sample_size,
    )

    end_time = time.time()
    elapsed = end_time - start_time
    print(f"Elapsed time: {elapsed:.6f} seconds")

if __name__ == "__main__":
    asyncio.run(main())
