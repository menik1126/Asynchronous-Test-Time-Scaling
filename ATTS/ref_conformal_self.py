"""Single-model conformal calibration.

The same model acts as both the draft generator and the PPL evaluator.
Only one SGLang server is needed.
"""
import argparse
import math
import time
import asyncio

import numpy as np
import openai
import requests
from transformers import AutoTokenizer
from tqdm.asyncio import tqdm

from ATTS.dataset import load_my_dataset

model_client = None
model_name = ""
tokenizer = None
use_chat_template = False


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


def build_init_prompt(question):
    return [
        {"role": "system", "content": "You are a math expert."},
        {"role": "user", "content": build_question(question)},
    ]


async def call_model_generate(prompt, max_tokens, temperature):
    message = build_init_prompt(prompt[0])
    global model_client, model_name
    response = await asyncio.to_thread(
        model_client.chat.completions.create,
        model=model_name,
        messages=message,
        temperature=temperature,
        max_tokens=max_tokens,
        stop=["\\boxed{"],
    )
    return response.choices[0].message.content


async def call_model_ppl(prompt, port):
    global tokenizer, use_chat_template

    if use_chat_template:
        history_text = "\n\n".join([f"{h}" for h in prompt[1]])
        template_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": build_question(prompt[0])}],
            tokenize=True,
            add_generation_prompt=True,
        )
        assistant_ids = tokenizer.encode(history_text)
        input_ids = template_ids + assistant_ids

        prefix_items = prompt[1][:-1]
        if prefix_items:
            prefix_text = "\n\n".join(prefix_items) + "\n\n"
        else:
            prefix_text = ""
        prefix_ids = tokenizer.encode(prefix_text)
        logprob_start_len = len(template_ids) + len(prefix_ids)

        payload = {
            "input_ids": input_ids,
            "sampling_params": {"temperature": 0, "max_new_tokens": 1},
            "return_logprob": True,
            "logprob_start_len": logprob_start_len,
            "top_logprobs_num": 1,
        }
    else:
        prompts = "\n\n".join([f"{prompt[1][i]}" for i in range(len(prompt[1]))])
        message = build_question(prompt[0]) + "\n" + prompts

        last_history_item = prompt[1][-1].strip("\n")
        position = message.find(last_history_item)
        if position == -1:
            raise ValueError("Prompt tokens not found in full tokens.")

        sub_message = message[:position]
        logprob_start_len = len(tokenizer.tokenize(sub_message))

        payload = {
            "text": message,
            "sampling_params": {"temperature": 0, "max_new_tokens": 1},
            "return_logprob": True,
            "logprob_start_len": logprob_start_len,
            "top_logprobs_num": 1,
        }

    response = await asyncio.to_thread(
        requests.post,
        f"http://localhost:{port}/generate",
        json=payload,
    )

    try:
        input_token_logprobs = response.json()["meta_info"]["input_token_logprobs"][1:]
        logprobs = [entry[0] for entry in input_token_logprobs if entry[0] is not None]
        if not logprobs:
            return float("inf")
        avg_neg_logprob = -sum(logprobs) / len(logprobs)
        return math.exp(avg_neg_logprob)
    except (KeyError, IndexError, ValueError) as e:
        print(f"Error parsing PPL response: {e}")
        return float("inf")


async def process_single_problem(problem, max_tokens, temperature, port):
    prompt = [problem, []]
    out = await call_model_generate(prompt, max_tokens, temperature)
    prompt[1].append(out)
    ppl = await call_model_ppl(prompt, port)
    return ppl


async def compute_ppl(problems, max_tokens, ppl_array_path, temperature, port, max_concurrent=16):
    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_with_semaphore(problem):
        async with semaphore:
            return await process_single_problem(problem, max_tokens, temperature, port)

    all_tasks = [asyncio.create_task(process_with_semaphore(p)) for p in problems]
    ppls = await tqdm.gather(*all_tasks, desc=f"Processing problems (max {max_concurrent} concurrent)")
    print(ppls)
    np.save(ppl_array_path, np.array(ppls))


async def main():
    parser = argparse.ArgumentParser(description="Single-model conformal calibration (self-evaluation).")
    parser.add_argument("--model_name", type=str, required=True, help="Model name (used for both generation and PPL).")
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--ppl_array_path", type=str, required=True, help="Path to save PPL array (.npy).")
    parser.add_argument("--model_port", type=int, default=40000, help="Port for the SGLang server.")
    parser.add_argument("--max_tokens", type=int, default=500)
    parser.add_argument("--sample_size", type=int, default=16)
    parser.add_argument("--max_concurrent", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--use_chat_template", action="store_true", default=False)
    args = parser.parse_args()

    global model_client, model_name, tokenizer, use_chat_template
    model_name = args.model_name
    use_chat_template = args.use_chat_template

    model_client = openai.Client(
        base_url=f"http://127.0.0.1:{args.model_port}/v1", api_key="None"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.use_default_system_prompt = True

    context, answer = load_my_dataset(args.dataset_name, args.sample_size)

    import os
    print(f"Single-model conformal calibration")
    print(f"  Model: {args.model_name}")
    print(f"  Dataset: {args.dataset_name} ({len(context)} samples)")
    print(f"  Saving to: {args.ppl_array_path}")

    for attempt in range(10):
        try:
            r = requests.get(
                f"http://127.0.0.1:{args.model_port}/get_model_info",
                timeout=5, proxies={"http": None, "https": None},
            )
            if r.status_code == 200:
                print(f"  Server on port {args.model_port}: OK")
                break
        except Exception as e:
            print(f"  Server on port {args.model_port}: {type(e).__name__}: {e}")
        print(f"  Retrying ({attempt+1}/10)...")
        time.sleep(3)
    else:
        raise RuntimeError(f"Server on port {args.model_port} not reachable")

    print("Testing /v1/chat/completions endpoint...")
    test_resp = model_client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": "1+1="}],
        max_tokens=5,
    )
    print(f"  Test response: {test_resp.choices[0].message.content!r} - OK")

    start_time = time.time()
    await compute_ppl(
        problems=context,
        max_tokens=args.max_tokens,
        ppl_array_path=args.ppl_array_path,
        temperature=args.temperature,
        port=args.model_port,
        max_concurrent=args.max_concurrent,
    )
    print(f"Elapsed: {time.time() - start_time:.3f}s")


if __name__ == "__main__":
    asyncio.run(main())
