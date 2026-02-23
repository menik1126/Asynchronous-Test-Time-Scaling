import argparse
import httpx
import asyncio
import numpy as np
import math
import time
import re
import os
import json
from tqdm.asyncio import tqdm
from tqdm import tqdm as sync_tqdm

from transformers import AutoTokenizer
from ATTS.dataset import load_my_dataset
from ATTS.async_agent import anyone_check

client_small = None
client_eval = None
small_model_semaphore = None
eval_model_semaphore = None

small_model_name = ""
eval_model_name = ""
tokenizer = None
small_tokenizer = None
max_retries = 3
extract_mode = "regex"
use_chat_template = False


async def _post_with_retry(client, url, json):
    for attempt in range(max_retries):
        try:
            resp = await client.post(url, json=json)
            resp.raise_for_status()
            return resp
        except (httpx.ReadError, httpx.ConnectError, httpx.RemoteProtocolError) as e:
            if attempt < max_retries - 1:
                wait = 2 * (attempt + 1)
                print(f"[Retry {attempt+1}/{max_retries}] {type(e).__name__}, retrying in {wait}s...", flush=True)
                await asyncio.sleep(wait)
            else:
                raise


def build_debug_prompt():
    message = [
        {
            "role": "system",
            "content": "You are a creative and expressive assistant. Feel free to write anything you want, in any format or style.",
        },
        {"role": "user", "content": "Go ahead and write freely. No need to stop."},
    ]
    return message


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


def build_cot(history):
    return "\n\n".join([f"{h}" for h in history])


def build_small_init_prompt(question):
    return [
        {"role": "system", "content": "You are a math expert."},
        {"role": "user", "content": build_question(question)},
    ]


def build_small_inner_prompt(question, history):
    return [
        {"role": "user", "content": build_question(question)},
        {"role": "assistant", "content": build_cot(history)},
    ]


def build_eval_prompt_for_generate(question, history):
    return [
        {"role": "user", "content": build_question(question)},
        {"role": "assistant", "content": build_cot(history)},
    ]


def build_eval_prompt_for_eval(question, history):
    prompts = "\n\n".join([f"{history[i]}" for i in range(len(history))])
    message = build_question(question) + "\n" + prompts
    return message


async def call_small_model(prompt, turn, max_tokens, idx, port, temperature):
    messages = (
        build_small_init_prompt(prompt[0])
        if turn == 0
        else build_small_inner_prompt(prompt[0], prompt[1])
    )

    global small_model_semaphore, client_small, small_model_name
    payload = {
        "model": small_model_name,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    async with small_model_semaphore:
        resp = await _post_with_retry(client_small, f"http://127.0.0.1:{port}/v1/chat/completions", payload)
        return resp.json()["choices"][0]["message"]["content"]


async def call_eval_model(prompt, max_tokens, idx, port, temperature):
    messages = build_eval_prompt_for_generate(prompt[0], prompt[1])
    global eval_model_semaphore, client_eval, eval_model_name
    payload = {
        "model": eval_model_name,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    async with eval_model_semaphore:
        resp = await _post_with_retry(client_eval, f"http://127.0.0.1:{port}/v1/chat/completions", payload)
        return resp.json()["choices"][0]["message"]["content"]


async def call_eval_model_ppl(prompt, idx, port):
    global client_eval, tokenizer, use_chat_template

    if use_chat_template:
        history_text = "\n\n".join([f"{h}" for h in prompt[1]])

        # Replicate the server's /v1/chat/completions tokenization exactly:
        #   template_ids = apply_chat_template([user], tokenize=True, add_generation_prompt=True)
        #   assistant_ids = tokenizer.encode(assistant_prefix)
        #   input_ids = template_ids + assistant_ids
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
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": 1,
            },
            "return_logprob": True,
            "logprob_start_len": logprob_start_len,
            "top_logprobs_num": 1,
        }
    else:
        message = build_eval_prompt_for_eval(prompt[0], prompt[1])

        last_history_item = prompt[1][-1].strip("\n")
        position = message.find(last_history_item)
        if position == -1:
            print(message)
            print("---------------------------")
            print(last_history_item)
            raise ValueError("Prompt tokens not found in full tokens.")

        sub_message = message[:position]
        logprob_start_len = len(tokenizer.tokenize(sub_message))

        payload = {
            "text": message,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": 1,
            },
            "return_logprob": True,
            "logprob_start_len": logprob_start_len,
            "top_logprobs_num": 1,
        }

    global eval_model_semaphore
    async with eval_model_semaphore:
        resp = await _post_with_retry(client_eval, f"http://127.0.0.1:{port}/generate", payload)
        data = resp.json()
        input_token_logprobs = data["meta_info"]["input_token_logprobs"][1:]
        logprobs = [entry[0] for entry in input_token_logprobs if entry[0] is not None]

        if not logprobs:
            print(f"No log probabilities returned for problem: {prompt[0]}", flush=True)
            return 0

        avg_neg_logprob = -sum(logprobs) / len(logprobs)
        return math.exp(avg_neg_logprob)


async def extract_answer_regex(history):
    answer = "invalid"
    temp = "\n\n".join([f"{history[i]}" for i in range(len(history))])

    matches = re.findall(r"\\boxed\{(.*?)\}", temp)
    if matches:
        answer = matches[-1]
    else:
        pattern = re.compile(r"ANSWER:\s+([A-Z])", re.IGNORECASE)
        matches = pattern.findall(temp)
        if matches:
            answer = matches[-1]

    return answer


EXTRACT_ANSWER_LLM_PROMPT = """You are tasked with extracting the final answer from a mathematical reasoning process.

Below is the reasoning history. Please identify and extract ONLY the final answer.

Rules:
- If the answer is inside \\boxed{{}}, extract the content within.
- If it is a multiple-choice question, return only the letter (A, B, C, or D).
- If it is a numerical answer, return only the number.
- Return ONLY the answer itself, nothing else. No explanation, no extra text.
- If you cannot find a clear answer, return exactly: invalid

Reasoning history:
{history}

Final answer:"""


async def extract_answer_llm(history):
    from openai import AsyncOpenAI, APITimeoutError

    openai_key = os.environ.get("OPENAI_API_KEY", "")
    openai_base = os.environ.get("OPENAI_BASE_URL") or None
    openai_model = os.environ.get("OPENAI_MODEL", "gpt-5.2")

    temp = "\n\n".join([f"{history[i]}" for i in range(len(history))])

    client = AsyncOpenAI(api_key=openai_key, base_url=openai_base)
    prompt_text = EXTRACT_ANSWER_LLM_PROMPT.format(history=temp)

    retries = 3
    delay = 1
    for attempt in range(retries):
        try:
            response = await client.chat.completions.create(
                model=openai_model,
                messages=[
                    {"role": "system", "content": "You are a precise answer extractor."},
                    {"role": "user", "content": prompt_text},
                ],
                temperature=0.0,
                max_tokens=50,
            )
            print("--------------------------------")
            print("extract_answer_llm response: ", response.choices[0].message.content.strip())
            print("--------------------------------")
            result = response.choices[0].message.content.strip()
            return result if result and result.lower() != "invalid" else "invalid"
        except APITimeoutError:
            if attempt < retries - 1:
                wait = delay * (2 ** attempt)
                print(f"[extract_answer_llm] Timeout, retrying in {wait}s...", flush=True)
                await asyncio.sleep(wait)
            else:
                print("[extract_answer_llm] All retries exhausted.", flush=True)
                return "invalid"
        except Exception as e:
            print(f"[extract_answer_llm] Error: {e}", flush=True)
            return "invalid"

    return "invalid"


async def extract_answer(history):
    global extract_mode
    if extract_mode == "llm":
        return await extract_answer_llm(history)
    return await extract_answer_regex(history)


async def process_single_problem(
    problem,
    small_model_max_tokens,
    evalator_max_tokens,
    ppl_array,
    turns,
    idx,
    small_model_port,
    eval_model_port,
    output_dir,
    small_model_temperature,
    eval_model_temperature,
):
    prompt = [problem, []]
    answer = "invalid"
    start_time = time.time()

    history_log = []

    for turn in range(turns):
        small_out = await call_small_model(
            prompt,
            turn,
            small_model_max_tokens,
            idx,
            small_model_port,
            small_model_temperature,
        )
        history_log.append({"turn": turn, "model": "small", "output": small_out})
        prompt[1].append(small_out)

        if not small_out:
            print("Small model returned empty output.", flush=True)
            break

        ppl = await call_eval_model_ppl(prompt, idx, eval_model_port)
        rank = np.sum(ppl_array < ppl)
        percent = rank / len(ppl_array)
        history_log.append(
            {"turn": turn, "model": "eval_ppl", "ppl": ppl, "percentile": percent}
        )

        if percent >= 0.6:
            eval_out = await call_eval_model(
                prompt,
                evalator_max_tokens,
                idx,
                eval_model_port,
                eval_model_temperature,
            )
            history_log.append(
                {"turn": turn, "model": "eval_generate", "output": eval_out}
            )
            prompt[1].append(eval_out)

        temp = await extract_answer(prompt[1])
        if temp != "invalid":
            answer = temp
            print("Early stop due to valid answer found.")
            break

    if answer == "invalid":
        answer = await extract_answer(prompt[1])

    end_time = time.time()
    duration = end_time - start_time

    result_data = {
        "problem_index": idx,
        "final_answer": answer,
        "duration_seconds": duration,
        "full_history": history_log,
        "question": problem,
    }

    output_filename = os.path.join(output_dir, f"problem_{idx:04d}.json")
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=4)

    return ()


async def compute_score(results, answers, repeats):
    generated_ans = [ans for ans, _ in results]
    group = len(generated_ans) // repeats
    right = 0
    pbar = sync_tqdm(range(group), desc="Evaluating", unit="question")
    for i in pbar:
        start = i * repeats
        end = (i + 1) * repeats
        outputs = generated_ans[start:end]
        correct_answer = answers[start]
        print(f"Generated answers: {outputs}")
        print(f"Correct answers: {correct_answer}")
        ans = await anyone_check(correct_answer, outputs)
        print(ans)
        if ans == "Match":
            right += 1
        pbar.set_postfix(acc=f"{right/(i+1):.2%}")

    print(f"Accuracy: {right / group:.2%}")


async def main():
    parser = argparse.ArgumentParser(
        description="Run a multi-turn, multi-agent evaluation."
    )
    parser.add_argument(
        "--small_model_name",
        type=str,
        required=True,
        help="Name of the small model for generating responses.",
    )
    parser.add_argument(
        "--eval_model_name",
        type=str,
        required=True,
        help="Name of the model to use for PPL evaluation.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="Name of the dataset to use (e.g., gpqa, math500).",
    )
    parser.add_argument(
        "--ppl_array_path",
        type=str,
        required=True,
        help="Path to the PPL results array (.npy file).",
    )
    parser.add_argument(
        "--turns",
        type=int,
        default=40,
        help="Maximum number of turns for the multi-agent loop.",
    )
    parser.add_argument(
        "--small_model_max_tokens",
        type=int,
        default=200,
        help="Maximum tokens for the small model's response.",
    )
    parser.add_argument(
        "--evalator_max_tokens",
        type=int,
        default=200,
        help="Maximum tokens for the evaluation model's response.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=16,
        help="Number of times to repeat each problem.",
    )
    parser.add_argument(
        "--small_model_port",
        type=int,
        default=52103,
        help="Port for the small model server.",
    )
    parser.add_argument(
        "--eval_model_port",
        type=int,
        default=52102,
        help="Port for the evaluation model server.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the results and history.",
    )
    parser.add_argument(
        "--small_model_temperature",
        type=float,
        default=0.6,
        help="Temperature for the small model generation.",
    )
    parser.add_argument(
        "--eval_model_temperature",
        type=float,
        default=0.6,
        help="Temperature for the evaluation model generation.",
    )
    parser.add_argument(
        "--small_model_concurrency",
        type=int,
        default=16,
        help="Concurrency limit for small model requests.",
    )
    parser.add_argument(
        "--eval_model_concurrency",
        type=int,
        default=8,
        help="Concurrency limit for evaluation model requests.",
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=3,
        help="Max retries for HTTP requests on connection errors.",
    )
    parser.add_argument(
        "--extract_mode",
        type=str,
        choices=["regex", "llm"],
        default="regex",
        help="Answer extraction mode: 'regex' for pattern matching, 'llm' for LLM-based extraction.",
    )
    parser.add_argument(
        "--use_chat_template",
        action="store_true",
        default=False,
        help="Apply chat template when building eval model prompts for PPL.",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    global client_small, client_eval, small_model_name, eval_model_name, tokenizer, small_tokenizer, small_model_semaphore, eval_model_semaphore, max_retries, extract_mode, use_chat_template
    max_retries = args.max_retries
    extract_mode = args.extract_mode
    use_chat_template = args.use_chat_template


    small_model_name = args.small_model_name
    eval_model_name = args.eval_model_name

    small_model_semaphore = asyncio.Semaphore(args.small_model_concurrency)
    eval_model_semaphore = asyncio.Semaphore(args.eval_model_concurrency)

    client_small = httpx.AsyncClient(
        timeout=24000.0,
        limits=httpx.Limits(max_connections=1000, max_keepalive_connections=1000),
    )
    client_eval = httpx.AsyncClient(
        timeout=24000.0,
        limits=httpx.Limits(max_connections=1000, max_keepalive_connections=1000),
    )

    tokenizer = AutoTokenizer.from_pretrained(args.eval_model_name)
    tokenizer.use_default_system_prompt = True
    small_tokenizer = AutoTokenizer.from_pretrained(args.small_model_name)
    small_tokenizer.use_default_system_prompt = True

    context, answer = load_my_dataset(args.dataset_name, args.repeats)
    ppl_array = np.load(args.ppl_array_path)

    total_unique_problems = len(answer) // args.repeats
    total_samples = len(context)

    processed_sample_indices = set()
    for filename in os.listdir(args.output_dir):
        if filename.startswith("problem_") and filename.endswith(".json"):
            try:
                sample_idx = int(filename.replace("problem_", "").replace(".json", ""))
                processed_sample_indices.add(sample_idx)
            except ValueError:
                continue

    unique_problems_to_process = []
    for unique_idx in range(total_unique_problems):
        start_idx = unique_idx * args.repeats
        end_idx = start_idx + args.repeats

        is_group_incomplete = any(
            (idx not in processed_sample_indices) for idx in range(start_idx, end_idx)
        )
        if is_group_incomplete:
            unique_problems_to_process.append(unique_idx)

    if not unique_problems_to_process:
        print("all tasks finish")
        all_results = []
        for idx in range(total_samples):
            filepath = os.path.join(args.output_dir, f"problem_{idx:04d}.json")
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                all_results.append((data["final_answer"], data["duration_seconds"]))
        await compute_score(all_results, answer, args.repeats)
        return

    print(f"find {len(unique_problems_to_process)} groups. handling...")

    start_time = time.time()

    for unique_idx in sync_tqdm(
        unique_problems_to_process, desc="Processing problem groups"
    ):
        tasks_to_run_for_group = []
        start_sample_idx = unique_idx * args.repeats
        end_sample_idx = start_sample_idx + args.repeats

        for sample_idx in range(start_sample_idx, end_sample_idx):
            if sample_idx not in processed_sample_indices:
                problem = context[sample_idx]
                task = asyncio.create_task(
                    process_single_problem(
                        problem,
                        args.small_model_max_tokens,
                        args.evalator_max_tokens,
                        ppl_array,
                        args.turns,
                        sample_idx,
                        args.small_model_port,
                        args.eval_model_port,
                        samples_per_turn_increment=args.samples_per_turn_increment,
                        args.output_dir,
                        args.small_model_temperature,
                        args.eval_model_temperature,
                    )
                )
                tasks_to_run_for_group.append(task)

        if tasks_to_run_for_group:
            await tqdm.gather(
                *tasks_to_run_for_group, desc=f"Group {unique_idx} samples"
            )

    end_time = time.time()
    print(f"time: {end_time - start_time:.3f} s")

    all_files_exist = True
    for idx in range(total_samples):
        filepath = os.path.join(args.output_dir, f"problem_{idx:04d}.json")
        if not os.path.exists(filepath):
            print(
                f"Error: required result file {filepath} is missing. Unable to compute the final score."
            )
            all_files_exist = False
            break

    if all_files_exist:
        all_results = []
        for idx in range(total_samples):
            filepath = os.path.join(args.output_dir, f"problem_{idx:04d}.json")
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                all_results.append((data["final_answer"], data["duration_seconds"]))
        await compute_score(all_results, answer, args.repeats)
    else:
        print(
            "Final score will not be computed due to missing result files. Please rerun the script to complete all tasks."
        )

    await client_small.aclose()
    await client_eval.aclose()


if __name__ == "__main__":
    asyncio.run(main())
