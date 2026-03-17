"""Single-model asynchronous test-time scaling.

The same model acts as both the draft generator and the PPL evaluator / continuer.
When the self-evaluated PPL percentile exceeds the threshold, the model generates
additional tokens to refine its own answer (instead of handing off to a target model).

PPL is computed via a separate prefill-only forward pass using the /generate endpoint.
The generation and PPL calls share the same prompt prefix, leveraging SGLang's prefix
caching so the PPL evaluation reuses cached KV states and is nearly free.
"""
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

client = None
model_semaphore = None
model_name = ""
tokenizer = None
max_retries = 3
extract_mode = "regex"


async def _post_with_retry(http_client, url, payload):
    for attempt in range(max_retries):
        try:
            resp = await http_client.post(url, json=payload)
            resp.raise_for_status()
            return resp
        except (httpx.ReadError, httpx.ConnectError, httpx.RemoteProtocolError) as e:
            if attempt < max_retries - 1:
                wait = 2 * (attempt + 1)
                print(f"[Retry {attempt+1}/{max_retries}] {type(e).__name__}, retrying in {wait}s...", flush=True)
                await asyncio.sleep(wait)
            else:
                raise


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


def build_init_prompt(question):
    return [
        {"role": "system", "content": "You are a math expert."},
        {"role": "user", "content": build_question(question)},
    ]


def build_continue_prompt(question, history):
    return [
        {"role": "system", "content": "You are a math expert."},
        {"role": "user", "content": build_question(question)},
        {"role": "assistant", "content": build_cot(history)},
    ]


async def call_model_generate(prompt, turn, max_tokens, idx, port, temperature):
    """Generate tokens via /v1/chat/completions (no logprobs)."""
    messages = (
        build_init_prompt(prompt[0])
        if turn == 0
        else build_continue_prompt(prompt[0], prompt[1])
    )

    global model_semaphore, client, model_name
    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    async with model_semaphore:
        resp = await _post_with_retry(client, f"http://127.0.0.1:{port}/v1/chat/completions", payload)
        data = resp.json()
        return data["choices"][0]["message"]["content"]


async def call_model_ppl(prompt, turn, idx, port):
    """Compute PPL of the latest output via prefill-only /generate call.

    Reconstructs the same token prefix that the generation call used,
    so SGLang's prefix cache provides a nearly-free forward pass.
    """
    global tokenizer, client, model_semaphore

    if turn == 0:
        messages = build_init_prompt(prompt[0])
    else:
        messages = [{"role": "system", "content": "You are a math expert."}, {"role": "user", "content": build_question(prompt[0])}]

    template_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True
    )

    all_history = prompt[1]

    if len(all_history) == 1:
        full_text = all_history[0]
        logprob_start_len = len(template_ids)
    else:
        prefix_text = "\n\n".join(all_history[:-1]) + "\n\n"
        full_text = "\n\n".join(all_history)
        prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
        logprob_start_len = len(template_ids) + len(prefix_ids)

    full_ids = tokenizer.encode(full_text, add_special_tokens=False)
    input_ids = template_ids + full_ids

    payload = {
        "input_ids": input_ids,
        "sampling_params": {"temperature": 0, "max_new_tokens": 1},
        "return_logprob": True,
        "logprob_start_len": logprob_start_len,
        "top_logprobs_num": 1,
    }

    async with model_semaphore:
        resp = await _post_with_retry(client, f"http://127.0.0.1:{port}/generate", payload)
        data = resp.json()
        input_token_logprobs = data["meta_info"]["input_token_logprobs"][1:]
        logprobs = [entry[0] for entry in input_token_logprobs if entry[0] is not None]

        if not logprobs:
            print(f"[idx={idx}] No logprobs at turn {turn}, ppl=inf", flush=True)
            return float("inf")

        avg_neg_logprob = -sum(logprobs) / len(logprobs)
        return math.exp(avg_neg_logprob)


def _extract_boxed_balanced(text):
    """Extract content from \\boxed{...} with balanced brace matching."""
    results = []
    pattern = "\\boxed{"
    i = 0
    while i < len(text):
        pos = text.find(pattern, i)
        if pos == -1:
            break
        start = pos + len(pattern)
        depth = 1
        j = start
        while j < len(text) and depth > 0:
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
            j += 1
        if depth == 0:
            results.append(text[start : j - 1])
        i = j
    return results


async def extract_answer_regex(history):
    answer = "invalid"
    temp = "\n\n".join([f"{history[i]}" for i in range(len(history))])
    matches = _extract_boxed_balanced(temp)
    if not matches:
        temp_direct = "".join(history)
        matches = _extract_boxed_balanced(temp_direct)
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
    aclient = AsyncOpenAI(api_key=openai_key, base_url=openai_base)
    prompt_text = EXTRACT_ANSWER_LLM_PROMPT.format(history=temp)

    for attempt in range(3):
        try:
            response = await aclient.chat.completions.create(
                model=openai_model,
                messages=[
                    {"role": "system", "content": "You are a precise answer extractor."},
                    {"role": "user", "content": prompt_text},
                ],
                temperature=0.0,
                max_tokens=50,
            )
            result = response.choices[0].message.content.strip()
            return result if result and result.lower() != "invalid" else "invalid"
        except APITimeoutError:
            if attempt < 2:
                await asyncio.sleep(2 ** attempt)
            else:
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
    max_tokens,
    continue_max_tokens,
    ppl_array,
    turns,
    idx,
    port,
    temperature,
    output_dir,
):
    """Multi-turn self-evaluation loop with a single model.

    Each turn:
      1. Model generates a draft chunk.
      2. PPL is computed via a prefill-only forward pass (prefix-cache friendly).
      3. If PPL percentile >= 0.6, model generates *more* tokens to refine.
      4. Check for a valid answer; early-stop if found.
    """
    prompt = [problem, []]
    answer = "invalid"
    start_time = time.time()
    history_log = []

    for turn in range(turns):
        out = await call_model_generate(prompt, turn, max_tokens, idx, port, temperature)

        if not out:
            print(f"[idx={idx}] Empty output at turn {turn}, retrying", flush=True)
            continue

        history_log.append({"turn": turn, "model": "draft", "output": out})
        prompt[1].append(out)

        temp_ans = await extract_answer(prompt[1])
        if temp_ans != "invalid":
            answer = temp_ans
            print(f"[idx={idx}] Early stop at turn {turn}", flush=True)
            break

        ppl = await call_model_ppl(prompt, turn, idx, port)
        rank = np.sum(ppl_array < ppl)
        percent = rank / len(ppl_array)
        history_log.append({"turn": turn, "model": "self_ppl", "ppl": ppl, "percentile": percent})

        if percent >= 0.6:
            continue_out = await call_model_generate(
                prompt, turn + 1, continue_max_tokens, idx, port, temperature
            )
            if continue_out:
                history_log.append({"turn": turn, "model": "self_continue", "output": continue_out})
                prompt[1].append(continue_out)

        temp_ans = await extract_answer(prompt[1])
        if temp_ans != "invalid":
            answer = temp_ans
            print(f"[idx={idx}] Early stop at turn {turn}", flush=True)
            break

    if answer == "invalid":
        answer = await extract_answer(prompt[1])

    result_data = {
        "problem_index": idx,
        "final_answer": answer,
        "duration_seconds": time.time() - start_time,
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
        ans = await anyone_check(correct_answer, outputs)
        if ans == "Match":
            right += 1
        pbar.set_postfix(acc=f"{right/(i+1):.2%}")
    print(f"Accuracy: {right / group:.2%}")


async def main():
    parser = argparse.ArgumentParser(description="Single-model async test-time scaling (self-evaluation).")
    parser.add_argument("--model_name", type=str, required=True, help="Model name (used for everything).")
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--ppl_array_path", type=str, required=True, help="Path to PPL array (.npy) from conformal calibration.")
    parser.add_argument("--model_port", type=int, default=40000)
    parser.add_argument("--max_tokens", type=int, default=500, help="Max tokens per draft turn.")
    parser.add_argument("--continue_max_tokens", type=int, default=500, help="Max tokens for continuation when PPL is high.")
    parser.add_argument("--turns", type=int, default=15, help="Max number of turns.")
    parser.add_argument("--repeats", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--concurrency", type=int, default=16, help="Max concurrent requests to the model.")
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--extract_mode", type=str, choices=["regex", "llm"], default="regex")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_problems", type=int, default=0, help="Max unique problems to process (0 = all). Useful for debugging.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    global client, model_name, model_semaphore, max_retries, extract_mode, tokenizer
    max_retries = args.max_retries
    extract_mode = args.extract_mode
    model_name = args.model_name

    model_semaphore = asyncio.Semaphore(args.concurrency)
    client = httpx.AsyncClient(
        timeout=24000.0,
        limits=httpx.Limits(max_connections=1000, max_keepalive_connections=1000),
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

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
        if any(idx not in processed_sample_indices for idx in range(start_idx, end_idx)):
            unique_problems_to_process.append(unique_idx)

    if args.max_problems > 0:
        unique_problems_to_process = unique_problems_to_process[:args.max_problems]

    if not unique_problems_to_process:
        print("All tasks already finished.")
        all_results = []
        for idx in range(total_samples):
            filepath = os.path.join(args.output_dir, f"problem_{idx:04d}.json")
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                all_results.append((data["final_answer"], data["duration_seconds"]))
        await compute_score(all_results, answer, args.repeats)
        return

    print(f"Single-model async evaluation (prefill PPL, prefix-cache friendly)")
    print(f"  Model: {args.model_name}")
    print(f"  Dataset: {args.dataset_name}")
    print(f"  {len(unique_problems_to_process)} groups to process")

    start_time = time.time()

    for unique_idx in sync_tqdm(unique_problems_to_process, desc="Processing problem groups"):
        tasks_to_run = []
        start_sample_idx = unique_idx * args.repeats
        end_sample_idx = start_sample_idx + args.repeats

        for sample_idx in range(start_sample_idx, end_sample_idx):
            if sample_idx not in processed_sample_indices:
                problem = context[sample_idx]
                task = asyncio.create_task(
                    process_single_problem(
                        problem,
                        args.max_tokens,
                        args.continue_max_tokens,
                        ppl_array,
                        args.turns,
                        sample_idx,
                        args.model_port,
                        args.temperature,
                        args.output_dir,
                    )
                )
                tasks_to_run.append(task)

        if tasks_to_run:
            await tqdm.gather(*tasks_to_run, desc=f"Group {unique_idx} samples")

    print(f"Time: {time.time() - start_time:.3f}s")

    if args.max_problems > 0:
        eval_problems = unique_problems_to_process
        eval_total = len(eval_problems) * args.repeats
        print(f"Debug mode: evaluating {len(eval_problems)} problems ({eval_total} samples)")
        all_results = []
        all_answers = []
        for unique_idx in eval_problems:
            start_sample_idx = unique_idx * args.repeats
            end_sample_idx = start_sample_idx + args.repeats
            for idx in range(start_sample_idx, end_sample_idx):
                filepath = os.path.join(args.output_dir, f"problem_{idx:04d}.json")
                if os.path.exists(filepath):
                    with open(filepath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        all_results.append((data["final_answer"], data["duration_seconds"]))
                        all_answers.append(answer[idx])
                else:
                    print(f"Warning: {filepath} missing.")
        if all_results:
            await compute_score(all_results, all_answers, args.repeats)
    else:
        all_files_exist = True
        for idx in range(total_samples):
            filepath = os.path.join(args.output_dir, f"problem_{idx:04d}.json")
            if not os.path.exists(filepath):
                print(f"Error: {filepath} missing.")
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
            print("Missing result files. Rerun to complete.")

    await client.aclose()


if __name__ == "__main__":
    asyncio.run(main())
