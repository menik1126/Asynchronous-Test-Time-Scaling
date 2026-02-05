import argparse
import httpx
import asyncio
import random
import numpy as np

import math
import time
import re
import os
import json
from tqdm.asyncio import tqdm
from tqdm import tqdm as sync_tqdm

from transformers import AutoTokenizer
from dataset import load_my_dataset

client_small = None
client_eval = None
semaphore = asyncio.Semaphore(4)

small_model_name = ""
eval_model_name = ""
tokenizer = None
small_tokenizer = None

ppl_array = None
percentile_threshold = None


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
        {"role": "user", "content": build_question(question)}
    ]


def build_small_inner_prompt(question, history):
    return [
        {"role": "user", "content": build_question(question)},
        {"role": "assistant", "content": build_cot(history)}
    ]


def build_eval_prompt_for_generate(question, history):
    return [
        {"role": "user", "content": build_question(question)},
        {"role": "assistant", "content": build_cot(history)}
    ]


def build_eval_prompt_for_eval(question, history):
    prompts = "\n\n".join([
        f"{history[i]}"
        for i in range(len(history))
    ])
    message = build_question(question) + "\n" + prompts
    return message


def load_ppl_array(ppl_array_path):
    global ppl_array
    if ppl_array_path and os.path.exists(ppl_array_path):
        ppl_array = np.load(ppl_array_path, allow_pickle=True).item()
        print(f"load ppl data, {len(ppl_array)} problems")
        for problem_idx, ppls in ppl_array.items():
            print(f"problem {problem_idx}: {len(ppls)} ppl")
        return True
    else:
        print(f"ppl data doesn't exist {ppl_array_path}")
        return False


async def call_eval_model_ppl(prompt, idx, port):
    global client_eval, tokenizer
    message = build_eval_prompt_for_eval(prompt[0], prompt[1])
    last_history_item = prompt[1][-1].strip('\n')

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

    global semaphore

    try:
        async with semaphore:
            resp = await client_eval.post(
                f"http://127.0.0.1:{port}/generate",
                json=payload,
                timeout=60.0
            )
            resp.raise_for_status()
            data = resp.json()
            input_token_logprobs = data['meta_info']['input_token_logprobs'][1:]
            logprobs = [entry[0] for entry in input_token_logprobs if entry[0] is not None]
            if not logprobs:
                print(f"No log probabilities returned for problem: {prompt[0]}", flush=True)
                return 0
            
            avg_neg_logprob = -sum(logprobs) / len(logprobs)
            
            return math.exp(avg_neg_logprob)

    except Exception as e:
        print(f"compute ppl error (sample {idx}): {e}", flush=True)
        return 0


def should_takeover_based_on_percentile(ppl_value, problem_group_idx):
    global ppl_array, percentile_threshold
    
    if ppl_array is None:
        return False
    
    if problem_group_idx not in ppl_array:
        print(f"group {problem_group_idx} in ppl doesn't exist")
        return False
    
    current_problem_ppls = ppl_array[problem_group_idx]
    print(f"group ppl data: {current_problem_ppls}")
    print(f"ppl data: {ppl_value}")
    
    current_problem_ppls_array = np.array(current_problem_ppls)
    
    rank = np.sum(current_problem_ppls_array < ppl_value)

    min_ppl = np.min(current_problem_ppls_array)
    max_ppl = np.max(current_problem_ppls_array)
    print(f"compare ppl (group {problem_group_idx}): current={ppl_value:.4f}, range=[{min_ppl:.4f}, {max_ppl:.4f}], rank={rank}/{len(current_problem_ppls_array)}")
    
    percentile = rank / len(current_problem_ppls)
    
    should_takeover = percentile >= percentile_threshold
    
    if should_takeover:
        print(f"take over: PPL={ppl_value:.4f}, percentage={percentile:.3f} >= {percentile_threshold}")
    
    return should_takeover


async def call_small_model(prompt, turn, max_tokens, idx, port):

    messages = (
        build_small_init_prompt(prompt[0]) if turn == 0 else build_small_inner_prompt(prompt[0], prompt[1])
    )
    
    global semaphore, client_small, small_model_name
    payload = {
        "model": small_model_name,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": max_tokens,
        "stop": ["\\boxed{"],
    }
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            async with semaphore:
                resp = await client_small.post(
                    f"http://127.0.0.1:{port}/v1/chat/completions",
                    json=payload,
                    timeout=60.0
                )
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"]
        except (httpx.ReadError, httpx.ConnectError, httpx.TimeoutException) as e:
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 2
                print(f"small model error (try {attempt + 1}/{max_retries}): {e}. wait {wait_time}s ...", flush=True)
                await asyncio.sleep(wait_time)
            else:
                print(f"small model error (sample {idx}): {e}", flush=True)
                return ""
        except Exception as e:
            print(f"small model error (sample {idx}): {e}", flush=True)
            return ""


async def call_eval_model(prompt, max_tokens, idx, port):
    messages = build_eval_prompt_for_generate(prompt[0], prompt[1])
    global semaphore, client_eval, eval_model_name
    payload = {
        "model": eval_model_name,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": max_tokens,
        "stop": ["\\boxed{"],
    }

    try:
        async with semaphore:
            resp = await client_eval.post(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                json=payload,
                timeout=60.0
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"eval model error (sample {idx}): {e}", flush=True)
        return ""


async def extract_answer(history):
    answer = "invalid"
    temp = "\n\n".join([
        f"{history[i]}"
        for i in range(len(history))
    ])

    matches = re.findall(r"\\boxed\{(.*?)\}", temp)
    if matches:
        answer = matches[-1].strip()
        return answer
    
    pattern = re.compile(r"ANSWER:\s*([A-Z])", re.IGNORECASE)
    matches = pattern.findall(temp)
    if matches:
        answer = matches[-1].strip()
        return answer
    
    patterns = [
        r"answer[:\s]*([A-Z])",
        r"the answer is[:\s]*([A-Z])",
        r"final answer[:\s]*([A-Z])",
        r"option[:\s]*([A-Z])",
        r"choice[:\s]*([A-Z])",
    ]
    
    for pattern_str in patterns:
        pattern = re.compile(pattern_str, re.IGNORECASE)
        matches = pattern.findall(temp)
        if matches:
            answer = matches[-1].strip()
            return answer
    
    lines = temp.strip().split('\n')
    for line in reversed(lines):
        line = line.strip()
        if line and len(line) <= 10:
            return line
    
    return answer


async def process_single_problem(problem, small_model_max_tokens, evalator_max_tokens, turns, idx, small_model_port, eval_model_port, output_dir, repeats=1, takeover_stats=None, takeover_budget=None):
    prompt = [problem, []]
    answer = "invalid"
    start_time = time.time()
    
    problem_group_idx = idx // repeats
    
    history_log = []
    problem_has_takeover = False
    temp = None
    for turn in range(turns):
        print(f"📊 Problem Group {problem_group_idx} (Sample {idx}) - Turn {turn+1}/{turns}", flush=True)
        small_out = await call_small_model(prompt, turn, small_model_max_tokens, idx, small_model_port)
        history_log.append({"turn": turn, "model": "small", "output": small_out})
        prompt[1].append(small_out)

        if not small_out:
            print("Small model returned empty output.", flush=True)
            break

        ppl = await call_eval_model_ppl(prompt, idx, eval_model_port)
        should_takeover = should_takeover_based_on_percentile(ppl, problem_group_idx)
        
        history_log.append({"turn": turn, "model": "eval_ppl", "ppl": ppl, "should_takeover": int(should_takeover)})
        
        if should_takeover:
            print(f"Turn {turn+1}: take over! (Sample {idx}, PPL={ppl:.4f})", flush=True)
            eval_out = await call_eval_model(prompt, evalator_max_tokens, idx, eval_model_port)
            history_log.append({"turn": turn, "model": "eval_generate", "output": eval_out})
            prompt[1].append(eval_out)
            problem_has_takeover = True
            if takeover_stats:
                takeover_stats['total_takeovers'] += 1
                if idx not in takeover_stats['sample_takeovers']:
                    takeover_stats['sample_takeovers'][idx] = 0
                takeover_stats['sample_takeovers'][idx] += 1
        else:
            print(f"Turn {turn+1}: not take over (Sample {idx}, PPL={ppl:.4f})", flush=True)
        
        if takeover_stats and takeover_budget is not None:
            is_last_sample_in_group = (idx % repeats == repeats - 1)
            is_last_turn = (turn == turns - 1)
            is_early_stop = (temp != "invalid")
            
            if (is_last_sample_in_group and (is_last_turn or is_early_stop)):
                theoretical_takeovers = takeover_budget
                current_group_takeovers = 0
                start_sample_idx = problem_group_idx * repeats
                end_sample_idx = start_sample_idx + repeats

                for sample_idx in range(start_sample_idx, end_sample_idx):
                    if sample_idx in takeover_stats.get('sample_takeovers', {}):
                        current_group_takeovers += takeover_stats['sample_takeovers'][sample_idx]
                
                accuracy = 1 - abs(current_group_takeovers - theoretical_takeovers) / theoretical_takeovers
                print(f"Group {problem_group_idx} finish - theory={theoretical_takeovers}, practical={current_group_takeovers}次, accuracy={accuracy:.1%}", flush=True)

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
        "question": problem
    }
    
    output_filename = os.path.join(output_dir, f"problem_{idx:04d}.json")
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, indent=4)
    
    if takeover_stats:
        if problem_has_takeover:
            takeover_stats['problems_with_takeover'].add(problem_group_idx)
        else:
            takeover_stats['problems_without_takeover'].add(problem_group_idx)
            takeover_stats['no_takeover_details'].append({
                'problem_group_idx': problem_group_idx,
                'sample_idx': idx,
                'ppl_values': [entry.get('ppl', 0) for entry in history_log if entry.get('model') == 'eval_ppl'],
                'question': problem[:200] + '...' if len(problem) > 200 else problem
            })
        
    return ()


async def compute_score(results, answers, repeats, takeover_stats=None, takeover_budget=None):
    generated_ans = [ans for ans, _ in results]
    group = len(generated_ans) // repeats
    right = 0
        
    for i in range(group):
        start = i * repeats
        end = (i + 1) * repeats
        outputs = generated_ans[start:end]
        correct_answer = answers[start]
        
        matched = False
        for output in outputs:
            if output != "invalid" and output == correct_answer:
                matched = True
                break
        
        print(f"group {i}: right={correct_answer}, all={outputs}, match={matched}")
        
        if matched:
            right += 1

    accuracy = right / group if group > 0 else 0
    print(f"accuracy: {accuracy:.2%} ({right}/{group})")
    
    invalid_count = sum(1 for ans in generated_ans if ans == "invalid")

    if takeover_stats:
        total_takeovers = takeover_stats['total_takeovers']
        print(f"take over: {total_takeovers}")


async def main():
    parser = argparse.ArgumentParser(description="Run a multi-turn, multi-agent evaluation.")
    parser.add_argument("--small_model_name", type=str, required=True,
                        help="Name of the small model for generating responses.")
    parser.add_argument("--eval_model_name", type=str, required=True,
                        help="Name of the model to use for PPL evaluation.")
    parser.add_argument("--dataset_name", type=str, required=True,
                        help="Name of the dataset to use (e.g., gpqa, math500).")

    parser.add_argument("--turns", type=int, default=15,
                        help="Maximum number of turns for the multi-agent loop.")
    parser.add_argument("--small_model_max_tokens", type=int, default=500,
                        help="Maximum tokens for the small model's response.")
    parser.add_argument("--evalator_max_tokens", type=int, default=500,
                        help="Maximum tokens for the evaluation model's response.")
    parser.add_argument("--repeats", type=int, default=16,
                        help="Number of times to repeat each problem.")
    parser.add_argument("--small_model_port", type=int, default=51101,
                        help="Port for the small model server.")
    parser.add_argument("--eval_model_port", type=int, default=51100,
                        help="Port for the evaluation model server.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save the results and history.")

    parser.add_argument("--takeover_budget", type=int, default=10,
                        help="Global budget for evaluation model takeovers (default: 10)")
    parser.add_argument("--ppl_array_path", type=str, default=None,
                        help="Path to PPL array file (.npy) for percentile calculation")
    parser.add_argument("--percentile_threshold", type=float, default=0.5,
                        help="Percentile threshold for triggering takeover (default: 0.5, 50th percentile)")
    
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    global client_small, client_eval, small_model_name, eval_model_name, tokenizer, small_tokenizer, percentile_threshold
    small_model_name = args.small_model_name
    eval_model_name = args.eval_model_name
    percentile_threshold = args.percentile_threshold
    
    if args.ppl_array_path:
        if not load_ppl_array(args.ppl_array_path):
            print("ppl path error")

    client_small = httpx.AsyncClient(
        timeout=240.0,
        limits=httpx.Limits(max_connections=1000, max_keepalive_connections=1000)
    )
    client_eval = httpx.AsyncClient(
        timeout=240.0,
        limits=httpx.Limits(max_connections=1000, max_keepalive_connections=1000)
    )

    tokenizer = AutoTokenizer.from_pretrained(args.eval_model_name)
    tokenizer.use_default_system_prompt = True
    small_tokenizer = AutoTokenizer.from_pretrained(args.small_model_name)
    small_tokenizer.use_default_system_prompt = True

    context, answer = load_my_dataset(args.dataset_name, args.repeats)
    
    total_unique_problems = len(answer) // args.repeats
    total_samples = len(context)
    
    print(f"{total_unique_problems} problems, {total_samples} samples")
    
    takeover_stats = {
        'total_takeovers': 0,
        'problems_with_takeover': set(),
        'problems_without_takeover': set(),
        'no_takeover_details': [],
        'sample_takeovers': {}
    }

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
        print("all problems finish")
        all_results = []
        for idx in range(total_samples):
            filepath = os.path.join(args.output_dir, f"problem_{idx:04d}.json")
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                all_results.append((data['final_answer'], data['duration_seconds']))
        await compute_score(all_results, answer, args.repeats, takeover_stats, args.takeover_budget)
        return

    print(f"find {len(unique_problems_to_process)} groups. handling...")
    
    start_time = time.time()
    
    for unique_idx in sync_tqdm(unique_problems_to_process, desc="Processing problem groups"):
        print(f"Processing Problem Group {unique_idx}")
        
        start_sample_idx = unique_idx * args.repeats
        end_sample_idx = start_sample_idx + args.repeats
        
        tasks_to_run_for_group = []
        
        for sample_idx in range(start_sample_idx, end_sample_idx):
            if sample_idx not in processed_sample_indices:
                problem = context[sample_idx]
                task = asyncio.create_task(
                    process_single_problem(
                        problem,
                        args.small_model_max_tokens,
                        args.evalator_max_tokens,
                        args.turns,
                        sample_idx,
                        args.small_model_port,
                        args.eval_model_port,
                        args.output_dir,
                        args.repeats,
                        takeover_stats,
                        args.takeover_budget
                    )
                )
                tasks_to_run_for_group.append(task)
        
        if tasks_to_run_for_group:
            await tqdm.gather(*tasks_to_run_for_group, desc=f"Group {unique_idx} samples")
            if takeover_stats and args.takeover_budget is not None:
                current_group_takeovers = sum(
                    takeover_stats.get('sample_takeovers', {}).get(sample_idx, 0)
                    for sample_idx in range(start_sample_idx, end_sample_idx)
                )
                
                theoretical_total_takeovers = args.takeover_budget
                takeover_accuracy = 1 - abs(current_group_takeovers - theoretical_total_takeovers) / theoretical_total_takeovers
                
            
    end_time = time.time()
    print(f"time: {end_time - start_time:.3f} s")
    
    total_problems_processed = len(takeover_stats['problems_with_takeover']) + len(takeover_stats['problems_without_takeover'])
    print(f"Total problem groups processed: {total_problems_processed}")
    print(f"Problem groups with takeover: {len(takeover_stats['problems_with_takeover'])}")
    print(f"Problem groups without takeover: {len(takeover_stats['problems_without_takeover'])}")
    print(f"Total takeover count: {takeover_stats['total_takeovers']}")

    if takeover_stats['problems_without_takeover']:
        print(f"Problem groups without takeover: {sorted(takeover_stats['problems_without_takeover'])}")
            
        for detail in takeover_stats['no_takeover_details']:
            print(f"Problem group {detail['problem_group_idx']} (sample {detail['sample_idx']}):")
            print(f"Question: {detail['question']}")
            print(f"PPL values: {[f'{p:.4f}' for p in detail['ppl_values']]}")
            print(f"Average PPL: {sum(detail['ppl_values'])/len(detail['ppl_values']):.4f}")
            print("-" * 40)

    if total_problems_processed > 0:
        takeover_rate = len(takeover_stats['problems_with_takeover']) / total_problems_processed * 100
        print(f"Takeover rate by problem group: {takeover_rate:.1f}%")

    all_files_exist = True
    for idx in range(total_samples):
        filepath = os.path.join(args.output_dir, f"problem_{idx:04d}.json")
        if not os.path.exists(filepath):
            print(f"Error: required result file {filepath} is missing. Final score cannot be computed.")
            all_files_exist = False
            break
            
    if all_files_exist:
        all_results = []
        for idx in range(total_samples):
            filepath = os.path.join(args.output_dir, f"problem_{idx:04d}.json")
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                all_results.append((data['final_answer'], data['duration_seconds']))
        await compute_score(all_results, answer, args.repeats, takeover_stats, args.takeover_budget)
    else:
        print("Final score will not be computed due to missing result files. Please rerun the script to complete all tasks.")


    await client_small.aclose()
    await client_eval.aclose()

if __name__ == "__main__":
    asyncio.run(main())