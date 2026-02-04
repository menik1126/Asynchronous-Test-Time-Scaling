import argparse
import httpx
import asyncio
import random

import math
import time
import re
import os
import json
from tqdm.asyncio import tqdm
from tqdm import tqdm as sync_tqdm

from transformers import AutoTokenizer
from dataset import load_my_dataset
from async_agent import anyone_check

# --- [Original code for global variables and helper functions goes here, unchanged] ---

# Global variables for model clients and tokenizer
client_small = None
client_eval = None
semaphore = asyncio.Semaphore(8)

small_model_name = ""
eval_model_name = ""
tokenizer = None
small_tokenizer = None

# 预选接管位置
takeover_positions = {}


def build_debug_prompt():
    message = [
        {"role": "system", "content": "You are a creative and expressive assistant. Feel free to write anything you want, in any format or style."},
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


async def call_small_model(prompt, turn, max_tokens, idx, port, debug_mode=False):
    if debug_mode:
        messages = build_debug_prompt()  # 使用debug提示
    else:
        messages = (
            build_small_init_prompt(prompt[0]) if turn == 0 else build_small_inner_prompt(prompt[0], prompt[1])
        )
    
    global semaphore, client_small, small_model_name
    payload = {
        "model": small_model_name,
        "messages": messages,
        "temperature": 0.8,
        "max_tokens": max_tokens,
    }
    
    async with semaphore:
        resp = await client_small.post(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            json=payload,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


async def call_eval_model(prompt, max_tokens, idx, port):
    messages = build_eval_prompt_for_generate(prompt[0], prompt[1])
    global semaphore, client_eval, eval_model_name
    payload = {
        "model": eval_model_name,
        "messages": messages,
        "temperature": 0.8,
        "max_tokens": max_tokens,
    }

    async with semaphore:
        resp = await client_eval.post(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            json=payload,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]





async def extract_answer(history):
    answer = "invalid"
    temp = "\n\n".join([
        f"{history[i]}"
        for i in range(len(history))
    ])

    matches = re.findall(r"\\boxed\{(.*?)\}", temp)
    if matches:
        answer = matches[-1]
    else:
        pattern = re.compile(r"ANSWER:\s+([A-Z])", re.IGNORECASE)
        matches = pattern.findall(temp)
        if matches:
            answer = matches[-1]

    return answer


async def process_single_problem(problem, small_model_max_tokens, evalator_max_tokens, turns, idx, small_model_port, eval_model_port, output_dir, debug_mode=False, repeats=1):
    prompt = [problem, []]
    answer = "invalid"
    start_time = time.time()
    
    # 计算问题组索引
    problem_group_idx = idx // repeats
    
    history_log = []

    for turn in range(turns):
        print(f"📊 Problem Group {problem_group_idx} (Sample {idx}) - Turn {turn+1}/{turns}", flush=True)
        small_out = await call_small_model(prompt, turn, small_model_max_tokens, idx, small_model_port, debug_mode)
        print(f"🔹 小模型输出 (Turn {turn+1}): {small_out[:200]}{'...' if len(small_out) > 200 else ''}")
        history_log.append({"turn": turn, "model": "small", "output": small_out})
        prompt[1].append(small_out)

        if not small_out:
            print("Small model returned empty output.", flush=True)
            break

        # 使用预选位置控制异步接管
        global takeover_positions
        should_takeover = False
        
        # 检查当前样本是否在预选的接管位置中
        if turn in takeover_positions and idx in takeover_positions[turn]:
            should_takeover = True
            print(f"🎯 Turn {turn+1}: 预选接管触发! (Sample {idx} 在预选位置中)", flush=True)
        else:
            print(f"⏭️  Turn {turn+1}: 跳过接管 (Sample {idx} 不在预选位置中)", flush=True)
        
        if should_takeover:
            eval_out = await call_eval_model(prompt, evalator_max_tokens, idx, eval_model_port)
            print(f"🔸 大模型输出 (Turn {turn+1}): {eval_out[:200]}{'...' if len(eval_out) > 200 else ''}")
            history_log.append({"turn": turn, "model": "eval_generate", "output": eval_out})
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
        "question": problem
    }
    
    output_filename = os.path.join(output_dir, f"problem_{idx:04d}.json")
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, indent=4)
        
    # We don't need to return anything, as the result is already saved.
    return ()


async def compute_score(results, answers, repeats):
    generated_ans = [ans for ans, _ in results]
    group = len(generated_ans) // repeats
    right = 0
    for i in range(group):
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

    print(f"Accuracy: {right / group:.2%}")


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
    parser.add_argument("--debug_mode", action="store_true",
                        help="Enable debug mode with simple prompts")
    
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    global client_small, client_eval, small_model_name, eval_model_name, tokenizer, small_tokenizer
    small_model_name = args.small_model_name
    eval_model_name = args.eval_model_name

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
    
    # 初始化每个turn的独立接管预算
    global takeover_positions
    max_turns = args.turns
    random.seed(42)  # 设置随机种子确保可重现
    for turn_num in range(max_turns):
        takeover_positions[turn_num] = set(random.sample(range(total_samples), min(args.takeover_budget, total_samples)))
        print(f"🎲 Turn {turn_num+1} 预选接管位置: {sorted(takeover_positions[turn_num])}")
    

    
    # 最终修正的、正确的断点恢复和分组处理逻辑
    
    # 第1步：找出所有已完成的单个采样任务的索引
    processed_sample_indices = set()
    for filename in os.listdir(args.output_dir):
        if filename.startswith("problem_") and filename.endswith(".json"):
            try:
                sample_idx = int(filename.replace("problem_", "").replace(".json", ""))
                processed_sample_indices.add(sample_idx)
            except ValueError:
                continue

    # 第2步：识别所有需要处理的唯一问题组
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
        print("所有问题都已完成处理。无需运行新任务。")
        all_results = []
        for idx in range(total_samples):
            filepath = os.path.join(args.output_dir, f"problem_{idx:04d}.json")
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                all_results.append((data['final_answer'], data['duration_seconds']))
        await compute_score(all_results, answer, args.repeats)
        return

    print(f"找到 {len(unique_problems_to_process)} 个需要处理的问题组。正在恢复...")
    
    start_time = time.time()
    
    # 第3步：按"问题组"为单位，只处理组内未完成的采样任务
    for unique_idx in sync_tqdm(unique_problems_to_process, desc="Processing problem groups"):
        print(f"🔄 Processing Problem Group {unique_idx}")
        
        tasks_to_run_for_group = []
        start_sample_idx = unique_idx * args.repeats
        end_sample_idx = start_sample_idx + args.repeats
        
        for sample_idx in range(start_sample_idx, end_sample_idx):
            # 检查这个采样是否已经完成
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
                        args.debug_mode,
                        args.repeats
                    )
                )
                tasks_to_run_for_group.append(task)
        
        # 在这里执行本组内的所有任务，并等待它们全部完成
        if tasks_to_run_for_group:
            await tqdm.gather(*tasks_to_run_for_group, desc=f"Group {unique_idx} samples")
            # 注意：这里不需要收集返回值，因为保存操作在任务内部已经完成
            
    end_time = time.time()
    print(f"耗时: {end_time - start_time:.3f} s")
    
    # 最后，在所有任务都完成之后，我们才去计算最终分数
    print("\n尝试计算最终分数...")
    
    all_files_exist = True
    for idx in range(total_samples):
        filepath = os.path.join(args.output_dir, f"problem_{idx:04d}.json")
        if not os.path.exists(filepath):
            print(f"错误：所需结果文件 {filepath} 缺失。无法计算最终分数。")
            all_files_exist = False
            break
            
    if all_files_exist:
        all_results = []
        for idx in range(total_samples):
            filepath = os.path.join(args.output_dir, f"problem_{idx:04d}.json")
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                all_results.append((data['final_answer'], data['duration_seconds']))
        await compute_score(all_results, answer, args.repeats)
    else:
        print("由于结果文件缺失，将不计算最终分数。请重新运行脚本以完成所有任务。")

    await client_small.aclose()
    await client_eval.aclose()

if __name__ == "__main__":
    asyncio.run(main())