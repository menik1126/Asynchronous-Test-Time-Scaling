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
from agent import anyone_check  # 修正导入

# --- [Original code for global variables and helper functions goes here, unchanged] ---

# Global variables for model clients and tokenizer
client_small = None
client_eval = None
semaphore = asyncio.Semaphore(4)  # 减少并发数，降低服务器压力

small_model_name = ""
eval_model_name = ""
tokenizer = None
small_tokenizer = None

# PPL数组数据（用于百分位数计算）
ppl_array = None
percentile_threshold = 0.5  # 百分位数阈值，默认50%分位数


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
    """加载PPL数组数据文件"""
    global ppl_array
    if ppl_array_path and os.path.exists(ppl_array_path):
        ppl_array = np.load(ppl_array_path)
        print(f"✅ 成功加载PPL数组数据: {ppl_array.shape}")
        return True
    else:
        print(f"❌ PPL数组文件不存在: {ppl_array_path}")
        return False


async def call_eval_model_ppl(prompt, idx, port):
    """
    Asynchronously calls the evaluation model to get the perplexity (PPL).
    """
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
    max_retries = 3
    for attempt in range(max_retries):
        try:
            async with semaphore:
                resp = await client_eval.post(
                    f"http://127.0.0.1:{port}/generate",
                    json=payload,
                    timeout=60.0  # 增加超时时间
                )
                resp.raise_for_status()
                data = resp.json()
                input_token_logprobs = data['meta_info']['input_token_logprobs'][1:]
                logprobs = [entry[0] for entry in input_token_logprobs if entry[0] is not None]
                #print(f"🔍 PPL in here: {logprobs}")
                if not logprobs:
                    print(f"No log probabilities returned for problem: {prompt[0]}", flush=True)
                    return 0
                
                avg_neg_logprob = -sum(logprobs) / len(logprobs)
                
                return math.exp(avg_neg_logprob)
        except (httpx.ReadError, httpx.ConnectError, httpx.TimeoutException) as e:
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 2  # 指数退避
                print(f"⚠️  PPL计算失败 (尝试 {attempt + 1}/{max_retries}): {e}. 等待 {wait_time}s 后重试...", flush=True)
                await asyncio.sleep(wait_time)
            else:
                print(f"❌ PPL计算最终失败 (样本 {idx}): {e}", flush=True)
                return 0  # 返回默认值
        except Exception as e:
            print(f"❌ PPL计算出现未知错误 (样本 {idx}): {e}", flush=True)
            return 0


def should_takeover_based_on_percentile(ppl_value):
    """基于PPL百分位数决定是否接管"""
    global ppl_array, percentile_threshold
    
    if ppl_array is None:
        return False
    rank = np.sum(ppl_array < ppl_value)
    # 添加调试信息
    if ppl_array is not None:
        min_ppl = np.min(ppl_array)
        max_ppl = np.max(ppl_array)
        print(f"🔍 PPL对比: 当前={ppl_value:.4f}, 历史范围=[{min_ppl:.4f}, {max_ppl:.4f}], 排名={rank}/{len(ppl_array)}")
    
    # 计算当前PPL在历史分布中的百分位数
    
    percentile = rank / len(ppl_array)
    
    should_takeover = percentile >= percentile_threshold
    
    if should_takeover:
        print(f"🎯 百分位数触发接管: PPL={ppl_value:.4f}, 百分位数={percentile:.3f} >= {percentile_threshold}")
    
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
        "stop": ["\\boxed{"],  # 添加停止条件，与历史PPL计算保持一致
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
                print(f"⚠️  小模型调用失败 (尝试 {attempt + 1}/{max_retries}): {e}. 等待 {wait_time}s 后重试...", flush=True)
                await asyncio.sleep(wait_time)
            else:
                print(f"❌ 小模型调用最终失败 (样本 {idx}): {e}", flush=True)
                return ""  # 返回空字符串
        except Exception as e:
            print(f"❌ 小模型调用出现未知错误 (样本 {idx}): {e}", flush=True)
            return ""


async def call_eval_model(prompt, max_tokens, idx, port):
    messages = build_eval_prompt_for_generate(prompt[0], prompt[1])
    global semaphore, client_eval, eval_model_name
    payload = {
        "model": eval_model_name,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": max_tokens,
        "stop": ["\\boxed{"],  # 添加停止条件
    }

    max_retries = 3
    for attempt in range(max_retries):
        try:
            async with semaphore:
                resp = await client_eval.post(
                    f"http://127.0.0.1:{port}/v1/chat/completions",
                    json=payload,
                    timeout=60.0
                )
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"]
        except (httpx.ReadError, httpx.ConnectError, httpx.TimeoutException) as e:
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 2
                print(f"⚠️  评估模型调用失败 (尝试 {attempt + 1}/{max_retries}): {e}. 等待 {wait_time}s 后重试...", flush=True)
                await asyncio.sleep(wait_time)
            else:
                print(f"❌ 评估模型调用最终失败 (样本 {idx}): {e}", flush=True)
                return ""  # 返回空字符串
        except Exception as e:
            print(f"❌ 评估模型调用出现未知错误 (样本 {idx}): {e}", flush=True)
            return ""


async def extract_answer(history):
    answer = "invalid"
    temp = "\n\n".join([
        f"{history[i]}"
        for i in range(len(history))
    ])

    # 尝试多种答案格式
    # 1. \boxed{} 格式
    matches = re.findall(r"\\boxed\{(.*?)\}", temp)
    if matches:
        answer = matches[-1].strip()
        return answer
    
    # 2. ANSWER: 格式
    pattern = re.compile(r"ANSWER:\s*([A-Z])", re.IGNORECASE)
    matches = pattern.findall(temp)
    if matches:
        answer = matches[-1].strip()
        return answer
    
    # 3. 查找常见的答案模式
    patterns = [
        r"answer[:\s]*([A-Z])",  # answer: A
        r"the answer is[:\s]*([A-Z])",  # the answer is A
        r"final answer[:\s]*([A-Z])",  # final answer A
        r"option[:\s]*([A-Z])",  # option A
        r"choice[:\s]*([A-Z])",  # choice A
    ]
    
    for pattern_str in patterns:
        pattern = re.compile(pattern_str, re.IGNORECASE)
        matches = pattern.findall(temp)
        if matches:
            answer = matches[-1].strip()
            return answer
    
    # 4. 如果都没有找到，返回最后一个非空行的内容（作为调试信息）
    lines = temp.strip().split('\n')
    for line in reversed(lines):
        line = line.strip()
        if line and len(line) <= 10:  # 假设答案不会太长
            return line
    
    return answer


async def process_single_problem(problem, small_model_max_tokens, evalator_max_tokens, turns, idx, small_model_port, eval_model_port, output_dir, repeats=1, takeover_stats=None, takeover_budget=None):
    prompt = [problem, []]
    answer = "invalid"
    start_time = time.time()
    
    # 计算问题组索引
    problem_group_idx = idx // repeats
    
    history_log = []
    problem_has_takeover = False
    temp = None
    for turn in range(turns):
        print(f"📊 Problem Group {problem_group_idx} (Sample {idx}) - Turn {turn+1}/{turns}", flush=True)
        small_out = await call_small_model(prompt, turn, small_model_max_tokens, idx, small_model_port)
        print(f"🔹 小模型输出 (Turn {turn+1}): {small_out[:200]}{'...' if len(small_out) > 200 else ''}")
        history_log.append({"turn": turn, "model": "small", "output": small_out})
        prompt[1].append(small_out)

        if not small_out:
            print("Small model returned empty output.", flush=True)
            break

        # 实时计算PPL并基于百分位数决定是否接管
        ppl = await call_eval_model_ppl(prompt, idx, eval_model_port)
        should_takeover = should_takeover_based_on_percentile(ppl)
        
        history_log.append({"turn": turn, "model": "eval_ppl", "ppl": ppl, "should_takeover": int(should_takeover)})
        
        if should_takeover:
            print(f"🎯 Turn {turn+1}: 百分位数触发接管! (Sample {idx}, PPL={ppl:.4f})", flush=True)
            eval_out = await call_eval_model(prompt, evalator_max_tokens, idx, eval_model_port)
            print(f"🔸 大模型输出 (Turn {turn+1}): {eval_out[:200]}{'...' if len(eval_out) > 200 else ''}")
            history_log.append({"turn": turn, "model": "eval_generate", "output": eval_out})
            prompt[1].append(eval_out)
            problem_has_takeover = True
            if takeover_stats:
                takeover_stats['total_takeovers'] += 1
                # 记录当前样本的接管次数
                if idx not in takeover_stats['sample_takeovers']:
                    takeover_stats['sample_takeovers'][idx] = 0
                takeover_stats['sample_takeovers'][idx] += 1
        else:
            print(f"⏭️  Turn {turn+1}: 跳过接管 (Sample {idx}, PPL={ppl:.4f})", flush=True)
        
        # 计算并打印接管率比较（方法二）- 仅在group内最后一个样本的最后一个turn打印
        if takeover_stats and takeover_budget is not None:
            # 检查是否是group内最后一个样本的最后一个turn
            is_last_sample_in_group = (idx % repeats == repeats - 1)  # 最后一个样本
            is_last_turn = (turn == turns - 1)  # 最后一个turn
            is_early_stop = (temp != "invalid")  # 提前停止
            
            if (is_last_sample_in_group and (is_last_turn or is_early_stop)):
                # 方法二：基于样本数量计算
                theoretical_takeovers = takeover_budget  # 理论接管次数
                # 计算当前group的实际接管次数
                current_group_takeovers = 0
                start_sample_idx = problem_group_idx * repeats
                end_sample_idx = start_sample_idx + repeats
                
                # 从takeover_stats中统计当前group的接管次数
                for sample_idx in range(start_sample_idx, end_sample_idx):
                    if sample_idx in takeover_stats.get('sample_takeovers', {}):
                        current_group_takeovers += takeover_stats['sample_takeovers'][sample_idx]
                
                accuracy = 1 - abs(current_group_takeovers - theoretical_takeovers) / theoretical_takeovers
                print(f"📈 Group {problem_group_idx} 完成 - 接管率比较: 理论={theoretical_takeovers}次, 实际={current_group_takeovers}次, 准确率={accuracy:.1%}", flush=True)

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
    
    # 更新接管统计
    if takeover_stats:
        if problem_has_takeover:
            takeover_stats['problems_with_takeover'].add(problem_group_idx)
        else:
            takeover_stats['problems_without_takeover'].add(problem_group_idx)
            # 记录没有接管的问题详情
            takeover_stats['no_takeover_details'].append({
                'problem_group_idx': problem_group_idx,
                'sample_idx': idx,
                'ppl_values': [entry.get('ppl', 0) for entry in history_log if entry.get('model') == 'eval_ppl'],
                'question': problem[:200] + '...' if len(problem) > 200 else problem
            })
        
    # We don't need to return anything, as the result is already saved.
    return ()


async def compute_score(results, answers, repeats, takeover_stats=None, takeover_budget=None):
    generated_ans = [ans for ans, _ in results]
    group = len(generated_ans) // repeats
    right = 0
    
    print(f"\n📊 答案统计:")
    print(f"总问题组数: {group}")
    print(f"重复次数: {repeats}")
    
    for i in range(group):
        start = i * repeats
        end = (i + 1) * repeats
        outputs = generated_ans[start:end]
        correct_answer = answers[start]
        
        # 简单的答案匹配（避免使用外部API）
        matched = False
        for output in outputs:
            if output != "invalid" and output == correct_answer:
                matched = True
                break
        
        print(f"问题组 {i}: 正确答案={correct_answer}, 生成答案={outputs}, 匹配={matched}")
        
        if matched:
            right += 1

    accuracy = right / group if group > 0 else 0
    print(f"\n🎯 最终准确率: {accuracy:.2%} ({right}/{group})")
    
    # 统计无效答案
    invalid_count = sum(1 for ans in generated_ans if ans == "invalid")
    print(f"⚠️  无效答案数量: {invalid_count}/{len(generated_ans)} ({invalid_count/len(generated_ans)*100:.1f}%)")
    
    # 显示最终的总接管数量
    if takeover_stats:
        total_takeovers = takeover_stats['total_takeovers']
        print(f"📊 最终总接管数量: {total_takeovers}")


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
    
    # 加载PPL数组数据
    if args.ppl_array_path:
        if not load_ppl_array(args.ppl_array_path):
            print("⚠️  无法加载PPL数组文件，将使用随机接管策略")
    else:
        print("⚠️  未提供PPL数组文件路径，将使用随机接管策略")

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
    
    print(f"📊 数据集信息: {total_unique_problems}个问题, {total_samples}个样本")
    if ppl_array is not None:
        print(f"📊 PPL数组信息: {ppl_array.shape}, 百分位数阈值: {percentile_threshold}")
        # 统计百分位数分布
        expected_takeover_count = int(len(ppl_array) * percentile_threshold)
        print(f"📊 预期接管样本数: {expected_takeover_count}/{len(ppl_array)} ({percentile_threshold*100:.1f}%分位数)")
    
    # 添加接管统计变量
    takeover_stats = {
        'total_takeovers': 0,
        'problems_with_takeover': set(),
        'problems_without_takeover': set(),
        'no_takeover_details': [],
        'sample_takeovers': {}  # 新增：每个样本的接管次数
    }
    

    
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
        await compute_score(all_results, answer, args.repeats, takeover_stats, args.takeover_budget)
        return

    print(f"找到 {len(unique_problems_to_process)} 个需要处理的问题组。正在恢复...")
    
    start_time = time.time()
    
    # 第3步：按"问题组"为单位，只处理组内未完成的采样任务
    for unique_idx in sync_tqdm(unique_problems_to_process, desc="Processing problem groups"):
        print(f"🔄 Processing Problem Group {unique_idx}")
        
        # 计算当前问题组的样本范围
        start_sample_idx = unique_idx * args.repeats
        end_sample_idx = start_sample_idx + args.repeats
        
        tasks_to_run_for_group = []
        
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
                        args.repeats,
                        takeover_stats,
                        args.takeover_budget
                    )
                )
                tasks_to_run_for_group.append(task)
        
        # 在这里执行本组内的所有任务，并等待它们全部完成
        if tasks_to_run_for_group:
            await tqdm.gather(*tasks_to_run_for_group, desc=f"Group {unique_idx} samples")
            # 注意：这里不需要收集返回值，因为保存操作在任务内部已经完成
            
            # 每个问题组完成后计算接管率准确率
            if takeover_stats and args.takeover_budget is not None:
                # 计算当前问题组的接管次数
                current_group_takeovers = sum(
                    takeover_stats.get('sample_takeovers', {}).get(sample_idx, 0)
                    for sample_idx in range(start_sample_idx, end_sample_idx)
                )
                
                theoretical_total_takeovers = args.takeover_budget
                takeover_accuracy = 1 - abs(current_group_takeovers - theoretical_total_takeovers) / theoretical_total_takeovers
                
                print(f"\n📈 问题组 {unique_idx} 完成后的接管率准确率统计:")
                print(f"理论总接管次数: {theoretical_total_takeovers} (预算: {args.takeover_budget})")
                print(f"当前问题组实际接管次数: {current_group_takeovers}")
                print(f"接管率准确率: {takeover_accuracy:.1%}")
                print(f"当前问题组样本数: {args.repeats}")
                print("-" * 50)
            
    end_time = time.time()
    print(f"耗时: {end_time - start_time:.3f} s")
    
    # 打印接管统计和没有被接管的部分
    print("\n" + "="*60)
    print("📊 接管情况统计")
    print("="*60)
    
    total_problems_processed = len(takeover_stats['problems_with_takeover']) + len(takeover_stats['problems_without_takeover'])
    print(f"总处理问题组数: {total_problems_processed}")
    print(f"有接管的问题组: {len(takeover_stats['problems_with_takeover'])}")
    print(f"无接管的问题组: {len(takeover_stats['problems_without_takeover'])}")
    print(f"总接管次数: {takeover_stats['total_takeovers']}")
    
    if takeover_stats['problems_without_takeover']:
        print(f"\n❌ 没有被接管的问题组: {sorted(takeover_stats['problems_without_takeover'])}")
        
        print("\n" + "="*60)
        print("🔍 没有被接管的详细情况")
        print("="*60)
        
        for detail in takeover_stats['no_takeover_details']:
            print(f"\n问题组 {detail['problem_group_idx']} (样本 {detail['sample_idx']}):")
            print(f"问题: {detail['question']}")
            print(f"PPL值: {[f'{p:.4f}' for p in detail['ppl_values']]}")
            print(f"平均PPL: {sum(detail['ppl_values'])/len(detail['ppl_values']):.4f}")
            print("-" * 40)
    
    # 计算接管率
    if total_problems_processed > 0:
        takeover_rate = len(takeover_stats['problems_with_takeover']) / total_problems_processed * 100
        print(f"\n📈 问题组接管率: {takeover_rate:.1f}%")
    
    print("="*60)
    
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
        await compute_score(all_results, answer, args.repeats, takeover_stats, args.takeover_budget)
    else:
        print("由于结果文件缺失，将不计算最终分数。请重新运行脚本以完成所有任务。")

    await client_small.aclose()
    await client_eval.aclose()

if __name__ == "__main__":
    asyncio.run(main())