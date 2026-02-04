import asyncio
import argparse
import json
import os
import time
import httpx
import random
import re
from transformers import AutoTokenizer
from tqdm import tqdm
from dataset import load_my_dataset
from async_agent import anyone_check

# Global variables (unchanged from your code)
client_small = None
client_eval = None
small_model_name = None
eval_model_name = None
tokenizer = None
small_tokenizer = None

def build_debug_prompt(problem, turn):
    """构建debug模式的prompt"""
    return f"DEBUG MODE - Turn {turn+1}: {problem[:100]}..."

async def call_small_model(prompt, turn, max_tokens, idx, port, debug_mode=False):
    """调用小模型"""
    global client_small, small_model_name
    
    if debug_mode:
        prompt_text = build_debug_prompt(prompt[0], turn)
    else:
        # 构建完整的对话历史
        conversation = [{"role": "user", "content": prompt[0]}]
        for i, response in enumerate(prompt[1]):
            conversation.append({"role": "assistant", "content": response})
        
        prompt_text = json.dumps(conversation, ensure_ascii=False)
    
    try:
        response = await client_small.post(
            f"http://localhost:{port}/generate",
            json={
                "model": small_model_name,
                "prompt": prompt_text,
                "max_tokens": max_tokens,
                "temperature": 0.7,
                "stop": ["</s>", "Human:", "Assistant:"]
            },
            timeout=60.0
        )
        result = response.json()
        output = result.get("text", [""])[0] if result.get("text") else ""
        print(f"🔹 小模型输出 (Turn {turn+1}): {output[:200]}{'...' if len(output) > 200 else ''}")
        return output
    except Exception as e:
        print(f"小模型调用失败: {e}")
        return ""

async def call_eval_model(prompt, max_tokens, idx, port):
    """调用大模型"""
    global client_eval, eval_model_name
    
    # 构建完整的对话历史
    conversation = [{"role": "user", "content": prompt[0]}]
    for i, response in enumerate(prompt[1]):
        conversation.append({"role": "assistant", "content": response})
    
    prompt_text = json.dumps(conversation, ensure_ascii=False)
    
    try:
        response = await client_eval.post(
            f"http://localhost:{port}/generate",
            json={
                "model": eval_model_name,
                "prompt": prompt_text,
                "max_tokens": max_tokens,
                "temperature": 0.7,
                "stop": ["</s>", "Human:", "Assistant:"]
            },
            timeout=60.0
        )
        result = response.json()
        output = result.get("text", [""])[0] if result.get("text") else ""
        print(f"🔸 大模型输出: {output[:200]}{'...' if len(output) > 200 else ''}")
        return output
    except Exception as e:
        print(f"大模型调用失败: {e}")
        return ""

async def process_single_problem(problem, small_model_max_tokens, evalator_max_tokens, turns, idx, small_model_port, eval_model_port, output_dir, debug_mode=False, repeats=1):
    # This function is not used in the new async flow.
    pass

async def simulate_async_evaluation_and_takeover_decision(samples_data, takeover_budget):
    """
    异步评分，同步决策。
    """
    print(f"\n🔄 Collecting scores and making takeover decisions...")
    
    # Create evaluation tasks for all samples and run them concurrently.
    evaluation_tasks = []
    for i, data in enumerate(samples_data):
        # The prompt for evaluation could be more sophisticated. Here we use the small model's output.
        prompt_for_eval = [data['problem'], [data['output']]]
        evaluation_tasks.append(asyncio.create_task(
            call_eval_model(prompt_for_eval, 50, data['sample_idx'], 51100) # Assuming a simple, fast eval model call
        ))
    
    # Wait for all evaluation tasks to complete.
    eval_outputs = await asyncio.gather(*evaluation_tasks)
    
    # Create a list of scores from the evaluation outputs (simulated).
    scores = []
    for i, eval_out in enumerate(eval_outputs):
        # Mock scoring logic. A real implementation would parse the eval_out to get a score.
        score = len(eval_out) + random.randint(0, 100)
        scores.append((i, score))
    
    # Sort scores to make a synchronous decision.
    scores.sort(key=lambda x: x[1], reverse=True)
    
    print(f"📊 Evaluation results collected:")
    for i, (sample_idx, score) in enumerate(scores):
        print(f"   Sample {samples_data[sample_idx]['sample_idx']}: Score {score}")
    
    # Select the top samples for takeover.
    takeover_samples = [samples_data[idx]['sample_idx'] for idx, _ in scores[:takeover_budget]]
    print(f"🎯 Samples selected for takeover: {takeover_samples}")
    
    return takeover_samples

async def generate_and_evaluate_task(conv_data, turn, small_model_max_tokens, evalator_max_tokens, sample_idx, small_model_port, eval_model_port, debug_mode):
    """
    Combined task to generate small model output and then evaluate it.
    This runs concurrently for each problem.
    """
    # Step 1: Call small model
    small_out = await call_small_model([conv_data['problem'], conv_data['history']], turn, small_model_max_tokens, sample_idx, small_model_port, debug_mode)
    
    # Step 2: Call evaluation model on the small model's output
    eval_out = await call_eval_model([conv_data['problem'], [small_out]], evalator_max_tokens, sample_idx, eval_model_port)
    
    # Return both the small model's output and the evaluation result
    return {
        'sample_idx': sample_idx,
        'problem': conv_data['problem'],
        'output': small_out,
        'turn': turn,
        'eval_output': eval_out,
        'score': len(eval_out) + random.randint(0, 100) # Mock score based on evaluation output
    }

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

async def process_problem_group_async(problems, small_model_max_tokens, evalator_max_tokens, turns, start_idx, small_model_port, eval_model_port, output_dir, debug_mode, repeats, takeover_budget):
    """
    异步处理问题组。小模型生成和打分并发，打分结果收集后同步决策。
    """
    print(f"🔄 Processing Problem Group {start_idx // repeats}")
    
    all_conversations = [
        {"problem": problem, "history": []} for problem in problems
    ]
    
    all_samples_data = []

    for turn in range(turns):
        print(f"\n🔄 第 {turn+1} 轮开始...")
        
        # Concurrently run the combined generation and evaluation task for all problems.
        combined_tasks = []
        for i, conv_data in enumerate(all_conversations):
            sample_idx = start_idx + i
            combined_tasks.append(
                asyncio.create_task(
                    generate_and_evaluate_task(conv_data, turn, small_model_max_tokens, evalator_max_tokens, sample_idx, small_model_port, eval_model_port, debug_mode)
                )
            )
        
        # Wait for all combined tasks to complete.
        turn_results_with_scores = await asyncio.gather(*combined_tasks)
        
        # Sort results based on the score to decide which samples to take over.
        turn_results_with_scores.sort(key=lambda x: x['score'], reverse=True)
        
        print(f"📊 Evaluation results collected:")
        for result in turn_results_with_scores:
            print(f"   Sample {result['sample_idx']}: Score {result['score']}")
            
        # Select the top samples for takeover.
        takeover_samples_data = turn_results_with_scores[:takeover_budget]
        takeover_samples = [data['sample_idx'] for data in takeover_samples_data]
        print(f"🎯 Samples selected for takeover: {takeover_samples}")
        
        # Update conversation history with the results from this turn.
        turn_results = []
        for i, result in enumerate(turn_results_with_scores):
            # Find the original problem's index
            problem_index_in_group = result['sample_idx'] - start_idx
            
            # Append small model output
            all_conversations[problem_index_in_group]['history'].append(result['output'])

            # If it's a takeover sample, append the eval output as well
            if result['sample_idx'] in takeover_samples:
                all_conversations[problem_index_in_group]['history'].append(result['eval_output'])
                print(f"🔸 样本 {result['sample_idx']} 被接管，大模型输出: {result['eval_output'][:100]}...")

            # Store results for final output
            turn_results.append({
                'sample_idx': result['sample_idx'],
                'problem': result['problem'],
                'output': result['output'],
                'turn': turn,
                'eval_output': result['eval_output'] if result['sample_idx'] in takeover_samples else None
            })
        
        all_samples_data.append(turn_results)

    # 保存最终结果 (unchanged from your code)
    start_time = time.time()
    for i, problem in enumerate(problems):
        sample_idx = start_idx + i
        
        final_conversation = [
            all_conversations[i]['problem']
        ] + all_conversations[i]['history']
        
        answer = await extract_answer(final_conversation[1:])
        
        end_time = time.time()
        duration = end_time - start_time
        
        result_data = {
            "problem_index": sample_idx,
            "final_answer": answer,
            "duration_seconds": duration,
            "full_history": final_conversation,
            "question": problem
        }
        
        output_filename = os.path.join(output_dir, f"problem_{sample_idx:04d}.json")
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=4, ensure_ascii=False)

async def compute_score(results, answers, repeats):
    # This function is unchanged
    generated_ans = [ans for ans, _ in results]
    group = len(generated_ans) // repeats
    right = 0
    for i in range(group):
        start = i * repeats
        end = start + repeats
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
    parser = argparse.ArgumentParser(description="Run a synchronous multi-turn, multi-agent evaluation.")
    parser.add_argument("--small_model_name", type=str, required=True,
                         help="Name of the small model for generating responses.")
    parser.add_argument("--eval_model_name", type=str, required=True,
                         help="Name of the model to use for evaluation.")
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
    parser.add_argument("--takeover_budget", type=int, default=5,
                         help="Number of samples to take over per turn.")
    parser.add_argument("--debug_mode", action="store_true",
                         help="Enable debug mode for simplified testing.")

    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 初始化HTTP客户端
    global client_small, client_eval, small_model_name, eval_model_name, tokenizer, small_tokenizer
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
    
    print(f"🎯 异步模式：小模型生成和打分并发，打分结果收集后同步决策")
    print(f"🎯 接管预算: {args.takeover_budget} 个样本/轮")
    print(f"🎯 总问题组数: {total_unique_problems}")
    print(f"🎯 总样本数: {total_samples}")
    
    # 处理每个问题组
    for unique_idx in tqdm(range(total_unique_problems), desc="Processing problem groups"):
        start_sample_idx = unique_idx * args.repeats
        end_sample_idx = start_sample_idx + args.repeats
        
        # 获取当前问题组的所有问题
        group_problems = context[start_sample_idx:end_sample_idx]
        
        # 异步处理问题组
        await process_problem_group_async(
            group_problems,
            args.small_model_max_tokens,
            args.evalator_max_tokens,
            args.turns,
            start_sample_idx,
            args.small_model_port,
            args.eval_model_port,
            args.output_dir,
            args.debug_mode,
            args.repeats,
            args.takeover_budget
        )
    
    # 计算最终分数
    print("\n尝试计算最终分数...")
    all_results = []
    for idx in range(total_samples):
        filepath = os.path.join(args.output_dir, f"problem_{idx:04d}.json")
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                all_results.append((data['final_answer'], data.get('duration_seconds', 0)))
        else:
            print(f"警告：结果文件 {filepath} 缺失")
            all_results.append(("invalid", 0))
    
    await compute_score(all_results, answer, args.repeats)
    
    await client_small.aclose()
    await client_eval.aclose()

if __name__ == "__main__":
    asyncio.run(main())