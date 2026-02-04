import os
import json
import argparse
import httpx
import asyncio
import numpy as np
import math
import time
import re
from tqdm.asyncio import tqdm
from tqdm import tqdm as sync_tqdm

from transformers import AutoTokenizer
from dataset import load_my_dataset
from async_agent import anyone_check


# --- Global variables for model clients and tokenizer ---
client_small = None
semaphore = None  # 将在main函数中初始化


def build_debug_prompt():
    """
    Builds a debug prompt for testing purposes.
    """
    message = [
        {"role": "system", "content": "You are a creative and expressive assistant. Feel free to write anything you want, in any format or style."},
        {"role": "user", "content": "Go ahead and write freely. No need to stop."},
    ]
    return message


def build_question(question):
    """
    Constructs the prompt for the small model based on the question type.
    """
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
    """
    Constructs the chain-of-thought prompt from the history.
    """
    return "\n\n".join([f"{h}" for h in history])


def build_small_init_prompt(question):
    """
    Builds the initial prompt for the small model.
    """
    return [
        {"role": "system", "content": "You are a math expert."},
        {"role": "user", "content": build_question(question)}
    ]


def build_small_inner_prompt(question, history):
    """
    Builds the inner turn prompt for the small model.
    """
    return [
        {"role": "user", "content": build_question(question)},
        {"role": "assistant", "content": build_cot(history)}
    ]


async def call_small_model(prompt, turn, max_tokens, idx, small_model_name, small_tokenizer, sglang_port):
    """
    Asynchronously calls the small model via the OpenAI-compatible API.
    """
    messages = (
        build_small_init_prompt(prompt[0]) if turn == 0 else build_small_inner_prompt(prompt[0], prompt[1])
    )

    global semaphore, client_small
    
    # 检查semaphore是否已初始化
    if semaphore is None:
        raise RuntimeError("Semaphore not initialized. Make sure main() function has been called.")
    
    payload = {
        "model": small_model_name,
        "messages": messages,
        "temperature": 0.6,
        "max_tokens": max_tokens,
    }

    async with semaphore:
        resp = await client_small.post(
            f"http://127.0.0.1:{sglang_port}/v1/chat/completions",
            json=payload,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


async def extract_answer(history):
    """
    Extracts the final answer from the model's history.
    """
    answer = "invalid"
    temp = "\n\n".join([
        f"{history[i]}"
        for i in range(len(history))
    ])

    # First try to find boxed answers (primary pattern)
    matches = re.findall(r"\\boxed\{(.*?)\}", temp)
    if matches:
        answer = matches[-1]
        return answer
    
    # Try to find "Final Answer:" or similar patterns (numerical answers)
    final_answer_patterns = [
        r"Final Answer[:\s]*\\boxed\{(.*?)\}",
        r"Final Answer[:\s]*([0-9]+)",
        r"Therefore[,\s]*the answer is[:\s]*([0-9]+)",
        r"So[,\s]*the answer is[:\s]*([0-9]+)",
        r"The answer is[:\s]*([0-9]+)",
        r"\\boxed\{([0-9]+)\}",  # Additional boxed pattern for numbers only
    ]
    
    for pattern in final_answer_patterns:
        matches = re.findall(pattern, temp, re.IGNORECASE)
        if matches:
            answer = matches[-1]
            return answer
    
    # Try to find multiple choice answers (A, B, C, D) - but be more specific
    # Only match if it's clearly formatted as a final answer
    choice_patterns = [
        r"Final Answer[:\s]*([A-D])\b",
        r"Therefore[,\s]*the answer is[:\s]*([A-D])\b", 
        r"So[,\s]*the answer is[:\s]*([A-D])\b",
        r"The answer is[:\s]*([A-D])\b",
        r"\\boxed\{([A-D])\}",  # Boxed choice answers
    ]
    
    for pattern in choice_patterns:
        matches = re.findall(pattern, temp, re.IGNORECASE)
        if matches:
            answer = matches[-1].upper()
            return answer
    
    # Last resort: look for standalone numbers at the end of text
    # Only match if it's clearly a final numerical answer
    end_number_pattern = r"(?:answer|result)[:\s]*([0-9]+)\s*\.?\s*$"
    matches = re.findall(end_number_pattern, temp, re.IGNORECASE | re.MULTILINE)
    if matches:
        answer = matches[-1]
        return answer

    return answer


async def process_single_problem(problem, small_model_max_tokens, turns, idx, small_model_name, small_tokenizer, output_dir, sglang_port):
    """
    Processes a single problem and saves the results.
    """
    prompt = [problem, []]
    answer = "invalid"
    start_time = time.time()
    
    history_log = []

    for turn in range(turns):
        small_out = await call_small_model(prompt, turn, small_model_max_tokens, idx, small_model_name, small_tokenizer, sglang_port)
        history_log.append({"turn": turn, "model": "small", "output": small_out})
        prompt[1].append(small_out)

        if not small_out:
            print(f"Small model returned empty output for problem", flush=True)
            print(f"history: {prompt[1]}", flush=True)
            break

        temp = await extract_answer(prompt[1])

        if temp != "invalid":
            answer = temp
            print("early stop")
            break

    if answer == "invalid":
        answer = await extract_answer(prompt[1])

    end_time = time.time()
    duration = end_time - start_time

    # Save the results to a file
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
    
    return answer, duration


async def fully_async_generate(problems, small_model_max_tokens, turns, small_model_name, small_tokenizer, start_idx, output_dir, sglang_port):
    """
    Generates answers for a list of problems asynchronously.
    The start_idx parameter is used to correctly label the files.
    """
    tasks = [
        asyncio.create_task(
            process_single_problem(p, small_model_max_tokens, turns, start_idx + idx, small_model_name, small_tokenizer, output_dir, sglang_port)
        )
        for idx, p in enumerate(problems)
    ]
    results = await tqdm.gather(*tasks, desc="Processing problems")
    durations = [duration for _, duration in results]
    avg_time = sum(durations) / len(durations)
    print(f"Average total time per problem: {avg_time:.4f}s")
    return results


async def compute_score(results, answers, repeats):
    """
    Computes the final accuracy score based on generated answers.
    """
    if not results:
        print("No results to compute score for.")
        return
        
    generated_ans = [ans for ans, _ in results]
    group = len(generated_ans) // repeats
    
    if group == 0:
        print(f"Insufficient results for score computation. Need at least {repeats} results, got {len(generated_ans)}.")
        return
        
    right = 0
    for i in range(group):
        start = i * repeats
        end = (i + 1) * repeats
        outputs = generated_ans[start:end]
        
        # Ensure we have the correct answer for this group
        if start >= len(answers):
            print(f"Warning: No correct answer available for group {i}")
            continue
            
        correct_answer = answers[start]
        print(f"Generated answers: {outputs}")
        print(f"Correct answers: {correct_answer}")
        ans = await anyone_check(correct_answer, outputs)
        print(ans)
        if ans == "Match":
            right += 1

    print(f"Accuracy: {right / group:.2%} ({right}/{group})")


async def main():
    """
    Main function to parse arguments, set up clients, and run the evaluation.
    """
    global client_small

    # Create a new ArgumentParser object
    parser = argparse.ArgumentParser(description="Process a dataset with a small language model and save results.")
    
    # Add command-line arguments
    parser.add_argument("--dataset_name", type=str, default="gpqa", help="Name of the dataset to load.")
    parser.add_argument("--turns", type=int, default=30, help="Number of turns for each problem.")
    parser.add_argument("--small_model_max_tokens", type=int, default=100, help="Maximum number of tokens for the small model.")
    parser.add_argument("--small_model_name", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", help="Name of the small model to use.")
    parser.add_argument("--sglang_port", type=int, default=60000, help="Port number for SGLang server.")
    
    # Parse command-line arguments
    args = parser.parse_args()

    # --- Dynamically create the output directory name ---
    small_model_base = args.small_model_name.replace('/', '_')
    output_dir = os.path.join("./results", f"{small_model_base}_{args.dataset_name}")
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving results to: {output_dir}")

    # Initialize small_tokenizer here, using the command-line argument
    small_tokenizer = AutoTokenizer.from_pretrained(args.small_model_name)
    small_tokenizer.use_default_system_prompt = True

    # Initialize global variables in the event loop
    global semaphore, client_small
    semaphore = asyncio.Semaphore(8)  # 在事件循环中创建Semaphore
    client_small = httpx.AsyncClient(
        timeout=240.0, 
        limits=httpx.Limits(
            max_connections=1000,         
            max_keepalive_connections=1000
        )
    )

    repeats = 16
    context, answer = load_my_dataset(args.dataset_name, repeats)
    
    total_problems = len(context) // repeats
    
    # --- Resumption logic: Check for existing files ---
    processed_data_indices = set()
    valid_processed_count = 0
    
    for filename in os.listdir(output_dir):
        if filename.startswith("problem_") and filename.endswith(".json"):
            try:
                # Extract the data index from the filename (e.g., "problem_0012.json")
                idx_str = filename.replace("problem_", "").replace(".json", "")
                data_idx = int(idx_str)
                
                # Validate the file content
                filepath = os.path.join(output_dir, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        # Check if file contains required fields
                        if all(key in data for key in ['final_answer', 'duration_seconds', 'problem_index']):
                            processed_data_indices.add(data_idx)
                            valid_processed_count += 1
                        else:
                            print(f"Warning: File {filename} is missing required fields, will be reprocessed")
                except (json.JSONDecodeError, FileNotFoundError, KeyError) as e:
                    print(f"Warning: File {filename} is corrupted or incomplete ({e}), will be reprocessed")
                    
            except ValueError:
                continue # Skip files that don't match the expected naming convention
    
    # Calculate which problem groups need processing based on data indices
    processed_problem_groups = set()
    for data_idx in processed_data_indices:
        problem_group_idx = data_idx // repeats
        if problem_group_idx < total_problems:
            # Check if all data points in this problem group are processed
            group_start = problem_group_idx * repeats
            group_end = (problem_group_idx + 1) * repeats
            group_data_indices = set(range(group_start, group_end))
            if group_data_indices.issubset(processed_data_indices):
                processed_problem_groups.add(problem_group_idx)
    
    unprocessed_problem_groups = [idx for idx in range(total_problems) if idx not in processed_problem_groups]
    
    if not unprocessed_problem_groups:
        print("All problem groups have been completely processed. Computing final score...")
        # Load all results for score computation with error handling
        all_results = []
        missing_files = []
        
        for data_idx in range(len(context)):
            filepath = os.path.join(output_dir, f"problem_{data_idx:04d}.json")
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if all(key in data for key in ['final_answer', 'duration_seconds']):
                        all_results.append((data['final_answer'], data['duration_seconds']))
                    else:
                        missing_files.append(filepath)
            except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
                missing_files.append(filepath)
                print(f"Warning: Cannot read {filepath}: {e}")
        
        if missing_files:
            print(f"Warning: {len(missing_files)} files are missing or corrupted. Score computation may be incomplete.")
        
        if len(all_results) > 0:
            await compute_score(all_results, answer, repeats)
        else:
            print("No valid results found for score computation.")
            
        await client_small.aclose()
        return

    print(f"Found {len(processed_problem_groups)} processed problem groups. Resuming from {len(processed_problem_groups)} out of {total_problems} problem groups.")
    print(f"Total valid processed files: {valid_processed_count}")
    
    results = []
    start = time.time()
    for problem_group_idx in sync_tqdm(unprocessed_problem_groups, desc="Processing problem groups"):
        # The problem group is based on the original full context
        group_start = problem_group_idx * repeats
        group_end = (problem_group_idx + 1) * repeats
        problem_group = context[group_start:group_end]
        
        result_group = await fully_async_generate(
            problem_group,
            args.small_model_max_tokens,
            args.turns,
            args.small_model_name,
            small_tokenizer,
            group_start, # Pass the starting data index for correct file naming
            output_dir,
            args.sglang_port
        )
        results.extend(result_group)

    end = time.time()
    print(f"Elapsed: {end - start:.3f} s")
    
    # After the run, load all results (processed and newly generated) to compute the final score
    all_results = []
    missing_files = []
    
    for data_idx in range(len(context)):
        filepath = os.path.join(output_dir, f"problem_{data_idx:04d}.json")
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if all(key in data for key in ['final_answer', 'duration_seconds']):
                    all_results.append((data['final_answer'], data['duration_seconds']))
                else:
                    missing_files.append(filepath)
                    print(f"Warning: File {filepath} is missing required fields.", flush=True)
        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            missing_files.append(filepath)
            print(f"Warning: Cannot read {filepath}: {e}", flush=True)

    if missing_files:
        print(f"Warning: {len(missing_files)} files are missing or corrupted. Score computation may be incomplete.", flush=True)
    
    if len(all_results) > 0:
        await compute_score(all_results, answer, repeats)
    else:
        print("No valid results found for final score computation.")
        
    await client_small.aclose()


if __name__ == "__main__":
    asyncio.run(main())