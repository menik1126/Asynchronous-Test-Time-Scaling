import __init__
import argparse
import concurrent.futures
import copy
import json
import math
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from typing import Dict, Tuple
import torch
import numpy as np
import ray
from openai import OpenAI
from skythought_evals.batch import Pipeline, init_engine_from_config
from skythought_evals.batch.env_config import EnvConfig
from skythought_evals.batch.workload import EvalWorkload
from skythought_evals.batch.workload import (
    load_config_from_path as load_ray_config_from_path,
)
from skythought_evals.models import ModelConfig, get_system_prompt_keys
from skythought_evals.tasks import (
    TASK_HANDLER_MAP,
    TASK_NAMES_TO_YAML,
    NUMINATaskHandler,
    TaskConfig,
    TaskHandler,
)
from skythought_evals.util.common import set_seed
from skythought_evals.util.metrics import pass_at_k
from skythought_evals.util.response import Response, SingleParsedResponse
from tqdm import tqdm
from vllm import LLM, SamplingParams, EngineArgs

module_dir = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RAY_CONFIG_RELATIVE_PATH = "ray_configs/ray_config.yaml"


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def fetch_response_openai(llm, model_name, max_tokens, temp, num_responses, prompt):
    model_name = model_name.replace("openai/", "")
    if "o1" in model_name:
        # O1 doesn't support system prompt
        # NOTE: might want to implement this inside handler instead
        for p in prompt:
            p["role"] = "user"

        response = llm.chat.completions.create(
            model=model_name,
            messages=prompt,
            n=num_responses,
            temperature=1,  # has to be 1
            max_completion_tokens=max_tokens,
        )
    else:
        response = llm.chat.completions.create(
            model=model_name,
            messages=prompt,
            n=num_responses,
            temperature=temp,
            max_tokens=max_tokens,
        )
    return response


def fetch_responses_ray(conversations, max_tokens, temp, args):
    config = load_ray_config_from_path(args.ray_config)
    config["model_id"] = args.model
    # use user-provided dtype from CLI
    config["engine_kwargs"]["dtype"] = args.dtype
    # use overrides if provided
    if args.ray_config_tensor_parallel_size:
        config["engine_kwargs"][
            "tensor_parallel_size"
        ] = args.ray_config_tensor_parallel_size

    if args.ray_config_num_replicas:
        config["env_config"]["num_replicas"] = args.ray_config_num_replicas

    engine_cfg = init_engine_from_config(config)
    ds = ray.data.from_items([(idx, conv) for idx, conv in enumerate(conversations)])
    num_replicas = config["env_config"].get("num_replicas", 1)
    if ds.count() < config["env_config"].get("batch_size", 1):
        config["env_config"]["batch_size"] = math.ceil(ds.count() / num_replicas)
    if num_replicas > 1 and num_replicas > ds.num_blocks():
        ds = ds.repartition(num_partitions=num_replicas)
    workload = EvalWorkload(
        dataset=ds,
        sampling_params={
            "n": args.n,
            "max_tokens": max_tokens,
            "temperature": temp,
            "top_p": args.top_p,
        },
    )
    pipeline = Pipeline(
        engine_cfg,
        env_config=EnvConfig(**config["env_config"]),
    )
    ds = pipeline(workload)
    responses = ds.materialize()
    return responses


def _parse_response_for_idx(
    response: Response, sample_idx: int, args
) -> Tuple[SingleParsedResponse, Dict[str, int]]:
    content = response.response[sample_idx].strip()
    response_entry = SingleParsedResponse(content=content)

    token_usage_for_response = {
        "completion_tokens": response.num_completion_tokens[sample_idx],
        "prompt_tokens": response.num_input_tokens,
    }
    if len(response.num_correct_tokens) > 0:
        token_usage_for_response['correct_tokens'] = response.num_correct_tokens[sample_idx]
        token_usage_for_response['time_spend'] = response.num_time_spend[sample_idx]
        token_usage_for_response['try_correct'] = response.num_try_correct[sample_idx]
    if len(response.num_accept) > 0:
        token_usage_for_response['correct_tokens'] = response.num_accept
    return response_entry, token_usage_for_response


# 全局结果字典，用于流式处理时维护状态
_streaming_results = {}

def _process_and_save_sample_immediately(handler, problem_data, sample_response, temp, problem_idx, sample_idx, result_file, existing_results):
    """
    立即处理和保存单个采样的结果
    """
    global _streaming_results
    try:
        # 获取问题key - 修复：正确获取problem_key，优先使用Question字段
        problem_key = str(problem_data.get('Question', problem_data.get('problem', problem_data.get('question', ''))))
        
        # 检查正确性
        response_entry = handler.update_results(problem_data, sample_response['generated_text'])
        
        # 初始化结果结构（如果不存在）
        if problem_key not in _streaming_results:
            _streaming_results[problem_key] = problem_data.copy()
            _streaming_results[problem_key]["responses"] = {}
            _streaming_results[problem_key]["token_usages"] = {}
        
        temp_str = str(temp)
        if temp_str not in _streaming_results[problem_key]["responses"]:
            _streaming_results[problem_key]["responses"][temp_str] = []
            _streaming_results[problem_key]["token_usages"][temp_str] = []
        
        # 确保列表足够长
        responses_list = _streaming_results[problem_key]["responses"][temp_str]
        token_usages_list = _streaming_results[problem_key]["token_usages"][temp_str]
        
        while len(responses_list) <= sample_idx:
            responses_list.append({})
        while len(token_usages_list) <= sample_idx:
            token_usages_list.append({})
        
        # 保存响应和token使用情况
        responses_list[sample_idx] = response_entry
        token_usages_list[sample_idx] = {
            "completion_tokens": sample_response.get('num_tokens', 0),
            "prompt_tokens": sample_response.get('question', 0),
            "generation_time": sample_response.get('generation_time', 0.0)
        }
        
        # 立即保存到JSONL
        jsonl_file = result_file.replace('.json', '.jsonl')
        if append_problem_result_jsonl(problem_key, _streaming_results[problem_key], jsonl_file):
            completed_samples = len([r for r in responses_list if r and r.get("content")])
            print(f"⚡ Problem '{problem_key}' sample {sample_idx+1} (temp={temp}) saved immediately ({completed_samples} completed)")
        
    except Exception as e:
        print(f"Warning: Failed to process and save sample immediately: {e}")


def inference(llm, conversations, max_tokens, temp, args, remaining_data=None, existing_results=None, handler=None, result_file=None):
    global _streaming_results
    
    # 初始化流式结果字典
    if existing_results:
        _streaming_results.update(existing_results)
    
    if args.spe_config is not None:
        from speculative.speculative_thinking import process_message
        r = process_message(conversations[0], llm, max_tokens, temp, top_p=args.top_p)
        responses = []
        
        # 🔥 流式处理：每个采样完成就立即处理和保存
        for i in tqdm(range(len(conversations)), desc="Processing Problems"):
            con = conversations[i]
            res = []
            
            # 检查是否有已存在的采样结果需要恢复
            existing_samples = 0
            if remaining_data and i < len(remaining_data):
                # 修复：正确获取problem_key，优先使用Question字段
                problem_key = str(remaining_data[i].get('Question', remaining_data[i].get('problem', remaining_data[i].get('question', ''))))
                # 检查_streaming_results中是否有已有的结果
                if problem_key in _streaming_results and "responses" in _streaming_results[problem_key]:
                    temp_str = str(temp)
                    if temp_str in _streaming_results[problem_key]["responses"]:
                        # 计算已有的有效采样数
                        existing_valid_samples = [
                            r for r in _streaming_results[problem_key]["responses"][temp_str] 
                            if r and isinstance(r, dict) and r.get("content")
                        ]
                        existing_samples = len(existing_valid_samples)
                        # 将已有的采样添加到res中
                        for existing_sample in existing_valid_samples:
                            mock_response = {
                                'generated_text': existing_sample["content"],
                                'num_tokens': existing_sample.get("completion_tokens", 0),
                                'correct_tokens': [],
                                'try_correct_num': 0,
                                'generation_time': 0.0,
                                'question': len(con) if isinstance(con, list) else 0
                            }
                            res.append(mock_response)
                        print(f"🔄 Restored {existing_samples} existing samples for problem {i}")
            
            # 流式生成和处理采样
            samples_needed = args.n - existing_samples
            for sample_idx in tqdm(range(samples_needed), desc=f"Problem {i+1} Sampling", leave=False):
                r = process_message(con, llm, max_tokens, temp, top_p=args.top_p)
                if r is None: 
                    print(f"Warning: Problem {i} sample {existing_samples + sample_idx+1}/{args.n} failed, continuing with next sample")
                    continue  # 继续尝试下一个采样，而不是break
                else: 
                    r['question'] = llm.get_prompt_len(con)
                    res.append(r)
                    
                    # 🔥 立即处理和保存这个采样
                    if handler and result_file and remaining_data and i < len(remaining_data):
                        total_sample_idx = existing_samples + len(res) - 1  # 总体采样索引
                        _process_and_save_sample_immediately(
                            handler, remaining_data[i], r, temp, i, total_sample_idx, result_file, existing_results
                        )

            # 保存已有的结果，即使采样数不足
            if len(res) > 0: 
                responses.append(Response.from_spe_response(res, i))
                if len(res) < args.n:
                    print(f"Warning: Problem {i} only got {len(res)}/{args.n} samples due to failures")
        
        # 🔥 流式处理完成，保存最终结果
        if result_file and _streaming_results:
            final_json_file = result_file
            with open(final_json_file, "w", encoding="utf-8") as file:
                json.dump(_streaming_results, file, ensure_ascii=False, indent=4, cls=NumpyEncoder)
            print(f"💾 Streaming results saved to {final_json_file}")
            print(f"📊 Processed {len(_streaming_results)} problems with streaming save logic")
    elif args.draft_model is not None:
        responses = []
        sampling_params = SamplingParams(
            max_tokens=max_tokens, temperature=temp, n=args.n, top_p=args.top_p
        )
        for i in tqdm(range(len(conversations))):
            con = conversations[i]
            res = []
            r = llm.chat( messages=[con], sampling_params=sampling_params, use_tqdm=True)
            responses.append(r[0])
        responses = [Response.from_spe_decoding_response(response) for response in responses]
    elif args.use_ray:
        responses = fetch_responses_ray(conversations, max_tokens, temp, args)
        responses = [
            Response.from_ray_response(response) for response in responses.iter_rows()
        ]
        # TODO/NOTE: This deepcopy is needed to avoid a SIGSEV error related to object cleanup with the ray object store and
        # the later use of ProcessPoolExecutor - see here: https://github.com/NovaSky-AI/SkyThought/pull/63#discussion_r1941899714
        # revisit the underlying issue and remove the deepcopy if possible
        responses = copy.deepcopy(responses)
        responses = sorted(responses, key=lambda x: x.index)
    elif args.model.startswith("openai"):
        fetch_partial = partial(
            fetch_response_openai, llm, args.model, max_tokens, temp, args.n
        )

        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as e:
            responses = list(e.map(fetch_partial, conversations))

        responses = [Response.from_openai_response(response) for response in responses]
    else:
        sampling_params = SamplingParams(
            max_tokens=max_tokens, temperature=temp, n=args.n, top_p=args.top_p
        )
        responses = llm.chat(
            messages=conversations, sampling_params=sampling_params, use_tqdm=True
        )
        responses = [Response.from_vllm_response(response) for response in responses]

    return responses


def load_existing_results(result_file):
    """Load existing results from JSON or JSONL format."""
    # 🔥 优先尝试JSONL文件（流式保存的最新数据）
    jsonl_file = result_file.replace('.json', '.jsonl')
    if os.path.exists(jsonl_file):
        print(f"Loading existing results from JSONL: {jsonl_file}")
        return _load_results_jsonl(jsonl_file)
    
    # 如果JSONL不存在，尝试JSON文件
    if not os.path.exists(result_file):
        return {}
    
    # Try JSONL format first (for new saves)
    if result_file.endswith('.jsonl'):
        return _load_results_jsonl(result_file)
    
    # Try JSON format (for backward compatibility)
    try:
        print(f"Loading existing results from JSON: {result_file}")
        with open(result_file, "r", encoding="utf-8") as f:
            records = json.load(f)
        return records
    except json.JSONDecodeError:
        # If JSON fails, try JSONL format
        print(f"JSON format failed, trying JSONL format for {result_file}")
        return _load_results_jsonl(result_file)


def _load_results_jsonl(result_file):
    """Load results from JSONL format."""
    results = {}
    try:
        with open(result_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                try:
                    data = json.loads(line)
                    results.update(data)
                except json.JSONDecodeError as e:
                    print(f"Warning: Invalid JSON on line {line_num} in {result_file}: {e}")
                    continue
    except Exception as e:
        print(f"Warning: Failed to load JSONL file {result_file}: {e}")
    
    return results


def append_problem_result_jsonl(problem_key, problem_data, result_file):
    """🔥 JSONL INCREMENTAL UPDATE: Update problem data by rewriting JSONL file."""
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(result_file), exist_ok=True)
        
        # Load existing data from JSONL file
        existing_data = {}
        if os.path.exists(result_file):
            existing_data = _load_results_jsonl(result_file)
        
        # Update the specific problem data
        existing_data[problem_key] = problem_data
        
        # Rewrite the entire JSONL file with updated data
        with open(result_file, 'w', encoding='utf-8') as f:
            for key, data in existing_data.items():
                line = json.dumps({key: data}, ensure_ascii=False, cls=NumpyEncoder)
                f.write(line + '\n')
        
        return True
    except Exception as e:
        print(f"Warning: Failed to update problem {problem_key} in JSONL: {e}")
        return False


def save_results_incrementally(results, result_file):
    """Save results incrementally to avoid data loss (kept for compatibility)."""
    try:
        with open(result_file, "w", encoding="utf-8") as file:
            json.dump(results, file, ensure_ascii=False, indent=4, cls=NumpyEncoder)
        return True
    except Exception as e:
        print(f"Warning: Failed to save results: {e}")
        return False


def perform_inference_and_check(
    handler: TaskHandler,
    temperatures,
    max_tokens,
    result_file,
    llm,
    model_config,
    args,
):
    results = load_existing_results(result_file)
    print(f"Loaded {len(results)} existing results.")
    train_data = handler.load_and_filter_dataset(
        args.start,
        args.end,
        split=args.split,
        subset=args.subset,
        difficulty=args.difficulty,
        args=args,
    )
    remaining_data = handler.process_remaining_data(train_data, results, args, temperatures)
    if not len(remaining_data):
        print("All results saved. Exiting....")
        return
    conversations = handler.make_conversations(
        remaining_data, model_config.system_prompt, model_config.user_template
    )
    temperature_to_scores = {}
    temperature_to_acc = {}
    responses = []
    for temp in temperatures:
        if len(conversations) == 0:
            print("No more data to process")
            continue

        responses = inference(llm, conversations, max_tokens, temp, args, remaining_data, results, handler, result_file)

        total_correct = 0
        total_finish = 0
        temperature_to_scores[temp] = {}
        with ProcessPoolExecutor(max_workers=32) as executor:
            future_to_task = {}
            token_usages = {}
            idx_to_actual_samples = {}  # 存储每个idx对应的实际采样数量
            for idx, response in enumerate(responses):
                if response.index is not None: idx = response.index
                # 使用实际的采样数量，而不是固定的args.n
                actual_samples = len(response.response)
                idx_to_actual_samples[idx] = actual_samples
                for sample_idx in range(actual_samples):
                    # response_entry at this point doesn't contain correctness check.
                    response_entry, token_usage_for_response = _parse_response_for_idx(
                        response, sample_idx, args
                    )
                    if idx not in token_usages:
                        token_usages[idx] = []
                    token_usages[idx].append(token_usage_for_response)
                    # submit correctness check for response
                    future_to_task[
                        executor.submit(
                            handler.update_results,
                            remaining_data[idx],
                            response_entry.content,
                        )
                    ] = (idx, sample_idx)

            # 定期保存机制初始化
            last_save_time = time.time()
            save_interval = 300  # 每5分钟保存一次
            
            for future in tqdm(
                as_completed(future_to_task),
                total=len(future_to_task),
                desc="Processing Generations",
            ):
                idx, sample_idx = future_to_task[future]
                
                # 定期保存检查 - 防止长时间运行时数据丢失
                current_time = time.time()
                if current_time - last_save_time > save_interval:
                    print(f"🕒 Periodic save triggered after {save_interval}s, saving current results...")
                    if save_results_incrementally(results, result_file):
                        print(f"✅ Periodic save completed to {result_file}")
                    last_save_time = current_time
                
                # TODO (sumanthrh): the returned entry is currently a dict and can be confusing.
                # this should also be a ParsedResponse object.
                response_entry: dict = future.result()
                total_correct += response_entry["correctness"]
                total_finish += 1

                problem_key = remaining_data[idx][handler.question_key]
                if problem_key not in results:
                    results[problem_key] = remaining_data[idx]
                    if isinstance(handler, NUMINATaskHandler):
                        results[problem_key]["messages"] = ""
                    results[problem_key]["responses"] = {}
                    results[problem_key]["token_usages"] = {}
                    prompt = conversations[idx][-1]["content"]
                    results[problem_key]["prompt"] = prompt
                    results[problem_key]["input_conversation"] = conversations[idx]
                    # 使用实际的采样数量初始化scores
                    actual_samples = idx_to_actual_samples[idx]
                    temperature_to_scores[temp][problem_key] = [
                        0 for _ in range(actual_samples)
                    ]

                if str(temp) not in results[problem_key]["responses"]:
                    # 使用实际的采样数量初始化responses
                    actual_samples = idx_to_actual_samples[idx]
                    results[problem_key]["responses"][str(temp)] = [
                        {} for _ in range(actual_samples)
                    ]

                results[problem_key]["responses"][str(temp)][
                    sample_idx
                ] = response_entry
                # do this only once per problem/idx
                if str(temp) not in results[problem_key]["token_usages"]:
                    results[problem_key]["token_usages"][str(temp)] = token_usages[idx]

                # update scores
                temperature_to_scores[temp][problem_key][sample_idx] = response_entry[
                    "correctness"
                ]
                
                # 🔥 JSONL REAL-TIME APPEND: 每完成一个采样就保存，真正的中间结果保存
                jsonl_file = result_file.replace('.json', '.jsonl')
                if append_problem_result_jsonl(problem_key, results[problem_key], jsonl_file):
                    problem_samples_completed = sum(1 for resp in results[problem_key]["responses"][str(temp)] if resp)
                    expected_samples = idx_to_actual_samples[idx]
                    print(f"⚡ Problem '{problem_key}' sample {sample_idx+1} (temp={temp}) saved immediately ({problem_samples_completed}/{expected_samples} completed)")

        print(f"Final acc: {total_correct}/{total_finish}")

        acc = round(total_correct / total_finish, 4) if total_finish > 0 else 0
        temperature_to_acc[f"{temp=}"] = acc
        print(json.dumps({"acc": acc}))

    pass_at_k_metrics = None
    if args.n > 1:
        pass_at_k_metrics = pass_at_k(args.n, temperature_to_scores)
        print(json.dumps({"pass_at_k": pass_at_k_metrics}))

    total_prompt_tokens = sum(
        results[key]["token_usages"][str(temp)][sample_idx]["prompt_tokens"]
        for sample_idx in range(args.n)
        for key in results
        for temp in temperatures
    )
    total_completion_tokens = sum(
        results[key]["token_usages"][str(temp)][sample_idx]["completion_tokens"]
        for sample_idx in range(args.n)
        for key in results
        for temp in temperatures
    )
    num_responses_total = len(responses) * args.n * len(temperatures)

    # Token usage summary
    result_dir, result_name = os.path.split(result_file)
    metrics_dir = os.path.join(result_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    # Construct the token usage result file path
    metrics_result_file = os.path.join(metrics_dir, result_name)

    # Prepare the token usage dictionary
    metrics_dict = {
        "completion_tokens": total_completion_tokens,
        "prompt_tokens": total_prompt_tokens,
        "avg_completion_tokens": (
            round(total_completion_tokens / num_responses_total, 3)
            if total_completion_tokens
            else 0
        ),
        "avg_prompt_tokens": (
            round(total_prompt_tokens / num_responses_total, 3)
            if total_prompt_tokens
            else 0
        ),
        "pass_at_k": pass_at_k_metrics,
        "accuracy": temperature_to_acc,
    }

    # Save the token usage dictionary to the result file
    with open(metrics_result_file, "w") as f:
        json.dump(metrics_dict, f, indent=4)

    print(f"Metrics saved to {metrics_result_file}")

    with open(result_file, "w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=4, cls=NumpyEncoder)


def perform_check(handler: TaskHandler, temperatures, result_file, args):
    results = load_existing_results(result_file)
    print(f"Loaded {len(results)} existing results.")

    train_data = handler.load_and_filter_dataset(
        args.start,
        args.end,
        split=args.split,
        subset=args.subset,
        difficulty=args.difficulty,
        args=args,
    )
    remaining_data = handler.process_remaining_data(train_data, {})

    tasks = []
    for item in remaining_data:
        problem_key = item[handler.question_key]
        # If this item exists in the results file, check each temperature
        if problem_key in results and "responses" in results[problem_key]:
            for temp in temperatures:
                if str(temp) in results[problem_key]["responses"]:
                    response_entries = results[problem_key]["responses"][str(temp)]
                    for sample_id, response_entry in enumerate(response_entries):
                        if sample_id > (args.n - 1):
                            continue
                        if True or response_entry["correctness"] is None:
                            processed = "processed_content" in response_entry
                            tasks.append(
                                (
                                    item,
                                    temp,
                                    (
                                        response_entry["processed_content"]
                                        if processed
                                        else response_entry["content"]
                                    ),
                                    sample_id,
                                )
                            )

    print(f"Found {len(tasks)} responses requiring reject sampling...")

    total_correct = 0
    total_finish = 0
    correct = {temp: {} for temp in temperatures}
    with ProcessPoolExecutor(max_workers=32) as executor:
        future_to_task = {
            executor.submit(handler.update_results, item, content): (
                item,
                temp,
                sample_id,
            )
            for (item, temp, content, sample_id) in tasks
        }

        # 4. Collect the results as they finish.
        for future in tqdm(
            as_completed(future_to_task),
            total=len(future_to_task),
            desc="Processing Reject Sampling",
        ):
            item, temp, sample_id = future_to_task[future]
            new_response_entry = future.result()
            total_correct += new_response_entry["correctness"]
            total_finish += 1

            # Update the corresponding record in results
            problem_key = item[handler.question_key]
            if problem_key not in correct[temp]:
                correct[temp][problem_key] = False
            if new_response_entry["correctness"]:
                correct[temp][problem_key] = True
            assert (
                problem_key in results
                and "responses" in results[problem_key]
                and str(temp) in results[problem_key]["responses"]
            )
            response_entry = results[problem_key]["responses"][str(temp)][sample_id]
            response_entry["correctness"] = new_response_entry["correctness"]
            response_entry["reason"] = new_response_entry["reason"]
            results[problem_key]["responses"][str(temp)][sample_id] = response_entry

    print(f"Final reject-sampling accuracy: {total_correct}/{total_finish}")
    # per temperature acc
    for temp in temperatures:
        temp_correct = sum(correct[temp].values())
        temp_total = len(correct[temp])
        temp_acc = round(temp_correct / temp_total, 4) if temp_total > 0 else 0
        print(f"Temperature {temp} acc: {temp_correct}/{temp_total} ({temp_acc})")

    with open(result_file, "w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=4, cls=NumpyEncoder)


def perform_inference_and_save(
    handler: TaskHandler,
    temperatures,
    max_tokens,
    result_file,
    llm,
    model_config,
    args,
):
    results = load_existing_results(result_file)
    print(f"Loaded {len(results)} existing results.")
    train_data = handler.load_and_filter_dataset(
        args.start,
        args.end,
        split=args.split,
        subset=args.subset,
        difficulty=args.difficulty,
        args=args,
    )
    remaining_data = handler.process_remaining_data(train_data, results, args, temperatures)
    if not len(remaining_data):
        print("All results saved. Exiting...")
        return
    conversations = handler.make_conversations(
        remaining_data, model_config.system_prompt, model_config.user_template
    )

    for temp in temperatures:
        if len(conversations) == 0:
            print("No more data to process")
            continue
        responses = inference(llm, conversations, max_tokens, temp, args, remaining_data, results, handler, result_file)

        completion_tokens = []
        prompt_tokens = []
        for idx, response in enumerate(responses):
            if response.index is not None: idx = response.index
            response_entries = []
            token_usages = []
            completion_token = 0
            # 使用实际的采样数量，而不是固定的args.n
            actual_samples = len(response.response)
            for sample_idx in range(actual_samples):
                response_entry, token_usage_for_response = _parse_response_for_idx(
                    response, sample_idx, args
                )
                token_usages.append(token_usage_for_response)
                completion_token += token_usage_for_response["completion_tokens"]
                response_entries.append(response_entry.to_dict())

            completion_token /= actual_samples if actual_samples > 0 else 1
            prompt_token = response.num_input_tokens
            prompt_tokens.append(prompt_token)
            completion_tokens.append(completion_token)

            problem_key = remaining_data[idx][
                handler.question_key
            ]  # can you use this idx
            if problem_key not in results:
                results[problem_key] = remaining_data[idx]
                if isinstance(handler, NUMINATaskHandler):
                    results[problem_key]["messages"] = ""
                results[problem_key]["responses"] = {}
                results[problem_key]["token_usages"] = {}
                prompt = conversations[idx][-1]["content"]
                results[problem_key]["prompt"] = prompt

            results[problem_key]["responses"][str(temp)] = response_entries
            results[problem_key]["token_usages"][str(temp)] = token_usages
            
            # 🔥 JSONL REAL-TIME APPEND: Append immediately after each problem completion
            jsonl_file = result_file.replace('.json', '.jsonl')
            if append_problem_result_jsonl(problem_key, results[problem_key], jsonl_file):
                print(f"⚡ Problem '{problem_key}' (temp={temp}) appended to JSONL immediately")

    # Token usage summary put into another subdirectory
    result_dir, result_name = os.path.split(result_file)
    metrics_dir = os.path.join(result_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    # Construct the token usage result file path
    metrics_result_file = os.path.join(metrics_dir, result_name)

    # Prepare the token usage dictionary
    metrics_dict = {
        "completion_tokens": sum(completion_tokens),
        "prompt_tokens": sum(prompt_tokens),
        "avg_completion_tokens": (
            round(sum(completion_tokens) / len(completion_tokens), 3)
            if completion_tokens
            else 0
        ),
        "avg_prompt_tokens": (
            round(sum(prompt_tokens) / len(prompt_tokens), 3) if prompt_tokens else 0
        ),
    }

    # Save the token usage dictionary to the result file
    with open(metrics_result_file, "w") as f:
        json.dump(metrics_dict, f, indent=4)

    print(f"Token usage saved to {metrics_result_file}")

    with open(result_file, "w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=4, cls=NumpyEncoder)


def perform_spe_inference_and_check(
    handler: TaskHandler,
    temperatures,
    max_tokens,
    result_file,
    llm,
    model_config,
    args,
):
    results = load_existing_results(result_file)
    print(f"Loaded {len(results)} existing results.")
    train_data = handler.load_and_filter_dataset(
        args.start,
        args.end,
        split=args.split,
        subset=args.subset,
        difficulty=args.difficulty,
        args=args,
    )
    remaining_data = handler.process_remaining_data(train_data, results, args, temperatures)
    if not len(remaining_data):
        print("All results saved. Exiting....")
        return
    conversations = handler.make_conversations(
        remaining_data, model_config.system_prompt, model_config.user_template
    )
    temperature_to_scores = {}
    temperature_to_acc = {}
    responses = []
    for temp in temperatures:
        if len(conversations) == 0:
            print("No more data to process")
            continue

        responses = inference(llm, conversations, max_tokens, temp, args, remaining_data, results, handler, result_file)

        total_correct = 0
        total_finish = 0
        temperature_to_scores[temp] = {}
        with ProcessPoolExecutor(max_workers=32) as executor:
            future_to_task = {}
            token_usages = {}
            idx_to_actual_samples = {}  # 存储每个idx对应的实际采样数量
            last_save_time = time.time()  # 定期保存机制
            save_interval = 300  # 每5分钟保存一次
            for idx, response in enumerate(responses):
                if response is not None:
                    # 使用实际的采样数量，而不是固定的args.n
                    actual_samples = len(response.response)
                    idx_to_actual_samples[idx] = actual_samples
                    for sample_idx in range(actual_samples):
                        # response_entry at this point doesn't contain correctness check.
                        response_entry, token_usage_for_response = _parse_response_for_idx(
                            response, sample_idx, args
                        )
                        if idx not in token_usages:
                            token_usages[idx] = []
                        token_usages[idx].append(token_usage_for_response)
                        # submit correctness check for response
                        future_to_task[
                            executor.submit(
                                handler.update_results,
                                remaining_data[idx],
                                response_entry.content,
                            )
                        ] = (idx, sample_idx)

            for future in tqdm(
                as_completed(future_to_task),
                total=len(future_to_task),
                desc="Processing Generations",
            ):
                idx, sample_idx = future_to_task[future]
                
                # 定期保存检查 - 防止长时间运行时数据丢失
                current_time = time.time()
                if current_time - last_save_time > save_interval:
                    print(f"🕒 Periodic save triggered after {save_interval}s, saving current results...")
                    if save_results_incrementally(results, result_file):
                        print(f"✅ Periodic save completed to {result_file}")
                    last_save_time = current_time
                
                # TODO (sumanthrh): the returned entry is currently a dict and can be confusing.
                # this should also be a ParsedResponse object.
                response_entry: dict = future.result()
                total_correct += response_entry["correctness"]
                total_finish += 1

                problem_key = remaining_data[idx][handler.question_key]
                if problem_key not in results:
                    results[problem_key] = remaining_data[idx]
                    if isinstance(handler, NUMINATaskHandler):
                        results[problem_key]["messages"] = ""
                    results[problem_key]["responses"] = {}
                    results[problem_key]["token_usages"] = {}
                    prompt = conversations[idx][-1]["content"]
                    results[problem_key]["prompt"] = prompt
                    results[problem_key]["input_conversation"] = conversations[idx]
                    # 使用实际的采样数量初始化scores
                    actual_samples = idx_to_actual_samples[idx]
                    temperature_to_scores[temp][problem_key] = [
                        0 for _ in range(actual_samples)
                    ]

                if str(temp) not in results[problem_key]["responses"]:
                    # 使用实际的采样数量初始化responses
                    actual_samples = idx_to_actual_samples[idx]
                    results[problem_key]["responses"][str(temp)] = [
                        {} for _ in range(actual_samples)
                    ]

                results[problem_key]["responses"][str(temp)][
                    sample_idx
                ] = response_entry
                # do this only once per problem/idx
                if str(temp) not in results[problem_key]["token_usages"]:
                    results[problem_key]["token_usages"][str(temp)] = token_usages[idx]

                # update scores
                temperature_to_scores[temp][problem_key][sample_idx] = response_entry[
                    "correctness"
                ]
                
                # 🔥 JSONL REAL-TIME APPEND: 每完成一个采样就保存，真正的中间结果保存
                jsonl_file = result_file.replace('.json', '.jsonl')
                if append_problem_result_jsonl(problem_key, results[problem_key], jsonl_file):
                    problem_samples_completed = sum(1 for resp in results[problem_key]["responses"][str(temp)] if resp)
                    expected_samples = idx_to_actual_samples[idx]
                    print(f"⚡ Problem '{problem_key}' sample {sample_idx+1} (temp={temp}) saved immediately ({problem_samples_completed}/{expected_samples} completed)")

        print(f"Final acc: {total_correct}/{total_finish}")

        acc = round(total_correct / total_finish, 4) if total_finish > 0 else 0
        temperature_to_acc[f"{temp=}"] = acc
        print(json.dumps({"acc": acc}))

    pass_at_k_metrics = None
    if args.n > 1:
        pass_at_k_metrics = pass_at_k(args.n, temperature_to_scores)
        print(json.dumps({"pass_at_k": pass_at_k_metrics}))

    total_prompt_tokens = sum(
        results[key]["token_usages"][str(temp)][sample_idx]["prompt_tokens"]
        for sample_idx in range(args.n)
        for key in results
        for temp in temperatures
    )
    total_completion_tokens = sum(
        results[key]["token_usages"][str(temp)][sample_idx]["completion_tokens"]
        for sample_idx in range(args.n)
        for key in results
        for temp in temperatures
    )
    num_responses_total = len(responses) * args.n * len(temperatures)

    # Token usage summary
    result_dir, result_name = os.path.split(result_file)
    metrics_dir = os.path.join(result_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    # Construct the token usage result file path
    metrics_result_file = os.path.join(metrics_dir, result_name)

    # Prepare the token usage dictionary
    metrics_dict = {
        "completion_tokens": total_completion_tokens,
        "prompt_tokens": total_prompt_tokens,
        "avg_completion_tokens": (
            round(total_completion_tokens / num_responses_total, 3)
            if total_completion_tokens
            else 0
        ),
        "avg_prompt_tokens": (
            round(total_prompt_tokens / num_responses_total, 3)
            if total_prompt_tokens
            else 0
        ),
        "pass_at_k": pass_at_k_metrics,
        "accuracy": temperature_to_acc,
    }

    # Save the token usage dictionary to the result file
    with open(metrics_result_file, "w") as f:
        json.dump(metrics_dict, f, indent=4)

    print(f"Metrics saved to {metrics_result_file}")

    with open(result_file, "w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=4, cls=NumpyEncoder)


def main():
    parser = argparse.ArgumentParser(
        description="Unified inference and checking for different datasets/tasks."
    )
    parser.add_argument(
        "--task",
        type=str,
        default='math500',
        choices=TASK_NAMES_TO_YAML.keys(),
        help="Task to process.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default='deepseek-ai/DeepSeek-R1-Distill-Qwen-32B',
        help="The model to run.",
    )
    parser.add_argument("--tp", type=int, default=2, help="Tensor Parallelism Degree")
    parser.add_argument(
        "--max_tokens", type=int, default=32768, help="Max tokens for the model."
    )
    parser.add_argument(
        "--max_model_len", type=int, default=None, help="Maximum model length for vLLM engine."
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Split to use for the dataset (e.g., train, test).",
    )
    parser.add_argument("--subset", type=str, help="Subset for the dataset.")
    parser.add_argument("--start", type=int, default=0, help="Start index.")
    parser.add_argument("--end", type=int, default=-1, help="End index.")
    parser.add_argument(
        "--difficulty",
        type=str,
        default=None,
        help="Difficulty level. Example: 'easy', 'medium', 'hard'.",
    )
    parser.add_argument(
        "--filter-difficulty",
        action="store_true",
        help="Optional filter difficulty, used for NUMINA.",
    )
    parser.add_argument(
        "--source",
        type=str,
        help="Source column filter for the dataset, used for NUMINA.",
    )
    parser.add_argument(
        "--result-dir", type=str, default="./eval", help="Result dir to save files."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Perform evaluation checks on generated samples.",
    )
    parser.add_argument("--inference", action="store_true", help="Perform inference.")
    parser.add_argument(
        "--temperatures",
        type=float,
        nargs="+",
        default=[0.6],
        help="Temperature for sampling.",
    )
    parser.add_argument(
        "--math-difficulty-lower-bound",
        type=int,
        default=None,
        help="Lowest difficulty level for math.",
    )
    parser.add_argument(
        "--math-difficulty-upper-bound",
        type=int,
        default=None,
        help="Highest difficulty level for math.",
    )
    parser.add_argument(
        "--system-prompt-template",
        type=str,
        default=None,
        help="System prompt template to use",
        choices=get_system_prompt_keys(),
    )
    parser.add_argument(
        "--n", type=int, default=1, help="Number of samples generated per problem."
    )
    parser.add_argument("--seed", type=int, default=41, help="Random seed.")
    parser.add_argument(
        "--use-ray", action="store_true", help="Use ray for scaling inference."
    )
    parser.add_argument(
        "--ray-config",
        type=str,
        default=None,
        help="Ray configuration file if using ray for scaling inference. By default, we use the example in ray_configs/ray_config.yaml",
    )
    parser.add_argument(
        "--ray-config-tensor-parallel-size",
        type=int,
        default=None,
        help="Ray configuration override for tensor parallel size per model replica",
    )
    parser.add_argument(
        "--ray-config-num-replicas",
        type=int,
        default=None,
        help="Ray configuration override for number of model replicas",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "auto", "float16", "bfloat16"],
        help="dtype for inference with vLLM. Full-precision by default."
        "'auto' refers to automatically inferring dtype for the model",
        default="bfloat16",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="Sampling parameter `top_p`",
    )
    parser.add_argument(
        "--spe_config", type=str, default=None, help="Path to speculative thinking config"
    )
    parser.add_argument(
        "--draft_model", type=str, default=None, help="Path to speculative thinking config"
    )
    parser.add_argument(
        "--num_speculative_tokens", type=int, default=5, help="Path to speculative thinking config"
    )
    args = parser.parse_args()
    # load ray config
    if args.use_ray:
        warnings.warn(
            "`tp` CLI argument is not compatible with `use-ray` and will be ignored. Please configure tensor parallel size in the `ray_config` YAML"
            " or override the value with the argument `ray-config-tensor-parallel-size` ",
            stacklevel=1,
        )
        if not args.ray_config:
            # load default
            args.ray_config = os.path.join(module_dir, DEFAULT_RAY_CONFIG_RELATIVE_PATH)
    set_seed(args.seed)
    # os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, range(args.tp)))
    # print(os.environ['CUDA_VISIBLE_DEVICES'], torch.cuda.device_count())
    # enable hf_transfer if not overriden by the user
    if os.environ.get("HF_HUB_ENABLE_HF_TRANSFER", None) is None:
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    if args.task not in TASK_NAMES_TO_YAML:
        raise ValueError(
            f"Task {args.task} not found. Should be one of {TASK_NAMES_TO_YAML.keys()}"
        )

    task_config = TaskConfig.from_yaml(TASK_NAMES_TO_YAML[args.task])
    handler_name = task_config.handler
    handler_cls = TASK_HANDLER_MAP[handler_name]
    handler = handler_cls(task_config)

    model_config = ModelConfig.from_model_id(args.model, args.system_prompt_template)

    temperatures = [1] if args.model.startswith("openai/o1") else args.temperatures

    if args.top_p < 1 and args.model.startswith("openai/o1"):
        print(
            "OpenAI o1 models do not support `top_p` sampling. Resetting `top_p` to 1"
        )
        args.top_p = 1

    print(f"Temperature: {temperatures}")
    max_tokens = args.max_tokens
    if temperatures == [0] and args.n > 1:
        args.n = 1
        print("Warning: Temperature 0 does not support multiple samples. Setting n=1.")

    # TODO: this can be cleaned up by allowing user override for any task_config with optional task_args
    # Currently kept here for consistency with old code
    args.split = args.split if args.split else handler.task_config.dataset_split
    args.subset = args.subset if args.subset else handler.task_config.dataset_subset
    if not args.difficulty and "difficulty" in handler.task_config.preprocess_config:
        args.difficulty = handler.task_config.preprocess_config["difficulty"]

    # create result dir if not exists
    if args.result_dir and not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)
    temperature_str = ",".join(map(str, temperatures))
    file_suffix = (
        f"{model_config.name}_{args.task}_{args.split}_subset_{args.subset}_filter_{args.filter_difficulty}"
        + f"_s{args.start}_e{args.end}_t{temperature_str}_n{args.n}"
    )
    if args.spe_config is not None:
        result_file = os.path.join(
            args.result_dir,
            f"{args.spe_config.split('/')[-1]}_{args.task}_{args.split}_subset_{args.subset}_filter_{args.filter_difficulty}_s{args.start}_e{args.end}_t{temperature_str}_n{args.n}.json",
        )
    elif (
        args.math_difficulty_lower_bound is not None
        or args.math_difficulty_upper_bound is not None
    ):
        result_file = os.path.join(
            args.result_dir,
            f"{model_config.name}_{file_suffix}_{args.math_difficulty_upper_bound}.json",
        )
    else:
        result_file = os.path.join(
            args.result_dir,
            f"{file_suffix}.json",
        )

    if args.check:
        # check if converted file exists
        if (
            args.math_difficulty_lower_bound is not None
            or args.math_difficulty_upper_bound is not None
        ):
            converted_file = f"{args.result_dir}/converted_{file_suffix}.json"
        else:
            converted_file = f"{args.result_dir}/converted_{file_suffix}.json"
        if os.path.exists(converted_file):
            result_file = converted_file
        perform_check(handler, temperatures, result_file, args)
        return
    else:
        if args.use_ray:
            llm = None
        elif args.spe_config is not None:
            from speculative.speculative_thinking import load_spe_model
            llm = load_spe_model(args.spe_config)
        elif args.draft_model is not None:
            # llm = LLM(
            #     model=args.model, tensor_parallel_size=args.tp, dtype=args.dtype,
            #     speculative_config={
            #         "model": args.draft_model,
            #         "num_speculative_tokens": args.num_speculative_tokens,
            #         "draft_tensor_parallel_size": 1,
            #         # "use_v2_block_manager": True
            #     },
            # )
            llm = LLM(
                model=args.model, tensor_parallel_size=args.tp, dtype=args.dtype,
                speculative_model=args.draft_model,
                num_speculative_tokens=args.num_speculative_tokens,
                use_v2_block_manager=True,
                max_model_len=args.max_model_len,
            )
        else:
            llm = (
                OpenAI()
                if args.model.startswith("openai")
                else LLM(
                    model=args.model, tensor_parallel_size=args.tp, dtype=args.dtype, max_model_len=args.max_model_len, # swap_space=32
                )
                # speculative model
            )
        if args.inference:
            perform_inference_and_save(
                handler, temperatures, max_tokens, result_file, llm, model_config, args
            )
        else:
            perform_inference_and_check(
                handler, temperatures, max_tokens, result_file, llm, model_config, args
            )


if __name__ == "__main__":
    main()
