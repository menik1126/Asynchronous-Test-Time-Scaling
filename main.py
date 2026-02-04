from dataset import (
    load_my_dataset,
    evaluate,
)
from agent import agent
from transformers import AutoTokenizer
import math
import time
import random
import copy
import os
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"

def sglang_wait_think(
    evalator,
    evalator_tokenizer,
    outputs_thinking_filtered,
    sampling_params,
    filtered_requests,
    evalator_max_tokens, 
    result_max_tokens,
    thinking_n_ignore,
    thinking_n_ignore_str,
    thinking_n_ignore_str_tok,
    thinking_start,
    thinking_end_max,
    thinking_end_max_tok,
    newline_tok,
    thinking_end_tok,
    thinking_end,
):
    
    sampling_params["max_new_tokens"] = int(evalator_max_tokens / thinking_n_ignore)
    requests_tmp = filtered_requests
    for i in range(thinking_n_ignore):
        print(f"sampleing params: {sampling_params}", flush=True)
        print(f"requests: {len(requests_tmp)}", flush=True)

        batch_size = 40
        outputs_tmp_wait = []

        for i in range(0, len(requests_tmp), batch_size):
            batch = requests_tmp[i:i + batch_size]
            print(f"[phase two] processing requests {i} to {i + len(batch) - 1}", flush=True)
            
            outputs_tmp = evalator.generate(
                input_ids=batch,
                sampling_params=sampling_params
            )
            outputs_tmp_wait.extend(outputs_tmp)

        requests_tmp_new = []
        for j, o in enumerate(outputs_tmp_wait):
            text = o["text"]
            cont = evalator_tokenizer(text)["input_ids"]
            # Final; do not generate further
            if (o["meta_info"]["finish_reason"]["type"] == "length") or (i == thinking_n_ignore - 1):
                outputs_thinking_filtered[j]["text"] += text
                outputs_thinking_filtered[j]["token_ids"] += cont
                outputs_thinking_filtered[j]["meta_info"]["finish_reason"]["type"] = o["meta_info"]["finish_reason"]["type"]
            else:
                
                if thinking_n_ignore_str is not None:
                    cont += thinking_n_ignore_str_tok
                    o["text"] += thinking_n_ignore_str
                if outputs_thinking_filtered[j] is not None:
                    outputs_thinking_filtered[j]["text"] += o["text"]
                    outputs_thinking_filtered[j]["token_ids"] += cont
            requests_tmp_new.append(requests_tmp[j] + cont)
        requests_tmp = requests_tmp_new
    print("phase two over", flush=True)
    for idx in list(range(len(filtered_requests))):
        if len(outputs_thinking_filtered[idx]["token_ids"]) > int(sampling_params["max_new_tokens"] * thinking_n_ignore):
            num = int(sampling_params["max_new_tokens"] * thinking_n_ignore)
            outputs_thinking_filtered[idx]["token_ids"] = outputs_thinking_filtered[idx]["token_ids"][:num]        
    
    for i, o in enumerate(outputs_thinking_filtered):
        cont = o["token_ids"]
        if o["meta_info"]["finish_reason"]["type"] == "length":
            if (o["text"][-1] == "\n") or (thinking_start[0] == "\n"):
                filtered_requests[i] += cont + thinking_end_max_tok
                outputs_thinking_filtered[i]["text"] = thinking_start + outputs_thinking_filtered[i]["text"] + thinking_end_max
            else:
                filtered_requests[i] += cont + newline_tok + thinking_end_max_tok
                outputs_thinking_filtered[i]["text"] = thinking_start + outputs_thinking_filtered[i]["text"] + "\n" + thinking_end_max
        else:
            filtered_requests[i] += cont + thinking_end_tok
            outputs_thinking_filtered[i]["text"] = thinking_start + outputs_thinking_filtered[i]["text"] + thinking_end
    
    sampling_params["max_new_tokens"]=result_max_tokens
    print(f"sampleing params: {sampling_params}", flush=True)
    print(f"requests: {len(filtered_requests)}", flush=True)

    batch_size = 40
    filtered_outputs = []

    for i in range(0, len(filtered_requests), batch_size):
        batch = filtered_requests[i:i + batch_size]
        print(f"[phase three] processing requests {i} to {i + len(batch) - 1}", flush=True)

        outputs = evalator.generate(
            input_ids=batch,
            sampling_params=sampling_params
        )

        filtered_outputs.extend(outputs)

    for i in range(len(filtered_outputs)):
        outputs_thinking_filtered[i]["text"] += filtered_outputs[i]["text"]
    print("phase three over", flush=True)
    return outputs_thinking_filtered

def thinking_generate(
        small_model, 
        evalator, 
        evalator_tokenizer, 
        origin_text, 
        requests, 
        stop, 
        filter_top, 
        small_model_max_tokens, 
        evalator_max_tokens, 
        result_max_tokens,
    ):
    until_thinking = ["<|im_start|>"]
    until_thinking.extend(stop)
    until_thinking_tok = evalator_tokenizer(until_thinking)["input_ids"]
    thinking_n_ignore = 1
    thinking_n_ignore_str = " <|im_start|> Think Wait "
    thinking_start = "<|im_start|>think"
    thinking_start_tok = evalator_tokenizer(thinking_start)["input_ids"]
    thinking_end = "<|im_start|>answer"
    thinking_end_tok = evalator_tokenizer(thinking_end)["input_ids"]
    thinking_end_max = thinking_end + "\nFinal Answer:"
    thinking_end_max_tok = evalator_tokenizer(thinking_end_max)["input_ids"]
    thinking_n_ignore_str_tok = evalator_tokenizer(thinking_n_ignore_str)["input_ids"]
    newline_tok = evalator_tokenizer("\n")["input_ids"]
    
    start_time = time.time()

    sampling_params = {
        "max_new_tokens": small_model_max_tokens,
        "temperature": 0.7,
    }
    print(f"sampleing params: {sampling_params}", flush=True)
    print(f"requests: {len(requests)}", flush=True)
    output_small_model = []
    temp_group_size = 40
    for i in range(0, len(requests), temp_group_size):
        batch_requests = requests[i:i + temp_group_size]
        
        batch_outputs = small_model.generate(
            input_ids=batch_requests,
            sampling_params=sampling_params,
        )
        
        output_small_model.extend(batch_outputs)
    print("phase one over", flush=True)
    next_input_text = []
    for idx, output in enumerate(output_small_model):
        text = origin_text[idx] + output["text"] + thinking_n_ignore_str
        next_input_text.append(text)
    end_time1 = time.time()
    top_indices = range(len(next_input_text))

    filtered_next_input_text = [next_input_text[i] for i in top_indices]
    filtered_requests = evalator_tokenizer(filtered_next_input_text)["input_ids"]
    outputs_thinking_filtered = [output_small_model[idx] for idx in top_indices]
    for i in range(len(outputs_thinking_filtered)):
        outputs_thinking_filtered[i]["text"] = filtered_next_input_text[i]
        outputs_thinking_filtered[i]["token_ids"] = filtered_requests[i]

    outputs_thinking_filtered = sglang_wait_think(
        evalator,
        evalator_tokenizer,
        outputs_thinking_filtered,
        sampling_params,
        filtered_requests,
        evalator_max_tokens, 
        result_max_tokens,
        thinking_n_ignore,
        thinking_n_ignore_str,
        thinking_n_ignore_str_tok,
        thinking_start,
        thinking_end_max,
        thinking_end_max_tok,
        newline_tok,
        thinking_end_tok,
        thinking_end,
    )

    end_time2 = time.time()
    elapsed_time_ms = (end_time1 - start_time) * 1000
    print(f"cost {elapsed_time_ms:.3f} ms", flush=True)
    elapsed_time_ms = (end_time2 - start_time) * 1000
    print(f"cost {elapsed_time_ms:.3f} ms", flush=True)
    idx = 0
    last_output = [None] * len(requests)
    for i in top_indices:
        last_output[i] = output_small_model[i]
        last_output[i]["text"] = outputs_thinking_filtered[idx]["text"]
        last_output[i]["text"] = last_output[i]["text"].removeprefix(thinking_start + origin_text[i])
        last_output[i]["meta_info"]["completion_tokens"] = len(model_tokenizer(last_output[i]["text"])["input_ids"])
        del last_output[i]["token_ids"]
        idx += 1

    for i in range(len(requests)):
        if last_output[i] is None:
            last_output[i] = output_small_model[i]

    return last_output

def my_rank(
    evalator,
    evalator_tokenizer, 
    content, 
    params, 
    origin_text,
    group_size,
):
    inputs = evalator_tokenizer(content)["input_ids"]
    request_token = evalator_tokenizer(origin_text)["input_ids"]
    
    answers_ppl = []
    for i in range(len(inputs)):
        output = evalator.generate(
            input_ids=inputs[i],
            sampling_params={
                "max_new_tokens": 1,
                "temperature": 0.0,
            },
            return_logprob=True,
            logprob_start_len=0,
            top_logprobs_num=1,
        )
        # print(output)
        prompt_logprobs = output["meta_info"]["input_token_logprobs"][len(request_token[i]):]
        avg_cot_logprob = -sum([item[0] for item in prompt_logprobs]) / len(prompt_logprobs)
        avg_cot_ppl = math.exp(avg_cot_logprob)
        answers_ppl.append(avg_cot_ppl)

    print(answers_ppl)

    index_list = []
    num_groups = len(answers_ppl) // group_size
    best_number = group_size // 2
    for g in range(num_groups):
        # normal remove
        group_index_map = []
        group_scores = answers_ppl[g * group_size:(g + 1) * group_size]

        # sorted_indices = sorted(range(group_size), key=lambda i: group_scores[i])
        # best_indices = sorted_indices[4:]
        # for indix in best_indices:
        #     group_index_map.append(indix + g * group_size)

        random.seed(int(time.time())) 
        random_numbers = random.sample(range(8), 4)
        for indix in random_numbers:
            group_index_map.append(indix + g * group_size)

        index_list.extend(group_index_map)

        # conformal remove
        # group_scores = answers_ppl[g * group_size:(g + 1) * group_size]
        # indices = [i for i, score in enumerate(group_scores) if score < 1.123527996400051]
        # pick_indices = []
        # if indices:
        #     for item in indices:
        #         pick_indices.append(item + g * group_size)
        # index_list.extend(pick_indices)
    print(index_list)
    return index_list

def remove_generate(
        small_model, 
        evalator, 
        evalator_tokenizer, 
        origin_text, 
        requests, 
        stop, 
        filter_top, 
        small_model_max_tokens, 
        group_size,
    ):
    until_thinking = ["<|im_start|>"]
    until_thinking.extend(stop)
    until_thinking_tok = evalator_tokenizer(until_thinking)["input_ids"]
    thinking_n_ignore = 1
    thinking_n_ignore_str = " <|im_start|> Think Wait "
    thinking_start = "<|im_start|>think"
    thinking_start_tok = evalator_tokenizer(thinking_start)["input_ids"]
    thinking_end = "<|im_start|>answer"
    thinking_end_tok = evalator_tokenizer(thinking_end)["input_ids"]
    thinking_end_max = thinking_end + "\nFinal Answer:"
    thinking_end_max_tok = evalator_tokenizer(thinking_end_max)["input_ids"]
    thinking_n_ignore_str_tok = evalator_tokenizer(thinking_n_ignore_str)["input_ids"]
    newline_tok = evalator_tokenizer("\n")["input_ids"]
    
    sampling_params = {
        "max_new_tokens": small_model_max_tokens,
        "temperature": 0.7,
    }
    output_small_model = small_model.generate(
        input_ids=requests,
        sampling_params=sampling_params,
    )
    print("phase one over", flush=True)
    next_input_text = []
    for idx, output in enumerate(output_small_model):
        text = origin_text[idx] + output["text"] + thinking_n_ignore_str
        next_input_text.append(text)

    index_list = my_rank(evalator, evalator_tokenizer, next_input_text, sampling_params, origin_text, group_size)
    replaced_outputs = []

    num_groups = len(output_small_model) // group_size

    for item in index_list:
        replaced_outputs.append(output_small_model[item])

    return replaced_outputs

def origin_generate(
        small_model, 
        requests, 
        small_model_max_tokens, 
    ):
    
    sampling_params = {
        "max_new_tokens": small_model_max_tokens,
        "temperature": 0.7,
        "stop": ["Human:", "<|im_end|>"],
    }
    output_small_model = small_model.generate(
        input_ids=requests,
        sampling_params=sampling_params,
    )
    print("phase one over", flush=True)

    return output_small_model

def direct_rank(
    evalator,
    evalator_tokenizer, 
    content, 
    params, 
    origin_text,
    group_size,
):
    inputs = evalator_tokenizer(content)["input_ids"]
    request_token = evalator_tokenizer(origin_text)["input_ids"]
    
    answers_ppl = []
    for i in range(len(inputs)):
        output = evalator.generate(
            input_ids=inputs[i],
            sampling_params={
                "max_new_tokens": 1,
                "temperature": 0.0,
            },
            return_logprob=True,
            logprob_start_len=0,
            top_logprobs_num=1,
        )
        prompt_logprobs = output["meta_info"]["input_token_logprobs"][len(request_token[i]):]
        avg_cot_logprob = -sum([item[0] for item in prompt_logprobs]) / len(prompt_logprobs)
        avg_cot_ppl = math.exp(avg_cot_logprob)
        answers_ppl.append(avg_cot_ppl)

    print(answers_ppl)
    random_index_list = []

    indexed_scores = list(enumerate(answers_ppl))

    sorted_scores = sorted(indexed_scores, key=lambda x: x[1])

    n = len(answers_ppl)
    top_25_percent = int(n * 0.50)
    bottom_75_percent = n - top_25_percent
    top_25_indices = [index for index, score in sorted_scores[:top_25_percent]]
    bottom_75_indices = [index for index, score in sorted_scores[top_25_percent:]]

    return top_25_indices, random_index_list

def cumulative_rank(
    evalator,
    evalator_tokenizer, 
    content, 
    params, 
    origin_text,
    group_size,
):
    inputs = evalator_tokenizer(content)["input_ids"]
    request_token = evalator_tokenizer(origin_text)["input_ids"]
    
    answers_ppl = []
    for i in range(len(inputs)):
        output = evalator.generate(
            input_ids=inputs[i],
            sampling_params={
                "max_new_tokens": 1,
                "temperature": 0.0,
            },
            return_logprob=True,
            logprob_start_len=0,
            top_logprobs_num=1,
        )
        prompt_logprobs = output["meta_info"]["input_token_logprobs"][len(request_token[i]):]
        avg_cot_logprob = -sum([item[0] for item in prompt_logprobs]) / len(prompt_logprobs)
        avg_cot_ppl = math.exp(avg_cot_logprob)
        answers_ppl.append(avg_cot_ppl)

    print(len(answers_ppl))

    index_list = []
    random_index_list = []
    hyper = 0.6988641
    print(f"hyper is {hyper}", flush=True)
    num_groups = len(answers_ppl) // group_size

    ##############################
    for g in range(num_groups):
        pick_indices = []
        group_scores = answers_ppl[g * group_size:(g + 1) * group_size]

        def softmax_normalize(group_scores):
            indexed_scores = [(score, idx) for idx, score in enumerate(group_scores)]
            indexed_scores.sort(key=lambda x: x[0], reverse=True)

            sorted_scores = [score for score, _ in indexed_scores]
            exp_scores = [math.exp(score) for score in sorted_scores]
            total_exp_score = sum(exp_scores)
            if total_exp_score > 0:
                normalized_scores = [exp_score / total_exp_score for exp_score in exp_scores]
            else:
                normalized_scores = [1.0 / len(group_scores)] * len(group_scores)
            cumulative_probs = []
            current_sum = 0
            for norm_score in normalized_scores:
                current_sum += norm_score
                cumulative_probs.append(current_sum)
            
            print(f"group cumulative is {cumulative_probs}", flush=True)
            pick_indices = []
            added = False
            prev_idx = None

            for idx, cum_prob in enumerate(cumulative_probs):
                if cum_prob > hyper:
                    if prev_idx is not None and not added:
                        random.seed(time.time())
                        if random.random() < 0.9:
                            original_idx = indexed_scores[prev_idx][1]
                            pick_indices.append(original_idx + g * group_size)
                    added = True

                    original_idx = indexed_scores[idx][1]
                    pick_indices.append(original_idx + g * group_size)
                else:
                    prev_idx = idx
            
            return pick_indices

        pick_indices = softmax_normalize(group_scores)
        index_list.extend(pick_indices)

    # no softmax version
    # for g in range(num_groups):
    #     pick_indices = []
    #     group_scores = answers_ppl[g * group_size:(g + 1) * group_size]

    #     total_score = sum(group_scores)
    #     if total_score > 0:
    #         normalized_scores = [score / total_score for score in group_scores]
    #     else:
    #         normalized_scores = [0] * len(group_scores)

    #     cumulative_probs = []
    #     current_sum = 0
    #     for norm_score in normalized_scores:
    #         current_sum += norm_score
    #         cumulative_probs.append(current_sum)
    #     print(f"group cumulative is {cumulative_probs}", flush=True)
    #     pick_indices = []
    #     for idx, cum_prob in enumerate(cumulative_probs):
    #         if cum_prob > hyper:
    #             pick_indices.append(idx + g * group_size)

    #     index_list.extend(pick_indices)

    return index_list, random_index_list

# def conformal_rank(
#     evalator,
#     evalator_tokenizer, 
#     content, 
#     params, 
#     origin_text,
#     group_size,
#     confidence,
#     conformal_hyperparam,
#     ranodm_enable,
# ):
    
#     inputs = evalator_tokenizer(content)["input_ids"]
#     request_token = evalator_tokenizer(origin_text)["input_ids"]
#     index_list = []
#     for i in range(len(inputs)):
#         print(f"{i} input", flush=True)
#         output = evalator.generate(
#             input_ids=inputs[i],
#             sampling_params={
#                 "max_new_tokens": 1,
#                 "temperature": 0.0,
#             },
#             return_logprob=True,
#             logprob_start_len=0,
#             top_logprobs_num=1,
#         )

#         prompt_logprobs = output["meta_info"]["input_token_logprobs"][len(request_token[i]):]
#         avg_cot_logprob = -sum([item[0] for item in prompt_logprobs]) / len(prompt_logprobs)
#         avg_cot_ppl = math.exp(avg_cot_logprob)
        
#         if avg_cot_ppl > conformal_hyperparam:
#             index_list.append(i)
#         elif ranodm_enable:
#             random.seed(int(time.time()))
#             if random.random() > confidence:
#                 index_list.append(i)
#     random_index_list = random.sample(range(len(inputs)), len(index_list))
#     return index_list, random_index_list

def naive_rank(
    evalator,
    evalator_tokenizer, 
    content, 
    params, 
    origin_text,
    group_size,
    confidence,
    conformal_hyperparam,
    ranodm_enable,
):
    
    inputs = evalator_tokenizer(content)["input_ids"]
    request_token = evalator_tokenizer(origin_text)["input_ids"]
    index_list = []
    avg_cot_ppls = []
    random_index_list = []
    for i in range(len(inputs)):
        print(f"{i} input", flush=True)
        output = evalator.generate(
            input_ids=inputs[i],
            sampling_params={
                "max_new_tokens": 1,
                "temperature": 0.0,
            },
            return_logprob=True,
            logprob_start_len=0,
            top_logprobs_num=1,
        )

        prompt_logprobs = output["meta_info"]["input_token_logprobs"][len(request_token[i]):]
        avg_cot_logprob = -sum([item[0] for item in prompt_logprobs]) / len(prompt_logprobs)
        avg_cot_ppl = math.exp(avg_cot_logprob)
        avg_cot_ppls.append(avg_cot_ppl)

    group = len(inputs) // group_size
    for g in range(group):
        start_idx = g * group_size
        end_idx = (g + 1) * group_size
        group_scores = avg_cot_ppls[start_idx:end_idx]
        group_scores = np.array(group_scores)
        exp_scores = np.exp(group_scores - np.max(group_scores))
        softmax_scores = exp_scores / np.sum(exp_scores)
        
        indexed_scores = [(score, start_idx + i) for i, score in enumerate(softmax_scores)]
        
        indexed_scores.sort(key=lambda x: x[0], reverse=True)
        
        current_sum = 0
        count = 0
        for score, _ in indexed_scores:
            current_sum += score
            count += 1
            if current_sum > confidence:
                break
        
        if ranodm_enable:
            random.seed(int(time.time()))
            V = (sum(indexed_scores[:count][0]) - confidence) / indexed_scores[count - 1][0]
            if random.random() <= V:
                count -= 1
        top_indices = [index for _, index in indexed_scores[:count]]
        index_list.extend(top_indices)

    return index_list, random_index_list

def top_rank(
    evalator,
    evalator_tokenizer, 
    content, 
    params, 
    origin_text,
    group_size,
    rank,
):
    
    inputs = evalator_tokenizer(content)["input_ids"]
    request_token = evalator_tokenizer(origin_text)["input_ids"]
    index_list = []
    avg_cot_ppls = []
    random_index_list = []
    for i in range(len(inputs)):
        print(f"{i} input", flush=True)
        output = evalator.generate(
            input_ids=inputs[i],
            sampling_params={
                "max_new_tokens": 1,
                "temperature": 0.0,
            },
            return_logprob=True,
            logprob_start_len=0,
            top_logprobs_num=1,
        )

        prompt_logprobs = output["meta_info"]["input_token_logprobs"][len(request_token[i]):]
        avg_cot_logprob = -sum([item[0] for item in prompt_logprobs]) / len(prompt_logprobs)
        avg_cot_ppl = math.exp(avg_cot_logprob)
        avg_cot_ppls.append(avg_cot_ppl)

    group = len(inputs) // group_size
    for g in range(group):
        start_idx = g * group_size
        end_idx = (g + 1) * group_size
        group_scores = avg_cot_ppls[start_idx:end_idx]
        
        indexed_scores = [(score, start_idx + i) for i, score in enumerate(group_scores)]
        indexed_scores.sort(key=lambda x: x[0], reverse=True)
        top_indices = [index for _, index in indexed_scores[:rank]]
        index_list.extend(top_indices)

    return index_list, random_index_list

def compare_calibration_rank(
    evalator,
    evalator_tokenizer, 
    content, 
    params, 
    origin_text,
    group_size,
    calibration_hyper,
):
    
    inputs = evalator_tokenizer(content)["input_ids"]
    request_token = evalator_tokenizer(origin_text)["input_ids"]
    index_list = []
    random_index_list = []
    for i in range(len(inputs)):
        print(f"{i} input", flush=True)
        output = evalator.generate(
            input_ids=inputs[i],
            sampling_params={
                "max_new_tokens": 1,
                "temperature": 0.0,
            },
            return_logprob=True,
            logprob_start_len=0,
            top_logprobs_num=1,
        )

        prompt_logprobs = output["meta_info"]["input_token_logprobs"][len(request_token[i]):]
        avg_cot_logprob = -sum([item[0] for item in prompt_logprobs]) / len(prompt_logprobs)
        avg_cot_ppl = math.exp(avg_cot_logprob)
        if avg_cot_ppl > calibration_hyper:
            index_list.append(i)

    return index_list, random_index_list

def compare_calibration_set(
    evalator,
    evalator_tokenizer, 
    content, 
    params, 
    origin_text,
    group_size,
    ppl_array,
    confidence,
):
    
    inputs = evalator_tokenizer(content)["input_ids"]
    request_token = evalator_tokenizer(origin_text)["input_ids"]
    index_list = []
    for i in range(len(inputs)):
        output = evalator.generate(
            input_ids=inputs[i][:2000],
            sampling_params={
                "max_new_tokens": 1,
                "temperature": 0.0,
            },
            return_logprob=True,
            logprob_start_len=len(request_token[i]),
            top_logprobs_num=1,
        )
        prompt_logprobs = output["meta_info"]["input_token_logprobs"][1:]
        avg_cot_logprob = -sum([item[0] for item in prompt_logprobs]) / len(prompt_logprobs)
        avg_cot_ppl = math.exp(avg_cot_logprob)
        count = 1
        for ppl in ppl_array:
            if ppl < avg_cot_ppl:
                count += 1
        p = count / (len(ppl_array) + 1)
        print(f"avg_cot_ppl is {avg_cot_ppl}, p is {p}, count is {count}, len is {len(ppl_array)}", flush=True)
        if p > confidence:
            index_list.append(i)

    return index_list

def prediction_generate(
        small_model, 
        evalator, 
        evalator_tokenizer, 
        origin_text, 
        requests, 
        stop, 
        filter_top, 
        small_model_max_tokens, 
        evalator_max_tokens, 
        result_max_tokens,
        group_size,
        confidence,
        conformal_hyperparam,
        ppl_array,
        random_enable=True,
    ):
    until_thinking = ["<|im_start|>"]
    until_thinking.extend(stop)
    until_thinking_tok = evalator_tokenizer(until_thinking)["input_ids"]
    thinking_n_ignore = 1
    thinking_n_ignore_str = " <|im_start|> Think Wait "
    thinking_start = "<|im_start|>think"
    thinking_start_tok = evalator_tokenizer(thinking_start)["input_ids"]
    thinking_end = "<|im_start|>answer"
    thinking_end_tok = evalator_tokenizer(thinking_end)["input_ids"]
    thinking_end_max = thinking_end + "\nFinal Answer:"
    thinking_end_max_tok = evalator_tokenizer(thinking_end_max)["input_ids"]
    thinking_n_ignore_str_tok = evalator_tokenizer(thinking_n_ignore_str)["input_ids"]
    newline_tok = evalator_tokenizer("\n")["input_ids"]
    
    start_time = time.time()

    sampling_params = {
        "max_new_tokens": small_model_max_tokens,
        "temperature": 0.7,
        "stop": ["Human:", "<|im_end|>"],
    }
    print(f"sampleing params: {sampling_params}", flush=True)
    print(f"requests: {len(requests)}", flush=True)
    output_small_model = []
    temp_group_size = 40
    for i in range(0, len(requests), temp_group_size):
        batch_requests = requests[i:i + temp_group_size]
        
        batch_outputs = small_model.generate(
            input_ids=batch_requests,
            sampling_params=sampling_params,
        )
        
        output_small_model.extend(batch_outputs)

    print("phase one over", flush=True)
    next_input_text = []
    for idx, output in enumerate(output_small_model):
        text = origin_text[idx] + output["text"] + thinking_n_ignore_str
        next_input_text.append(text)
    end_time1 = time.time()

    # top_indices = my_rank(evalator, evalator_tokenizer, next_input_text, sampling_params, origin_text, group_size)
    # top_indices, random_indices = cumulative_rank(evalator, evalator_tokenizer, next_input_text, sampling_params, origin_text, group_size)
    
    # top_indices = naive_rank(evalator, evalator_tokenizer, next_input_text, sampling_params, origin_text, group_size, confidence, conformal_hyperparam, random_enable)
    # top_indices = top_rank(evalator, evalator_tokenizer, next_input_text, sampling_params, origin_text, group_size, rank=2)
    # top_indices = compare_calibration_rank(evalator, evalator_tokenizer, next_input_text, sampling_params, origin_text, group_size, calibration_hyper)
    top_indices = compare_calibration_set(evalator, evalator_tokenizer, next_input_text, sampling_params, origin_text, group_size, ppl_array, confidence)
    
    # top_indices = random.sample(range(664 * 16), len(top_indices))
    print(top_indices)
    print(len(top_indices))

    # top_indices = []
    # for i in range(0, len(output_small_model), group_size):
    #     group_start = i
    #     group_end = min(i + group_size, len(output_small_model))
    #     group_indices = list(range(group_start, group_end))
    #     selected_group_indices = random.sample(group_indices, min(6, len(group_indices)))
    #     top_indices.extend(selected_group_indices)

    filtered_next_input_text = [next_input_text[i] for i in top_indices]
    filtered_requests = evalator_tokenizer(filtered_next_input_text)["input_ids"]
    outputs_thinking_filtered = [output_small_model[idx] for idx in top_indices]
    for i in range(len(outputs_thinking_filtered)):
        outputs_thinking_filtered[i]["text"] = filtered_next_input_text[i]
        outputs_thinking_filtered[i]["token_ids"] = filtered_requests[i]

    outputs_thinking_filtered = sglang_wait_think(
        evalator,
        evalator_tokenizer,
        outputs_thinking_filtered,
        sampling_params,
        filtered_requests,
        evalator_max_tokens, 
        result_max_tokens,
        thinking_n_ignore,
        thinking_n_ignore_str,
        thinking_n_ignore_str_tok,
        thinking_start,
        thinking_end_max,
        thinking_end_max_tok,
        newline_tok,
        thinking_end_tok,
        thinking_end,
    )

    ##############################################################

    # filtered_next_input_text = [next_input_text[i] for i in random_indices]
    # filtered_requests = evalator_tokenizer(filtered_next_input_text)["input_ids"]
    # outputs_thinking_filtered1 = [output_small_model1[idx] for idx in random_indices]
    # for i in range(len(outputs_thinking_filtered1)):
    #     outputs_thinking_filtered1[i]["text"] = filtered_next_input_text[i]
    #     outputs_thinking_filtered1[i]["token_ids"] = filtered_requests[i]

    # outputs_thinking_filtered1 = sglang_wait_think(
    #     evalator,
    #     evalator_tokenizer,
    #     outputs_thinking_filtered1,
    #     sampling_params,
    #     filtered_requests,
    #     evalator_max_tokens, 
    #     result_max_tokens,
    #     thinking_n_ignore,
    #     thinking_n_ignore_str,
    #     thinking_n_ignore_str_tok,
    #     thinking_start,
    #     thinking_end_max,
    #     thinking_end_max_tok,
    #     newline_tok,
    #     thinking_end_tok,
    #     thinking_end,
    # )

    end_time2 = time.time()
    elapsed_time_ms = (end_time1 - start_time) * 1000
    print(f"cost {elapsed_time_ms:.3f} ms", flush=True)
    elapsed_time_ms = (end_time2 - start_time) * 1000
    print(f"cost {elapsed_time_ms:.3f} ms", flush=True)
    idx = 0
    last_output = [None] * len(requests)
    for i in top_indices:
        last_output[i] = output_small_model[i]
        last_output[i]["text"] = outputs_thinking_filtered[idx]["text"]
        last_output[i]["text"] = last_output[i]["text"].removeprefix(thinking_start + origin_text[i])
        last_output[i]["meta_info"]["completion_tokens"] = len(model_tokenizer(last_output[i]["text"])["input_ids"])
        del last_output[i]["token_ids"]
        idx += 1

    for i in range(len(requests)):
        if last_output[i] is None:
            last_output[i] = output_small_model[i]

    ########################################################
    
    # idx = 0
    # random_last_output = [None] * len(requests)
    # for i in random_indices:
    #     random_last_output[i] = output_small_model1[i]
    #     random_last_output[i]["text"] = outputs_thinking_filtered1[idx]["text"]
    #     random_last_output[i]["text"] = random_last_output[i]["text"].removeprefix(thinking_start + origin_text[i])
    #     random_last_output[i]["meta_info"]["completion_tokens"] = len(model_tokenizer(random_last_output[i]["text"])["input_ids"])
    #     del random_last_output[i]["token_ids"]
    #     idx += 1

    # for i in range(len(requests)):
    #     if random_last_output[i] is None:
    #         random_last_output[i] = output_small_model1[i]

    return last_output


if __name__ == "__main__":
    repeats = 4
    confidence = 0.7
    conformal_hyperparam = 1
    random_enable = True
    context, answer = load_my_dataset("aime24", repeats)
    ppl_array = np.load('/zju_0038/xj/sglang-parallel-test-time-scaling/ppls_aime24_4_s1.1-14B_llama.npy')
    dictionary = dict()
    for item in zip(context, answer):
        dictionary[item[0]] = item[1]

    evalator = None
    evalator_tokenizer = None
    small_model = None
    model_tokenizer = None

    # small_model = agent("Qwen/Qwen2.5-7B-Instruct", gpu=0).model
    # model_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    # small_model = agent("meta-llama/Llama-3.1-70B-Instruct", gpu=0, tp=2, dp=1).model
    # model_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-70B-Instruct")
    small_model = agent("meta-llama/Llama-3.1-8B-Instruct", gpu=0).model
    model_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    # small_model = agent("simplescaling/s1.1-14B", gpu=0).model
    # model_tokenizer = AutoTokenizer.from_pretrained("simplescaling/s1.1-14B")

    evalator = agent("simplescaling/s1.1-14B", gpu=1).model
    evalator_tokenizer = AutoTokenizer.from_pretrained("simplescaling/s1.1-14B") 
    # evalator = agent("simplescaling/s1.1-7B", gpu=1).model
    # evalator_tokenizer = AutoTokenizer.from_pretrained("simplescaling/s1.1-7B")
    # evalator = agent("meta-llama/Llama-3.1-70B-Instruct", gpu=1, tp=2, dp=1).model
    # evalator_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-70B-Instruct")
    context_encoding = model_tokenizer(context)["input_ids"]
    until = ["Question:", "</s>", "<|im_end|>"]

    generation_mode = "prediction"
    print(f"generation mode is {generation_mode}", flush=True)
    small_model_max_tokens = 10000
    evalator_max_tokens = 5000
    result_max_tokens = 200
    
    # small_model_max_tokens = 1000
    # evalator_max_tokens = 1500
    # result_max_tokens = 1000
    filter_top = 20

    if generation_mode == "thinking":
        cont = thinking_generate(
            small_model=small_model,
            evalator=evalator,
            evalator_tokenizer=evalator_tokenizer,
            origin_text=context,
            requests=context_encoding,
            stop=until,
            filter_top=filter_top,
            small_model_max_tokens=small_model_max_tokens,
            evalator_max_tokens=evalator_max_tokens,
            result_max_tokens=result_max_tokens,
        )
    elif generation_mode == "remove":
        cont = remove_generate(
            small_model=small_model,
            evalator=evalator,
            evalator_tokenizer=evalator_tokenizer,
            origin_text=context,
            requests=context_encoding,
            stop=until,
            filter_top=filter_top,
            small_model_max_tokens=small_model_max_tokens,
            group_size=repeats
        )
    elif generation_mode == "prediction":
        cont = prediction_generate(
            small_model=small_model,
            evalator=evalator,
            evalator_tokenizer=evalator_tokenizer,
            origin_text=context,
            requests=context_encoding,
            stop=until,
            filter_top=filter_top,
            small_model_max_tokens=small_model_max_tokens,
            evalator_max_tokens=evalator_max_tokens,
            result_max_tokens=result_max_tokens,
            group_size=repeats,
            confidence=confidence,
            conformal_hyperparam=conformal_hyperparam,
            ppl_array=ppl_array,
            random_enable=random_enable,
        )
    elif generation_mode == "origin":
        cont = origin_generate(
            small_model=small_model,
            requests=context_encoding,
            small_model_max_tokens=small_model_max_tokens,
        )
    
    # new_context = []
    # for i in range(0, len(context), repeats):
    #     new_context.extend(context[i:i + repeats // 2]) 
    # sglang_outputs = list(zip(new_context, cont))
    # evaluate(dictionary, sglang_outputs, repeats // 2)

    sglang_outputs = list(zip(context, cont))
    evaluate(dictionary, sglang_outputs, repeats)

    # random_sglang_outputs = list(zip(context, random_cont))
    # evaluate(dictionary, random_sglang_outputs, repeats)
