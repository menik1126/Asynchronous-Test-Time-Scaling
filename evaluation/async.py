import httpx
import asyncio
import numpy as np
import math
from transformers import AutoTokenizer
from dataset import load_my_dataset
from async_agent import anyone_check
import time
import re
from tqdm.asyncio import tqdm
from tqdm import tqdm as sync_tqdm

client_small = None
client_eval = None
semaphore = asyncio.Semaphore(8)
small_model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
eval_model_name = "Qwen/QwQ-32B"
tokenizer = AutoTokenizer.from_pretrained("Qwen/QwQ-32B")
tokenizer.use_default_system_prompt = True
small_tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
small_tokenizer.use_default_system_prompt = True

def build_debug_prompt():
    message = [
        {"role": "system", "content": "You are a creative and expressive assistant. Feel free to write anything you want, in any format or style."},
        {"role": "user", "content": "Go ahead and write freely. No need to stop."},
    ]

    return message

def build_question(question):
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
    # return " <think> " + "\n\n".join([f"\n{h}\n" for h in history])
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

async def call_small_model(prompt, turn, max_tokens, idx):

    # messages = build_debug_prompt()
    messages = (
        build_small_init_prompt(prompt[0]) if turn == 0 else build_small_inner_prompt(prompt[0], prompt[1])
    )

    #print(f"history: {prompt[1]}", flush=True)

    # print(f"Small model prompt: {messages}", flush=True)

    global semaphore, client_small, small_model_name

    payload = {
        "model": small_model_name,
        "messages": messages,
        "temperature": 0.8,
        "max_tokens": max_tokens,
        # "stop": ["\n\n"],
    }

    async with semaphore:
        resp = await client_small.post(
            "http://127.0.0.1:51101/v1/chat/completions",
            json=payload,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

async def call_eval_model(prompt, max_tokens, idx):

    messages = build_eval_prompt_for_generate(prompt[0], prompt[1])
    # messages = build_debug_prompt()
    global semaphore, client_eval, eval_model_name
    payload = {
        "model": eval_model_name,
        "messages": messages,
        "temperature": 0.8,
        "max_tokens": max_tokens,
        # "stop": ["\n\n"],
    }

    async with semaphore:
        resp = await client_eval.post(
            "http://127.0.0.1:51100/v1/chat/completions",
            json=payload,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

async def call_eval_model_ppl(prompt, idx):

    global client_eval, tokenizer
    message = build_eval_prompt_for_eval(prompt[0], prompt[1])

    # tokens_full = tokenizer.tokenize(message)
    prompt[1][-1] = prompt[1][-1].strip('\n')
    # tokens_prompt = tokenizer.tokenize(prompt[1][-1])

    position = message.find(prompt[1][-1])
    sub_message = message[:position]
    logprob_start_len = len(tokenizer.tokenize(sub_message))
    if logprob_start_len == -1:
        print(message)
        print("---------------------------")
        print(prompt[1][-1])
        raise ValueError("Prompt tokens not found in full tokens.")

    target_part = tokenizer.tokenize(prompt[1][-1])
    # print(f"logprob_start_len: {logprob_start_len}", flush=True)
    # print(f"Target part lens: {len(target_part)}", flush=True)

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
    async with semaphore:
        resp = await client_eval.post(
            "http://127.0.0.1:51100/generate",
            json=payload,
        )

        resp.raise_for_status()
        data = resp.json()
        input_token_logprobs = data['meta_info']['input_token_logprobs'][1:]
        logprobs = [entry[0] for entry in input_token_logprobs if entry[0] is not None]

        if not logprobs:
            print(f"No log probabilities returned for history: {prompt[1]}", flush=True)
            raise ValueError(f"No log probabilities returned for problem: {prompt[0]}")

        avg_neg_logprob = -sum(logprobs) / len(logprobs)
        return math.exp(avg_neg_logprob)

async def extract_answer(history):

    answer = "invalid"

    temp = "\n\n".join([

        f"{history[i]}"

        for i in range(len(history))

    ])

    # print(f"lens is {len(history)}")

    # print(f"History: {temp}", flush=True)

    # print("-------------------------------------------------")



    matches = re.findall(r"\\boxed\{(.*?)\}", temp)
    if matches:
        answer = matches[-1]
    else:
        pattern = re.compile(r"ANSWER:\s+([A-Z])", re.IGNORECASE)
        matches = pattern.findall(temp)
        # matches = re.findall(r"ANSWER:\s+([A-Z])", temp)
        if matches:
            answer = matches[-1]

    return answer

async def process_single_problem(problem, small_model_max_tokens, evalator_max_tokens, ppl_array, turns, idx):

    prompt = [problem, []]

    answer = "invalid"

    start_time = time.time()



    for turn in range(turns):

        small_out = await call_small_model(prompt, turn, small_model_max_tokens, idx)

        # print(f"Small model output: {small_out}", flush=True)

        prompt[1].append(small_out)

        if not small_out:

            print(f"Small model returned empty output for problem", flush=True)

            print(f"history: {prompt[1]}", flush=True)

            break

           

        ppl = await call_eval_model_ppl(prompt, idx)

        rank = np.sum(ppl_array < ppl)

        percent = rank / len(ppl_array)

        if percent >= 0.7:

            eval_out = await call_eval_model(prompt, evalator_max_tokens, idx)

            # print(f"Eval model output: {eval_out}", flush=True)

            prompt[1].append(eval_out)



        temp = await extract_answer(prompt[1])

        if temp != "invalid":

            answer = temp

            print("early stop")

            break



    if answer == "invalid":

        answer = await extract_answer(prompt[1])



    end_time = time.time()

    duration = end_time - start_time

    return answer, duration





async def fully_async_generate(problems, small_model_max_tokens, evalator_max_tokens, ppl_array, turns):
    tasks = [
        asyncio.create_task(
            process_single_problem(p, small_model_max_tokens, evalator_max_tokens, ppl_array, turns, idx)
        )
        for idx, p in enumerate(problems)
    ]
    results = await tqdm.gather(*tasks, desc="Processing problems")
    durations = [duration for _, duration in results]
    avg_time = sum(durations) / len(durations)
    print(f"Average total time per problem: {avg_time:.4f}s")
    return results

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
    global client_small, client_eval

    client_small = httpx.AsyncClient(
        timeout=240.0, 
        limits=httpx.Limits(
            max_connections=1000,         
            max_keepalive_connections=1000
        )
    )

    client_eval = httpx.AsyncClient(
        timeout=240.0,
        limits=httpx.Limits(
            max_connections=1000,     
            max_keepalive_connections=1000
        )
    )

    repeats = 16
    context, answer = load_my_dataset("gpqa", repeats)
    ppl_array = np.load('/zju_0038/xj/sglang-parallel-test-time-scaling/ppls_gpqa_64_qwq_32_r1_1.npy')
    # ppl_array = None
    turns = 15
    small_model_max_tokens = 500
    evalator_max_tokens = 500

    results = []
    start = time.time()
    total_problems = int(len(context) / repeats)
    for idx in sync_tqdm(range(total_problems), desc="Processing problem groups"):
        problem = context[idx * repeats: (idx + 1) * repeats]
        result = await fully_async_generate(
            problem,
            small_model_max_tokens,
            evalator_max_tokens,
            ppl_array,
            turns,
        )
        results.extend(result)

    end = time.time()
    print(f"Elapsed: {end - start:.3f} s")
    await compute_score(results, answer, repeats)
    await client_small.aclose()
    await client_eval.aclose()

if __name__ == "__main__":

    asyncio.run(main())
