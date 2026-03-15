"""Single-model asynchronous test-time scaling with rejection sampling + cross-sample forking.

Extends ref_async_self_reject.py with a shared sample pool. When a sample is
rejected, instead of blindly resampling from its own prefix, it can **fork**
from the globally best sample (scored by avg_ppl - alpha * current_turn).

Key differences from ref_async_self_reject.py:
  - All samples of the same problem share a SamplePool with async locking.
  - Each sample tracks cumulative PPL history (avg_ppl over accepted turns).
  - On rejection, the sample forks from the best-scored peer in the pool,
    copying its full history and PPL record, then continues with a diversity
    perturbation and higher temperature.
  - A progress reward (alpha * current_turn) prevents short chains from
    dominating the score purely due to having fewer turns.
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
from copy import deepcopy
from dataclasses import dataclass, field

from tqdm.asyncio import tqdm
from tqdm import tqdm as sync_tqdm
from transformers import AutoTokenizer

from ATTS.dataset import load_my_dataset
from ATTS.async_agent import anyone_check

client = None
http_semaphore = None
model_name = ""
tokenizer = None
max_retries = 3
extract_mode = "regex"


# ---------------------------------------------------------------------------
# SampleState + SamplePool
# ---------------------------------------------------------------------------

@dataclass
class SampleState:
    idx: int
    history: list = field(default_factory=list)
    ppl_history: list = field(default_factory=list)
    current_turn: int = 0
    finished: bool = False

    @property
    def avg_ppl(self) -> float:
        if not self.ppl_history:
            return float("inf")
        return sum(self.ppl_history) / len(self.ppl_history)

    def score(self, alpha: float) -> float:
        """Lower is better. Avg PPL penalised, progress rewarded."""
        return self.avg_ppl - alpha * self.current_turn


class SamplePool:
    """Thread-safe (asyncio-safe) shared state for all samples of one problem."""

    def __init__(self):
        self._states: dict[int, SampleState] = {}
        self._lock = asyncio.Lock()

    def register(self, idx: int):
        self._states[idx] = SampleState(idx=idx)

    async def update(self, idx: int, history: list, ppl_history: list, turn: int):
        async with self._lock:
            st = self._states[idx]
            st.history = list(history)
            st.ppl_history = list(ppl_history)
            st.current_turn = turn

    async def mark_finished(self, idx: int):
        async with self._lock:
            self._states[idx].finished = True

    async def best_peer(self, exclude_idx: int, alpha: float):
        """Return a *snapshot* of the best-scored peer, or None."""
        async with self._lock:
            candidates = [
                s for s in self._states.values()
                if s.idx != exclude_idx
                and len(s.history) > 0
                and not s.finished
            ]
            if not candidates:
                return None
            best = min(candidates, key=lambda s: s.score(alpha))
            return SampleState(
                idx=best.idx,
                history=list(best.history),
                ppl_history=list(best.ppl_history),
                current_turn=best.current_turn,
            )


# ---------------------------------------------------------------------------
# Network helpers (identical to ref_async_self_reject.py)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Prompt builders (identical to ref_async_self_reject.py)
# ---------------------------------------------------------------------------

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
        {"role": "user", "content": build_question(question)},
        {"role": "assistant", "content": build_cot(history)},
    ]


# ---------------------------------------------------------------------------
# Model calls (identical to ref_async_self_reject.py)
# ---------------------------------------------------------------------------

async def call_model_generate(prompt, turn, max_tokens, idx, port, temperature):
    messages = (
        build_init_prompt(prompt[0])
        if turn == 0
        else build_continue_prompt(prompt[0], prompt[1])
    )

    global http_semaphore, client, model_name
    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    async with http_semaphore:
        resp = await _post_with_retry(client, f"http://127.0.0.1:{port}/v1/chat/completions", payload)
        data = resp.json()
        return data["choices"][0]["message"]["content"]


async def call_model_ppl(prompt, turn, idx, port):
    global tokenizer, client, http_semaphore

    if turn == 0:
        messages = build_init_prompt(prompt[0])
    else:
        messages = [{"role": "user", "content": build_question(prompt[0])}]

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

    async with http_semaphore:
        resp = await _post_with_retry(client, f"http://127.0.0.1:{port}/generate", payload)
        data = resp.json()
        input_token_logprobs = data["meta_info"]["input_token_logprobs"][1:]
        logprobs = [entry[0] for entry in input_token_logprobs if entry[0] is not None]

        if not logprobs:
            print(f"[idx={idx}] No logprobs at turn {turn}, ppl=inf", flush=True)
            return float("inf")

        avg_neg_logprob = -sum(logprobs) / len(logprobs)
        return math.exp(avg_neg_logprob)


# ---------------------------------------------------------------------------
# Answer extraction (identical to ref_async_self_reject.py)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Core logic – process_single_problem with cross-sample forking
# ---------------------------------------------------------------------------

DIVERSITY_PROMPTS = [
    "Wait, let me reconsider this problem from a different angle.\n\n",
    "Hmm, I think there might be a simpler approach. Let me try again.\n\n",
    "Let me verify my reasoning by trying an alternative method.\n\n",
    "I should double-check this. Let me rethink step by step.\n\n",
]


async def process_single_problem(
    problem,
    max_tokens,
    ppl_array,
    turns,
    idx,
    port,
    temperature,
    output_dir,
    ppl_threshold,
    max_reject_attempts,
    pool: SamplePool,
    alpha: float,
    fork_temperature: float,
    fork_gap: float = 0.05,
):
    """Multi-turn loop with rejection sampling + cross-sample forking.

    Each turn:
      1. Generate a draft.
      2. Compute PPL via prefill (prefix-cache friendly).
      3. If PPL percentile >= threshold → reject.
         After max_reject_attempts failures, try to fork from the best peer.
         If no peer available, keep the best rejected output.
      4. If PPL percentile < threshold → accept.
      5. Publish updated state to the shared pool.
      6. Check for a valid answer; early-stop if found.
    """
    prompt = [problem, []]
    answer = "invalid"
    start_time = time.time()
    history_log = []
    ppl_record = []
    forked_from = None

    for turn in range(turns):
        best_out = None
        best_ppl = float("inf")
        accepted = False

        cur_temp = fork_temperature if forked_from is not None else temperature
        if forked_from is not None:
            forked_from = None

        for attempt in range(max_reject_attempts):
            out = await call_model_generate(prompt, turn, max_tokens, idx, port, cur_temp)

            if not out:
                print(f"[idx={idx}] Empty output at turn {turn} attempt {attempt}", flush=True)
                break

            prompt[1].append(out)

            temp_ans = await extract_answer(prompt[1])
            if temp_ans != "invalid":
                ppl = await call_model_ppl(prompt, turn, idx, port)
                ppl_record.append(float(ppl))
                history_log.append({
                    "turn": turn, "attempt": attempt,
                    "model": "draft", "output": out,
                    "ppl": float(ppl), "percentile": 0.0,
                    "accepted": True,
                    "accepted_reason": "answer_found",
                })
                answer = temp_ans
                accepted = True
                print(f"[idx={idx}] Answer found at turn {turn} attempt {attempt}, skip PPL check", flush=True)
                break

            ppl = await call_model_ppl(prompt, turn, idx, port)
            prompt[1].pop()

            rank = np.sum(ppl_array < ppl)
            percent = rank / len(ppl_array)

            history_log.append({
                "turn": turn, "attempt": attempt,
                "model": "draft", "output": out,
                "ppl": float(ppl), "percentile": float(percent),
                "accepted": bool(percent < ppl_threshold),
            })

            if ppl < best_ppl:
                best_ppl = ppl
                best_out = out

            if percent < ppl_threshold:
                accepted = True
                prompt[1].append(out)
                ppl_record.append(float(ppl))
                break
            else:
                if attempt < max_reject_attempts - 1:
                    print(
                        f"[idx={idx}] Rejected at turn {turn} attempt {attempt} "
                        f"(ppl={ppl:.2f}, pct={percent:.2f})",
                        flush=True,
                    )

        if answer != "invalid":
            await pool.update(idx, prompt[1], ppl_record, turn + 1)
            print(f"[idx={idx}] Early stop at turn {turn}", flush=True)
            break

        if not accepted:
            # --- Cross-sample fork ---
            peer = await pool.best_peer(exclude_idx=idx, alpha=alpha)

            my_avg_ppl = (
                sum(ppl_record) / len(ppl_record) if ppl_record else float("inf")
            )
            my_score = my_avg_ppl - alpha * turn
            peer_avg_ppl = peer.avg_ppl if peer else float("inf")
            peer_score = peer.score(alpha) if peer else float("inf")

            score_gap = my_score - peer_score if peer is not None else 0.0
            ppl_gap = my_avg_ppl - peer_avg_ppl if peer is not None else 0.0
            if peer is not None and score_gap >= fork_gap and ppl_gap >= 0.02:
                prompt[1] = list(peer.history)
                ppl_record = list(peer.ppl_history)
                forked_turn = peer.current_turn
                forked_from = peer.idx

                diversity_hint = DIVERSITY_PROMPTS[idx % len(DIVERSITY_PROMPTS)]
                prompt[1].append(diversity_hint)
                ppl_record.append(best_ppl if best_ppl < float("inf") else 3.0)

                history_log.append({
                    "turn": turn, "event": "fork",
                    "forked_from": peer.idx,
                    "peer_score": float(peer_score),
                    "my_score": float(my_score),
                    "score_gap": float(score_gap),
                    "my_avg_ppl": float(my_avg_ppl),
                    "peer_avg_ppl": float(peer_avg_ppl),
                    "ppl_gap": float(ppl_gap),
                    "peer_turn": forked_turn,
                })

                print(
                    f"[idx={idx}] Forked from idx={peer.idx} at turn {turn} "
                    f"(my_ppl={my_avg_ppl:.3f}, peer_ppl={peer_avg_ppl:.3f}, ppl_gap={ppl_gap:.3f})",
                    flush=True,
                )
            elif best_out is not None:
                prompt[1].append(best_out)
                ppl_record.append(float(best_ppl))
                print(
                    f"[idx={idx}] All {max_reject_attempts} attempts rejected at turn {turn}, "
                    f"keeping best (ppl={best_ppl:.2f})",
                    flush=True,
                )
            else:
                print(f"[idx={idx}] Empty output at turn {turn}", flush=True)
                break

        await pool.update(idx, prompt[1], ppl_record, turn + 1)

        temp_ans = await extract_answer(prompt[1])
        if temp_ans != "invalid":
            answer = temp_ans
            print(f"[idx={idx}] Early stop at turn {turn}", flush=True)
            break

    if answer == "invalid":
        answer = await extract_answer(prompt[1])

    await pool.mark_finished(idx)

    result_data = {
        "problem_index": idx,
        "final_answer": answer,
        "duration_seconds": time.time() - start_time,
        "full_history": history_log,
        "question": problem,
        "ppl_history": ppl_record,
        "avg_ppl": sum(ppl_record) / len(ppl_record) if ppl_record else None,
    }

    output_filename = os.path.join(output_dir, f"problem_{idx:04d}.json")
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=4)

    return ()


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    parser = argparse.ArgumentParser(
        description="Single-model async TTS with rejection sampling + cross-sample forking."
    )
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--ppl_array_path", type=str, required=True)
    parser.add_argument("--model_port", type=int, default=40000)
    parser.add_argument("--max_tokens", type=int, default=500)
    parser.add_argument("--turns", type=int, default=15)
    parser.add_argument("--repeats", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--extract_mode", type=str, choices=["regex", "llm"], default="regex")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--ppl_threshold", type=float, default=0.6)
    parser.add_argument("--max_reject_attempts", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.1,
                        help="Progress reward weight in fork scoring (default: 0.1).")
    parser.add_argument("--fork_temperature", type=float, default=1.0,
                        help="Temperature for generation right after a fork (default: 1.0).")
    parser.add_argument("--fork_gap", type=float, default=0.05,
                        help="Minimum composite score gap (my_score - peer_score) required to fork (default: 0.05).")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    global client, model_name, http_semaphore, max_retries, extract_mode, tokenizer
    max_retries = args.max_retries
    extract_mode = args.extract_mode
    model_name = args.model_name

    http_semaphore = asyncio.Semaphore(64)
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

    print(f"Single-model async evaluation (rejection sampling + cross-sample forking)")
    print(f"  Model: {args.model_name}")
    print(f"  Dataset: {args.dataset_name}")
    print(f"  PPL threshold: {args.ppl_threshold}")
    print(f"  Max reject attempts: {args.max_reject_attempts}")
    print(f"  Alpha (progress reward): {args.alpha}")
    print(f"  Fork temperature: {args.fork_temperature}")
    print(f"  Fork gap (score): {args.fork_gap}")
    print(f"  {len(unique_problems_to_process)} groups to process")

    start_time = time.time()

    all_tasks = []
    pools = {}
    for unique_idx in unique_problems_to_process:
        pool = SamplePool()
        pools[unique_idx] = pool

        start_sample_idx = unique_idx * args.repeats
        end_sample_idx = start_sample_idx + args.repeats

        for sample_idx in range(start_sample_idx, end_sample_idx):
            if sample_idx not in processed_sample_indices:
                pool.register(sample_idx)
                problem = context[sample_idx]
                task = asyncio.create_task(
                    process_single_problem(
                        problem,
                        args.max_tokens,
                        ppl_array,
                        args.turns,
                        sample_idx,
                        args.model_port,
                        args.temperature,
                        args.output_dir,
                        args.ppl_threshold,
                        args.max_reject_attempts,
                        pool,
                        args.alpha,
                        args.fork_temperature,
                        args.fork_gap,
                    )
                )
                all_tasks.append(task)

    print(f"  Launched {len(all_tasks)} tasks across {len(unique_problems_to_process)} groups (all parallel)")
    await tqdm.gather(*all_tasks, desc="All samples")

    print(f"Time: {time.time() - start_time:.3f}s")

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
