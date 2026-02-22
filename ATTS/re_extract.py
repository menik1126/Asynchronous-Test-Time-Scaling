"""
Re-extract final answers from existing problem_XXXX.json files using LLM,
then compute accuracy against ground-truth.

Usage:
    python -m ATTS.re_extract --input_dir <results_dir> --dataset_name math500 --repeats 16 [--dry_run]

Environment variables (same as async_agent.py):
    OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL
"""

import argparse
import asyncio
import json
import os
import re
from glob import glob

from openai import AsyncOpenAI, APITimeoutError
from tqdm import tqdm as sync_tqdm

from ATTS.dataset import load_my_dataset
from ATTS.async_agent import anyone_check


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


def extract_answer_regex(history):
    temp = "\n\n".join(history)
    matches = re.findall(r"\\boxed\{(.*?)\}", temp)
    if matches:
        return matches[-1]
    pattern = re.compile(r"ANSWER:\s+([A-Z])", re.IGNORECASE)
    matches = pattern.findall(temp)
    if matches:
        return matches[-1]
    return "invalid"


async def extract_answer_llm(history, client, model):
    temp = "\n\n".join(history)
    prompt_text = EXTRACT_ANSWER_LLM_PROMPT.format(history=temp)

    retries = 3
    delay = 1
    for attempt in range(retries):
        try:
            response = await client.chat.completions.create(
                model=model,
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
            if attempt < retries - 1:
                wait = delay * (2 ** attempt)
                print(f"  [Timeout] retrying in {wait}s...", flush=True)
                await asyncio.sleep(wait)
            else:
                print("  [Timeout] all retries exhausted.", flush=True)
                return "invalid"
        except Exception as e:
            print(f"  [Error] {e}", flush=True)
            return "invalid"
    return "invalid"


def rebuild_history(full_history):
    """Reconstruct the output history list from full_history entries."""
    return [
        entry["output"]
        for entry in full_history
        if "output" in entry and entry["output"]
    ]


async def process_one(filepath, client, model, dry_run):
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    history = rebuild_history(data["full_history"])
    if not history:
        print(f"  {os.path.basename(filepath)}: no history, skipped")
        return None

    old_answer = data.get("final_answer", "invalid")
    regex_answer = extract_answer_regex(history)
    llm_answer = await extract_answer_llm(history, client, model)

    changed = old_answer != llm_answer
    tag = "CHANGED" if changed else "same"
    print(
        f"  {os.path.basename(filepath)}: "
        f"old={old_answer!r}  regex={regex_answer!r}  llm={llm_answer!r}  [{tag}]"
    )

    if not dry_run:
        data["final_answer"] = llm_answer
        data["final_answer_regex"] = regex_answer
        data["final_answer_llm"] = llm_answer
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    return {"file": os.path.basename(filepath), "old": old_answer, "regex": regex_answer, "llm": llm_answer, "changed": changed}


async def compute_score(generated_answers, gt_answers, repeats):
    group = len(generated_answers) // repeats
    right = 0
    pbar = sync_tqdm(range(group), desc="Evaluating", unit="question")
    for i in pbar:
        start = i * repeats
        end = (i + 1) * repeats
        outputs = generated_answers[start:end]
        correct_answer = gt_answers[start]
        print(f"Generated answers: {outputs}")
        print(f"Correct answer: {correct_answer}")
        ans = await anyone_check(correct_answer, outputs)
        print(ans)
        if ans == "Match":
            right += 1
        pbar.set_postfix(acc=f"{right/(i+1):.2%}")

    print(f"\nAccuracy: {right / group:.2%}")


async def main():
    parser = argparse.ArgumentParser(description="Re-extract answers from result JSONs using LLM, then compute accuracy.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing problem_XXXX.json files.")
    parser.add_argument("--dataset_name", type=str, required=True, help="Dataset name (e.g., math500, aime25, gpqa).")
    parser.add_argument("--repeats", type=int, default=16, help="Number of repeats per problem.")
    parser.add_argument("--dry_run", action="store_true", help="Only print, do not overwrite files.")
    parser.add_argument("--concurrency", type=int, default=8, help="Max concurrent LLM requests.")
    args = parser.parse_args()

    openai_key = os.environ.get("OPENAI_API_KEY", "")
    openai_base = os.environ.get("OPENAI_BASE_URL") or None
    openai_model = os.environ.get("OPENAI_MODEL", "gpt-5.2")

    client = AsyncOpenAI(api_key=openai_key, base_url=openai_base)
    semaphore = asyncio.Semaphore(args.concurrency)

    files = sorted(glob(os.path.join(args.input_dir, "problem_*.json")))
    if not files:
        print(f"No problem_*.json files found in {args.input_dir}")
        return

    print(f"Found {len(files)} files in {args.input_dir}")
    if args.dry_run:
        print("[DRY RUN] Files will NOT be modified.\n")

    # --- Step 1: Re-extract answers ---
    print("=== Step 1: Re-extracting answers ===")

    async def bounded(fp):
        async with semaphore:
            return await process_one(fp, client, openai_model, args.dry_run)

    results = await asyncio.gather(*[bounded(fp) for fp in files])
    results = [r for r in results if r is not None]

    changed_count = sum(1 for r in results if r["changed"])
    print(f"\nDone. {len(results)} files processed, {changed_count} answers changed.")

    # --- Step 2: Compute accuracy ---
    print("\n=== Step 2: Computing accuracy ===")
    _, gt_answers = load_my_dataset(args.dataset_name, args.repeats)

    llm_answers = []
    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)
        llm_answers.append(data.get("final_answer_llm", data.get("final_answer", "invalid")))

    if len(llm_answers) != len(gt_answers):
        print(f"Warning: file count ({len(llm_answers)}) != dataset size ({len(gt_answers)}). Truncating to min.")
        n = min(len(llm_answers), len(gt_answers))
        llm_answers = llm_answers[:n]
        gt_answers = gt_answers[:n]

    await compute_score(llm_answers, gt_answers, args.repeats)


if __name__ == "__main__":
    asyncio.run(main())
