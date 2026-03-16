"""Quick test: run fork version on a single problem group to observe fork behavior."""
import asyncio
import sys
import os

sys.path.insert(0, "/home/hku/Asynchronous-Test-Time-Scaling")
os.chdir("/home/hku/Asynchronous-Test-Time-Scaling")
os.environ["OPENAI_API_KEY"] = "sk-f7Oh115pfz6REQeFKesLFOhrk85Yd8ySvnqmRDZ08oDT8nyr"
os.environ["OPENAI_BASE_URL"] = "https://chatapi.littlewheat.com/v1"
os.environ["OPENAI_MODEL"] = "gpt-4o"

async def main():
    import numpy as np
    import httpx
    from transformers import AutoTokenizer
    from tqdm.asyncio import tqdm

    import ATTS.ref_async_self_reject_fork as mod
    from ATTS.dataset import load_my_dataset

    model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    port = 40002
    repeats = 16
    max_tokens = 500
    turns = 15
    temperature = 0.8
    concurrency = 16
    ppl_threshold = 0.6
    max_reject_attempts = 3
    alpha = 0.01
    fork_temperature = 1.0
    fork_gap = 0.02
    output_dir = "results/fork_test_single"

    mod.model_name = model_name
    mod.max_retries = 3
    mod.extract_mode = "regex"
    mod.model_semaphore = asyncio.Semaphore(concurrency)
    mod.client = httpx.AsyncClient(
        timeout=24000.0,
        limits=httpx.Limits(max_connections=1000, max_keepalive_connections=1000),
    )
    mod.tokenizer = AutoTokenizer.from_pretrained(model_name)

    context, answer = load_my_dataset("math500", repeats)
    ppl_array = np.load("evaluation/ppls_self_prefill_math500_deepseek-ai_DeepSeek-R1-Distill-Llama-8B_s16_t500_temp0.8.npy")

    pool = mod.SamplePool()
    tasks = []
    for sample_idx in range(0, repeats):
        pool.register(sample_idx)
        task = asyncio.create_task(
            mod.process_single_problem(
                context[sample_idx],
                max_tokens, ppl_array, turns, sample_idx, port,
                temperature, output_dir, ppl_threshold,
                max_reject_attempts, pool, alpha, fork_temperature, fork_gap,
            )
        )
        tasks.append(task)

    await tqdm.gather(*tasks, desc="Test group 0")
    await mod.client.aclose()

    print("\n=== Results ===")
    import json
    for i in range(repeats):
        fp = f"{output_dir}/problem_{i:04d}.json"
        with open(fp) as f:
            d = json.load(f)
        forks = [h for h in d["full_history"] if h.get("event") == "fork"]
        ans_found = [h for h in d["full_history"] if h.get("accepted_reason") == "answer_found"]
        max_turn = max((h["turn"] for h in d["full_history"]), default=0)
        print(f"  idx={i:2d}: answer={d['final_answer']!r:>10s}, turns={max_turn}, "
              f"forks={len(forks)}, ans_shortcut={len(ans_found)}, "
              f"avg_ppl={d.get('avg_ppl','N/A')}")
        for fk in forks:
            print(f"          fork@turn{fk['turn']}: from idx={fk['forked_from']}, "
                  f"my_ppl={fk['my_avg_ppl']:.3f}, peer_ppl={fk['peer_avg_ppl']:.3f}, "
                  f"score_gap={fk['score_gap']:.3f}")

if __name__ == "__main__":
    asyncio.run(main())
