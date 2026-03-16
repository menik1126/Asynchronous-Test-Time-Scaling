"""Quick test: run 1 problem with 4 samples to verify fork mechanism."""
import asyncio
import json
import sys
import os
import tempfile

sys.path.insert(0, os.path.dirname(__file__))

async def main():
    from ATTS.ref_async_self_reject_fork import (
        SamplePool, SampleState, process_single_problem,
        call_model_generate, call_model_ppl, extract_answer,
    )
    import ATTS.ref_async_self_reject_fork as mod
    import numpy as np
    import httpx
    from transformers import AutoTokenizer

    model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    port = 40002
    ppl_array_path = "evaluation/ppls_self_prefill_math500_deepseek-ai_DeepSeek-R1-Distill-Llama-8B_s16_t500_temp0.8.npy"

    mod.model_name = model_name
    mod.model_semaphore = asyncio.Semaphore(4)
    mod.client = httpx.AsyncClient(
        timeout=24000.0,
        limits=httpx.Limits(max_connections=100, max_keepalive_connections=100),
    )
    mod.tokenizer = AutoTokenizer.from_pretrained(model_name)
    mod.extract_mode = "regex"
    mod.max_retries = 3

    ppl_array = np.load(ppl_array_path)

    # Use a hard math problem as test
    problem = (
        "Let $P(x)$ be a monic polynomial of degree 3. Suppose that $P(x)$ has "
        "remainder $R(x)$ when it is divided by $(x - 1)(x - 4),$ and remainder "
        "$2R(x)$ when it is divided by $(x - 2)(x - 3).$ Given that $P(0) = 5,$ "
        "find $P(5).$"
    )

    output_dir = tempfile.mkdtemp(prefix="fork_test_")
    print(f"Output dir: {output_dir}")
    print(f"PPL array shape: {ppl_array.shape}, mean: {ppl_array.mean():.2f}")
    print()

    n_samples = 4
    turns = 8
    ppl_threshold = 0.6
    max_reject_attempts = 2
    alpha = 0.1
    fork_temperature = 1.0
    fork_min_gap = 0.1

    pool = SamplePool()
    for i in range(n_samples):
        pool.register(i)

    tasks = []
    for i in range(n_samples):
        task = asyncio.create_task(
            process_single_problem(
                problem=problem,
                max_tokens=500,
                ppl_array=ppl_array,
                turns=turns,
                idx=i,
                port=port,
                temperature=0.8,
                output_dir=output_dir,
                ppl_threshold=ppl_threshold,
                max_reject_attempts=max_reject_attempts,
                pool=pool,
                alpha=alpha,
                fork_temperature=fork_temperature,
                fork_min_gap=fork_min_gap,
            )
        )
        tasks.append(task)

    await asyncio.gather(*tasks)

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    for i in range(n_samples):
        filepath = os.path.join(output_dir, f"problem_{i:04d}.json")
        with open(filepath) as f:
            data = json.load(f)

        history = data["full_history"]
        forks = [h for h in history if h.get("event") == "fork"]
        accepted = [h for h in history if h.get("accepted") is True]
        rejected = [h for h in history if h.get("accepted") is False]

        print(f"\n--- Sample {i} ---")
        print(f"  Final answer: {data['final_answer']}")
        print(f"  Duration: {data['duration_seconds']:.1f}s")
        print(f"  Avg PPL: {data['avg_ppl']:.3f}" if data['avg_ppl'] else "  Avg PPL: N/A")
        print(f"  PPL history: {[f'{p:.2f}' for p in data['ppl_history']]}")
        print(f"  Accepted turns: {len(accepted)}, Rejected: {len(rejected)}, Forks: {len(forks)}")

        for fork in forks:
            print(f"    FORK at turn {fork['turn']}: "
                  f"from idx={fork['forked_from']}, "
                  f"peer_score={fork['peer_score']:.3f}, "
                  f"my_score={fork['my_score']:.3f}")

    await mod.client.aclose()
    print(f"\nFull results saved to: {output_dir}")


if __name__ == "__main__":
    asyncio.run(main())
