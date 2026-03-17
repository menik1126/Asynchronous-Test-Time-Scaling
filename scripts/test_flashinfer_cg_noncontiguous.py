"""
Test: Does flashinfer CUDA graph decode produce correct results with
non-contiguous KV page indices (simulating KV cache compression)?

Compares CG wrapper vs non-CG wrapper output for identical inputs.
If they differ, the bug is in flashinfer's CG decode kernel.
"""
import torch
import flashinfer
import numpy as np

torch.manual_seed(42)
device = "cuda:0"

# Model-like params (DeepSeek-R1-Distill-Llama-8B)
num_qo_heads = 32
num_kv_heads = 8
head_dim = 128
page_size = 1  # sglang uses page_size=1

# Test scenarios
scenarios = [
    {"name": "contiguous_short",  "seq_len": 100,  "compress_to": None},
    {"name": "contiguous_long",   "seq_len": 2000, "compress_to": None},
    {"name": "compress_short",    "seq_len": 600,  "compress_to": 500},
    {"name": "compress_medium",   "seq_len": 2000, "compress_to": 1000},
    {"name": "compress_long",     "seq_len": 6000, "compress_to": 3000},
    {"name": "compress_extreme",  "seq_len": 10000,"compress_to": 4000},
    # Multi-request batch
    {"name": "batch_compress",    "batch": [
        {"seq_len": 600,  "compress_to": 500},
        {"seq_len": 2000, "compress_to": 1000},
        {"seq_len": 600,  "compress_to": 500},
    ]},
    {"name": "batch_mixed",       "batch": [
        {"seq_len": 100,  "compress_to": None},   # no compression
        {"seq_len": 2000, "compress_to": 1000},    # compressed
        {"seq_len": 100,  "compress_to": None},    # no compression
        {"seq_len": 6000, "compress_to": 3000},    # heavily compressed
    ]},
]


def make_kv_cache(max_pages):
    """Create a paged KV cache [max_pages, 2, page_size, num_kv_heads, head_dim]."""
    return torch.randn(
        max_pages, 2, page_size, num_kv_heads, head_dim,
        dtype=torch.float16, device=device
    )


def build_indices(seq_len, compress_to, max_pages_start):
    """Build page indices. If compress_to is set, select non-contiguous pages."""
    if compress_to is None:
        indices = torch.arange(max_pages_start, max_pages_start + seq_len, device=device, dtype=torch.int32)
        return indices, seq_len
    else:
        full_indices = torch.arange(max_pages_start, max_pages_start + seq_len, device=device)
        # Simulate SnapKV selection: keep first 128 (protect), random middle, last 64 (recent)
        protect = 128
        recent = 64
        budget = compress_to - protect - recent
        middle_pool = torch.arange(protect, seq_len - recent, device=device)
        perm = torch.randperm(len(middle_pool), device=device)[:budget]
        selected_middle = middle_pool[perm].sort().values
        selected = torch.cat([
            torch.arange(protect, device=device),
            selected_middle,
            torch.arange(seq_len - recent, seq_len, device=device),
        ])
        indices = (full_indices[selected] + max_pages_start).to(torch.int32)
        return indices, len(indices)


def run_scenario(scenario):
    print(f"\n{'='*60}")
    print(f"Scenario: {scenario['name']}")

    max_total_pages = 20000
    kv_cache = make_kv_cache(max_total_pages)

    if "batch" in scenario:
        batch_configs = scenario["batch"]
    else:
        batch_configs = [{"seq_len": scenario["seq_len"], "compress_to": scenario.get("compress_to")}]

    batch_size = len(batch_configs)

    # Build per-request indices
    all_indices = []
    kv_lens = []
    offset = 0
    for cfg in batch_configs:
        idx, kv_len = build_indices(cfg["seq_len"], cfg["compress_to"], offset)
        all_indices.append(idx)
        kv_lens.append(kv_len)
        offset += cfg["seq_len"] + 100  # gap between requests

    # Build indptr and concatenated indices
    indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    for i, l in enumerate(kv_lens):
        indptr[i + 1] = indptr[i] + l
    indices = torch.cat(all_indices)
    last_page_len = torch.ones(batch_size, dtype=torch.int32, device=device)

    # Query: random
    q = torch.randn(batch_size, num_qo_heads, head_dim, dtype=torch.float16, device=device)

    print(f"  batch_size={batch_size}, kv_lens={kv_lens}, total_kv={indices.shape[0]}")
    print(f"  indptr={indptr.tolist()}")
    print(f"  indices non-contiguous: {not torch.equal(indices, torch.arange(indices[0].item(), indices[0].item()+len(indices), device=device, dtype=torch.int32))}")

    # --- Non-CG wrapper ---
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    wrapper_nocg = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace, "NHD", use_cuda_graph=False)
    wrapper_nocg.plan(
        indptr.clone(), indices.clone(), last_page_len.clone(),
        num_qo_heads, num_kv_heads, head_dim, page_size,
        pos_encoding_mode="NONE", q_data_type=torch.float16,
    )
    out_nocg = wrapper_nocg.run(q, kv_cache)

    # --- CG wrapper ---
    max_kv_indices = max_total_pages
    wrapper_cg = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace, "NHD", use_cuda_graph=True,
        paged_kv_indptr_buffer=torch.zeros(batch_size + 1, dtype=torch.int32, device=device),
        paged_kv_indices_buffer=torch.zeros(max_kv_indices, dtype=torch.int32, device=device),
        paged_kv_last_page_len_buffer=torch.ones(batch_size, dtype=torch.int32, device=device),
    )

    # First plan (capture phase) - use dummy contiguous indices
    dummy_indptr = torch.arange(batch_size + 1, dtype=torch.int32, device=device) * 10
    dummy_indices = torch.arange(batch_size * 10, dtype=torch.int32, device=device)
    dummy_last = torch.ones(batch_size, dtype=torch.int32, device=device)
    wrapper_cg.plan(
        dummy_indptr, dummy_indices, dummy_last,
        num_qo_heads, num_kv_heads, head_dim, page_size,
        pos_encoding_mode="NONE", q_data_type=torch.float16,
    )

    # Capture CUDA graph
    s = torch.cuda.Stream(device=device)
    s.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(s):
        # Warm up
        _ = wrapper_cg.run(q, kv_cache)
    torch.cuda.current_stream(device).wait_stream(s)

    g = torch.cuda.CUDAGraph()
    cg_q = q.clone()
    cg_out = torch.empty_like(out_nocg)

    with torch.cuda.graph(g, stream=s):
        cg_out_ref = wrapper_cg.run(cg_q, kv_cache)

    # Now replay with actual (possibly non-contiguous) indices
    wrapper_cg._paged_kv_indptr_buf.copy_(indptr)
    wrapper_cg._paged_kv_indices_buf[:len(indices)].copy_(indices)
    wrapper_cg._paged_kv_last_page_len_buf.copy_(last_page_len)

    # Re-plan with correct indptr (this is what sglang does via fast_decode_plan)
    wrapper_cg.plan(
        indptr.clone(), indices.clone(), last_page_len.clone(),
        num_qo_heads, num_kv_heads, head_dim, page_size,
        pos_encoding_mode="NONE", q_data_type=torch.float16,
    )

    cg_q.copy_(q)
    g.replay()
    out_cg = cg_out_ref.clone()

    # Compare
    max_diff = (out_nocg - out_cg).abs().max().item()
    mean_diff = (out_nocg - out_cg).abs().mean().item()
    rel_diff = ((out_nocg - out_cg).abs() / (out_nocg.abs() + 1e-8)).mean().item()

    # Check per-request
    for i in range(batch_size):
        req_diff = (out_nocg[i] - out_cg[i]).abs().max().item()
        req_mean = (out_nocg[i] - out_cg[i]).abs().mean().item()
        status = "OK" if req_diff < 0.01 else "MISMATCH"
        print(f"  req {i}: max_diff={req_diff:.6f}, mean_diff={req_mean:.6f} [{status}]")

    status = "PASS" if max_diff < 0.01 else "FAIL"
    print(f"  Overall: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}, rel_diff={rel_diff:.6f} [{status}]")

    return max_diff < 0.01


if __name__ == "__main__":
    print("Testing flashinfer CUDA graph with non-contiguous KV indices")
    print(f"flashinfer version: {flashinfer.__version__}")
    print(f"Device: {device}")

    passed = 0
    failed = 0
    for scenario in scenarios:
        try:
            ok = run_scenario(scenario)
            if ok:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed out of {len(scenarios)}")
    if failed > 0:
        print("CONCLUSION: flashinfer CG decode has issues with non-contiguous KV indices")
    else:
        print("CONCLUSION: flashinfer CG decode works correctly (issue may be in sglang integration)")
