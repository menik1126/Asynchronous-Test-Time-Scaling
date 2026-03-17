"""
Test: Does capturing CUDA graph with large seq_lens and replaying with
smaller seq_lens produce correct results?

If yes → "capture with large seq_lens" is a valid fix for the CG + compression bug.
If no → empty chunks cause issues in flashinfer's split-k reduce.
"""
import torch
import flashinfer

torch.manual_seed(42)
device = "cuda:0"

num_qo_heads = 32
num_kv_heads = 8
head_dim = 128
page_size = 1

max_total_pages = 500000
kv_cache = torch.randn(max_total_pages, 2, page_size, num_kv_heads, head_dim,
                        dtype=torch.float16, device=device)

workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)

test_cases = [
    # (name, capture_seq_lens, replay_seq_lens)
    ("shrink_small",    [1000] * 16,  [100] * 16),
    ("shrink_10x",      [5000] * 16,  [500] * 16),
    ("shrink_compress", [642] * 16,   [558] * 16),
    ("shrink_extreme",  [10000] * 16, [500] * 16),
    ("shrink_mixed",    [10000] * 16, [100, 500, 1000, 2000, 100, 500, 1000, 2000,
                                        100, 500, 1000, 2000, 100, 500, 1000, 2000]),
    ("grow_small",      [100] * 16,   [500] * 16),   # original bug case
    ("grow_large",      [100] * 16,   [5000] * 16),  # original bug case worse
    ("same_size",       [500] * 16,   [500] * 16),   # sanity check
]

for test_name, capture_lens, replay_lens in test_cases:
    print(f"\n{'='*60}")
    print(f"Test: {test_name}")
    print(f"  Capture seq_lens: {capture_lens[:3]}... (sum={sum(capture_lens)})")
    print(f"  Replay  seq_lens: {replay_lens[:3]}... (sum={sum(replay_lens)})")

    batch_size = len(capture_lens)

    # Pre-allocate CG buffers
    max_kv = max(sum(capture_lens), sum(replay_lens)) + 1000
    kv_indptr_buf = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    kv_indices_buf = torch.zeros(max_kv, dtype=torch.int32, device=device)
    kv_last_page_len_buf = torch.ones(batch_size, dtype=torch.int32, device=device)

    wrapper_cg = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace, "NHD", use_cuda_graph=True,
        paged_kv_indptr_buffer=kv_indptr_buf,
        paged_kv_indices_buffer=kv_indices_buf,
        paged_kv_last_page_len_buffer=kv_last_page_len_buf,
    )
    wrapper_nocg = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace, "NHD", use_cuda_graph=False)

    # Build CAPTURE indices (contiguous)
    cap_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    offset = 0
    for i, sl in enumerate(capture_lens):
        cap_indptr[i + 1] = cap_indptr[i] + sl
    cap_indices = torch.arange(sum(capture_lens), dtype=torch.int32, device=device)

    kv_indptr_buf[:batch_size + 1].copy_(cap_indptr)
    kv_indices_buf[:len(cap_indices)].copy_(cap_indices)

    wrapper_cg.plan(
        cap_indptr, cap_indices, kv_last_page_len_buf,
        num_qo_heads, num_kv_heads, head_dim, page_size,
        pos_encoding_mode="NONE", q_data_type=torch.float16,
    )

    q = torch.randn(batch_size, num_qo_heads, head_dim, dtype=torch.float16, device=device)

    # Capture
    s = torch.cuda.Stream(device=device)
    s.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(s):
        _ = wrapper_cg.run(q, kv_cache)
    torch.cuda.current_stream(device).wait_stream(s)

    g = torch.cuda.CUDAGraph()
    cg_q = q.clone()
    with torch.cuda.graph(g, stream=s):
        cg_out = wrapper_cg.run(cg_q, kv_cache)

    # Build REPLAY indices (non-contiguous, simulating compression)
    rep_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    all_rep_indices = []
    offset = 0
    for i, sl in enumerate(replay_lens):
        # Non-contiguous: select scattered pages
        full_range = torch.arange(offset, offset + sl + 100, device=device)
        step = max(1, (sl + 100) // sl)
        selected = full_range[::step][:sl]
        if len(selected) < sl:
            selected = full_range[:sl]
        all_rep_indices.append(selected.to(torch.int32))
        rep_indptr[i + 1] = rep_indptr[i] + sl
        offset += sl + 200
    rep_indices = torch.cat(all_rep_indices)

    # Update CG buffers and re-plan
    kv_indptr_buf[:batch_size + 1].copy_(rep_indptr)
    kv_indices_buf[:len(rep_indices)].copy_(rep_indices)

    wrapper_cg.plan(
        rep_indptr, rep_indices, kv_last_page_len_buf,
        num_qo_heads, num_kv_heads, head_dim, page_size,
        pos_encoding_mode="NONE", q_data_type=torch.float16,
    )

    # Replay
    cg_q.copy_(q)
    g.replay()
    out_cg = cg_out.clone()

    # Non-CG reference
    wrapper_nocg.plan(
        rep_indptr.clone(), rep_indices.clone(),
        torch.ones(batch_size, dtype=torch.int32, device=device),
        num_qo_heads, num_kv_heads, head_dim, page_size,
        pos_encoding_mode="NONE", q_data_type=torch.float16,
    )
    out_nocg = wrapper_nocg.run(q, kv_cache)

    max_diff = (out_cg - out_nocg).abs().max().item()
    mean_diff = (out_cg - out_nocg).abs().mean().item()
    status = "PASS" if max_diff < 0.01 else "FAIL"
    print(f"  Result: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f} [{status}]")

    del wrapper_cg, wrapper_nocg, g
    torch.cuda.empty_cache()
