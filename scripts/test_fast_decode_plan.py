"""
Test: Reproduce the exact sglang CG decode path with fast_decode_plan.
Simulates the sequence: capture with dummy data -> replay with compressed data.
"""
import torch
import flashinfer
import sys
sys.path.insert(0, "/home/hku/Asynchronous-Test-Time-Scaling/third_party/sglang-kv-compress-dev")

torch.manual_seed(42)
device = "cuda:0"

num_qo_heads = 32
num_kv_heads = 8
head_dim = 128
page_size = 1

max_total_pages = 20000
kv_cache = torch.randn(max_total_pages, 2, page_size, num_kv_heads, head_dim,
                        dtype=torch.float16, device=device)

batch_size = 16  # sglang uses padded bs

# Pre-allocate CG buffers (like sglang does)
kv_indptr_buf = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
max_kv_indices = 200000
kv_indices_buf = torch.zeros(max_kv_indices, dtype=torch.int32, device=device)
kv_last_page_len_buf = torch.ones(batch_size, dtype=torch.int32, device=device)

workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)

# CG wrapper
wrapper_cg = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
    workspace, "NHD", use_cuda_graph=True,
    paged_kv_indptr_buffer=kv_indptr_buf,
    paged_kv_indices_buffer=kv_indices_buf,
    paged_kv_last_page_len_buffer=kv_last_page_len_buf,
)

# Non-CG wrapper
wrapper_nocg = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace, "NHD", use_cuda_graph=False)

# === CAPTURE PHASE: use contiguous indices (like normal decode before compression) ===
capture_seq_lens = [100] * batch_size
capture_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
for i, l in enumerate(capture_seq_lens):
    capture_indptr[i + 1] = capture_indptr[i] + l
capture_indices = torch.arange(sum(capture_seq_lens), dtype=torch.int32, device=device)

kv_indptr_buf.copy_(capture_indptr)
kv_indices_buf[:len(capture_indices)].copy_(capture_indices)

wrapper_cg.plan(
    capture_indptr, capture_indices, kv_last_page_len_buf,
    num_qo_heads, num_kv_heads, head_dim, page_size,
    pos_encoding_mode="NONE", q_data_type=torch.float16,
)

q = torch.randn(batch_size, num_qo_heads, head_dim, dtype=torch.float16, device=device)

# Capture graph
s = torch.cuda.Stream(device=device)
s.wait_stream(torch.cuda.current_stream(device))
with torch.cuda.stream(s):
    _ = wrapper_cg.run(q, kv_cache)
torch.cuda.current_stream(device).wait_stream(s)

g = torch.cuda.CUDAGraph()
cg_q = q.clone()
with torch.cuda.graph(g, stream=s):
    cg_out = wrapper_cg.run(cg_q, kv_cache)

print("Graph captured successfully")

# === REPLAY PHASE: simulate compressed (non-contiguous) indices ===
# Simulating: 16 requests, each originally had 600 tokens, compressed to 500
# Some with different lengths (like in real ATTS)
test_configs = [
    ("uniform_compress", [500] * batch_size),
    ("mixed_compress",   [500, 500, 500, 500, 300, 300, 300, 300, 800, 800, 800, 800, 200, 200, 200, 200]),
    ("one_very_long",    [500]*15 + [5000]),
    ("all_very_long",    [3000] * batch_size),
]

for test_name, seq_lens in test_configs:
    print(f"\n=== {test_name}: seq_lens={seq_lens[:4]}... ===")
    
    # Build non-contiguous indices (simulating SnapKV selection)
    all_indices = []
    offset = 0
    for sl in seq_lens:
        full = torch.arange(offset, offset + sl + 100, device=device)  # original had sl+100 tokens
        # Select non-contiguous: keep first 128, random middle, last 64
        protect = min(128, sl)
        recent = min(64, sl - protect)
        budget = sl - protect - recent
        if budget > 0:
            middle = torch.arange(protect, sl + 100 - recent, device=device)
            perm = torch.randperm(len(middle), device=device)[:budget]
            selected = torch.cat([
                torch.arange(protect, device=device),
                middle[perm].sort().values,
                torch.arange(sl + 100 - recent, sl + 100, device=device),
            ])
        else:
            selected = torch.arange(sl, device=device)
        
        page_idx = (full[selected[:sl]] + offset).to(torch.int32)
        all_indices.append(page_idx)
        offset += sl + 200
    
    indices = torch.cat(all_indices)
    indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    for i, sl in enumerate(seq_lens):
        indptr[i + 1] = indptr[i] + sl
    
    # === Method 1: sglang's fast_decode_plan path ===
    # Step 1: Write to CG buffers (like create_flashinfer_kv_indices_triton does)
    kv_indptr_buf[:batch_size + 1].copy_(indptr)
    kv_indices_buf[:len(indices)].copy_(indices)
    # Note: rest of kv_indices_buf still has OLD data from capture phase!
    
    # Step 2: Call plan (like fast_decode_plan, but using standard plan since we can't import fast_decode_plan easily)
    wrapper_cg.plan(
        kv_indptr_buf[:batch_size + 1], 
        indices,  # Note: flashinfer's plan copies this to the buffer anyway in CG mode
        kv_last_page_len_buf,
        num_qo_heads, num_kv_heads, head_dim, page_size,
        pos_encoding_mode="NONE", q_data_type=torch.float16,
    )
    
    # Step 3: Replay
    cg_q.copy_(q)
    g.replay()
    out_cg = cg_out.clone()
    
    # === Method 2: Non-CG reference ===
    wrapper_nocg.plan(
        indptr.clone(), indices.clone(), torch.ones(batch_size, dtype=torch.int32, device=device),
        num_qo_heads, num_kv_heads, head_dim, page_size,
        pos_encoding_mode="NONE", q_data_type=torch.float16,
    )
    out_nocg = wrapper_nocg.run(q, kv_cache)
    
    # Compare
    max_diff = (out_cg - out_nocg).abs().max().item()
    mean_diff = (out_cg - out_nocg).abs().mean().item()
    
    # Per-request
    worst_req = -1
    worst_diff = 0
    for i in range(batch_size):
        req_diff = (out_cg[i] - out_nocg[i]).abs().max().item()
        if req_diff > worst_diff:
            worst_diff = req_diff
            worst_req = i
    
    status = "PASS" if max_diff < 0.01 else "FAIL"
    print(f"  max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}, worst_req={worst_req}(diff={worst_diff:.6f}) [{status}]")
