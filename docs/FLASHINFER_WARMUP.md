# FlashInfer Kernel Pre-compilation Guide

## Background

### What is sm90?

**sm90** refers to NVIDIA's Hopper architecture (H100/H200 GPUs), Streaming Multiprocessor version 9.0.

FlashInfer compiles optimized CUDA kernels for different GPU architectures:
- **sm90 kernels**: Optimized for H100/H200, better performance but more complex compilation
- **Generic kernels**: Compatible across architectures, stable but slightly slower

### Why pre-compile?

FlashInfer uses **JIT (Just-In-Time) compilation**:
- The first time a kernel config is used, it compiles CUDA code on the fly
- sm90 kernels are slow to compile (minutes or longer)
- If compilation fails mid-way (resource contention, library mismatch, etc.), the server **hangs**

**Symptom:**
```
flashinfer.jit: Loading JIT ops: batch_prefill_..._sm90
(no output for ~10 minutes, then timeout)
```

### Root cause: libcudart version mismatch

This hang is caused by a mismatch between the system CUDA toolkit and pip-installed CUDA runtime:

| Component | Version | Has `cudaGetDriverEntryPointByVersion` |
|-----------|---------|---------------------------------------|
| System nvcc (compiler) | 12.9 | - |
| System libcudart.so.12 | 12.9.79 | Yes |
| Pip nvidia-cuda-runtime-cu12 | 12.4.127 | **No** |

When `nvcc 12.9` compiles sm90 kernels, the output `.so` references symbols only available in CUDA 12.9+. But at runtime, Python loads the pip-installed `libcudart 12.4`, which lacks those symbols, causing `undefined symbol` errors or hangs.

---

## Solutions

### Option 1: Pre-compile kernels (recommended)

Run once after installation, before starting servers:

```bash
source .sglang/bin/activate
bash scripts/precompile_kernels.sh
```

This script automatically:
1. Patches `libcudart.so.12` if the pip version doesn't match the system version
2. Cleans up any incomplete compilations
3. Pre-compiles all commonly used kernels (decode + prefill, including sm90)

**Verify:**
```bash
python3 scripts/warmup_flashinfer_kernels.py --check
```

Expected: `Found N compiled kernel(s)` with no incomplete compilations.

### Option 2: Disable sm90 (fallback)

If sm90 compilation keeps failing:

```bash
bash scripts/precompile_kernels.sh --no-sm90
```

Then launch servers with sm90 disabled:
```bash
export FLASHINFER_DISABLE_SM90=1
bash scripts/launch_sglang_servers.sh
```

**Performance impact**: Generic kernels are ~5-15% slower than sm90 on H100/H200.

---

## Cache Location & Persistence

Compiled kernels are cached at:
```
~/.cache/flashinfer/{arch}/cached_ops/
```

- `{arch}`: Auto-detected from GPU (e.g. `90` for Hopper, `80` for Ampere)
- Each kernel config gets a subdirectory containing the compiled `.so`

### Docker persistence

Mount the cache as a volume to avoid recompilation across container restarts:

```yaml
# docker-compose.yml
services:
  sglang:
    volumes:
      - flashinfer-cache:/root/.cache/flashinfer

volumes:
  flashinfer-cache:
```

Or with `docker run`:
```bash
docker run -v ./cache/flashinfer:/root/.cache/flashinfer ...
```

---

## Troubleshooting

### Compilation hangs

**Symptom**: `Loading JIT ops: ..._sm90` with no `Finished loading` for minutes.

**Fix**:
```bash
# Clean incomplete compilations and retry
python3 scripts/warmup_flashinfer_kernels.py --clear-incomplete
python3 scripts/warmup_flashinfer_kernels.py
```

### undefined symbol: cudaGetDriverEntryPointByVersion

**Cause**: Pip-installed `libcudart.so.12` is older than the system nvcc version.

**Fix**: `precompile_kernels.sh` handles this automatically. To fix manually:
```bash
cp /usr/local/cuda/lib64/libcudart.so.12 \
   .sglang/lib/python3.11/site-packages/nvidia/cuda_runtime/lib/libcudart.so.12
```

### Check compiled kernels

```bash
find ~/.cache/flashinfer/ -name "*.so" -exec ls -lh {} \;
```

### Check JIT log

```bash
tail -50 ~/.cache/flashinfer/90/flashinfer_jit.log
```

Look for pairs of `Loading JIT ops: xxx` / `Finished loading JIT ops: xxx`. Missing `Finished` means compilation failed or hung.

### Full reset

```bash
rm -rf ~/.cache/flashinfer/
python3 scripts/warmup_flashinfer_kernels.py
```

---

## Performance Comparison

| Setup | First startup | Subsequent startups | Inference perf |
|-------|--------------|--------------------:|----------------|
| No pre-compilation + sm90 | 10+ min (may hang) | Normal | Best |
| **Pre-compiled + sm90** | **~30s** | **Normal** | **Best** |
| sm90 disabled | ~30s | Normal | -5~15% |

**Recommendation**: Use pre-compiled + sm90 for production.
