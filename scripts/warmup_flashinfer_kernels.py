#!/usr/bin/env python3
"""
FlashInfer Kernel Warmup Script
================================
Pre-compile flashinfer kernels to avoid JIT compilation during first inference.

This script triggers compilation of commonly used kernels by performing
dummy operations. The compiled kernels will be cached in:
    ~/.cache/flashinfer/{arch}/cached_ops/

Usage:
    python3 scripts/warmup_flashinfer_kernels.py [--disable-sm90] [--model-path MODEL]

Options:
    --disable-sm90      Disable sm90 optimizations (use generic kernels)
    --model-path        Path to model to determine token config (default: deepseek-ai/DeepSeek-R1-Distill-Llama-8B)
    --dtype             Data type for kernels (default: bfloat16, options: bfloat16, float16)
"""

import argparse
import os
import sys
import time
import torch
from pathlib import Path

# Add sglang to path
project_dir = Path(__file__).parent.parent
sys.path.insert(0, str(project_dir / "third_party" / "sglang-0.4.3.post4" / "python"))

def check_flashinfer_cache():
    """Check current state of flashinfer cache."""
    cache_dir = Path.home() / ".cache" / "flashinfer"
    if not cache_dir.exists():
        print(f"⚠️  FlashInfer cache directory not found: {cache_dir}")
        return
    
    print(f"\n📂 FlashInfer cache directory: {cache_dir}")
    
    # Find all compiled kernels
    so_files = list(cache_dir.glob("**/*.so"))
    print(f"✅ Found {len(so_files)} compiled kernel(s)")
    
    # Find all incomplete compilations (has build.ninja but no .so)
    for arch_dir in cache_dir.iterdir():
        if not arch_dir.is_dir():
            continue
        cached_ops = arch_dir / "cached_ops"
        if not cached_ops.exists():
            continue
        
        for kernel_dir in cached_ops.iterdir():
            if not kernel_dir.is_dir():
                continue
            
            has_ninja = (kernel_dir / "build.ninja").exists()
            has_so = any(kernel_dir.glob("*.so"))
            
            if has_ninja and not has_so:
                print(f"⚠️  Incomplete compilation: {kernel_dir.name}")


def warmup_kernels(disable_sm90=False, dtype_str="bfloat16"):
    """Trigger compilation of common kernels."""
    print("\n" + "="*80)
    print("Starting FlashInfer Kernel Warmup")
    print("="*80)
    
    if disable_sm90:
        os.environ["FLASHINFER_DISABLE_SM90"] = "1"
        print("✓ SM90 optimizations DISABLED (using generic kernels)")
    else:
        print("✓ SM90 optimizations ENABLED")
    
    # Import after setting env vars
    try:
        import flashinfer
        print(f"✓ FlashInfer version: {flashinfer.__version__}")
    except ImportError as e:
        print(f"✗ Failed to import flashinfer: {e}")
        print("\nMake sure you're in the correct Python environment:")
        print("  source .sglang/bin/activate")
        sys.exit(1)
    
    # Determine dtype
    if dtype_str == "bfloat16":
        dtype = torch.bfloat16
    elif dtype_str == "float16":
        dtype = torch.float16
    else:
        raise ValueError(f"Unsupported dtype: {dtype_str}")
    
    device = "cuda:0"
    print(f"✓ Using device: {device}, dtype: {dtype}")
    
    # Common configurations for DeepSeek-R1-Distill-Llama-8B and similar models
    configs = [
        {
            "name": "Decode (single token)",
            "batch_size": 1,
            "qo_len": 1,  # decode: 1 token at a time
            "kv_len": 128,
            "num_qo_heads": 32,
            "num_kv_heads": 8,  # GQA
            "head_dim": 128,
        },
        {
            "name": "Prefill (short)",
            "batch_size": 1,
            "qo_len": 128,
            "kv_len": 128,
            "num_qo_heads": 32,
            "num_kv_heads": 8,
            "head_dim": 128,
        },
        {
            "name": "Prefill (medium)",
            "batch_size": 1,
            "qo_len": 512,
            "kv_len": 512,
            "num_qo_heads": 32,
            "num_kv_heads": 8,
            "head_dim": 128,
        },
    ]
    
    print(f"\n🔨 Compiling {len(configs)} kernel configuration(s)...")
    print("-" * 80)
    
    for i, config in enumerate(configs, 1):
        print(f"\n[{i}/{len(configs)}] {config['name']}")
        print(f"  Config: batch={config['batch_size']}, qo_len={config['qo_len']}, "
              f"kv_len={config['kv_len']}, heads={config['num_qo_heads']}/{config['num_kv_heads']}")
        
        try:
            start_time = time.time()
            
            # Trigger compilation
            if config["qo_len"] == 1:
                # Decode path: q must be 2D (num_qo_heads, head_dim)
                #              k/v must be 3D (kv_len, num_kv_heads, head_dim)
                q_decode = torch.randn(
                    config["num_qo_heads"], config["head_dim"],
                    dtype=dtype, device=device,
                )
                k_decode = torch.randn(
                    config["kv_len"], config["num_kv_heads"], config["head_dim"],
                    dtype=dtype, device=device,
                )
                v_decode = torch.randn(
                    config["kv_len"], config["num_kv_heads"], config["head_dim"],
                    dtype=dtype, device=device,
                )
                _ = flashinfer.single_decode_with_kv_cache(q_decode, k_decode, v_decode)
            else:
                # Prefill path: q must be 3D (qo_len, num_qo_heads, head_dim)
                #               k/v must be 3D (kv_len, num_kv_heads, head_dim)
                q_prefill = torch.randn(
                    config["qo_len"], config["num_qo_heads"], config["head_dim"],
                    dtype=dtype, device=device,
                )
                k_prefill = torch.randn(
                    config["kv_len"], config["num_kv_heads"], config["head_dim"],
                    dtype=dtype, device=device,
                )
                v_prefill = torch.randn(
                    config["kv_len"], config["num_kv_heads"], config["head_dim"],
                    dtype=dtype, device=device,
                )
                _ = flashinfer.single_prefill_with_kv_cache(q_prefill, k_prefill, v_prefill, causal=True)
            
            torch.cuda.synchronize()
            elapsed = time.time() - start_time
            
            print(f"  ✓ Compiled successfully in {elapsed:.2f}s")
            
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*80)
    print("Warmup Complete!")
    print("="*80)


def clear_incomplete_compilations():
    """Remove incomplete kernel compilations."""
    cache_dir = Path.home() / ".cache" / "flashinfer"
    if not cache_dir.exists():
        print("No cache directory to clean.")
        return
    
    removed = 0
    for arch_dir in cache_dir.iterdir():
        if not arch_dir.is_dir():
            continue
        cached_ops = arch_dir / "cached_ops"
        if not cached_ops.exists():
            continue
        
        for kernel_dir in cached_ops.iterdir():
            if not kernel_dir.is_dir():
                continue
            
            has_ninja = (kernel_dir / "build.ninja").exists()
            has_so = any(kernel_dir.glob("*.so"))
            
            if has_ninja and not has_so:
                print(f"🗑️  Removing incomplete: {kernel_dir.name}")
                import shutil
                shutil.rmtree(kernel_dir)
                # Also remove lock file
                lock_file = kernel_dir.parent / f"{kernel_dir.name}.lock"
                if lock_file.exists():
                    lock_file.unlink()
                removed += 1
    
    print(f"\n✓ Removed {removed} incomplete compilation(s)")


def main():
    parser = argparse.ArgumentParser(
        description="Pre-compile FlashInfer kernels to avoid JIT delays",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--disable-sm90",
        action="store_true",
        help="Disable sm90 optimizations (use generic kernels)",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float16"],
        help="Data type for kernels (default: bfloat16)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check current cache status without compiling",
    )
    parser.add_argument(
        "--clear-incomplete",
        action="store_true",
        help="Remove incomplete compilations from cache",
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("FlashInfer Kernel Warmup Tool")
    print("="*80)
    
    if args.check:
        check_flashinfer_cache()
        return
    
    if args.clear_incomplete:
        print("\n🧹 Clearing incomplete compilations...")
        clear_incomplete_compilations()
        print()
    
    check_flashinfer_cache()
    
    # Run warmup
    warmup_kernels(
        disable_sm90=args.disable_sm90,
        dtype_str=args.dtype,
    )
    
    # Check again
    check_flashinfer_cache()


if __name__ == "__main__":
    main()
