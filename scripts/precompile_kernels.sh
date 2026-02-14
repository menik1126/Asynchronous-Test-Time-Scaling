#!/bin/bash
# Pre-compile FlashInfer kernels to avoid JIT hang on first startup.
# Usage: ./scripts/precompile_kernels.sh [--no-sm90]
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Fix libcudart version mismatch (pip 12.4 vs system 12.9)
SYS="/usr/local/cuda/lib64/libcudart.so.12"
VENV="$(python3 -c 'import nvidia.cuda_runtime, pathlib; print(pathlib.Path(nvidia.cuda_runtime.__file__).parent / "lib" / "libcudart.so.12")')"
[ -f "$SYS" ] && [ -f "$VENV" ] && ! cmp -s "$SYS" "$VENV" && \
    cp "$VENV" "${VENV}.bak" 2>/dev/null; cp "$SYS" "$VENV" && echo "Updated venv libcudart."

ARGS=""; [ "$1" = "--no-sm90" ] && ARGS="--disable-sm90"
python3 "${SCRIPT_DIR}/warmup_flashinfer_kernels.py" --clear-incomplete $ARGS
python3 "${SCRIPT_DIR}/warmup_flashinfer_kernels.py" $ARGS
