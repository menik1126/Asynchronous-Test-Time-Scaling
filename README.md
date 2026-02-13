<h1 style="font-size: 9em; white-space: nowrap;">[ICLR2026🔥] ATTS: Asynchronous Test-Time Scaling</h1>

Official implementation of **ATTS: Asynchronous Test-Time Scaling via Conformal Prediction**.

> ATTS achieves up to **56.7x speedup** and **4.14x throughput** improvement in test-time scaling while maintaining statistical guarantees through conformal prediction.

<div align="center">
  <a href="https://arxiv.org/abs/2509.15148"><img src="https://img.shields.io/badge/arXiv-2509.15148-b31b1b.svg" alt="arXiv" /></a>
  <a href="https://iclr.cc/Conferences/2026"><img src="https://iclr.cc/static/core/img/iclr-navbar-logo.svg" alt="ICLR 2026" height="20" /></a>
</div>

## 🔧 Installation

### ⚠️ Critical: Install SGLang 0.4.3.post4 and sgl-kernel

**ATTS requires SGLang 0.4.3.post4 and sgl-kernel 0.0.3.post6.** Since `sgl-kernel` is not on PyPI, we provide a pre-built wheel in `third_party/sgl_kernel.zip`. Use the following steps (from the **repo root**).

**1. Prerequisites**

- Python 3.11
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- CUDA Toolkit (with `nvcc` in `PATH` if you build from source)
- PyTorch with CUDA support (will be installed via `requirements.txt`)

```bash
# Optional: ensure CUDA is in PATH
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

**2. Create virtual environment and activate**

```bash
uv venv .sglang --python 3.11
source .sglang/bin/activate
```

**3. Install sgl-kernel from bundled wheel**

Unzip the pre-built sgl-kernel and copy it into the venv’s site-packages (paths assume you are in repo root for step 2, then `cd third_party` here):

```bash
cd third_party
unzip sgl_kernel.zip
cp -r 0.0.3.post6-cp39-abi3-manylinux2014_x86_64/sgl_kernel ../.sglang/lib/python3.11/site-packages/
cp -r 0.0.3.post6-cp39-abi3-manylinux2014_x86_64/sgl_kernel-0.0.3.post6.dist-info ../.sglang/lib/python3.11/site-packages/
rm -rf 0.0.3.post6-cp39-abi3-manylinux2014_x86_64/
```

**4. Install remaining dependencies (SGLang and extras)**

Still in `third_party/`:

```bash
uv pip install -r requirements.txt
cd ..
```

This installs SGLang 0.4.3.post4 from the bundled source (`third_party/sglang-0.4.3.post4/python`) and all other dependencies.

**5. Verify installation**

```bash
python -c "import sglang; print('sglang version:', sglang.__version__); from sglang import Engine; print('OK')"
```

Expected output: `sglang version: 0.4.3.post4` and `OK`.

## 🚀 Quick Start

### 1. Launch SGLang Servers

Start the SGLang servers for inference (run from repo root):

```bash
bash scripts/launch_sglang_servers.sh
```

**Default Configuration:**
- Small Model: `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` (Port 40000)
- Eval Model: `Qwen/QwQ-32B` (Port 40001)

**To customize models or GPUs**, edit the configuration variables in `scripts/launch_sglang_servers.sh`:
```bash
SMALL_MODEL="your-model"
EVAL_MODEL="your-eval-model"
SMALL_MODEL_DEVICE="0"
EVAL_MODEL_DEVICES="1,2"
```

**To stop servers:**
```bash
kill $(cat small_model.pid) $(cat eval_model.pid)
```

### 2. Prepare PPL Arrays

Generate perplexity arrays for conformal calibration (run from repo root):

```bash
bash scripts/suite_conformal.sh
```

This script will:
- Launch SGLang servers for small and evaluation models
- Compute PPL arrays for the specified datasets
- Save results to `.npy` files

### 3. Run Evaluation

#### Conformal Prediction Method

Run from repo root so the `ATTS` package is on the path:

```bash
python -m ATTS.ref_conformal \
    --small_model_name "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" \
    --eval_model_name "Qwen/QwQ-32B" \
    --dataset_name "aime24" \
    --ppl_array_path "ppls_aime24.npy" \
    --small_model_port 40000 \
    --eval_model_port 40001
```

#### Asynchronous Inference

```bash
python -m ATTS.ref_async \
    --small_model_name "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" \
    --eval_model_name "Qwen/QwQ-32B" \
    --dataset_name "aime24"
```

## 📊 Supported Datasets

- AIME24, AIME25
- AMC23
- MATH500
- Olympiad
- GPQA

## 🎯 Model Combinations

The framework supports various draft-target model pairs:
- DeepSeek-R1-Distill (1.5B/8B) + QwQ-32B
- Qwen2.5-7B + QwQ-32B
- Llama-3.1-8B + simplescaling/s1.1-32B

## 📝 Configuration

- **ATTS Python code** lives under **`ATTS/`** (e.g. `ref_conformal.py`, `ref_async.py`, `dataset.py`). Run from repo root: `python -m ATTS.ref_conformal ...`.
- **Shell scripts** (launch, suite, profiling, etc.) are under **`scripts/`**. Run them from the repo root, e.g. `bash scripts/suite_conformal.sh`.

Edit variables in the shell scripts to customize:
- `SAMPLE_SIZE`: Number of samples per question (default: 16)
- `SMALL_MODEL_MAX_TOKENS`: Max tokens for small model (default: 500)
- `SMALL_MODEL_TEMPERATURE`: Sampling temperature (default: 0.8)
- `CUDA_VISIBLE_DEVICES`: GPU allocation

---

## 🧪 Testing the Baselines

We provide two baselines in this repo for comparison with ATTS. You can reproduce them as follows.

### Baseline 1: SpecReason (`specreason/`)

[SpecReason](https://arxiv.org/abs/2504.07891) is a speculative-reasoning baseline (draft + target with vLLM). To **test** it:

1. **Environment** (from repo root):
   ```bash
   conda create -n specreason python=3.12 -y && conda activate specreason
   pip install vllm datasets
   ```
   For vLLM speculative decoding you may need to install from source; see [specreason/README.md](specreason/README.md).

2. **Start two vLLM servers** (e.g. in two terminals; 32B on 30000, 1.5B on 30001):
   ```bash
   VLLM_USE_V1=0 vllm serve Qwen/QwQ-32B --dtype auto -tp 2 --max_model_len 8192 --gpu-memory-utilization 0.8 --enable-prefix-caching --port 30000
   VLLM_USE_V1=0 vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --dtype auto -tp 2 --max_model_len 8192 --gpu-memory-utilization 0.1 --enable-prefix-caching --port 30001
   ```

3. **Run the SpecReason baseline** (single problem, optional: change `--problem_id` / `--dataset_name`):
   ```bash
   cd specreason
   mkdir -p results && OUTPUT_DIR=./results
   python spec_reason.py --dataset_name aime --problem_id 60 --repeat_id 0 --score_threshold 7.0 --score_method greedy --token_budget 8192 --output_dir "$OUTPUT_DIR"
   ```
   Results go to `specreason/results/`. For full datasets and batch scripts see [specreason/README.md](specreason/README.md) and `specreason/spec_reason_della_*.sh`.

### Baseline 2: Speculative Thinking (`speculative_thinking/`)

Speculative thinking baseline (SkyThought-style evals with sglang/vLLM). To **test** it:

1. **Environment** (from repo root):
   ```bash
   cd speculative_thinking
   python -m venv .venv && source .venv/bin/activate
   pip install sglang vllm   # see speculative_thinking/skythought_evals/requirements.txt for full deps
   ```

2. **Test normal (non-speculative) model** (no draft model):
   ```bash
   python ./skythought_evals/eval.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
       --evals amc23 --n 1 --result-dir ./eval_out --tp 2 --output-file ./eval_out/32B.txt
   ```

3. **Test speculative thinking** (draft + target). Pick a config from `speculative/config/` (e.g. `1b_14b.yml`) or add your own, then:
   ```bash
   python ./skythought_evals/eval.py --evals amc23 --n 1 --result-dir ./eval_out \
       --tp 3 --output-file ./eval_out/1b_14b.txt --spe_config ./speculative/config/1b_14b.yml
   ```
   Results are written to the paths given by `--result-dir` and `--output-file`. More options and config format: [speculative_thinking/README.md](speculative_thinking/README.md).

---

## 📄 Citation

```bibtex
@article{xiong2025atts,
  title={ATTS: Asynchronous Test-Time Scaling via Conformal Prediction},
  author={Xiong, Jing and Chen, Qiujiang and Ye, Fanghua and Wan, Zhongwei and Zheng, Chuanyang and Zhao, Chenyang and Shen, Hui and Li, Alexander Hanbo and Tao, Chaofan and Tan, Haochen and others},
  journal={arXiv preprint arXiv:2509.15148},
  url={https://arxiv.org/abs/2509.15148},
  year={2025}
}
```

## 📧 Contact

For questions or issues, please open a GitHub issue.
