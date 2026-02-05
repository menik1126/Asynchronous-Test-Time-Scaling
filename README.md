<h1 style="font-size: 0.1em; white-space: nowrap;">ATTS: Asynchronous Test-Time Scaling via Conformal Prediction</h1>

Official implementation of **ATTS: Asynchronous Test-Time Scaling via Conformal Prediction**.

> ATTS achieves up to **56.7x speedup** and **4.14x throughput** improvement in test-time scaling while maintaining statistical guarantees through conformal prediction.

<div align="center">
  <a href="https://arxiv.org/abs/2509.15148"><img src="https://img.shields.io/badge/arXiv-2509.15148-b31b1b.svg" alt="arXiv" /></a>
  <a href="https://iclr.cc/Conferences/2026"><img src="https://iclr.cc/static/core/img/iclr-navbar-logo.svg" alt="ICLR 2026" height="20" /></a>
</div>

## 🔧 Installation

### ⚠️ Critical: Install Specific SGLang Version

**You MUST install SGLang version 0.4.3.post4:**

```bash
pip install "sglang[all]==0.4.3.post4"
```

### Other Dependencies

```bash
pip install transformers torch numpy tqdm asyncio openai httpx nvtx
```

## 🚀 Quick Start

### 1. Launch SGLang Servers

Start the SGLang servers for inference:

```bash
bash launch_sglang_servers.sh
```

**Default Configuration:**
- Small Model: `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` (Port 40000)
- Eval Model: `Qwen/QwQ-32B` (Port 40001)

**To customize models or GPUs**, edit the configuration variables in `launch_sglang_servers.sh`:
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

Generate perplexity arrays for conformal calibration:

```bash
bash suite_conformal.sh
```

This script will:
- Launch SGLang servers for small and evaluation models
- Compute PPL arrays for the specified datasets
- Save results to `.npy` files

### 3. Run Evaluation

#### Conformal Prediction Method

```bash
python ref_conformal.py \
    --small_model_name "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" \
    --eval_model_name "Qwen/QwQ-32B" \
    --dataset_name "aime24" \
    --ppl_array_path "ppls_aime24.npy" \
    --small_model_port 40000 \
    --eval_model_port 40001
```

#### Asynchronous Inference

```bash
python ref_async.py \
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

Edit variables in shell scripts to customize:
- `SAMPLE_SIZE`: Number of samples per question (default: 16)
- `SMALL_MODEL_MAX_TOKENS`: Max tokens for small model (default: 500)
- `SMALL_MODEL_TEMPERATURE`: Sampling temperature (default: 0.8)
- `CUDA_VISIBLE_DEVICES`: GPU allocation

---

## 📦 Related Components

This repository also includes two related sub-projects under the same roof.

### SpecReason (`specreason/`)

[SpecReason](https://arxiv.org/abs/2504.07891) implements fast inference-time compute via speculative reasoning with vLLM.

**How to use:**

1. **Environment:** Create a conda env and install vLLM 0.8.2 (see `specreason/README.md` for vLLM speculative decoding fix if needed).
   ```bash
   conda create -n specreason python=3.12 -y && conda activate specreason
   pip install vllm datasets
   ```

2. **Launch two vLLM servers** (e.g. 32B base on port 30000, 1.5B small on port 30001):
   ```bash
   VLLM_USE_V1=0 vllm serve Qwen/QwQ-32B --dtype auto -tp 2 --max_model_len 8192 --gpu-memory-utilization 0.8 --enable-prefix-caching --port 30000
   VLLM_USE_V1=0 vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --dtype auto -tp 2 --max_model_len 8192 --gpu-memory-utilization 0.1 --enable-prefix-caching --port 30001
   ```

3. **Run SpecReason:**
   ```bash
   cd specreason
   mkdir -p results && OUTPUT_DIR=./results
   python spec_reason.py --dataset_name aime --problem_id 60 --repeat_id 0 --score_threshold 7.0 --score_method greedy --token_budget 8192 --output_dir "$OUTPUT_DIR"
   ```

Full details: [specreason/README.md](specreason/README.md).

### Speculative Thinking (`speculative_thinking/`)

Speculative thinking evaluation with sglang/vLLM (SkyThought-style evals).

**How to use:**

1. **Environment:** Use the project’s venv (Python 3.10, sglang, vLLM). From repo root:
   ```bash
   cd speculative_thinking
   source .venv/bin/activate   # or ./activate_env.sh if present
   ```

2. **Eval normal model:**
   ```bash
   python ./skythought_evals/eval.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
       --evals amc23 --n 1 --result-dir ./eval1/amc2323 \
       --tp 2 --output-file ./eval1/amc2323/32B.txt
   ```

3. **Eval speculative thinking:** Add a config under `speculative/config/` (see existing `.yml` there), then:
   ```bash
   python ./skythought_evals/eval.py --evals amc23 --n 1 --result-dir ./eval1/amc2323 \
       --tp 3 --output-file ./eval1/amc2323/1b_14b.txt --spe_config ./speculative/config/1b_14b.yml
   ```

Full details: [speculative_thinking/README.md](speculative_thinking/README.md).

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
