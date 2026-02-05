<h1 style="font-size: 1.15em; white-space: nowrap;">ATTS: Asynchronous Test-Time Scaling via Conformal Prediction</h1>

Official implementation of **ATTS: Asynchronous Test-Time Scaling via Conformal Prediction**.

> ATTS achieves up to **56.7x speedup** and **4.14x throughput** improvement in test-time scaling while maintaining statistical guarantees through conformal prediction.

<div align="center">
  <a href="https://arxiv.org/abs/2509.15148"><img src="https://img.shields.io/badge/arXiv-2509.15148-b31b1b.svg" alt="arXiv" /></a>
  <a href="https://iclr.cc/Conferences/2026"><img src="https://iclr.cc/static/core/img/iclr-navbar-logo.svg" alt="ICLR" height="20" style="vertical-align: middle;" /> ICLR 2026</a>
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
