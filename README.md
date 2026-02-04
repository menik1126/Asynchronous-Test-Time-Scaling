# ATTS: Asynchronous Test-Time Scaling via Conformal Prediction

Official implementation of **ATTS: Asynchronous Test-Time Scaling via Conformal Prediction**.

> ATTS achieves up to **56.7x speedup** and **4.14x throughput** improvement in test-time scaling while maintaining statistical guarantees through conformal prediction.

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

First, start the SGLang servers for inference:

```bash
# Launch with default models (DeepSeek-R1-Distill-Llama-8B + QwQ-32B)
bash launch_sglang_servers.sh

# Or specify custom models and GPUs
bash launch_sglang_servers.sh \
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" \  # Small model
    "Qwen/QwQ-32B" \                               # Eval model
    "0" \                                           # Small model GPU
    "1,2" \                                         # Eval model GPUs
    "0.9" \                                         # Memory fraction
    "2"                                             # Tensor parallel size
```

The script will:
- Launch small model server on port 40000
- Launch evaluation model server on port 40001
- Wait for both servers to be ready
- Save PIDs to `small_model.pid` and `eval_model.pid`

**To stop servers:**
```bash
# Using saved PIDs
kill $(cat small_model.pid) $(cat eval_model.pid)

# Or manually
kill <SMALL_MODEL_PID> <EVAL_MODEL_PID>
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
  year={2025}
}
```

## 📧 Contact

For questions or issues, please open a GitHub issue.
