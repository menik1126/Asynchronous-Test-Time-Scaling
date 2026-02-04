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

### 1. Start SGLang Servers

First, start the SGLang servers for inference:

```bash
# Start servers with default models (DeepSeek-R1-Distill-Llama-8B + QwQ-32B)
bash start_servers.sh

# Or specify custom models
bash start_servers.sh "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" "Qwen/QwQ-32B"

# Configure GPUs (optional)
SMALL_GPU=0 EVAL_GPU=1,2 bash start_servers.sh
```

The servers will run on:
- **Small Model**: `http://127.0.0.1:40000`
- **Evaluation Model**: `http://127.0.0.1:40001`

To stop the servers:
```bash
bash stop_servers.sh
```

**Note**: The `suite_*.sh` scripts automatically start and stop servers, so you don't need to run `start_servers.sh` when using them.

### 2. Prepare PPL Arrays (Optional)

Generate perplexity arrays for conformal calibration:

```bash
bash suite_conformal.sh
```

This script will automatically:
- Launch SGLang servers
- Compute PPL arrays for the specified datasets
- Save results to `.npy` files
- Clean up servers

### 3. Run Evaluation

#### Option A: Manual Evaluation (Requires servers running)

**Conformal Prediction Method:**
```bash
# Make sure servers are running first (bash start_servers.sh)
python ref_conformal.py \
    --small_model_name "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" \
    --eval_model_name "Qwen/QwQ-32B" \
    --dataset_name "aime24" \
    --ppl_array_path "ppls_aime24.npy" \
    --small_model_port 40000 \
    --eval_model_port 40001
```

**Asynchronous Inference:**
```bash
# Make sure servers are running first (bash start_servers.sh)
python ref_async.py \
    --small_model_name "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" \
    --eval_model_name "Qwen/QwQ-32B" \
    --dataset_name "aime24"
```

#### Option B: Automated Suite (Handles servers automatically)

```bash
# Runs complete pipeline: start servers → evaluate → stop servers
bash suite_async.sh        # Asynchronous evaluation
bash suite_conformal.sh    # Conformal prediction evaluation
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
