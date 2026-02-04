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

First, launch two SGLang servers (small model and evaluation model):

```bash
# Terminal 1: Start small model server (e.g., DeepSeek-R1-Distill-Llama-8B)
CUDA_VISIBLE_DEVICES=0 python3 -m sglang.launch_server \
    --model-path deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --tp 1 \
    --mem-fraction-static 0.9 \
    --host 0.0.0.0 \
    --port 40000

# Terminal 2: Start evaluation model server (e.g., QwQ-32B)
CUDA_VISIBLE_DEVICES=1,2 python3 -m sglang.launch_server \
    --model-path Qwen/QwQ-32B \
    --tp 2 \
    --mem-fraction-static 0.9 \
    --host 0.0.0.0 \
    --port 40001
```

**Important Notes:**
- Small model uses port `40000`, evaluation model uses port `40001`
- Adjust `CUDA_VISIBLE_DEVICES` and `--tp` based on your GPU setup
- Wait for both servers to be ready before proceeding

### 2. Prepare PPL Arrays

Generate perplexity arrays for conformal calibration:

```bash
bash suite_conformal.sh
```

This script will automatically:
- Launch SGLang servers for configured model pairs
- Compute PPL arrays for the specified datasets
- Save results to `.npy` files

**Or manually run:**

```bash
python baseline.py \
    --small_model_name "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" \
    --eval_model_name "Qwen/QwQ-32B" \
    --dataset_name "aime24" \
    --ppl_array_path "ppls_aime24.npy"
```

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
