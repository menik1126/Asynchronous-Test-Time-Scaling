export CUDA_VISIBLE_DEVICES=0
python -m sglang.launch_server \
    --model-path Qwen/Qwen2-7B \
    --draft-model-path Qwen/Qwen2-7B \
    --inference-model draft \
    --host 0.0.0.0 \
    --port 30020
