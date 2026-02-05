export CUDA_VISIBLE_DEVICES=1,2
python /root/xj/sglang-parallel-test-time-scaling/speculative_thinking/skythought_evals/eval.py \
    --evals aime24 --n 16 --result-dir ./eval_sglang/aime24 \
    --tp 2 --output-file ./eval_sglang/aime24/1b_32b_sglang.txt --spe_config ./speculative/config/1b_32b_sglang.yml \
    --start 0 --end 15