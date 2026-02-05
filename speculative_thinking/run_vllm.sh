export CUDA_VISIBLE_DEVICES=1,2
python /home/xiongjing/speculative_thinking/skythought_evals/eval.py \
    --evals amc23 --n 16 --result-dir ./eval1/amc2323 \
    --tp 2 --output-file ./eval1/amc2323/1b_32b.txt --spe_config ./speculative/config/1b_32b.yml