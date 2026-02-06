export CUDA_VISIBLE_DEVICES=0
python -m sglang.launch_server \
    --model-path simplescaling/s1.1-7B \
    --inference-model target \
    --host 0.0.0.0 \
    --disable-cuda-graph \
    --disable-overlap-schedule \
    --port 30030
