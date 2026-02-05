#!/bin/bash

#SBATCH  -J stage
#SBATCH  -N 1
#SBATCH  --gres=gpu:2
#SBATCH --qos=high_priority
#SBATCH  --output=qj/sglang-parallel-test-time-scaling/qiu_sglang_server.out
#SBATCH  --error=qj/sglang-parallel-test-time-scaling/qiu_sglang_server.out
#SBATCH  --time=7-00:00:00 
#SBATCH  --partition=LocalQ
#SBATCH  --ntasks=1

# 激活虚拟环境
source /home/xiongjing/resource_dir/qiu/.sglang/bin/activate


nvidia-smi

pwd
# bash run.sh llama2-7b-chat synthetic
# bash run.sh llama3-8b-instruct-fullkv synthetic FullKV 512 3 1
# bash run.sh llama3-8b-instruct-uncomp-multi synthetic uncomp 512 3 5
# bash run.sh llama3-8b-instruct-uncomp-stage-single synthetic uncomp_stage 512 3 1

wait_for_server() {
    local port=$1
    while true; do
        # Try to connect to the server
        curl -s http://localhost:$port/get_model_info > /dev/null
        if [ $? -eq 0 ]; then
            echo "Server on port $port is ready!"
            break
        else
            echo "Waiting for server on port $port to start..."
            sleep 10  # Wait 10 seconds before retrying
        fi
    done

}
# CUDA_VISIBLE_DEVICES=3 python3 -m sglang.launch_server --model-path simplescaling/s1.1-14B --host 0.0.0.0 --port 40000 --mem-fraction-static 0.8   &

# CUDA_VISIBLE_DEVICES=4 python3 -m sglang.launch_server --model-path Qwen/Qwen2.5-7B-Instruct --host 0.0.0.0 --port 40001 --mem-fraction-static 0.7  &

# kill -9 $(lsof eval.log | awk 'NR>1 {print $2}' | sort -u)
# kill -9 $(lsof small.log | awk 'NR>1 {print $2}' | sort -u)

# wait_for_server 40000
# wait_for_server 40001
# wait

# CUDA_VISIBLE_DEVICES=3 python3 -m sglang.launch_server --model-path simplescaling/s1.1-14B --host 0.0.0.0 --port 40000 --mem-fraction-static 0.8 &

# CUDA_VISIBLE_DEVICES=4 python3 -m sglang.launch_server --model-path Zyphra/ZR1-1.5B --host 0.0.0.0 --port 40001 --mem-fraction-static 0.7 &

CUDA_VISIBLE_DEVICES=3 python3 -m sglang.launch_server --model-path Qwen/QwQ-32B --host 0.0.0.0 --port 8001   &

CUDA_VISIBLE_DEVICES=4 python3 -m sglang.launch_server --model-path deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --host 0.0.0.0 --port 30001  &

wait_for_server 8001
wait_for_server 30001

wait
