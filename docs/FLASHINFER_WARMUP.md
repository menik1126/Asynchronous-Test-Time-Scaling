# FlashInfer Kernel 预编译指南

## 问题背景

### 什么是 sm90？

**sm90** 是 NVIDIA Hopper 架构（H100 GPU）的 Streaming Multiprocessor 版本 9.0。

FlashInfer 会为不同的 GPU 架构编译优化的 CUDA kernel：
- **sm90 kernels**：专为 H100 优化，性能更好但编译更复杂
- **通用 kernels**：兼容多种架构，稳定但性能略低

### 为什么需要预编译？

FlashInfer 使用 **JIT (Just-In-Time) 编译**：
- 第一次使用某个 kernel 配置时，会**现场编译** CUDA 代码
- sm90 的 kernel 编译非常慢（可能需要几分钟甚至更久）
- 如果编译过程中出现问题（资源不足、并发冲突等），可能导致**卡住或失败**

**症状**：
```
2026-02-13 14:04:34 - INFO - flashinfer.jit: Loading JIT ops: batch_prefill_..._sm90
（此后约 10 分钟没有任何日志，最终超时）
```

---

## 解决方案

### 方案 1：预编译 sm90 kernels（推荐）

在 Docker 容器启动后、运行服务前，手动触发预编译：

```bash
# 1. 激活 Python 环境
cd /home/hku/Asynchronous-Test-Time-Scaling-Server/Asynchronous-Test-Time-Scaling
source .sglang/bin/activate

# 2. 检查当前缓存状态
python3 scripts/warmup_flashinfer_kernels.py --check

# 3. 清理未完成的编译（如果有）
python3 scripts/warmup_flashinfer_kernels.py --clear-incomplete

# 4. 预编译常用 kernels（启用 sm90 优化）
python3 scripts/warmup_flashinfer_kernels.py

# 5. 再次检查，确认 .so 文件已生成
python3 scripts/warmup_flashinfer_kernels.py --check
```

**预期输出**：
```
================================================================================
FlashInfer Kernel Warmup Tool
================================================================================

📂 FlashInfer cache directory: /home/hku/.cache/flashinfer
✅ Found 2 compiled kernel(s)

================================================================================
Starting FlashInfer Kernel Warmup
================================================================================
✓ SM90 optimizations ENABLED
✓ FlashInfer version: 0.2.2.post1
✓ Using device: cuda:0, dtype: torch.bfloat16

🔨 Compiling 3 kernel configuration(s)...
--------------------------------------------------------------------------------

[1/3] Decode (single token)
  Config: batch=1, qo_len=1, kv_len=128, heads=32/8
  ✓ Compiled successfully in 2.34s

[2/3] Prefill (short)
  Config: batch=1, qo_len=128, kv_len=128, heads=32/8
  ✓ Compiled successfully in 45.67s    ← sm90 prefill 编译较慢是正常的

[3/3] Prefill (medium)
  Config: batch=1, qo_len=512, kv_len=512, heads=32/8
  ✓ Compiled successfully in 3.12s

================================================================================
Warmup Complete!
================================================================================
```

**编译完成后**，启动服务时会直接加载缓存的 `.so` 文件，不再 JIT 编译。

---

### 方案 2：禁用 sm90 优化（临时规避）

如果 sm90 编译一直失败或卡住，可以禁用它：

```bash
# 预编译通用 kernels（不使用 sm90）
python3 scripts/warmup_flashinfer_kernels.py --disable-sm90
```

然后在启动服务时也禁用 sm90：

```bash
export FLASHINFER_DISABLE_SM90=1
./scripts/launch_sglang_servers.sh
```

**性能影响**：通用 kernel 比 sm90 慢约 5-15%，但稳定性更好。

---

## 在 Docker 镜像中集成预编译

### 方法 1：在 Dockerfile 中预编译

在构建 Docker 镜像时运行预编译：

```dockerfile
# 在安装 sglang 之后
RUN source .sglang/bin/activate && \
    python3 scripts/warmup_flashinfer_kernels.py --clear-incomplete && \
    python3 scripts/warmup_flashinfer_kernels.py
```

**注意**：
- 需要在**有 GPU 的环境**中构建镜像（`docker build` 时需 `--gpus all`）
- 或者使用 **multi-stage build**，在运行时容器中首次启动时预编译

### 方法 2：在容器启动脚本中预编译

创建 `/entrypoint.sh`：

```bash
#!/bin/bash
set -e

# 首次启动时预编译
WARMUP_FLAG="/home/hku/.cache/flashinfer/.warmup_done"
if [ ! -f "$WARMUP_FLAG" ]; then
    echo "First run: warming up FlashInfer kernels..."
    cd /home/hku/Asynchronous-Test-Time-Scaling-Server/Asynchronous-Test-Time-Scaling
    source .sglang/bin/activate
    python3 scripts/warmup_flashinfer_kernels.py --clear-incomplete
    python3 scripts/warmup_flashinfer_kernels.py
    touch "$WARMUP_FLAG"
fi

# 启动服务
exec "$@"
```

在 `docker-compose.yml` 或启动命令中使用：

```yaml
services:
  sglang:
    entrypoint: ["/entrypoint.sh"]
    command: ["./scripts/launch_sglang_servers.sh"]
```

---

## 缓存位置与持久化

FlashInfer 编译好的 kernel 缓存在：

```
$HOME/.cache/flashinfer/{arch}/cached_ops/
```

- `{arch}`：根据 GPU 自动检测，如 `90` (Hopper), `80` (Ampere) 等
- 每个 kernel 配置一个子目录，包含：
  - `*.so`：编译好的共享库
  - `build.ninja`、`*.o.d`：编译中间文件

### Docker 中持久化缓存

在 `docker-compose.yml` 中挂载卷：

```yaml
services:
  sglang:
    volumes:
      - flashinfer-cache:/home/hku/.cache/flashinfer

volumes:
  flashinfer-cache:
```

或使用主机目录：

```yaml
volumes:
  - ./cache/flashinfer:/home/hku/.cache/flashinfer
```

这样重启容器后不需要重新编译。

---

## 故障排查

### 1. 编译卡住 / 超时

**症状**：`Loading JIT ops: ..._sm90` 后长时间没有 `Finished loading`

**原因**：
- 系统资源不足（CPU、内存）
- 多个进程同时编译导致死锁
- CUDA Toolkit 版本与驱动不匹配

**解决**：
```bash
# 清理未完成的编译
python3 scripts/warmup_flashinfer_kernels.py --clear-incomplete

# 单独编译（确保没有其他进程使用 GPU）
python3 scripts/warmup_flashinfer_kernels.py
```

### 2. 检查已编译的 kernels

```bash
find ~/.cache/flashinfer/ -name "*.so" -exec ls -lh {} \;
```

每个有 `.so` 文件的配置都已成功编译。

### 3. 查看编译日志

```bash
tail -100 ~/.cache/flashinfer/90/flashinfer_jit.log
```

如果看到：
```
Loading JIT ops: xxx
Finished loading JIT ops: xxx  ← 说明成功
```

如果只有 `Loading` 没有 `Finished`，说明编译失败或卡住。

### 4. 完全重置缓存

```bash
rm -rf ~/.cache/flashinfer/
python3 scripts/warmup_flashinfer_kernels.py
```

---

## 性能对比

| 配置 | 首次启动时间 | 后续启动时间 | 推理性能 |
|------|-------------|-------------|----------|
| **无预编译 + sm90** | 10+ 分钟（可能卡住） | 正常 | 最佳 |
| **预编译 + sm90** | 正常（30 秒） | 正常 | 最佳 |
| **禁用 sm90** | 正常（30 秒） | 正常 | -5~15% |

**推荐**：在生产环境中使用**预编译 + sm90**。

---

## 参考

- FlashInfer 官方文档: https://github.com/flashinfer-ai/flashinfer
- CUDA Compute Capability: https://developer.nvidia.com/cuda-gpus
- SGLang 文档: https://github.com/sgl-project/sglang
