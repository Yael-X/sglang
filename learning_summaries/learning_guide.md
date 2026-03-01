# SGLang 学习指南与实验手册

**创建时间**: 2026-03-02 03:00  
**基于分析**: 7 个核心模块，16,000+ 行代码  
**目标读者**: SGLang 学习者、贡献者、应用开发者

---

## 一、学习路径

### 1.1 入门路径 (2-3 周)

```
第 1 周：架构理解
├─ Day 1-2: 阅读 architechture.md
│   └─ 理解三层架构 (Frontend/Runtime/Hardware)
├─ Day 3-4: Scheduler 调度器
│   └─ 理解 Continuous Batching 机制
└─ Day 5-7: RadixAttention
    └─ 理解 KV Cache 复用原理

第 2 周：核心实现
├─ Day 1-3: KV Cache Manager
│   └─ 理解三层内存池架构
├─ Day 4-7: Model Runner
    └─ 理解模型执行与优化

第 3 周：高级主题
├─ Day 1-3: FlashInfer Attention
│   └─ 理解注意力后端
├─ Day 4-5: Token Dispatcher
│   └─ 理解 MoE 通信
└─ Day 6-7: 综合实验
    └─ 性能测试与优化
```

### 1.2 进阶路径 (1-2 月)

```
阶段 1: 源码精读
├─ 逐模块阅读核心代码
├─ 绘制详细流程图
└─ 记录关键设计决策

阶段 2: 实验验证
├─ 性能基准测试
├─ 参数调优实验
└─ 对比不同配置

阶段 3: 贡献代码
├─ 修复简单 bug
├─ 优化文档
└─ 实现小功能
```

---

## 二、实验手册

### 2.1 环境搭建

```bash
# 1. 克隆仓库
git clone https://github.com/sgl-project/sglang.git
cd sglang

# 2. 安装依赖
pip install -e "python[all]"

# 3. 验证安装
python -c "import sglang; print(sglang.__version__)"

# 4. 启动服务
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --port 30000
```

### 2.2 实验 1: Continuous Batching 性能测试

**目标**: 验证 Continuous Batching 相比静态批处理的性能提升

**步骤**:
```python
import sglang as sgl
import time

@sgl.function
def batch_generate(s, prompt):
    s += prompt
    s += sgl.gen("output", max_tokens=100)

# 准备数据
prompts = ["Hello, "] * 32  # 32 个相同请求

# 静态批处理 (模拟)
start = time.time()
for prompt in prompts:
    sglang.run(batch_generate, prompt)
static_time = time.time() - start

# Continuous Batching
start = time.time()
sglang.run(batch_generate, prompts)  # 批量传入
cb_time = time.time() - start

print(f"Static: {static_time:.2f}s")
print(f"Continuous Batching: {cb_time:.2f}s")
print(f"Speedup: {static_time/cb_time:.2f}x")
```

**预期结果**: 2-3x 加速

### 2.3 实验 2: RadixAttention 复用率测试

**目标**: 测量多轮对话场景下的 KV Cache 复用率

**步骤**:
```python
import sglang as sgl

@sgl.function
def multi_turn_chat(s, history):
    for turn in history:
        s += turn["role"] + ": " + turn["content"] + "\n"
        s += sgl.gen("response", max_tokens=50)

# 多轮对话
history = [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi there!"},
    {"role": "user", "content": "How are you?"},
    # ... 更多轮次
]

# 运行并查看日志
# 关注 RadixCache match_prefix 命中率
```

**预期结果**: 50-70% 重复计算减少

### 2.4 实验 3: 调度策略对比

**目标**: 对比 FCFS vs LPM 调度策略

**配置**:
```bash
# FCFS (先来先服务)
python -m sglang.launch_server \
    --schedule-policy fcfs

# LPM (最长前缀匹配)
python -m sglang.launch_server \
    --schedule-policy lpm
```

**测试场景**:
- 多轮对话 (LPM 优势)
- 独立请求 (FCFS 足够)
- 混合负载

**指标**:
- 平均延迟 (P50, P99)
- 吞吐量 (tokens/s)
- KV Cache 命中率

### 2.5 实验 4: CUDA Graph 优化

**目标**: 测量 CUDA Graph 对 Decode 延迟的影响

**配置**:
```bash
# 启用 CUDA Graph
python -m sglang.launch_server \
    --disable-cuda-graph  # 禁用

python -m sglang.launch_server \
    # 默认启用  # 启用
```

**测试**:
```python
# Decode 延迟测试
import time

@sgl.function
def decode(s, prompt):
    s += prompt
    for _ in range(100):
        s += sgl.gen("token", max_tokens=1)

start = time.time()
decode.run("Hello, ")
decode_time = time.time() - start

print(f"Decode latency: {decode_time/100*1000:.2f}ms/token")
```

**预期结果**: 2-5x 延迟降低

---

## 三、代码疑点清单

### 3.1 待深入问题

#### Q1: Overlap Scheduler 的 FutureMap 实现

**位置**: `scheduler.py::event_loop_overlap`

**疑点**:
```python
# FutureMap 如何管理异步结果？
# batch_result 如何与 last_batch 对应？
self.result_queue.append((batch.copy(), batch_result))
```

**待验证**:
- FutureMap 的内存管理
- 错误处理机制
- 边界情况 (空批次)

#### Q2: MoE TP/EP 协同

**位置**: `parallel_state.py::initialize_model_parallel`

**疑点**:
```python
# TP 和 EP 如何协同工作？
# 专家路由时 TP 组内如何通信？
moe_tp_size = tensor_model_parallel_size // moe_ep_size
```

**待验证**:
- 专家分配策略
- All-to-All 通信优化
- 负载均衡

#### Q3: Pipeline Parallel 与 Attention 协同

**位置**: `scheduler_pp_mixin.py`

**疑点**:
```python
# PP 如何与 Continuous Batching 协同？
# 跨 stage 的 KV Cache 如何管理？
```

**待验证**:
- 流水线气泡处理
- KV Cache 跨 GPU 传输
- 1F1B 调度策略

#### Q4: FP8 量化精度影响

**位置**: `memory_pool.py`, `flashinfer_backend.py`

**疑点**:
```python
# FP8 量化对模型精度的影响？
# 哪些层适合量化？
hidden_states_fp8, scale = sglang_per_token_group_quant_fp8(...)
```

**待验证**:
- 量化敏感度分析
- 混合精度策略
- 校准方法

### 3.2 实验建议

#### 实验 A: 量化精度测试

```python
# 对比 FP16 vs FP8
model_fp16 = load_model(dtype=torch.float16)
model_fp8 = load_model(dtype=torch.float8_e4m3fn)

# 在 benchmark 上测试
ppl_fp16 = evaluate_perplexity(model_fp16)
ppl_fp8 = evaluate_perplexity(model_fp8)

print(f"PPL delta: {ppl_fp8 - ppl_fp16}")
```

#### 实验 B: 专家负载均衡

```python
# 记录专家使用频率
expert_usage = Counter()

for request in requests:
    topk_ids = route_experts(request)
    expert_usage.update(topk_ids)

# 分析负载分布
print(expert_usage.most_common())
```

---

## 四、性能优化建议

### 4.1 配置优化

#### 显存优化
```bash
# 调整 KV Cache 大小
python -m sglang.launch_server \
    --mem-fraction-static 0.8  # 默认 0.9

# 调整 page size
python -m sglang.launch_server \
    --page-size 32  # 默认 1
```

#### 吞吐量优化
```bash
# 增大最大并发请求
python -m sglang.launch_server \
    --max-running-requests 1024

# 启用重叠调度
python -m sglang.launch_server \
    --enable-overlap-schedule  # 默认启用
```

#### 延迟优化
```bash
# 启用 CUDA Graph
python -m sglang.launch_server \
    # 默认启用

# 启用 FlashInfer
python -m sglang.launch_server \
    --attention-backend flashinfer
```

### 4.2 代码优化

#### 减少 Host-Device 拷贝
```python
# Bad: 每次 plan 都拷贝 indptr
kv_indptr = kv_indptr.cpu()
wrapper.plan(kv_indptr=kv_indptr)

# Good: 使用缓存
global_override_indptr_cpu = kv_indptr
wrapper.plan()  # 自动使用缓存
```

#### 优化 RadixCache 插入
```python
# Bad: 频繁插入小请求
for req in requests:
    radix_cache.insert(req)

# Good: 批量插入
batch_insert(radix_cache, requests)
```

---

## 五、常见问题 FAQ

### Q: 如何选择 schedule-policy？

**A**:
- **多轮对话**: LPM (最长前缀匹配)
- **独立请求**: FCFS (先来先服务)
- **混合负载**: LPM + Priority

### Q: 显存不足怎么办？

**A**:
1. 减小 `--mem-fraction-static`
2. 增大 `--page-size`
3. 启用 KV Cache 量化 (`--kv-cache-dtype fp8`)
4. 减少 `--max-running-requests`

### Q: 如何提高吞吐量？

**A**:
1. 增大 batch size (`--max-running-requests`)
2. 启用 Continuous Batching (默认)
3. 使用 FlashInfer 后端
4. 启用 CUDA Graph

### Q: 如何降低延迟？

**A**:
1. 启用 CUDA Graph (默认)
2. 使用 Tensor Cores (FP8)
3. 减小 batch size
4. 启用 Low Latency 模式 (MoE)

---

## 六、贡献指南

### 6.1 文档贡献

**适合新手**:
- 修正拼写错误
- 补充代码注释
- 添加使用示例
- 翻译文档

**步骤**:
```bash
# 1. Fork 仓库
git clone https://github.com/YOUR_USERNAME/sglang.git

# 2. 创建分支
git checkout -b docs/improve-xxx

# 3. 修改文档
vim docs/xxx.md

# 4. 提交 PR
git add .
git commit -m "docs: improve xxx section"
git push origin docs/improve-xxx
```

### 6.2 代码贡献

**适合新手**:
- 修复简单 bug (good first issue)
- 添加单元测试
- 优化日志输出
- 改进错误信息

**步骤**:
```bash
# 1. 选择 issue
# 查看 GitHub Issues 标签 "good first issue"

# 2. 本地开发
git checkout -b fix/xxx

# 3. 编写测试
pytest tests/xxx/test_xxx.py

# 4. 提交 PR
git add .
git commit -m "fix: resolve xxx issue"
git push origin fix/xxx
```

---

## 七、资源链接

### 7.1 官方资源

- **GitHub**: https://github.com/sgl-project/sglang
- **文档**: https://docs.sglang.ai
- **Discord**: https://discord.gg/sglang

### 7.2 学习资源

- **FlashInfer**: https://github.com/flashinfer-ai/flashinfer
- **vLLM**: https://github.com/vllm-project/vllm
- **PagedAttention 论文**: https://arxiv.org/abs/2309.06180

### 7.3 相关技术

- **CUDA Programming**: https://docs.nvidia.com/cuda/
- **Triton**: https://github.com/openai/triton
- **DeepEP**: https://github.com/deepseek-ai/DeepEP

---

**持续更新中...** 🚀
