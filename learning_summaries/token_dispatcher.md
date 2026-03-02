# Token Dispatcher (DeepEP) MoE 通信深度解析

**分析时间**: 2026-03-02 03:30  
**代码路径**: `python/sglang/srt/layers/moe/token_dispatcher/deepep.py`  
**优先级**: P1 - MoE 核心模块

---

## 一、模块定位

### 1.1 核心职责

Token Dispatcher 是 MoE 架构的**通信引擎**，负责：
- Token 到专家的路由分发 (Dispatch)
- 专家计算结果合并 (Combine)
- DeepEP 通信优化 (Normal/Low Latency 模式)

### 1.2 DeepEP 架构

```
DeepEP = Deep Learning Expert Parallel
- Normal 模式：Prefill 阶段，大批量 tokens
- Low Latency 模式：Decode 阶段，小批量 tokens
```

---

## 二、核心类结构

### 2.1 DeepEPDispatcher

```python
class DeepEPDispatcher(BaseDispatcher):
    def __init__(self, deepep_mode=DeepEPMode.AUTO, ...):
        # 根据模式创建对应的分发器
        if deepep_mode.enable_low_latency():
            self._low_latency_dispatcher = _DeepEPDispatcherImplLowLatency(...)
        if deepep_mode.enable_normal():
            self._normal_dispatcher = _DeepEPDispatcherImplNormal(...)
```

### 2.2 DeepEPBuffer (单例)

```python
class DeepEPBuffer:
    _buffer = None  # 单例
    
    @classmethod
    def get_deepep_buffer(cls, group, hidden_size, ...):
        if cls._buffer is not None:
            return cls._buffer
        
        # 创建 DeepEP Buffer
        cls._buffer = Buffer(
            group,
            num_nvl_bytes,    # NVLink 缓冲区
            num_rdma_bytes,   # RDMA 缓冲区
            low_latency_mode=deepep_mode.enable_low_latency(),
        )
        return cls._buffer
```

---

## 三、DeepEP 两种模式

### 3.1 Normal 模式

**适用场景**: Prefill 阶段，大批量 tokens

**特点**:
- 使用 NVLink/RDMA 优化
- 吞吐量优先
- 批量处理

```python
class DeepEPNormalDispatchOutput(NamedTuple):
    hidden_states: torch.Tensor
    hidden_states_scale: Optional[torch.Tensor]  # FP8 量化
    topk_ids: torch.Tensor                        # 专家 ID
    topk_weights: torch.Tensor                    # 专家权重
    num_recv_tokens_per_expert: List[int]         # 每专家 token 数
    
    @property
    def format(self) -> DispatchOutputFormat:
        return DispatchOutputFormat.DEEPEP_NORMAL
```

### 3.2 Low Latency 模式

**适用场景**: Decode 阶段，小批量 tokens

**特点**:
- 优化延迟
- 适合实时推理
- 单 token 处理

```python
class DeepEPLLDispatchOutput(NamedTuple):
    hidden_states: torch.Tensor
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    masked_m: torch.Tensor      # 掩码
    expected_m: int             # 期望 token 数
    
    @property
    def format(self) -> DispatchOutputFormat:
        return DispatchOutputFormat.DEEPEP_LL
```


### 3.3 运行时模式切换（代码对齐）

DeepEP 不是只在初始化时确定模式。运行时可以通过：

```python
DeepEPBuffer.set_dispatch_mode(mode)
# mode.is_low_latency() -> set_dispatch_mode_as_low_latency()
# mode.is_normal()      -> set_dispatch_mode_as_normal()
```

在进入 Low Latency 前会清理 Normal 模式缓冲区，避免模式切换后的缓冲不一致。

---

## 四、Dispatch 流程

### 4.1 Normal 模式 Dispatch

```python
def dispatch(self, hidden_states: torch.Tensor, topk_output: TopKOutput):
    # 1. 获取 top-k 专家选择
    topk_ids = topk_output.topk_ids      # [batch_size, top_k]
    topk_weights = topk_output.topk_weights
    
    # 2. 计算每专家 token 分布
    num_recv_tokens_per_expert = self.count_expert_tokens(topk_ids, num_experts)
    
    # 3. All-to-All 通信 (NVLink/RDMA)
    hidden_states_recv = self.buffer.dispatch(
        hidden_states,
        topk_ids,
        num_recv_tokens_per_expert,
    )
    
    # 4. 返回分发的 hidden states
    return DeepEPNormalDispatchOutput(
        hidden_states=hidden_states_recv,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        num_recv_tokens_per_expert=num_recv_tokens_per_expert,
    )
```

### 4.2 Low Latency 模式 Dispatch

```python
def dispatch_low_latency(self, hidden_states: torch.Tensor, topk_output: TopKOutput):
    # 1. 低延迟模式优化
    # - 减少内存拷贝
    # - 使用专用缓冲区
    
    # 2. 快速 All-to-All
    hidden_states_recv = self.buffer.dispatch_low_latency(
        hidden_states,
        topk_ids,
    )
    
    # 3. 返回结果
    return DeepEPLLDispatchOutput(
        hidden_states=hidden_states_recv,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        masked_m=masked_m,
        expected_m=expected_m,
    )
```

---

## 五、Combine 流程

### 5.1 Normal 模式 Combine

```python
def combine(self, expert_output: torch.Tensor, dispatch_output: DeepEPNormalDispatchOutput):
    # 1. 专家计算完成
    # expert_output: [total_tokens, hidden_size]
    
    # 2. All-to-All 逆向通信
    combined_hidden = self.buffer.combine(
        expert_output,
        dispatch_output.topk_ids,
        dispatch_output.num_recv_tokens_per_expert,
    )
    
    # 3. 加权求和
    final_output = combined_hidden * dispatch_output.topk_weights
    
    return final_output
```

---

## 六、通信优化技术

### 6.1 NVLink 优化

```python
# NVLink 缓冲区大小计算
num_nvl_bytes = config.get_nvl_buffer_size_hint(
    hidden_bytes=hidden_size * param_bytes,
    num_ranks=group.size(),
)
```

### 6.2 RDMA 优化

```python
# RDMA 缓冲区大小计算
num_rdma_bytes = max(
    Buffer.get_low_latency_rdma_size_hint(
        num_max_dispatch_tokens_per_rank,
        hidden_size,
        group.size(),
        num_experts,
    ),
)
```

### 6.3 FP8 量化

```python
if not _is_npu:
    # FP8 量化减少通信量
    hidden_states_fp8, scale = sglang_per_token_group_quant_fp8(
        hidden_states, torch.float8_e4m3fn
    )
```

---

## 七、与现有文档关联

### 7.1 补充 parallel_communication.md

| 现有内容 | 补充内容 |
|---------|---------|
| DeepEP 架构 | 详细实现代码 |
| 模式对比 | Normal/LL 输出结构 |
| 初始化流程 | Buffer 单例管理 |

---

## 八、总结

**Token Dispatcher 核心价值**:
1. ✅ MoE 专家路由核心
2. ✅ DeepEP 双模式优化
3. ✅ NVLink/RDMA 通信加速
4. ✅ FP8 量化减少带宽

---

**分析用时**: 20 分钟  
**代码行数**: ~900 行  
**产出文档**: token_dispatcher.md
