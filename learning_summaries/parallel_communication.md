# SGLang从推理到MOE的DeepEP并行策略和通信流程

## 目录

1. [并行策略架构](#一并行策略架构)
2. [并行组初始化](#二并行组初始化)
3. [DeepEP通信流程](#三deepep通信流程)
4. [完整的MoE推理流程](#四完整的moe推理流程)
5. [DeepEP之前的通信流程详解](#五deepep之前的通信流程详解)
6. [完整通信流程图](#六完整通信流程图)
7. [通信优化技术总结](#七通信优化技术总结)
8. [关键通信原语实现](#八关键通信原语实现)

---

## 一、并行策略架构

SGLang采用多层次的并行策略来支持大规模MoE模型的推理：

### 1.1 主要并行维度

| 并行维度 | 说明 | 典型应用 |
|---------|------|----------|
| **Tensor Parallelism (TP)** | 将模型权重在多个GPU间切分 | QKV投影、MLP/MoE层 |
| **Expert Parallelism (EP)** | 将MoE专家分配到不同GPU | MoE专家计算 |
| **Pipeline Parallelism (PP)** | 将模型层分配到不同GPU | 大模型分层 |
| **Data Parallelism (DP)** | 处理不同的请求批次 | 多请求并发 |

### 1.2 并行层次结构

```
总GPU数 = TP × PP × DP
MoE并行: EP ⊆ TP
MoE_TP = TP / EP
```

**示例配置**：
- 8个GPU，TP=4，PP=2，DP=1
- MoE模型：总专家数256，EP=4，则每个TP组有64个专家
- MoE_TP = 4/4 = 1，每个专家在TP组内不切分

---

## 二、并行组初始化

在[parallel_state.py](../python/sglang/srt/distributed/parallel_state.py#L1557)中的`initialize_model_parallel`函数中初始化所有并行组。

### 2.1 Tensor Parallel组初始化

```python
# 构建tensor model-parallel组
num_tensor_model_parallel_groups = world_size // tensor_model_parallel_size
group_ranks = []
for i in range(num_tensor_model_parallel_groups):
    ranks = list(
        range(i * tensor_model_parallel_size, (i + 1) * tensor_model_parallel_size)
    )
    group_ranks.append(ranks)

_TP = init_model_parallel_group(
    group_ranks,
    get_world_group().local_rank,
    backend,
    use_message_queue_broadcaster=True,
    group_name="tp",
)
```

**示例**：8个GPU，TP=4
- TP组0: [0, 1, 2, 3]
- TP组1: [4, 5, 6, 7]

### 2.2 MoE Expert Parallel组初始化

```python
moe_ep_size = expert_model_parallel_size
moe_tp_size = tensor_model_parallel_size // moe_ep_size

# 构建MoE专家并行组 (MOE_EP)
if moe_ep_size == tensor_model_parallel_size:
    _MOE_EP = _TP
else:
    group_ranks = []
    for i in range(num_tensor_model_parallel_groups):
        for j in range(moe_tp_size):
            st = i * tensor_model_parallel_size + j
            en = (i + 1) * tensor_model_parallel_size + j
            ranks = list(range(st, en, moe_tp_size))
            group_ranks.append(ranks)
    _MOE_EP = init_model_parallel_group(
        group_ranks,
        get_world_group().local_rank,
        backend,
        group_name="moe_ep",
    )
```

**示例**：`TP=4, EP=2, world=8`

先算出：
- `num_tensor_model_parallel_groups = world_size // TP = 8 // 4 = 2`
- `moe_tp_size = TP // EP = 4 // 2 = 2`

按源码中的双层循环展开：

- `i=0, j=0`：`st=0*4+0=0`，`en=(0+1)*4+0=4`，`ranks=range(0,4,2)=[0,2]`
- `i=0, j=1`：`st=0*4+1=1`，`en=(0+1)*4+1=5`，`ranks=range(1,5,2)=[1,3]`
- `i=1, j=0`：`st=1*4+0=4`，`en=(1+1)*4+0=8`，`ranks=range(4,8,2)=[4,6]`
- `i=1, j=1`：`st=1*4+1=5`，`en=(1+1)*4+1=9`，`ranks=range(5,9,2)=[5,7]`

因此会得到 **4 个 EP 组（每组 2 个 rank）**：
- EP组0: `[0, 2]`
- EP组1: `[1, 3]`
- EP组2: `[4, 6]`
- EP组3: `[5, 7]`

### 2.3 MoE Tensor Parallel组初始化

```python
# 构建MoE张量并行组 (MOE_TP)
if moe_tp_size == tensor_model_parallel_size:
    _MOE_TP = _TP
else:
    group_ranks = []
    for i in range(num_tensor_model_parallel_groups):
        for j in range(moe_ep_size):
            st = i * tensor_model_parallel_size + j * moe_tp_size
            en = i * tensor_model_parallel_size + (j + 1) * moe_tp_size
            ranks = list(range(st, en))
            group_ranks.append(ranks)
    _MOE_TP = init_model_parallel_group(
        group_ranks,
        get_world_group().local_rank,
        backend,
        group_name="moe_tp",
    )
```

**示例**：8个GPU，TP=4，EP=2
- MoE_TP组0: [0, 1] (EP组0内的TP)
- MoE_TP组1: [2, 3] (EP组0内的TP)
- MoE_TP组2: [4, 5] (EP组1内的TP)
- MoE_TP组3: [6, 7] (EP组1内的TP)

### 2.4 Pipeline Parallel组初始化

```python
# 构建pipeline model-parallel组
num_pipeline_model_parallel_groups = world_size // pipeline_model_parallel_size
group_ranks = []
for i in range(num_pipeline_model_parallel_groups):
    ranks = list(range(i, world_size, num_pipeline_model_parallel_groups))
    group_ranks.append(ranks)

_PP = init_model_parallel_group(
    group_ranks,
    get_world_group().local_rank,
    backend,
    use_custom_allreduce=False,
    group_name="pp",
)
```

**示例**：8个GPU，PP=2
- PP组0: [0, 2, 4, 6]
- PP组1: [1, 3, 5, 7]

---

## 三、DeepEP通信流程

DeepEP是专门为MoE专家并行优化的通信库，支持两种模式。

### 3.1 DeepEP架构

在[deepep.py](../python/sglang/srt/layers/moe/token_dispatcher/deepep.py#L731)中定义了`DeepEPDispatcher`：

```python
class DeepEPDispatcher(BaseDispatcher):
    def __init__(self, deepep_mode=DeepEPMode.AUTO, ...):
        # 根据模式创建对应的分发器
        if self.deepep_mode.enable_low_latency():
            self._low_latency_dispatcher = _DeepEPDispatcherImplLowLatency(...)
        if self.deepep_mode.enable_normal():
            self._normal_dispatcher = _DeepEPDispatcherImplNormal(...)
```

### 3.2 DeepEP模式

| 模式 | 适用场景 | 特点 |
|------|---------|------|
| **Normal模式** | Prefill阶段，大批量tokens | 使用NVLink/RDMA优化，吞吐量优先 |
| **Low Latency模式** | Decode阶段，小批量tokens | 优化延迟，适合实时推理 |
| **AUTO模式** | 自动选择 | 根据batch size自动选择模式 |

### 3.3 Normal模式通信流程

**Dispatch A阶段**（[deepep.py#L380](../python/sglang/srt/layers/moe/token_dispatcher/deepep.py#L380)）：

```python
def dispatch_a(self, hidden_states, topk_output):
    topk_weights, topk_ids = topk_output.topk_weights, topk_output.topk_ids
    topk_ids = topk_ids.to(torch.int64)
    
    # 可选的FP8量化以减少通信量
    if ENABLE_JIT_DEEPGEMM:
        hidden_states = sglang_per_token_group_quant_fp8(hidden_states, 128, ...)
    
    # 捕获事件用于异步通信
    previous_event = Buffer.capture() if self.async_finish else None
    return hidden_states, topk_ids, topk_weights, previous_event
```

**Dispatch B阶段**（[deepep.py#L403](../python/sglang/srt/layers/moe/token_dispatcher/deepep.py#L403)）：

```python
def dispatch_b(self, hidden_states, topk_ids, topk_weights, previous_event):
    # 计算分发布局
    (num_tokens_per_rank, num_tokens_per_rdma_rank, 
     num_tokens_per_expert, is_token_in_rank, previous_event) = \
        buffer.get_dispatch_layout(topk_ids, self.num_experts, ...)
    
    # 执行DeepEP分发
    (recv_x, recv_topk_ids, recv_topk_weights, 
     num_recv_tokens_per_expert, self.handle, event) = \
        buffer.dispatch(
            x, topk_idx=topk_ids, topk_weights=topk_weights,
            num_tokens_per_rank=num_tokens_per_ranks,
            num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
            is_token_in_rank=is_token_in_rank,
            num_tokens_per_expert=num_tokens_per_expert,
            expert_alignment=128 if ENABLE_JIT_DEEPGEMM else 1,
            config=DeepEPConfig.get_instance().normal_dispatch_config,
        )
    
    return DeepEPNormalDispatchOutput(...)
```

**Combine A阶段**（[deepep.py#L487](../python/sglang/srt/layers/moe/token_dispatcher/deepep.py#L487)）：

```python
def combine_a(self, hidden_states, topk_ids, topk_weights):
    if ENABLE_JIT_DEEPGEMM or _use_aiter or _is_npu:
        output = hidden_states
    else:
        raise NotImplementedError()
    
    previous_event = Buffer.capture() if self.async_finish else None
    return output, previous_event
```

**Combine B阶段**（[deepep.py#L502](../python/sglang/srt/layers/moe/token_dispatcher/deepep.py#L502)）：

```python
def combine_b(self, output, previous_event):
    hidden_states, event = self._combine_core(output, previous_event)
    event.current_stream_wait() if self.async_finish else ()
    
    # DeepEP combine操作
    combined_x, _, event = buffer.combine(
        x, self.handle, async_finish=self.async_finish,
        previous_event=previous_event,
        allocate_on_comm_stream=previous_event is not None,
        config=DeepEPConfig.get_instance().normal_combine_config,
    )
    return combined_x
```

### 3.4 Low Latency模式通信流程

**Dispatch A阶段**（[deepep.py#L546](../python/sglang/srt/layers/moe/token_dispatcher/deepep.py#L546)）：

```python
def dispatch_a(self, hidden_states, topk_output):
    buffer = self._get_buffer()
    topk_weights, topk_ids = topk_output.topk_weights, topk_output.topk_ids
    topk_ids = topk_ids.to(torch.int64)
    
    # 计算期望的token数量
    expected_m = (
        hidden_states.shape[0] * buffer.group_size * topk_ids.shape[1]
        + self.num_experts
    ) // self.num_experts
    
    # Low latency dispatch
    hidden_states, masked_m, event, hook = self._dispatch_core(
        hidden_states, topk_ids
    )
    return (hidden_states, topk_ids, topk_weights, masked_m, expected_m, event, hook)
```

**Dispatch B阶段**（[deepep.py#L572](../python/sglang/srt/layers/moe/token_dispatcher/deepep.py#L572)）：

```python
def dispatch_b(self, hidden_states, topk_ids, topk_weights, masked_m, expected_m, event, hook):
    hook() if self.return_recv_hook else event.current_stream_wait()
    
    return DeepEPLLDispatchOutput(...)
```

**Combine阶段**（[deepep.py#L639](../python/sglang/srt/layers/moe/token_dispatcher/deepep.py#L639)）：

```python
def combine_a(self, hidden_states, topk_ids, topk_weights):
    hidden_states, event, hook = self._combine_core(
        hidden_states, topk_ids, topk_weights
    )
    return hidden_states, event, hook

def combine_b(self, hidden_states, event, hook):
    # 支持计算和通信的overlap
    overlap_args = self.overlap_args
    if overlap_args is not None:
        overlap_args.stream.wait_stream(self.device_module.current_stream())
    
    hook() if self.return_recv_hook else event.current_stream_wait()
    
    if overlap_args is not None:
        self.device_module.current_stream().wait_stream(overlap_args.stream)
    
    return hidden_states
```

---

## 四、完整的MoE推理流程

### 4.1 MiniMaxM2MoE的forward流程

在[MiniMaxM2MoE](../python/sglang/srt/models/minimax_m2.py#L319)中：

```python
class MiniMaxM2MoE(nn.Module):
    def forward_deepep(self, hidden_states, forward_batch):
        # 1. Gate计算路由logits
        router_logits, _ = self.gate(hidden_states.to(torch.float32))
        
        # 2. TopK选择专家
        topk_output = self.topk(
            hidden_states,
            router_logits,
            num_token_non_padded=forward_batch.num_token_non_padded,
            expert_location_dispatch_info=ExpertLocationDispatchInfo.init_new(
                layer_id=self.layer_id,
            ),
        )
        
        # 3. 专家计算
        final_hidden_states = self.experts(
            hidden_states=hidden_states,
            topk_output=topk_output,
        )
        return final_hidden_states
```

### 4.2 FusedMoE的forward流程

在[FusedMoE](../python/sglang/srt/layers/moe/fused_moe_triton/layer.py#L958)中：

```python
class FusedMoE(torch.nn.Module):
    def forward_impl(self, hidden_states, topk_output):
        # 1. Token分发到对应专家
        dispatch_output = self.dispatcher.dispatch(
            hidden_states=hidden_states, 
            topk_output=topk_output
        )
        
        # 2. 执行MoE核心计算
        combine_input = self.run_moe_core(dispatch_output=dispatch_output)
        
        # 3. 组合专家输出
        final_hidden_states = self.dispatcher.combine(combine_input=combine_input)
        
        # 4. 如果需要，进行all_reduce
        if self.reduce_results and (self.moe_tp_size > 1 or self.moe_ep_size > 1):
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)
        
        return final_hidden_states
```

### 4.3 MoE推理流程图

```
输入hidden_states
    ↓
[Gate] → router_logits
    ├─ ReplicatedLinear（无通信）
    └─ 本地计算
    ↓
[TopK] → topk_ids, topk_weights
    ├─ 本地计算
    └─ 输出：每个token选择的top-k专家
    ↓
[DeepEP Dispatcher]
    ├─ Dispatch A：准备+量化
    │   ├─ FP8量化（可选）
    │   └─ 捕获事件
    ↓
    ├─ Dispatch B：DeepEP分发
    │   ├─ 计算分发布局
    │   ├─ NVLink通信（同节点）
    │   ├─ RDMA通信（跨节点）
    │   └─ FP8量化传输
    ↓
[Expert Compute]
    ├─ GEMM1：hidden → intermediate
    ├─ Activation (SiLU/GeLU)
    └─ GEMM2：intermediate → output
    ↓
[DeepEP Combine]
    ├─ Combine A：准备
    ├─ Combine B：DeepEP组合
    │   ├─ 收集各专家输出
    │   └─ 加权求和
    ↓
[All Reduce] (如果moe_tp_size > 1 or moe_ep_size > 1)
    ├─ Custom AllReduce
    ├─ Quick AllReduce
    ├─ PyMSCCL++
    └─ 标准NCCL
    ↓
输出final_hidden_states
```

---

## 五、DeepEP之前的通信流程详解

### 5.1 Embedding层通信

在[VocabParallelEmbedding](../python/sglang/srt/layers/vocab_parallel_embedding.py#L161)中：

```python
class VocabParallelEmbedding(torch.nn.Module):
    """Embedding parallelized in vocabulary dimension."""
    
    def __init__(self, num_embeddings, embedding_dim, enable_tp=True, ...):
        if enable_tp:
            tp_rank = get_tensor_model_parallel_rank()
            self.tp_size = get_tensor_model_parallel_world_size()
        else:
            tp_rank = 0
            self.tp_size = 1
        
        # 将vocab按TP大小切分
        self.num_embeddings_per_partition = divide(
            self.num_embeddings_padded, self.tp_size
        )
```

**通信特点**：
- **无通信**：Embedding层本身不需要通信，每个rank只负责vocab的一部分
- **权重切分**：vocab_size被切分到不同TP rank
- **输入广播**：input_ids在所有rank上相同

### 5.2 Attention层通信

在[LlamaAttention](../python/sglang/srt/models/llama.py#L119)中：

```python
class LlamaAttention(nn.Module):
    def __init__(self, config, hidden_size, num_heads, num_kv_heads, ...):
        tp_size = get_tensor_model_parallel_world_size()
        
        # QKV投影：ColumnParallel
        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            ...
        )
        
        # 输出投影：RowParallel
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            ...
        )
```

**QKV投影通信**（[QKVParallelLinear](../python/sglang/srt/layers/linear.py#L786)）：

```python
class QKVParallelLinear(ColumnParallelLinear):
    def forward(self, input_):
        # 1. 本地GEMM计算
        output_parallel = self.quant_method.apply(self, input_, bias)
        
        # 2. All-gather收集所有TP rank的结果
        if self.gather_output:
            output = tensor_model_parallel_all_gather(output_parallel)
        else:
            output = output_parallel
        return output, output_bias
```

**输出投影通信**（[RowParallelLinear](../python/sglang/srt/layers/linear.py#L1412)）：

```python
class RowParallelLinear(LinearBase):
    def forward(self, input_, skip_all_reduce=False):
        # 1. 输入切分（如果需要）
        if self.input_is_parallel:
            input_parallel = input_
        else:
            splitted_input = split_tensor_along_last_dim(
                input_, num_partitions=self.tp_size
            )
            input_parallel = splitted_input[self.tp_rank]
        
        # 2. 本地GEMM计算
        output_parallel = self.quant_method.apply(self, input_parallel, bias=bias_)
        
        # 3. All-Reduce聚合结果
        if self.reduce_results and self.tp_size > 1 and not skip_all_reduce:
            output = tensor_model_parallel_all_reduce(output_parallel)
        else:
            output = output_parallel
        return output, output_bias
```

### 5.3 LayerNorm和残差连接

在[LlamaDecoderLayer](../python/sglang/srt/models/llama.py#L245)中：

```python
class LlamaDecoderLayer(nn.Module):
    def forward(self, positions, hidden_states, forward_batch, residual):
        # Self Attention前
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        
        # Attention计算（包含QKV和Output的TP通信）
        hidden_states = self.self_attn(positions, hidden_states, forward_batch)
        
        # MLP前
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        
        # MLP计算（包含TP通信）
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual
```

**通信特点**：
- **LayerNorm**：本地计算，无通信
- **残差连接**：本地加法，无通信
- **LayerCommunicator**：在MoE模型中用于优化通信模式

### 5.4 MLP层通信

在[LlamaMLP](../python/sglang/srt/models/llama.py#L65)中：

```python
class LlamaMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size, ...):
        # Gate+Up投影：ColumnParallel
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,  # gate和up融合
            bias=False,
            ...
        )
        
        # Down投影：RowParallel
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias-bias=False,
            reduce_results=True,  # 需要all-reduce
            ...
        )
    
    def forward(self, x, forward_batch=None, use_reduce_scatter=False):
        # 1. Gate+Up投影
        gate_up, _ = self.gate_up_proj(x)
        
        # 2. 激活函数
        x = self.act_fn(gate_up)
        
        # 3. Down投影（包含all-reduce）
        x, _ = self.down_proj(x, skip_all_reduce=use_reduce_scatter)
        return x
```

**MLP通信流程**：
```
hidden_states
    ↓
[Gate+Up GEMM] → gate_up_parallel (每个rank计算部分)
    ↓
[All-Gather] → gate_up_full (收集所有rank)
    ↓
[SiLU激活] → intermediate
    ↓
[Down GEMM] → output_parallel (每个rank计算部分)
    ↓
[All-Reduce] → output (聚合所有rank)
```

### 5.5 LayerCommunicator优化通信

在MoE模型中，使用[LayerCommunicator](../python/sglang/srt/layers/communicator.py#L336)来优化通信：

```python
class LayerCommunicator:
    def prepare_attn(self, hidden_states, residual, forward_batch):
        # 1. TP Reduce-Scatter（如果启用）
        if get_attn_tp_context().input_scattered:
            hidden_states, residual = self._tp_reduce_scatter(
                hidden_states, residual
            )
        
        # 2. LayerNorm（本地计算）
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        
        # 3. 简单通信
        hidden_states = self._communicate_simple_fn(
            hidden_states=hidden_states,
            forward_batch=forward_batch,
            context=self._context,
        )
        return hidden_states, residual
    
    def prepare_mlp(self, hidden_states, residual, forward_batch):
        # 1. All-Reduce + LayerNorm融合
        return self._communicate_with_all_reduce_and_layer_norm_fn(
            hidden_states=hidden_states,
            residual=residual,
            forward_batch=forward_batch,
            layernorm=self.post_attention_layernorm,
            context=self._context,
        )
    
    def postprocess_layer(self, hidden_states, residual, forward_batch):
        # 1. 可选的Reduce-Scatter
        return self._communicate_summable_tensor_pair_fn(
            hidden_states=hidden_states,
            residual=residual,
            forward_batch=forward_batch,
            context=self._context,
            allow_reduce_scatter=self.allow_reduce_scatter,
        )
```

**Reduce-Scatter优化**（[communicator.py#L515](../python/sglang/srt/layers/communicator.py#L515)）：

```python
def _tp_reduce_scatter(self, hidden_states, residual):
    # Reduce-Scatter：All-Reduce + Scatter的组合
    # 比单独的All-Reduce + Scatter更高效
    local_tokens = hidden_states.shape[0] // self._context.tp_size
    output = hidden_states.new_empty(local_tokens, *hidden_states.shape[1:])
    get_tp_group().reduce_scatter_tensor(output, hidden_states)
    
    if residual is not None:
        residual = residual.tensor_split(self._context.tp_size)[
            self._context.tp_rank
        ]
    return output, residual
```

---

## 六、完整通信流程图

```
输入tokens (所有rank相同)
    ↓
[VocabParallelEmbedding]
    ├─ 无通信
    └─ 权重切分：vocab_size / tp_size
    ↓
hidden_states
    ↓
┌─────────────────────────────────────────────────────────────┐
│              Transformer Layer N                          │
└─────────────────────────────────────────────────────────────┘
    ↓
[Input LayerNorm]
    ├─ 无通信
    └─ 本地RMSNorm
    ↓
[Attention]
    ↓
    ├─ [QKVParallelLinear]
    │   ├─ GEMM计算（本地）
    │   ├─ All-Gather（TP通信）
    │   └─ 输出：qkv_full
    ↓
    ├─ [RoPE]（旋转位置编码，本地）
    ↓
    ├─ [RadixAttention]
    │   ├─ KV Cache读写
    │   ├─ FlashAttention
    │   └─ 输出：attn_output
    ↓
    ├─ [RowParallelLinear - o_proj]
    │   ├─ GEMM计算（本地）
    │   ├─ All-Reduce（TP通信）
    │   └─ 输出：hidden_states
    ↓
[Post-Attention LayerNorm]
    ├─ 无通信
    └─ 本地RMSNorm
    ↓
[MoE/MLP]
    ↓
    ├─ [ReplicatedLinear - gate]
    │   ├─ 无通信
    │   └─ 输出：router_logits
    ↓
    ├─ [TopK]
    │   ├─ 本地计算
    │   └─ 输出：topk_ids, topk_weights
    ↓
    ├─ [DeepEP Dispatcher]
    │   ├─ Dispatch A：准备+量化
    │   ├─ Dispatch B：DeepEP分发
    │   │   ├─ NVLink通信（同节点）
    │   │   ├─ RDMA通信（跨节点）
    │   │   └─ FP8量化传输
    │   └─ 输出：dispatched_tokens
    ↓
    ├─ [Expert Compute]
    │   ├─ GEMM1：hidden → intermediate
    │   ├─ Activation
    │   └─ GEMM2：intermediate → output
    ↓
    ├─ [DeepEP Combine]
    │   ├─ Combine A：准备
    │   ├─ Combine B：DeepEP组合
    │   │   ├─ 收集各专家输出
    │   │   └─ 加权求和
    │   └─ 输出：hidden_states
    ↓
    ├─ [All-Reduce]（如果moe_tp_size > 1 or moe_ep_size > 1）
    │   ├─ Custom AllReduce
    │   ├─ Quick AllReduce
    │   ├─ PyMSCCL++
    │   └─ 标准NCCL
    ↓
hidden_states
    ↓
[LogitsProcessor]
    ↓
输出logits
```

---

## 七、通信优化技术总结

### 7.1 TP通信优化

| 优化技术 | 应用场景 | 文件位置 | 说明 |
|---------|---------|----------|------|
| All-Gather | ColumnParallelLinear输出收集 | [linear.py#L454](../python/sglang/srt/layers/linear.py#L454) | 收集所有TP rank的部分结果 |
| All-Reduce | RowParallelLinear结果聚合 | [linear.py#L1435](../python/sglang/srt/layers/linear.py#L1435) | 聚合所有TP rank的部分结果 |
| Reduce-Scatter | LayerCommunicator优化 | [communicator.py#L527](../python/sglang/srt/layers/communicator.py#L527) | All-Reduce + Scatter的组合优化 |
| Custom AllReduce | 大tensor优化 | [parallel_state.py#L595](../python/sglang/srt/distributed/parallel_state.py#L595) | 自定义的all-reduce实现 |
| Quick AllReduce | 小tensor优化 | [parallel_state.py#L599](../python/sglang/srt/distributed/parallel_state.py#L599) | 针对小tensor优化的all-reduce |
| PyMSCCL++ | MSCCL++优化 | [parallel_state.py#L606](../python/sglang/srt/distributed/parallel_state.py#L606) | 使用PyMSCCL++库 |
| Torch Symmetric Memory | 对称内存优化 | [parallel_state.py#L612](../python/sglang/srt/distributed/parallel_state.py#L612) | 使用对称内存减少拷贝 |

### 7.2 DeepEP通信优化

| 优化技术 | 应用场景 | 文件位置 | 说明 |
|---------|---------|----------|------|
| NVLink | 同节点通信 | [deepep.py#L458](../python/sglang/srt/layers/moe/token_dispatcher/deepep.py#L458) | 使用NVLink进行高速通信 |
| RDMA | 跨节点通信 | [deepep.py#L458](../python/sglang/srt/layers/moe/token_dispatcher/deepep.py#L458) | 使用RDMA进行跨节点通信 |
| FP8量化 | 减少通信带宽 | [deepep.py#L393](../python/sglang/srt/layers/moe/token_dispatcher/deepep.py#L393) | 使用FP8量化减少数据传输量 |
| 异步通信 | 计算通信重叠 | [deepep.py#L400](../python/sglang/srt/layers/moe/token_dispatcher/deepep.py#L400) | 使用异步事件实现计算和通信重叠 |
| Normal模式 | Prefill大批量 | [deepep.py#L372](../python/sglang/srt/layers/moe/token_dispatcher/deepep.py#L372) | 优化吞吐量的模式 |
| Low Latency模式 | Decode小批量 | [deepep.py#L534](../python/sglang/srt/layers/moe/token_dispatcher/deepep.py#L534) | 优化延迟的模式 |

### 7.3 通信优化策略

**1. 根据tensor大小选择通信原语**：

```python
def all_reduce(self, input_):
    # 单GPU优化
    if self.world_size == 1:
        return input_
    
    # 根据tensor大小和配置选择最优实现
    if self.ca_comm and self.ca_comm.should_custom_ar(input_):
        return outplace_all_reduce(input_, method="ca")
    elif self.qr_comm and self.qr_comm.should_quick_allreduce(input_):
        return outplace_all_reduce(input_, method="qr")
    elif self.pymscclpp_comm and self.pymscclpp_comm.should_mscclpp_allreduce(input_):
        return outplace_all_reduce(input_, method="pymscclpp")
    elif self.torch_symm_mem_comm and self.torch_symm_mem_comm.should_torch_symm_mem_allreduce(input_):
        return outplace_all_reduce(input_, method="torch_symm_mem")
    else:
        inplace_all_reduce(input_, group_name=self.unique_name)
        return input_
```

**2. DeepEP模式自动选择**：

```python
def _get_impl(self):
    is_extend_in_batch = get_is_extend_in_batch()
    resolved_deepep_mode = self.deepep_mode.resolve(is_extend_in_batch)
    
    if resolved_deepep_mode == DeepEPMode.NORMAL:
        return self._normal_dispatcher  # Prefill阶段
    elif resolved_deepep_mode == DeepEPMode.LOW_LATENCY:
        return self._low_latency_dispatcher  # Decode阶段
```

---

## 八、关键通信原语实现

### 8.1 All-Gather

在[parallel_state.py](../python/sglang/srt/distributed/parallel_state.py#L764)中：

```python
def all_gather_into_tensor(self, output: torch.Tensor, input: torch.Tensor):
    pynccl_comm = self.pynccl_comm
    if pynccl_comm is not None and (
        not pynccl_comm.disabled or self.is_symmetric_memory_enabled()
    ):
        with pynccl_comm.change_state(
            enable=True, stream=get_current_device_stream_fast()
        ):
            pynccl_comm.all_gather_into_tensor(output, input)
    else:
        torch.distributed.all_gather_into_tensor(
            output, input, group=self.device_group
        )
```

### 8.2 All-Reduce

在[parallel_state.py](../python/sglang/srt/distributed/parallel_state.py#L527)中：

```python
def all_reduce(self, input_: torch.Tensor):
    # 单GPU优化
    if self.world_size == 1:
        return input_
    
    # 选择最优的all-reduce实现
    if self.ca_comm and self.ca_comm.should_custom_ar(input_):
        return outplace_all_reduce(input_, method="ca")
    elif self.qr_comm and self.qr_comm.should_quick_allreduce(input_):
        return outplace_all_reduce(input_, method="qr")
    elif self.pymscclpp_comm and self.pymscclpp_comm.should_mscclpp_allreduce(input_):
        return outplace_all_reduce(input_, method="pymscclpp")
    elif self.torch_symm_mem_comm and self.torch_symm_mem_comm.should_torch_symm_mem_allreduce(input_):
        return outplace_all_reduce(input_, method="torch_symm_mem")
    else:
        inplace_all_reduce(input_, group_name=self.unique_name)
        return input_
```

### 8.3 Reduce-Scatter

在[parallel_state.py](../python/sglang/srt/distributed/parallel_state.py#L657)中：

```python
def _reduce_scatter_tensor(self, output: torch.Tensor, input: torch.Tensor):
    pynccl_comm = self.pynccl_comm
    if pynccl_comm is not None and (
        not pynccl_comm.disabled or self.is_symmetric_memory_enabled()
    ):
        with pynccl_comm.change_state(
            enable=True, stream=get_current_device_stream_fast()
        ):
            pynccl_comm.reduce_scatter(output, input)
    else:
        torch.distributed.reduce_scatter_tensor(
            output, input, group=self.device_group
        )
    return output
```

### 8.4 DeepEP Buffer

在[deepep.py](../python/sglang/srt/layers/moe/token_dispatcher/deepep.py#L136)中：

```python
class DeepEPBuffer:
    _buffer = None
    _dispatch_mode: Optional[DeepEPDispatchMode] = None
    _hidden_size: Optional[int] = None
    _num_max_dispatch_tokens_per_rank: Optional[int] = None
    _num_experts: Optional[int] = None
    
    @classmethod
    def get_deepep_buffer(
        cls,
        group: dist.ProcessGroup,
        hidden_size: int,
        param_bytes: int,
        deepep_mode: DeepEPMode,
        num_max_dispatch_tokens_per_rank: int = -1,
        num_experts: int = -1,
    ):
        if cls._buffer is not None:
            return cls._buffer
        
        # 计算buffer大小
        num_nvl_bytes, num_rdma_bytes = 0, 0
        if deepep_mode.enable_normal():
            hidden_bytes = hidden_size * param_bytes
            for config in (
                DeepEPConfig.get_instance().normal_dispatch_config
                or Buffer.get_dispatch_config(group.size()),
                DeepEPConfig.get_instance().normal_combine_config
                or Buffer.get_combine_config(group.size()),
            ):
                num_nvl_bytes = max(
                    config.get_nvl_buffer_size_hint(hidden_bytes, group.size()),
                    num_nvl_bytes,
                )
                num_rdma_bytes = max(
                    config.get_rdma_buffer_size_hint(hidden_bytes, group.size()),
                    num_rdma_bytes,
                )
        
        # 创建DeepEP buffer
        cls._buffer = Buffer(
            group,
            num_nvl_bytes,
            num_rdma_bytes,
            low_latency_mode=deepep_mode == DeepEPMode.LOW_LATENCY,
            num_qps_per_rank=num_qps_per_rank,
        )
        
        return cls._buffer
```

---

## 九、总结

SGLang的MoE DeepEP并行策略和通信流程具有以下特点：

### 9.1 多层次并行

- **TP (Tensor Parallelism)**：权重切分，适用于Attention和MLP层
- **EP (Expert Parallelism)**：专家切分，专门优化MoE层
- **PP (Pipeline Parallelism)**：层切分，适用于大模型
- **DP (Data Parallelism)**：数据切分，处理多请求

### 9.2 高效通信优化

- **多种通信原语**：All-Gather、All-Reduce、Reduce-Scatter等
- **多种通信后端**：Custom AllReduce、Quick AllReduce、PyMSCCL++、Torch Symmetric Memory
- **DeepEP优化**：NVLink、RDMA、FP8量化、异步通信
- **模式自适应**：根据batch size自动选择Normal或Low Latency模式

### 9.3 计算通信重叠

- **异步事件**：使用Buffer.capture()和事件同步
- **流水线重叠**：Dispatch和Expert计算重叠
- **Reduce-Scatter**：All-Reduce和Scatter的组合优化

### 9.4 内存优化

- **对称内存**：减少内存拷贝
- **In-place操作**：减少内存分配
- **FP8量化**：减少通信带宽

这些优化使得SGLang能够高效地运行大规模MoE模型，在多GPU环境下实现高吞吐量和低延迟的推理性能。
