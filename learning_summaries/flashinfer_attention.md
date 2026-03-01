# FlashInfer Attention 后端深度解析

**分析时间**: 2026-03-02 02:00  
**代码路径**: `python/sglang/srt/layers/attention/flashinfer_backend.py` (1,663 行)  
**优先级**: P1 - 关键性能模块

---

## 一、模块定位

### 1.1 核心职责

FlashInfer Attention 是 SGLang 的**高性能注意力后端**，负责：
- Prefill 阶段注意力计算
- Decode 阶段注意力计算
- Paged KV Cache 支持
- 多后端支持 (FlashInfer/Triton/FA3)

### 1.2 架构位置

```
Model Runner
    │
    │ ForwardBatch
    ▼
RadixAttention.forward()
    │
    ├─→ FlashInferAttnBackend ⭐ ← 本模块
    │   ├─ forward_extend()  (Prefill)
    │   └─ forward_decode()  (Decode)
    │
    ├─→ TritonAttnBackend (备选)
    └─→ TorchNativeBackend (回退)
```

---

## 二、核心类结构

### 2.1 FlashInferAttnBackend

```python
class FlashInferAttnBackend(AttentionBackend):
    def __init__(self, model_runner, skip_prefill=False, ...):
        # 后端配置
        self.prefill_backend = "fa2"      # Flash Attention 2
        self.decode_backend = "fa2"
        
        # 注意力头配置
        self.num_attention_heads = model_config.num_attention_heads
        self.num_kv_heads = model_config.get_num_kv_heads()
        self.head_dim = model_config.head_dim
        
        # FlashInfer 包装器
        self.prefill_wrappers: List[BatchPrefillWithPagedKVCacheWrapper]
        self.decode_wrappers: List[BatchDecodeWithPagedKVCacheWrapper]
        
        # 工作空间缓冲区 (全局共享)
        global_workspace_buffer: torch.Tensor
```

### 2.2 包装器分发

```python
class WrapperDispatch(Enum):
    SLIDING_WINDOW = auto()      # 滑动窗口注意力
    CROSS_ATTENTION = auto()     # 交叉注意力

# 根据模型类型决定包装器数量
if sliding_window:
    self.num_wrappers = 2  # 普通 + 滑动窗口
elif cross_attention:
    self.num_wrappers = 2  # self + cross
else:
    self.num_wrappers = 1
```

---

## 三、Prefill 阶段实现

### 3.1 forward_extend

```python
def forward_extend(
    self,
    q: torch.Tensor,          # Query: [batch_size, num_q_heads, head_dim]
    kv_cache: torch.Tensor,   # KV Cache
    attn_metadata: PrefillMetadata,
) -> torch.Tensor:
    # 1. 构建 Paged KV Cache 索引
    kv_indptr = attn_metadata.kv_indptr
    kv_indices = attn_metadata.kv_indices
    kv_last_page_len = attn_metadata.kv_last_page_len
    
    # 2. 计划 (Plan) - 准备 FlashInfer 所需数据结构
    wrapper = self.prefill_wrappers[0]
    wrapper.plan(
        qo_indptr=qo_indptr,      # Q/O 序列长度指针
        kv_indptr=kv_indptr,      # KV 序列长度指针
        kv_indices=kv_indices,    # KV 索引
        kv_last_page_len=kv_last_page_len,
        num_qo_heads=self.num_qo_heads,
        num_kv_heads=self.num_kv_heads,
        head_dim=self.head_dim,
    )
    
    # 3. 执行注意力计算
    o = wrapper.run(
        q=q,
        kv_cache=kv_cache,
        sm_scale=self.sm_scale,
        logits_soft_cap=self.logits_soft_cap,
    )
    
    return o
```

### 3.2 Ragged vs Paged

**Ragged 模式** (连续内存):
```python
# 适用于无缓存情况
wrapper = BatchPrefillWithRaggedKVCacheWrapper()
```

**Paged 模式** (分页内存):
```python
# 适用于有缓存复用
wrapper = BatchPrefillWithPagedKVCacheWrapper()
```

---

## 四、Decode 阶段实现

### 4.1 forward_decode

```python
def forward_decode(
    self,
    q: torch.Tensor,          # Query: [batch_size, num_q_heads, head_dim]
    kv_cache: torch.Tensor,   # KV Cache
    attn_metadata: DecodeMetadata,
) -> torch.Tensor:
    # 1. 使用 Tensor Cores (如果支持)
    use_tensor_cores = self.decode_use_tensor_cores
    
    # 2. 计划
    wrapper = self.decode_wrappers[0]
    wrapper.plan(
        kv_indptr=kv_indptr,
        kv_indices=kv_indices,
        kv_last_page_len=kv_last_page_len,
        num_qo_heads=self.num_qo_heads,
        num_kv_heads=self.num_kv_heads,
        head_dim=self.head_dim,
        use_tensor_cores=use_tensor_cores,
    )
    
    # 3. 执行
    o = wrapper.run(
        q=q,
        kv_cache=kv_cache,
    )
    
    return o
```

### 4.2 Tensor Core 优化

```python
def should_use_tensor_core(kv_cache_dtype, num_q_heads, num_kv_heads):
    # FP8 量化 → 使用 Tensor Cores
    if kv_cache_dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
        return True
    
    # MLA 架构 → 使用 Tensor Cores
    if num_q_heads != num_kv_heads:
        return True
    
    return False
```

---

## 五、性能优化技术

### 5.1 工作空间共享

```python
# 全局共享工作空间缓冲区
global_workspace_buffer = None

def init_workspace():
    global global_workspace_buffer
    if global_workspace_buffer is None:
        global_workspace_buffer = torch.empty(
            workspace_size,
            dtype=torch.uint8,
            device='cuda',
        )
    
    # 所有包装器共享同一缓冲区
    wrapper.set_workspace_buffer(global_workspace_buffer)
```

**收益**: 减少重复分配，提高内存利用率

### 5.2 Indptr 优化

```python
# 避免 Host-to-Device 拷贝
global_override_indptr_cpu = None

def plan(..., kv_indptr, ...):
    # 使用 CPU 缓存的 indptr
    if global_override_indptr_cpu is not None:
        kv_indptr = global_override_indptr_cpu
```

**收益**: 减少 PCIe 传输延迟

### 5.3 确定性模式

```python
if enable_deterministic:
    # 使用 Tensor Cores 确保确定性
    self.decode_use_tensor_cores = True
    
    # 设置 split tile sizes
    self.prefill_split_tile_size = 4096
    self.decode_split_tile_size = 2048
    
    # 禁用 CUDA Graph KV split
    self.disable_cuda_graph_kv_split = True
```

---

## 六、多后端支持

### 6.1 后端注册

```python
ATTENTION_BACKENDS = [
    "flashinfer",    # FlashInfer (默认，最快)
    "triton",        # Triton (易定制)
    "fa3",           # Flash Attention 3
    "fa4",           # Flash Attention 4
    "flashmla",      # Flash MLA
    "cutlass_mla",   # CUTLASS MLA
    "trtllm_mla",    # TRT-LLM MLA
    "ascend",        # Ascend NPU
    "nsa",           # NSA
]
```

### 6.2 后端选择

```python
def get_attention_backend(backend_name: str):
    if backend_name == "flashinfer":
        return FlashInferAttnBackend
    elif backend_name == "triton":
        return TritonAttnBackend
    elif backend_name == "fa3":
        return FlashAttention3Backend
    # ...
```

---

## 七、MLA 支持

### 7.1 MLA 架构

```
Multi-Head Latent Attention (MLA)
- 低秩 KV 投影
- 减少显存占用
- 适合长序列
```

### 7.2 FlashInfer MLA

```python
class FlashInferMLABackend(AttentionBackend):
    def forward(self, q, kv_cache, ...):
        # MLA 特殊处理
        # - 低秩投影
        # - 压缩 KV
        # - 特殊注意力模式
```

---

## 八、与现有文档关联

### 8.1 补充 model_runner.md

| 现有内容 | 补充内容 |
|---------|---------|
| 后端列表 | FlashInfer 详细实现 |
| CUDA Graph | FlashInfer 工作空间管理 |
| 量化支持 | FP8 + Tensor Core 协同 |

### 8.2 架构图补充

```
FlashInferAttnBackend
    │
    ├─ Prefill 路径
    │   ├─ Ragged 模式 (无缓存)
    │   └─ Paged 模式 (有缓存) ⭐
    │
    ├─ Decode 路径
    │   ├─ Tensor Core 模式
    │   └─ CUDA Core 模式
    │
    └─ MLA 路径
        ├─ 低秩投影
        └─ 压缩注意力
```

---

## 九、性能对比

| 后端 | Prefill | Decode | 特点 |
|------|---------|--------|------|
| FlashInfer | ⭐⭐⭐ | ⭐⭐⭐ | 最快，功能全 |
| Triton | ⭐⭐ | ⭐⭐ | 易定制 |
| FA3 | ⭐⭐⭐ | ⭐⭐ | 新，待优化 |
| Torch Native | ⭐ | ⭐ | 回退方案 |

---

## 十、总结

**FlashInfer Attention 核心价值**:
1. ✅ 高性能 Prefill/Decode
2. ✅ Paged KV Cache 支持
3. ✅ Tensor Core 优化
4. ✅ 多后端统一接口
5. ✅ MLA 架构支持

---

**分析用时**: 25 分钟  
**代码行数**: 1,663 行  
**产出文档**: flashinfer_attention.md
