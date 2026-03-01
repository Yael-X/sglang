# ScheduleBatch 批次管理深度解析

**分析时间**: 2026-03-02 02:30  
**代码路径**: `python/sglang/srt/managers/schedule_batch.py` (2,426 行)  
**优先级**: P1 - 关键调度模块

---

## 一、模块定位

### 1.1 核心职责

ScheduleBatch 是 Scheduler 与 ModelRunner 之间的**数据桥梁**，负责：
- 请求状态管理 (Req)
- 批次构建与管理 (ScheduleBatch → ModelWorkerBatch → ForwardBatch)
- 采样参数管理
- 完成原因追踪

### 1.2 数据流

```
Scheduler (CPU)
    │
    │ Req + 调度决策
    ▼
ScheduleBatch (CPU) ← 本模块
    │
    │ 转换为 GPU 数据
    ▼
ModelWorkerBatch (CPU→GPU)
    │
    │ 张量数据
    ▼
ForwardBatch (GPU) → ModelRunner
```

---

## 二、核心数据结构

### 2.1 Req (请求)

```python
@dataclass
class Req:
    # 基本信息
    rid: str                          # 请求 ID
    origin_input_text: str            # 原始输入文本
    origin_input_ids: List[int]       # 原始 token IDs
    
    # 状态追踪
    output_ids: List[int]             # 已生成 tokens
    finished: bool                    # 是否完成
    finish_reason: BaseFinishReason   # 完成原因
    
    # KV Cache 管理
    req_pool_idx: int                 # 请求池索引
    prefix_indices: torch.Tensor      # 前缀 KV 索引
    cache_protected_len: int          # 受保护的 KV 长度
    last_node: TreeNode               # RadixCache 节点
    
    # 采样参数
    sampling_params: SamplingParams   # 采样配置
    stream: bool                      # 是否流式输出
    
    # 多模态支持
    mm_inputs: Optional[MultimodalInputs]  # 多模态输入
    mm_hash: Optional[int]                 # 多模态 hash
```

### 2.2 ScheduleBatch

```python
@dataclass
class ScheduleBatch:
    # 请求列表
    reqs: List[Req]                   # 当前批次的所有请求
    
    # 批次信息
    batch_size: int                   # 批次大小
    forward_mode: ForwardMode         # 前向模式 (prefill/decode)
    
    # Token 数据
    input_ids: torch.Tensor           # 输入 tokens
    positions: torch.Tensor           # 位置信息
    seq_lens: List[int]               # 序列长度
    prefix_lens: List[int]            # 前缀长度
    
    # KV Cache
    req_pool_indices: torch.Tensor    # 请求池索引
    kv_indices: torch.Tensor          # KV Cache 索引
    
    # 采样信息
    sampling_info: SamplingBatchInfo  # 批量采样信息
    
    # 特殊功能
    has_grammar: bool                 # 是否有语法约束
    is_spec_v2: bool                  # 是否投机采样 v2
```

### 2.3 ModelWorkerBatch

```python
@dataclass
class ModelWorkerBatch:
    # ScheduleBatch 的 GPU 子集
    input_ids: torch.Tensor           # 输入 tokens (GPU)
    positions: torch.Tensor           # 位置 (GPU)
    req_pool_indices: torch.Tensor    # 请求池索引 (GPU)
    seq_lens: List[int]               # 序列长度
    kv_indices: torch.Tensor          # KV 索引 (GPU)
    
    # 转发给 ForwardBatch
    def to_forward_batch(self) -> ForwardBatch:
        return ForwardBatch(...)
```

---

## 三、批次构建流程

### 3.1 从 Scheduler 到 ScheduleBatch

```python
# Scheduler 中构建批次
def get_next_batch_to_run(self):
    # 1. 从 waiting_queue 选择请求
    selected_reqs = self.schedule_policy.select(self.waiting_queue)
    
    # 2. 构建 ScheduleBatch
    batch = ScheduleBatch(
        reqs=selected_reqs,
        batch_size=len(selected_reqs),
        forward_mode=self.determine_forward_mode(selected_reqs),
    )
    
    # 3. 填充批次数据
    batch.prepare_for_extend()  # Prefill 模式
    # 或
    batch.prepare_for_decode()  # Decode 模式
    
    return batch
```

### 3.2 Prefill 模式准备

```python
def prepare_for_extend(self):
    self.forward_mode = ForwardMode.EXTEND
    
    # 拼接所有请求的 input_ids
    self.input_ids = torch.cat([
        torch.tensor(r.origin_input_ids, dtype=torch.int32)
        for r in self.reqs
    ])
    
    # 计算位置信息
    self.positions = torch.cat([
        torch.arange(len(r.prefix_indices), len(r.origin_input_ids))
        for r in self.reqs
    ])
    
    # 序列长度和前缀长度
    self.seq_lens = [len(r.origin_input_ids) for r in self.reqs]
    self.prefix_lens = [len(r.prefix_indices) for r in self.reqs]
```

### 3.3 Decode 模式准备

```python
def prepare_for_decode(self):
    self.forward_mode = ForwardMode.DECODE
    
    # 只取最后一个 token
    self.input_ids = torch.tensor([
        r.output_ids[-1] if r.output_ids else r.origin_input_ids[-1]
        for r in self.reqs
    ], dtype=torch.int32)
    
    # 位置是当前序列长度
    self.positions = torch.tensor([
        len(r.origin_input_ids) + len(r.output_ids) - 1
        for r in self.reqs
    ])
    
    self.seq_lens = [
        len(r.origin_input_ids) + len(r.output_ids)
        for r in self.reqs
    ]
```

---

## 四、完成原因管理

### 4.1 完成原因类型

```python
class BaseFinishReason:
    def __init__(self, is_error: bool = False):
        self.is_error = is_error

class FINISH_MATCHED_TOKEN(BaseFinishReason):
    """遇到 stop token"""
    def __init__(self, matched: Union[int, List[int]]):
        self.matched = matched

class FINISH_MATCHED_STR(BaseFinishReason):
    """遇到 stop string"""
    def __init__(self, matched: str):
        self.matched = matched

class FINISH_LENGTH(BaseFinishReason):
    """达到最大长度"""
    def __init__(self, length: int):
        self.length = length

class FINISH_ABORT(BaseFinishReason):
    """用户中止"""
    def __init__(self):
        super().__init__(is_error=True)
```

### 4.2 完成检查

```python
def check_finished(self, req: Req, new_token_id: int) -> Optional[BaseFinishReason]:
    # 1. 检查 stop tokens
    if new_token_id in req.sampling_params.stop_token_ids:
        return FINISH_MATCHED_TOKEN(new_token_id)
    
    # 2. 检查最大长度
    if len(req.output_ids) >= req.sampling_params.max_new_tokens:
        return FINISH_LENGTH(req.sampling_params.max_new_tokens)
    
    # 3. 检查 stop strings
    decoded_text = self.tokenizer.decode(req.output_ids)
    for stop_str in req.sampling_params.stop_strings:
        if stop_str in decoded_text:
            return FINISH_MATCHED_STR(stop_str)
    
    return None
```

---

## 五、采样参数管理

### 5.1 SamplingParams

```python
@dataclass
class SamplingParams:
    # 基础参数
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    
    # 约束
    max_new_tokens: int = 128
    stop_token_ids: List[int] = None
    stop_strings: List[str] = None
    
    # 高级功能
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    repetition_penalty: float = 1.0
    
    # 语法约束
    grammar: Optional[str] = None  # JSON Schema / Regex
```

### 5.2 批量采样信息

```python
@dataclass
class SamplingBatchInfo:
    # 温度采样
    temperatures: torch.Tensor      # [batch_size]
    top_ps: torch.Tensor            # [batch_size]
    top_ks: torch.Tensor            # [batch_size]
    
    # 惩罚
    presence_penalties: torch.Tensor
    frequency_penalties: torch.Tensor
    
    # 采样器
    sampler: Sampler
    
    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        # 应用温度
        logits = logits / self.temperatures.unsqueeze(1)
        
        # 应用惩罚
        logits = self.apply_penalty(logits)
        
        # Top-K / Top-P 采样
        return self.sampler(logits, self.top_ps, self.top_ks)
```

---

## 六、多模态支持

### 6.1 多模态输入

```python
@dataclass
class MultimodalInputs:
    modalities: List[str]           # 模态类型 (image/video/audio)
    mm_data: Dict[str, torch.Tensor]  # 多模态数据
    mm_hashes: List[int]            # 多模态 hash
    mm_positions: Dict[str, List[Tuple[int, int]]]  # 位置信息
```

### 6.2 Pad Value 计算

```python
def _compute_pad_value(hash: int) -> int:
    """计算多模态 pad 值，避免与 text token 重叠"""
    return MM_PAD_SHIFT_VALUE + (hash % (1 << 30))

# MM_PAD_SHIFT_VALUE = 1,000,000
# 确保 pad_values 在有效 text token 范围之外
```

---

## 七、与现有文档关联

### 7.1 补充 scheduler_analysis.md

| 现有内容 | 补充内容 |
|---------|---------|
| 批次构建 | ScheduleBatch 详细结构 |
| 请求管理 | Req 完整字段说明 |
| 采样 | SamplingParams 详解 |

### 7.2 数据流补充

```
Scheduler
    │
    │ get_next_batch_to_run()
    ▼
ScheduleBatch (CPU)
    │ reqs, input_ids, positions, seq_lens
    │
    │ to_model_worker_batch()
    ▼
ModelWorkerBatch (CPU→GPU)
    │ GPU 张量
    │
    │ to_forward_batch()
    ▼
ForwardBatch (GPU)
    │
    ▼
ModelRunner.forward()
```

---

## 八、关键性能优化

### 8.1 增量 Detokenization

```python
# 避免重复解码整个序列
INCREMENTAL_DETOKENIZATION_OFFSET = 5

def incremental_detokenize(self, new_token_ids: List[int]):
    # 只解码新增的 tokens
    new_text = self.tokenizer.decode(
        new_token_ids[INCREMENTAL_DETOKENIZATION_OFFSET:]
    )
    self.output_text += new_text
```

### 8.2 语法约束

```python
class BaseGrammarObject:
    def accept_token(self, token_id: int):
        """接受 token，更新语法状态"""
        pass
    
    def get_allowed_tokens(self) -> List[int]:
        """获取语法允许的下一个 tokens"""
        pass

# 在采样时应用语法约束
if req.grammar:
    allowed_tokens = req.grammar.get_allowed_tokens()
    logits[~allowed_tokens] = -float('inf')
```

---

## 九、总结

**ScheduleBatch 核心价值**:
1. ✅ Scheduler ↔ ModelRunner 数据桥梁
2. ✅ 请求状态完整管理
3. ✅ 批次构建与转换
4. ✅ 采样参数批量处理
5. ✅ 多模态支持

---

**分析用时**: 25 分钟  
**代码行数**: 2,426 行  
**产出文档**: schedule_batch.md
