# SGLang 深度解析任务计划

## 执行周期
- **启动**: 2026-03-02 01:18 (GMT+8)
- **终止**: 2026-03-02 07:30 (GMT+8)
- **总时长**: 6 小时 12 分钟

## 现有文档分析

### architechture.md (7KB)
- ✅ 三层架构图 (Frontend/Runtime/Hardware)
- ✅ Mermaid + PlantUML 双格式
- ✅ 核心组件标注

### parallel_communication.md (34KB)
- ✅ 并行策略详解 (TP/EP/PP/DP)
- ✅ DeepEP 通信流程
- ✅ 代码引用 + 示例

## 待分析模块清单

### 优先级 P0 (核心架构)
1. **Scheduler 调度器** - `python/sglang/srt/managers/scheduler.py`
2. **RadixAttention** - `python/sglang/srt/managers/radix_attention.py`
3. **KV Cache Manager** - `python/sglang/srt/memory_pool/`
4. **Model Runner** - `python/sglang/srt/model_executor/model_runner.py`

### 优先级 P1 (关键组件)
5. **Token Dispatcher** - `python/sglang/srt/layers/moe/token_dispatcher/`
6. **FlashInfer Attention** - `python/sglang/srt/layers/attention/`
7. **Scheduler Batch** - `python/sglang/srt/managers/schedule_batch.py`
8. **Detokenizer** - `python/sglang/srt/managers/detokenizer.py`

### 优先级 P2 (扩展模块)
9. **Health Manager** - `python/sglang/srt/managers/health_manager.py`
10. **Session Manager** - `python/sglang/srt/managers/session_manager.py`
11. **Cache Manager** - `python/sglang/srt/cache/`

## 输出格式标准

每个模块输出包含：
1. **模块定位** - 代码路径 + 职责
2. **核心类/函数** - 关键代码结构
3. **流程解析** - 算法逻辑详解
4. **Mermaid 图** - 流程图/类图
5. **代码片段** - 关键实现 (≤500 行)
6. **与现有文档关联** - 补充/修正点

## 时间分配

| 时间段 | 任务 | 产出 |
|--------|------|------|
| 01:18-01:45 | 环境准备 + 文档分析 | 任务计划.md |
| 01:45-02:30 | Scheduler 调度器 | scheduler_analysis.md |
| 02:30-03:15 | RadixAttention | radix_attention.md |
| 03:15-04:00 | KV Cache Manager | kv_cache_manager.md |
| 04:00-04:45 | Model Runner | model_runner.md |
| 04:45-05:30 | Token Dispatcher | token_dispatcher.md |
| 05:30-06:15 | FlashInfer Attention | flashinfer_attention.md |
| 06:15-07:00 | 补充模块 + 整理 | 其他分析.md |
| 07:00-07:30 | 最终报告汇总 | summary_report.md |

## 进度追踪

- [ ] Scheduler 调度器
- [ ] RadixAttention
- [ ] KV Cache Manager
- [ ] Model Runner
- [ ] Token Dispatcher
- [ ] FlashInfer Attention
- [ ] 补充模块
- [ ] 最终报告

## 注意事项

1. 每次分析代码量 ≤3000 行
2. 每个模块产出独立 Markdown
3. 使用 Mermaid 绘图
4. 代码引用标注行号
5. 20 分钟无产出则跳过

---

**当前状态**: 准备就绪 (01:23)
**下一步**: 开始分析 Scheduler 调度器
