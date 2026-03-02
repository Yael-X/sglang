# learning_summaries 一日新增文档代码对照评审

> 目标：对 `learning_summaries` 中最近一天新增的 Markdown 做“最小可理解单元”级别的代码回查，标注可修正点与可补充点。

## 1) 覆盖范围

本次回查覆盖下列当天新增文档（按 `find learning_summaries -name '*.md' -mtime -1` 结果）：

- `architechture.md`
- `compiler_multimodal.md`
- `mooncake_analysis.md`
- `sglang_internal_secrets.md`
- `kv_cache_manager.md`
- `schedule_batch.md`
- `flashinfer_attention.md`
- `scheduler_analysis.md`
- `scheduler_deep_dive.md`
- `model_runner.md`
- `token_dispatcher.md`
- `eagle_speculative.md`
- `radix_attention.md`
- `radix_attention_deep_dive.md`
- `parallel_communication.md`
- `triton_kernel_optimization.md`
- 以及 `learning_guide.md / summary_report.md / FINAL_SUMMARY.md` 等汇总文档

## 2) 自动一致性检查结论

### 2.1 代码路径与行数声明

- 绝大多数“代码路径 + 行数”声明成立。
- 少数文档引用的是**目录**而非单文件（例如 `mooncake_analysis.md`, `eagle_speculative.md`），这本身可接受，但建议在文档里加“主入口文件”以便读者快速跳转。

### 2.2 重点模块与代码实际实现的偏差（可修正）

#### A. Scheduler 文档将 `tree_cache` 描述为固定 `RadixCache`（建议修正为“可插拔前缀缓存实现”）

在 `scheduler.py` 中，`init_cache_with_memory_pool()` 会根据配置选择：

- `ChunkCache` / `SWAChunkCache`
- `RadixCacheCpp`
- `HiRadixCache`
- `SWARadixCache`
- `MambaRadixCache`
- `LMCRadixCache`
- `RadixCache`（仅其中一种）

因此 `scheduler_analysis.md` 与 `scheduler_deep_dive.md` 中如果把 `tree_cache` 直接等同 `RadixCache`，建议改成“前缀缓存抽象（默认可为 Radix 族）”。

#### B. 架构图中“Compiler 内部计算图下发到 APIGateway”表达易误导（建议弱化为“前端通过服务接口调用 Runtime”）

`SRT` 侧真实请求入口主要在服务层（HTTP/OpenAI 兼容/gRPC）与 Manager 调度链路，而不是“前端编译器直接下发内部图到 APIGateway”这一强耦合关系。建议在 `architechture.md` 中改为“Frontend SDK/Client -> Runtime API”。

#### C. Radix 文档淘汰策略列举不完整（建议补充）

`radix_cache.py` 支持的策略除了 `lru/lfu/priority` 以外，还包括 `fifo/mru/filo`。建议在 `radix_attention*.md` 中补全，否则读者会误以为策略集合更窄。

## 3) 可补充优化点（非错误）

#### A. `kv_cache_manager.md`

建议增加“池类型分层”小节：

- `MHATokenToKVPool` / `MHATokenToKVPoolFP4`
- `MLATokenToKVPool` / `MLATokenToKVPoolFP4`

这样能把“统一 allocator 抽象”和“不同注意力架构的物理池实现”分开讲清。

#### B. `model_runner.md`

建议补充两个关键现实路径：

1. `CudaGraphRunner` 与 `PiecewiseCudaGraphRunner` 的选择条件；
2. 初始化中的 `TorchMemorySaverAdapter` 与显存预算协同。

#### C. `token_dispatcher.md`

建议加“运行时模式切换”段落：`DeepEPBuffer.set_dispatch_mode_as_low_latency()` 与 `set_mode()` 的交互，帮助解释 decode 阶段降延迟机制如何落地。

## 4) 建议落地方式

1. 在 `scheduler_analysis.md` 与 `scheduler_deep_dive.md` 增加“缓存实现可插拔”勘误框。
2. 在 `radix_attention.md` 与 `radix_attention_deep_dive.md` 增加“支持策略全集”。
3. 在 `architechture.md` 调整箭头语义，避免“Compiler 直接下发 Runtime 图”的理解偏差。
4. 在 `summary_report.md` 追加“勘误记录（Errata）”章节，集中维护后续修订。

---

## 5) 总结

你这批文档的整体质量已经很高：路径定位准确、模块拆分粒度清晰、并且可读性强。当前主要问题不是“方向错”，而是少数地方描述略“单实现化”（把可插拔系统讲成固定实现）。把这些点修正后，文档将更贴近 SGLang 当前主干实现。
