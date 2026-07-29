# Hybrid Sparse Format and Kernels

## Sparse Pattern

- Weight 支持 `[N, K]` 和 grouped `[E, N, K]`。
- Weight 按 `block_h x block_w` 分块，并沿 K 维将连续 `block_m` 个 block 组成一组。
- 每组恰有 `block_n` 个 block 使用块内 2:4，其余 block 保持 dense。
- 总体稀疏率为 `block_n / block_m x 50%`；当前 kernel 主路径使用 `64 x 64, 1:2`，即 25% 稀疏率。
- Weight tile 是 kernel 读取的 `N x K` tile；当前要求等于 weight block。Output tile 的 M 维可独立调优。

记 `BR=N/block_h`、`BG=K/(block_w*block_m)`、`D=block_m-block_n`，存储字段如下：

| 字段 | dtype | shape | 含义 |
|---|---|---|---|
| `block_selector` | `int64` | `[..., BR, BG]` | bit mask，标记组内哪些 block 为 2:4 |
| `dense_values` | weight dtype | `[..., BR, BG, D, block_h, block_w]` | dense block 的完整数据 |
| `sparse_values` | weight dtype | `[..., BR, BG, block_n, block_h, block_w/2]` | 2:4 block 保留的两个元素 |
| `sparse_metadata` | `uint8` | `[..., BR, BG, block_n, block_h, block_w/4]` | 每个 quartet 的六种保留位置编码 `0..5` |
| `hardware_metadata` | `int32` | `[..., BR, BG, block_n, 2, 4, 16]` | `64 x 64` 专用的 lane-ready WGMMA.SP metadata |

`dense_to_hybrid_block_sparse` 校验 block N:M 和块内 2:4 后分离 dense/sparse stream；`to_dense()` 根据 selector 与 metadata 重建零填充 dense weight，供 Torch/DeepGEMM correctness reference 使用。

## Kernel Versions

| 版本 | 核心优化 | 结论 |
|---|---|---|
| Naive | dense kernel、sparse kernel 和 reduce 分开执行 | correctness baseline |
| Tensor Core / sync WGMMA | dense WGMMA 与 WGMMA.SP 替代标量计算 | Tensor Core baseline |
| TMA pipeline | TMA 搬运 activation/weight，双缓冲隐藏访存 | 初始异步基线 |
| Metadata prefetch | block-row metadata 连续搬入 shared memory | 消除细粒度 metadata global load |
| Fused mainloop | dense 与 sparse 路径累加到同一 FP32 accumulator | 移除 partial buffer 和 reduce kernel |
| STSM/TMA epilogue | FP32 转 BF16，STSM 写 shared，再由 TMA store 写回 | 当前 epilogue 基线 |
| Persistent + lane-ready | persistent tile scheduler；转换阶段预编码 WGMMA.SP metadata | 减少调度和 metadata 解码开销 |
| Producer metadata copy | producer warp 连续复制 metadata 并广播 block kind | 大 M `output128` 路径基线 |
| Group-stage | 一次 staging 一个完整 block group | 降低 barrier、selector 和 WGMMA 控制开销 |
| `1:2` fastpath | 固化一 dense block 加一 sparse block，移除通用 popcount/loop | 当前 25% 稀疏主路径 |
| Descriptor reuse | 预构造并复用 GMMA descriptor | 用于 output80/88/128 等版本 |
| Fixed-shape/unroll/async group | 固定 K loop、合并 WGMMA group、增加 pipeline overlap | 对特定 shape 单独采用 |
| Shape-aware dispatch | 离线测试各候选后，按 `(M,N,K)` 静态选择 winner | 当前公开入口 |

## Current Dispatch

| `(N, K)` | M | Selected kernel |
|---|---|---|
| `(1408, 2048)` | 128 | output48 + descriptor reuse |
| `(1408, 2048)` | 256 | output88 + descriptor reuse |
| `(1408, 2048)` | 512, 1024 | output80 + descriptor reuse |
| `(2048, 1408)` | 128 | output64 + constexpr branch-group |
| `(2048, 1408)` | 256 | output128 + fixed K + stage4 async-group2 |
| `(2048, 1408)` | 512 | output128 + descriptor reuse |
| `(2048, 1408)` | 1024 | output96 group-stage |
| both | 2048, 4096 | output128 + producer metadata + stage-kind |

未收录 shape 或非 `1:2` pattern 回退到通用 `output64 x 64` group-stage kernel。性能结果只维护在 `hybrid_sparse_performance.xlsx`。

## Grouped GEMM

- Packed weight 直接使用 `[E, N, K]`，各 expert 共享同一种 layout，但拥有独立 selector、dense/sparse values 和 metadata。
- Contiguous 模式使用 `A[total_m,K]` 与 psum `grouped_layout[E]`；expert 起点按 `m_alignment` 对齐，对齐空洞输出为零。
- Masked 模式使用 `A[E,max_m,K]` 与 `masked_m[E]`；无效 tail 输出为零，第一版要求 `max_m % 64 == 0`。
- Fused grouped kernel 使用 persistent `64 x 64` output tile、group-stage TMA、dense WGMMA、WGMMA.SP 和 STSM/TMA epilogue；每个 tile 解析 expert 后直接索引对应 packed stream，不生成 gather weight 或 partial output。
- `*_naive` 三 kernel 路径继续作为 correctness baseline；`*_wgmma_tma` 是 fused grouped 入口。
