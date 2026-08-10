# Hybrid Sparse Format and Kernels

## Sparse Pattern

- Weight 支持 `[N, K]` 和 grouped `[E, N, K]`。
- Weight 按 `block_h x block_w` 分块，并沿 K 维将连续 `block_m` 个 block 组成一组。
- 每组恰有 `block_n` 个 block 使用块内 2:4，其余 block 保持 dense。
- 总体稀疏率为 `block_n / block_m x 50%`；当前 kernel 主路径使用 `64 x 64, 1:2`，即 25% 稀疏率。
- Weight tile 是 kernel 读取的 `N x K` tile；当前要求等于 weight block。Output tile 的 M 维可独立调优。

### Variables

| 变量 | 含义 |
|---|---|
| `E` | expert 数；普通 GEMM 无此维度 |
| `N` | weight 行数，即 output-channel 维度 |
| `K` | reduction 维度 |
| `block_h` | weight block 在 N 维的高度 |
| `block_w` | weight block 在 K 维的宽度，必须是 4 的倍数 |
| `block_m` | 沿 K 维每组包含的 block 数 |
| `block_n` | 每组执行块内 2:4 的 block 数 |
| `BR` | block row 数，`N / block_h` |
| `BG` | block group 数，`K / (block_w * block_m)` |
| `D` | 每组 dense block 数，`block_m - block_n` |
| `...` | 可选前导维度；grouped weight 中为 `[E]` |

### Packed Storage

| 字段 | dtype | shape | 含义 |
|---|---|---|---|
| `original_shape` | Python tuple | `[N,K]` 或 `[E,N,K]` | 压缩前 weight shape |
| `layout` | config | - | `block_h/block_w/block_n/block_m` |
| `block_selector` | `int64` | `[...,BR,BG]` | 每一 bit 对应组内一个 block；`1` 表示 2:4，`0` 表示 dense |
| `dense_values` | weight dtype | `[...,BR,BG,D,block_h,block_w]` | dense block 的完整元素 |
| `sparse_values` | weight dtype | `[...,BR,BG,block_n,block_h,block_w/2]` | 每个 2:4 quartet 保留的两个元素 |
| `sparse_metadata` | `uint8` | `[...,BR,BG,block_n,block_h,block_w/4]` | 每个 quartet 的非零位置编码 `0..5` |
| `hardware_metadata` | `int32` | `[...,BR,BG,block_n,2,4,16]` | `64 x 64` sparse block 专用的 lane-ready WGMMA.SP metadata |

`block_selector` 决定 sparse block 在原 group 中的位置。`dense_values` 和 `sparse_values` 只保存各自的紧凑 stream，stream 内按原 block 索引升序排列。

### Encoding Example

以 `block_h=4, block_w=8, block_n:block_m=1:2` 为例，一个 group 包含 `B0/B1` 两个 block。若 `B0` dense、`B1` 2:4：

```text
block_selector = 0b10
dense_values   = [B0]
sparse_values  = [compress(B1)]
```

`B1` 的某一行为：

```text
[3.5, 0, -1.2, 0] [0, 2.0, 0, -4.0]
       quartet 0            quartet 1
```

保留位置的通用编码为：

| code | 保留位置 |
|---:|---|
| `0` | `(0,1)` |
| `1` | `(0,2)` |
| `2` | `(0,3)` |
| `3` | `(1,2)` |
| `4` | `(1,3)` |
| `5` | `(2,3)` |

因此该行压缩为：

```text
sparse_values[row]   = [3.5, -1.2, 2.0, -4.0]
sparse_metadata[row] = [1, 4]
```

`block_w/2=4` 是保留的 value 数；`block_w/4=2` 是 quartet 数，每个 quartet 使用一个 `uint8` code。

### Hardware Metadata

`sparse_metadata` 是与硬件无关的通用格式，用于验证和 `to_dense()`。对 `64 x 64` sparse block，weight conversion 还会生成 `hardware_metadata`：

```text
pair code 0..5
  -> WGMMA.SP 4-bit code: 0x4, 0x8, 0xC, 0x9, 0xD, 0xE
  -> 按 K tile / warp / lane 重排
  -> 每 8 个 4-bit code 打包为一个 int32
```

例如通用 code `1` 表示保留 `(0,2)`，转换为 WGMMA.SP code `0x8`。优化 kernel 可直接将 lane-ready `int32` 传给 `wgmma.mma_async.sp`，无需在 mainloop 内查表和重排。

当前同时保存通用和硬件 metadata。对 BF16 `64 x 64, 1:2` 的一个 block group：

| 数据 | 大小 |
|---|---:|
| `dense_values` | `8192 B` |
| `sparse_values` | `4096 B` |
| `sparse_metadata` | `1024 B` |
| `hardware_metadata` | `512 B` |
| `block_selector` | `8 B` |
| 合计 | `13832 B` |

`sparse_metadata` 占两套 metadata 的 `66.67%`，占当前 packed weight 的 `7.40%`。若部署格式只保留 `hardware_metadata`，总大小从 dense weight 的 `84.42%` 降至 `78.17%`。

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
