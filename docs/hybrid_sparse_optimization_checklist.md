# Hybrid Sparse Kernel Optimization Checklist

勾选表示实验已经完成，不代表该方案最终被采用。性能数据见
[`hybrid_sparse_performance.xlsx`](hybrid_sparse_performance.xlsx)。

标准测试使用 NVIDIA H20、BF16/FP32 accumulation，以及
`M = 128, 256, 512, 1024, 2048, 4096`。

## 待尝试

- [ ] **P0：联合 K-block merge 与更深 pipeline**，仅在增加 TMA stage 后重新评估 `merge_k=2`，避免两级 buffer 成对释放阻断预取。
- [ ] **P1：CTA tile swizzle**，调整 tile 调度顺序以提高 activation 或 weight 的 L2 复用率。
- [ ] **P1：grouped GEMM persistent scheduler**，让固定数量的 CTA 持续领取不均匀 expert tile。
- [ ] **P3：TMA multicast/CTA cluster**，仅在 operand 复用和并行 wave 足够时评估 cluster 共享收益。

## 已完成

- [x] **两级 TMA pipeline**，相比同步 WGMMA 显著加速并成为初始优化基线。
- [x] **weight block shape 调优**，`64 x 64` 在 M=128 最快，而 `128 x 128` 在 M=512 更有优势。
- [x] **双 consumer warpgroup**，occupancy 提高但 sparse kernel 变慢，因此保留代码但不采用。
- [x] **shared-memory bank conflict 分析**，NCU 未发现冲突，因此不继续调整当前 TMA swizzle。
- [x] **memory bandwidth 分析**，DRAM 利用率较低，确认当前主要受 latency 而非 HBM bandwidth 限制。
- [x] **block-row metadata prefetch**，标准 shape 总延迟降低 `6.6%–15.1%`，因此保留为当前首选版本。
- [x] **融合 dense/sparse mainloop**，在同一 FP32 accumulator 中累加两条路径并移除 partial buffer 与 reduce kernel。
- [x] **BF16 STSM/TMA epilogue**，先转换 BF16，再用 STSM 写入 swizzled shared memory，最后由 TMA 写回 global memory。
- [x] **普通 GEMM persistent scheduler**，使用 `3 CTA/SM` 的 grid-stride tile 调度；单独使用时仍慢于静态 STSM，作为后续 persistent 优化基础保留。
- [x] **预编码硬件 metadata**，weight conversion 直接生成 lane-ready WGMMA.SP words，移除 mainloop byte-code 解码，并降低寄存器、shared memory 和 warp 指令量。
- [x] **`128 x 32` weight tile + 三级 TMA pipeline**，Stage 3 仅在 M=2048 相对 Stage 2 改善 `8.8%`，其他 M 持平或回退，且整体仍慢于当前 `64 x 64` fused kernel，因此不加入 shape-aware dispatch。
- [x] **`merge_k=2` WGMMA group**，CTA barrier stall 从 `67.1%` 降至 `57.5%`，但 NCU duration 增加 `4.9%` 且多数 shape 回退，因此保留实现但不采用。
- [x] **`64 x 64` weight block 的 TMA pipeline depth**，Stage 3 的 NCU duration 从 `62.24 us` 降至 `60.93 us`；Stage 4/5 将理论 occupancy 降至 `25%` 并造成中大 M 回退，因此选择 Stage 3。
- [x] **warpgroup 寄存器重分配**，producer/math 分别设置为 `40/128`，但 kernel 仍使用 `60 registers/thread`，两次配对 NCU 的平均延迟变化小于 `0.1%`，因此保留实现但不采用。
- [x] **direct lane-ready metadata load**，shared memory 从 `66.56 KB` 降至 `58.37 KB`，但 long-scoreboard 从 `1.69` 升至 `2.88 cycles/inst`，因此不采用。
- [x] **persistent epilogue overlap**，CTA barrier stall 从 `16.05` 降至 `7.03 cycles/inst`，但延迟仅改善约 `0.3%`，作为后续 metadata 优化基础保留。
- [x] **metadata register prefetch**，long-scoreboard 小幅下降，但指令数增加约 `5.5%` 且 NCU 延迟回退，因此不采用。
- [x] **stage-local metadata TMA**，降低 math warp 的 global-load stall，但每个 sparse block 增加一次 TMA transaction，普通 benchmark 改善而 NCU cache-control 结果回退。
- [x] **producer-warp metadata copy**，用连续 `512 B` vector copy 替代额外 metadata TMA；NCU 延迟从 `62.05 us` 降至 `59.87 us`，作为后续 output tile 优化的 metadata 基线。
- [x] **`128 x 64` output tile 与双 math warpgroup**，两个 warpgroup 共享同一 `64 x 64` weight tile 和 metadata；在大 M 上减少 CTA、TMA load 和控制指令。
- [x] **producer 广播 stage block kind**，consumer 不再重复读取 global block selector；NCU global-load 从 `112,640` 降至 `22,528`，总指令从 `19.76 M` 降至 `18.42 M`，为当前最佳版本。
- [x] **按 block group 广播完整 selector**，shared control load 数量下降，但稳定计时略慢于 stage-kind，因此保留实现但不采用。
- [x] **验证 `1.25x` 目标**，在 `4096 x 2048 x 1408` 上三轮各 100 次配对计时中，stage-kind 相对 DeepGEMM 为 `1.261x`、`1.259x`、`1.262x`。
- [x] **验证全部标准 shape**，12 个 shape 均完成重复配对测试，其中 8 个快于 DeepGEMM、2 个达到至少 `1.25x`；`M=512/1024` 加测至 7 轮。
- [x] **整组 N:M staging**，一次加载并提交完整 outer block group，`M=128,N=1408,K=2048` 的 NCU 指令数下降 `36%`。
- [x] **小 M token tile 调优**，新增 `48 x 64`、`80 x 64` 和 `96 x 64` output tile，其中 48/96 分别解决 M=128/256 的 CTA wave 不足或尾波问题。
- [x] **`1:2` block dispatch fastpath**，移除通用 block loop 和 popcount，同时保留其他 N:M 的通用 fallback。
- [x] **group-stage metadata TMA**，消除 L1TEX scoreboard 并将 shared-store bank conflicts 从 `6379` 降至 `697`，但 warm-cache benchmark 无收益，因此不进入默认 dispatch。
- [x] **shape-aware dispatch**，为 8 个已验证 Qwen MoE shape 固化 `48/64/80/96/128` token tile 选择，其他 shape 和非 `1:2` N:M 回退到通用 group-stage。
