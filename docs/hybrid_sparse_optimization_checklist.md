# Hybrid Sparse Kernel Optimization Checklist

勾选表示实验已经完成，不代表该方案最终被采用。性能数据见
[`hybrid_sparse_performance.xlsx`](hybrid_sparse_performance.xlsx)。

标准测试使用 NVIDIA H20、BF16/FP32 accumulation，以及
`M = 128, 256, 512, 1024, 2048, 4096`。

## 待尝试

- [ ] **P0：联合 K-block merge 与更深 pipeline**，仅在增加 TMA stage 后重新评估 `merge_k=2`，避免两级 buffer 成对释放阻断预取。
- [ ] **P1：CTA tile swizzle**，调整 tile 调度顺序以提高 activation 或 weight 的 L2 复用率。
- [ ] **P1：shape-aware dispatch**，保持 `64 x 64` 为默认路径，仅继续评估 fused、lane-ready 的 `128 x 128` 等候选版本。
- [ ] **P1：grouped GEMM persistent scheduler**，让固定数量的 CTA 持续领取不均匀 expert tile。
- [ ] **P2：persistent epilogue overlap**，让 TMA store 与同一 CTA 的下一 output tile mainloop 重叠。
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
