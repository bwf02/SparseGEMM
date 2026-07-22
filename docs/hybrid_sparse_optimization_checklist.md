# Hybrid Sparse Kernel Optimization Checklist

勾选表示实验已经完成，不代表该方案最终被采用，详细数据与结论见
[`hybrid_sparse_optimization_results.md`](hybrid_sparse_optimization_results.md)。

标准测试使用 NVIDIA H20、BF16/FP32 accumulation，以及
`M = 128, 256, 512, 1024, 2048, 4096`。

## 待尝试

- [ ] **P0：预编码硬件 metadata**，在 weight conversion 阶段直接生成 lane-ready WGMMA.SP metadata，移除 mainloop 中的 byte-code 解码。
- [ ] **P0：增加 TMA pipeline stage**，保持 `64 x 64` weight block 并依次测试 3、4、5 stage 的延迟和 occupancy。
- [ ] **P0：合并多个 K tile**，先测试 `merge_k=2`，减少 WGMMA `commit_group/wait_group` 的执行频率。
- [ ] **P0：重分配 warpgroup 寄存器**，减少 producer 寄存器并增加 math warpgroup 可用寄存器。
- [ ] **P1：CTA tile swizzle**，调整 tile 调度顺序以提高 activation 或 weight 的 L2 复用率。
- [ ] **P1：shape-aware dispatch**，小 M 使用 `64 x 64`，较大 M 根据实测选择 `128 x 128` 等版本。
- [ ] **P1：grouped GEMM persistent scheduler**，让固定数量的 CTA 持续领取不均匀 expert tile。
- [ ] **P2：融合 dense/sparse mainloop**，在同一 accumulator 中累加两条路径并移除 partial buffer 与 reduce kernel。
- [ ] **P2：融合 STSM/TMA epilogue**，通过 shared-memory staging 和 TMA store 写出最终结果并尝试重叠下一 tile。
- [ ] **P3：TMA multicast/CTA cluster**，仅在 operand 复用和并行 wave 足够时评估 cluster 共享收益。

## 已完成

- [x] **两级 TMA pipeline**，相比同步 WGMMA 显著加速并成为初始优化基线。
- [x] **weight block shape 调优**，`64 x 64` 在 M=128 最快，而 `128 x 128` 在 M=512 更有优势。
- [x] **双 consumer warpgroup**，occupancy 提高但 sparse kernel 变慢，因此保留代码但不采用。
- [x] **shared-memory bank conflict 分析**，NCU 未发现冲突，因此不继续调整当前 TMA swizzle。
- [x] **memory bandwidth 分析**，DRAM 利用率较低，确认当前主要受 latency 而非 HBM bandwidth 限制。
- [x] **block-row metadata prefetch**，标准 shape 总延迟降低 `6.6%–15.1%`，因此保留为当前首选版本。
