"""Benchmark naive hybrid grouped GEMM against DeepGEMM grouped BF16."""

import argparse

import torch

import deep_gemm
from deep_gemm.testing import bench_kineto
from deep_gemm.utils import align
from sparse_gemm.hybrid_sparse import (
    HybridBlockSparseLayout,
    dense_to_hybrid_block_sparse,
    hybrid_block_sparse_grouped_contiguous_naive,
    hybrid_block_sparse_grouped_masked_naive,
)

from bench_hybrid_sparse import make_hybrid_mask


HYBRID_GROUPED_KERNEL_NAMES = (
    "hybrid_sparse_grouped_dense_naive",
    "hybrid_sparse_grouped_2_4_naive",
    "hybrid_sparse_grouped_reduce_naive",
)


def make_packed_weight(
    experts: int, n: int, k: int, layout: HybridBlockSparseLayout
):
    source = torch.randn(
        experts, n, k, device="cuda", dtype=torch.bfloat16
    )
    mask = torch.stack([make_hybrid_mask(weight, layout) for weight in source])
    return dense_to_hybrid_block_sparse(source, mask, layout)


def time_hybrid(fn, num_tests: int, flush_l2: bool) -> tuple[float, ...]:
    return bench_kineto(
        fn,
        HYBRID_GROUPED_KERNEL_NAMES,
        num_tests=num_tests,
        suppress_kineto_output=True,
        flush_l2=flush_l2,
    )


def print_result(
    mode: str,
    valid_m: int,
    n: int,
    k: int,
    sparsity: float,
    hybrid_times: tuple[float, ...],
    deepgemm_time: float,
) -> None:
    hybrid_time = sum(hybrid_times)
    dense_flops = 2 * valid_m * n * k
    executed_flops = dense_flops * (1.0 - sparsity)
    print(
        f"{mode:10s} {valid_m:8d} | "
        f"{hybrid_time * 1e6:10.1f} {deepgemm_time * 1e6:11.1f} "
        f"{deepgemm_time / hybrid_time:8.3f}x | "
        f"{dense_flops / hybrid_time / 1e12:10.2f} "
        f"{executed_flops / hybrid_time / 1e12:10.2f} "
        f"{dense_flops / deepgemm_time / 1e12:10.2f} | "
        f"{hybrid_times[0] * 1e6:8.1f} "
        f"{hybrid_times[1] * 1e6:8.1f} "
        f"{hybrid_times[2] * 1e6:8.1f}"
    )


def benchmark_contiguous(
    packed_weight,
    tokens_per_expert: int,
    num_tests: int,
    flush_l2: bool,
) -> None:
    experts, n, k = packed_weight.original_shape
    m_alignment = deep_gemm.get_mk_alignment_for_contiguous_layout()
    ends = []
    previous_end = 0
    for expert in range(experts):
        start = 0 if expert == 0 else align(previous_end, m_alignment)
        previous_end = start + tokens_per_expert
        ends.append(previous_end)
    total_m = align(previous_end, m_alignment)

    a = torch.randn(total_m, k, device="cuda", dtype=torch.bfloat16)
    grouped_layout = torch.tensor(ends, device="cuda", dtype=torch.int32)
    dense_weight = packed_weight.to_dense().contiguous()
    hybrid_out = torch.empty(total_m, n, device="cuda", dtype=torch.bfloat16)
    deepgemm_out = torch.empty_like(hybrid_out)

    hybrid_fn = lambda: hybrid_block_sparse_grouped_contiguous_naive(
        a, packed_weight, grouped_layout, m_alignment, out=hybrid_out
    )
    deepgemm_fn = lambda: deep_gemm.m_grouped_bf16_gemm_nt_contiguous(
        a,
        dense_weight,
        deepgemm_out,
        grouped_layout,
        use_psum_layout=True,
        ensure_zero_padding=True,
        expected_m_for_psum_layout=tokens_per_expert,
    )
    hybrid_fn()
    deepgemm_fn()
    torch.cuda.synchronize()
    previous_end = 0
    valid_rows = torch.zeros(total_m, device="cuda", dtype=torch.bool)
    for expert, end in enumerate(ends):
        start = 0 if expert == 0 else align(previous_end, m_alignment)
        torch.testing.assert_close(
            hybrid_out[start:end],
            deepgemm_out[start:end],
            rtol=2e-2,
            atol=2e-2,
        )
        valid_rows[start:end] = True
        previous_end = end
    if torch.any(~valid_rows):
        torch.testing.assert_close(
            hybrid_out[~valid_rows], torch.zeros_like(hybrid_out[~valid_rows])
        )

    hybrid_times = time_hybrid(hybrid_fn, num_tests, flush_l2)
    deepgemm_time = bench_kineto(
        deepgemm_fn,
        "bf16_gemm",
        num_tests=num_tests,
        suppress_kineto_output=True,
        flush_l2=flush_l2,
    )
    print_result(
        "contiguous",
        experts * tokens_per_expert,
        n,
        k,
        packed_weight.layout.sparsity,
        hybrid_times,
        deepgemm_time,
    )


def benchmark_masked(
    packed_weight,
    tokens_per_expert: int,
    max_m: int,
    num_tests: int,
    flush_l2: bool,
) -> None:
    experts, n, k = packed_weight.original_shape
    a = torch.randn(experts, max_m, k, device="cuda", dtype=torch.bfloat16)
    masked_m = torch.full(
        (experts,), tokens_per_expert, device="cuda", dtype=torch.int32
    )
    dense_weight = packed_weight.to_dense().contiguous()
    hybrid_out = torch.empty(
        experts, max_m, n, device="cuda", dtype=torch.bfloat16
    )
    deepgemm_out = torch.empty_like(hybrid_out)

    hybrid_fn = lambda: hybrid_block_sparse_grouped_masked_naive(
        a, packed_weight, masked_m, out=hybrid_out
    )
    deepgemm_fn = lambda: deep_gemm.m_grouped_bf16_gemm_nt_masked(
        a, dense_weight, deepgemm_out, masked_m, tokens_per_expert
    )
    hybrid_fn()
    deepgemm_fn()
    torch.cuda.synchronize()
    torch.testing.assert_close(
        hybrid_out[:, :tokens_per_expert],
        deepgemm_out[:, :tokens_per_expert],
        rtol=2e-2,
        atol=2e-2,
    )

    hybrid_times = time_hybrid(hybrid_fn, num_tests, flush_l2)
    deepgemm_time = bench_kineto(
        deepgemm_fn,
        "bf16_gemm",
        num_tests=num_tests,
        suppress_kineto_output=True,
        flush_l2=flush_l2,
    )
    print_result(
        "masked",
        experts * tokens_per_expert,
        n,
        k,
        packed_weight.layout.sparsity,
        hybrid_times,
        deepgemm_time,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--tokens-per-expert", type=int, default=8)
    parser.add_argument(
        "--max-m",
        type=int,
        default=64,
        help="masked activation capacity; DeepGEMM SM90 baseline requires at least 64",
    )
    parser.add_argument("--n", type=int, default=1408)
    parser.add_argument("--k", type=int, default=2048)
    parser.add_argument("--block-n", type=int, default=1)
    parser.add_argument("--block-m", type=int, default=2)
    parser.add_argument("--num-tests", type=int, default=30)
    parser.add_argument("--no-flush-l2", action="store_true")
    args = parser.parse_args()
    if args.experts <= 0 or args.tokens_per_expert <= 0:
        parser.error("--experts and --tokens-per-expert must be positive")
    if args.max_m < args.tokens_per_expert:
        parser.error("--max-m must be at least --tokens-per-expert")
    if args.max_m < 64:
        parser.error("--max-m must be at least 64 for the DeepGEMM SM90 baseline")
    if args.num_tests <= 0:
        parser.error("--num-tests must be positive")
    return args


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(0)
    layout = HybridBlockSparseLayout(64, 64, args.block_n, args.block_m)
    layout.validate_shape((args.n, args.k))
    packed_weight = make_packed_weight(args.experts, args.n, args.k, layout)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(
        f"Experts: {args.experts}, N={args.n}, K={args.k}, "
        f"tokens/expert={args.tokens_per_expert}, max_m={args.max_m}"
    )
    print(
        f"Hybrid layout: 64x64, sparse blocks {layout.block_n}:{layout.block_m}, "
        f"element sparsity {layout.sparsity:.1%}"
    )
    print("Timing: CUDA kernel time from bench_kineto; packing is excluded")
    print(
        "mode        valid-M | hybrid(us) deepgemm(us)  speedup | "
        "hyb-eff-TF hyb-exec-TF   dg-TFLOPS | dense(us) sparse(us) reduce(us)"
    )
    benchmark_contiguous(
        packed_weight,
        args.tokens_per_expert,
        args.num_tests,
        not args.no_flush_l2,
    )
    benchmark_masked(
        packed_weight,
        args.tokens_per_expert,
        args.max_m,
        args.num_tests,
        not args.no_flush_l2,
    )


if __name__ == "__main__":
    main()
