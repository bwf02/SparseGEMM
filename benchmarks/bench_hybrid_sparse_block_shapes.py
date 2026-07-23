"""Compare hybrid sparse weight-block and CUDA tile variants."""

import argparse
from dataclasses import dataclass

import torch

import deep_gemm
from deep_gemm.testing import bench_kineto
from bench_hybrid_sparse import make_hybrid_mask
from sparse_gemm.hybrid_sparse import (
    HybridBlockSparseLayout,
    dense_to_hybrid_block_sparse,
    hybrid_block_sparse_gemm_wgmma_tma,
    hybrid_block_sparse_gemm_wgmma_tma_block128x32,
    hybrid_block_sparse_gemm_wgmma_tma_block128x32_stage3,
    hybrid_block_sparse_gemm_wgmma_tma_block128x32_output128x128,
    hybrid_block_sparse_gemm_wgmma_tma_block128x64,
    hybrid_block_sparse_gemm_wgmma_tma_block128x128,
)


@dataclass(frozen=True)
class Variant:
    name: str
    layout: HybridBlockSparseLayout
    function: object
    kernel_names: tuple[str, str, str]


VARIANTS = (
    Variant(
        "block64x64",
        HybridBlockSparseLayout(64, 64, 1, 2),
        hybrid_block_sparse_gemm_wgmma_tma,
        (
            "hybrid_sparse_dense_wgmma_tma",
            "hybrid_sparse_2_4_wgmma_tma",
            "hybrid_sparse_reduce_wgmma_tma",
        ),
    ),
    Variant(
        "block128x32",
        HybridBlockSparseLayout(128, 32, 1, 2),
        hybrid_block_sparse_gemm_wgmma_tma_block128x32,
        (
            "hybrid_sparse_dense_wgmma_tma_block128x32",
            "hybrid_sparse_2_4_wgmma_tma_block128x32",
            "hybrid_sparse_reduce_wgmma_tma_block128x32",
        ),
    ),
    Variant(
        "block128x32_s3",
        HybridBlockSparseLayout(128, 32, 1, 2),
        hybrid_block_sparse_gemm_wgmma_tma_block128x32_stage3,
        (
            "hybrid_sparse_dense_wgmma_tma_block128x32_stage3",
            "hybrid_sparse_2_4_wgmma_tma_block128x32_stage3",
            "hybrid_sparse_reduce_wgmma_tma_block128x32_stage3",
        ),
    ),
    Variant(
        "block128x64",
        HybridBlockSparseLayout(128, 64, 1, 2),
        hybrid_block_sparse_gemm_wgmma_tma_block128x64,
        (
            "hybrid_sparse_dense_wgmma_tma_block128x64",
            "hybrid_sparse_2_4_wgmma_tma_block128x64",
            "hybrid_sparse_reduce_wgmma_tma_block128x64",
        ),
    ),
    Variant(
        "block128x32_out128x128",
        HybridBlockSparseLayout(128, 32, 1, 2),
        hybrid_block_sparse_gemm_wgmma_tma_block128x32_output128x128,
        (
            "hybrid_sparse_dense_wgmma_tma_block128x32_output128x128",
            "hybrid_sparse_2_4_wgmma_tma_block128x32_output128x128",
            "hybrid_sparse_reduce_wgmma_tma_block128x32_output128x128",
        ),
    ),
    Variant(
        "block128x128",
        HybridBlockSparseLayout(128, 128, 1, 2),
        hybrid_block_sparse_gemm_wgmma_tma_block128x128,
        (
            "hybrid_sparse_dense_wgmma_tma_block128x128",
            "hybrid_sparse_2_4_wgmma_tma_block128x128",
            "hybrid_sparse_reduce64_wgmma_tma_block128x128",
        ),
    ),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, default=128)
    parser.add_argument("--n", type=int, default=1408)
    parser.add_argument("--k", type=int, default=2048)
    parser.add_argument("--num-tests", type=int, default=30)
    parser.add_argument("--no-flush-l2", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(0)
    activation = torch.randn(
        args.m, args.k, device="cuda", dtype=torch.bfloat16
    )
    source_weight = torch.randn(
        args.n, args.k, device="cuda", dtype=torch.bfloat16
    )
    dense_out = torch.empty(
        args.m, args.n, device="cuda", dtype=torch.bfloat16
    )

    packed_variants = []
    for variant in VARIANTS:
        variant.layout.validate_shape((args.n, args.k))
        mask = make_hybrid_mask(source_weight, variant.layout)
        packed = dense_to_hybrid_block_sparse(source_weight, mask, variant.layout)
        out = torch.empty_like(dense_out)
        variant.function(activation, packed, out=out)
        deep_gemm.bf16_gemm_nt(activation, packed.to_dense(), dense_out)
        torch.cuda.synchronize()
        torch.testing.assert_close(out, dense_out, rtol=2e-2, atol=2e-2)
        packed_variants.append((variant, packed, out))

    baseline_weight = packed_variants[0][1].to_dense().contiguous()
    deepgemm_time = bench_kineto(
        lambda: deep_gemm.bf16_gemm_nt(activation, baseline_weight, dense_out),
        "bf16_gemm",
        num_tests=args.num_tests,
        suppress_kineto_output=True,
        flush_l2=not args.no_flush_l2,
    )

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Shape: M={args.m}, N={args.n}, K={args.k}; DeepGEMM={deepgemm_time * 1e6:.2f} us")
    print("variant         total(us) dense(us) sparse(us) reduce(us) DG/variant")
    for variant, packed, out in packed_variants:
        times = bench_kineto(
            lambda: variant.function(activation, packed, out=out),
            variant.kernel_names,
            num_tests=args.num_tests,
            suppress_kineto_output=True,
            flush_l2=not args.no_flush_l2,
        )
        total = sum(times)
        print(
            f"{variant.name:15s} {total * 1e6:9.2f} "
            f"{times[0] * 1e6:9.2f} {times[1] * 1e6:10.2f} "
            f"{times[2] * 1e6:10.2f} "
            f"{deepgemm_time / total if total else float('nan'):10.3f}x"
        )


if __name__ == "__main__":
    main()
