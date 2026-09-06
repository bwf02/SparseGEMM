"""Fixed-shape MoE batch baselines for SlideSparse/cuSPARSELt and Sputnik."""

from __future__ import annotations

import ctypes
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent


def prune_2_of_8(weight: torch.Tensor) -> torch.Tensor:
    """Magnitude-prune exactly two values in every contiguous group of eight."""
    if weight.shape[-1] % 8:
        raise ValueError("K must be divisible by 8")
    grouped = weight.reshape(*weight.shape[:-1], -1, 8)
    prune = grouped.float().abs().topk(2, dim=-1, largest=False).indices
    return grouped.scatter(-1, prune, 0).reshape_as(weight).contiguous()


def prune_2_of_4(weight: torch.Tensor) -> torch.Tensor:
    """Magnitude-prune exactly two values in every contiguous group of four."""
    if weight.shape[-1] % 4:
        raise ValueError("K must be divisible by 4")
    grouped = weight.reshape(*weight.shape[:-1], -1, 4)
    prune = grouped.float().abs().topk(2, dim=-1, largest=False).indices
    return grouped.scatter(-1, prune, 0).reshape_as(weight).contiguous()


def slide_weight_2_of_8(weight: torch.Tensor) -> torch.Tensor:
    """Apply SlideSparse's greedy 6:8-to-2:4 weight expansion."""
    if weight.shape[-1] % 8:
        raise ValueError("K must be divisible by 8")
    groups = weight.reshape(*weight.shape[:-1], -1, 8)
    allocated = torch.zeros_like(groups, dtype=torch.bool)
    windows = []
    for start in (0, 2, 4):
        values = groups[..., start : start + 4]
        candidates = values.ne(0) & ~allocated[..., start : start + 4]
        selected = candidates & (candidates.cumsum(dim=-1) <= 2)
        windows.append(torch.where(selected, values, 0))
        allocated[..., start : start + 4] |= selected
    if torch.any(groups.ne(0) & ~allocated):
        raise ValueError("weight does not satisfy SlideSparse's 2:8 allocation")
    return torch.stack(windows, dim=-2).flatten(-3).contiguous()


def slide_activation_2_of_8(activation: torch.Tensor) -> torch.Tensor:
    """Duplicate activation windows corresponding to SlideSparse 6:8 weights."""
    if activation.shape[-1] % 8:
        raise ValueError("K must be divisible by 8")
    groups = activation.reshape(*activation.shape[:-1], -1, 8)
    return torch.stack(
        (groups[..., 0:4], groups[..., 2:6], groups[..., 4:8]), dim=-2
    ).flatten(-3).contiguous()


def dense_to_fixed_csr(weight: torch.Tensor):
    """Convert [B,N,K] fixed-density weights to Sputnik's packed CSR tensors."""
    if weight.ndim != 3:
        raise ValueError("weight must have shape [batch, N, K]")
    batches, n, _ = weight.shape
    mask = weight.ne(0)
    per_row = mask.sum(-1)
    if not torch.all(per_row == per_row[0, 0]):
        raise ValueError("Sputnik batch requires identical nnz per row")
    nnz_per_row = int(per_row[0, 0])
    nonzeros = n * nnz_per_row
    values = weight[mask].reshape(batches, nonzeros).contiguous()
    columns = mask.nonzero(as_tuple=False)[:, 2].reshape(batches, nonzeros)
    if columns.max().item() > 32767:
        raise ValueError("Sputnik FP16 column indices require K <= 32767")
    columns = columns.to(torch.int16).contiguous()
    offsets = torch.arange(
        0, nonzeros + 1, nnz_per_row, device=weight.device, dtype=torch.int32
    ).expand(batches, -1).contiguous()
    row_indices = torch.arange(n, device=weight.device, dtype=torch.int32)
    row_indices = row_indices.expand(batches, -1).contiguous()
    return row_indices, values, offsets, columns, nonzeros


class SlideSparseBatch:
    """cuSPARSELt strided-batch plan for SlideSparse-expanded 16-bit operands."""

    def __init__(self, weight: torch.Tensor, m: int, *, allocate_output: bool = True,
                 compressed_reference: torch.Tensor | None = None,
                 workspace_reference: torch.Tensor | None = None):
        if weight.dtype not in (torch.float16, torch.bfloat16) or not weight.is_cuda or weight.ndim != 3:
            raise ValueError("weight must be CUDA FP16 or BF16 [batch,N,K]")
        self.lib = ctypes.CDLL(str(ROOT / "build/libslidesparse_batch.so"))
        self._bind()
        self.shape = tuple(weight.shape)
        self.m = m
        self.dtype = weight.dtype
        batches, n, k = self.shape
        dtype_code = 0 if weight.dtype == torch.float16 else 1
        self.ctx = self.lib.slidesparse_batch_create(batches, m, n, k, dtype_code)
        if not self.ctx:
            self._raise("create")
        compressed_bytes = self.lib.slidesparse_batch_compressed_size(self.ctx)
        compress_ws_bytes = self.lib.slidesparse_batch_compress_workspace_size(self.ctx)
        matmul_ws_bytes = self.lib.slidesparse_batch_matmul_workspace_size(self.ctx)
        # Initialize padding so bytewise layout validation is deterministic.
        self.compressed = torch.zeros(compressed_bytes, device=weight.device, dtype=torch.uint8)
        compress_ws = torch.empty(compress_ws_bytes, device="cuda", dtype=torch.uint8)
        if workspace_reference is not None:
            if workspace_reference.device != weight.device or workspace_reference.numel() < matmul_ws_bytes:
                self.close()
                raise ValueError("Shared cuSPARSELt workspace is too small or on another device")
            self.workspace = workspace_reference
        else:
            self.workspace = torch.empty(matmul_ws_bytes, device=weight.device, dtype=torch.uint8)
        status = self.lib.slidesparse_batch_compress(
            self.ctx, weight.data_ptr(), self.compressed.data_ptr(),
            compress_ws.data_ptr() if compress_ws_bytes else 0, self._stream()
        )
        if status:
            self._raise("compress")
        torch.cuda.current_stream().synchronize()
        if compressed_reference is not None:
            if not torch.equal(self.compressed, compressed_reference):
                self.close()
                raise ValueError("cuSPARSELt compressed layout differs across M plans")
            self.compressed = compressed_reference
        self.output = (
            torch.empty((batches, m, n), device=weight.device, dtype=self.dtype)
            if allocate_output else None
        )

    def _bind(self):
        lib = self.lib
        lib.slidesparse_batch_create.argtypes = [ctypes.c_int] * 5
        lib.slidesparse_batch_create.restype = ctypes.c_void_p
        for name in ("compressed_size", "compress_workspace_size", "matmul_workspace_size"):
            fn = getattr(lib, f"slidesparse_batch_{name}")
            fn.argtypes = [ctypes.c_void_p]
            fn.restype = ctypes.c_size_t
        lib.slidesparse_batch_compress.argtypes = [ctypes.c_void_p] * 5
        lib.slidesparse_batch_compress.restype = ctypes.c_int
        lib.slidesparse_batch_matmul.argtypes = [ctypes.c_void_p] * 6
        lib.slidesparse_batch_matmul.restype = ctypes.c_int
        lib.slidesparse_batch_destroy.argtypes = [ctypes.c_void_p]
        lib.slidesparse_batch_last_error.restype = ctypes.c_char_p

    @staticmethod
    def _stream():
        return torch.cuda.current_stream().cuda_stream

    def _raise(self, operation):
        message = self.lib.slidesparse_batch_last_error().decode()
        raise RuntimeError(f"cuSPARSELt {operation} failed: {message}")

    def __call__(self, activation: torch.Tensor) -> torch.Tensor:
        if tuple(activation.shape) != (self.shape[0], self.m, self.shape[2]):
            raise ValueError("activation shape does not match the cached batch plan")
        if activation.dtype != self.dtype or not activation.is_cuda or not activation.is_contiguous():
            raise ValueError(f"activation must be contiguous CUDA {self.dtype}")
        output = self.output
        if output is None:
            output = torch.empty(
                (self.shape[0], self.m, self.shape[1]),
                device=activation.device, dtype=self.dtype,
            )
        status = self.lib.slidesparse_batch_matmul(
            self.ctx, self.compressed.data_ptr(), activation.data_ptr(),
            output.data_ptr(), self.workspace.data_ptr() if self.workspace.numel() else 0,
            self._stream(),
        )
        if status:
            self._raise("matmul")
        return output

    def close(self):
        if getattr(self, "ctx", None):
            self.lib.slidesparse_batch_destroy(self.ctx)
            self.ctx = None

    def __del__(self):
        self.close()


class SputnikBatch:
    """Single-launch fixed-shape batch adaptation of Sputnik FP16 SpMM."""

    def __init__(self, weight: torch.Tensor, m: int):
        if weight.dtype != torch.float16 or not weight.is_cuda:
            raise ValueError("weight must be CUDA FP16")
        self.lib = ctypes.CDLL(str(ROOT / "build/libsputnik_batch.so"))
        self.lib.sputnik_spmm_batch_fp16.argtypes = [ctypes.c_int] * 5 + [ctypes.c_void_p] * 7
        self.lib.sputnik_spmm_batch_fp16.restype = ctypes.c_int
        self.row_indices, self.values, self.row_offsets, self.columns, self.nonzeros = (
            dense_to_fixed_csr(weight)
        )
        self.shape = tuple(weight.shape)
        self.m = m
        self.output_nm = torch.empty(
            (self.shape[0], self.shape[1], m), device="cuda", dtype=torch.float16
        )

    def prepare_activation(self, activation: torch.Tensor) -> torch.Tensor:
        expected = (self.shape[0], self.m, self.shape[2])
        if tuple(activation.shape) != expected:
            raise ValueError(f"expected activation {expected}, got {tuple(activation.shape)}")
        return activation.transpose(1, 2).contiguous()

    def run_prepared(self, activation_km: torch.Tensor) -> torch.Tensor:
        expected = (self.shape[0], self.shape[2], self.m)
        if tuple(activation_km.shape) != expected or not activation_km.is_contiguous():
            raise ValueError(f"expected contiguous prepared activation {expected}")
        if activation_km.dtype != torch.float16 or not activation_km.is_cuda:
            raise ValueError("prepared activation must be CUDA FP16")
        status = self.lib.sputnik_spmm_batch_fp16(
            self.shape[0], self.shape[1], self.shape[2], self.m, self.nonzeros,
            self.row_indices.data_ptr(), self.values.data_ptr(), self.row_offsets.data_ptr(),
            self.columns.data_ptr(), activation_km.data_ptr(), self.output_nm.data_ptr(),
            torch.cuda.current_stream().cuda_stream,
        )
        if status:
            raise RuntimeError(f"Sputnik batched SpMM launch failed with CUDA error {status}")
        return self.output_nm.transpose(1, 2)

    def __call__(self, activation: torch.Tensor) -> torch.Tensor:
        return self.run_prepared(self.prepare_activation(activation))
