"""BF16 MoE baseline using the native cuBLAS Grouped GEMM API."""

import ctypes as ct
import os

import torch


class CublasGrouped:
    """Compute X[e, :M_e] @ W[e].T without computing padded rows.

    Host dimension arrays and device pointer arrays are built once, outside
    timing. Each active expert is one group containing one GEMM. The API
    uses column-major matrices, so compute Y.T = W @ X.T without copies.
    """

    def __init__(self, activation, weight, counts, out):
        tensors = (activation, weight, out)
        if any(t.ndim != 3 or not t.is_cuda or t.dtype != torch.bfloat16
               or not t.is_contiguous() for t in tensors):
            raise ValueError("Expected contiguous CUDA BF16 rank-three tensors")
        if any(t.device != activation.device for t in tensors):
            raise ValueError("All tensors must be on the same device")
        e, capacity, k = activation.shape
        if weight.shape[0] != e or weight.shape[2] != k:
            raise ValueError("Incompatible weight shape")
        n = weight.shape[1]
        if out.shape != (e, capacity, n):
            raise ValueError("Incompatible output shape")
        rows = counts.tolist()
        if len(rows) != e or any(m < 0 or m > capacity for m in rows):
            raise ValueError("Invalid expert row counts")
        active = [i for i, m in enumerate(rows) if m]
        self.activation, self.weight, self.out = tensors
        self.device = activation.device
        self.handle = ct.c_void_p()
        self.lib = ct.CDLL(os.environ.get("CUBLAS_LIBRARY", "libcublas.so.12"))
        p, i = ct.c_void_p, ct.c_int
        self.lib.cublasCreate_v2.argtypes = [ct.POINTER(p)]
        self.lib.cublasDestroy_v2.argtypes = [p]
        self.lib.cublasSetStream_v2.argtypes = [p, p]
        self.lib.cublasGetVersion_v2.argtypes = [p, ct.POINTER(i)]
        self.lib.cublasGemmGroupedBatchedEx.argtypes = [
            p, p, p, p, p, p, p, p, i, p, p, i, p, p, p, i, p, i, p, i,
        ]
        with torch.cuda.device(self.device):
            self._check(self.lib.cublasCreate_v2(ct.byref(self.handle)))
        version = i()
        self._check(self.lib.cublasGetVersion_v2(self.handle, ct.byref(version)))
        self.version = version.value
        self.group_count = len(active)
        size = self.group_count
        self.transa = (i * size)(*[1] * size)
        self.transb = (i * size)(*[0] * size)
        self.m = (i * size)(*[n] * size)
        self.n = (i * size)(*[rows[e] for e in active])
        self.k = (i * size)(*[k] * size)
        self.lda = (i * size)(*[k] * size)
        self.ldb = (i * size)(*[k] * size)
        self.ldc = (i * size)(*[n] * size)
        self.group_size = (i * size)(*[1] * size)
        self.alpha = (ct.c_float * size)(*[1.0] * size)
        self.beta = (ct.c_float * size)(*[0.0] * size)
        self.pointers = [
            torch.tensor([t[e].data_ptr() for e in active],
                         dtype=torch.int64, device=self.device)
            for t in (weight, activation, out)
        ]

    @staticmethod
    def _check(status):
        if status:
            raise RuntimeError(f"cuBLAS status {status}")

    def __call__(self):
        if not self.handle.value:
            raise RuntimeError("cuBLAS handle is closed")
        if not self.group_count:
            return self.out
        self._check(self.lib.cublasSetStream_v2(
            self.handle, torch.cuda.current_stream(self.device).cuda_stream))
        # CUDA_R_16BF = 14; CUBLAS_COMPUTE_32F = 68 (FP32 accumulation).
        self._check(self.lib.cublasGemmGroupedBatchedEx(
            self.handle, self.transa, self.transb, self.m, self.n, self.k,
            self.alpha, self.pointers[0].data_ptr(), 14, self.lda,
            self.pointers[1].data_ptr(), 14, self.ldb, self.beta,
            self.pointers[2].data_ptr(), 14, self.ldc, self.group_count,
            self.group_size, 68,
        ))
        return self.out

    def close(self):
        if self.handle.value:
            self._check(self.lib.cublasDestroy_v2(self.handle))
            self.handle = ct.c_void_p()

    def __del__(self):
        if getattr(self, "handle", None) and self.handle.value:
            self.lib.cublasDestroy_v2(self.handle)
