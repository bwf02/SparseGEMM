#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "sputnik/spmm/cuda_spmm.h"

extern "C" int sputnik_spmm_batch_fp16(
    int batches, int n_rows, int k, int m_tokens, int nonzeros,
    const int* row_indices, const void* values, const int* row_offsets,
    const int16_t* column_indices, const void* activation_km, void* output_nm,
    cudaStream_t stream) {
  return static_cast<int>(sputnik::CudaSpmmBatched(
      batches, n_rows, k, m_tokens, nonzeros, row_indices,
      static_cast<const half2*>(values), row_offsets,
      reinterpret_cast<const short2*>(column_indices),
      static_cast<const half2*>(activation_km), static_cast<half2*>(output_nm),
      stream));
}
