#include <cuda_runtime.h>
#include <cusparseLt.h>

#include <cstdint>
#include <cstdio>
#include <new>

namespace {

thread_local char last_error[512] = "";

struct Context {
  cusparseLtHandle_t handle{};
  cusparseLtMatDescriptor_t weight{}, activation{}, output{};
  cusparseLtMatmulDescriptor_t matmul{};
  cusparseLtMatmulAlgSelection_t algorithm{};
  cusparseLtMatmulPlan_t plan{};
  size_t compressed_size = 0;
  size_t compress_workspace_size = 0;
  size_t matmul_workspace_size = 0;
};

bool check(cusparseStatus_t status, const char* call) {
  if (status == CUSPARSE_STATUS_SUCCESS) return true;
  std::snprintf(last_error, sizeof(last_error), "%s failed: %d", call,
                static_cast<int>(status));
  return false;
}

#define CHECK_OR_RETURN(call, value) \
  do { if (!check((call), #call)) return (value); } while (0)

}  // namespace

extern "C" {

const char* slidesparse_batch_last_error() { return last_error; }

void* slidesparse_batch_create(int batches, int m, int n, int k, int dtype_code) {
  last_error[0] = '\0';
  if (batches <= 0 || m <= 0 || n <= 0 || k <= 0 || n % 8 || k % 16 ||
      (dtype_code != 0 && dtype_code != 1)) {
    std::snprintf(last_error, sizeof(last_error),
                  "invalid configuration: batches=%d M=%d N=%d K=%d dtype=%d",
                  batches, m, n, k, dtype_code);
    return nullptr;
  }
  Context* ctx = new (std::nothrow) Context;
  if (!ctx) return nullptr;
  int32_t count = batches;
  int64_t weight_stride = static_cast<int64_t>(n) * k;
  int64_t activation_stride = static_cast<int64_t>(m) * k;
  int64_t output_stride = static_cast<int64_t>(m) * n;
  cudaDataType_t dtype = dtype_code == 0 ? CUDA_R_16F : CUDA_R_16BF;
  if (!check(cusparseLtInit(&ctx->handle), "cusparseLtInit")) goto fail;

  if (!check(cusparseLtStructuredDescriptorInit(
          &ctx->handle, &ctx->weight, k, n, k, 16, dtype,
          CUSPARSE_ORDER_COL, CUSPARSELT_SPARSITY_50_PERCENT),
          "cusparseLtStructuredDescriptorInit")) goto fail;
  if (!check(cusparseLtDenseDescriptorInit(
          &ctx->handle, &ctx->activation, k, m, k, 16, dtype,
          CUSPARSE_ORDER_COL), "cusparseLtDenseDescriptorInit(A)")) goto fail;
  if (!check(cusparseLtDenseDescriptorInit(
          &ctx->handle, &ctx->output, n, m, n, 16, dtype,
          CUSPARSE_ORDER_COL), "cusparseLtDenseDescriptorInit(D)")) goto fail;

  if (!check(cusparseLtMatDescSetAttribute(
          &ctx->handle, &ctx->weight, CUSPARSELT_MAT_NUM_BATCHES,
          &count, sizeof(count)), "set weight batches")) goto fail;
  if (!check(cusparseLtMatDescSetAttribute(
          &ctx->handle, &ctx->weight, CUSPARSELT_MAT_BATCH_STRIDE,
          &weight_stride, sizeof(weight_stride)), "set weight stride")) goto fail;
  if (!check(cusparseLtMatDescSetAttribute(
          &ctx->handle, &ctx->activation, CUSPARSELT_MAT_NUM_BATCHES,
          &count, sizeof(count)), "set activation batches")) goto fail;
  if (!check(cusparseLtMatDescSetAttribute(
          &ctx->handle, &ctx->activation, CUSPARSELT_MAT_BATCH_STRIDE,
          &activation_stride, sizeof(activation_stride)), "set activation stride")) goto fail;
  if (!check(cusparseLtMatDescSetAttribute(
          &ctx->handle, &ctx->output, CUSPARSELT_MAT_NUM_BATCHES,
          &count, sizeof(count)), "set output batches")) goto fail;
  if (!check(cusparseLtMatDescSetAttribute(
          &ctx->handle, &ctx->output, CUSPARSELT_MAT_BATCH_STRIDE,
          &output_stride, sizeof(output_stride)), "set output stride")) goto fail;

  if (!check(cusparseLtMatmulDescriptorInit(
          &ctx->handle, &ctx->matmul, CUSPARSE_OPERATION_TRANSPOSE,
          CUSPARSE_OPERATION_NON_TRANSPOSE, &ctx->weight, &ctx->activation,
          &ctx->output, &ctx->output, CUSPARSE_COMPUTE_32F),
          "cusparseLtMatmulDescriptorInit")) goto fail;
  if (!check(cusparseLtMatmulAlgSelectionInit(
          &ctx->handle, &ctx->algorithm, &ctx->matmul,
          CUSPARSELT_MATMUL_ALG_DEFAULT),
          "cusparseLtMatmulAlgSelectionInit")) goto fail;
  if (!check(cusparseLtMatmulPlanInit(
          &ctx->handle, &ctx->plan, &ctx->matmul, &ctx->algorithm),
          "cusparseLtMatmulPlanInit")) goto fail;
  if (!check(cusparseLtSpMMACompressedSize(
          &ctx->handle, &ctx->plan, &ctx->compressed_size,
          &ctx->compress_workspace_size), "cusparseLtSpMMACompressedSize")) goto fail;
  if (!check(cusparseLtMatmulGetWorkspace(
          &ctx->handle, &ctx->plan, &ctx->matmul_workspace_size),
          "cusparseLtMatmulGetWorkspace")) goto fail;
  return ctx;

fail:
  delete ctx;
  return nullptr;
}

size_t slidesparse_batch_compressed_size(void* opaque) {
  return static_cast<Context*>(opaque)->compressed_size;
}
size_t slidesparse_batch_compress_workspace_size(void* opaque) {
  return static_cast<Context*>(opaque)->compress_workspace_size;
}
size_t slidesparse_batch_matmul_workspace_size(void* opaque) {
  return static_cast<Context*>(opaque)->matmul_workspace_size;
}

int slidesparse_batch_compress(void* opaque, const void* weight, void* compressed,
                               void* workspace, cudaStream_t stream) {
  Context* ctx = static_cast<Context*>(opaque);
  return check(cusparseLtSpMMACompress(&ctx->handle, &ctx->plan, weight,
                                      compressed, workspace, stream),
               "cusparseLtSpMMACompress") ? 0 : -1;
}

int slidesparse_batch_matmul(void* opaque, const void* compressed,
                             const void* activation, void* output,
                             void* workspace, cudaStream_t stream) {
  Context* ctx = static_cast<Context*>(opaque);
  float alpha = 1.0f, beta = 0.0f;
  return check(cusparseLtMatmul(&ctx->handle, &ctx->plan, &alpha, compressed,
                               activation, &beta, output, output, workspace,
                               &stream, 1), "cusparseLtMatmul") ? 0 : -1;
}

void slidesparse_batch_destroy(void* opaque) {
  Context* ctx = static_cast<Context*>(opaque);
  if (!ctx) return;
  cusparseLtMatmulPlanDestroy(&ctx->plan);
  cusparseLtMatmulAlgSelectionDestroy(&ctx->algorithm);
  cusparseLtMatDescriptorDestroy(&ctx->output);
  cusparseLtMatDescriptorDestroy(&ctx->activation);
  cusparseLtMatDescriptorDestroy(&ctx->weight);
  cusparseLtDestroy(&ctx->handle);
  delete ctx;
}

}  // extern "C"
