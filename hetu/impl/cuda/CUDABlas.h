#pragma once

#include "hetu/impl/utils/cuda_utils.h"
#include <cublas_v2.h>

namespace hetu {
namespace impl {

cublasHandle_t GetCublasHandle(int32_t device_id);

template <typename T>
inline void cublas_dot(cublasHandle_t cublas_handle, int32_t n, const T* x,
                       int32_t incx, const T* y, int32_t incy, T* result) {
  HT_NOT_IMPLEMENTED << "cublas_dot is not implemented for type "
                     << typeid(T).name();
}

template <>
void cublas_dot<float16>(cublasHandle_t cublas_handle, int32_t n, const float16* x,
                         int32_t incx, const float16* y, int32_t incy, float16* result);

template <>
void cublas_dot<bfloat16>(cublasHandle_t cublas_handle, int32_t n, const bfloat16* x,
                          int32_t incx, const bfloat16* y, int32_t incy, bfloat16* result);

template <>
void cublas_dot<float>(cublasHandle_t cublas_handle, int32_t n, const float* x,
                       int32_t incx, const float* y, int32_t incy, float* result);

template <>
void cublas_dot<double>(cublasHandle_t cublas_handle, int32_t n, const double* x,
                       int32_t incx, const double* y, int32_t incy, double* result);

template <typename T>
inline void cublas_gemv(cublasHandle_t cublas_handle, cublasOperation_t trans,
                        int32_t m, int32_t n, const void* alpha, const T* A,
                        int32_t lda, const T* x, int32_t incx, const void* beta,
                        T* y, int32_t incy) {
  HT_NOT_IMPLEMENTED << "cublas_gemv is not implemented for type "
                     << typeid(T).name();
}

template <>
void cublas_gemv<float16>(cublasHandle_t cublas_handle, cublasOperation_t trans,
                          int32_t m, int32_t n, const void* alpha, const float16* A,
                          int32_t lda, const float16* x, int32_t incx,
                          const void* beta, float16* y, int32_t incy);

template <>
void cublas_gemv<bfloat16>(cublasHandle_t cublas_handle, cublasOperation_t trans,
                          int32_t m, int32_t n, const void* alpha, const bfloat16* A,
                          int32_t lda, const bfloat16* x, int32_t incx,
                          const void* beta, bfloat16* y, int32_t incy);

template <>
void cublas_gemv<float>(cublasHandle_t cublas_handle, cublasOperation_t trans,
                        int32_t m, int32_t n, const void* alpha, const float* A,
                        int32_t lda, const float* x, int32_t incx,
                        const void* beta, float* y, int32_t incy);

template <>
void cublas_gemv<double>(cublasHandle_t cublas_handle, cublasOperation_t trans,
                        int32_t m, int32_t n, const void* alpha, const double* A,
                        int32_t lda, const double* x, int32_t incx,
                        const void* beta, double* y, int32_t incy);


template <typename T>
inline void cublas_gemm(cublasHandle_t cublas_handle, cublasOperation_t trans_a,
                        cublasOperation_t trans_b, int32_t m, int32_t n,
                        int32_t k, const void* alpha, const T* A, int32_t lda,
                        const T* B, int32_t ldb, const void* beta, T* C,
                        int32_t ldc) {
  HT_NOT_IMPLEMENTED << "cublas_gemm is not implemented for type "
                     << typeid(T).name();
}

template <>
void cublas_gemm<float16>(cublasHandle_t cublas_handle, cublasOperation_t trans_a,
                        cublasOperation_t trans_b, int32_t m, int32_t n,
                        int32_t k, const void* alpha, const float16* A,
                        int32_t lda, const float16* B, int32_t ldb,
                        const void* beta, float16* C, int32_t ldc);

template <>
void cublas_gemm<bfloat16>(cublasHandle_t cublas_handle, cublasOperation_t trans_a,
                           cublasOperation_t trans_b, int32_t m, int32_t n,
                           int32_t k, const void* alpha, const bfloat16* A,
                           int32_t lda, const bfloat16* B, int32_t ldb,
                           const void* beta, bfloat16* C, int32_t ldc);

template <>
void cublas_gemm<float>(cublasHandle_t cublas_handle, cublasOperation_t trans_a,
                        cublasOperation_t trans_b, int32_t m, int32_t n,
                        int32_t k, const void* alpha, const float* A,
                        int32_t lda, const float* B, int32_t ldb,
                        const void* beta, float* C, int32_t ldc);

template <>
void cublas_gemm<double>(cublasHandle_t cublas_handle,
                         cublasOperation_t trans_a, cublasOperation_t trans_b,
                         int32_t m, int32_t n, int32_t k, const void* alpha,
                         const double* A, int32_t lda, const double* B,
                         int32_t ldb, const void* beta, double* C,
                         int32_t ldc);

template <typename T>
inline void
cublas_batch_gemm(cublasHandle_t cublas_handle, cublasOperation_t trans_a,
                  cublasOperation_t trans_b, int32_t m, int32_t n, int32_t k,
                  const void* alpha, const T* A, int32_t lda, int32_t strideA,
                  const T* B, int32_t ldb, int32_t strideB, const void* beta, T* C,
                  int32_t ldc, int32_t strideC, int32_t batch_count) {
  HT_NOT_IMPLEMENTED << "cublas_gemm is not implemented for type "
                     << typeid(T).name();
}

template <>
void cublas_batch_gemm<float16>(cublasHandle_t cublas_handle,
                              cublasOperation_t trans_a,
                              cublasOperation_t trans_b, int32_t m, int32_t n,
                              int32_t k, const void* alpha, const float16* A,
                              int32_t lda, int32_t strideA, const float16* B,
                              int32_t ldb, int32_t strideB, const void* beta,
                              float16* C, int32_t ldc, int32_t strideC,
                              int32_t batch_count);

template <>
void cublas_batch_gemm<bfloat16>(cublasHandle_t cublas_handle,
                               cublasOperation_t trans_a,
                               cublasOperation_t trans_b, int32_t m, int32_t n,
                               int32_t k, const void* alpha, const bfloat16* A,
                               int32_t lda, int32_t strideA, const bfloat16* B,
                               int32_t ldb, int32_t strideB, const void* beta,
                               bfloat16* C, int32_t ldc, int32_t strideC,
                               int32_t batch_count);

template <>
void cublas_batch_gemm<float>(cublasHandle_t cublas_handle,
                              cublasOperation_t trans_a,
                              cublasOperation_t trans_b, int32_t m, int32_t n,
                              int32_t k, const void* alpha, const float* A,
                              int32_t lda, int32_t strideA, const float* B,
                              int32_t ldb, int32_t strideB, const void* beta,
                              float* C, int32_t ldc, int32_t strideC,
                              int32_t batch_count);

template <>
void cublas_batch_gemm<double>(cublasHandle_t cublas_handle,
                               cublasOperation_t trans_a,
                               cublasOperation_t trans_b, int32_t m, int32_t n,
                               int32_t k, const void* alpha, const double* A,
                               int32_t lda, int32_t strideA, const double* B,
                               int32_t ldb, int32_t strideB, const void* beta,
                               double* C, int32_t ldc, int32_t strideC,
                               int32_t batch_count);

} // namespace impl
} // namespace hetu

namespace {
inline const char* cublasGetErrorString(cublasStatus_t status) {
  switch (status) {
    case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED:
      return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
    case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";
    case CUBLAS_STATUS_LICENSE_ERROR: return "CUBLAS_STATUS_LICENSE_ERROR";
    default: return "CUBLAS_UNKNOWN_ERROR";
  }
}
} // namespace

#define CUBLAS_CALL(f)                                                         \
  for (cublasStatus_t status = (f); status != CUBLAS_STATUS_SUCCESS;           \
       status = CUBLAS_STATUS_SUCCESS)                                         \
  __HT_FATAL_SILENT(hetu::cuda::cuda_error)                                    \
    << "Cublas call " << #f << " failed: " << cublasGetErrorString(status)
