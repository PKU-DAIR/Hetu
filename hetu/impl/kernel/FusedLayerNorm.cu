#include "hetu/core/ndarray.h"
#include "hetu/core/memory_pool.h"
#include "hetu/impl/stream/CUDAStream.h"
#include "hetu/impl/cuda/CUDADnn.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/cuda_utils.h"
#include "hetu/impl/kernel/Binary.cuh"
#include "hetu/impl/utils/cuda_math.h"
#include "hetu/impl/utils/offset_calculator.cuh"

namespace hetu {
namespace impl {

template <typename T>
__device__ __forceinline__ T WARP_SHFL(T value, int srcLane, int width = warpSize, unsigned int mask = 0xffffffff)
{
#if !defined(USE_ROCM)
    return __shfl_sync(mask, value, srcLane, width);
#else
    return __shfl(value, srcLane, width);
#endif
}

template <typename T>
__device__ __forceinline__ T WARP_SHFL_XOR(T value, int laneMask, int width = warpSize, unsigned int mask = 0xffffffff)
{
#if !defined(USE_ROCM)
    return __shfl_xor_sync(mask, value, laneMask, width);
#else
    return __shfl_xor(value, laneMask, width);
#endif
}

template<typename U> __device__
void cuWelfordOnlineSum(
  const U curr,
  U& mu,
  U& sigma2,
  U& count)
{
  count = count + U(1);
  U delta = curr - mu;
  U lmean = mu + delta / count;
  mu = lmean;
  U delta2 = curr - lmean;
  sigma2 = sigma2 + delta * delta2;
}

template<typename U> __device__
void cuChanOnlineSum(
  const U muB,
  const U sigma2B,
  const U countB,
  U& mu,
  U& sigma2,
  U& count)
{
  U delta = muB - mu;
  U nA = count;
  U nB = countB;
  count = count + countB;
  U nX = count;
  if (nX > U(0)) {
    nA = nA / nX;
    nB = nB / nX;
    mu = nA*mu + nB*muB;
    sigma2 = sigma2 + sigma2B + delta * delta * nA * nB * nX;
  } else {
    mu = U(0);
    sigma2 = U(0);
  }
}

template<typename U> __device__
void cuRMSOnlineSum(
  const U curr,
  U& sigma2)
{
  sigma2 = sigma2 + curr * curr;
}

template<typename U> __device__
void cuChanRMSOnlineSum(
  const U sigma2B,
  U& sigma2)
{
  sigma2 = sigma2 + sigma2B;
}


template<typename T, typename U> __device__
void cuWelfordMuSigma2(
  const T* __restrict__ vals,
  const int n1,
  const int n2,
  const int i1,
  U& mu,
  U& sigma2,
  U* buf,
  bool rms_only)
{
  // Assumptions:
  // 1) blockDim.x == warpSize
  // 2) Tensor is contiguous
  // 3) 2*blockDim.y*sizeof(U)+blockDim.y*sizeof(int) shared memory available.
  //
  // compute variance and mean over n2
  U count = U(0);
  mu= U(0);
  sigma2 = U(0);
  if (i1 < n1) {
    // one warp normalizes one n1 index,
    // synchronization is implicit
    // initialize with standard Welford algorithm
    const int numx = blockDim.x * blockDim.y;
    const int thrx = threadIdx.x + threadIdx.y * blockDim.x;
    const T* lvals = vals + i1*n2;
    int l = 4*thrx;
    for (;  l+3 < n2;  l+=4*numx) {
      for (int k = 0;  k < 4;  ++k) {
        U curr = static_cast<U>(lvals[l+k]);
        if (!rms_only) {
          cuWelfordOnlineSum<U>(curr,mu,sigma2,count);
        } else {
          cuRMSOnlineSum<U>(curr, sigma2);
        }
      }
    }
    for (;  l < n2;  ++l) {
      U curr = static_cast<U>(lvals[l]);
      if (!rms_only) {
        cuWelfordOnlineSum<U>(curr,mu,sigma2,count);
      } else {
       cuRMSOnlineSum<U>(curr, sigma2);
      }
    }
    // intra-warp reductions
    for (int l = 0;  l <= 4;  ++l) {
      int srcLaneB = (threadIdx.x+(1<<l))&31;
      U sigma2B = WARP_SHFL(sigma2, srcLaneB);
      if (!rms_only) {
        U muB = WARP_SHFL(mu, srcLaneB);
        U countB = WARP_SHFL(count, srcLaneB);
        cuChanOnlineSum<U>(muB,sigma2B,countB,mu,sigma2,count);
      } else {
        cuChanRMSOnlineSum<U>(sigma2B, sigma2);
      }
    }
    // threadIdx.x == 0 has correct values for each warp
    // inter-warp reductions
    if (blockDim.y > 1) {
      U* ubuf = (U*)buf;
      U* ibuf = (U*)(ubuf + blockDim.y);
      for (int offset = blockDim.y/2;  offset > 0;  offset /= 2) {
        // upper half of warps write to shared
        if (threadIdx.x == 0 && threadIdx.y >= offset && threadIdx.y < 2*offset) {
          const int wrt_y = threadIdx.y - offset;
          if (!rms_only) {
            ubuf[2*wrt_y] = mu;
            ibuf[wrt_y] = count;
          }
          ubuf[2*wrt_y+1] = sigma2;
        }
        __syncthreads();
        // lower half merges
        if (threadIdx.x == 0 && threadIdx.y < offset) {
          U sigma2B = ubuf[2*threadIdx.y+1];
          if (!rms_only) {
            U muB = ubuf[2*threadIdx.y];
            U countB = ibuf[threadIdx.y];
            cuChanOnlineSum<U>(muB,sigma2B,countB,mu,sigma2,count);
          } else {
            cuChanRMSOnlineSum<U>(sigma2B,sigma2);
          }
        }
        __syncthreads();
      }
      // threadIdx.x = 0 && threadIdx.y == 0 only thread that has correct values
      if (threadIdx.x == 0 && threadIdx.y == 0) {
        if (!rms_only) {
          ubuf[0] = mu;
        }
        ubuf[1] = sigma2;
      }
      __syncthreads();
      if (!rms_only) {
        mu = ubuf[0];
      }
      sigma2 = ubuf[1]/U(n2);
      // don't care about final value of count, we know count == n2
    } else {
      if (!rms_only) {
        mu = WARP_SHFL(mu, 0);
      }
      sigma2 = WARP_SHFL(sigma2/U(n2), 0);
    }
  }
}

template<> __device__
void cuWelfordMuSigma2(
  const float16* __restrict__ vals,
  const int n1,
  const int n2,
  const int i1,
  float& mu,
  float& sigma2,
  float* buf,
  bool rms_only)
{
  // Assumptions:
  // 1) blockDim.x == warpSize
  // 2) Tensor is contiguous
  // 3) 2*blockDim.y*sizeof(U)+blockDim.y*sizeof(int) shared memory available.
  //
  // compute variance and mean over n2
  float count = 0.0f;
  mu= float(0);
  sigma2 = float(0);
  if (i1 < n1) {
    // one warp normalizes one n1 index,
    // synchronization is implicit
    // initialize with standard Welford algorithm
    const int numx = blockDim.x * blockDim.y;
    const int thrx = threadIdx.x + threadIdx.y * blockDim.x;
    const float16* lvals = vals + i1*n2;
    int l = 8*thrx;
    if ((((size_t)lvals)&3) != 0) {
      // 16 bit alignment
      // first thread consumes first point
      if (thrx == 0) {
        float curr = static_cast<float>(lvals[0]);
        if (!rms_only) {
          cuWelfordOnlineSum(curr,mu,sigma2,count);
        } else {
          cuRMSOnlineSum(curr, sigma2);
        }

      }
      ++l;
    }
    // at this point, lvals[l] are 32 bit aligned for all threads.
    for (;  l+7 < n2;  l+=8*numx) {
      for (int k = 0;  k < 8;  k+=2) {
        float2 curr = __half22float2(*((__half2*)(lvals+l+k)));
        if (!rms_only) {
          cuWelfordOnlineSum(curr.x,mu,sigma2,count);
          cuWelfordOnlineSum(curr.y,mu,sigma2,count);
        } else {
          cuRMSOnlineSum(curr.x, sigma2);
          cuRMSOnlineSum(curr.y, sigma2);
        }
      }
    }
    for (;  l < n2;  ++l) {
      float curr = static_cast<float>(lvals[l]);
      if (!rms_only) {
        cuWelfordOnlineSum(curr,mu,sigma2,count);
      } else {
        cuRMSOnlineSum(curr, sigma2);
      }
    }
    // intra-warp reductions
    for (int l = 0;  l <= 4;  ++l) {
      int srcLaneB = (threadIdx.x+(1<<l))&31;
      float sigma2B = WARP_SHFL(sigma2, srcLaneB);
      if (!rms_only) {
        float muB = WARP_SHFL(mu, srcLaneB);
        float countB = WARP_SHFL(count, srcLaneB);
        cuChanOnlineSum(muB,sigma2B,countB,mu,sigma2,count);
      } else {
        cuChanRMSOnlineSum(sigma2B, sigma2);
      }
    }
    // threadIdx.x == 0 has correct values for each warp
    // inter-warp reductions
    if (blockDim.y > 1) {
      float* ubuf = (float*)buf;
      float* ibuf = (float*)(ubuf + blockDim.y);
      for (int offset = blockDim.y/2;  offset > 0;  offset /= 2) {
        // upper half of warps write to shared
        if (threadIdx.x == 0 && threadIdx.y >= offset && threadIdx.y < 2*offset) {
          const int wrt_y = threadIdx.y - offset;
          ubuf[2*wrt_y+1] = sigma2;
          if (!rms_only) {
            ubuf[2*wrt_y] = mu;
            ibuf[wrt_y] = count;
          }
        }
        __syncthreads();
        // lower half merges
        if (threadIdx.x == 0 && threadIdx.y < offset) {
          float sigma2B = ubuf[2*threadIdx.y+1];
          if (!rms_only) {
            float muB = ubuf[2*threadIdx.y];
            float countB = ibuf[threadIdx.y];
            cuChanOnlineSum(muB,sigma2B,countB,mu,sigma2,count);
          } else {
            cuChanRMSOnlineSum(sigma2B, sigma2);
          }
        }
        __syncthreads();
      }
      // threadIdx.x = 0 && threadIdx.y == 0 only thread that has correct values
      if (threadIdx.x == 0 && threadIdx.y == 0) {
        if (!rms_only) {
          ubuf[0] = mu;
        }
        ubuf[1] = sigma2;
      }
      __syncthreads();
      if (!rms_only) {
        mu = ubuf[0];
      }
      sigma2 = ubuf[1]/float(n2);
      // don't care about final value of count, we know count == n2
    } else {
      if (!rms_only) {
        mu = WARP_SHFL(mu, 0);
      }
      sigma2 = WARP_SHFL(sigma2/float(n2), 0);
    }
  }
}

template<typename U> __device__ U rsqrt(U v) {
  return U(1) / hetu::cuda::cuda_sqrt(v);
}
template<> __device__ float rsqrt(float v) {
  return rsqrtf(v);
}
template<> __device__ double rsqrt(double v) {
  return rsqrt(v);
}

namespace {
// This is the un-specialized struct.  Note that we prevent instantiation of this
// struct by putting an undefined symbol in the function body so it won't compile.
//  template <typename T>
//  struct SharedMemory
//  {
//      // Ensure that we won't compile any un-specialized types
//      __device__ T *getPointer()
//      {
//          extern __device__ void error(void);
//          error();
//          return NULL;
//      }
//  };
// https://github.com/NVIDIA/apex/issues/246
template <typename T>
struct SharedMemory;

template <>
struct SharedMemory <float16>
{
    __device__ float16 *getPointer()
    {
        extern __shared__ __half s_float16[];
        return (float16*)s_float16;
    }
};

template <>
struct SharedMemory <bfloat16>
{
    __device__ bfloat16 *getPointer()
    {
        extern __shared__ __nv_bfloat16 s_bfloat16[];
        return (bfloat16*)s_bfloat16;
    }
};

template <>
struct SharedMemory <float>
{
    __device__ float *getPointer()
    {
        extern __shared__ float s_float[];
        return s_float;
    }
};

template <>
struct SharedMemory <double>
{
    __device__ double *getPointer()
    {
        extern __shared__ double s_double[];
        return s_double;
    }
};
}

template<typename T, typename U, typename V> __device__
void cuApplyLayerNorm_(
  V* __restrict__ output_vals,
  U* __restrict__ mean,
  U* __restrict__ invvar,
  const T* __restrict__ vals,
  const int n1,
  const int n2,
  const U epsilon,
  const V* __restrict__ gamma,
  const V* __restrict__ beta,
  bool rms_only
  )
{
  // Assumptions:
  // 1) blockDim.x == warpSize
  // 2) Tensors are contiguous
  //
  for (auto i1=blockIdx.y; i1 < n1; i1 += gridDim.y) {
    SharedMemory<U> shared;
    U* buf = shared.getPointer();
    U mu,sigma2;
    cuWelfordMuSigma2(vals,n1,n2,i1,mu,sigma2,buf,rms_only);

    const T* lvals = vals + i1*n2;
    V* ovals = output_vals + i1*n2;
    U c_invvar = rsqrt(sigma2 + epsilon);
    const int numx = blockDim.x * blockDim.y;
    const int thrx = threadIdx.x + threadIdx.y * blockDim.x;
    if (gamma != NULL && (beta != NULL || rms_only)) {
      for (int i = thrx;  i < n2;  i+=numx) {
        U curr = static_cast<U>(lvals[i]);
        if (!rms_only) {
          ovals[i] = gamma[i] * static_cast<V>(c_invvar * (curr - mu)) + beta[i];
        } else {
          ovals[i] = gamma[i] * static_cast<V>(c_invvar * curr);
        }

      }
    } 
    else {
      for (int i = thrx;  i < n2;  i+=numx) {
        U curr = static_cast<U>(lvals[i]);
        if (!rms_only) {
          ovals[i] = static_cast<V>(c_invvar * (curr - mu));
        } else {
          ovals[i] = static_cast<V>(c_invvar * curr);
        }
      }
    }
    if (threadIdx.x == 0 && threadIdx.y == 0) {
      if (!rms_only) {
        mean[i1] = mu;
      }
      invvar[i1] = c_invvar;
    }
    __syncthreads();
  }
}

template<typename T, typename U, typename V=T> __global__
void cuApplyLayerNorm(
  V* __restrict__ output_vals,
  U* __restrict__ mean,
  U* __restrict__ invvar,
  const T* __restrict__ vals,
  const int n1,
  const int n2,
  const U epsilon,
  const V* __restrict__ gamma,
  const V* __restrict__ beta
  )
{
  cuApplyLayerNorm_<T, U, V>(output_vals, mean, invvar, vals, n1, n2, epsilon, gamma, beta, false);
}

template<typename T, typename U, typename V=T> __global__
void cuApplyRMSNorm(
  V* __restrict__ output_vals,
  U* __restrict__ invvar,
  const T* __restrict__ vals,
  const int n1,
  const int n2,
  const U epsilon,
  const V* __restrict__ gamma)
{
  cuApplyLayerNorm_<T, U, V>(output_vals, NULL, invvar, vals, n1, n2, epsilon, gamma, NULL, true);
}


template<typename V> __device__
V clamp_by_magnitude(V curr_gamma, double eps)
{
  const V kMinGamma = V(eps);
  if (curr_gamma >= 0) {
    if (curr_gamma < kMinGamma) {
      return kMinGamma;
    } else {
      return curr_gamma;
    }
  } else {
    if (curr_gamma > -kMinGamma) {
      return -kMinGamma;
    } else {
      return curr_gamma;
    }
  }
}


template<typename T, typename U, typename V, bool MemoryEfficient> __device__
void cuLoadWriteStridedInputs(
    const int i1_block,
    const int thr_load_row_off,
    const int thr_load_col_off,
    const int i2_off,
    const int row_stride,
    U* warp_buf1,
    U* warp_buf2,
    const T* input_or_output,
    const V* dout,
    const int i1_end,
    const int n2,
    const U* __restrict__ mean,
    const U* __restrict__ invvar,
    const V* __restrict__ gamma,
    const V* __restrict__ beta,
    const double eps,
    bool rms_only
    )
{
  int i1 = i1_block+thr_load_row_off;
  if (i1 < i1_end) {
    for (int k = 0;  k < blockDim.y;  ++k) {
      int i2 = i2_off + k;
      int load_idx = i1*n2+i2;
      int write_idx = thr_load_row_off*row_stride+thr_load_col_off+k;
      if (i2<n2) {
        U c_h = static_cast<U>(input_or_output[load_idx]);
        U curr_dout = static_cast<U>(dout[load_idx]);
        if (!rms_only) {
          warp_buf1[write_idx] = curr_dout;
          if (MemoryEfficient) {
            U curr_beta = static_cast<U>(beta[i2]);
            warp_buf2[write_idx] = curr_dout * (c_h - curr_beta) / static_cast<U>(clamp_by_magnitude(gamma[i2], eps));
          } else {
            warp_buf2[write_idx] = curr_dout * (c_h - mean[i1]) * invvar[i1];
          }
        } else {
          if (MemoryEfficient) {
            warp_buf2[write_idx] = curr_dout * (c_h) / static_cast<U>(clamp_by_magnitude(gamma[i2], eps));
          } else {
            warp_buf2[write_idx] = curr_dout * (c_h) * invvar[i1];
          }
        }
      } else {
        if (!rms_only) {
          warp_buf1[write_idx] = U(0);
        }
        warp_buf2[write_idx] = U(0);
      }
    }
  } else {
    for (int k = 0;  k < blockDim.y;  ++k) {
      int write_idx = thr_load_row_off*row_stride+thr_load_col_off+k;
      if (!rms_only) {
        warp_buf1[write_idx] = U(0);
      }
      warp_buf2[write_idx] = U(0);
    }
  }
}

template<typename T, typename U, typename V, bool MemoryEfficient> __device__
void cuLoadAddStridedInputs(
    const int i1_block,
    const int thr_load_row_off,
    const int thr_load_col_off,
    const int i2_off,
    const int row_stride,
    U* warp_buf1,
    U* warp_buf2,
    const T* input_or_output,
    const V* dout,
    const int i1_end,
    const int n2,
    const U* __restrict__ mean,
    const U* __restrict__ invvar,
    const V* __restrict__ gamma,
    const V* __restrict__ beta,
    const double eps,
    bool rms_only
    )
{
  int i1 = i1_block+thr_load_row_off;
  if (i1 < i1_end) {
    for (int k = 0;  k < blockDim.y;  ++k) {
      int i2 = i2_off + k;
      int load_idx = i1*n2+i2;
      int write_idx = thr_load_row_off*row_stride+thr_load_col_off+k;
      if (i2<n2) {
        U c_h = static_cast<U>(input_or_output[load_idx]);
        U curr_dout = static_cast<U>(dout[load_idx]);
        if (!rms_only) {
          U curr_beta = static_cast<U>(beta[i2]);
          warp_buf1[write_idx] += curr_dout;
          if (MemoryEfficient) {
            warp_buf2[write_idx] += curr_dout * (c_h - curr_beta) / static_cast<U>(clamp_by_magnitude(gamma[i2], eps));
          } else {
            warp_buf2[write_idx] += curr_dout * (c_h - mean[i1]) * invvar[i1];
          }
        } else {
          if (MemoryEfficient) {
            warp_buf2[write_idx] += curr_dout * (c_h) / static_cast<U>(clamp_by_magnitude(gamma[i2], eps));
          } else {
            warp_buf2[write_idx] += curr_dout * (c_h) * invvar[i1];
          }
        }
      }
    }
  }
}


template<typename T, typename U, typename V, bool MemoryEfficient> __global__
void cuComputePartGradGammaBeta(
    const V* __restrict__ dout,
    const T* __restrict__ input_or_output,
    const int n1,
    const int n2,
    const U* __restrict__ mean,
    const U* __restrict__ invvar,
    U epsilon,
    const V* __restrict__ gamma,
    const V* __restrict__ beta,
    U* part_grad_gamma,
    U* part_grad_beta,
    const double eps,
    bool rms_only)
{
    const int numsegs_n1 = (n1+blockDim.y*blockDim.y-1) / (blockDim.y*blockDim.y);
    const int segs_per_block = (numsegs_n1 + gridDim.y - 1) / gridDim.y;
    const int i1_beg = blockIdx.y * segs_per_block * blockDim.y*blockDim.y;
    const int i1_beg_plus_one = (blockIdx.y+1) * segs_per_block * blockDim.y*blockDim.y;
    const int i1_end = i1_beg_plus_one < n1 ? i1_beg_plus_one : n1;
    const int row_stride = blockDim.x+1;
    const int thr_load_col_off = (threadIdx.x*blockDim.y)&(blockDim.x-1);
    const int thr_load_row_off = (threadIdx.x*blockDim.y)/blockDim.x + threadIdx.y*blockDim.y;
    const int i2_off = blockIdx.x * blockDim.x + thr_load_col_off;
    SharedMemory<U> shared;
    U* buf = shared.getPointer(); // buf has at least blockDim.x * blockDim.y * blockDim.y + (blockDim.y - 1)*(blockDim.x/blockDim.y) elements
    U* warp_buf1 = (U*)buf;
    U* warp_buf2 = warp_buf1 + blockDim.y * blockDim.y * row_stride;
    // compute partial sums from strided inputs
    // do this to increase number of loads in flight
    cuLoadWriteStridedInputs<T, U, V, MemoryEfficient>(i1_beg,thr_load_row_off,thr_load_col_off,i2_off,row_stride,warp_buf1,warp_buf2,input_or_output,dout,i1_end,n2,mean,invvar,gamma,beta,eps, rms_only);
    for (int i1_block = i1_beg+blockDim.y*blockDim.y;  i1_block < i1_end;  i1_block+=blockDim.y*blockDim.y) {
      cuLoadAddStridedInputs<T, U, V, MemoryEfficient>(i1_block,thr_load_row_off,thr_load_col_off,i2_off,row_stride,warp_buf1,warp_buf2,input_or_output,dout,i1_end,n2,mean,invvar,gamma,beta,eps, rms_only);
    }
    __syncthreads();
    // inter-warp reductions
    // sum within each warp
    U acc1 = U(0);
    U acc2 = U(0);
    for (int k = 0;  k < blockDim.y;  ++k) {
      int row1 = threadIdx.y + k*blockDim.y;
      int idx1 = row1*row_stride + threadIdx.x;
      if (!rms_only) {
        acc1 += warp_buf1[idx1];
      }
      acc2 += warp_buf2[idx1];
    }
    if (!rms_only) {
      warp_buf1[threadIdx.y*row_stride+threadIdx.x] = acc1;
    }
    warp_buf2[threadIdx.y*row_stride+threadIdx.x] = acc2;
    __syncthreads();
    // sum all warps
    for (int offset = blockDim.y/2;  offset > 1;  offset /= 2) {
      if (threadIdx.y < offset) {
        int row1 = threadIdx.y;
        int row2 = threadIdx.y + offset;
        int idx1 = row1*row_stride + threadIdx.x;
        int idx2 = row2*row_stride + threadIdx.x;
        if (!rms_only) {
          warp_buf1[idx1] += warp_buf1[idx2];
        }
        warp_buf2[idx1] += warp_buf2[idx2];
      }
      __syncthreads();
    }
    int i2 = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadIdx.y == 0 && i2 < n2) {
      int row1 = threadIdx.y;
      int row2 = threadIdx.y + 1;
      int idx1 = row1*row_stride + threadIdx.x;
      int idx2 = row2*row_stride + threadIdx.x;
      if (!rms_only) {
        part_grad_beta[blockIdx.y*n2+i2] = warp_buf1[idx1] + warp_buf1[idx2];
      }
      part_grad_gamma[blockIdx.y*n2+i2] = warp_buf2[idx1] + warp_buf2[idx2];
    }
}

template<typename U, typename V> __global__
void cuComputeGradGammaBeta(
    const U* part_grad_gamma,
    const U* part_grad_beta,
    const int part_size,
    const int n1,
    const int n2,
    V* grad_gamma,
    V* grad_beta,
    bool rms_only)
{
    // sum partial gradients for gamma and beta
    SharedMemory<U> shared;
    U* buf = shared.getPointer();
    int i2 = blockIdx.x * blockDim.x + threadIdx.x;
    if (i2 < n2) {
      // each warp does sequential reductions until reduced part_size is num_warps
      int num_warp_reductions = part_size / blockDim.y;
      U sum_gamma = U(0);
      U sum_beta = U(0);
      const U* part_grad_gamma_ptr = part_grad_gamma + threadIdx.y * num_warp_reductions * n2 + i2;
      const U* part_grad_beta_ptr = part_grad_beta + threadIdx.y * num_warp_reductions * n2 + i2;
      for (int warp_offset = 0;  warp_offset < num_warp_reductions;  ++warp_offset) {
        sum_gamma += part_grad_gamma_ptr[warp_offset*n2];
        if (!rms_only) {
          sum_beta += part_grad_beta_ptr[warp_offset*n2];
        }
      }
      // inter-warp reductions
      const int nbsize3 = blockDim.x * blockDim.y / 2;
      for (int offset = blockDim.y/2;  offset >= 1;  offset /= 2) {
        // top half write to shared memory
        if (threadIdx.y >= offset && threadIdx.y < 2*offset) {
          const int write_idx = (threadIdx.y - offset) * blockDim.x + threadIdx.x;
          buf[write_idx] = sum_gamma;
          if (!rms_only) {
            buf[write_idx+nbsize3] = sum_beta;
          }
        }
        __syncthreads();
        // bottom half sums
        if (threadIdx.y < offset) {
          const int read_idx = threadIdx.y * blockDim.x + threadIdx.x;
          sum_gamma += buf[read_idx];
          if (!rms_only) {
            sum_beta += buf[read_idx+nbsize3];
          }
        }
        __syncthreads();
      }
      // write out fully summed gradients
      if (threadIdx.y == 0) {
        grad_gamma[i2] = sum_gamma;
        if (!rms_only) {
          grad_beta[i2] = sum_beta;
        }
      }
    }
}


template<typename T, typename U, typename V, bool MemoryEfficient> __global__
void cuComputeGradInput(
    const V* __restrict__ dout,
    const T* __restrict__ input_or_output,
    const int n1,
    const int n2,
    const U* __restrict__ mean,
    const U* __restrict__ invvar,
    U epsilon,
    const V* gamma,
    const V* beta,
    T* grad_input,
    const double eps,
    bool rms_only)
{
  for (auto i1=blockIdx.y; i1 < n1; i1 += gridDim.y) {
    U sum_loss1 = U(0);
    U sum_loss2 = U(0);
    const T* k_h = input_or_output + i1*n2;
    const V* k_dout = dout + i1*n2;
    const U c_invvar = invvar[i1];
    const U c_mean = !MemoryEfficient ? mean[i1] : 0.;
    const int numx = blockDim.x * blockDim.y;
    const int thrx = threadIdx.x + threadIdx.y * blockDim.x;
    if (gamma != NULL) {
      int l = 4*thrx;
      for (;  l+3 < n2;  l+=4*numx) {
        for (int k = 0;  k < 4;  ++k) {
          const U c_h = static_cast<U>(k_h[l+k]);
          const U c_loss = static_cast<U>(k_dout[l+k]);
          if (!rms_only) {
            sum_loss1 += c_loss * gamma[l+k];
            if (MemoryEfficient) {
              sum_loss2 += c_loss * (c_h - beta[l+k]);
            } else {
              sum_loss2 += c_loss * gamma[l+k] * (c_h - c_mean) * c_invvar;
            }
          } else {
            if (MemoryEfficient) {
              sum_loss2 += c_loss * c_h;
            } else {
              sum_loss2 += c_loss * gamma[l+k] * (c_h) * c_invvar;
            }
          }
        }
      }
      for (;  l < n2;  ++l) {
        const U c_h = static_cast<U>(k_h[l]);
        const U c_loss = static_cast<U>(k_dout[l]);
        if (!rms_only) {
          sum_loss1 += c_loss * gamma[l];
          if (MemoryEfficient) {
            sum_loss2 += c_loss * (c_h - beta[l]);
          } else {
            sum_loss2 += c_loss * gamma[l] * (c_h - c_mean) * c_invvar;
          }
        } else {
          if (MemoryEfficient) {
            sum_loss2 += c_loss * c_h;
          } else {
            sum_loss2 += c_loss * gamma[l] * (c_h) * c_invvar;
          }
        }
      }
    } else {
      int l = 4*thrx;
      for (;  l+3 < n2;  l+=4*numx) {
        for (int k = 0;  k < 4;  ++k) {
          const U c_h = static_cast<U>(k_h[l+k]);
          const U c_loss = static_cast<U>(k_dout[l+k]);
          if (!rms_only) {
            sum_loss1 += c_loss;
            if (MemoryEfficient) {
              sum_loss2 += c_loss * c_h;
            } else {
              sum_loss2 += c_loss * (c_h - c_mean) * c_invvar;
            }
          } else {
            if (MemoryEfficient) {
              sum_loss2 += c_loss * c_h;
            } else {
              sum_loss2 += c_loss * (c_h) * c_invvar;
            }
          }
        }
      }
      for (;  l < n2;  ++l) {
        const U c_h = static_cast<U>(k_h[l]);
        const U c_loss = static_cast<U>(k_dout[l]);
        if (!rms_only) {
          sum_loss1 += c_loss;
          if (MemoryEfficient) {
            sum_loss2 += c_loss * c_h;
          } else {
            sum_loss2 += c_loss * (c_h - c_mean) * c_invvar;
          }
        } else {
          if (MemoryEfficient) {
            sum_loss2 += c_loss * c_h;
          } else {
            sum_loss2 += c_loss * (c_h) * c_invvar;
          }
        }
      }
    }
    // intra-warp reductions
    for (int mask = blockDim.x/2;  mask > 0;  mask /= 2) {
      if (!rms_only) {
        sum_loss1 += WARP_SHFL_XOR(sum_loss1, mask);
      }
      sum_loss2 += WARP_SHFL_XOR(sum_loss2, mask);
    }
    // inter-warp reductions
    if (blockDim.y > 1) {
      SharedMemory<U> shared;
      U* buf = shared.getPointer();
      for (int offset = blockDim.y/2;  offset > 0;  offset /= 2) {
        // upper half of warps write to shared
        if (threadIdx.y >= offset && threadIdx.y < 2*offset) {
          const int wrt_i = (threadIdx.y - offset) * blockDim.x + threadIdx.x;
          if (!rms_only) {
            buf[2*wrt_i] = sum_loss1;
          }
          buf[2*wrt_i+1] = sum_loss2;
        }
        __syncthreads();
        // lower half merges
        if (threadIdx.y < offset) {
          const int read_i = threadIdx.y * blockDim.x + threadIdx.x;
          if (!rms_only) {
            sum_loss1 += buf[2*read_i];
          }
          sum_loss2 += buf[2*read_i+1];
        }
        __syncthreads();
      }
      if (threadIdx.y == 0) {
        if (!rms_only) {
          buf[2*threadIdx.x] = sum_loss1;
        }
        buf[2*threadIdx.x+1] = sum_loss2;
      }
      __syncthreads();
      if (threadIdx.y !=0) {
        if (!rms_only) {
          sum_loss1 = buf[2*threadIdx.x];
        }
        sum_loss2 = buf[2*threadIdx.x+1];
      }
    }
    // all threads now have the two sums over l
    U fH = (U)n2;
    U term1 = (U(1) / fH) * c_invvar;
    T* k_grad_input = grad_input + i1*n2;
    if (gamma != NULL) {
      for (int l = thrx;  l < n2;  l+=numx) {
        const U c_h = static_cast<U>(k_h[l]);
        const U c_loss = static_cast<U>(k_dout[l]);
        const U k_gamma = static_cast<U>(clamp_by_magnitude(gamma[l], eps));
        U f_grad_input = fH * c_loss * k_gamma;
        if (!rms_only) {
          const U k_beta = beta[l];
          f_grad_input -= sum_loss1;
          if (MemoryEfficient) {
            f_grad_input -= (c_h - k_beta) / k_gamma * sum_loss2;
          } else {
            f_grad_input -= (c_h - c_mean) * c_invvar * sum_loss2;
          }
        } else {
          if (MemoryEfficient) {
            f_grad_input -= c_h / k_gamma * sum_loss2;
          } else {
            f_grad_input -= c_h * c_invvar * sum_loss2;
          }
        }
        f_grad_input *= term1;
        k_grad_input[l] = static_cast<T>(f_grad_input);
      }
    } else {
      for (int l = thrx;  l < n2;  l+=numx) {
        const U c_h = static_cast<U>(k_h[l]);
        const U c_loss = static_cast<U>(k_dout[l]);
        U f_grad_input = fH * c_loss;
        if (!rms_only) {
          f_grad_input -= sum_loss1;
          if (MemoryEfficient) {
            f_grad_input -= c_h * sum_loss2;
          } else {
            f_grad_input -= (c_h - c_mean) * c_invvar * sum_loss2;
          }
        } else {
          if (MemoryEfficient) {
            f_grad_input -= c_h * sum_loss2;
          } else {
            f_grad_input -= c_h * c_invvar * sum_loss2;
          }
        }
        f_grad_input *= term1;
        k_grad_input[l] = static_cast<T>(f_grad_input);
      }
    }
    // prevent race where buf is written again before reads are done
    __syncthreads();
  }
}

void FusedLayerNormCuda(const NDArray& in_arr, const NDArray& ln_scale,
                        const NDArray& ln_bias, NDArray& mean_arr, NDArray& var_arr,
                        NDArray& out_arr, int64_t reduce_dims, 
                        float eps, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(in_arr);
  HT_ASSERT_SAME_DEVICE(in_arr, ln_scale);
  HT_ASSERT_SAME_DEVICE(in_arr, ln_bias);
  HT_ASSERT_SAME_DEVICE(in_arr, mean_arr); 
  HT_ASSERT_SAME_DEVICE(in_arr, var_arr); 
  HT_ASSERT_SAME_DEVICE(in_arr, out_arr);

  CUDAStream cuda_stream(stream);
  hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());

  int ndim = in_arr->ndim();
  int base_dim = 1, last_dim = 1;
  for (int i = 0; i < ndim - reduce_dims; ++i)
    base_dim *= in_arr->shape(i);
  for (int i = ndim - reduce_dims; i < ndim; ++i)
    last_dim *= in_arr->shape(i);
  const dim3 threads(32,4,1);
  auto dprops = Device::dprop(in_arr->device().index());
  const uint64_t maxGridY = dprops.maxGridSize[1];
  size_t n1 = in_arr->numel() / last_dim, n2 = last_dim;
  const dim3 blocks(1, std::min((uint64_t)n1, maxGridY), 1);
  HT_DISPATCH_FLOATING_TYPES(
    in_arr->dtype(), spec_t, "FusedLayerNormCuda", [&]() {
      int nshared = threads.y > 1 ?
          threads.y * sizeof(float) + (threads.y / 2) * sizeof(float) : 0;
      cuApplyLayerNorm<<<blocks, threads, nshared, cuda_stream>>>(
        out_arr->data_ptr<spec_t>(), mean_arr->data_ptr<float>(), 
        var_arr->data_ptr<float>(), in_arr->data_ptr<spec_t>(), 
        n1, n2, float(eps), ln_scale->data_ptr<spec_t>(), 
        ln_bias->data_ptr<spec_t>()); 
  });
  NDArray::MarkUsedBy({in_arr, ln_scale, ln_bias, mean_arr, var_arr, out_arr}, stream);
}

void FusedRMSNormCuda(const NDArray& in_arr, const NDArray& ln_scale,
                      NDArray& var_arr, NDArray& out_arr, int64_t reduce_dims, 
                      float eps, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(in_arr);
  HT_ASSERT_SAME_DEVICE(in_arr, ln_scale);
  HT_ASSERT_SAME_DEVICE(in_arr, var_arr); 
  HT_ASSERT_SAME_DEVICE(in_arr, out_arr);

  CUDAStream cuda_stream(stream);
  hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());

  int ndim = in_arr->ndim();
  int base_dim = 1, last_dim = 1;
  for (int i = 0; i < ndim - reduce_dims; ++i)
    base_dim *= in_arr->shape(i);
  for (int i = ndim - reduce_dims; i < ndim; ++i)
    last_dim *= in_arr->shape(i);
  const dim3 threads(32,4,1);
  auto dprops = Device::dprop(in_arr->device().index());
  const uint64_t maxGridY = dprops.maxGridSize[1];
  size_t n1 = in_arr->numel() / last_dim, n2 = last_dim;
  const dim3 blocks(1, std::min((uint64_t)n1, maxGridY), 1);
  HT_DISPATCH_FLOATING_TYPES(
    in_arr->dtype(), spec_t, "FusedRMSNormCuda", [&]() {
      int nshared = threads.y > 1 ?
          threads.y * sizeof(float) + (threads.y / 2) * sizeof(float) : 0;
      cuApplyRMSNorm<<<blocks, threads, nshared, cuda_stream>>>(
        out_arr->data_ptr<spec_t>(), var_arr->data_ptr<float>(), 
        in_arr->data_ptr<spec_t>(), n1, n2, float(eps), 
        ln_scale->data_ptr<spec_t>()); 
  });
  NDArray::MarkUsedBy({in_arr, ln_scale, var_arr, out_arr}, stream);
}

void FusedLayerNormGradientCuda(const NDArray& out_grads, const NDArray& in_arr,
                                const NDArray& ln_scale, const NDArray& ln_bias, NDArray& grad_arr,
                                NDArray& grad_scale, NDArray& grad_bias,
                                const NDArray& mean_arr, const NDArray& var_arr,
                                int64_t reduce_dims, float eps, bool inplace, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(out_grads);
  HT_ASSERT_SAME_DEVICE(out_grads, ln_scale);
  HT_ASSERT_SAME_DEVICE(out_grads, in_arr);
  HT_ASSERT_SAME_DEVICE(out_grads, mean_arr); 
  HT_ASSERT_SAME_DEVICE(out_grads, var_arr); 
  HT_ASSERT_SAME_DEVICE(out_grads, grad_scale);
  HT_ASSERT_SAME_DEVICE(out_grads, grad_arr);
  HT_ASSERT_SAME_DEVICE(out_grads, grad_bias);

  CUDAStream cuda_stream(stream);
  hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
  cudnnHandle_t handle = hetu::impl::GetCudnnHandle(cuda_stream.device_id());

  int ndim = out_grads->ndim();
  size_t total_elements = 1;

  int last_2dim = in_arr->shape(ndim - 1) * in_arr->shape(ndim - 2);

  HTAxes reduce_axes_before = {}, reduce_axes_after = {};
  for (int i = 0; i < ndim; ++i) {
    if (i < ndim - reduce_dims)
      reduce_axes_before.emplace_back(i);
    else
      reduce_axes_after.emplace_back(i);
  }

  for (int i = 0; i < ndim; ++i)
    total_elements *= out_grads->shape(i);
  int lastdim = 1;
  for (size_t i = 0; i < reduce_dims; ++i) {
    lastdim *= out_grads->shape(ndim - 1 - i);
  }

  size_t size = total_elements;
  if (size == 0)
    return;
  size_t n1 = in_arr->numel() / lastdim, n2 = lastdim;
  HT_DISPATCH_FLOATING_TYPES(
      in_arr->dtype(), spec_t, "FusedLayerNormGradientCuda", [&]() {
      const int part_size = 16;
      const dim3 threads2(32,4,1);
      const dim3 blocks2((n2+threads2.x-1)/threads2.x,part_size,1);
      const int nshared2_a = 2 * sizeof(float) * threads2.y * threads2.y * (threads2.x + 1);
      const int nshared2_b = threads2.x * threads2.y * sizeof(float);
      const int nshared2 = nshared2_a > nshared2_b ? nshared2_a : nshared2_b;
      const auto part_grad_dtype =
        (in_arr->dtype() == DataType::FLOAT16 || in_arr->dtype() == DataType::BFLOAT16) ?
        DataType::FLOAT32 : in_arr->dtype();
      HTShape part_shape = {int64_t(part_size), int64_t(n2)};
      auto part_grad_gamma = NDArray::empty(part_shape, in_arr->device(), part_grad_dtype, stream.stream_index());
      auto part_grad_beta = NDArray::empty_like(part_grad_gamma, stream.stream_index());
      if (inplace) { 
        cuComputePartGradGammaBeta<spec_t, float, spec_t, true><<<blocks2, threads2, nshared2, cuda_stream>>>(
                        out_grads->data_ptr<spec_t>(),
                        in_arr->data_ptr<spec_t>(),
                        n1,n2,
                        mean_arr->data_ptr<float>(),
                        var_arr->data_ptr<float>(),
                        float(eps),
                        ln_scale->data_ptr<spec_t>(),
                        ln_bias->data_ptr<spec_t>(),
                        part_grad_gamma->data_ptr<float>(),
                        part_grad_beta->data_ptr<float>(),
                        double(eps),
                        false);
      }
      else {
        cuComputePartGradGammaBeta<spec_t, float, spec_t, false><<<blocks2, threads2, nshared2, cuda_stream>>>(
                        out_grads->data_ptr<spec_t>(),
                        in_arr->data_ptr<spec_t>(),
                        n1,n2,
                        mean_arr->data_ptr<float>(),
                        var_arr->data_ptr<float>(),
                        float(eps),
                        ln_scale->data_ptr<spec_t>(),
                        ln_bias->data_ptr<spec_t>(),
                        part_grad_gamma->data_ptr<float>(),
                        part_grad_beta->data_ptr<float>(),
                        double(eps),
                        false);
      }

        const dim3 threads3(32,8,1);
        const dim3 blocks3((n2+threads2.x-1)/threads2.x,1,1);
        const int nshared3 = threads3.x * threads3.y * sizeof(float);
          cuComputeGradGammaBeta<<<blocks3, threads3, nshared3, cuda_stream>>>(
                          part_grad_gamma->data_ptr<float>(),
                          part_grad_beta->data_ptr<float>(),
                          part_size,
                          n1,n2,
                          grad_scale->data_ptr<spec_t>(),
                          grad_bias->data_ptr<spec_t>(),
                          false);

        auto dprops = Device::dprop(in_arr->device().index());
        const uint64_t maxGridY = dprops.maxGridSize[1];
        const dim3 blocks1(1, std::min((uint64_t)n1, maxGridY), 1);
        // const dim3 blocks1(1, (uint64_t)n1, 1);
        const dim3 threads1(32,4,1);
        int nshared =
                threads1.y > 1 ?
                threads1.y * threads1.x * sizeof(float) : 0;
        if (inplace) {
          cuComputeGradInput<spec_t, float, spec_t, true><<<blocks1, threads1, nshared, cuda_stream>>>(
                  out_grads->data_ptr<spec_t>(),
                  in_arr->data_ptr<spec_t>(),
                  n1,n2,
                  mean_arr->data_ptr<float>(),
                  var_arr->data_ptr<float>(),
                  float(eps),
                  ln_scale->data_ptr<spec_t>(),
                  ln_bias->data_ptr<spec_t>(),
                  grad_arr->data_ptr<spec_t>(),
                  eps,
                  false);
        }
        else {
          cuComputeGradInput<spec_t, float, spec_t, false><<<blocks1, threads1, nshared, cuda_stream>>>(
                  out_grads->data_ptr<spec_t>(),
                  in_arr->data_ptr<spec_t>(),
                  n1,n2,
                  mean_arr->data_ptr<float>(),
                  var_arr->data_ptr<float>(),
                  float(eps),
                  ln_scale->data_ptr<spec_t>(),
                  ln_bias->data_ptr<spec_t>(),
                  grad_arr->data_ptr<spec_t>(),
                  eps,
                  false);
        }
      });
  NDArray::MarkUsedBy({out_grads, in_arr, ln_scale, ln_bias, grad_arr,
                       grad_scale, grad_bias, mean_arr, var_arr}, stream);
}

void FusedRMSNormGradientCuda(const NDArray& out_grads, const NDArray& in_arr,
                              const NDArray& ln_scale, NDArray& grad_arr,
                              NDArray& grad_scale, const NDArray& var_arr,
                              int64_t reduce_dims, float eps, bool inplace, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(out_grads);
  HT_ASSERT_SAME_DEVICE(out_grads, ln_scale);
  HT_ASSERT_SAME_DEVICE(out_grads, in_arr);
  HT_ASSERT_SAME_DEVICE(out_grads, var_arr); 
  HT_ASSERT_SAME_DEVICE(out_grads, grad_scale);
  HT_ASSERT_SAME_DEVICE(out_grads, grad_arr);

  CUDAStream cuda_stream(stream);
  hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
  cudnnHandle_t handle = hetu::impl::GetCudnnHandle(cuda_stream.device_id());

  int ndim = out_grads->ndim();
  size_t total_elements = 1;

  int last_2dim = in_arr->shape(ndim - 1) * in_arr->shape(ndim - 2);

  HTAxes reduce_axes_before = {}, reduce_axes_after = {};
  for (int i = 0; i < ndim; ++i) {
    if (i < ndim - reduce_dims)
      reduce_axes_before.emplace_back(i);
    else
      reduce_axes_after.emplace_back(i);
  }

  for (int i = 0; i < ndim; ++i)
    total_elements *= out_grads->shape(i);
  int lastdim = 1;
  for (size_t i = 0; i < reduce_dims; ++i) {
    lastdim *= out_grads->shape(ndim - 1 - i);
  }

  size_t size = total_elements;
  if (size == 0)
    return;
  size_t n1 = in_arr->numel() / lastdim, n2 = lastdim;
  HT_DISPATCH_FLOATING_TYPES(
      in_arr->dtype(), spec_t, "FusedRMSNormGradientCuda", [&]() {
      const int part_size = 16;
      const dim3 threads2(32,4,1);
      const dim3 blocks2((n2+threads2.x-1)/threads2.x,part_size,1);
      const int nshared2_a = 2 * sizeof(float) * threads2.y * threads2.y * (threads2.x + 1);
      const int nshared2_b = threads2.x * threads2.y * sizeof(float);
      const int nshared2 = nshared2_a > nshared2_b ? nshared2_a : nshared2_b;
      const auto part_grad_dtype =
        (in_arr->dtype() == DataType::FLOAT16 || in_arr->dtype() == DataType::BFLOAT16) ?
        DataType::FLOAT32 : in_arr->dtype();
      HTShape part_shape = {int64_t(part_size), int64_t(n2)};
      auto part_grad_gamma = NDArray::empty(part_shape, in_arr->device(), part_grad_dtype, stream.stream_index());
      if (inplace) { 
        cuComputePartGradGammaBeta<spec_t, float, spec_t, true>
        <<<blocks2, threads2, nshared2, cuda_stream>>>(
                        out_grads->data_ptr<spec_t>(),
                        in_arr->data_ptr<spec_t>(),
                        n1,n2,
                        var_arr->data_ptr<float>(),
                        var_arr->data_ptr<float>(),
                        float(eps),
                        ln_scale->data_ptr<spec_t>(),
                        ln_scale->data_ptr<spec_t>(),
                        part_grad_gamma->data_ptr<float>(),
                        part_grad_gamma->data_ptr<float>(),
                        double(eps),
                        true);
      }
      else {
        cuComputePartGradGammaBeta<spec_t, float, spec_t, false><<<blocks2, threads2, nshared2, cuda_stream>>>(
                        out_grads->data_ptr<spec_t>(),
                        in_arr->data_ptr<spec_t>(),
                        n1,n2,
                        var_arr->data_ptr<float>(),
                        var_arr->data_ptr<float>(),
                        float(eps),
                        ln_scale->data_ptr<spec_t>(),
                        ln_scale->data_ptr<spec_t>(),
                        part_grad_gamma->data_ptr<float>(),
                        part_grad_gamma->data_ptr<float>(),
                        double(eps),
                        true);
      }

        const dim3 threads3(32,8,1);
        const dim3 blocks3((n2+threads2.x-1)/threads2.x,1,1);
        const int nshared3 = threads3.x * threads3.y * sizeof(float);
          cuComputeGradGammaBeta<<<blocks3, threads3, nshared3, cuda_stream>>>(
                          part_grad_gamma->data_ptr<float>(),
                          part_grad_gamma->data_ptr<float>(),
                          part_size,
                          n1,n2,
                          grad_scale->data_ptr<spec_t>(),
                          grad_scale->data_ptr<spec_t>(),
                          true);

        auto dprops = Device::dprop(in_arr->device().index());
        const uint64_t maxGridY = dprops.maxGridSize[1];
        const dim3 blocks1(1, std::min((uint64_t)n1, maxGridY), 1);
        // const dim3 blocks1(1, (uint64_t)n1, 1);
        const dim3 threads1(32,4,1);
        int nshared =
                threads1.y > 1 ?
                threads1.y * threads1.x * sizeof(float) : 0;
        if (inplace) {
          cuComputeGradInput<spec_t, float, spec_t, true><<<blocks1, threads1, nshared, cuda_stream>>>(
                  out_grads->data_ptr<spec_t>(),
                  in_arr->data_ptr<spec_t>(),
                  n1,n2,
                  var_arr->data_ptr<float>(),
                  var_arr->data_ptr<float>(),
                  float(eps),
                  ln_scale->data_ptr<spec_t>(),
                  ln_scale->data_ptr<spec_t>(),
                  grad_arr->data_ptr<spec_t>(),
                  eps,
                  true);
        }
        else {
          cuComputeGradInput<spec_t, float, spec_t, false><<<blocks1, threads1, nshared, cuda_stream>>>(
                  out_grads->data_ptr<spec_t>(),
                  in_arr->data_ptr<spec_t>(),
                  n1,n2,
                  var_arr->data_ptr<float>(),
                  var_arr->data_ptr<float>(),
                  float(eps),
                  ln_scale->data_ptr<spec_t>(),
                  ln_scale->data_ptr<spec_t>(),
                  grad_arr->data_ptr<spec_t>(),
                  eps,
                  true);
        }
      });
  NDArray::MarkUsedBy({out_grads, in_arr, ln_scale, grad_arr,
                       grad_scale, var_arr}, stream);
}

} // namespace impl
} // namespace hetu
