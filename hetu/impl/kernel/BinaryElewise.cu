#include "hetu/core/ndarray.h"
#include "hetu/core/stream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/stream/CUDAStream.h"
#include "hetu/impl/utils/cuda_utils.h"
#include "hetu/impl/kernel/Binary.cuh"
#include "hetu/impl/utils/offset_calculator.cuh"
#include "hetu/impl/kernel/Vectorized.cuh"

namespace hetu {
namespace impl {

inline void merge_broadcast_shape(const NDArray& inputA, const NDArray& inputB, NDArray& output, size_t& num_dims,
                                  HTShape& merge_shapeA, HTShape& merge_shapeB, HTShape& merge_out_shape,
                                  HTStride& merge_strideA, HTStride& merge_strideB, HTStride& merge_out_stride) {
  size_t numel_A = inputA->numel();
  size_t numel_B = inputB->numel();
  size_t ndim_A = inputA->ndim();
  size_t ndim_B = inputB->ndim();
  size_t ndim_out = std::max(ndim_A, ndim_B);
  HTShape out_shape = output->shape();
  HT_ASSERT(out_shape.size() > 0);
  auto MakeGetShape = [ndim_out](size_t ndim, HTShape shape) {
    int64_t diff = ndim_out - ndim;
    return [diff, shape](size_t idx) {
      return idx < diff ? 1 : shape[idx - diff];
    };
  };
  auto MakeGetStride = [ndim_out](size_t ndim, HTStride stride) {
    int64_t diff = ndim_out - ndim;
    return [diff, stride](size_t idx) {
      return idx < diff ? 0 : stride[idx - diff];
    };
  };
  auto get_shape_A = MakeGetShape(ndim_A, inputA->shape());
  auto get_shape_B = MakeGetShape(ndim_B, inputB->shape());
  auto get_shape_out = MakeGetShape(ndim_out, out_shape);
  auto get_stride_A = MakeGetStride(ndim_A, inputA->stride());
  auto get_stride_B = MakeGetStride(ndim_B, inputB->stride());
  auto get_stride_out = MakeGetStride(ndim_out, output->stride());
  num_dims = 0;
  bool prev_broadcast_A = false;
  bool prev_broadcast_B = false;
  bool prev_broadcast_out = false;
  for (int i = 0; i < ndim_out; i++) {
    int64_t dim_A = get_shape_A(i);
    int64_t dim_B = get_shape_B(i);
    int64_t dim_out = get_shape_out(i);
    int64_t stride_A = get_stride_A(i);
    int64_t stride_B = get_stride_B(i);
    int64_t stride_out = get_stride_out(i);
    int64_t broadcast_dim = std::max(std::max(dim_A, dim_B), dim_out);
    bool broadcast_A = (dim_A == 1);
    bool broadcast_B = (dim_B == 1);
    bool broadcast_out = (dim_out == 1);
    if (broadcast_dim == 1) {
      continue;
    } else if (num_dims != 0 &&
               (prev_broadcast_A == broadcast_A &&
                prev_broadcast_B == broadcast_B &&
                prev_broadcast_out == broadcast_out) && 
               (merge_strideA[num_dims - 1] == dim_A * stride_A &&
                merge_strideB[num_dims - 1] == dim_B * stride_B &&
                merge_out_stride[num_dims - 1] == dim_out * stride_out)) {
      merge_shapeA[num_dims - 1] *= dim_A;
      merge_shapeB[num_dims - 1] *= dim_B;
      merge_out_shape[num_dims - 1] *= dim_out;
      merge_strideA[num_dims - 1] = stride_A;
      merge_strideB[num_dims - 1] = stride_B;
      merge_out_stride[num_dims - 1] = stride_out;
    } else {
      merge_shapeA.push_back(dim_A);
      merge_shapeB.push_back(dim_B);
      merge_out_shape.push_back(dim_out);
      merge_strideA.push_back(stride_A);
      merge_strideB.push_back(stride_B);
      merge_out_stride.push_back(stride_out);
      num_dims++;
      prev_broadcast_A = broadcast_A;
      prev_broadcast_B = broadcast_B;
      prev_broadcast_out = broadcast_out;
    }
  }
}

template <typename spec_a_t, typename spec_b_t, typename out_t>
int get_vectorize_size(size_t num_dims, const HTShape& shapeA, const HTShape& shapeB,
                       spec_a_t* inputA, spec_b_t* inputB, out_t* output) {
  auto is_aligned = [&](int vec_size) -> bool {
    bool is_A_supported = (shapeA[num_dims - 1] == 1) ||
                          (shapeA[num_dims - 1] % vec_size == 0 &&
                           reinterpret_cast<uint64_t>(inputA) % (vec_size * sizeof(spec_a_t)) == 0);
    bool is_B_supported = (shapeB[num_dims - 1] == 1) ||
                          (shapeB[num_dims - 1] % vec_size == 0 &&
                           reinterpret_cast<uint64_t>(inputB) % (vec_size * sizeof(spec_b_t)) == 0);
    if (is_A_supported && is_B_supported &&
       (reinterpret_cast<uint64_t>(output) % (vec_size * sizeof(out_t)) == 0)) {
      return true;
    }
    return false;
  };
  for (int vec_size = 4; vec_size >= 1; vec_size /= 2) {
    if (is_aligned(vec_size)) {
      return vec_size;
    }
  }
}

template <int vec_size_A, int vec_size_B, typename spec_a_t, typename spec_b_t,
          typename out_t, typename arr_t, typename func_t>
__global__ void vectorize_broadcast_loop_kernel(const spec_a_t* inputA, const spec_b_t* inputB, size_t size,
                                                out_t* output, size_t num_dims, const int64_t idx_mask_A,
                                                const int64_t idx_mask_B, arr_t strideA,
                                                arr_t strideB, arr_t out_stride,
                                                func_t op) {
  constexpr int vec_size_out = vec_size_A > vec_size_B ? vec_size_A : vec_size_B;
  auto step = blockDim.x * gridDim.x;
  #pragma unroll
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += step) {
    int offset_A = 0;
    int offset_B = 0;
    int remainder = i;
    #pragma unroll
    for (int j = 0; j < num_dims - 1; j++) {
      int out_idx = remainder / out_stride[j];
      remainder = remainder - out_idx * out_stride[j];
      offset_A += bool(idx_mask_A & (1 << j)) ? out_idx * strideA[j] : 0;
      offset_B += bool(idx_mask_B & (1 << j)) ? out_idx * strideB[j] : 0;
    }
    offset_A += bool(idx_mask_A & (1 << (num_dims - 1))) ? remainder * strideA[num_dims - 1] : 0;
    offset_B += bool(idx_mask_B & (1 << (num_dims - 1))) ? remainder * strideB[num_dims - 1] : 0;
    auto vec_inputA_ptr = reinterpret_cast<const aligned_vector<spec_a_t, vec_size_A>*>(inputA);
    auto vec_inputB_ptr = reinterpret_cast<const aligned_vector<spec_b_t, vec_size_B>*>(inputB);
    auto vec_output_ptr = reinterpret_cast<aligned_vector<out_t, vec_size_out>*>(output);
    auto vec_inputA = vec_inputA_ptr[offset_A];
    auto vec_inputB = vec_inputB_ptr[offset_B];
    aligned_vector<out_t, vec_size_out> vec_output;
    #pragma unroll
    for (int j = 0; j < vec_size_out; j++) {
      const spec_a_t valA = (vec_size_A == vec_size_out) ? vec_inputA.val[j] : vec_inputA.val[0];
      const spec_b_t valB = (vec_size_B == vec_size_out) ? vec_inputB.val[j] : vec_inputB.val[0];
      vec_output.val[j] = op(valA, valB);
    }
    vec_output_ptr[i] = vec_output;
  }
}

template <typename spec_a_t, typename spec_b_t, typename out_t, typename func_t>
void launch_vectorize_broadcast_loop_kernel(const NDArray& inputA, const NDArray& inputB, NDArray& output,
                                            size_t num_dims, HTShape& shapeA, HTShape& shapeB,
                                            HTShape& out_shape, HTStride& strideA, HTStride& strideB,
                                            HTStride& out_stride, const Stream& stream, const func_t& op) {
  int vec_size = get_vectorize_size(num_dims, shapeA, shapeB,
                                    inputA->data_ptr<spec_a_t>(), inputB->data_ptr<spec_b_t>(),
                                    output->data_ptr<out_t>());
  int vec_size_A = 1;
  int vec_size_B = 1;
  auto stride_div_vec_size = [num_dims](HTStride& stride, int vec_size) {
    for (int i = 0; i < num_dims - 1; i++) {
      stride[i] /= vec_size;
    }
  };
  if (shapeA[num_dims - 1] != 1) {
    shapeA[num_dims - 1] /= vec_size;
    stride_div_vec_size(strideA, vec_size);
    vec_size_A = vec_size;
  }
  if (shapeB[num_dims - 1] != 1) {
    shapeB[num_dims - 1] /= vec_size;
    stride_div_vec_size(strideB, vec_size);
    vec_size_B = vec_size;
  }
  out_shape[num_dims - 1] /= vec_size;
  stride_div_vec_size(out_stride, vec_size);
  size_t out_size = output->numel();
  size_t size = out_size / vec_size;
  int64_t idx_mask_A = 0, idx_mask_B = 0;
  for (int i = 0; i < num_dims; i++) {
    idx_mask_A |= (shapeA[i] == 1 ? 0 : (1 << i));
    idx_mask_B |= (shapeB[i] == 1 ? 0 : (1 << i));
  }
  #define DISPATCH_VECTORIZE_KERNEL(vec_size_A, vec_size_B)                                             \
    vectorize_broadcast_loop_kernel<vec_size_A, vec_size_B, spec_a_t, spec_b_t, out_t>                  \
    <<<grid, block, 0, cuda_stream>>>(inputA->data_ptr<spec_a_t>(), inputB->data_ptr<spec_b_t>(), size, \
                                      output->data_ptr<out_t>(), num_dims, idx_mask_A,                  \
                                      idx_mask_B, strideA_arr, strideB_arr,                             \
                                      out_stride_arr, op);

  #define DISPATCH_VEC_SIZE(vec_size_A, vec_size_B)                                            \
  if (vec_size_A == 1 && vec_size_B == 1) {                                                    \
    DISPATCH_VECTORIZE_KERNEL(1, 1);                                                           \
  } else if (vec_size_A == 1 && vec_size_B == 2) {                                             \
    DISPATCH_VECTORIZE_KERNEL(1, 2);                                                           \
  } else if (vec_size_A == 1 && vec_size_B == 4) {                                             \
    DISPATCH_VECTORIZE_KERNEL(1, 4);                                                           \
  } else if (vec_size_A == 2 && vec_size_B == 1) {                                             \
    DISPATCH_VECTORIZE_KERNEL(2, 1);                                                           \
  } else if (vec_size_A == 4 && vec_size_B == 1) {                                             \
    DISPATCH_VECTORIZE_KERNEL(4, 1);                                                           \
  } else if (vec_size_A == 2 && vec_size_B == 2) {                                             \
    DISPATCH_VECTORIZE_KERNEL(2, 2);                                                           \
  } else if (vec_size_A == 4 && vec_size_B == 4) {                                             \
    DISPATCH_VECTORIZE_KERNEL(4, 4);                                                           \
  } else {                                                                                     \
    HT_RUNTIME_ERROR << "Unexpected vectorization size: " << vec_size_A << ", " << vec_size_B; \
    __builtin_unreachable();                                                                   \
  }

  auto device_id = stream.device_index();
  hetu::cuda::CUDADeviceGuard guard(device_id);
  CUDAStream cuda_stream(stream);
  dim3 grid(DIVUP(size, BLOCK_WORK_SIZE));
  dim3 block(NUM_THREADS);

  #define ALLOC_NDIM_BUFFER(ndim)                                      \
  auto strideA_arr = hetu::cuda::to_int64_buffer<ndim>(strideA);       \
  auto strideB_arr = hetu::cuda::to_int64_buffer<ndim>(strideB);       \
  auto out_stride_arr = hetu::cuda::to_int64_buffer<ndim>(out_stride);

  #define DISPATCH_NDIM(ndim, vec_size_A, vec_size_B) \
  if (ndim == 1) {                                    \
    ALLOC_NDIM_BUFFER(1)                              \
    DISPATCH_VEC_SIZE(vec_size_A, vec_size_B)         \
  } else if (ndim == 2) {                             \
    ALLOC_NDIM_BUFFER(2)                              \
    DISPATCH_VEC_SIZE(vec_size_A, vec_size_B)         \
  } else if (ndim == 3) {                             \
    ALLOC_NDIM_BUFFER(3)                              \
    DISPATCH_VEC_SIZE(vec_size_A, vec_size_B)         \
  } else if (ndim == 4) {                             \
    ALLOC_NDIM_BUFFER(4)                              \
    DISPATCH_VEC_SIZE(vec_size_A, vec_size_B)         \
  } else {                                            \
    ALLOC_NDIM_BUFFER(HT_MAX_NDIM)                    \
    DISPATCH_VEC_SIZE(vec_size_A, vec_size_B)         \
  }

  DISPATCH_NDIM(num_dims, vec_size_A, vec_size_B);
}

template <int nt, int vt, typename spec_a_t, typename spec_b_t, typename out_t, typename func_t>
__global__ void broadcast_elewise_kernel(const spec_a_t* inputA, const spec_b_t* inputB, size_t size,
                                         out_t* output, size_t num_dims, const int64_t idx_mask_A,
                                         const int64_t idx_mask_B, const int64_t* strideA,
                                         const int64_t* strideB, const int64_t* out_stride, func_t op,
                                         const OffsetCalculator* out_offset_calculator) {
  int tid = threadIdx.x;
  int nv = nt * vt;
  int idx = nv * blockIdx.x + tid;
  #pragma unroll
  for (int i = 0; i < vt; i++) {
    if (idx < size) {
      int offset_A = 0;
      int offset_B = 0;
      int remainder = idx;
      #pragma unroll
      for (int j = 0; j < num_dims - 1; j++) {
        int out_idx = remainder / out_stride[j];
        offset_A += bool(idx_mask_A & (1 << j)) ? out_idx * strideA[j] : 0;
        offset_B += bool(idx_mask_B & (1 << j)) ? out_idx * strideB[j] : 0;
        remainder = remainder - out_idx * out_stride[j];
      }
      offset_A += bool(idx_mask_A & (1 << (num_dims - 1))) ? remainder * strideA[num_dims - 1] : 0;
      offset_B += bool(idx_mask_B & (1 << (num_dims - 1))) ? remainder * strideB[num_dims - 1] : 0;
      auto out_offset = out_offset_calculator->get(idx);
      output[out_offset] = op(inputA[offset_A], inputB[offset_B]);
      idx += nt;
    }
  }
}

template <typename spec_a_t, typename spec_b_t, typename out_t, typename func_t>
void launch_broadcast_loop_kernel(const NDArray& inputA, const NDArray& inputB, NDArray& output,
                                  size_t num_dims, HTShape& shapeA, HTShape& shapeB,
                                  HTShape& out_shape, HTStride& strideA, HTStride& strideB,
                                  HTStride& out_stride, const Stream& stream, const func_t& op) {
  bool contiguous = output->is_contiguous() &&
                    inputA->is_contiguous() &&
                    inputB->is_contiguous();
  if (contiguous) {
    launch_vectorize_broadcast_loop_kernel<spec_a_t, spec_b_t, out_t>(inputA, inputB, output,
                                                                      num_dims, shapeA, shapeB,
                                                                      out_shape, strideA, strideB,
                                                                      out_stride, stream, op);
  } else {
    size_t size = output->numel();
    constexpr int unroll_factor = sizeof(DataType2Size(output->dtype())) >= 4 ? 2 : 4;
    dim3 block(NUM_THREADS);
    dim3 grid(DIVUP(size, block.x * unroll_factor));
    CUDAStream cuda_stream(stream);
    int64_t idx_mask_A = 0, idx_mask_B = 0;
    for (int i = 0; i < num_dims; i++) {
      idx_mask_A |= (shapeA[i] == 1 ? 0 : (1 << i));
      idx_mask_B |= (shapeB[i] == 1 ? 0 : (1 << i));
    }
    NDArrayMeta broadcast_out_meta = output->meta();
    broadcast_out_meta.set_shape(out_shape)
                      .set_stride(out_stride);
    auto broadcast_output = NDArray(broadcast_out_meta, output->storage(), output->storage_offset());
    NDArray out_offset_calculator_arr;
    OffsetCalculator *out_offset_calculator;
    std::tie(out_offset_calculator_arr, out_offset_calculator) =
      AllocOffsetCalculator(broadcast_output, stream);
    out_stride = Shape2Stride(out_shape);
    std::vector<int64_t> broadcast_metadata;
    broadcast_metadata.insert(broadcast_metadata.end(), strideA.begin(), strideA.end());
    broadcast_metadata.insert(broadcast_metadata.end(), strideB.begin(), strideB.end());
    broadcast_metadata.insert(broadcast_metadata.end(), out_stride.begin(), out_stride.end());
    auto broadcast_metadata_arr = hetu::cuda::to_int64_ndarray(broadcast_metadata, cuda_stream.device_id());
    int64_t *strideA_ptr = broadcast_metadata_arr->data_ptr<int64_t>();
    int64_t *strideB_ptr = strideA_ptr + strideA.size();
    int64_t *out_stride_ptr = strideB_ptr + strideB.size();
    hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
    broadcast_elewise_kernel<NUM_THREADS, unroll_factor><<<grid, block, 0, cuda_stream>>>(
      inputA->data_ptr<spec_a_t>(), inputB->data_ptr<spec_b_t>(), size, output->data_ptr<out_t>(),
      num_dims, idx_mask_A, idx_mask_B, strideA_ptr, strideB_ptr,
      out_stride_ptr, op, out_offset_calculator);
    NDArray::MarkUsedBy({broadcast_metadata_arr, out_offset_calculator_arr}, stream);
  }
}

#define BinaryElewiseCudaHelper(inputA, inputB, output, op, stream, name)                 \
  do {                                                                                    \
    HT_ASSERT_CUDA_DEVICE(inputA);                                                        \
    HT_ASSERT_SAME_DEVICE(inputA, output);                                                \
    HT_ASSERT_SAME_DEVICE(inputB, output);                                                \
    size_t sizeA = inputA->numel();                                                       \
    size_t sizeB = inputB->numel();                                                       \
    if (output->numel() == 0) {                                                           \
      return;                                                                             \
    }                                                                                     \
    if (sizeA == sizeB) {                                                                 \
      auto size = sizeA;                                                                  \
      if (inputA->dtype() == inputB->dtype()) {                                           \
        HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(                                           \
          inputA->dtype(), spec_t, name, [&]() {                                          \
            using InType = std::tuple<spec_t, spec_t>;                                    \
            using OutType = thrust::tuple<spec_t>;                                        \
            launch_loop_kernel<InType, OutType>({inputA, inputB}, {output}, size, stream, \
                op<spec_t, spec_t>());                                                    \
          });                                                                             \
        NDArray::MarkUsedBy(                                                              \
          {inputA, inputB, output}, stream);                                              \
      } else {                                                                            \
        HT_DISPATCH_TRIPLE_FLOATING_TYPES(                                                \
          inputA->dtype(), inputB->dtype(),                                               \
          output->dtype(), spec_a_t, spec_b_t, spec_c_t, name, [&]() {                    \
            launch_loop_kernel<spec_a_t, spec_b_t, spec_c_t>(                             \
                inputA, inputB, output, size, stream,                                     \
                op<spec_a_t, spec_b_t>());                                                \
          });                                                                             \
        NDArray::MarkUsedBy(                                                              \
          {inputA, inputB, output}, stream);                                              \
      }                                                                                   \
    } else {                                                                              \
      size_t num_dims = 0;                                                                \
      HTShape shapeA, shapeB, out_shape;                                                  \
      HTStride strideA, strideB, out_stride;                                              \
      merge_broadcast_shape(inputA, inputB, output, num_dims,                             \
                            shapeA, shapeB, out_shape,                                    \
                            strideA, strideB, out_stride);                                \
      if (inputA->dtype() == inputB->dtype()) {                                           \
        HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(                                           \
          inputA->dtype(), spec_t, name, [&]() {                                          \
            using InType = std::tuple<spec_t, spec_t>;                                    \
            using OutType = thrust::tuple<spec_t>;                                        \
            launch_broadcast_loop_kernel<spec_t, spec_t, spec_t>(                         \
              inputA, inputB, output, num_dims,                                           \
              shapeA, shapeB, out_shape,                                                  \
              strideA, strideB, out_stride,                                               \
              stream, op<spec_t, spec_t>());                                              \
        });                                                                               \
        NDArray::MarkUsedBy(                                                              \
          {inputA, inputB, output}, stream);                                              \
      } else {                                                                            \
        HT_NOT_IMPLEMENTED                                                                \
          << name << " across different data types is not supported yet";                 \
      }                                                                                   \
    }                                                                                     \
  } while (0);

void AddElewiseCuda(const NDArray& inputA, const NDArray& inputB,
                    NDArray& output, const Stream& stream) {
  BinaryElewiseCudaHelper(inputA, inputB, output, kplus, stream,
                          "AddElewiseCuda");
}

void SubElewiseCuda(const NDArray& inputA, const NDArray& inputB,
                    NDArray& output, const Stream& stream) {
  BinaryElewiseCudaHelper(inputA, inputB, output, kminus, stream,
                          "SubElewiseCuda");
}

void MulElewiseCuda(const NDArray& inputA, const NDArray& inputB,
                    NDArray& output, const Stream& stream) {
  BinaryElewiseCudaHelper(inputA, inputB, output, kmultiplies, stream,
                          "MulElewiseCuda");
}

void DivElewiseCuda(const NDArray& inputA, const NDArray& inputB,
                    NDArray& output, const Stream& stream) {
  BinaryElewiseCudaHelper(inputA, inputB, output, kdivides, stream,
                          "DivElewiseCuda");
}

} // namespace impl
} // namespace hetu
