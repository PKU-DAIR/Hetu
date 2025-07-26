#include "hetu/core/ndarray.h"
#include "hetu/impl/stream/CUDAStream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/cuda_utils.h"
#include "hetu/impl/utils/offset_calculator.cuh"
#include "hetu/impl/kernel/Vectorized.cuh"

namespace hetu {
namespace impl {

constexpr int CONCAT_BATCH_SIZE = 32;
constexpr int MAX_CONCAT_NDIM = 4;

struct ConcatSizeStride {
  uint64_t shape[MAX_CONCAT_NDIM];
  uint64_t stride[MAX_CONCAT_NDIM];
};

struct ConcatOffsetCalculator {
  static inline __device__ uint64_t compute(
      const uint64_t tensor_size[MAX_CONCAT_NDIM],
      const uint64_t tensor_stride[MAX_CONCAT_NDIM],
      const uint64_t dim_size,
      const unsigned int concat_dim,
      const unsigned int dim,
      uint64_t linear_index) {
    uint64_t offset = 0;

    #pragma unroll
    for (int i = dim - 1; i >= 1; --i) {
      uint64_t cur_dim_size = i == concat_dim ? dim_size : tensor_size[i];
      uint64_t next_dim_index = linear_index / cur_dim_size;
      uint64_t cur_dim_index = linear_index - cur_dim_size * next_dim_index;
      uint64_t cur_dim_offset = cur_dim_index * tensor_stride[i];
      offset += cur_dim_offset;
      linear_index = next_dim_index;
    }

    return offset + linear_index * tensor_stride[0];
  }
};

template <typename spec_t, int batch_size>
struct ConcatInputMeta {
  const spec_t* input[batch_size];
  uint64_t offset[batch_size];
  uint64_t dim_size[batch_size];
  uint64_t numel[batch_size];
};

template <typename spec_t, int batch_size>
struct ConcatInputMetaWithOffsetCalculator {
  const spec_t* input[batch_size];
  uint64_t offset[batch_size];
  uint64_t dim_size[batch_size];
  uint64_t numel[batch_size];
  bool is_contiguous[batch_size];
  ConcatSizeStride size_stride[batch_size];
};

template <typename spec_t, int batch_size>
struct ConcatGradientInputMeta {
  spec_t* grad_input[batch_size];
  uint64_t offset[batch_size];
  uint64_t dim_size[batch_size];
  uint64_t numel[batch_size];
};

template <typename spec_t, int batch_size>
struct ConcatGradientInputMetaWithOffsetCalculator {
  spec_t* grad_input[batch_size];
  uint64_t offset[batch_size];
  uint64_t dim_size[batch_size];
  uint64_t numel[batch_size];
  bool is_contiguous[batch_size];
  ConcatSizeStride size_stride[batch_size];
};

template <typename spec_t>
inline std::tuple<dim3, dim3> get_grid_config(
  unsigned int max_elements_per_tensor,
  int input_num,
  int multiProcessorCount) {
  
  constexpr unsigned int threads_per_block = 256;
  constexpr unsigned int thread_work_size = 4;
  constexpr unsigned int max_tb_per_sm = 32;

  unsigned int max_threads = (max_elements_per_tensor + thread_work_size - 1) / thread_work_size;
  unsigned int thread_blocks = (max_threads + threads_per_block - 1) / threads_per_block;

  thread_blocks = std::min(multiProcessorCount * max_tb_per_sm, thread_blocks);

  dim3 block = dim3(threads_per_block);
  dim3 grid = dim3(thread_blocks, (long long)input_num);

  return std::make_tuple(grid, block);
}

template <typename spec_t, int batch_size>
__global__ void concat_batched_copy_contig(
  spec_t* __restrict__ output,
  const ConcatInputMeta<spec_t, batch_size> inputs,
  const ConcatSizeStride output_size_stride,
  const int concat_dim,
  const int ndim,
  const uint64_t dim_stride) {

  const uint64_t base_tid = blockIdx.x * blockDim.x * 4 + threadIdx.x;
  const uint64_t numel = inputs.numel[blockIdx.y];
  
  if (base_tid >= numel) return;

  const spec_t* __restrict__ data = inputs.input[blockIdx.y];
  const uint64_t data_offset = inputs.offset[blockIdx.y] * dim_stride;
  const uint64_t base_stride = blockDim.x * gridDim.x * 4;

  #pragma unroll 4
  for (int i = 0; i < 4; ++i) {
    const uint64_t tid = base_tid + i * blockDim.x;
    if (tid >= numel) continue;

    uint64_t element_offset = ConcatOffsetCalculator::compute(output_size_stride.shape, output_size_stride.stride, inputs.dim_size[blockIdx.y], concat_dim, ndim, tid);
    output[data_offset + element_offset] = data[tid];
  }

  for (uint64_t tid = base_tid + base_stride; 
       tid < numel; 
       tid += base_stride) {
    const uint64_t element_offset = ConcatOffsetCalculator::compute(output_size_stride.shape, output_size_stride.stride, inputs.dim_size[blockIdx.y], concat_dim, ndim, tid);
    output[data_offset + element_offset] = data[tid];
  }
}

template <typename spec_t, int batch_size>
__global__ void concat_batched_copy(
  spec_t* __restrict__ output,
  const ConcatInputMetaWithOffsetCalculator<spec_t, batch_size> inputs,
  const ConcatSizeStride output_size_stride,
  const int concat_dim,
  const int ndim,
  const uint64_t dim_stride) {

  const uint64_t base_tid = blockIdx.x * blockDim.x * 4 + threadIdx.x;
  const uint64_t numel = inputs.numel[blockIdx.y];
  const bool is_contig = inputs.is_contiguous[blockIdx.y];
  
  if (base_tid >= numel) return;

  const spec_t* __restrict__ data = inputs.input[blockIdx.y];
  const uint64_t data_offset = inputs.offset[blockIdx.y] * dim_stride;
  const uint64_t base_stride = blockDim.x * gridDim.x * 4;

  ConcatSizeStride in_size_stride = inputs.size_stride[blockIdx.y];

  #pragma unroll 4
  for (int i = 0; i < 4; ++i) {
    const uint64_t tid = base_tid + i * blockDim.x;
    if (tid >= numel) continue;

    uint64_t element_offset = ConcatOffsetCalculator::compute(output_size_stride.shape, output_size_stride.stride, inputs.dim_size[blockIdx.y], concat_dim, ndim, tid);

    if (is_contig) {
      output[data_offset + element_offset] = data[tid];
    } else {
      const uint64_t in_offset = ConcatOffsetCalculator::compute(in_size_stride.shape, in_size_stride.stride, inputs.dim_size[blockIdx.y], concat_dim, ndim, tid);
      output[data_offset + element_offset] = data[in_offset];
    }
  }

  for (uint64_t tid = base_tid + base_stride; 
       tid < numel; 
       tid += base_stride) {
    const uint64_t element_offset = ConcatOffsetCalculator::compute(output_size_stride.shape, output_size_stride.stride, inputs.dim_size[blockIdx.y], concat_dim, ndim, tid);
    if (is_contig) {
      output[data_offset + element_offset] = data[tid];
    } else {
      const uint64_t in_offset = ConcatOffsetCalculator::compute(in_size_stride.shape, in_size_stride.stride, inputs.dim_size[blockIdx.y], concat_dim, ndim, tid);
      output[data_offset + element_offset] = data[in_offset];
    }
  }
}

template <typename spec_t, int batch_size, int vec_size>
__global__ void concat_batched_copy_vectorized(
  spec_t* output,
  ConcatInputMeta<spec_t, batch_size> inputs,
  const ConcatSizeStride output_size_stride,
  const int concat_dim,
  const int ndim,
  const uint64_t dim_stride) {

  const int bid = blockIdx.y;
  const spec_t* data = inputs.input[bid];
  const uint64_t data_offset = inputs.offset[bid] * dim_stride;
  const uint64_t numel = inputs.numel[bid];
  const uint64_t dim_size = inputs.dim_size[bid];
  
  uint64_t tid = (blockIdx.x * blockDim.x + threadIdx.x) * vec_size;
  if (tid >= numel) return;

  using vec_t = aligned_vector<spec_t, vec_size>;
  const vec_t* data_vec = reinterpret_cast<const vec_t*>(data);
  vec_t* out_vec = reinterpret_cast<vec_t*>(output + data_offset);

  while (tid < numel) {
    const uint64_t valid_count = min(static_cast<uint64_t>(vec_size), numel - tid);
    const uint64_t base_out_offset = ConcatOffsetCalculator::compute(output_size_stride.shape, output_size_stride.stride, dim_size, concat_dim, ndim, tid);

    if (valid_count == vec_size) {
      const vec_t val = data_vec[tid / vec_size];
      *reinterpret_cast<vec_t*>(&output[data_offset + base_out_offset]) = val;
    } else {
      #pragma unroll(vec_size)
      for (int i = 0; i < vec_size; ++i) {
        if (i < valid_count) {
          const uint64_t element_offset = ConcatOffsetCalculator::compute(output_size_stride.shape, output_size_stride.stride, dim_size, concat_dim, ndim, tid + i);
          output[data_offset + element_offset] = data[tid + i];
        }
      }
    }
    tid += blockDim.x * gridDim.x * vec_size;
  }
}

template <typename spec_t, int batch_size>
void parallel_concat(const NDArrayList& inputs, NDArray& output,
                     size_t dim, bool is_contig, const Stream& stream) {
  int ndim = output->ndim();
  HT_ASSERT(ndim <= MAX_CONCAT_NDIM)
    << "Currently only support up to 4D concat, but got "
    << ndim << "D concat";

  ConcatSizeStride output_size_stride;
  for (size_t i = 0; i < output->ndim(); ++i) {
    output_size_stride.shape[i] = output->shape(i);
    output_size_stride.stride[i] = output->stride(i);
  }

  if (is_contig) {
    ConcatInputMeta<spec_t, batch_size> inputs_meta;
    uint64_t dim_stride = output->stride(dim);
    uint64_t max_elements_per_tensor = 0;
    int batch_counter = 0;
    int64_t offset = 0;

    for (unsigned i = 0; i < inputs.size(); i += batch_size) {
      NDArrayList data{};
      for (batch_counter = 0;
          batch_counter < batch_size && (i + batch_counter) < inputs.size();
          ++batch_counter) {
        int64_t dim_size = 0;
        if (inputs[i + batch_counter]->numel() > 0) {
          dim_size = inputs[i + batch_counter]->shape(dim);
        }

        inputs_meta.input[batch_counter] = inputs[i + batch_counter]->data_ptr<spec_t>();
        inputs_meta.offset[batch_counter] = offset;
        inputs_meta.dim_size[batch_counter] = dim_size;
        inputs_meta.numel[batch_counter] = inputs[i + batch_counter]->numel();
        data.push_back(inputs[i + batch_counter]);
        offset += dim_size;
        max_elements_per_tensor = std::max(
          max_elements_per_tensor,
          inputs_meta.numel[batch_counter]);
      }

      // Skip if the tensor is empty. Otherwise, the grid dim is invalid
      if (max_elements_per_tensor == 0)
        continue;
      
      CUDAStream cuda_stream(stream);
      hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
      
      dim3 block, grid;
      int multiProcessorCount;
      CudaDeviceGetAttribute(&multiProcessorCount,
        cudaDevAttrMultiProcessorCount, cuda_stream.device_id());
      if (sizeof(spec_t) > 2) {
        std::tie(grid, block) = get_grid_config<spec_t>(
          max_elements_per_tensor, batch_counter, multiProcessorCount);
      } else {
        block = dim3(32 * 16);
        grid = dim3(2LL * multiProcessorCount, (long long) batch_counter);
      }
      
      int vec_size = 4;
      for (int i = 0; i < batch_counter; i++) {
        vec_size = std::min(vec_size, get_vectorize_size(inputs_meta.input[i]));
      }
      switch (vec_size) {
        case 4:
          concat_batched_copy_vectorized<spec_t, batch_size, 4><<<grid, block, 0, cuda_stream>>>(
            output->data_ptr<spec_t>(), inputs_meta, output_size_stride, dim, ndim, dim_stride);
          break;
        case 2:
          concat_batched_copy_vectorized<spec_t, batch_size, 2><<<grid, block, 0, cuda_stream>>>(
            output->data_ptr<spec_t>(), inputs_meta, output_size_stride, dim, ndim, dim_stride);
          break;
        case 1:
          concat_batched_copy_contig<spec_t, batch_size><<<grid, block, 0, cuda_stream>>>(
            output->data_ptr<spec_t>(), inputs_meta, output_size_stride, dim, ndim, dim_stride);
          break;
        default:
          HT_RUNTIME_ERROR << "Unexpected vectorization size";
          __builtin_unreachable();
      }
    }
  } else {
    ConcatInputMetaWithOffsetCalculator<spec_t, batch_size> inputs_meta;
    uint64_t dim_stride = output->stride(dim);
    uint64_t max_elements_per_tensor = 0;
    int batch_counter = 0;
    int64_t offset = 0;
    for (unsigned i = 0; i < inputs.size(); i += batch_size) {
      NDArrayList data{};
      for (batch_counter = 0;
          batch_counter < batch_size && (i + batch_counter) < inputs.size();
          ++batch_counter) {
        int64_t dim_size = 0;
        if (inputs[i + batch_counter]->numel() > 0) {
          dim_size = inputs[i + batch_counter]->shape(dim);
        }

        inputs_meta.input[batch_counter] = inputs[i + batch_counter]->data_ptr<spec_t>();
        inputs_meta.offset[batch_counter] = offset;
        inputs_meta.dim_size[batch_counter] = dim_size;
        inputs_meta.numel[batch_counter] = inputs[i + batch_counter]->numel();
        inputs_meta.is_contiguous[batch_counter] = inputs[i + batch_counter]->is_contiguous();
        for (size_t j = 0; j < inputs[i + batch_counter]->ndim(); ++j) {
          inputs_meta.size_stride[batch_counter].shape[j] = inputs[i + batch_counter]->shape(j);
          inputs_meta.size_stride[batch_counter].stride[j] = inputs[i + batch_counter]->stride(j);
        }
        data.push_back(inputs[i + batch_counter]);
        offset += dim_size;
        max_elements_per_tensor = std::max(max_elements_per_tensor, inputs_meta.numel[batch_counter]);
      }

      // Skip if the tensor is empty. Otherwise, the grid dim is invalid
      if (max_elements_per_tensor == 0)
        continue;
      
      CUDAStream cuda_stream(stream);
      hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
      
      dim3 block, grid;
      int multiProcessorCount;
      CudaDeviceGetAttribute(&multiProcessorCount,
        cudaDevAttrMultiProcessorCount, cuda_stream.device_id());
      block = dim3(32 * 16);
      grid = dim3(2LL * multiProcessorCount, (long long) batch_counter);
      
      concat_batched_copy<spec_t, batch_size><<<grid, block, 0, cuda_stream>>>(
        output->data_ptr<spec_t>(), inputs_meta, output_size_stride, dim, ndim, dim_stride);
    }
  }
}

void ConcatCuda(const NDArrayList& inputs, NDArray& output,
                size_t dim, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(output);
  for (auto& input : inputs) {
    HT_ASSERT_SAME_DEVICE(input, output);
  }
  auto dtype = inputs[0]->dtype();
  for (const auto& input : inputs) {
    HT_ASSERT_SAME_DTYPE(input, output);
  }
  if (output->numel() == 0) {
    return;  
  }

  auto input_num = inputs.size();
  bool all_contiguous = true;
  for (const auto& input : inputs) 
    all_contiguous &= input->is_contiguous();
  
  for (const auto& input : inputs) {
    if (input->numel() == 0) continue;
    HT_ASSERT(input->ndim() == output->ndim())
      << "All tensors must have same number of dimensions";
    for (size_t i = 0; i < input->ndim(); i++) {
      if (i == dim) continue;
      HT_ASSERT(input->shape(i) == output->shape(i))
        << "All tensors must have same shape except concat dimension";
    }
  }
  
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    output->dtype(), spec_t, "ConcatCuda", [&]() {
      if (input_num > 1 && (all_contiguous || output->is_contiguous())) {
        if (all_contiguous) {
          parallel_concat<spec_t, CONCAT_BATCH_SIZE>(
            inputs, output, dim, all_contiguous, stream);
        } else {
          parallel_concat<spec_t, CONCAT_BATCH_SIZE / 2>(
            inputs, output, dim, all_contiguous, stream);
        }
      } else {
        int64_t offset = 0;
        for (const auto& input : inputs) {
          if (input->numel() == 0 && input->ndim() == 1)
            continue;
          int64_t dim_size = input->shape(dim);
          auto begin_pos = HTShape(output->shape().size(), 0);
          begin_pos[dim] = offset;
          auto slice_shape = output->shape();
          slice_shape[dim] = dim_size;
          NDArray slice_out = NDArray::slice(output, begin_pos, slice_shape, stream.stream_index());
          NDArray::copy(input, stream.stream_index(), slice_out);
          offset += dim_size;
        }
      }
  });
  NDArray::MarkUsedBy(inputs, stream);
  NDArray::MarkUsedBy({output}, stream);
}

template <typename spec_t, int batch_size>
__global__ void concat_gradient_batched_copy_contig(
  const spec_t* output_grad,
  ConcatGradientInputMeta<spec_t, batch_size> grads,
  const ConcatSizeStride out_grad_size_stride,
  const int concat_dim,
  const int ndim,
  uint64_t dim_stride) {

  const uint64_t base_tid = blockIdx.x * blockDim.x * 4 + threadIdx.x;
  const uint64_t numel = grads.numel[blockIdx.y];
  
  if (base_tid >= numel) return;

  spec_t* grad = grads.grad_input[blockIdx.y];
  const uint64_t data_offset = grads.offset[blockIdx.y] * dim_stride;
  const uint64_t base_stride = blockDim.x * gridDim.x * 4;

  #pragma unroll 4
  for (int i = 0; i < 4; ++i) {
    const uint64_t tid = base_tid + i * blockDim.x;
    if (tid >= numel) continue;
    
    const uint64_t element_offset = ConcatOffsetCalculator::compute(out_grad_size_stride.shape, out_grad_size_stride.stride, grads.dim_size[blockIdx.y], concat_dim, ndim, tid);
    grad[tid] = output_grad[data_offset + element_offset];
  }

  for (uint64_t tid = base_tid + base_stride; 
       tid < numel; 
       tid += base_stride) {
    const uint64_t element_offset = ConcatOffsetCalculator::compute(out_grad_size_stride.shape, out_grad_size_stride.stride, grads.dim_size[blockIdx.y], concat_dim, ndim, tid);
    grad[tid] = output_grad[data_offset + element_offset];
  }
}

template <typename spec_t, int batch_size>
__global__ void concat_gradient_batched_copy(
  const spec_t* output_grad,
  ConcatGradientInputMetaWithOffsetCalculator<spec_t, batch_size> grads,
  const ConcatSizeStride out_grad_size_stride,
  const int concat_dim,
  const int ndim,
  uint64_t dim_stride) {

  const uint64_t base_tid = blockIdx.x * blockDim.x * 4 + threadIdx.x;
  const uint64_t numel = grads.numel[blockIdx.y];
  const bool is_contig = grads.is_contiguous[blockIdx.y];
  
  if (base_tid >= numel) return;

  spec_t* grad = grads.grad_input[blockIdx.y];
  const uint64_t data_offset = grads.offset[blockIdx.y] * dim_stride;
  const uint64_t base_stride = blockDim.x * gridDim.x * 4;

  ConcatSizeStride grad_size_stride = grads.size_stride[blockIdx.y];

  #pragma unroll 4
  for (int i = 0; i < 4; ++i) {
    const uint64_t tid = base_tid + i * blockDim.x;
    if (tid >= numel) continue;
    
    const uint64_t element_offset = ConcatOffsetCalculator::compute(grad_size_stride.shape, grad_size_stride.stride, grads.dim_size[blockIdx.y], concat_dim, ndim, tid);

    if (is_contig) {
      grad[tid] = output_grad[data_offset + element_offset];
    } else {
      const uint64_t in_offset = ConcatOffsetCalculator::compute(grad_size_stride.shape, grad_size_stride.stride, grads.dim_size[blockIdx.y], concat_dim, ndim, tid);
      grad[in_offset] = output_grad[data_offset + element_offset];
    }
  }

  for (uint64_t tid = base_tid + base_stride; 
       tid < numel; 
       tid += base_stride) {
    const uint64_t element_offset = ConcatOffsetCalculator::compute(grad_size_stride.shape, grad_size_stride.stride, grads.dim_size[blockIdx.y], concat_dim, ndim, tid);

    if (is_contig) {
      grad[tid] = output_grad[data_offset + element_offset];
    } else {
      const uint64_t in_offset = ConcatOffsetCalculator::compute(grad_size_stride.shape, grad_size_stride.stride, grads.dim_size[blockIdx.y], concat_dim, ndim, tid);
      grad[in_offset] = output_grad[data_offset + element_offset];
    }
  }
}

template <typename spec_t, int batch_size, int vec_size>
__global__ void concat_gradient_batched_copy_vectorized(
    const spec_t* output_grad,
    ConcatGradientInputMeta<spec_t, batch_size> grads,
    const ConcatSizeStride out_grad_size_stride,
    const int concat_dim,
    const int ndim,
    uint64_t dim_stride) {

  const int bid = blockIdx.y;
  spec_t* grad = grads.grad_input[bid];
  const uint64_t data_offset = grads.offset[bid] * dim_stride;
  const uint64_t numel = grads.numel[bid];
  const uint64_t dim_size = grads.dim_size[bid];
  
  uint64_t tid = (blockIdx.x * blockDim.x + threadIdx.x) * vec_size;
  if (tid >= numel) return;

  using vec_t = aligned_vector<spec_t, vec_size>;
  vec_t* grad_vec = reinterpret_cast<vec_t*>(grad);
  const vec_t* out_grad_vec = reinterpret_cast<const vec_t*>(output_grad + data_offset);

  while (tid < numel) {
    const uint64_t valid_count = min(static_cast<uint64_t>(vec_size), numel - tid);
    const uint64_t base_out_offset = ConcatOffsetCalculator::compute(out_grad_size_stride.shape, out_grad_size_stride.stride, dim_size, concat_dim, ndim, tid);

    if (valid_count == vec_size) {
      const vec_t val = out_grad_vec[base_out_offset / vec_size];
      grad_vec[tid / vec_size] = val;
    } else {
      #pragma unroll(vec_size)
      for (int i = 0; i < vec_size; ++i) {
        if (i < valid_count) {
          const uint64_t element_offset = ConcatOffsetCalculator::compute(out_grad_size_stride.shape, out_grad_size_stride.stride, dim_size, concat_dim, ndim, tid + i);
          grad[tid + i] = output_grad[data_offset + element_offset];
        }
      }
    }
    tid += blockDim.x * gridDim.x * vec_size;
  }
}

template <typename spec_t, int batch_size>
void parallel_concat_gradient(
    const NDArray& output_grad,
    NDArrayList& input_grads,
    size_t dim,
    bool is_contig,
    const Stream& stream) {
  
  int ndim = output_grad->ndim();
  HT_ASSERT(ndim <= MAX_CONCAT_NDIM)
    << "Currently only support up to 4D concat, but got "
    << ndim << "D concat";
  
  ConcatSizeStride out_grad_size_stride;
  for (size_t i = 0; i < output_grad->ndim(); ++i) {
    out_grad_size_stride.shape[i] = output_grad->shape(i);
    out_grad_size_stride.stride[i] = output_grad->stride(i);
  }

  if (is_contig) {
    ConcatGradientInputMeta<spec_t, batch_size> grads_meta;
    uint64_t dim_stride = output_grad->stride(dim);
    uint64_t max_elements_per_tensor = 0;
    int batch_counter = 0;
    int64_t offset = 0;

    for (unsigned i = 0; i < input_grads.size(); i += batch_size) {
      NDArrayList grad_batch{};
      for (batch_counter = 0;
          batch_counter < batch_size && (i + batch_counter) < input_grads.size();
          ++batch_counter) {
        int64_t dim_size = 0;
        if (input_grads[i + batch_counter]->numel() > 0) {
          dim_size = input_grads[i + batch_counter]->shape(dim);
        }

        grads_meta.grad_input[batch_counter] = input_grads[i + batch_counter]->data_ptr<spec_t>();
        grads_meta.offset[batch_counter] = offset;
        grads_meta.dim_size[batch_counter] = dim_size;
        grads_meta.numel[batch_counter] = input_grads[i + batch_counter]->numel();
        grad_batch.push_back(input_grads[i + batch_counter]);
        offset += dim_size;
        max_elements_per_tensor = std::max(
          max_elements_per_tensor, 
          grads_meta.numel[batch_counter]);
      }

      if (max_elements_per_tensor == 0) {
        continue;
      }

      CUDAStream cuda_stream(stream);
      hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());

      dim3 block, grid;
      int multiProcessorCount;
      CudaDeviceGetAttribute(
        &multiProcessorCount,
        cudaDevAttrMultiProcessorCount,
        cuda_stream.device_id());

      if (sizeof(spec_t) > 2) {
        std::tie(grid, block) = get_grid_config<spec_t>(
          max_elements_per_tensor,
          batch_counter,
          multiProcessorCount);
      } else {
        block = dim3(32 * 16);
        grid = dim3(2LL * multiProcessorCount, (long long)batch_counter);
      }

      int vec_size = 4;
      for (int i = 0; i < batch_counter; i++) {
        vec_size = std::min(vec_size, get_vectorize_size(grads_meta.grad_input[i]));
      }
      switch (vec_size) {
        case 4:
          concat_gradient_batched_copy_vectorized<spec_t, batch_size, 4>
            <<<grid, block, 0, cuda_stream>>>(
              output_grad->data_ptr<spec_t>(),
              grads_meta,
              out_grad_size_stride,
              dim,
              ndim,
              dim_stride);
          break;
        case 2:
          concat_gradient_batched_copy_vectorized<spec_t, batch_size, 2>
            <<<grid, block, 0, cuda_stream>>>(
              output_grad->data_ptr<spec_t>(),
              grads_meta,
              out_grad_size_stride,
              dim,
              ndim,
              dim_stride);
          break;
        case 1:
          concat_gradient_batched_copy_contig<spec_t, batch_size>
            <<<grid, block, 0, cuda_stream>>>(
              output_grad->data_ptr<spec_t>(),
              grads_meta,
              out_grad_size_stride,
              dim,
              ndim,
              dim_stride);
          break;
        default:
          HT_RUNTIME_ERROR << "Unexpected vectorization size";
          __builtin_unreachable();
      }
    }
  } else {
    ConcatGradientInputMetaWithOffsetCalculator<spec_t, batch_size> grads_meta;
    uint64_t dim_stride = output_grad->stride(dim);
    uint64_t max_elements_per_tensor = 0;
    int batch_counter = 0;
    int64_t offset = 0;
    
    for (unsigned i = 0; i < input_grads.size(); i += batch_size) {
      NDArrayList grad_batch{};
      for (batch_counter = 0;
          batch_counter < batch_size && (i + batch_counter) < input_grads.size();
          ++batch_counter) {
        int64_t dim_size = 0;
        if (input_grads[i + batch_counter]->numel() > 0) {
          dim_size = input_grads[i + batch_counter]->shape(dim);
        }

        grads_meta.grad_input[batch_counter] = input_grads[i + batch_counter]->data_ptr<spec_t>();
        grads_meta.offset[batch_counter] = offset;
        grads_meta.dim_size[batch_counter] = dim_size;
        grads_meta.numel[batch_counter] = input_grads[i + batch_counter]->numel();
        grads_meta.is_contiguous[batch_counter] = input_grads[i + batch_counter]->is_contiguous();
        for (size_t j = 0; j < input_grads[i + batch_counter]->ndim(); ++j) {
          grads_meta.size_stride[batch_counter].shape[j] = input_grads[i + batch_counter]->shape(j);
          grads_meta.size_stride[batch_counter].stride[j] = input_grads[i + batch_counter]->stride(j);
        }
        grad_batch.push_back(input_grads[i + batch_counter]);
        offset += dim_size;
        max_elements_per_tensor = std::max(
          max_elements_per_tensor, 
          grads_meta.numel[batch_counter]);
      }

      if (max_elements_per_tensor == 0) {
        continue;
      }

      CUDAStream cuda_stream(stream);
      hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());

      dim3 block, grid;
      int multiProcessorCount;
      CudaDeviceGetAttribute(
        &multiProcessorCount,
        cudaDevAttrMultiProcessorCount,
        cuda_stream.device_id());

      block = dim3(32 * 16);
      grid = dim3(2LL * multiProcessorCount, (long long)batch_counter);

      concat_gradient_batched_copy<spec_t, batch_size>
        <<<grid, block, 0, cuda_stream>>>(
          output_grad->data_ptr<spec_t>(),
          grads_meta,
          out_grad_size_stride,
          dim,
          ndim,
          dim_stride);
    }
  }
}

void ConcatGradientCuda(
    const NDArray& output_grad,
    NDArrayList& input_grads,
    size_t dim,
    const Stream& stream) {

  HT_ASSERT_CUDA_DEVICE(output_grad);
  for (auto& grad : input_grads) {
    HT_ASSERT_SAME_DEVICE(grad, output_grad);
    HT_ASSERT_SAME_DTYPE(grad, output_grad);
  }
  if (output_grad->numel() == 0) {
    return;
  }

  bool has_data = false;
  for (const auto& grad : input_grads) {
    if (grad->numel() > 0) {
      has_data = true;
      break;
    }
  }
  if (!has_data) {
    return;
  }

  bool all_contiguous = true;
  for (const auto& grad : input_grads)
    all_contiguous &= grad->is_contiguous();

  for (const auto& grad : input_grads) {
    if (grad->numel() == 0) continue;
    HT_ASSERT(grad->ndim() == output_grad->ndim())
      << "All tensors must have same number of dimensions";
    for (size_t i = 0; i < grad->ndim(); i++) {
      if (i == dim) continue;
      HT_ASSERT(grad->shape(i) == output_grad->shape(i))
        << "All tensors must have same shape except concat dimension";
    }
  }

  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    output_grad->dtype(), spec_t, "ConcatGradientCuda", [&]() {
      if (input_grads.size() > 1 || (all_contiguous && output_grad->is_contiguous())) {
        if (all_contiguous) {
          parallel_concat_gradient<spec_t, CONCAT_BATCH_SIZE>(
            output_grad, 
            input_grads, 
            dim, 
            all_contiguous, 
            stream
          );
        } else {
          parallel_concat_gradient<spec_t, CONCAT_BATCH_SIZE/2>(
            output_grad, 
            input_grads, 
            dim, 
            all_contiguous, 
            stream
          );
        }
      } else {
        int64_t offset = 0;
        for (auto& input_grad : input_grads) {
          if (input_grad->numel() == 0 && input_grad->ndim() == 1)
            continue;
          int64_t dim_size = input_grad->shape(dim);
          auto begin_pos = HTShape(output_grad->shape().size(), 0);
          begin_pos[dim] = offset;
          auto slice_shape = output_grad->shape();
          slice_shape[dim] = dim_size;
          NDArray slice_out_grad = NDArray::slice(output_grad, begin_pos, slice_shape, stream.stream_index());
          NDArray::copy(slice_out_grad, stream.stream_index(), input_grad);
          offset += dim_size;
        }
      }
  });

  NDArray::MarkUsedBy({output_grad}, stream);
  NDArray::MarkUsedBy(input_grads, stream);
}

} // namespace impl
} // namespace hetu
