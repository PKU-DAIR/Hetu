#include "hetu/core/ndarray.h"
#include "hetu/impl/stream/CUDAStream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/cuda_utils.h"
#include "hetu/impl/utils/offset_calculator.cuh"
#include "hetu/impl/kernel/Vectorized.cuh"

namespace hetu {
namespace impl {

constexpr int CONCAT_BATCH_SIZE = 128;

template <typename spec_t, int batch_size>
struct ConcatInputMeta {
  const spec_t* input[batch_size];
  uint64_t offset[batch_size];
  uint64_t dim_size[batch_size];
  uint64_t numel[batch_size];
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
__global__ void concat_batched_copy(
  spec_t* output,
  ConcatInputMeta<spec_t, batch_size> inputs,
  const int concat_dim,
  uint64_t dim_stride,
  OffsetCalculator** input_offset_calculators,
  const OffsetCalculator* output_offset_calculator) {

  uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t numel = inputs.numel[blockIdx.y];
  if (tid >= numel)
    return;
  
  const spec_t* data = inputs.input[blockIdx.y];
  uint64_t offset = inputs.offset[blockIdx.y];
  uint64_t dim_size = inputs.dim_size[blockIdx.y];
  uint64_t data_offset = offset * dim_stride;
  uint64_t stride = gridDim.x * blockDim.x;

  while (tid < numel) {
    uint64_t element_offset = output_offset_calculator->get(tid, dim_size, concat_dim);
    uint64_t in_offset = input_offset_calculators[blockIdx.y]->get(tid);
    output[data_offset + element_offset] = data[in_offset];
    tid += stride;
  }
}

template <typename spec_t, int batch_size, int vec_size>
__global__ void concat_batched_copy_vectorized(
  spec_t* output,
  ConcatInputMeta<spec_t, batch_size> inputs,
  const int concat_dim,
  uint64_t dim_stride,
  OffsetCalculator** input_offset_calculators,
  const OffsetCalculator* output_offset_calculator) {

  uint64_t in_offset = (blockIdx.x * blockDim.x + threadIdx.x) * vec_size;
  uint64_t in_stride = gridDim.x * blockDim.x * vec_size;
  uint64_t numel = inputs.numel[blockIdx.y];
  if (in_offset >= numel) {
    return;
  }

  const spec_t* data = inputs.input[blockIdx.y];
  uint64_t offset = inputs.offset[blockIdx.y];
  uint64_t dim_size = inputs.dim_size[blockIdx.y];
  uint64_t data_offset = offset * dim_stride;

  uint64_t vec_element_offsets[vec_size];
  spec_t results[vec_size];

  while (in_offset + vec_size <= numel) {
    #pragma unroll
    for (int i = 0; i < vec_size; ++i) {
      vec_element_offsets[i] = output_offset_calculator->get(in_offset + i, dim_size, concat_dim);
    }

    using vec_spec_t = aligned_vector<spec_t, vec_size>;
    ((vec_spec_t*)results)[0] = const_cast<vec_spec_t*>((vec_spec_t*)(data + in_offset))[0];

    #pragma unroll
    for (int i = 0; i < vec_size; ++i) {
      output[data_offset + vec_element_offsets[i]] = results[i];
    }
    in_offset += in_stride;
  }

  while (in_offset < numel) {
    vec_element_offsets[0] = output_offset_calculator->get(in_offset, dim_size, concat_dim);
    output[data_offset + vec_element_offsets[0]] = data[in_offset];
    in_offset++;
  }
}

template <typename spec_t, int batch_size>
void parallel_concat(const NDArrayList& inputs, NDArray& output,
                     size_t dim, bool is_contig, const Stream& stream) {
  ConcatInputMeta<spec_t, batch_size> inputs_meta;
  uint64_t dim_stride = output->stride(dim);
  unsigned int max_elements_per_tensor = 0;

  int batch_counter = 0;
  int64_t offset = 0;
  auto [out_offset_arr, out_offset_calculator] = AllocOffsetCalculator(output, stream, true);
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
      max_elements_per_tensor = std::max(max_elements_per_tensor, inputs_meta.numel[batch_counter]);
    }

    // Skip if the tensor is empty. Otherwise, the grid dim is invalid
    if (max_elements_per_tensor == 0)
      continue;
    
    dim3 block, grid;
    int multiProcessorCount;
    CudaDeviceGetAttribute(&multiProcessorCount,
      cudaDevAttrMultiProcessorCount, cuda_stream.device_id());
    if (is_contig && sizeof(spec_t) > 2) {
      std::tie(grid, block) = get_grid_config<spec_t>(
        max_elements_per_tensor, batch_counter, multiProcessorCount);
    } else {
      block = dim3(32 * 16);
      grid = dim3(2LL * multiProcessorCount, (long long) batch_counter);
    }

    CUDAStream cuda_stream(stream);
    hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
    auto [in_offset_arrs, in_offset_ptrs] = AllocOffsetCalculator(data, stream);
    NDArray in_offset_ptrs_arr = hetu::cuda::to_ptr_ndarray(in_offset_ptrs, device_id);
    auto** in_offset_calculators = in_offset_ptrs_arr->data_ptr<OffsetCalculator*>();

    if (is_contig) {
      int vec_size = get_vectorize_size(inputs...);
      switch (vec_size) {
        case 4:
          concat_batched_copy_vectorized<spec_t, batch_size, 4><<<grid, block, 0, cuda_stream>>>(
            output->data_ptr<spec_t>(), inputs_meta, dim, dim_stride,
            in_offset_calculators, out_offset_calculator);
          break;
        case 2:
          concat_batched_copy_vectorized<spec_t, batch_size, 2><<<grid, block, 0, cuda_stream>>>(
            output->data_ptr<spec_t>(), inputs_meta, dim, dim_stride,
            in_offset_calculators, out_offset_calculator);
          break;
        case 1:
          concat_batched_copy<spec_t, batch_size><<<grid, block, 0, cuda_stream>>>(
            output->data_ptr<spec_t>(), inputs_meta, dim, dim_stride,
            in_offset_calculators, out_offset_calculator);
          break;
        default:
          HT_RUNTIME_ERROR << "Unexpected vectorization size";
          __builtin_unreachable();
      }
    } else {
      concat_batched_copy<spec_t, batch_size><<<grid, block, 0, cuda_stream>>>(
        output->data_ptr<spec_t>(), inputs_meta, dim, dim_stride,
        in_offset_calculators, out_offset_calculator);
    }

    in_offset_arrs.push_back(in_offset_ptrs_arr);
    NDArray::MarkUsedBy(in_offset_arrs, stream);
  }
  NDArray::MarkUsedBy({out_offset_arr}, stream);
}

void ConcatCuda(const NDArrayList& inputs, NDArray& output,
                size_t dim, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(output);
  for (auto& input : inputs) {
    HT_ASSERT_SAME_DEVICE(input, output);
  }
  auto dtype = inputs[0]->dtype();
  for (const auto& input : inputs) {
    HT_ASSERT_SAME_DTYPE(input, dtype);
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
    HT_ASSERT(input->ndim() == output->ndim(),
              "All tensors must have same number of dimensions");
    for (size_t i = 0; i < input->ndim(); i++) {
      if (i == dim) continue;
      HT_ASSERT(input->shape(i) == output->shape(i),
                "All tensors must have same shape except concat dimension");
    }
  }
  
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    output->dtype(), spec_t, "ConcatCuda", [&]() {
      if (input_num > 1 && (all_contiguous || output->is_contiguous())) {
        if (all_contiguous) {
          parallel_concat<spec_t, CONCAT_BATCH_SIZE>(
            inputs, output, dim, all_contiguous, stream);
        } else {
          parallel_concat<spec_t, CONCAT_BATCH_SIZE/2>(
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
          NDArray slice_out = NDArray::slice(input, begin_pos, slice_shape, stream.stream_index());
          NDArray::copy(input, stream.stream_index(), slice_out);
          offset += dim_size;
        }
      }
  });
  NDArray::MarkUsedBy(inputs, stream);
  NDArray::MarkUsedBy({output}, stream);
}

template <typename spec_t, int batch_size>
__global__ void concat_gradient_batched_copy(
  const spec_t* output_grad,
  ConcatInputMeta<spec_t, batch_size> grads,
  const int concat_dim,
  uint64_t dim_stride,
  const OffsetCalculator* out_grad_offset_calculator,
  OffsetCalculator** in_grad_offset_calculators) {

  uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t numel = grads.numel[blockIdx.y];
  if (tid >= numel)
    return;
  
  const spec_t* grad = grads.input[blockIdx.y];
  uint64_t offset = grads.offset[blockIdx.y];
  uint64_t dim_size = grads.dim_size[blockIdx.y];
  uint64_t data_offset = offset * dim_stride;
  uint64_t stride = gridDim.x * blockDim.x;

  while (tid < numel) {
    uint64_t element_offset = out_grad_offset_calculator->get(tid, dim_size, concat_dim);
    uint64_t in_offset = in_grad_offset_calculators[blockIdx.y]->get(tid);
    grad[in_offset] = output_grad[data_offset + element_offset];
    tid += stride;
  }
}

template <typename spec_t, int batch_size, int vec_size>
__global__ void concat_gradient_batched_copy_vectorized(
    const spec_t* output_grad,
    ConcatInputMeta<spec_t, batch_size> grads,
    const int concat_dim,
    uint64_t dim_stride,
    const OffsetCalculator* out_grad_offset_calculator,
    OffsetCalculator** in_grad_offset_calculators) {

  uint64_t in_offset = (blockIdx.x * blockDim.x + threadIdx.x) * vec_size;
  uint64_t in_stride = gridDim.x * blockDim.x * vec_size;
  uint64_t numel = grads.numel[blockIdx.y];
  if (in_offset >= numel) {
    return;
  }

  spec_t* grad = grads.input[blockIdx.y];
  uint64_t offset = grads.offset[blockIdx.y];
  uint64_t dim_size = grads.dim_size[blockIdx.y];
  uint64_t data_offset = offset * dim_stride;

  uint64_t out_element_offsets[vec_size];
  spec_t values[vec_size];

  while (in_offset + vec_size <= numel) {
    #pragma unroll
    for (int i = 0; i < vec_size; ++i) {
      out_element_offsets[i] = data_offset + out_grad_offset_calculator->get(
        in_offset + i, dim_size, concat_dim);
    }

    #pragma unroll
    for (int i = 0; i < vec_size; ++i) {
      values[i] = output_grad[out_element_offsets[i]];
    }

    using vec_spec_t = aligned_vector<spec_t, vec_size>;
    auto in_grad_offset = in_grad_offset_calculators[blockIdx.y]->get(in_offset);
    ((vec_spec_t*)(grad + in_grad_offset))[0] = ((vec_spec_t*)values)[0];

    in_offset += in_stride;
  }

  while (in_offset < numel) {
    auto out_offset = data_offset + out_grad_offset_calculator->get(
      in_offset, dim_size, concat_dim);
    auto in_grad_offset = in_grad_offset_calculators[blockIdx.y]->get(in_offset);
    grad[in_grad_offset] = output_grad[out_offset];
    in_offset++;
  }
}

template <typename spec_t, int batch_size>
void parallel_concat_gradient(
    const NDArray& output_grad,
    NDArrayList& input_grads,
    size_t dim,
    bool is_contig,
    const Stream& stream) {
  
  ConcatInputMeta<spec_t, batch_size> grads_meta;
  uint64_t dim_stride = output_grad->stride(dim);
  unsigned int max_elements_per_tensor = 0;

  int batch_counter = 0;
  int64_t offset = 0;
  
  auto [out_grad_offset_arr, out_grad_offset_calculator] = 
    AllocOffsetCalculator(output_grad, stream, true);

  for (unsigned i = 0; i < input_grads.size(); i += batch_size) {
    NDArrayList grad_batch{};
    
    for (batch_counter = 0;
         batch_counter < batch_size && (i + batch_counter) < input_grads.size();
         ++batch_counter) {
      
      int64_t dim_size = 0;
      if (input_grads[i + batch_counter]->numel() > 0) {
        dim_size = input_grads[i + batch_counter]->shape(dim);
      }

      grads_meta.input[batch_counter] = input_grads[i + batch_counter]->data_ptr<spec_t>();
      grads_meta.offset[batch_counter] = offset;
      grads_meta.dim_size[batch_counter] = dim_size;
      grads_meta.numel[batch_counter] = input_grads[i + batch_counter]->numel();
      grad_batch.push_back(input_grads[i + batch_counter]);
      
      offset += dim_size;
      max_elements_per_tensor = std::max(
        max_elements_per_tensor, 
        grads_meta.numel[batch_counter]
      );
    }

    if (max_elements_per_tensor == 0) {
      continue;
    }

    dim3 block, grid;
    int multiProcessorCount;
    CUDAStream cuda_stream(stream);
    CudaDeviceGetAttribute(
      &multiProcessorCount,
      cudaDevAttrMultiProcessorCount,
      cuda_stream.device_id()
    );

    if (is_contig && sizeof(spec_t) > 2) {
      std::tie(grid, block) = get_grid_config<spec_t>(
        max_elements_per_tensor,
        batch_counter,
        multiProcessorCount
      );
    } else {
      block = dim3(32 * 16);
      grid = dim3(2LL * multiProcessorCount, (long long)batch_counter);
    }

    hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());

    auto [in_grad_offset_arrs, in_grad_offset_ptrs] = 
      AllocOffsetCalculator(grad_batch, stream);
    NDArray in_grad_offset_ptrs_arr = 
      hetu::cuda::to_ptr_ndarray(in_grad_offset_ptrs, cuda_stream.device_id());
    auto** in_grad_offset_calculators = 
      in_grad_offset_ptrs_arr->data_ptr<OffsetCalculator*>();

    if (is_contig) {
      int vec_size = get_vectorize_size(input_grads...);
      switch (vec_size) {
        case 4:
          concat_gradient_batched_copy_vectorized<spec_t, batch_size, 4>
            <<<grid, block, 0, cuda_stream>>>(
              output_grad->data_ptr<spec_t>(),
              grads_meta,
              dim,
              dim_stride,
              out_grad_offset_calculator,
              in_grad_offset_calculators
            );
          break;
        case 2:
          concat_gradient_batched_copy_vectorized<spec_t, batch_size, 2>
            <<<grid, block, 0, cuda_stream>>>(
              output_grad->data_ptr<spec_t>(),
              grads_meta,
              dim,
              dim_stride,
              out_grad_offset_calculator,
              in_grad_offset_calculators
            );
          break;
        case 1:
          concat_gradient_batched_copy<spec_t, batch_size>
            <<<grid, block, 0, cuda_stream>>>(
              output_grad->data_ptr<spec_t>(),
              grads_meta,
              dim,
              dim_stride,
              out_grad_offset_calculator,
              in_grad_offset_calculators
            );
          break;
        default:
          HT_RUNTIME_ERROR << "Unexpected vectorization size";
          __builtin_unreachable();
      }
    } else {
      concat_gradient_batched_copy<spec_t, batch_size>
        <<<grid, block, 0, cuda_stream>>>(
          output_grad->data_ptr<spec_t>(),
          grads_meta,
          dim,
          dim_stride,
          out_grad_offset_calculator,
          in_grad_offset_calculators
        );
    }

    in_grad_offset_arrs.push_back(in_grad_offset_ptrs_arr);
    NDArray::MarkUsedBy(in_grad_offset_arrs, stream);
  }

  NDArray::MarkUsedBy({out_grad_offset_arr}, stream);
}

void ConcatGradientCuda(
    const NDArray& output_grad,
    NDArrayList& input_grads,
    size_t dim,
    const Stream& stream) {

  HT_ASSERT_CUDA_DEVICE(output_grad);
  for (auto& grad : input_grads) {
    HT_ASSERT_SAME_DEVICE(grad, output_grad);
    HT_ASSERT_SAME_DTYPE(grad, output_grad->dtype());
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
    HT_ASSERT(grad->ndim() == output_grad->ndim(),
              "All tensors must have same number of dimensions");
    for (size_t i = 0; i < grad->ndim(); i++) {
      if (i == dim) continue;
      HT_ASSERT(grad->shape(i) == output_grad->shape(i),
                "All tensors must have same shape except concat dimension");
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
        for (const auto& input_grad : input_grads) {
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
