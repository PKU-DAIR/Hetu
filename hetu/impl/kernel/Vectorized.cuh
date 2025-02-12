#pragma once

#include "hetu/core/ndarray.h"
#include "hetu/core/stream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/stream/CUDAStream.h"
#include "hetu/impl/utils/cuda_utils.h"
#include "hetu/impl/utils/offset_calculator.cuh"
#include <thrust/tuple.h>
#include <tuple>
#include <utility>

namespace hetu {
namespace impl {

static constexpr uint32_t NUM_THREADS = 256;
static constexpr uint32_t THREAD_WORK_SIZE = 4;
static constexpr uint32_t BLOCK_WORK_SIZE = NUM_THREADS * THREAD_WORK_SIZE;

template <typename T, int vt>
struct VectorizedStorage {
  T data[vt];
  
  __device__ T& operator[](int i) { return data[i]; }
  __device__ const T& operator[](int i) const { return data[i]; }
};

template <typename spec_t>
int get_vectorize_size(const spec_t* ptr) {
  uint64_t address = reinterpret_cast<uint64_t>(ptr);
  constexpr int vec2_alignment = std::alignment_of<aligned_vector<spec_t, 2>>::value;
  constexpr int vec4_alignment = std::alignment_of<aligned_vector<spec_t, 4>>::value;
  if (address % vec4_alignment == 0) {
    return 4;
  } else if (address % vec2_alignment == 0) {
    return 2;
  }
  return 1;
}

template <typename spec_t, typename T, typename... Args>
int get_vectorize_size(const spec_t* ptr, const T* arg, const Args*... args) {
  return std::min(get_vectorize_size(ptr),
                  get_vectorize_size(arg, args...));
}

template <typename ins_t, int end, int current=0>
struct load_multiple_inputs {
  __device__ static void apply(void** input_ptrs, ins_t* inputs, int linear_idx, int j) {
    using arg_t = std::tuple_element_t<current, ins_t>;
    std::get<current>(inputs[j]) = *(reinterpret_cast<arg_t*>(input_ptrs[current]) + linear_idx);
    load_multiple_inputs<ins_t, end, current + 1>::apply(input_ptrs, inputs, linear_idx, j);
  }

  __device__ static void apply(void** input_ptrs, ins_t* inputs, int* linear_idx, int j) {
    using arg_t = std::tuple_element_t<current, ins_t>;
    std::get<current>(inputs[j]) = *(reinterpret_cast<arg_t*>(input_ptrs[current]) + linear_idx[current]);
    load_multiple_inputs<ins_t, end, current + 1>::apply(input_ptrs, inputs, linear_idx, j);
  }
};

template <typename ins_t, int end>
struct load_multiple_inputs<ins_t, end, end> {
  __device__ static void apply(void** input_ptrs, ins_t* inputs, int linear_idx, int j) {}

  __device__ static void apply(void** input_ptrs, ins_t* inputs, int* linear_idx, int j) {}
};

// Base case for zero inputs
template <typename ins_t>
struct load_multiple_inputs<ins_t, 0, 0> {
  __device__ static void apply(void** input_ptrs, ins_t* inputs, int linear_idx, int j) {}

  __device__ static void apply(void** input_ptrs, ins_t* inputs, int* linear_idx, int j) {}
};

template <typename outs_t, int num_outputs, int current=0>
struct store_multiple_outputs {
  __device__ static void apply(void** output_ptrs, outs_t outputs, int linear_idx) {
    using out_t = typename thrust::tuple_element<current, outs_t>::type;
    out_t *to = reinterpret_cast<out_t*>(output_ptrs[current]) + linear_idx;
    *to = thrust::get<current>(outputs);
    store_multiple_outputs<outs_t, num_outputs, current + 1>::apply(output_ptrs, outputs, linear_idx);
  }

  __device__ static void apply(void** output_ptrs, outs_t outputs, int* linear_idx) {
    using out_t = typename thrust::tuple_element<current, outs_t>::type;
    out_t *to = reinterpret_cast<out_t*>(output_ptrs[current]) + linear_idx[current];
    *to = thrust::get<current>(outputs);
    store_multiple_outputs<outs_t, num_outputs, current + 1>::apply(output_ptrs, outputs, linear_idx);
  }
};

template <typename outs_t, int num_outputs>
struct store_multiple_outputs<outs_t, num_outputs, num_outputs> {
  __device__ static void apply(void** output_ptrs, outs_t outputs, int linear_idx) {}

  __device__ static void apply(void** output_ptrs, outs_t outputs, int* linear_idx) {}
};

// Helper to cast input pointers to their respective types
template <typename InTuple, typename NDArrayList, size_t... Is>
auto get_input_ptrs_impl(const NDArrayList& inputs, std::index_sequence<Is...>) {
  return std::make_tuple(
    inputs[Is]->template data_ptr<typename std::tuple_element<Is, InTuple>::type>()...
  );
}

template <typename InTuple, typename NDArrayList>
auto get_input_ptrs(const NDArrayList& inputs) {
  return get_input_ptrs_impl<InTuple>(inputs, 
    std::make_index_sequence<std::tuple_size<InTuple>::value>{});
}

template <typename in_t, typename out_t, typename func_t>
__device__ inline void unroll_kernel_for_multi_outputs_impl(void** input_ptrs, void** output_ptrs,
                                                            int remaining, int base_idx, func_t op) {
  constexpr int num_outputs = thrust::tuple_size<out_t>::value;
  constexpr int nargs = std::tuple_size<in_t>::value;

  int thread_idx = threadIdx.x;
  out_t results[THREAD_WORK_SIZE];
  in_t inputs[THREAD_WORK_SIZE];
  #pragma unroll
  for (int i = 0; i < THREAD_WORK_SIZE; i++) {
    int index = thread_idx + i * NUM_THREADS;
    if (index >= remaining) {
      break;
    }
    int linear_idx = index + base_idx;
    load_multiple_inputs<in_t, nargs>::apply(input_ptrs, inputs, linear_idx, i);
    results[i] = hetu::impl::apply(op, inputs[i]);
  }
  #pragma unroll
  for (int i = 0; i < THREAD_WORK_SIZE; i++) {
    int index = thread_idx + i * NUM_THREADS;
    if (index >= remaining) {
      break;
    }
    int linear_idx = index + base_idx;
    store_multiple_outputs<out_t, num_outputs>::apply(output_ptrs, results[i], linear_idx);
  }
}

template <typename in_t, typename out_t, int num_outputs, int num_inputs, typename func_t>
__global__ void unroll_kernel_for_multi_outputs(void** input_ptrs, void** output_ptrs,
                                                size_t size, func_t op) {
  int base_idx = BLOCK_WORK_SIZE * blockIdx.x;
  int remaining = size - base_idx;
  unroll_kernel_for_multi_outputs_impl<in_t, out_t>(input_ptrs, output_ptrs, remaining, base_idx, op);
}

template <typename func_t, typename out_t, typename... IN_t>
__device__ void unroll_kernel_impl(func_t op, int remaining, int base_idx, out_t* output, const IN_t*... inputs) {
  int thread_idx = threadIdx.x;
  out_t results[THREAD_WORK_SIZE];
  #pragma unroll
  for (int i = 0; i < THREAD_WORK_SIZE; i++) {
    int index = thread_idx + i * NUM_THREADS;
    if (index >= remaining) {
      break;
    }
    int linear_idx = index + base_idx;
    results[i] = op(inputs[linear_idx]...);
  }
  #pragma unroll
  for (int i = 0; i < THREAD_WORK_SIZE; i++) {
    int index = thread_idx + i * NUM_THREADS;
    if (index >= remaining) {
      break;
    }
    int linear_idx = index + base_idx;
    output[linear_idx] = results[i];
  }
}

template <typename func_t, typename out_t>
__device__ void unroll_kernel_impl(func_t op, int remaining, int base_idx, out_t* output) {
  int thread_idx = threadIdx.x;
  out_t results[THREAD_WORK_SIZE];
  #pragma unroll
  for (int i = 0; i < THREAD_WORK_SIZE; i++) {
    int index = thread_idx + i * NUM_THREADS;
    if (index >= remaining) {
      break;
    }
    int linear_idx = index + base_idx;
    results[i] = op(linear_idx);
  }
  #pragma unroll
  for (int i = 0; i < THREAD_WORK_SIZE; i++) {
    int index = thread_idx + i * NUM_THREADS;
    if (index >= remaining) {
      break;
    }
    int linear_idx = index + base_idx;
    output[linear_idx] = results[i];
  }
}

template <typename func_t, typename out_t, typename... IN_t>
__global__ void unroll_kernel(func_t op, size_t size, out_t* output, const IN_t*... inputs) {
  int base_idx = BLOCK_WORK_SIZE * blockIdx.x;
  int remaining = size - base_idx;
  unroll_kernel_impl(op, remaining, base_idx, output, inputs...);
}

template <int vec_size, typename func_t, typename out_t, typename... IN_t>
__device__ void vectorize_kernel_impl(func_t op, aligned_vector<out_t, vec_size>* output,
                                      const aligned_vector<IN_t, vec_size>*... inputs) {
  int loop_size = THREAD_WORK_SIZE / vec_size;
  #pragma unroll
  for (int i = 0; i < loop_size; i++) {
    int index = threadIdx.x + i * NUM_THREADS;
    aligned_vector<out_t, vec_size> ret;
    #pragma unroll
    for (int j = 0; j < vec_size; j++) {
      ret.val[j] = op(inputs[index].val[j]...);
    }
    output[index] = ret;
  }
}

template <int vec_size, typename func_t, typename out_t>
__device__ void vectorize_kernel_impl(func_t op, aligned_vector<out_t, vec_size>* output) {
  int base_idx = BLOCK_WORK_SIZE * blockIdx.x;
  int loop_size = THREAD_WORK_SIZE / vec_size;
  #pragma unroll
  for (int i = 0; i < loop_size; i++) {
    int index = threadIdx.x + i * NUM_THREADS;
    int linear_idx = base_idx + index * vec_size;
    aligned_vector<out_t, vec_size> ret;
    #pragma unroll
    for (int j = 0; j < vec_size; j++) {
      ret.val[j] = op(linear_idx + j);
    }
    output[index] = ret;
  }
}

template <int vec_size, typename func_t, typename out_t, typename... IN_t>
__global__ void vectorize_kernel(func_t op, size_t size, out_t* output, const IN_t*... inputs) {
  int base_idx = BLOCK_WORK_SIZE * blockIdx.x;
  int remaining = size - base_idx;
  if (remaining < BLOCK_WORK_SIZE) {
    // do a naive unrolled loop to handle the reminder
    unroll_kernel_impl(op, remaining, base_idx, output, inputs...);
  } else {
    // use vectorize memory load/store to handle a full `block_work_size` data
    vectorize_kernel_impl<vec_size>(op, reinterpret_cast<aligned_vector<out_t, vec_size>*>(output + base_idx),
                                    reinterpret_cast<const aligned_vector<IN_t, vec_size>*>(
                                    const_cast<IN_t*>(inputs) + base_idx)...);
  }
}

template <typename func_t, typename out_t, typename... IN_t>
static inline void launch_vectorized_kernel(const func_t& op, size_t size, const Stream& stream,
                                            out_t* output, const IN_t*... inputs) {
  int64_t grid = DIVUP(size, BLOCK_WORK_SIZE);
  CUDAStream cuda_stream(stream);
  hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
  int vec_size = get_vectorize_size(output, inputs...);
  switch (vec_size) {
    case 4:
      vectorize_kernel<4><<<grid, NUM_THREADS, 0, cuda_stream>>>(
        op, size, output, inputs...);
      break;
    case 2:
      vectorize_kernel<2><<<grid, NUM_THREADS, 0, cuda_stream>>>(
        op, size, output, inputs...);
      break;
    case 1:
      unroll_kernel<<<grid, NUM_THREADS, 0, cuda_stream>>>(
        op, size, output, inputs...);
      break;
    default:
      HT_RUNTIME_ERROR << "Unexpected vectorization size";
      __builtin_unreachable();
  }
}

template <int nt, int vt, int num_outputs, int num_inputs,
          typename InTuple, typename OutTuple, typename FuncT>
__global__ void loop_kernel(
  void** input_ptrs, void** output_ptrs, size_t size, FuncT op,
  OffsetCalculator** in_offset_calculators, 
  OffsetCalculator** out_offset_calculators) {
    
  int tid = threadIdx.x;
  int nv = nt * vt;
  int idx = nv * blockIdx.x + tid;

  VectorizedStorage<InTuple, vt> inputs;
  VectorizedStorage<OutTuple, vt> results;

  #pragma unroll
  for (int i = 0; i < vt; i++) {
    if (idx < size) {
      // Get offsets for inputs and outputs
      int in_offsets[num_inputs];
      int out_offsets[num_outputs];
            
      #pragma unroll
      for (int j = 0; j < num_inputs; j++) {
        in_offsets[j] = in_offset_calculators[j]->get(idx);
      }
      #pragma unroll
      for (int j = 0; j < num_outputs; j++) {
        out_offsets[j] = out_offset_calculators[j]->get(idx);
      }

      // Load inputs
      load_multiple_inputs<InTuple, num_inputs>::apply(
          input_ptrs, &inputs[i], in_offsets, 0);

      // Apply operation
      results[i] = hetu::impl::apply(op, inputs[i]);

      // Store outputs
      store_multiple_outputs<OutTuple, num_outputs>::apply(
          output_ptrs, results[i], out_offsets);

      idx += nt;
    }
  }
}

template <typename InTuple, typename OutTuple, typename FuncT>
void launch_loop_kernel(
  const NDArrayList& inputs, 
  const NDArrayList& outputs, 
  size_t size,
  const Stream& stream,
  FuncT op) {
    
  constexpr int num_inputs = std::tuple_size<InTuple>::value;
  constexpr int num_outputs = thrust::tuple_size<OutTuple>::value;
    
  HT_ASSERT(num_inputs == inputs.size())
    << "Input count mismatch";
  HT_ASSERT(num_outputs == outputs.size())
    << "Output count mismatch";

  // Check contiguous
  bool all_contiguous = true;
  for (const auto& nd : inputs) 
    all_contiguous &= nd->is_contiguous();
  for (const auto& nd : outputs) 
    all_contiguous &= nd->is_contiguous();

  // Prepare data pointers
  auto device_id = stream.device_index();

  CUDAStream cuda_stream(stream);
  hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());

  if (all_contiguous) {
    if constexpr (num_outputs == 1) {
      // Single output: use vectorized kernel if possible
      using OutType = typename thrust::tuple_element<0, OutTuple>::type;
      OutType* out_ptr = reinterpret_cast<OutType*>(outputs[0]->raw_data_ptr());
      auto in_ptrs_tuple = get_input_ptrs<InTuple>(inputs);
      std::apply([&](auto... args) {
          launch_vectorized_kernel(op, size, stream, out_ptr, args...);
      }, in_ptrs_tuple);
    } else {
      // Multiple outputs: use unrolled kernel
      std::vector<void*> data_ptrs(num_outputs + num_inputs);
      for (int i = 0; i < num_outputs; ++i)
        data_ptrs[i] = outputs[i]->raw_data_ptr();
      for (int i = 0; i < num_inputs; ++i)
        data_ptrs[num_outputs + i] = inputs[i]->raw_data_ptr();
      NDArray ptrs_arr = hetu::cuda::to_ptr_ndarray(data_ptrs, device_id);
      void** output_ptrs = ptrs_arr->data_ptr<void*>();
      void** input_ptrs = output_ptrs + num_outputs;
      int64_t grid = DIVUP(size, BLOCK_WORK_SIZE);
      unroll_kernel_for_multi_outputs<InTuple, OutTuple, num_outputs, num_inputs>
          <<<grid, NUM_THREADS, 0, cuda_stream>>>(input_ptrs, output_ptrs, size, op);
      NDArray::MarkUsedBy({ptrs_arr}, stream);
    }
  } else {
    std::vector<void*> data_ptrs(num_outputs + num_inputs);
    for (int i = 0; i < num_outputs; ++i)
      data_ptrs[i] = outputs[i]->raw_data_ptr();
    for (int i = 0; i < num_inputs; ++i)
      data_ptrs[num_outputs + i] = inputs[i]->raw_data_ptr();
    NDArray ptrs_arr = hetu::cuda::to_ptr_ndarray(data_ptrs, device_id);
    void** output_ptrs = ptrs_arr->data_ptr<void*>();
    void** input_ptrs = output_ptrs + num_outputs;
    // Non-contiguous: use general loop kernel with offset calculators
    constexpr int unroll_factor = num_outputs == 1 ? (sizeof(DataType2Size(outputs[0]->dtype())) >= 4 ? 2 : 4) : 2;
    dim3 block(NUM_THREADS);
    dim3 grid(DIVUP(size, unroll_factor * block.x));

    // Prepare offset calculators
    NDArrayList data = inputs;
    data.insert(data.end(), outputs.begin(), outputs.end());
    auto [offset_arrs, offset_ptrs] = AllocOffsetCalculator(data, stream);
    NDArray offset_ptrs_arr = hetu::cuda::to_ptr_ndarray(offset_ptrs, device_id);
    auto** in_calcs = offset_ptrs_arr->data_ptr<OffsetCalculator*>();
    auto** out_calcs = in_calcs + num_inputs;

    // Launch kernel
    loop_kernel<NUM_THREADS, unroll_factor, num_outputs, num_inputs, InTuple, OutTuple>
        <<<grid, block, 0, cuda_stream>>>(
            input_ptrs, output_ptrs, size, op, in_calcs, out_calcs);

    offset_arrs.push_back(offset_ptrs_arr);
    NDArray::MarkUsedBy(offset_arrs, stream);
    NDArray::MarkUsedBy({ptrs_arr}, stream);
  }
}

/**
 * Vectorized kernel with index
 */

template <typename in_t, typename out_t, typename func_t>
__device__ inline void unroll_kernel_for_multi_outputs_with_idx_impl(
  void** input_ptrs, 
  void** output_ptrs,
  int remaining, 
  int base_idx, 
  func_t op) {
    
  constexpr int num_outputs = thrust::tuple_size<out_t>::value;
  constexpr int num_inputs = std::tuple_size<in_t>::value;

  int thread_idx = threadIdx.x;
  out_t results[THREAD_WORK_SIZE];
  in_t inputs[THREAD_WORK_SIZE];
    
  // Load inputs and compute results
  #pragma unroll
  for (int i = 0; i < THREAD_WORK_SIZE; i++) {
    int index = thread_idx + i * NUM_THREADS;
    if (index >= remaining) break;
    int linear_idx = base_idx + index;
    load_multiple_inputs<in_t, num_inputs>::apply(input_ptrs, inputs, linear_idx, i);
    results[i] = hetu::impl::apply_with_idx(op, linear_idx, inputs[i]);
  }

  // Store outputs
  #pragma unroll
  for (int i = 0; i < THREAD_WORK_SIZE; i++) {
    int index = thread_idx + i * NUM_THREADS;
    if (index >= remaining) break;
    int linear_idx = base_idx + index;
    store_multiple_outputs<out_t, num_outputs>::apply(output_ptrs, results[i], linear_idx);
  }
}

template <typename in_t, typename out_t, int num_outputs, int num_inputs, typename func_t>
__global__ void unroll_kernel_for_multi_outputs_with_idx(
  void** input_ptrs, 
  void** output_ptrs, 
  size_t size, 
  func_t op) {
    
  int base_idx = BLOCK_WORK_SIZE * blockIdx.x;
  int remaining = size - base_idx;
  unroll_kernel_for_multi_outputs_with_idx_impl<in_t, out_t>(
    input_ptrs, output_ptrs, remaining, base_idx, op);
}

template <typename func_t, typename out_t, typename... IN_t>
__device__ void unroll_kernel_with_idx_impl(
  func_t op, 
  int remaining, 
  int base_idx, 
  out_t* output, 
  const IN_t*... inputs) {
    
  int thread_idx = threadIdx.x;
  out_t results[THREAD_WORK_SIZE];
  #pragma unroll
  for (int i = 0; i < THREAD_WORK_SIZE; i++) {
    int index = thread_idx + i * NUM_THREADS;
    if (index >= remaining) break;
    int linear_idx = base_idx + index;
    auto input_tuple = std::make_tuple(inputs[linear_idx]...);
    results[i] = hetu::impl::apply_with_idx(op, linear_idx, input_tuple);
  }
  #pragma unroll
  for (int i = 0; i < THREAD_WORK_SIZE; i++) {
    int index = thread_idx + i * NUM_THREADS;
    if (index >= remaining) break;
    int linear_idx = base_idx + index;
    output[linear_idx] = results[i];
  }
}

template <typename func_t, typename out_t>
__device__ void unroll_kernel_with_idx_impl(
  func_t op, 
  int remaining, 
  int base_idx, 
  out_t* output) {
    
  int thread_idx = threadIdx.x;
  out_t results[THREAD_WORK_SIZE];
  #pragma unroll
  for (int i = 0; i < THREAD_WORK_SIZE; i++) {
    int index = thread_idx + i * NUM_THREADS;
    if (index >= remaining) break;
    int linear_idx = base_idx + index;
    results[i] = op(linear_idx);
  }
  #pragma unroll
  for (int i = 0; i < THREAD_WORK_SIZE; i++) {
    int index = thread_idx + i * NUM_THREADS;
    if (index >= remaining) break;
    int linear_idx = base_idx + index;
    output[linear_idx] = results[i];
  }
}

template <typename func_t, typename out_t, typename... IN_t>
__global__ void unroll_kernel_with_idx(
  func_t op, 
  size_t size, 
  out_t* output, 
  const IN_t*... inputs) {
    
  int base_idx = BLOCK_WORK_SIZE * blockIdx.x;
  int remaining = size - base_idx;
  unroll_kernel_with_idx_impl(op, remaining, base_idx, output, inputs...);
}

template <int vec_size, typename func_t, typename out_t, typename... IN_t>
__device__ void vectorize_kernel_with_idx_impl(func_t op, aligned_vector<out_t, vec_size>* output,
                                               const aligned_vector<IN_t, vec_size>*... inputs) {
  int base_idx = BLOCK_WORK_SIZE * blockIdx.x;
  int loop_size = THREAD_WORK_SIZE / vec_size;
  #pragma unroll
  for (int i = 0; i < loop_size; i++) {
    int index = threadIdx.x + i * NUM_THREADS;
    int linear_idx = base_idx + index * vec_size;
    aligned_vector<out_t, vec_size> ret;
    #pragma unroll
    for (int j = 0; j < vec_size; j++) {
      auto input_tuple = std::make_tuple(inputs[index].val[j]...);
      ret.val[j] = hetu::impl::apply_with_idx(op, linear_idx + j, input_tuple);
    }
    output[index] = ret;
  }
}

template <int vec_size, typename func_t, typename out_t>
__device__ void vectorize_kernel_with_idx_impl(func_t op, aligned_vector<out_t, vec_size>* output) {
  int base_idx = BLOCK_WORK_SIZE * blockIdx.x;
  int loop_size = THREAD_WORK_SIZE / vec_size;
  #pragma unroll
  for (int i = 0; i < loop_size; i++) {
    int index = threadIdx.x + i * NUM_THREADS;
    int linear_idx = base_idx + index * vec_size;
    aligned_vector<out_t, vec_size> ret;
    #pragma unroll
    for (int j = 0; j < vec_size; j++) {
      ret.val[j] = op(linear_idx + j);
    }
    output[index] = ret;
  }
}

template <int vec_size, typename func_t, typename out_t, typename... IN_t>
__global__ void vectorize_kernel_with_idx(func_t op, size_t size, out_t* output, const IN_t*... inputs) {
  int base_idx = BLOCK_WORK_SIZE * blockIdx.x;
  int remaining = size - base_idx;
  if (remaining < BLOCK_WORK_SIZE) {
    // do a naive unrolled loop to handle the reminder
    unroll_kernel_with_idx_impl(op, remaining, base_idx, output, inputs...);
  } else {
    // use vectorize memory load/store to handle a full `block_work_size` data
    vectorize_kernel_with_idx_impl<vec_size>(op, reinterpret_cast<aligned_vector<out_t, vec_size>*>(output + base_idx),
                                             reinterpret_cast<const aligned_vector<IN_t, vec_size>*>(
                                             const_cast<IN_t*>(inputs) + base_idx)...);
  }
}

template <typename func_t, typename out_t, typename... IN_t>
static inline void launch_vectorized_kernel_with_idx(const func_t& op, size_t size, const Stream& stream,
                                                     out_t* output, const IN_t*... inputs) {
  int64_t grid = DIVUP(size, BLOCK_WORK_SIZE);
  CUDAStream cuda_stream(stream);
  hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
  int vec_size = get_vectorize_size(output, inputs...);
  switch (vec_size) {
    case 4:
      vectorize_kernel_with_idx<4><<<grid, NUM_THREADS, 0, cuda_stream>>>(
        op, size, output, inputs...);
      break;
    case 2:
      vectorize_kernel_with_idx<2><<<grid, NUM_THREADS, 0, cuda_stream>>>(
        op, size, output, inputs...);
      break;
    case 1:
      unroll_kernel_with_idx<<<grid, NUM_THREADS, 0, cuda_stream>>>(
        op, size, output, inputs...);
      break;
    default:
      HT_RUNTIME_ERROR << "Unexpected vectorization size";
      __builtin_unreachable();
  }
}

template <int nt, int vt, int num_outputs, int num_inputs,
          typename InTuple, typename OutTuple, typename func_t>
__global__ void loop_kernel_with_idx(
  void** input_ptrs, 
  void** output_ptrs, 
  size_t size, 
  func_t op,
  OffsetCalculator** in_offset_calculators,
  OffsetCalculator** out_offset_calculators) {
    
  int tid = threadIdx.x;
  int nv = nt * vt;
  int idx = nv * blockIdx.x + tid;

  VectorizedStorage<InTuple, vt> inputs;
  VectorizedStorage<OutTuple, vt> results;

  #pragma unroll
  for (int i = 0; i < vt; i++) {
    if (idx < size) {
      // Get output offsets
      int out_offsets[num_outputs];
      #pragma unroll
      for (int j = 0; j < num_outputs; j++) {
        out_offsets[j] = out_offset_calculators[j]->get(idx);
      }

      if constexpr (num_inputs > 0) {
        int in_offsets[num_inputs];
        #pragma unroll
        for (int j = 0; j < num_inputs; j++) {
          in_offsets[j] = in_offset_calculators[j]->get(idx);
        }

        // Load inputs
        load_multiple_inputs<InTuple, num_inputs>::apply(
          input_ptrs, &inputs[i], in_offsets, 0);

        // Apply operation with index
        results[i] = hetu::impl::apply_with_idx(op, idx, inputs[i]);
      } else {
        // Apply operation with index
        results[i] = op(idx);
      }

      // Store outputs
      store_multiple_outputs<OutTuple, num_outputs>::apply(
          output_ptrs, results[i], out_offsets);

      idx += nt;
    }
  }
}

template <typename InTuple, typename OutTuple, typename FuncT>
void launch_loop_kernel_with_idx(
  const NDArrayList& inputs,
  const NDArrayList& outputs,
  size_t size,
  const Stream& stream,
  FuncT op) {
    
  constexpr int num_inputs = std::tuple_size<InTuple>::value;
  constexpr int num_outputs = thrust::tuple_size<OutTuple>::value;
  HT_ASSERT(num_inputs == inputs.size())
    << "Input count mismatch";
  HT_ASSERT(num_outputs == outputs.size())
    << "Output count mismatch";

  // Check if all tensors are contiguous
  bool all_contiguous = true;
  for (const auto& nd : inputs) all_contiguous &= nd->is_contiguous();
  for (const auto& nd : outputs) all_contiguous &= nd->is_contiguous();

  // Prepare data pointers
  auto device_id = stream.device_index();

  CUDAStream cuda_stream(stream);
  hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());

  if (all_contiguous) {
    // Use unrolled kernel for contiguous tensors
    int64_t grid = DIVUP(size, BLOCK_WORK_SIZE);
        
    if constexpr (num_outputs == 1) {
      using OutType = typename thrust::tuple_element<0, OutTuple>::type;
      OutType* out_ptr = reinterpret_cast<OutType*>(outputs[0]->raw_data_ptr());
      auto in_ptrs_tuple = get_input_ptrs<InTuple>(inputs);
      std::apply([&](auto... args) {
        launch_vectorized_kernel_with_idx(op, size, stream, out_ptr, args...);
      }, in_ptrs_tuple);
    } else {
      // Multiple outputs: use specialized multi-output kernel
      std::vector<void*> data_ptrs(num_outputs + num_inputs);
      for (int i = 0; i < num_outputs; ++i)
        data_ptrs[i] = outputs[i]->raw_data_ptr();
      for (int i = 0; i < num_inputs; ++i)
        data_ptrs[num_outputs + i] = inputs[i]->raw_data_ptr();
      NDArray ptrs_arr = hetu::cuda::to_ptr_ndarray(data_ptrs, device_id);
      void** output_ptrs = ptrs_arr->data_ptr<void*>();
      void** input_ptrs = output_ptrs + num_outputs;
      unroll_kernel_for_multi_outputs_with_idx<InTuple, OutTuple, num_outputs, num_inputs>
          <<<grid, NUM_THREADS, 0, cuda_stream>>>(input_ptrs, output_ptrs, size, op);
      NDArray::MarkUsedBy({ptrs_arr}, stream);
    }
  } else {
    // Use loop kernel with offset calculators for non-contiguous tensors
    std::vector<void*> data_ptrs(num_outputs + num_inputs);
    for (int i = 0; i < num_outputs; ++i)
      data_ptrs[i] = outputs[i]->raw_data_ptr();
    for (int i = 0; i < num_inputs; ++i)
      data_ptrs[num_outputs + i] = inputs[i]->raw_data_ptr();
    NDArray ptrs_arr = hetu::cuda::to_ptr_ndarray(data_ptrs, device_id);
    void** output_ptrs = ptrs_arr->data_ptr<void*>();
    void** input_ptrs = output_ptrs + num_outputs;

    constexpr int unroll_factor = num_outputs == 1 ? (sizeof(DataType2Size(outputs[0]->dtype())) >= 4 ? 2 : 4) : 2;
    dim3 block(NUM_THREADS);
    dim3 grid(DIVUP(size, unroll_factor * block.x));

    // Prepare offset calculators
    NDArrayList data = inputs;
    data.insert(data.end(), outputs.begin(), outputs.end());
    auto [offset_arrs, offset_ptrs] = AllocOffsetCalculator(data, stream);
    NDArray offset_ptrs_arr = hetu::cuda::to_ptr_ndarray(offset_ptrs, device_id);
    OffsetCalculator** in_calcs = offset_ptrs_arr->data_ptr<OffsetCalculator*>();
    OffsetCalculator** out_calcs = in_calcs + num_inputs;

    // Launch kernel
    loop_kernel_with_idx<NUM_THREADS, unroll_factor, num_outputs, num_inputs, InTuple, OutTuple>
        <<<grid, block, 0, cuda_stream>>>(
            input_ptrs, output_ptrs, size, op, in_calcs, out_calcs);

    offset_arrs.push_back(offset_ptrs_arr);
    NDArray::MarkUsedBy(offset_arrs, stream);
    NDArray::MarkUsedBy({ptrs_arr}, stream);
  }
}

} // namespace impl
} // namespace hetu