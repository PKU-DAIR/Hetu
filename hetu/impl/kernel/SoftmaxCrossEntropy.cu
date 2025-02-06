#include "hetu/core/ndarray.h"
#include "hetu/core/memory_pool.h"
#include "hetu/impl/stream/CUDAStream.h"
#include "hetu/impl/cuda/CUDADnn.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/cuda_utils.h"
#include "hetu/impl/utils/offset_calculator.cuh"
#include "hetu/impl/kernel/Vectorized.cuh"

namespace hetu {
namespace impl {

void SoftmaxCrossEntropyCuda(const NDArray& input, const NDArray& label,
                             NDArray& output, const Stream& stream) {
  CUDAStream cuda_stream(stream);
  hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
  cudnnHandle_t handle = hetu::impl::GetCudnnHandle(cuda_stream.device_id());
  size_t indim = input->ndim();
  HT_ASSERT(indim == label->ndim() && indim == output->ndim() + 1)
    << "Indim is " << indim << ", Label dim is " << label->ndim()
    << ", Output dim is " << output->ndim();
  int n_ = 1;
  for (int i = 0; i < indim - 1; ++i) {
    n_ *= input->shape(i);
  }
  int c_ = input->shape(indim - 1);
  size_t size = n_ * c_;

  if (size == 0)
    return;

  cudnnDataType_t datatype = to_cudnn_DataType(input->dtype());
  cudnnIndicesType_t indicetype = to_cudnn_IndicidesType(input->dtype());

  dim3 blocks, threads;
  threads.x = MIN(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  blocks.x = DIVUP(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  OffsetCalculator *label_offset_calculator, *temp_offset_calculator;
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input->dtype(), spec_t, "SoftmaxCrossEntropyCuda", [&]() {
      spec_t alpha = 1.0;
      spec_t beta = 0.0;

      float alpha_f = 1.0f;
      float beta_f = 0.0f;

      cudnnTensorDescriptor_t desc;
      CUDNN_CALL(cudnnCreateTensorDescriptor(&desc));
      CUDNN_CALL(cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW, datatype,
                                            n_, c_, 1, 1));
      NDArray contig_input = NDArray::contiguous(input, stream.stream_index());
      NDArray temp_ = NDArray::empty_like(input);

      if (contig_input->dtype() == DataType::FLOAT16 || contig_input->dtype() == DataType::BFLOAT16) {
      CUDNN_CALL(cudnnSoftmaxForward(
          handle, CUDNN_SOFTMAX_LOG, CUDNN_SOFTMAX_MODE_INSTANCE, &alpha_f, desc,
          (const void*) contig_input->data_ptr<spec_t>(), &beta_f, desc, (void*)temp_->data_ptr<spec_t>()));     
      }
      else {
      CUDNN_CALL(cudnnSoftmaxForward(
          handle, CUDNN_SOFTMAX_LOG, CUDNN_SOFTMAX_MODE_INSTANCE, &alpha, desc,
          (const void*) contig_input->data_ptr<spec_t>(), &beta, desc, (void*)temp_->data_ptr<spec_t>()));             
      }   

      using InType = std::tuple<spec_t, spec_t>;
      using OutType = thrust::tuple<spec_t>;
      launch_loop_kernel<InType, OutType>({temp_, label}, {temp_}, size, stream,
                                         [] __device__ (spec_t a, spec_t b) {
                                           return -a * b;
                                          });
      NDArray::sum(temp_, {1}, false, stream.stream_index(), output);
      NDArray::MarkUsedBy({contig_input}, stream);
    });
  NDArray::MarkUsedBy({input, label, output}, stream);
}

template <typename spec_t>
__global__ void softmax_cross_entropy_gradient_kernel(
  const spec_t* pred, const spec_t* y_, const spec_t* grad_data,
  spec_t* output_data, int last_dim, size_t size,
  const OffsetCalculator* y_offset_calculator,
  const OffsetCalculator* grad_offset_calculator,
  const OffsetCalculator* out_offset_calculator) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;
  auto out_offset = out_offset_calculator->get(idx);
  auto y_offset = y_offset_calculator->get(idx);
  auto grad_offset = grad_offset_calculator->get(idx / last_dim);
  output_data[out_offset] = (pred[idx] - y_[y_offset]) * grad_data[grad_offset];
}

void SoftmaxCrossEntropyGradientCuda(const NDArray& input_y,
                                     const NDArray& label, const NDArray& grad,
                                     NDArray& output, const Stream& stream) {
  CUDAStream cuda_stream(stream);
  hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
  cudnnHandle_t handle = hetu::impl::GetCudnnHandle(cuda_stream.device_id());
  size_t indim = input_y->ndim();
  HT_ASSERT(indim == label->ndim() && indim == output->ndim() &&
            indim == grad->ndim() + 1)
    << "Indim is " << indim << ", Label dim is " << label->ndim()
    << ", Output dim is " << output->ndim();
  int n_ = 1;
  for (int i = 0; i < indim - 1; ++i) {
    n_ *= input_y->shape(i);
  }
  int c_ = input_y->shape(indim - 1);
  size_t size = n_ * c_;

  cudnnDataType_t datatype = to_cudnn_DataType(input_y->dtype());

  dim3 blocks, threads;
  threads.x = MIN(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  blocks.x = DIVUP(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);

  int64_t workspace_size;
  if (input_y->dtype() == DataType::FLOAT16 || input_y->dtype() == DataType::BFLOAT16) {
    workspace_size = size * sizeof(float);
  } else {
    workspace_size = size * DataType2Size(input_y->dtype());
  }
  auto workspace_arr =
    NDArray::empty({workspace_size}, grad->device(), kInt8, stream.stream_index());

  OffsetCalculator *label_offset_calculator, *grad_offset_calculator,
                   *out_offset_calculator;
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input_y->dtype(), spec_t, "SoftmaxCrossEntropyCuda", [&]() {
      spec_t alpha = 1.0;
      spec_t beta = 0.0;

      float alpha_f = 1.0f;
      float beta_f = 0.0f;

      cudnnTensorDescriptor_t desc;
      CUDNN_CALL(cudnnCreateTensorDescriptor(&desc));
      CUDNN_CALL(cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW, datatype,
                                            n_, c_, 1, 1));
      NDArray contig_input_y = NDArray::contiguous(input_y, stream.stream_index());
      if (contig_input_y->dtype() == DataType::FLOAT16 || contig_input_y->dtype() == DataType::BFLOAT16) {
      CUDNN_CALL(cudnnSoftmaxForward(
        handle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE, &alpha_f,
        desc, contig_input_y->data_ptr<spec_t>(), &beta_f, desc, 
        workspace_arr->raw_data_ptr()));
      } else {
      CUDNN_CALL(cudnnSoftmaxForward(
        handle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE, &alpha,
        desc, contig_input_y->data_ptr<spec_t>(), &beta, desc, 
        workspace_arr->raw_data_ptr()));        
      }

      NDArray label_offset_calculator_arr, grad_offset_calculator_arr,
               out_offset_calculator_arr;
      std::tie(label_offset_calculator_arr, label_offset_calculator) =
        AllocOffsetCalculator(label, stream);
      std::tie(grad_offset_calculator_arr, grad_offset_calculator) =
        AllocOffsetCalculator(grad, stream);
      std::tie(out_offset_calculator_arr, out_offset_calculator) =
        AllocOffsetCalculator(output, stream);
      softmax_cross_entropy_gradient_kernel<spec_t>
        <<<blocks, threads, 0, cuda_stream>>>(
          (const spec_t*) workspace_arr->raw_data_ptr(), label->data_ptr<spec_t>(),
          grad->data_ptr<spec_t>(), output->data_ptr<spec_t>(), c_, size,
          label_offset_calculator, grad_offset_calculator, out_offset_calculator);

      CUDNN_CALL(cudnnDestroyTensorDescriptor(desc));
      NDArray::MarkUsedBy({contig_input_y, label_offset_calculator_arr,
                           grad_offset_calculator_arr, out_offset_calculator_arr},
                           stream);
    });
  NDArray::MarkUsedBy({input_y, label, grad, workspace_arr}, stream);
}

} // namespace impl
} // namespace hetu
