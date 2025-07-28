#include "hetu/core/ndarray.h"
#include "hetu/core/memory_pool.h"
#include "hetu/impl/stream/CUDAStream.h"
#include "hetu/impl/cuda/CUDADnn.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/cuda_utils.h"
#include <chrono>

namespace hetu {
namespace impl {

void Conv3dCuda(const NDArray& input_x, const NDArray& input_f, NDArray& output,
                const int padding_d, const int padding_h, const int padding_w,
                const int stride_d, const int stride_h, const int stride_w, 
                const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(input_x);
  HT_ASSERT_SAME_DEVICE(input_x, input_f);
  HT_ASSERT_SAME_DEVICE(input_x, output);

  CUDAStream cuda_stream(stream);
  hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
  cudnnHandle_t handle = hetu::impl::GetCudnnHandle(cuda_stream.device_id());

  cudnnDataType_t datatype = to_cudnn_DataType(input_x->dtype());

  int inputDims[5] = {input_x->shape(0), input_x->shape(1), input_x->shape(2), input_x->shape(3), input_x->shape(4)};
  int inputStrides[5] = {input_x->stride(0), input_x->stride(1), input_x->stride(2), input_x->stride(3), input_x->stride(4)};

  int filterDims[5] = {input_f->shape(0), input_f->shape(1), input_f->shape(2), input_f->shape(3), input_f->shape(4)};
  int filterStrides[5] = {input_f->stride(0), input_f->stride(1), input_f->stride(2), input_f->stride(3), input_f->stride(4)}; 

  int outputDims[5] = {output->shape(0), output->shape(1), output->shape(2), output->shape(3), output->shape(4)};
  int outputStrides[5] = {output->stride(0), output->stride(1), output->stride(2), output->stride(3), output->stride(4)};


  // input
  cudnnTensorDescriptor_t input_desc;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&input_desc));
  CUDNN_CALL(cudnnSetTensorNdDescriptor(input_desc, datatype, 5, inputDims, inputStrides));

  // filter
  cudnnFilterDescriptor_t filter_desc;
  CUDNN_CALL(cudnnCreateFilterDescriptor(&filter_desc));
  CUDNN_CALL(cudnnSetFilterNdDescriptor(filter_desc, datatype,
                                        CUDNN_TENSOR_NCHW, 5, filterDims));

  // output
  cudnnTensorDescriptor_t out_desc;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
  CUDNN_CALL(cudnnSetTensorNdDescriptor(out_desc, datatype, 5, outputDims, outputStrides));

  // convolution
  cudnnConvolutionDescriptor_t conv_desc;
  CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
  int conv_padding[3] = {padding_d, padding_h, padding_w};
  int conv_stride[3] = {stride_d, stride_h, stride_w};
  int conv_dilation[3] = {1, 1, 1};
  CUDNN_CALL(cudnnSetConvolutionNdDescriptor(conv_desc, 3, conv_padding, conv_stride, conv_dilation, CUDNN_CROSS_CORRELATION, datatype));

  if (input_x->dtype() == DataType::FLOAT16)
    CUDNN_CALL(cudnnSetConvolutionMathType(conv_desc, CUDNN_TENSOR_OP_MATH));


  // algorithm
  cudnnConvolutionFwdAlgo_t algo;
  size_t workspace_size = 0;
  NDArray workspace;

#if defined(CUDNN_MAJOR) && ((CUDNN_MAJOR >= 8))
  // workaround here
  // TODO: using cudnnFindConvolutionForwardAlgorithm in CuDNN 8 instead
  int return_algo_cnt = CUDNN_CONVOLUTION_FWD_ALGO_COUNT;
  cudnnConvolutionFwdAlgoPerf_t
    perf_results[CUDNN_CONVOLUTION_FWD_ALGO_COUNT];
  CUDNN_CALL(cudnnGetConvolutionForwardAlgorithm_v7(
    handle, input_desc, filter_desc, conv_desc, out_desc,
    CUDNN_CONVOLUTION_FWD_ALGO_COUNT, &return_algo_cnt, perf_results));

  void* tmp_work_data = nullptr;
  bool flag = false;
  for (int i = 0; i < return_algo_cnt; ++i) {
    CUDNN_CALL(cudnnGetConvolutionForwardWorkspaceSize(
      handle, input_desc, filter_desc, conv_desc, out_desc,
      perf_results[i].algo, &workspace_size));
    if (cudaMalloc(&tmp_work_data, workspace_size) == cudaSuccess) {
      algo = perf_results[i].algo;
      CudaFree(tmp_work_data);
      flag = true;
      break;
    }
  }
  HT_RUNTIME_ERROR_IF(!flag) << "Memory insufficient to create workspace";
#else
  CUDNN_CALL(cudnnGetConvolutionForwardAlgorithm(
      handle, input_desc, filter_desc, conv_desc, out_desc, 
      CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &algo));
#endif

  CUDNN_CALL(cudnnGetConvolutionForwardWorkspaceSize(
    handle, input_desc, filter_desc, conv_desc, out_desc, algo,
    &workspace_size));

  if (workspace_size != 0) {
    workspace = NDArray::empty({static_cast<int64_t>(workspace_size)},
                               input_x->device(), kInt8, stream.stream_index());
  }

  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input_x->dtype(), spec_t, "Conv3dCuda", [&]() {
      void* workspace_ptr =
        workspace.is_defined() ? workspace->raw_data_ptr() : nullptr;

      spec_t alpha = 1.0f;
      spec_t beta = 0.0f;

      float alpha_f = 1.0f;
      float beta_f = 0.0f;

      if (input_x->dtype() == DataType::FLOAT16 || input_x->dtype() == DataType::BFLOAT16) {
        CUDNN_CALL(cudnnConvolutionForward(handle, &alpha_f, input_desc, input_x->data_ptr<spec_t>(),
                                           filter_desc, input_f->data_ptr<spec_t>(), conv_desc,
                                           algo, workspace_ptr, workspace_size, &beta_f,
                                           out_desc, output->data_ptr<spec_t>()));
      } else {
        CUDNN_CALL(cudnnConvolutionForward(handle, &alpha, input_desc, input_x->data_ptr<spec_t>(),
                                           filter_desc, input_f->data_ptr<spec_t>(), conv_desc,
                                           algo, workspace_ptr, workspace_size, &beta,
                                           out_desc, output->data_ptr<spec_t>()));
      }
      CUDNN_CALL(cudnnDestroyTensorDescriptor(out_desc));
      CUDNN_CALL(cudnnDestroyConvolutionDescriptor(conv_desc));
      CUDNN_CALL(cudnnDestroyFilterDescriptor(filter_desc));
      CUDNN_CALL(cudnnDestroyTensorDescriptor(input_desc));
    });

  NDArray::MarkUsedBy({input_x, input_f, output, workspace}, stream);
  return;
}

void Conv3dGradientofFilterCuda(const NDArray& input_x,
                                const NDArray& gradient_y, NDArray& gradient_f,
                                const int padding_d, const int padding_h, const int padding_w,
                                const int stride_d, const int stride_h, const int stride_w,
                                const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(input_x);
  HT_ASSERT_SAME_DEVICE(input_x, gradient_y);
  HT_ASSERT_SAME_DEVICE(input_x, gradient_f);

  CUDAStream cuda_stream(stream);
  hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
  cudnnHandle_t handle = hetu::impl::GetCudnnHandle(cuda_stream.device_id());

  cudnnDataType_t datatype = to_cudnn_DataType(input_x->dtype());

  // input


  int input_dims[5] = {input_x->shape(0), input_x->shape(1), input_x->shape(2), input_x->shape(3), input_x->shape(4)};
  int input_strides[5] = {input_x->stride(0), input_x->stride(1), input_x->stride(2), input_x->stride(3), input_x->stride(4)};

  int dy_dims[5] = {gradient_y->shape(0), gradient_y->shape(1), gradient_y->shape(2), gradient_y->shape(3), gradient_y->shape(4)};
  int dy_strides[5] = {gradient_y->stride(0), gradient_y->stride(1), gradient_y->stride(2), gradient_y->stride(3), gradient_y->stride(4)};

  int df_dims[5] = {gradient_f->shape(0), gradient_f->shape(1), gradient_f->shape(2), gradient_f->shape(3), gradient_f->shape(4)};
  int df_strides[5] = {gradient_f->stride(0), gradient_f->stride(1), gradient_f->stride(2), gradient_f->stride(3), gradient_f->stride(4)};

  // input
  cudnnTensorDescriptor_t input_desc;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&input_desc));
  CUDNN_CALL(cudnnSetTensorNdDescriptor(input_desc, datatype, 5, input_dims, input_strides));

  // dy
  cudnnTensorDescriptor_t dy_desc;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&dy_desc));
  CUDNN_CALL(cudnnSetTensorNdDescriptor(dy_desc, datatype, 5, dy_dims, dy_strides));


  // dw
  cudnnFilterDescriptor_t df_desc;
  CUDNN_CALL(cudnnCreateFilterDescriptor(&df_desc));
  CUDNN_CALL(cudnnSetFilterNdDescriptor(df_desc, datatype, CUDNN_TENSOR_NCHW, 5, df_dims));


  // conv3d
  cudnnConvolutionDescriptor_t conv_desc;
  CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
  int conv_padding[3] = {padding_d, padding_h, padding_w};
  int conv_stride[3] = {stride_d, stride_h, stride_w};
  int conv_dilation[3] = {1, 1, 1};
  CUDNN_CALL(cudnnSetConvolutionNdDescriptor(conv_desc, 3, conv_padding, conv_stride, conv_dilation, CUDNN_CROSS_CORRELATION, datatype));

  if (input_x->dtype() == DataType::FLOAT16 || input_x->dtype() == DataType::BFLOAT16)
    CUDNN_CALL(cudnnSetConvolutionMathType(conv_desc, CUDNN_TENSOR_OP_MATH));

  // algo
  cudnnConvolutionBwdFilterAlgo_t algo;
  size_t workspace_size = 0;
  NDArray workspace;

#if defined(CUDNN_MAJOR) && ((CUDNN_MAJOR >= 8))
  // TODO: using cudnnFindConvolutionBackwardFilterAlgorithm in CuDNN 8
  // instead algo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT;
  int return_algo_cnt = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT;
  cudnnConvolutionBwdFilterAlgoPerf_t
    perf_results[CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT];
  CUDNN_CALL(cudnnGetConvolutionBackwardFilterAlgorithm_v7(
    handle, input_desc, dy_desc, conv_desc, df_desc,
    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT, &return_algo_cnt,
    perf_results));

  void* tmp_work_data = nullptr;
  bool flag = false;
  for (int i = 0; i < return_algo_cnt; ++i) {
    CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(
      handle, input_desc, dy_desc, conv_desc, df_desc, perf_results[i].algo,
      &workspace_size));
    if (cudaMalloc(&tmp_work_data, workspace_size) == cudaSuccess) {
      algo = perf_results[i].algo;
      CudaFree(tmp_work_data);
      flag = true;
      break;
    }
  }
  HT_RUNTIME_ERROR_IF(!flag) << "Memory insufficient to create workspace";
#else
  CUDNN_CALL(cudnnGetConvolutionBackwardFilterAlgorithm(
      handle, input_desc, dy_desc, conv_desc, df_desc,
      CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, 0, &algo));
#endif
  CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(
    handle, input_desc, dy_desc, conv_desc, df_desc, algo,
    &workspace_size));

  if (workspace_size != 0) {
    workspace = NDArray::empty({static_cast<int64_t>(workspace_size)},
                               input_x->device(), kInt8, stream.stream_index());
  }

  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input_x->dtype(), spec_t, "Conv2dGradientofFilterCuda", [&]() {
      void* workspace_ptr =
        workspace.is_defined() ? workspace->raw_data_ptr() : nullptr;

      spec_t alpha = 1.0;
      spec_t beta = 0.0;

      float alpha_f = 1.0f;
      float beta_f = 0.0f;

      if (input_x->dtype() == DataType::FLOAT16 || input_x->dtype() == DataType::BFLOAT16) {
        CUDNN_CALL(cudnnConvolutionBackwardFilter(
          handle, &alpha_f, input_desc, input_x->data_ptr<spec_t>(), dy_desc, gradient_y->data_ptr<spec_t>(), 
          conv_desc, algo, workspace_ptr, workspace_size, &beta_f, df_desc, gradient_f->data_ptr<spec_t>()));
      } else {
        CUDNN_CALL(cudnnConvolutionBackwardFilter(
          handle, &alpha, input_desc, input_x->data_ptr<spec_t>(), dy_desc, gradient_y->data_ptr<spec_t>(), 
          conv_desc, algo, workspace_ptr, workspace_size, &beta, df_desc, gradient_f->data_ptr<spec_t>()));
      }
      CUDNN_CALL(cudnnDestroyTensorDescriptor(dy_desc));
      CUDNN_CALL(cudnnDestroyConvolutionDescriptor(conv_desc));
      CUDNN_CALL(cudnnDestroyFilterDescriptor(df_desc));
      CUDNN_CALL(cudnnDestroyTensorDescriptor(input_desc));
    });
  NDArray::MarkUsedBy({input_x, gradient_y, gradient_f, workspace}, stream);
}


void Conv3dGradientofDataCuda(const NDArray& input_f, const NDArray& gradient_y,
                              NDArray& gradient_x, const int padding_d, const int padding_h,
                              const int padding_w, const int stride_d, const int stride_h,
                              const int stride_w, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(input_f);
  HT_ASSERT_SAME_DEVICE(input_f, gradient_y);
  HT_ASSERT_SAME_DEVICE(input_f, gradient_x);

  CUDAStream cuda_stream(stream);
  hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
  cudnnHandle_t handle = hetu::impl::GetCudnnHandle(cuda_stream.device_id());

  cudnnDataType_t datatype = to_cudnn_DataType(input_f->dtype());

  // filter
  int filter_dim[5] = {input_f->shape(0), input_f->shape(1), input_f->shape(2), input_f->shape(3), input_f->shape(4)};
  int filter_strides[5] = {input_f->stride(0), input_f->stride(1), input_f->stride(2), input_f->stride(3), input_f->stride(4)};

  // dy
  int dy_dims[5] = {gradient_y->shape(0), gradient_y->shape(1), gradient_y->shape(2), gradient_y->shape(3), gradient_y->shape(4)};
  int dy_strides[5] = {gradient_y->stride(0), gradient_y->stride(1), gradient_y->stride(2), gradient_y->stride(3), gradient_y->stride(4)};
  
  // dx
  int dx_dims[5] = {gradient_x->shape(0), gradient_x->shape(1), gradient_x->shape(2), gradient_x->shape(3), gradient_x->shape(4)};
  int dx_strides[5] = {gradient_x->stride(0), gradient_x->stride(1), gradient_x->stride(2), gradient_x->stride(3), gradient_x->stride(4)};

  // filter
  cudnnFilterDescriptor_t filter_desc;
  CUDNN_CALL(cudnnCreateFilterDescriptor(&filter_desc));
  CUDNN_CALL(cudnnSetFilterNdDescriptor(filter_desc, datatype, CUDNN_TENSOR_NCHW, 5, filter_dim));

  // dy
  cudnnTensorDescriptor_t dy_desc;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&dy_desc));
  CUDNN_CALL(cudnnSetTensorNdDescriptor(dy_desc, datatype, 5, dy_dims, dy_strides));


  // dx
  cudnnTensorDescriptor_t dx_desc;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&dx_desc));
  CUDNN_CALL(cudnnSetTensorNdDescriptor(dx_desc, datatype, 5, dx_dims, dx_strides));


  // conv3d
  cudnnConvolutionDescriptor_t conv_desc;
  CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
  int conv_padding[3] = {padding_d, padding_h, padding_w};
  int conv_stride[3] = {stride_d, stride_h, stride_w};
  int conv_dilation[3] = {1, 1, 1};
  CUDNN_CALL(cudnnSetConvolutionNdDescriptor(conv_desc, 3, conv_padding, conv_stride, conv_dilation, CUDNN_CROSS_CORRELATION, datatype));

  if (input_f->dtype() == DataType::FLOAT16 || input_f->dtype() == DataType::BFLOAT16)
    CUDNN_CALL(cudnnSetConvolutionMathType(conv_desc, CUDNN_TENSOR_OP_MATH));

  // algo
  cudnnConvolutionBwdDataAlgo_t algo;
  size_t workspace_size = 0;
  NDArray workspace;

#if defined(CUDNN_MAJOR) && ((CUDNN_MAJOR >= 8))
  // TODO: using cudnnFindConvolutionBackwardDataAlgorithm in CuDNN 8
  // instead
  int return_algo_cnt = CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT;
  cudnnConvolutionBwdDataAlgoPerf_t
    perf_results[CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT];
  CUDNN_CALL(cudnnGetConvolutionBackwardDataAlgorithm_v7(
    handle, filter_desc, dy_desc, conv_desc, dx_desc,
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT, &return_algo_cnt, perf_results));

  void* tmp_work_data = nullptr;
  bool flag = false;
  for (int i = 0; i < return_algo_cnt; ++i) {
    CUDNN_CALL(cudnnGetConvolutionBackwardDataWorkspaceSize(
      handle, filter_desc, dy_desc, conv_desc, dx_desc,
      perf_results[i].algo, &workspace_size));
    if (cudaMalloc(&tmp_work_data, workspace_size) == cudaSuccess) {
      algo = perf_results[i].algo;
      CudaFree(tmp_work_data);
      flag = true;
      break;
    }
  }
  HT_RUNTIME_ERROR_IF(!flag) << "Memory insufficient to create workspace";
#else
  CUDNN_CALL(cudnnGetConvolutionBackwardDataAlgorithm(
      handle, filter_desc, dy_desc, conv_desc, dx_desc,
      CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST, 0, &algo));
#endif
  CUDNN_CALL(cudnnGetConvolutionBackwardDataWorkspaceSize(
    handle, filter_desc, dy_desc, conv_desc, dx_desc, algo,
    &workspace_size));

  if (workspace_size != 0) {
    workspace = NDArray::empty({static_cast<int64_t>(workspace_size)},
                               input_f->device(), kInt8, stream.stream_index());
  }

  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input_f->dtype(), spec_t, "Conv2dGradientofDataCuda", [&]() {
      void* workspace_ptr =
        workspace.is_defined() ? workspace->raw_data_ptr() : nullptr;

      spec_t alpha = 1.0;
      spec_t beta = 0.0;

      float alpha_f = 1.0f;
      float beta_f = 0.0f;

      if (input_f->dtype() == DataType::FLOAT16 || input_f->dtype() == DataType::BFLOAT16) {
        CUDNN_CALL(cudnnConvolutionBackwardData(
          handle, &alpha_f, filter_desc, input_f->data_ptr<spec_t>(), dy_desc, gradient_y->data_ptr<spec_t>(), 
          conv_desc, algo, workspace_ptr, workspace_size, &beta_f, dx_desc, gradient_x->data_ptr<spec_t>()));
      } else {
        CUDNN_CALL(cudnnConvolutionBackwardData(
          handle, &alpha, filter_desc, input_f->data_ptr<spec_t>(), dy_desc, gradient_y->data_ptr<spec_t>(), 
          conv_desc, algo, workspace_ptr, workspace_size, &beta, dx_desc, gradient_x->data_ptr<spec_t>()));        
      }
      CUDNN_CALL(cudnnDestroyTensorDescriptor(dy_desc));
      CUDNN_CALL(cudnnDestroyConvolutionDescriptor(conv_desc));
      CUDNN_CALL(cudnnDestroyTensorDescriptor(dx_desc));
      CUDNN_CALL(cudnnDestroyFilterDescriptor(filter_desc));
    });
  NDArray::MarkUsedBy({input_f, gradient_y, gradient_x, workspace}, stream);
}

} // namespace impl
} // namespace hetu
