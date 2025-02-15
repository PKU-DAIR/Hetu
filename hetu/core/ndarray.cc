#include "hetu/core/ndarray.h"
#include "hetu/core/stream.h"
#include "hetu/core/memory_pool.h"
#include "hetu/graph/ops/kernel_links.h"
#include "hetu/impl/utils/cuda_utils.h"
#include "hetu/impl/utils/dispatch.h"
#include "hetu/impl/utils/ndarray_utils.h"
#include "hetu/impl/stream/CUDAStream.h"
#include <numeric>
#include <iterator>

namespace hetu {

// A silly serialization method for debugging.
void NDArrayDef::Serialize(std::ostream& os, size_t n_print) const {
  os << "NDArray([";
  size_t size = numel();
  n_print = MIN(n_print, size);
  if (n_print > 0 && dtype() != kFloat4 && dtype() != kNFloat4) {
    wait(); // ensure all async kernels on this array have completed
    HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
      dtype(), spec_t, __FUNCTION__, [&]() {
        if (is_cpu()) {
          const spec_t* ptr = data_ptr<spec_t>();
          os << ptr[0];
          for (size_t i = 1; i < n_print; i++)
            os << ", " << ptr[i];
        } else {
          hetu::cuda::CUDADeviceGuard guard(device().index());
          const spec_t* dev_ptr = data_ptr<spec_t>();
          std::vector<spec_t> host_vec(n_print);
          CudaMemcpy(host_vec.data(), dev_ptr, n_print * sizeof(spec_t),
                     cudaMemcpyDeviceToHost);
          os << host_vec[0];
          for (size_t i = 1; i < n_print; i++)
            os << ", " << host_vec[i];
        }
      });
  }
  else if (n_print > 0){
    if (is_cpu()) {
      const uint8_t* ptr = data_ptr<uint8_t>();
      os << int(ptr[0] >> 4) << ", " << int(ptr[0] & 0xF);
      for (size_t i = 1; i < n_print / 2; i++)
        os << ", " << int(ptr[i] >> 4) << ", " << int(ptr[i] & 0xF);
    } else {
      hetu::cuda::CUDADeviceGuard guard(device().index());
      const uint8_t* dev_ptr = data_ptr<uint8_t>();
      std::vector<uint8_t> host_vec(n_print);
      CudaMemcpy(host_vec.data(), dev_ptr, (n_print / 2) * sizeof(uint8_t),
                  cudaMemcpyDeviceToHost);
      os << int(host_vec[0] >> 4) << ", " << int(host_vec[0] & 0xF);
      for (size_t i = 1; i < n_print / 2; i++)
        os << ", " << int(host_vec[i] >> 4) << ", " << int(host_vec[i] & 0xF);
    }
  }
  os << "], dtype=" << dtype() << ", shape=" << shape() << ", stride=" << stride()
     << ", dynamic_shape=" << dynamic_shape() << ", device=" << device() << ")";
}

std::ostream& operator<<(std::ostream& os, const NDArray& data) {
  if (!data.is_defined())
    os << "NDArray()";
  else 
    data->Serialize(os);
  return os;
}

NDArray NDArray::EMPTY;
const StreamIndex NDArray::DEFAULT_STREAM = kComputingStream;

std::future<void> NDArrayDef::wait_async() const {
  return GetMemoryPool(device())->WaitDataSpace(storage()->data_ptr(), true);
}

void NDArrayDef::wait() const {
  GetMemoryPool(device())->WaitDataSpace(storage()->data_ptr(), false);
}

void NDArray::MarkUsedBy(const NDArray& array, const Stream& stream) {
  if (stream.is_blocking() || !array.is_defined())
    return;
  GetMemoryPool(array->device())
    ->MarkDataSpaceUsedByStream(array->storage()->data_ptr(), stream);
}

void NDArray::MarkUsedBy(const NDArrayList& arrays, const Stream& stream) {
  if (stream.is_blocking() || arrays.empty())
    return;

  if (arrays.size() == 1) {
    NDArray::MarkUsedBy(arrays.front(), stream);
    return;
  }

  if (arrays.size() == 2) {
    // Minor optimization (by skipping boring checks) for H2D or D2H ops
    if (!arrays[0].is_defined() || !arrays[1].is_defined() ||
        arrays[0]->device() != arrays[1]->device()) {
      NDArray::MarkUsedBy(arrays[0], stream);
      NDArray::MarkUsedBy(arrays[1], stream);
    } else {
      DataPtrList data_ptrs{arrays[0]->storage()->data_ptr(),
                            arrays[1]->storage()->data_ptr()};
      GetMemoryPool(arrays[0]->device())
        ->MarkDataSpacesUsedByStream(data_ptrs, stream);
    }
    return;
  }

  Device device;
  bool first_defined_array = true;
  bool same_device = true;
  for (const auto& array : arrays) {
    if (!array.is_defined())
      continue;
    if (first_defined_array) {
      device = array->device();
      first_defined_array = false;
      continue;
    }
    if (array->device() != device) {
      same_device = false;
      break;
    }
  }

  if (device.is_undetermined()) {
    // All arrays are undefined
    return;
  } else if (same_device) {
    // All arrays are on the same device
    DataPtrList data_ptrs;
    data_ptrs.reserve(arrays.size());
    for (const auto& array : arrays) {
      if (!array.is_defined())
        continue;
      data_ptrs.emplace_back(array->storage()->data_ptr());
    }
    GetMemoryPool(device)->MarkDataSpacesUsedByStream(data_ptrs, stream);
  } else {
    // Codes below are unoptimized. But there should not be too many 
    // devices involved.
    std::unordered_set<Device> involved_devices;
    for (const auto& array : arrays) {
      if (!array.is_defined())
        continue;
      involved_devices.insert(array->device());
    }
    DataPtrList data_ptrs;
    data_ptrs.reserve(arrays.size());
    for (auto& device : involved_devices) {
      data_ptrs.clear();
      for (const auto& array : arrays) {
        if (!array.is_defined() || array->device() != device)
          continue;
        data_ptrs.emplace_back(array->storage()->data_ptr());
      }
      GetMemoryPool(device)->MarkDataSpacesUsedByStream(data_ptrs, stream);
    }
  }
}

// deprecated: dynamic shape at inference when using different seq_len
NDArray NDArray::to(const NDArray& input, const Device& device, DataType dtype,
                    StreamIndex stream_id, NDArray& output) {
  bool same_device = device.is_undetermined() || device == input->device();
  bool same_dtype = dtype == kUndeterminedDataType || dtype == input->dtype();
  if (same_device && same_dtype) {
    return NDArray(input->meta(), input->storage(), input->storage_offset());
  } else {
    const auto& target_device = same_device ? input->device() : device;
    const auto& target_dtype = same_dtype ? input->dtype() : dtype;
    NDArray out = output.is_defined()
      ? output
      : NDArray::empty(input->shape(), target_device, target_dtype, stream_id,
                       input->dynamic_shape());
    if (output.is_defined()) {
      // Unlike many other kernels, the DataTransfer kernel cannot check
      // whether the devices and dtypes are valid. Hence we check them here.
      HT_ASSERT(output->device() == target_device)
      << output->device() << "," << target_device; 
      HT_ASSERT(output->dtype() == target_dtype);
    }
    Stream stream(input->is_cuda() ? input->device() : target_device,
                  stream_id);
    HT_DISPATCH_KERNEL_CPU_AND_CUDA(stream.device_type(), __FUNCTION__,
                                    hetu::impl::DataTransfer, input, out,
                                    stream);
    return out;
  }
}

NDArray NDArray::abs(const NDArray& input, StreamIndex stream_id,
                     NDArray& output) {
  Stream stream(input->device(), stream_id);
  NDArray out = output.is_defined() ? output : NDArray::empty_like(input);
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(input->device().type(), __FUNCTION__,
                                  hetu::impl::Abs, input, out, stream);
  return out;
}

NDArray NDArray::add(const NDArray& x, const NDArray& y, StreamIndex stream_id,
                     NDArray& output) {
  auto output_shape = NDArrayMeta::Broadcast(x->shape(), y->shape());
  HT_ASSERT(!output_shape.empty())
    << "Shapes cannot be broadcast together: " << x->shape() << " vs. "
    << y->shape();
  NDArray out = output.is_defined()
    ? output
    : NDArray::empty(output_shape, x->device(), x->dtype(), stream_id);
  Stream stream(x->device(), stream_id);
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(x->device().type(), __FUNCTION__,
                                  hetu::impl::AddElewise, x, y, out, stream);
  return out;
}

NDArray NDArray::add(const NDArray& input, double scalar, StreamIndex stream_id,
                     NDArray& output) {
  NDArray out = output.is_defined() ? output : NDArray::empty_like(input);
  Stream stream(input->device(), stream_id);
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(input->device().type(), __FUNCTION__,
                                  hetu::impl::AddConst, input, scalar, out,
                                  stream);
  return out;
}

NDArray NDArray::sub(const NDArray& x, const NDArray& y, StreamIndex stream_id,
                     NDArray& output) {
  auto output_shape = NDArrayMeta::Broadcast(x->shape(), y->shape());
  HT_ASSERT(!output_shape.empty())
    << "Shapes cannot be broadcast together: " << x->shape() << " vs. "
    << y->shape();
  NDArray out = output.is_defined()
    ? output
    : NDArray::empty(output_shape, x->device(), x->dtype(), stream_id);
  Stream stream(x->device(), stream_id);
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(x->device().type(), __FUNCTION__,
                                  hetu::impl::SubElewise, x, y, out, stream);
  return out;
}

NDArray NDArray::sub(const NDArray& input, double scalar, StreamIndex stream_id,
                     NDArray& output) {
  return NDArray::add(input, -scalar, stream_id, output);
}

NDArray NDArray::sub(double scalar, const NDArray& input, StreamIndex stream_id,
                     NDArray& output) {
  NDArray out = output.is_defined() ? output : NDArray::empty_like(input);
  Stream stream(input->device(), stream_id);
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(input->device().type(), __FUNCTION__,
                                  hetu::impl::SubConst, input, scalar, out,
                                  stream);
  return out;
}

NDArray NDArray::neg(const NDArray& input, StreamIndex stream_id,
                     NDArray& output) {
  NDArray out = output.is_defined() ? output : NDArray::empty_like(input);
  Stream stream(input->device(), stream_id);
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(input->device().type(), __FUNCTION__,
                                  hetu::impl::Opposite, input, out, stream);
  return out;
}

NDArray NDArray::mul(const NDArray& x, const NDArray& y, StreamIndex stream_id,
                     NDArray& output) {
  auto output_shape = NDArrayMeta::Broadcast(x->shape(), y->shape());
  HT_ASSERT(!output_shape.empty())
    << "Shapes cannot be broadcast together: " << x->shape() << " vs. "
    << y->shape();
  NDArray out = output.is_defined()
    ? output
    : NDArray::empty(output_shape, x->device(), x->dtype(), stream_id);
  Stream stream(x->device(), stream_id);
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(x->device().type(), __FUNCTION__,
                                  hetu::impl::MulElewise, x, y, out, stream);
  return out;
}

NDArray NDArray::mul(const NDArray& input, double scalar, StreamIndex stream_id,
                     NDArray& output) {
  NDArray out = output.is_defined() ? output : NDArray::empty_like(input);
  Stream stream(input->device(), stream_id);
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(input->device().type(), __FUNCTION__,
                                  hetu::impl::MulConst, input, scalar, out,
                                  stream);
  return out;
}

NDArray NDArray::div(const NDArray& x, const NDArray& y, StreamIndex stream_id,
                     NDArray& output) {
  auto output_shape = NDArrayMeta::Broadcast(x->shape(), y->shape());
  HT_ASSERT(!output_shape.empty())
    << "Shapes cannot be broadcast together: " << x->shape() << " vs. "
    << y->shape();
  NDArray out = output.is_defined()
    ? output
    : NDArray::empty(output_shape, x->device(), x->dtype(), stream_id);
  Stream stream(x->device(), stream_id);
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(x->device().type(), __FUNCTION__,
                                  hetu::impl::DivElewise, x, y, out, stream);
  return out;
}

NDArray NDArray::div(const NDArray& input, double scalar, StreamIndex stream_id,
                     NDArray& output) {
  return NDArray::mul(input, 1.0 / scalar, stream_id, output);
}

NDArray NDArray::div(double scalar, const NDArray& input, StreamIndex stream_id,
                     NDArray& output) {
  NDArray out = output.is_defined() ? output : NDArray::empty_like(input);
  Stream stream(input->device(), stream_id);
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(input->device().type(), __FUNCTION__,
                                  hetu::impl::DivConst, input, scalar, out,
                                  stream);
  return out;
}

NDArray NDArray::pow(const NDArray& input, double exponent,
                     StreamIndex stream_id, NDArray& output) {
  NDArray out = output.is_defined() ? output : NDArray::empty_like(input);
  Stream stream(input->device(), stream_id);
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(input->device().type(), __FUNCTION__,
                                  hetu::impl::Pow, input, exponent, out,
                                  stream);
  return out;
}

NDArray NDArray::sqrt(const NDArray& input, StreamIndex stream_id,
                      NDArray& output) {
  NDArray out = output.is_defined() ? output : NDArray::empty_like(input);
  Stream stream(input->device(), stream_id);
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(input->device().type(), __FUNCTION__,
                                  hetu::impl::Sqrt, input, out, stream);
  return out;
}

NDArray NDArray::reciprocal(const NDArray& input, StreamIndex stream_id,
                            NDArray& output) {
  NDArray out = output.is_defined() ? output : NDArray::empty_like(input);
  Stream stream(input->device(), stream_id);
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(input->device().type(), __FUNCTION__,
                                  hetu::impl::Reciprocal, input, out, stream);
  return out;
}

NDArray NDArray::sigmoid(const NDArray& input, StreamIndex stream_id,
                         NDArray& output) {
  NDArray out = output.is_defined() ? output : NDArray::empty_like(input);
  Stream stream(input->device(), stream_id);
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(input->device().type(), __FUNCTION__,
                                  hetu::impl::Sigmoid, input, out, stream);
  return out;
}

NDArray NDArray::relu(const NDArray& input, StreamIndex stream_id,
                      NDArray& output) {
  NDArray out = output.is_defined() ? output : NDArray::empty_like(input);
  Stream stream(input->device(), stream_id);
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(input->device().type(), __FUNCTION__,
                                  hetu::impl::Relu, input, out, stream);
  return out;
}

NDArray NDArray::gelu(const NDArray& input, StreamIndex stream_id,
                      NDArray& output) {
  NDArray out = output.is_defined() ? output : NDArray::empty_like(input);
  Stream stream(input->device(), stream_id);
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(input->device().type(), __FUNCTION__,
                                  hetu::impl::Gelu, input, out, stream);
  return out;
}

NDArray NDArray::swiglu(const NDArray& input, StreamIndex stream_id,
                        NDArray& output){
  Stream stream(input->device(), stream_id);
 
  if(output.is_defined()){
    HT_DISPATCH_KERNEL_CPU_AND_CUDA(input->device().type(), __FUNCTION__,
                                  hetu::impl::Swiglu, input, output, stream);
    return output; 
  } else {
    HTShape shape = input->shape();
    shape.back() /= 2;
    NDArray out = NDArray::empty(shape, input->device(), 
                      input->dtype(), stream_id, input->dynamic_shape());
    HT_DISPATCH_KERNEL_CPU_AND_CUDA(input->device().type(), __FUNCTION__,
                                  hetu::impl::Swiglu, input, out, stream);
    return out;
  }
}

NDArray NDArray::tanh(const NDArray& input, StreamIndex stream_id,
                      NDArray& output) {
  NDArray out = output.is_defined() ? output : NDArray::empty_like(input);
  Stream stream(input->device(), stream_id);
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(input->device().type(), __FUNCTION__,
                                  hetu::impl::Tanh, input, out, stream);
  return out;
}

NDArray NDArray::elu(const NDArray& input, double alpha, double scale,
                     StreamIndex stream_id,
                     NDArray& output) {
  NDArray out = output.is_defined() ? output : NDArray::empty_like(input);
  Stream stream(input->device(), stream_id);
  HT_DISPATCH_KERNEL_CUDA_ONLY(input->device().type(), __FUNCTION__,
                               hetu::impl::Elu, input, alpha, scale, out, stream);
  return out;
}

NDArray NDArray::hardshrink(const NDArray& input, double lambda,
                            StreamIndex stream_id,
                            NDArray& output) {
  NDArray out = output.is_defined() ? output : NDArray::empty_like(input);
  Stream stream(input->device(), stream_id);
  HT_DISPATCH_KERNEL_CUDA_ONLY(input->device().type(), __FUNCTION__,
                               hetu::impl::Hardshrink, input, lambda, out, stream);
  return out;
}

NDArray NDArray::hardsigmoid(const NDArray& input,
                             StreamIndex stream_id,
                             NDArray& output) {
  NDArray out = output.is_defined() ? output : NDArray::empty_like(input);
  Stream stream(input->device(), stream_id);
  HT_DISPATCH_KERNEL_CUDA_ONLY(input->device().type(), __FUNCTION__,
                               hetu::impl::Hardsigmoid, input, out, stream);
  return out;
}

NDArray NDArray::hardtanh(const NDArray& input, double min_val, double max_val,
                          StreamIndex stream_id,
                          NDArray& output) {
  NDArray out = output.is_defined() ? output : NDArray::empty_like(input);
  Stream stream(input->device(), stream_id);
  HT_DISPATCH_KERNEL_CUDA_ONLY(input->device().type(), __FUNCTION__,
                               hetu::impl::Hardtanh, input, min_val, max_val, out, stream);
  return out;
}

NDArray NDArray::hardswish(const NDArray& input,
                           StreamIndex stream_id,
                           NDArray& output) {
  NDArray out = output.is_defined() ? output : NDArray::empty_like(input);
  Stream stream(input->device(), stream_id);
  HT_DISPATCH_KERNEL_CUDA_ONLY(input->device().type(), __FUNCTION__,
                               hetu::impl::Hardswish, input, out, stream);
  return out;
}

// Used for softmax_lse correctness in parallel-attn op only
NDArray NDArray::logsigmoid(const NDArray& input, bool inverse,
                            StreamIndex stream_id,
                            NDArray& output) {
  NDArray out = output.is_defined() ? output : NDArray::empty_like(input);
  Stream stream(input->device(), stream_id);
  HT_DISPATCH_KERNEL_CUDA_ONLY(input->device().type(), __FUNCTION__,
                               hetu::impl::Logsigmoid, input, out, inverse, stream);
  return out;
}

NDArray NDArray::silu(const NDArray& input,
                      StreamIndex stream_id,
                      NDArray& output) {
  NDArray out = output.is_defined() ? output : NDArray::empty_like(input);
  Stream stream(input->device(), stream_id);
  HT_DISPATCH_KERNEL_CUDA_ONLY(input->device().type(), __FUNCTION__,
                               hetu::impl::Silu, input, out, stream);
  return out;
}

NDArray NDArray::mish(const NDArray& input,
                      StreamIndex stream_id,
                      NDArray& output) {
  NDArray out = output.is_defined() ? output : NDArray::empty_like(input);
  Stream stream(input->device(), stream_id);
  HT_DISPATCH_KERNEL_CUDA_ONLY(input->device().type(), __FUNCTION__,
                               hetu::impl::Mish, input, out, stream);
  return out;
}

NDArray NDArray::softplus(const NDArray& input, double beta, double threshold,
                          StreamIndex stream_id,
                          NDArray& output) {
  NDArray out = output.is_defined() ? output : NDArray::empty_like(input);
  Stream stream(input->device(), stream_id);
  HT_DISPATCH_KERNEL_CUDA_ONLY(input->device().type(), __FUNCTION__,
                               hetu::impl::Softplus, input, beta, threshold, out, stream);
  return out;
}

NDArray NDArray::softshrink(const NDArray& input, double lambda,
                          StreamIndex stream_id,
                          NDArray& output) {
  NDArray out = output.is_defined() ? output : NDArray::empty_like(input);
  Stream stream(input->device(), stream_id);
  HT_DISPATCH_KERNEL_CUDA_ONLY(input->device().type(), __FUNCTION__,
                               hetu::impl::Softshrink, input, lambda, out, stream);
  return out;
}

NDArray NDArray::exp(const NDArray& input, StreamIndex stream_id,
                     NDArray& output) {
  NDArray out = output.is_defined() ? output : NDArray::empty_like(input);
  Stream stream(input->device(), stream_id);
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(input->device().type(), __FUNCTION__,
                                  hetu::impl::Exp, input, out, stream);
  return out;
}

NDArray NDArray::log(const NDArray& input, StreamIndex stream_id,
                     NDArray& output) {
  NDArray out = output.is_defined() ? output : NDArray::empty_like(input);
  Stream stream(input->device(), stream_id);
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(input->device().type(), __FUNCTION__,
                                  hetu::impl::Log, input, out, stream);
  return out;
}

NDArray NDArray::ceil(const NDArray& input, StreamIndex stream_id,
                      NDArray& output) {
  NDArray out = output.is_defined() ? output : NDArray::empty_like(input);
  Stream stream(input->device(), stream_id);
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(input->device().type(), __FUNCTION__,
                                  hetu::impl::Ceil, input, out, stream);
  return out;
}

NDArray NDArray::floor(const NDArray& input, StreamIndex stream_id,
                       NDArray& output) {
  NDArray out = output.is_defined() ? output : NDArray::empty_like(input);
  Stream stream(input->device(), stream_id);
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(input->device().type(), __FUNCTION__,
                                  hetu::impl::Floor, input, out, stream);
  return out;
}

NDArray NDArray::round(const NDArray& input, StreamIndex stream_id,
                       NDArray& output) {
  NDArray out = output.is_defined() ? output : NDArray::empty_like(input);
  Stream stream(input->device(), stream_id);
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(input->device().type(), __FUNCTION__,
                                  hetu::impl::Round, input, out, stream);
  return out;
}

NDArray NDArray::reduce(const NDArray& input, ReductionType red_type,
                        const HTAxes& axes, bool keepdims,
                        StreamIndex stream_id, NDArray& output) {
  auto output_shape = NDArrayMeta::Reduce(input->shape(), axes, keepdims);
  NDArray out = output.is_defined()
    ? output
    : NDArray::empty(output_shape, input->device(), input->dtype(), stream_id);
  Stream stream(input->device(), stream_id);
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(input->device().type(), __FUNCTION__,
                                  hetu::impl::Reduce, input, out, axes, red_type,
                                  stream);
  return out;
}

/*
  Matrix product of two tensors.
  The behavior depends on the dimensionality of input tensors:
  - If both tensors are 1-dimensional (1D), return the dot product (scalar).
  - If both tensors are 2D, return matrix-matrix product.
  - If tensors are 1D - 2D, a 1 is prepended to first tensor's dimension
    for the purpose of the matrix multiply.
    After the matrix multiply, the prepended dimension is removed.
  - If tensors are 2D - 1D, return matrix-vector product.
  - If one of the tensors is ND (N >= 3) and the other is 1D or 2D, fold the 
    first N-1 dimensions of the ND tensor to form a matrix and go back to
    the previous two cases, reshape it and return.
  - Otherwise, broadcast and fold the batched dimensions if
    there's more than one, return bmm.
*/
NDArray NDArray::matmul(const NDArray& x, const NDArray& y, bool trans_left,
                        bool trans_right, StreamIndex stream_id,
                        NDArray& output) {
  auto dim_x = x->ndim();
  auto dim_y = y->ndim();

  HT_ASSERT(dim_x != 0 && dim_y != 0)
    << "Invalid ndims for matrix multiplication: "
    << "Both arguments to matmul need to be at least 1D, but they are"
    << dim_x << "D and " << dim_y << "D. ";
  
  NDArray out;
  Stream stream(x->device(), stream_id);

  if (dim_x == 1 && dim_y == 1) {
    out = output.is_defined()
      ? output
      : NDArray::empty({1}, x->device(), x->dtype(), stream_id);
    HT_DISPATCH_KERNEL_CPU_AND_CUDA(x->device().type(), __FUNCTION__,
                                    hetu::impl::Dot, x, y, out, stream);
  } else if (dim_x == 2 && dim_y == 2) {
    HT_ASSERT(x->shape(trans_left ? 0 : 1) == y->shape(trans_right ? 1 : 0))
      << "Invalid shapes for matrix multiplication: " << x->shape()
      << " (transpose = " << trans_left << ") vs. " << y->shape()
      << " (transpose = " << trans_right << "). ";
    out = output.is_defined()
      ? output
      : NDArray::empty(
          {x->shape(trans_left ? 1 : 0), y->shape(trans_right ? 0 : 1)},
          x->device(), x->dtype(), stream_id);
    HT_DISPATCH_KERNEL_CPU_AND_CUDA(x->device().type(), __FUNCTION__,
                                    hetu::impl::MatMul, x, trans_left, y,
                                    trans_right, out, stream);
  } else if (dim_x == 1 && dim_y == 2) {
    auto out_ = NDArray::empty({1, y->shape(trans_right ? 0 : 1)}, x->device(),
                               x->dtype(), stream_id);
    out_ = NDArray::matmul(NDArray::unsqueeze(x, 0), y, false, trans_right, stream_id, out_);
    out = NDArray::squeeze(out_, 0);
    if (output.is_defined()) {
      NDArray::copy(out, stream_id, output);
      out = output;
    }
  } else if (dim_x == 2 && dim_y == 1) {
    HT_ASSERT(x->shape(trans_left ? 0 : 1) == y->shape(0))
      << "Invalid shapes for matrix multiplication: " << x->shape()
      << " (transpose = " << trans_left << ") vs. " << y->shape()
      << " (transpose = " << trans_right << "). ";
    out = output.is_defined()
      ? output
      : NDArray::empty({x->shape(trans_left ? 1 : 0)}, x->device(), x->dtype(),
                       stream_id);
    HT_DISPATCH_KERNEL_CPU_AND_CUDA(x->device().type(), __FUNCTION__,
                                    hetu::impl::MatVecMul, x, trans_left, y, 
                                    out, stream);
  } else if ((dim_x >= 3 && (dim_y == 1 || dim_y == 2))
          || ((dim_x == 1 || dim_x == 2) && dim_y >= 3)) {
    const auto transpose = dim_y > dim_x;
    const auto x_ = transpose ? y : x;
    const auto y_ = transpose ? x : y;
    const auto dim_y_ = transpose ? dim_x : dim_y;
    const auto output_transpose = dim_y_ == 2 && transpose;

    if (transpose) {
      std::swap(trans_left, trans_right);
      trans_left = !trans_left;
      trans_right = !trans_right;
    }
    
    auto x_shape = x_->shape();
    NDArray x_trans = x_;
    if (trans_left) {
      std::iter_swap(x_shape.end() - 2, x_shape.end() - 1);
      const auto dim_x_ = transpose ? dim_y : dim_x;
      auto ndims_x_ = HTAxes(dim_x_);
      std::iota(ndims_x_.begin(), ndims_x_.end(), 0);
      std::iter_swap(ndims_x_.end() - 2, ndims_x_.end() - 1);
      x_trans = NDArray::permute(x_, ndims_x_, stream_id);
    }
    
    auto output_shape = HTShape(x_shape.begin(), x_shape.end() - 1);
    auto folded_dim = std::accumulate(output_shape.begin(), output_shape.end(), 1,
                                      [](int64_t x, int64_t y) { return x * y; });
    if (dim_y_ == 2) {
      output_shape.emplace_back(y_->shape(trans_right ? 0 : 1));
    }
    const auto x_folded = NDArray::reshape(x_trans, {folded_dim, x_shape.back()}, stream_id);
    auto folded_shape = HTShape({folded_dim});
    if (dim_y_ == 2) {
      folded_shape.emplace_back(y_->shape(trans_right ? 0 : 1));
    }
    auto out_folded = NDArray::empty(folded_shape, x->device(), x->dtype(), stream_id);

    if (output_transpose) {
      auto out_trans = NDArray::empty(output_shape, x->device(), x->dtype(), stream_id);
      out_folded = NDArray::matmul(x_folded, y_, false, trans_right, stream_id, out_folded);
      out_trans = NDArray::reshape(out_folded, output_shape, stream_id);
      std::iter_swap(output_shape.end() - 2, output_shape.end() - 1);
      out = output.is_defined()
        ? output
        : NDArray::empty(output_shape, x->device(), x->dtype(), stream_id);
      const auto dim_out = out->ndim();
      auto ndims_out = HTAxes(dim_out);
      std::iota(ndims_out.begin(), ndims_out.end(), 0);
      std::iter_swap(ndims_out.end() - 2, ndims_out.end() - 1);
      NDArray::contiguous(NDArray::permute(out_trans, ndims_out, stream_id),
                          stream_id, out);
    } else {
      out_folded = NDArray::matmul(x_folded, y_, false, trans_right, stream_id, out_folded);
      if (output.is_defined()) {
        NDArray::copy(NDArray::reshape(out_folded, output_shape, stream_id),
                      stream_id, output);
        out = output;
      } else {
        out = NDArray::reshape(out_folded, output_shape, stream_id);
      }
    }
  } else {
    const auto x_shape = x->shape();
    const auto y_shape = y->shape();

    const auto n = x_shape.cend()[-2];
    const auto m_x = x_shape.back();
    const auto m_y = y_shape.cend()[-2];
    const auto p = y_shape.back();

    const auto batch_shape_x = HTShape(x_shape.begin(), x_shape.end() - 2);
    const auto batch_shape_y = HTShape(y_shape.begin(), y_shape.end() - 2);
    auto output_shape = NDArrayMeta::Broadcast(batch_shape_x, batch_shape_y);
    const auto batch_size = std::accumulate(output_shape.begin(), output_shape.end(), 1,
                                            [](int64_t x, int64_t y) { return x * y; });

    const auto broadcast_shape_x = [&output_shape, n, m_x] {
                                      HTShape ret(output_shape);
                                      ret.emplace_back(n);
                                      ret.emplace_back(m_x);
                                      return ret; }();
    const auto broadcast_shape_y = [&output_shape, m_y, p] {
                                      HTShape ret(output_shape);
                                      ret.emplace_back(m_y);
                                      ret.emplace_back(p);
                                      return ret; }();
    auto broadcast_x = (x_shape == broadcast_shape_x) ? x : NDArray::empty(broadcast_shape_x, x->device(), x->dtype(), stream_id);
    broadcast_x = NDArray::reshape(x_shape != broadcast_shape_x
                                      ? NDArray::broadcast(x, broadcast_shape_x, stream_id, broadcast_x)
                                      : x,
                                    {batch_size, n, m_x}, stream_id);
    auto broadcast_y = (y_shape == broadcast_shape_y) ? y : NDArray::empty(broadcast_shape_y, y->device(), y->dtype(), stream_id);
    broadcast_y = NDArray::reshape(y_shape != broadcast_shape_y
                                      ? NDArray::broadcast(y, broadcast_shape_y, stream_id, broadcast_y)
                                      : y,
                                    {batch_size, m_y, p}, stream_id);

    output_shape.emplace_back(trans_left ? m_x : n);
    output_shape.emplace_back(trans_right ? m_y : p);

    out =
      NDArray::empty({batch_size, output_shape.cend()[-2], output_shape.back()},
                     x->device(), x->dtype(), stream_id);
    HT_DISPATCH_KERNEL_CPU_AND_CUDA(x->device().type(), __FUNCTION__,
                                  hetu::impl::BatchMatMul, broadcast_x, trans_left,
                                  broadcast_y, trans_right, out, stream);
    if (output.is_defined()) {
      NDArray::copy(NDArray::reshape(out, output_shape, stream_id),
                    stream_id, output);
      out = output;
    } else {
      out = NDArray::reshape(out, output_shape, stream_id);
    }
  }
  return out;
}

NDArray NDArray::matmul4bit(const NDArray& x, const NDArray& y, 
                            const NDArray& absmax,  const NDArray& datatype,
                            bool trans_left, bool trans_right, 
                            int blocksize,
                            StreamIndex stream_id, NDArray& output) {
  HT_ASSERT(x->shape(trans_left ? 0 : 1) == y->shape(trans_right ? 1 : 0))
      << "Invalid shapes for matrix multiplication: " << x->shape(trans_left ? 0 : 1)
      << " (transpose = " << trans_left << ") vs. " << y->shape(trans_right ? 1 : 0)
      << " (transpose = " << trans_right << "). ";
  NDArray out = output.is_defined()
              ? output
              : NDArray::empty(
                  {x->shape(trans_left ? 1 : 0), y->shape(trans_right ? 0 : 1)},
                  x->device(), x->dtype(), stream_id);
  Stream stream(x->device(), stream_id);
  HT_DISPATCH_KERNEL_CUDA_ONLY(x->device().type(), __FUNCTION__,
                                hetu::impl::MatMul4Bit, x, trans_left, y,
                                trans_right, absmax, datatype, 
                                out, blocksize, stream);
  return out;
}

NDArray NDArray::bmm(const NDArray& x, const NDArray& y,
                     bool trans_left, bool trans_right,
                     StreamIndex stream_id, NDArray& output) {
const HTShape& a = x->shape();
  const HTShape& b = y->shape();
  int ndims = a.size() - 2;
  HT_ASSERT(a.size() >= 2 && b.size() >= 2 && a.size() == b.size() &&
            a.at(trans_left ? ndims : ndims + 1) ==
              b.at(trans_right ? ndims + 1 : ndims))
    << "Invalid input shapes for:"
    << " (shape_left) " << a << " (shape_right) " << b << " (transpose_left) "
    << trans_left << " (transpose_right) " << trans_right;
  HTShape shape = {};
  int64_t batch_size = 1;
  for (int i = 0; i < ndims; ++i) {
    HT_ASSERT(a[i] == b[i]);
    batch_size *= a[i];
  }
  if (output.is_defined())
    output = NDArray::contiguous(output, stream_id);
  shape.emplace_back(batch_size);
  shape.emplace_back(a.at(trans_left ? ndims + 1 : ndims));
  shape.emplace_back(b.at(trans_right ? ndims : ndims + 1));
  auto x_ = NDArray::reshape(x, {batch_size, a.at(ndims), a.at(ndims + 1)}, stream_id);
  auto y_ = NDArray::reshape(y, {batch_size, b.at(ndims), b.at(ndims + 1)}, stream_id);
  NDArray out = output.is_defined()
    ? NDArray::reshape(output, shape, stream_id)
    : NDArray::empty(shape, x->device(), x->dtype(), stream_id);
  Stream stream(x->device(), stream_id);
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(x->device().type(), __FUNCTION__,
                                  hetu::impl::BatchMatMul, x_, trans_left, y_,
                                  trans_right, out, stream);
  return out;
}

NDArray NDArray::dot(const NDArray& x, const NDArray& y, StreamIndex stream_id,
                     NDArray& output) {
  auto output_shape = NDArrayMeta::Broadcast(x->shape(), y->shape());
  HT_ASSERT(!output_shape.empty())
    << "Shapes cannot be broadcast together: " << x->shape() << " vs. "
    << y->shape();
  NDArray out = output.is_defined()
    ? output
    : NDArray::empty(output_shape, x->device(), x->dtype(), stream_id);
  Stream stream(x->device(), stream_id);
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(x->device().type(), __FUNCTION__,
                                  hetu::impl::MatDot, x, y, out, stream);
  return out;
}

NDArray NDArray::index_add(const NDArray& x, const NDArray& ids,
                           const NDArray& y, int64_t dim, StreamIndex stream_id,
                           NDArray& output) {
  NDArray tmp = NDArray::empty(x->shape(), x->device(), x->dtype(), stream_id);
  NDArray out = output.is_defined()
    ? output
    : NDArray::empty(x->shape(), x->device(), x->dtype(), stream_id);
  Stream stream(x->device(), stream_id);
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(x->device().type(), __FUNCTION__,
                                  hetu::impl::IndexAdd, y, ids, tmp, dim,
                                  stream);
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(x->device().type(), __FUNCTION__,
                                  hetu::impl::AddElewise, x, tmp, out, stream);
  return out;
}

NDArray NDArray::reshape(const NDArray& input, const HTShape& new_shape,
                         StreamIndex stream_id, NDArray& output) {
  if (output.is_defined()) {
    output = NDArray::view(output, input->shape());
    NDArray::contiguous(input, stream_id, output);
    output = NDArray::view(output, new_shape);
    return output;
  }
  else {
    NDArray out = NDArray::contiguous(input, stream_id);
    return NDArray::view(out, new_shape);
  }
}

NDArray NDArray::view(const NDArray& input, const HTShape& view_shape) {
  NDArrayMeta output_meta = input->meta();
  output_meta.view(view_shape);
  return NDArray(output_meta, input->storage(), input->storage_offset());
}

NDArray NDArray::squeeze(const NDArray& input) {
  NDArrayMeta output_meta = input->meta();
  output_meta.squeeze();
  return NDArray(output_meta, input->storage(), input->storage_offset());
}

NDArray NDArray::squeeze(const NDArray& input, int64_t dim) {
  NDArrayMeta output_meta = input->meta();
  output_meta.squeeze(dim);
  return NDArray(output_meta, input->storage(), input->storage_offset());
}

NDArray NDArray::unsqueeze(const NDArray& input, int64_t dim) {
  NDArrayMeta output_meta = input->meta();
  output_meta.unsqueeze(dim);
  return NDArray(output_meta, input->storage(), input->storage_offset());
}

NDArray NDArray::flatten(const NDArray& input, int64_t start_dim,
                         int64_t end_dim) {
  NDArrayMeta output_meta = input->meta();
  output_meta.flatten(start_dim, end_dim);
  return NDArray(output_meta, input->storage(), input->storage_offset());
}

NDArray NDArray::permute(const NDArray& input, const HTAxes& dims,
                         StreamIndex stream_id) {
  auto output_meta = NDArrayMeta().set_dtype(input->dtype())
                                  .set_shape(input->shape())
                                  .set_stride(input->stride())
                                  .set_device(input->device());
  output_meta.permute(dims);
  NDArray out = NDArray(output_meta, input->storage(), input->storage_offset());
  return out;
}

NDArray NDArray::movedim(const NDArray& input, int64_t src, int64_t dst,
                         StreamIndex stream_id) {
  int64_t len = input->ndim();
  src = NDArrayMeta::ParseAxis(src, len);
  dst = NDArrayMeta::ParseAxis(dst, len);
  HTAxes dims(len);
  if (src < dst) {
    for (int i = 0; i < src; ++i) {
      dims[i] = i;
    }
    for (int i = src; i < dst; ++i) {
      dims[i] = i + 1;
    }
    dims[dst] = src;
    for (int i = dst + 1; i < len; ++i) {
      dims[i] = i;
    }
  } else if (src > dst) {
    for (int i = 0; i < dst; ++i) {
      dims[i] = i;
    }
    dims[dst] = src;
    for (int i = dst + 1; i < src + 1; ++i) {
      dims[i] = i - 1;
    }
    for (int i = src + 1; i < len; ++i) {
      dims[i] = i;
    }
  } else {
    for (int i = 0; i < len; ++i) {
      dims[i] = i;
    }
  }
  return NDArray::permute(input, dims, stream_id);
}

NDArray NDArray::adddim(const NDArray& input, int64_t dim, int64_t size,
                        StreamIndex stream_id, NDArray& output) {
  int64_t len = input->ndim();
  dim = NDArrayMeta::ParseAxis(dim, len);
  HT_ASSERT(size > 0);
  HTAxes dims(len);
  HTShape output_shape = {};
  for (int i = 0; i < dim; ++i) {
    output_shape.emplace_back(input->shape(i));
  }
  output_shape.emplace_back(size);
  for (int i = dim + 1; i < len; ++i) {
    output_shape.emplace_back(input->shape(i));
  }
  NDArray out = output.is_defined()
    ? output
    : NDArray::empty(output_shape, input->device(), input->dtype(), stream_id);
  Stream stream(input->device(), stream_id);
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(input->device().type(), __FUNCTION__,
                                  hetu::impl::BroadcastShape, input, out,
                                  output_shape, stream);
  return out;
}

NDArray NDArray::diagonal(const NDArray& input, int64_t dim1, int64_t dim2,
                          int64_t offset, StreamIndex stream_id) {
  int64_t ndim = input->ndim();
  int64_t dim1_ = NDArrayMeta::ParseAxis(dim1, ndim);
  int64_t dim2_ = NDArrayMeta::ParseAxis(dim2, ndim);
  HT_ASSERT(dim1_ != dim2_)
  << "diagonal dimensions cannot be identical. dim1: " << dim1 << ", dim2: " << dim2;

  int64_t diag_size;
  int64_t storage_offset = input->storage_offset();
  if (offset >= 0) {
    diag_size = std::min(input->shape(dim1_), input->shape(dim2_) - offset);
  } else {
    diag_size = std::min(input->shape(dim1_) + offset, input->shape(dim2_));
  }

  HT_ASSERT(diag_size > 0);
  
  if (offset >= 0) {
    storage_offset += offset * input->stride(dim2_);
  } else if (offset < 0) {
    storage_offset -= offset * input->stride(dim1_);
  }

  HTShape output_shape(input->shape().begin(), input->shape().end());
  HTStride output_stride(input->stride().begin(), input->stride().end());
  output_shape.erase(output_shape.begin() + std::max(dim1_, dim2_));
  output_stride.erase(output_stride.begin() + std::max(dim1_, dim2_));
  output_shape.erase(output_shape.begin() + std::min(dim1_, dim2_));
  output_stride.erase(output_stride.begin() + std::min(dim1_, dim2_));
  output_shape.emplace_back(diag_size);
  output_stride.emplace_back(input->stride(dim1_) + input->stride(dim2_));

  auto output_meta = NDArrayMeta().set_dtype(input->dtype())
                                  .set_shape(output_shape)
                                  .set_stride(output_stride)
                                  .set_device(input->device());
  NDArray out = NDArray(output_meta, input->storage(), storage_offset);
  return out;
}

NDArray NDArray::diagonal_grad(const NDArray& input, int64_t dim1, int64_t dim2,
                               StreamIndex stream_id, NDArray& output) {
  HTShape output_shape = {};
  int64_t len = input->ndim();
  dim1 = NDArrayMeta::ParseAxis(dim1, len + 1);
  dim2 = NDArrayMeta::ParseAxis(dim2, len + 1);
  HT_ASSERT(dim1 < dim2);
  for (int i = 0; i < dim2; ++i) {
    output_shape.emplace_back(input->shape(i));
  }
  output_shape.emplace_back(input->shape(dim1));
  for (int i = dim2; i < len; ++i) {
    output_shape.emplace_back(input->shape(i));
  }
  NDArray out = output.is_defined()
    ? output
    : NDArray::empty(output_shape, input->device(), input->dtype(), stream_id);
  Stream stream(input->device(), stream_id);
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(input->device().type(), input->dtype(),
                                  hetu::impl::DiagonalGradient, input, out,
                                  dim1, dim2, stream);
  return out;
}

NDArray NDArray::as_strided(const NDArray& input, const HTShape& outshape,
                            const HTStride& stride, int64_t storage_offset,
                            StreamIndex stream_id) {
  auto output_meta = NDArrayMeta().set_dtype(input->dtype())
                                  .set_shape(outshape)
                                  .set_stride(stride)
                                  .set_device(input->device());
  NDArray out = NDArray(output_meta, input->storage(), storage_offset);
  return out;
}

NDArrayList NDArray::split(const NDArray& input, size_t num_chunks,
                           int64_t axis, StreamIndex stream_id) {
  auto parsed_axis = NDArrayMeta::ParseAxis(axis, input->ndim());
  HT_ASSERT(parsed_axis == 0) << "Currently we only support split on axis 0.";
  HT_ASSERT(num_chunks <= static_cast<size_t>(input->shape(parsed_axis)) || input->shape(parsed_axis) == 0)
    << "Cannot split axis " << axis << " of shape " << input->shape()
    << " into " << num_chunks << " chunks";
  auto avg_chunk_size = DIVUP(input->shape(parsed_axis), num_chunks);
  HTShape chunks(num_chunks, avg_chunk_size);
  chunks[num_chunks - 1] =
    input->shape(parsed_axis) - (num_chunks - 1) * avg_chunk_size;
  return split(input, chunks, axis, stream_id);
}

NDArrayList NDArray::split(const NDArray& input, const HTShape& chunks,
                           int64_t axis, StreamIndex stream_id) {
  auto parsed_axis = NDArrayMeta::ParseAxis(axis, input->ndim());
  if (parsed_axis == 0) {
    auto split_shapes = NDArrayMeta::Split(input->shape(), chunks, 0);
    size_t interval = input->shape(0) == 0 ? 0 : input->numel() / input->shape(0);
    NDArrayList ret;
    ret.reserve(split_shapes.size());
    auto offset = input->storage_offset();
    for (size_t i = 0; i < split_shapes.size(); i++) {
      auto split_meta = input->meta();
      split_meta.set_shape(split_shapes[i]);
      ret.emplace_back(split_meta, input->storage(), offset);
      offset += chunks[i] * interval;
    }
    return ret;
  } else {
    HT_NOT_IMPLEMENTED << "Currently we only support split on axis 0.";
    __builtin_unreachable();
  }
}

//_____________________________________________________________________________________

NDArray NDArray::avgpool(const NDArray& input, const size_t kernel_H,
                         const size_t kernel_W, const size_t padding,
                         const size_t stride,
                         StreamIndex stream_id,
                         NDArray& output) {  
  NDArray out;
  if (output.is_defined())
    out = output;
  else {
    int64_t N = input->shape(0);
    int64_t C = input->shape(1);
    int64_t H = input->shape(2);
    int64_t W = input->shape(3);
    int64_t p_H = (H + 2 * padding - kernel_H) / stride + 1;
    int64_t p_W = (W + 2 * padding - kernel_W) / stride + 1;
    out = NDArray::empty({N, C, p_H, p_W}, input->device(), input->dtype(), stream_id);
  }
  Stream stream(input->device(), stream_id);
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(input->device().type(), __FUNCTION__, hetu::impl::AvgPool,
                                  input, kernel_H, kernel_W,
                                  out, padding, stride,
                                  stream);
  return out;
}

NDArray NDArray::arrayset(NDArray& input, double value,
                          StreamIndex stream_id) {  
  Stream stream(input->device(), stream_id);
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(
      input->device().type(), __FUNCTION__, hetu::impl::ArraySet, 
      input, value, stream);  
  return input;
}

NDArrayList NDArray::batchnorm(const NDArray& input, const NDArray& bn_scale, const NDArray& bn_bias,
                               NDArray& running_mean, NDArray& running_var,
                               double momentum, double eps,
                               StreamIndex stream_id,
                               NDArray& output,
                               NDArray& save_mean,
                               NDArray& save_var) {
  NDArray out = output.is_defined() ? output : NDArray::empty_like(input);
  NDArray savemean = save_mean.is_defined() ? save_mean : NDArray::empty({input->shape(1)}, input->device(), input->dtype(), stream_id);
  NDArray savevar = save_var.is_defined() ? save_var : NDArray::empty({input->shape(1)}, input->device(), input->dtype(), stream_id);
  Stream stream(input->device(), stream_id);
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(
    input->device().type(), __FUNCTION__, hetu::impl::BatchNorm, input,
    bn_scale, bn_bias, out, momentum, eps,
    running_mean, running_var, 
    savemean, savevar, stream);
  return {out, savemean, savevar};
}

NDArray NDArray::broadcast(const NDArray& input, const HTShape& shape,
                           StreamIndex stream_id,
                           NDArray& output) {
  NDArray out = output.is_defined() ? output : NDArray::empty(shape, input->device(), input->dtype(), stream_id);
  Stream stream(input->device(), stream_id);
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(input->device().type(), __FUNCTION__,
                                  hetu::impl::Broadcast, input,
                                  out, stream);
  return out;
}

NDArray NDArray::broadcast(const NDArray& input, const HTShape& shape,
                           const HTAxes& add_axes,
                           StreamIndex stream_id,
                           NDArray& output) {
  NDArray out = output.is_defined() ? output : NDArray::empty(shape, input->device(), input->dtype(), stream_id);
  Stream stream(input->device(), stream_id);
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(input->device().type(), __FUNCTION__,
                                  hetu::impl::BroadcastShape, input,
                                  out, add_axes, stream);
  return out;
}

NDArray NDArray::conv2d(const NDArray& input, const NDArray& filter, 
                        const HTShape& padding, const HTShape& stride,
                        StreamIndex stream_id,
                        NDArray& output) {
  NDArray out;
  if (output.is_defined()) 
    out = output;
  else {
    int64_t N = input->shape(0);
    int64_t H = input->shape(2);
    int64_t W = input->shape(3);
    int64_t f_O = filter->shape(0);
    int64_t f_H = filter->shape(2);
    int64_t f_W = filter->shape(3);
    int64_t out_H = (H + 2 * padding[0] - f_H) / stride[0] + 1;
    int64_t out_W = (W + 2 * padding[1] - f_W) / stride[1] + 1;
    out = NDArray::empty({N, f_O, out_H, out_W}, input->device(),
                         input->dtype(), stream_id);
  }
  Stream stream(input->device(), stream_id);
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(input->device().type(), __FUNCTION__, hetu::impl::Conv2d,
                                  input, filter, out,
                                  padding[0], padding[1],
                                  stride[0], stride[1], stream);
  return out;
}

NDArray NDArray::cos(const NDArray& input,
                     StreamIndex stream_id,
                     NDArray& output) {
  NDArray out = output.is_defined() ? output : NDArray::empty_like(input);
  Stream stream(input->device(), stream_id);
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(input->device().type(), __FUNCTION__,
                                  hetu::impl::Cos, input,
                                  out, stream);
  return out;
}

NDArray NDArray::dequantization(const NDArray& input, 
                                NDArray& absmax, 
                                DataType dqtype,
                                int64_t blocksize, 
                                StreamIndex stream_id,
                                const NDArray& code,
                                NDArray& output) {
  NDArray out;
  if (input->dtype() == DataType::INT8)
    HT_ASSERT(code.is_defined());
  if (output.is_defined() && output->dtype() == dqtype) 
    out = output;
  else 
    out = NDArray::empty(input->shape(), input->device(), dqtype, stream_id);
  Stream stream(input->device(), stream_id);
  HT_DISPATCH_KERNEL_CUDA_ONLY(input->device().type(), __FUNCTION__,
                               hetu::impl::DeQuantization, input, absmax, code,
                               out, blocksize, stream);
  return out;
}

// TODO: support dynamic if output is not defined
NDArray NDArray::embedding(const NDArray& input, const NDArray& id,
                           StreamIndex stream_id,
                           NDArray& output) {
  NDArray out;
  if (output.is_defined()) 
    out = output;
  else {
    HTShape output_shape = id->shape();
    output_shape.emplace_back(input->shape(1));
    out =
      NDArray::empty(output_shape, input->device(), input->dtype(), stream_id);
  }
  Stream stream(input->device(), stream_id);
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(input->device().type(), __FUNCTION__,
                                  hetu::impl::EmbeddingLookup, input,
                                  id, out, stream);
  return out;
}

NDArrayList NDArray::fused_layernorm(const NDArray& input, const NDArray& ln_scale, const NDArray& ln_bias, 
                               const HTShape& normalized_shape, double eps,
                               StreamIndex stream_id,
                               NDArray& output,
                               NDArray& save_mean,
                               NDArray& save_var) {
  NDArray out = output.is_defined() ? output : NDArray::empty_like(input);
  HTShape local_shape = input->shape();
  int ndim = local_shape.size();
  local_shape[ndim - 1] = 1;
  NDArray savemean = save_mean.is_defined()
    ? save_mean
    : NDArray::empty(normalized_shape, input->device(), input->dtype(),
                     stream_id);
  NDArray savevar = save_var.is_defined()
    ? save_var
    : NDArray::empty(normalized_shape, input->device(), input->dtype(),
                     stream_id);
  Stream stream(input->device(), stream_id);
  HT_DISPATCH_KERNEL_CUDA_ONLY(input->device().type(), __FUNCTION__,
                               hetu::impl::FusedLayerNorm, input,
                               ln_scale, ln_bias, savemean, savevar, 
                               out, normalized_shape.size(), eps, stream);  
  return {out, savemean, savevar};
}

NDArrayList NDArray::fused_rmsnorm(const NDArray& input, const NDArray& ln_scale,
                                   const HTShape& normalized_shape, double eps,
                                   StreamIndex stream_id,
                                   NDArray& output,
                                   NDArray& save_var) {
  NDArray out = output.is_defined() ? output : NDArray::empty_like(input);
  HTShape local_shape = input->shape();
  int ndim = local_shape.size();
  local_shape[ndim - 1] = 1;
  NDArray savevar = save_var.is_defined()
    ? save_var
    : NDArray::empty(normalized_shape, input->device(), input->dtype(),
                     stream_id);
  Stream stream(input->device(), stream_id);
  HT_DISPATCH_KERNEL_CUDA_ONLY(input->device().type(), __FUNCTION__,
                               hetu::impl::FusedRMSNorm, input,
                               ln_scale, savevar, 
                               out, normalized_shape.size(), eps, stream);  
  return {out, savevar};
}

NDArray NDArray::gather(const NDArray& input, const NDArray& id, int64_t dim,
                        StreamIndex stream_id,
                        NDArray& output) {
  NDArray out = output.is_defined() ? output : NDArray::empty(id->shape(), input->device(), input->dtype(), stream_id);
  Stream stream(input->device(), stream_id);
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(input->device().type(), __FUNCTION__,
                                  hetu::impl::Gather, input,
                                  id, out, dim, stream);
  return out;
}

NDArrayList NDArray::instancenorm(const NDArray& input, double eps,
                                  StreamIndex stream_id,
                                  NDArray& output,
                                  NDArray& save_mean,
                                  NDArray& save_var) {
  NDArray out = output.is_defined() ? output : NDArray::empty_like(input);
  HTShape local_shape = input->shape();
  local_shape[3] = 1;
  local_shape[2] = 1;
  NDArray savemean = save_mean.is_defined()
    ? save_mean
    : NDArray::empty(local_shape, input->device(), input->dtype(), stream_id);
  NDArray savevar = save_var.is_defined()
    ? save_var
    : NDArray::empty(local_shape, input->device(), input->dtype(), stream_id);
  Stream stream(input->device(), stream_id);
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(
    input->device().type(), __FUNCTION__, hetu::impl::InstanceNorm, input,
    savemean, savevar, out, eps, stream);
  return {out, savemean, savevar};  
}

NDArray NDArray::kldiv(const NDArray& preds, const NDArray& labels,
                       ReductionType reduction,
                       StreamIndex stream_id,
                       NDArray& output) {
  NDArray out;
  if (output.is_defined()) 
    out = output;
  else {
    if (reduction != kNONE)
      out = NDArray::empty({1}, preds->device(), preds->dtype(), stream_id);
    else
      out = NDArray::empty(preds->shape(), preds->device(), preds->dtype(),
                           stream_id);
  }
  Stream stream(preds->device(), stream_id);
  NDArray unreduced =
    reduction == kNONE ? out : NDArray::empty_like(preds);
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(preds->device().type(), __FUNCTION__,
                                  hetu::impl::KLDivLoss, preds,
                                  labels, unreduced, stream);
  if (reduction != kNONE) {
    NDArray::reduce(unreduced, reduction, HTAxes(), false, stream_id, out);
  }
  return out;
}

NDArrayList NDArray::layernorm(const NDArray& input, const NDArray& bn_scale, const NDArray& bn_bias, 
                               const HTShape& normalized_shape, double eps,
                               StreamIndex stream_id,
                               NDArray& output,
                               NDArray& save_mean,
                               NDArray& save_var) {
  NDArray out = output.is_defined() ? output : NDArray::empty_like(input);
  HTShape local_shape = input->shape();
  int ndim = local_shape.size();
  local_shape[ndim - 1] = 1;
  NDArray savemean = save_mean.is_defined()
    ? save_mean
    : NDArray::empty(normalized_shape, input->device(), input->dtype(),
                     stream_id);
  NDArray savevar = save_var.is_defined()
    ? save_var
    : NDArray::empty(normalized_shape, input->device(), input->dtype(),
                     stream_id);
  Stream stream(input->device(), stream_id);
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(input->device().type(), __FUNCTION__,
                                  hetu::impl::LayerNorm, input,
                                  bn_scale, bn_bias, savemean, savevar, 
                                  out, normalized_shape.size(), eps, stream);  
  return {out, savemean, savevar};
}

NDArray NDArray::leakyrelu(const NDArray& input, double alpha,
                           StreamIndex stream_id,
                           NDArray& output) {
  NDArray out = output.is_defined() ? output : NDArray::empty_like(input);
  Stream stream(input->device(), stream_id);
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(input->device().type(), __FUNCTION__,
                                  hetu::impl::LeakyRelu, input,
                                  alpha, out, stream);
  return out;
}

NDArray NDArray::linear(const NDArray& a, const NDArray& b, const NDArray& bias, 
                        bool trans_a, bool trans_b,
                        StreamIndex stream_id,
                        NDArray& output) {
  HT_ASSERT(a->ndim() == 2 && b->ndim() == 2 &&
            a->shape(trans_a ? 0 : 1) == b->shape(trans_b ? 1 : 0))
    << "Invalid shapes for matrix multiplication: " << a->shape()
    << " (transpose = " << trans_a << ") vs. " << b->shape()
    << " (transpose = " << trans_b << "). ";
  NDArray out = output.is_defined()
    ? output
    : NDArray::empty({a->shape(trans_a ? 1 : 0), b->shape(trans_b ? 0 : 1)},
                     a->device(), a->dtype(), stream_id);
  if (a->shape(trans_a ? 0 : 1) == 0) {
    NDArray::zeros_(out, stream_id);
    return out;
  }
  Stream stream(a->device(), stream_id);
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(a->device().type(), __FUNCTION__, hetu::impl::Linear,
                                  a, trans_a, b, trans_b,
                                  bias, out, stream);
  return out; 
}

NDArray NDArray::mseloss(const NDArray& preds, const NDArray& labels,
                         ReductionType reduction,
                         StreamIndex stream_id,
                         NDArray& output) {
  NDArray out;
  if (output.is_defined()) 
    out = output;
  else {
    if (reduction != kNONE)
      out = NDArray::empty({1}, preds->device(), preds->dtype(), stream_id);
    else
      out = NDArray::empty(preds->shape(), preds->device(), preds->dtype(),
                           stream_id);
  }
  Stream stream(preds->device(), stream_id);
  NDArray unreduced =
    reduction == kNONE ? out : NDArray::empty_like(preds);
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(preds->device().type(), __FUNCTION__,
                                  hetu::impl::MSELoss, preds,
                                  labels, unreduced, stream);
  if (reduction != kNONE) {
    NDArray::reduce(unreduced, reduction, HTAxes(), false, stream_id, out);
  }
  return out;
}

NDArray NDArray::nllloss(const NDArray& preds, const NDArray& labels,
                         ReductionType reduction,
                         StreamIndex stream_id,
                         NDArray& output) {
  NDArray out;
  if (output.is_defined()) 
    out = output;
  else {
    if (reduction != kNONE)
      out = NDArray::empty({1}, preds->device(), preds->dtype(), stream_id);
    else
      out = NDArray::empty(labels->shape(), preds->device(), preds->dtype(),
                           stream_id);
  }
  Stream stream(preds->device(), stream_id);
  NDArray unreduced = reduction == kNONE
    ? out
    : NDArray::empty(labels->shape(), preds->device(), preds->dtype(),
                     stream_id);
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(preds->device().type(), __FUNCTION__,
                                  hetu::impl::NLLLoss, preds,
                                  labels, unreduced, stream);
  if (reduction != kNONE) {
    NDArray::reduce(unreduced, reduction, HTAxes(), false, stream_id, out);
  }
  return out;
}

NDArray NDArray::norm(const NDArray& input, int64_t p, int64_t dim, 
                      bool keepdim,
                      StreamIndex stream_id,
                      NDArray& output) {
  NDArray out;
  if (output.is_defined())
    out = output;
  else {
    HTShape outshape = input->shape();
    int64_t axi = dim >= 0 ? dim: dim + outshape.size();
    if (keepdim) 
      outshape[axi] = 1;
    else 
      outshape.erase(outshape.begin() + axi);
    out = NDArray::empty(outshape, input->device(), input->dtype(), stream_id);
  }
  Stream stream(input->device(), stream_id);
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(input->device().type(), __FUNCTION__, hetu::impl::Norm,
                                  input, out, dim, p, stream);
  return out;
}

NDArray NDArray::onehot(const NDArray& input, size_t num_classes,
                        StreamIndex stream_id,
                        NDArray& output) {
  HTShape out_shape = input->shape();
  out_shape.emplace_back(num_classes);
  NDArray out = output.is_defined()
    ? output
    : NDArray::empty(out_shape, input->device(), input->dtype(), stream_id);
  Stream stream(input->device(), stream_id);
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(input->device().type(), __FUNCTION__,
                                  hetu::impl::Onehot, input,
                                  num_classes, out, stream);
  return out;
}

NDArray NDArray::pad(const NDArray& input, const HTShape& paddings, 
                     std::string mode, double constant,
                     StreamIndex stream_id,
                     NDArray& output) {
  HTShape Infer = input->shape();
  size_t len = paddings.size();
  for (size_t i = 0; i < 4; ++i) {
    if (i >= (4 - len / 2)) {
      Infer[i] = Infer[i] + paddings[(i - (4 - len / 2)) * 2] +
        paddings[(i - (4 - len / 2)) * 2 + 1];
    }
  }
  NDArray out = output.is_defined()
    ? output
    : NDArray::empty(Infer, input->device(), input->dtype(), stream_id);
  Stream stream(input->device(), stream_id);
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(input->device().type(), __FUNCTION__, hetu::impl::Pad,
                                  input, out, paddings,
                                  stream, mode, constant);
  return out;
}

NDArrayList NDArray::quantization(const NDArray& input,
                                  DataType qtype, 
                                  int64_t blocksize,
                                  bool stochastic,
                                  StreamIndex stream_id,
                                  const NDArray& code,
                                  NDArray& absmax,
                                  NDArray& output) {
  NDArray absmax_, out;
  if (qtype == DataType::INT8)
    HT_ASSERT(code.is_defined());
  if (absmax.is_defined()) 
    absmax_ = absmax;
  else {
    HTShape absmax_shape = {int64_t(input->numel() / blocksize)};
    absmax_ = NDArray::empty(absmax_shape, input->device(), kFloat32, stream_id);
  }
  if (output.is_defined() && output->dtype() == qtype) 
    out = output;
  else 
    out = NDArray::empty(input->shape(), input->device(), qtype, stream_id);
  Stream stream(input->device(), stream_id);
  HT_DISPATCH_KERNEL_CUDA_ONLY(input->device().type(), __FUNCTION__,
                               hetu::impl::Quantization, input, absmax_, code,
                               out, blocksize, stochastic, stream);
  return {absmax_, out};
}

NDArray NDArray::repeat(const NDArray& input, HTShape repeats,
                        StreamIndex stream_id,
                        NDArray& output) {
  NDArray out;
  if (output.is_defined()) 
    out = output;
  else {
    HTShape output_shape = repeats;
    HT_ASSERT(output_shape.size() >= input->ndim());
    for (size_t i = 0; i < input->ndim(); ++i) {
      output_shape[i + output_shape.size() - input->ndim()] *= input->shape(i); 
    }
    out = NDArray::empty(output_shape, input->device(), input->dtype(), stream_id);
  }
  Stream stream(input->device(), stream_id);
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(input->device().type(), __FUNCTION__,
                                  hetu::impl::Repeat, input,
                                  out, stream);
  return out;
}

NDArray NDArray::roll(const NDArray& input, HTShape shifts, HTAxes dims,
                      StreamIndex stream_id,
                      NDArray& output) {
  NDArray out = output.is_defined() ? output : NDArray::empty_like(input);
  Stream stream(input->device(), stream_id);
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(input->device().type(), __FUNCTION__,
                                  hetu::impl::Roll, input,
                                  shifts, dims,
                                  out, stream);
  return out;
}

NDArray NDArray::sin(const NDArray& input,
                     StreamIndex stream_id,
                     NDArray& output) {
  NDArray out = output.is_defined() ? output : NDArray::empty_like(input);
  Stream stream(input->device(), stream_id);
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(input->device().type(), __FUNCTION__,
                                  hetu::impl::Sin, input,
                                  out, stream);
  return out;
}

NDArray NDArray::slice(const NDArray& input, const HTShape& begin_pos,
                       const HTShape& output_shape, StreamIndex stream_id) {
  HT_ASSERT(!input->is_dynamic())
  << "Output definition doesn't support dynamic input in NDArray::slice.";

  const auto& in_shape = input->shape();
  const auto& in_stride = input->stride();
  HTShape out_shape = output_shape;
  
  HT_ASSERT(begin_pos.size() == in_shape.size()
         && output_shape.size() == in_shape.size());

  size_t ndim = in_shape.size();
  auto storage_offset = input->storage_offset();
  for (int64_t i = 0; i < ndim; i++) {
    int64_t start_val = begin_pos[i];
    int64_t end_val = begin_pos[i] + out_shape[i];
    if (start_val < 0) {
      start_val += in_shape[i];
    }
    if (end_val < 0) {
      end_val += in_shape[i];
    }
    if (start_val < 0) {
      start_val = 0;
    } else if (start_val >= in_shape[i]) {
      start_val = in_shape[i];
    }
    if (end_val < start_val) {
      end_val = start_val;
    } else if (end_val >= in_shape[i]) {
      end_val = in_shape[i];
    }
    storage_offset += start_val * in_stride[i];
    out_shape[i] = end_val - start_val;
  }

  if (input->numel() == 0) {
    storage_offset = 0;
  }
  
  auto output_meta = NDArrayMeta().set_dtype(input->dtype())
                                  .set_shape(out_shape)
                                  .set_stride(in_stride)
                                  .set_device(input->device());
  auto out = NDArray(output_meta, input->storage(), storage_offset);
  return out;
}

NDArray NDArray::softmax(const NDArray& input,
                         int64_t dim,
                         StreamIndex stream_id,
                         NDArray& output) {
  NDArray out = output.is_defined() ? output : NDArray::empty_like(input);
  Stream stream(input->device(), stream_id);
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(input->device().type(), __FUNCTION__,
                                  hetu::impl::Softmax, input,
                                  out, dim, stream);
  return out;
}

NDArray NDArray::sceloss(const NDArray& preds, const NDArray& labels,
                         ReductionType reduction,
                         StreamIndex stream_id,
                         NDArray& output) {
  NDArray out;
  HTShape output_shape = HTShape(preds->shape().begin(), preds->shape().end() - 1);
  if (output.is_defined()) 
    out = output;
  else {
    if (reduction != kNONE)
      out = NDArray::empty({1}, preds->device(), preds->dtype(), stream_id);
    else
      out = NDArray::empty(output_shape, preds->device(), preds->dtype(),
                           stream_id);
  }
  Stream stream(preds->device(), stream_id);
  NDArray unreduced = reduction == kNONE
    ? out
    : NDArray::empty(output_shape, preds->device(), preds->dtype(), stream_id);
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(preds->device().type(), __FUNCTION__,
                                  hetu::impl::SoftmaxCrossEntropy, preds,
                                  labels, unreduced, stream);
  if (reduction != kNONE) {
    NDArray::reduce(unreduced, reduction, HTAxes(), false, stream_id, out);
  }
  return out;
}

NDArray NDArray::sceloss(const NDArray& preds, const NDArray& labels, const int64_t ignored_index, 
                         ReductionType reduction,
                         StreamIndex stream_id,
                         NDArray& output) {
  NDArray out;
  HTShape output_shape = HTShape(preds->shape().begin(), preds->shape().end() - 1);
  if (output.is_defined()) 
    out = output;
  else {
    if (reduction != kNONE)
      out = NDArray::empty({1}, preds->device(), preds->dtype(), stream_id);
    else
      out = NDArray::empty(output_shape, preds->device(), preds->dtype(),
                           stream_id);
  }
  Stream stream(preds->device(), stream_id);
  NDArray unreduced = reduction == kNONE
    ? out
    : NDArray::empty(output_shape, preds->device(), preds->dtype(), stream_id);
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(preds->device().type(), __FUNCTION__,
                                  hetu::impl::SoftmaxCrossEntropySparse, preds,
                                  labels, unreduced, ignored_index, stream);
  if (reduction != kNONE) {
    NDArray::reduce(unreduced, reduction, HTAxes(), false, stream_id, out);
  }
  return out;
}            

NDArray NDArray::triu(const NDArray& input, bool lower, int64_t diagonal, 
                      StreamIndex stream_id,
                      NDArray& output) {
  NDArray out = output.is_defined() ? output : NDArray::empty_like(input);
  Stream stream(input->device(), stream_id);
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(input->device().type(), __FUNCTION__, hetu::impl::TriuTril,
                                  input, output, lower, diagonal, stream);
  return out;
}

NDArray NDArray::where(const NDArray& cond, const NDArray& x, const NDArray& y, 
                       StreamIndex stream_id,
                       NDArray& output) {
  NDArray out = output.is_defined() ? output : NDArray::empty_like(x);
  Stream stream(x->device(), stream_id);
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(x->device().type(), __FUNCTION__, 
                                  hetu::impl::Where, cond, x, y, out, stream);
  return out;
}

NDArray NDArray::maxpool(const NDArray& input, const size_t kernel_H,
                         const size_t kernel_W, const size_t padding,
                         const size_t stride,
                         StreamIndex stream_id,
                         NDArray& output) {  
  NDArray out;
  if (output.is_defined())
    out = output;
  else {
    int64_t N = input->shape(0);
    int64_t C = input->shape(1);
    int64_t H = input->shape(2);
    int64_t W = input->shape(3);
    int64_t p_H = (H + 2 * padding - kernel_H) / stride + 1;
    int64_t p_W = (W + 2 * padding - kernel_W) / stride + 1;
    out = NDArray::empty({N, C, p_H, p_W}, input->device(), input->dtype(),
                         stream_id);
  }
  Stream stream(input->device(), stream_id);
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(input->device().type(), __FUNCTION__, hetu::impl::MaxPool,
                                  input, kernel_H, kernel_W,
                                  out, padding, stride,
                                  stream);
  return out;
}

//_____________________________________________________________________________________

NDArray NDArray::cat(const NDArrayList& inputs, int axis,
                     StreamIndex stream_id,
                     NDArray& output) {
  auto parsed_axis = NDArrayMeta::ParseAxis(axis, inputs.at(0)->ndim());
  std::vector<HTShape> shapes;
  shapes.reserve(inputs.size());
  std::transform(inputs.begin(), inputs.end(), std::back_inserter(shapes),
                 [](const NDArray& x) { return x->shape(); });
  auto cat_shape = NDArrayMeta::Concat(shapes, 0);
  NDArray ret = output.is_defined()
    ? output
    : NDArray::empty(cat_shape, inputs.at(0)->device(), inputs.at(0)->dtype(),
                     stream_id);
  Stream stream(inputs.at(0)->device(), stream_id);
  HT_DISPATCH_KERNEL_CUDA_ONLY(ret->device().type(), __FUNCTION__, hetu::impl::Concat,
                               inputs, ret, parsed_axis, stream);
  return ret;
}

// deprecated: dynamic shape at inference when using different seq_len
NDArray NDArray::empty(const HTShape& shape, const Device& device,
                       DataType dtype, StreamIndex stream_id,
                       const HTShape& dynamic_shape) {
  auto meta = NDArrayMeta()
                .set_device(device)
                .set_dtype(dtype)
                .set_shape(shape)
                .set_dynamic_shape(dynamic_shape);
  auto storage = std::make_shared<NDArrayStorage>(AllocFromMemoryPool(
    device, (dtype == kFloat4 || dtype == kNFloat4) ?  
    ((meta.numel() + 1) / 2) * DataType2Size(dtype) : 
    meta.numel() * DataType2Size(dtype), Stream(device, stream_id)));
  // mempool debug use
  // HT_LOG_INFO << "NDArray empty with " << meta;
  // if (storage->is_new_malloc())
  //   HT_LOG_INFO << "malloc new empty NDArray with " << meta;
  return NDArray(meta, storage);
}

NDArray NDArray::empty_like(const NDArray& other, StreamIndex stream_id) {
  return NDArray::empty(other->shape(), other->device(), other->dtype(),
                        stream_id, other->dynamic_shape());
}

// deprecated: dynamic shape at inference when using different seq_len
NDArray NDArray::full(const HTShape& shape, double fill_value,
                      const Device& device, DataType dtype,
                      StreamIndex stream_id,
                      const HTShape& dynamic_shape) {
  NDArray out = NDArray::empty(shape, device, dtype, stream_id, dynamic_shape);
  return NDArray::full_(out, fill_value, stream_id);
}

// deprecated: dynamic shape at inference when using different seq_len
NDArray NDArray::full_like(const NDArray& other, double fill_value,
                           StreamIndex stream_id) {
  return NDArray::full(other->shape(), fill_value, other->device(),
                       other->dtype(), stream_id, other->dynamic_shape());
}

NDArray NDArray::full_(NDArray& data, double fill_value,
                       StreamIndex stream_id) {
  Stream stream(data->device(), stream_id);
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(data->device().type(), __FUNCTION__,
                                  hetu::impl::ArraySet, data, fill_value,
                                  stream);
  return data;
}

NDArray NDArray::copy(const NDArray& input, StreamIndex stream_id,
                      NDArray& output) {
  NDArray out = output.is_defined() ? output : NDArray::empty_like(input);
  if (input->numel() == 0) {
    return out;
  }
  Device out_device = input->device();
  if (out->device().is_cuda())
    out_device = out->device();
  Stream stream(out_device, stream_id);
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(out_device.type(), __FUNCTION__,
                                  hetu::impl::DataTransfer, input, out, stream);
  return out;
}

NDArray NDArray::contiguous(const NDArray& input, StreamIndex stream_id,
                            NDArray& output) {
  if (input->is_contiguous()) {
    if (output.is_defined()) {
      HT_ASSERT(input->shape() == output->shape())
        << "Input and output shape mismatch in NDArray::contiguous. "
        << "input shape: " << input->shape() << ", output shape: "
        << output->shape();
      HT_ASSERT(output->is_contiguous())
        << "Output should be contiguous in NDArray::contiguous.";
      NDArray::copy(input, stream_id, output);
      return output;
    }
    else {
      return input;
    }
  }
  NDArray out = output.is_defined() ? output : NDArray::empty_like(input);
  Stream stream(input->device(), stream_id);
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(input->device().type(), __FUNCTION__,
                                  hetu::impl::DataTransfer, input, out, stream);
  return out;
}

// deprecated: dynamic shape at inference when using different seq_len
NDArray NDArray::rand(const HTShape& shape, const Device& device,
                      DataType dtype, double lb, double ub, uint64_t seed,
                      StreamIndex stream_id,
                      const HTShape& dynamic_shape) {
  NDArray out = NDArray::empty(shape, device, dtype, stream_id, dynamic_shape);
  return NDArray::uniform_(out, lb, ub, seed, stream_id);
}

// deprecated: dynamic shape at inference when using different seq_len
NDArray NDArray::randn(const HTShape& shape, const Device& device,
                       DataType dtype, double mean, double stddev,
                       uint64_t seed, StreamIndex stream_id,
                       const HTShape& dynamic_shape) {
  NDArray out = NDArray::empty(shape, device, dtype, stream_id, dynamic_shape);
  return NDArray::normal_(out, mean, stddev, seed, stream_id);
}

NDArray NDArray::uniform_(NDArray& data, double lb, double ub, uint64_t seed,
                          StreamIndex stream_id) {
  Stream stream(data->device(), stream_id);
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(data->device().type(), __FUNCTION__,
                                  hetu::impl::UniformInits, data, lb, ub, seed,
                                  stream);
  return data;
}

NDArray NDArray::normal_(NDArray& data, double mean, double stddev,
                         uint64_t seed, StreamIndex stream_id) {
  Stream stream(data->device(), stream_id);
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(data->device().type(), __FUNCTION__,
                                  hetu::impl::NormalInits, data, mean, stddev,
                                  seed, stream);
  return data;
}

NDArray NDArray::truncated_normal_(NDArray& data, double mean, double stddev,
                                   double lb, double ub, uint64_t seed,
                                   StreamIndex stream_id) {
  Stream stream(data->device(), stream_id);
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(data->device().type(), __FUNCTION__,
                                  hetu::impl::TruncatedNormalInits, data, mean,
                                  stddev, lb, ub, seed, stream);
  return data;
}

} // namespace hetu