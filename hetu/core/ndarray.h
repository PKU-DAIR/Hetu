#pragma once

#include "hetu/common/except.h"
#include "hetu/utils/shared_ptr_wrapper.h"
#include "hetu/core/reduction_type.h"
#include "hetu/core/binary_type.h"
#include "hetu/core/ndarray_meta.h"
#include "hetu/core/ndarray_storage.h"
#include "hetu/core/stream.h"
#include <future>

namespace hetu {

class NDArrayDef;
class NDArray;
using NDArrayList = std::vector<NDArray>;

class NDArray : public shared_ptr_wrapper<NDArrayDef> {
 public:
  NDArray() = default;
  NDArray(const NDArrayMeta& meta,
          std::shared_ptr<NDArrayStorage> storage,
          int64_t storage_offset = 0)
  : shared_ptr_wrapper<NDArrayDef>() {
    _ptr = make_ptr<NDArrayDef>(meta, storage, storage_offset);
  }
  using shared_ptr_wrapper<NDArrayDef>::operator=;

 private:
  static NDArray EMPTY;

 public:
  static const StreamIndex DEFAULT_STREAM;

  static void MarkUsedBy(const NDArray& array, const Stream& stream);

  static void MarkUsedBy(const NDArrayList& arrays, const Stream& stream);

  static NDArray to(const NDArray& input,
                    const Device& device = Device(kUndeterminedDevice),
                    DataType = kUndeterminedDataType,
                    StreamIndex stream_id = DEFAULT_STREAM,
                    NDArray& output = EMPTY);

  static inline NDArray cuda(const NDArray& input, DeviceIndex dev_id = 0,
                             StreamIndex stream_id = DEFAULT_STREAM,
                             NDArray& output = EMPTY) {
    return NDArray::to(input, Device(kCUDA, dev_id), kUndeterminedDataType,
                       stream_id, output);
  }

  static inline NDArray cpu(const NDArray& input,
                            StreamIndex stream_id = DEFAULT_STREAM,
                            NDArray& output = EMPTY) {
    return NDArray::to(input, Device(kCPU), kUndeterminedDataType, stream_id,
                       output);
  }

  static inline NDArray toUInt8(const NDArray& input,
                                StreamIndex stream_id = DEFAULT_STREAM,
                                NDArray& output = EMPTY) {
    return NDArray::to(input, Device(kUndeterminedDevice), kUInt8, stream_id,
                       output);
  }

  static inline NDArray toInt8(const NDArray& input,
                               StreamIndex stream_id = DEFAULT_STREAM,
                               NDArray& output = EMPTY) {
    return NDArray::to(input, Device(kUndeterminedDevice), kInt8, stream_id,
                       output);
  }

  static inline NDArray toInt16(const NDArray& input,
                                StreamIndex stream_id = DEFAULT_STREAM,
                                NDArray& output = EMPTY) {
    return NDArray::to(input, Device(kUndeterminedDevice), kInt16, stream_id,
                       output);
  }

  static inline NDArray toInt32(const NDArray& input,
                                StreamIndex stream_id = DEFAULT_STREAM,
                                NDArray& output = EMPTY) {
    return NDArray::to(input, Device(kUndeterminedDevice), kInt32, stream_id,
                       output);
  }

  static inline NDArray toInt64(const NDArray& input,
                                StreamIndex stream_id = DEFAULT_STREAM,
                                NDArray& output = EMPTY) {
    return NDArray::to(input, Device(kUndeterminedDevice), kInt64, stream_id,
                       output);
  }

  static inline NDArray toFloat32(const NDArray& input,
                                  StreamIndex stream_id = DEFAULT_STREAM,
                                  NDArray& output = EMPTY) {
    return NDArray::to(input, Device(kUndeterminedDevice), kFloat32, stream_id,
                       output);
  }

  static inline NDArray toFloat64(const NDArray& input,
                                  StreamIndex stream_id = DEFAULT_STREAM,
                                  NDArray& output = EMPTY) {
    return NDArray::to(input, Device(kUndeterminedDevice), kFloat64, stream_id,
                       output);
  }

  static NDArray abs(const NDArray& input,
                     StreamIndex stream_id = DEFAULT_STREAM,
                     NDArray& output = EMPTY);

  static NDArray add(const NDArray& x, const NDArray& y,
                     StreamIndex stream_id = DEFAULT_STREAM,
                     NDArray& output = EMPTY);

  static NDArray add(const NDArray& input, double scalar,
                     StreamIndex stream_id = DEFAULT_STREAM,
                     NDArray& output = EMPTY);

  static NDArray sub(const NDArray& x, const NDArray& y,
                     StreamIndex stream_id = DEFAULT_STREAM,
                     NDArray& output = EMPTY);

  static NDArray sub(const NDArray& input, double scalar,
                     StreamIndex stream_id = DEFAULT_STREAM,
                     NDArray& output = EMPTY);

  static NDArray sub(double scalar, const NDArray& input,
                     StreamIndex stream_id = DEFAULT_STREAM,
                     NDArray& output = EMPTY);

  static NDArray neg(const NDArray& input,
                     StreamIndex stream_id = DEFAULT_STREAM,
                     NDArray& output = EMPTY);

  static NDArray mul(const NDArray& x, const NDArray& y,
                     StreamIndex stream_id = DEFAULT_STREAM,
                     NDArray& output = EMPTY);

  static NDArray mul(const NDArray& input, double scalar,
                     StreamIndex stream_id = DEFAULT_STREAM,
                     NDArray& output = EMPTY);

  static NDArray div(const NDArray& x, const NDArray& y,
                     StreamIndex stream_id = DEFAULT_STREAM,
                     NDArray& output = EMPTY);

  static NDArray div(const NDArray& input, double scalar,
                     StreamIndex stream_id = DEFAULT_STREAM,
                     NDArray& output = EMPTY);

  static NDArray div(double scalar, const NDArray& input,
                     StreamIndex stream_id = DEFAULT_STREAM,
                     NDArray& output = EMPTY);

  static NDArray pow(const NDArray& input, double exponent = 2.0,
                     StreamIndex stream_id = DEFAULT_STREAM,
                     NDArray& output = EMPTY);

  static NDArray sqrt(const NDArray& input,
                      StreamIndex stream_id = DEFAULT_STREAM,
                      NDArray& output = EMPTY);

  static NDArray reciprocal(const NDArray& input,
                            StreamIndex stream_id = DEFAULT_STREAM,
                            NDArray& output = EMPTY);

  // activations
  static NDArray sigmoid(const NDArray& input,
                         StreamIndex stream_id = DEFAULT_STREAM,
                         NDArray& output = EMPTY);

  static NDArray relu(const NDArray& input,
                      StreamIndex stream_id = DEFAULT_STREAM,
                      NDArray& output = EMPTY);

  static NDArray gelu(const NDArray& input,
                      StreamIndex stream_id = DEFAULT_STREAM,
                      NDArray& output = EMPTY);

  static NDArray swiglu(const NDArray& input,
                        StreamIndex stream_id = DEFAULT_STREAM,
                        NDArray& output = EMPTY);

  static NDArray tanh(const NDArray& input,
                      StreamIndex stream_id = DEFAULT_STREAM,
                      NDArray& output = EMPTY);
  
  static NDArray elu(const NDArray& input, double alpha, double scale,
                     StreamIndex stream_id = DEFAULT_STREAM,
                     NDArray& output = EMPTY);
  
  static NDArray hardshrink(const NDArray& input, double lambda,
                            StreamIndex stream_id = DEFAULT_STREAM,
                            NDArray& output = EMPTY);

  static NDArray hardsigmoid(const NDArray& input,
                            StreamIndex stream_id = DEFAULT_STREAM,
                            NDArray& output = EMPTY);

  static NDArray hardtanh(const NDArray& input, double min_val, double max_val,
                          StreamIndex stream_id = DEFAULT_STREAM,
                          NDArray& output = EMPTY);

  static NDArray hardswish(const NDArray& input,
                           StreamIndex stream_id = DEFAULT_STREAM,
                           NDArray& output = EMPTY);

  static NDArray logsigmoid(const NDArray& input, bool inverse = false, 
                            StreamIndex stream_id = DEFAULT_STREAM,
                            NDArray& output = EMPTY);

  static NDArray silu(const NDArray& input,
                      StreamIndex stream_id = DEFAULT_STREAM,
                      NDArray& output = EMPTY);

  static NDArray mish(const NDArray& input,
                      StreamIndex stream_id = DEFAULT_STREAM,
                      NDArray& output = EMPTY);

  static NDArray softplus(const NDArray& input, double beta, double threshold,
                          StreamIndex stream_id = DEFAULT_STREAM,
                          NDArray& output = EMPTY);

  static NDArray softshrink(const NDArray& input, double lambda,
                            StreamIndex stream_id = DEFAULT_STREAM,
                            NDArray& output = EMPTY);                           

  static NDArray exp(const NDArray& input,
                     StreamIndex stream_id = DEFAULT_STREAM,
                     NDArray& output = EMPTY);

  static NDArray log(const NDArray& input,
                     StreamIndex stream_id = DEFAULT_STREAM,
                     NDArray& output = EMPTY);

  static NDArray ceil(const NDArray& input,
                      StreamIndex stream_id = DEFAULT_STREAM,
                      NDArray& output = EMPTY);

  static NDArray floor(const NDArray& input,
                       StreamIndex stream_id = DEFAULT_STREAM,
                       NDArray& output = EMPTY);

  static NDArray round(const NDArray& input,
                       StreamIndex stream_id = DEFAULT_STREAM,
                       NDArray& output = EMPTY);

  static NDArray reduce(const NDArray& input, ReductionType red_type,
                        const HTAxes& axes = HTAxes(), bool keepdims = false,
                        StreamIndex stream_id = DEFAULT_STREAM,
                        NDArray& output = EMPTY);

  static NDArray sum(const NDArray& input, const HTAxes& axes = HTAxes(),
                     bool keepdims = false,
                     StreamIndex stream_id = DEFAULT_STREAM,
                     NDArray& output = EMPTY) {
    return NDArray::reduce(input, kSUM, axes, keepdims, stream_id, output);
  }

  static NDArray mean(const NDArray& input, const HTAxes& axes = HTAxes(),
                      bool keepdims = false,
                      StreamIndex stream_id = DEFAULT_STREAM,
                      NDArray& output = EMPTY) {
    return NDArray::reduce(input, kMEAN, axes, keepdims, stream_id, output);
  }

  static NDArray prod(const NDArray& input, const HTAxes& axes = HTAxes(),
                      bool keepdims = false,
                      StreamIndex stream_id = DEFAULT_STREAM,
                      NDArray& output = EMPTY) {
    return NDArray::reduce(input, kPROD, axes, keepdims, stream_id, output);
  }

  static NDArray max(const NDArray& input, const HTAxes& axes = HTAxes(),
                     bool keepdims = false,
                     StreamIndex stream_id = DEFAULT_STREAM,
                     NDArray& output = EMPTY) {
    return NDArray::reduce(input, kMAX, axes, keepdims, stream_id, output);
  }

  static NDArray min(const NDArray& input, const HTAxes& axes = HTAxes(),
                     bool keepdims = false,
                     StreamIndex stream_id = DEFAULT_STREAM,
                     NDArray& output = EMPTY) {
    return NDArray::reduce(input, kMIN, axes, keepdims, stream_id, output);
  }

  static NDArray matmul(const NDArray& x, const NDArray& y,
                        bool trans_left = false, bool trans_right = false,
                        StreamIndex stream_id = DEFAULT_STREAM,
                        NDArray& output = EMPTY);

  static NDArray matmul4bit(const NDArray& x, const NDArray& y, 
                            const NDArray& absmax, const NDArray& datatype,
                            bool trans_left = false, bool trans_right = false,
                            int blocksize = 4096,
                            StreamIndex stream_id = DEFAULT_STREAM,
                            NDArray& output = EMPTY);

  static NDArray bmm(const NDArray& x, const NDArray& y,
                     bool trans_left = false, bool trans_right = false,
                     StreamIndex stream_id = DEFAULT_STREAM,
                     NDArray& output = EMPTY);

  static NDArray dot(const NDArray& x, const NDArray& y,
                     StreamIndex stream_id = DEFAULT_STREAM,
                     NDArray& output = EMPTY);

  static NDArray index_add(const NDArray& x, const NDArray& ids,
                           const NDArray& y, int64_t dim = 0,
                           StreamIndex stream_id = DEFAULT_STREAM,
                           NDArray& output = EMPTY);

  static NDArray reshape(const NDArray& input, const HTShape& new_shape,
                         StreamIndex stream_id = DEFAULT_STREAM, NDArray& output = EMPTY);

  static NDArray view(const NDArray& input, const HTShape& view_shape);

  static NDArray squeeze(const NDArray& input);

  static NDArray squeeze(const NDArray& input, int64_t dim);

  static NDArray unsqueeze(const NDArray& input, int64_t dim);

  static NDArray flatten(const NDArray& input, int64_t start_dim = 0,
                         int64_t end_dim = -1);

  static NDArray permute(const NDArray& input, const HTAxes& dims,
                         StreamIndex stream_id = DEFAULT_STREAM);

  static NDArray movedim(const NDArray& input, int64_t src, int64_t dst,
                         StreamIndex stream_id = DEFAULT_STREAM);

  static NDArray adddim(const NDArray& input, int64_t dim, int64_t size,
                        StreamIndex stream_id = DEFAULT_STREAM,
                        NDArray& output = EMPTY);

  static NDArray diagonal(const NDArray& input, int64_t dim1, int64_t dim2,
                          int64_t offset, StreamIndex stream_id = DEFAULT_STREAM);

  static NDArray diagonal_grad(const NDArray& input, int64_t dim1, int64_t dim2,
                               StreamIndex stream_id = DEFAULT_STREAM,
                               NDArray& output = EMPTY);
  
  static NDArray as_strided(const NDArray& input, const HTShape& outshape, const HTStride& stride,
                            int64_t storage_offset = 0, StreamIndex stream_id = DEFAULT_STREAM);

  static NDArrayList split(const NDArray& input, size_t num_chunks,
                           int64_t axis = 0,
                           StreamIndex stream_id = DEFAULT_STREAM);

  static NDArrayList split(const NDArray& input, const HTShape& chunks,
                           int64_t axis = 0,
                           StreamIndex stream_id = DEFAULT_STREAM);

//____________________________________________________________________________________________

  static NDArray arrayset(NDArray& input, double value,
                          StreamIndex stream_id = DEFAULT_STREAM);


  static NDArray avgpool(const NDArray& input, const size_t kernel_H,
                         const size_t kernel_W, const size_t padding,
                         const size_t stride,
                         StreamIndex stream_id = DEFAULT_STREAM,
                         NDArray& output = EMPTY);

  static NDArrayList batchnorm(const NDArray& input, const NDArray& bn_scale, const NDArray& bn_bias,
                               NDArray& running_mean, NDArray& running_var,
                               double momentum = 0.1, double eps = 1e-5,
                               StreamIndex stream_id = DEFAULT_STREAM,
                               NDArray& output = EMPTY,
                               NDArray& save_mean = EMPTY,
                               NDArray& save_var = EMPTY);
   
  static NDArray broadcast(const NDArray& input, const HTShape& shape,
                           StreamIndex stream_id = DEFAULT_STREAM,
                           NDArray& output = EMPTY);

  static NDArray broadcast(const NDArray& input, const HTShape& shape,
                           const HTAxes& add_axes,
                           StreamIndex stream_id = DEFAULT_STREAM,
                           NDArray& output = EMPTY);

  static NDArray conv2d(const NDArray& input, const NDArray& filter, 
                        const HTShape& padding, const HTShape& stride,
                        StreamIndex stream_id = DEFAULT_STREAM,
                        NDArray& output = EMPTY);

  static NDArray cos(const NDArray& input,
                     StreamIndex stream_id = DEFAULT_STREAM,
                     NDArray& output = EMPTY);

  static NDArray dequantization(const NDArray& input, NDArray& absmax, DataType dqtype,
                                int64_t blocksize, StreamIndex stream_id = DEFAULT_STREAM,
                                const NDArray& code = EMPTY,
                                NDArray& output = EMPTY);

  static NDArray embedding(const NDArray& input, const NDArray& id,
                           StreamIndex stream_id = DEFAULT_STREAM,
                           NDArray& output = EMPTY);

  static NDArrayList fused_layernorm(const NDArray& input, const NDArray& ln_scale, const NDArray& ln_bias, 
                                     const HTShape& normalized_shape, double eps = 0.01,
                                     StreamIndex stream_id = DEFAULT_STREAM,
                                     NDArray& output = EMPTY,
                                     NDArray& save_mean = EMPTY,
                                     NDArray& save_var = EMPTY);

  static NDArrayList fused_rmsnorm(const NDArray& input, const NDArray& ln_scale, 
                                   const HTShape& normalized_shape, double eps = 0.01,
                                   StreamIndex stream_id = DEFAULT_STREAM,
                                   NDArray& output = EMPTY,
                                   NDArray& save_var = EMPTY);

  static NDArray gather(const NDArray& input, const NDArray& id, int64_t dim,
                        StreamIndex stream_id = DEFAULT_STREAM,
                        NDArray& output = EMPTY);

  static NDArrayList instancenorm(const NDArray& input, double eps = 1e-7,
                                  StreamIndex stream_id = DEFAULT_STREAM,
                                  NDArray& output = EMPTY,
                                  NDArray& save_mean = EMPTY,
                                  NDArray& save_var = EMPTY);

  static NDArray kldiv(const NDArray& preds, const NDArray& labels,
                       ReductionType reduction = kMEAN,
                       StreamIndex stream_id = DEFAULT_STREAM,
                       NDArray& output = EMPTY);

  static NDArrayList layernorm(const NDArray& input, const NDArray& bn_scale, const NDArray& bn_bias, 
                               const HTShape& normalized_shape, double eps = 0.01,
                               StreamIndex stream_id = DEFAULT_STREAM,
                               NDArray& output = EMPTY,
                               NDArray& save_mean = EMPTY,
                               NDArray& save_var = EMPTY);

  static NDArray leakyrelu(const NDArray& input, double alpha,
                           StreamIndex stream_id = DEFAULT_STREAM,
                           NDArray& output = EMPTY);

  static NDArray linear(const NDArray& a, const NDArray& b, const NDArray& bias, 
                        bool trans_a = false, bool trans_b = true,
                        StreamIndex stream_id = DEFAULT_STREAM,
                        NDArray& output = EMPTY);

  static NDArray maxpool(const NDArray& input, const size_t kernel_H,
                         const size_t kernel_W, const size_t padding,
                         const size_t stride,
                         StreamIndex stream_id = DEFAULT_STREAM,
                         NDArray& output = EMPTY);

  static NDArray mseloss(const NDArray& preds, const NDArray& labels,
                         ReductionType reduction = kMEAN,
                         StreamIndex stream_id = DEFAULT_STREAM,
                         NDArray& output = EMPTY);

  static NDArray nllloss(const NDArray& preds, const NDArray& labels,
                         ReductionType reduction = kMEAN,
                         StreamIndex stream_id = DEFAULT_STREAM,
                         NDArray& output = EMPTY);

  static NDArray norm(const NDArray& input, int64_t p = 1, int64_t dim = 0, 
                      bool keepdim = false,
                      StreamIndex stream_id = DEFAULT_STREAM,
                      NDArray& output = EMPTY);

  static NDArray onehot(const NDArray& input, size_t num_classes,
                        StreamIndex stream_id = DEFAULT_STREAM,
                        NDArray& output = EMPTY);

  static NDArray pad(const NDArray& input, const HTShape& paddings, 
                     std::string mode, double constant,
                     StreamIndex stream_id = DEFAULT_STREAM,
                     NDArray& output = EMPTY);

  static NDArrayList quantization(const NDArray& input,
                                  DataType qtype, 
                                  int64_t blocksize,
                                  bool stochastic = false,
                                  StreamIndex stream_id = DEFAULT_STREAM,
                                  const NDArray& code = EMPTY,
                                  NDArray& absmax = EMPTY,
                                  NDArray& output = EMPTY);

  static NDArray repeat(const NDArray& input, HTShape repeats,
                        StreamIndex stream_id = DEFAULT_STREAM,
                        NDArray& output = EMPTY);

  static NDArray roll(const NDArray& input, HTShape shifts, HTAxes dims,
                      StreamIndex stream_id = DEFAULT_STREAM,
                      NDArray& output = EMPTY);

  static NDArray sin(const NDArray& input,
                     StreamIndex stream_id = DEFAULT_STREAM,
                     NDArray& output = EMPTY);

  static NDArray slice(const NDArray& input, const HTShape& begin_pos,
                       const HTShape& output_shape,
                       StreamIndex stream_id = DEFAULT_STREAM);

  static NDArray softmax(const NDArray& input, int64_t dim,
                         StreamIndex stream_id = DEFAULT_STREAM,
                         NDArray& output = EMPTY);

  static NDArray sceloss(const NDArray& preds, const NDArray& labels,
                         ReductionType reduction = kMEAN,
                         StreamIndex stream_id = DEFAULT_STREAM,
                         NDArray& output = EMPTY);     

  static NDArray sceloss(const NDArray& preds, const NDArray& labels, const int64_t ignored_index = -1, 
                         ReductionType reduction = kMEAN,
                         StreamIndex stream_id = DEFAULT_STREAM,
                         NDArray& output = EMPTY);                     

  static NDArray triu(const NDArray& input, bool lower = false, int64_t diagonal = 0, 
                      StreamIndex stream_id = DEFAULT_STREAM,
                      NDArray& output = EMPTY);   

  static NDArray where(const NDArray& cond, const NDArray& x, const NDArray& y, 
                       StreamIndex stream_id = DEFAULT_STREAM,
                       NDArray& output = EMPTY); 
//____________________________________________________________________________________________

  static NDArray cat(const NDArrayList& inputs, int axis = 0,
                     StreamIndex stream_id = DEFAULT_STREAM,
                     NDArray& output = EMPTY);

  static inline NDArray zeros(const HTShape& shape,
                              const Device& device = Device(kCPU),
                              DataType dtype = kFloat32,
                              StreamIndex stream_id = DEFAULT_STREAM,
                              const HTShape& dynamic_shape = {}) {
    return NDArray::full(shape, 0, device, dtype, stream_id, dynamic_shape);
  }

  static inline NDArray zeros_like(const NDArray& other,
                                   StreamIndex stream_id = DEFAULT_STREAM) {
    return NDArray::full_like(other, 0, stream_id);
  }

  static inline NDArray zeros_(NDArray& data,
                               StreamIndex stream_id = DEFAULT_STREAM) {
    return NDArray::full_(data, 0, stream_id);
  }

  static inline NDArray ones(const HTShape& shape,
                             const Device& device = Device(kCPU),
                             DataType dtype = kFloat32,
                             StreamIndex stream_id = DEFAULT_STREAM,
                             const HTShape& dynamic_shape = {}) {
    return NDArray::full(shape, 1, device, dtype, stream_id, dynamic_shape);
  }

  static inline NDArray ones_like(const NDArray& other,
                                  StreamIndex stream_id = DEFAULT_STREAM) {
    return NDArray::full_like(other, 1, stream_id);
  }

  static inline NDArray ones_(NDArray& data,
                              StreamIndex stream_id = DEFAULT_STREAM) {
    return NDArray::full_(data, 1, stream_id);
  }

  static NDArray empty(const HTShape& shape,
                       const Device& device = Device(kCPU),
                       DataType dtype = kFloat32,
                       StreamIndex stream_id = DEFAULT_STREAM,
                       const HTShape& dynamic_shape = {},
                       bool shared_memory = false);

  static NDArray empty_like(const NDArray& other,
                            StreamIndex stream_id = DEFAULT_STREAM);

  static NDArray full(const HTShape& shape, double fill_value,
                      const Device& device = Device(kCPU),
                      DataType dtype = kFloat32,
                      StreamIndex stream_id = DEFAULT_STREAM,
                      const HTShape& dynamic_shape = {});

  static NDArray full_like(const NDArray& other, double fill_value,
                           StreamIndex stream_id = DEFAULT_STREAM);

  static NDArray full_(NDArray& data, double fill_value,
                       StreamIndex stream_id = DEFAULT_STREAM);

  static NDArray copy(const NDArray& input,
                      StreamIndex stream_id = DEFAULT_STREAM,
                      NDArray& output = EMPTY);

  static NDArray copy_multi_slices(const NDArray& input, const std::vector<std::pair<int32_t, int32_t>> indices,
                                   int64_t axis = 0, StreamIndex stream_id = DEFAULT_STREAM,
                                   NDArray& output = EMPTY);

  static NDArray revert_copy_multi_slices(const NDArray& input, const std::vector<std::pair<int32_t, int32_t>> indices,
                                          int64_t axis = 0, StreamIndex stream_id = DEFAULT_STREAM, 
                                          NDArray& output = EMPTY);

  static NDArray contiguous(const NDArray& input,
                            StreamIndex stream_id = DEFAULT_STREAM,
                            NDArray& output = EMPTY);

  static NDArray rand(const HTShape& shape, const Device& device = Device(kCPU),
                      DataType dtype = kFloat32, double lb = 0.0,
                      double ub = 1.0, uint64_t seed = 0,
                      StreamIndex stream_id = DEFAULT_STREAM,
                      const HTShape& dynamic_shape = {});

  static NDArray randn(const HTShape& shape,
                       const Device& device = Device(kCPU),
                       DataType dtype = kFloat32, double mean = 0.0,
                       double stddev = 1.0, uint64_t seed = 0,
                       StreamIndex stream_id = DEFAULT_STREAM,
                       const HTShape& dynamic_shape = {});

  static NDArray uniform_(NDArray& data, double lb = 0.0, double ub = 1.0,
                          uint64_t seed = 0,
                          StreamIndex stream_id = DEFAULT_STREAM);

  static NDArray normal_(NDArray& data, double mean = 0.0, double stddev = 1.0,
                         uint64_t seed = 0,
                         StreamIndex stream_id = DEFAULT_STREAM);

  static NDArray truncated_normal_(NDArray& data, double mean = 0.0,
                                   double stddev = 1.0, double lb = 0.0,
                                   double ub = 1.0, uint64_t seed = 0,
                                   StreamIndex stream_id = DEFAULT_STREAM);
};

inline NDArray operator+(const NDArray& x, const NDArray& y) {
  return NDArray::add(x, y);
}
inline NDArray operator+(const NDArray& input, double scalar) {
  return NDArray::add(input, scalar);
}
inline NDArray operator+(double scalar, const NDArray& input) {
  return NDArray::add(input, scalar);
}
inline NDArray operator-(const NDArray& x, const NDArray& y) {
  return NDArray::sub(x, y);
}
inline NDArray operator-(const NDArray& input, double scalar) {
  return NDArray::sub(input, scalar);
}
inline NDArray operator-(double scalar, const NDArray& input) {
  return NDArray::sub(scalar, input);
}
inline NDArray operator*(const NDArray& x, const NDArray& y) {
  return NDArray::mul(x, y);
}
inline NDArray operator*(const NDArray& input, double scalar) {
  return NDArray::mul(input, scalar);
}
inline NDArray operator*(double scalar, const NDArray& input) {
  return NDArray::mul(input, scalar);
}
inline NDArray operator/(const NDArray& x, const NDArray& y) {
  return NDArray::div(x, y);
}
inline NDArray operator/(const NDArray& input, double scalar) {
  return NDArray::div(input, scalar);
}
inline NDArray operator/(double scalar, const NDArray& input) {
  return NDArray::div(scalar, input);
}

class NDArrayDef : public shared_ptr_target {
 public:
  NDArrayDef(const NDArrayMeta& meta,
             std::shared_ptr<NDArrayStorage> storage,
             int64_t storage_offset = 0)
  : _meta(meta), _storage(storage) {
    HT_ASSERT(_meta.dtype != kUndeterminedDataType &&
              _meta.device != kUndeterminedDevice)
      << "Invalid meta: " << _meta;
    HT_VALUE_ERROR_IF(_storage == nullptr) << "Storage is not provided";
    size_t bytes_per_value = DataType2Size(meta.dtype);
    HT_ASSERT_GE(storage_offset, 0)
        << "Storage offset should be greater or equal to 0";
    HT_ASSERT_GE(_storage->size(), min_storage_size(storage_offset))
      << "Storage size is not sufficient";
    _storage_offset = storage_offset;
  }

  NDArrayDef(NDArray& other)
  : NDArrayDef(other->_meta, other->_storage, other->_storage_offset) {}

  // disable copy constructor and move constructor
  NDArrayDef(const NDArrayDef& other) = delete;
  NDArrayDef& operator=(const NDArrayDef& other) = delete;
  NDArrayDef(NDArrayDef&& other) = delete;
  NDArrayDef& operator=(NDArrayDef&& other) = delete;

  inline void* raw_data_ptr() {
    return const_cast<void*>(
      const_cast<const NDArrayDef*>(this)->raw_data_ptr());
  }

  inline const void* raw_data_ptr() const {
    HT_ASSERT_NE(_storage, nullptr) << "Storage is not initialized";
    auto* ptr = static_cast<uint8_t*>(_storage->mutable_data());
    ptr += (dtype() == kFloat4 || dtype() == kNFloat4) 
           ? ((_storage_offset + 1) / 2) * DataType2Size(dtype())
           : _storage_offset * DataType2Size(dtype());
    return static_cast<void*>(ptr);
  }

  template <typename T>
  inline T* data_ptr() {
    return static_cast<T*>(raw_data_ptr());
  }

  template <typename T>
  inline const T* data_ptr() const {
    return static_cast<const T*>(raw_data_ptr());
  }

  inline int64_t shm_offset() {
    HT_ASSERT(_storage->data_ptr().shared_memory)
      << "Only share memory has shmoffset.";
    return _storage->data_ptr().shmoffset;
  }

  template <typename T>
  inline T item() {
    HT_VALUE_ERROR_IF(numel() != 1)
      << "Cannot convert data with shape " << shape() << " to a scalar value";
    return *(data_ptr<T>());
  }

  const NDArrayMeta& meta() const {
    return _meta;
  }

  size_t ndim() const {
    return _meta.ndim();
  }

  size_t numel() const {
    return _meta.numel();
  }

  size_t dynamic_numel() const {
    return _meta.dynamic_numel();
  }

  DataType dtype() const {
    return _meta.dtype;
  }

  Device device() const {
    return _meta.device;
  }

  bool is_cpu() const {
    return _meta.device.is_cpu();
  }

  bool is_cuda() const {
    return _meta.device.is_cuda();
  }

  bool is_new_malloc() const {
    return _storage->is_new_malloc();
  }

  bool can_quantization() const {
    return _meta.dtype == kFloat32 ||
           _meta.dtype == kFloat16 ||
           _meta.dtype == kBFloat16;
  }

  bool is_quantization() const {
    return _meta.dtype == kInt8 ||
           _meta.dtype == kFloat4 ||
           _meta.dtype == kNFloat4;
  }

  bool is_equal(const NDArray& other) const {
    return _meta == other->meta() &&
           _storage == other->storage() &&
           _storage_offset == other->storage_offset();
  }

  const HTShape& shape() const {
    return _meta.shape;
  }

  int64_t shape(int64_t axis) const {
    int64_t ndim_ = ndim();
    HT_ASSERT(axis >= -ndim_ && axis < ndim_)
      << "shape axis should in range [" << -ndim_ << ", "
      << ndim_ << "), but it's " << axis;
    axis = axis < 0 ? axis + ndim_ : axis;
    return _meta.shape[axis];
  }

  const HTStride& stride() const {
    return _meta.stride;
  }

  int64_t stride(int64_t axis) const {
    int64_t ndim_ = ndim();
    HT_ASSERT(axis >= -ndim_ && axis < ndim_)
      << "stride axis should in range [" << -ndim_ << ", "
      << ndim_ << "), but it's " << axis;
    axis = axis < 0 ? axis + ndim_ : axis;
    return _meta.stride[axis];
  }

  bool is_contiguous() const {
    if (ndim() < 1 || numel() <= 1) { return true; }
    int64_t ndim_ = ndim();
    int64_t contiguous_stride = 1;
    for (int64_t i = ndim_ - 1; i >= 0; i--) {
      if (stride(i) != contiguous_stride)
        return false;
      contiguous_stride *= shape(i);
    }
    return true;
  }

  bool is_dynamic() const {
    return !_meta.dynamic_shape.empty();
  }

  const HTShape& dynamic_shape() const {
    // dynamic_shape is empty means shape = dynamic_shape
    if (_meta.dynamic_shape.empty())
      return _meta.shape;
    return _meta.dynamic_shape;
  }

  int64_t dynamic_shape(size_t axis) const {
    if (_meta.dynamic_shape.empty())
      return _meta.shape[axis];
    return _meta.dynamic_shape[axis];
  }

  const std::shared_ptr<NDArrayStorage>& storage() const {
    return _storage;
  }

  std::shared_ptr<NDArrayStorage>& storage() {
    return _storage;
  }

  size_t storage_size() const {
    return _storage->size();
  }

  size_t min_storage_size(int64_t storage_offset) const {
    // Question: it's used for non-contiguous cases
    // why not use the following code?
    /*
    int64_t storage_size = storage_offset + 1;
    int64_t dim = ndim();
    for (int64_t i = 0; i < dim; i++) {
      const auto& size_i = shape(i);
      if (size_i == 0) {
        return (dtype() == kFloat4 || dtype() == kNFloat4) ?  
               ((storage_offset + 1) / 2) * DataType2Size(dtype()) : 
               storage_offset * DataType2Size(dtype());
      }
      storage_size += (size_i - 1) * stride(i);
    }
    return (dtype() == kFloat4 || dtype() == kNFloat4) ?  
           ((storage_size + 1) / 2) * DataType2Size(dtype()) : 
           storage_size * DataType2Size(dtype());
    */
    // return (storage_offset + numel()) * DataType2Size(dtype());
    return (dtype() == kFloat4 || dtype() == kNFloat4) ?  
           ((storage_offset + numel() + 1) / 2) * DataType2Size(dtype()) : 
           (storage_offset + numel()) * DataType2Size(dtype());
  }

  int64_t storage_offset() const {
    return _storage_offset;
  }

  void set_meta(const NDArrayMeta& meta) {
    _meta = meta;
  }

  std::future<void> wait_async() const;

  void wait() const;

 protected:
  friend class NDArray;
  void Serialize(std::ostream& os, size_t n_print = 10) const;
  friend std::ostream& operator<<(std::ostream&, const NDArray&);

  NDArrayMeta _meta;
  std::shared_ptr<NDArrayStorage> _storage;
  int64_t _storage_offset;
};

std::ostream& operator<<(std::ostream&, const NDArray&);

inline bool IsCopiable(const NDArray& x, const NDArray& y) {
  return IsCopiable(x->meta(), y->meta());
}

inline bool IsExchangable(const NDArray& x, const NDArray& y) {
  return IsExchangable(x->meta(), y->meta());
}

inline bool IsBroadcastable(const NDArray& x, const NDArray& y) {
  return IsBroadcastable(x->meta(), y->meta());
}

inline bool IsConcatable(const NDArray& x, const NDArray& y, const NDArray& z,
                         int64_t axis) {
  return IsConcatable(x->meta(), y->meta(), z->meta(), axis);
}

inline bool IsConcatable(const NDArray& x, const NDArray& y, int64_t axis) {
  return IsConcatable(x->meta(), y->meta(), axis);
}

} // namespace hetu
