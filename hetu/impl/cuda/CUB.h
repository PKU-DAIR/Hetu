#include <cub/cub.cuh>
#include "hetu/impl/stream/CUDAStream.h"
#include "hetu/impl/utils/cuda_utils.h"



namespace hetu {
namespace impl {


// #define CUB_WRAPPER(func, ...) do {                                   \
//   size_t temp_storage_bytes = 0;                                                   \
//   func(nullptr, temp_storage_bytes, __VA_ARGS__);                                   \
//   std::cout << "temp_storage_bytes: " << temp_storage_bytes << std::endl;            \
//                                                                                    \
//   void* temp_storage_ptr = nullptr;                                                \
//   cudaError_t alloc_err = cudaMalloc(&temp_storage_ptr, temp_storage_bytes);        \
//   if (alloc_err != cudaSuccess) {                                                  \
//       std::cerr << "CUDA malloc failed: " << cudaGetErrorString(alloc_err) << std::endl; \
//       return;                                                                       \
//   }                                                                                \
//   std::cout << "alloc successfully" << std::endl;                                   \
//   func(temp_storage_ptr, temp_storage_bytes, __VA_ARGS__);                          \
//                                                                                    \
//   cudaError_t err = cudaGetLastError();                                            \
//   if (err != cudaSuccess) {                                                         \
//       std::cerr << "CUDA error after calling CUB function: " << cudaGetErrorString(err) << std::endl; \
//       cudaDeviceSynchronize();                                                      \
//   }                                                                                \
//   std::cout << "after ";                                                             \
//   cudaFree(temp_storage_ptr);                                                      \
// } while (false)


constexpr int max_cub_size = std::numeric_limits<int>::max() / 2 + 1; // 2**30

namespace {
template <typename scalar_t>
struct SumOp {
  __device__ scalar_t operator () (scalar_t a, scalar_t b) const {
    return a + b;
  }
};
}

template<typename InputIteratorT1, typename InputIteratorT2, typename OutputIteratorT, class ScanOpT>
__global__ void transform_vals(InputIteratorT1 a, InputIteratorT2 b, OutputIteratorT out, ScanOpT scan_op){
  // NOTE: out here not the final scan output, but an intermediate of the accumulation type.
  using acc_t = typename std::iterator_traits<OutputIteratorT>::value_type;
  *out = scan_op(static_cast<acc_t>(*a), static_cast<acc_t>(*b));
}

template<typename ValueT, typename InputIteratorT>
struct chained_iterator {
  using iterator_category = std::random_access_iterator_tag;
  using difference_type   = std::ptrdiff_t;
  using value_type        = ValueT;
  using pointer           = ValueT*;
  using reference         = ValueT&;

  InputIteratorT iter;
  ValueT *first;
  difference_type offset = 0;

  __device__ ValueT operator[](difference_type i) {
    i += offset;
    if (i == 0) {
      return *first;
    } else {
      return ValueT(iter[i - 1]);
    }
  }
  __device__ chained_iterator operator+(difference_type i) {
    return chained_iterator{iter, first, i};
  }
  __device__ ValueT operator*() {
    return (*this)[0];
  }
};

struct CountMaskOp {
  __device__ int64_t operator() (const uint8_t &x) const {
    return x != 0;
  }
};

template<typename InputIteratorT, typename OutputIteratorT, typename ScanOpT, typename InitValueT, int max_cub_size=max_cub_size>
inline void exclusive_scan(InputIteratorT input, OutputIteratorT output, ScanOpT scan_op, InitValueT init_value, int64_t num_items, const Stream& stream) {
    HT_ASSERT(num_items < max_cub_size) << "CUB exclusive_scan only supports input size < 2**30";
    int size_cub = std::min<int64_t>(max_cub_size, num_items);

    CUDAStream cuda_stream(stream);
    size_t temp_storage_bytes = 0;    

    hetu::cuda::CUDADeviceGuard guard(stream.device_index());
    CUDA_CALL(cub::DeviceScan::ExclusiveScan(nullptr, temp_storage_bytes, input, output, scan_op, init_value, size_cub, cuda_stream));
    NDArray temp_storage = NDArray::empty({(int64_t)temp_storage_bytes}, stream.device(), kInt8, stream.stream_index());
    CUDA_CALL(cub::DeviceScan::ExclusiveScan(temp_storage->data_ptr<void>(), temp_storage_bytes, input, output, scan_op, init_value, size_cub, cuda_stream));

    NDArray::MarkUsedBy({temp_storage}, stream);
    return;
}

}
}