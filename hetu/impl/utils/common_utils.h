#pragma once

#include "hetu/impl/utils/ndarray_utils.h"
#include "hetu/impl/utils/dispatch.h"

namespace hetu {
namespace impl {

template<typename spec_t, int vec_size>
struct alignas(sizeof(spec_t) * vec_size) aligned_vector {
  spec_t val[vec_size];

  __host__ __device__ spec_t& operator[](int idx) {
    return val[idx];
  }

  __host__ __device__ const spec_t& operator[](int idx) const {
    return val[idx];
  }
};

inline size_t numel(const HTShape& shape) {
  size_t num = 1;
  for (auto& s : shape) {
    num *= s;
  }
  return num;
}

inline HTStride Shape2Stride(const HTShape& shape) {
  auto size = shape.size();
  HTStride stride(size);
  if (size > 0) {
    stride[size - 1] = 1;
    for (auto d = size - 1; d > 0; d--) {
      stride[d - 1] = stride[d] * shape[d];
    }
  }
  return stride;
}

inline bool IsContiguous(const NDArrayList& arrays) {
  for (const auto& array : arrays) {
    if (!array->is_contiguous()) {
      return false;
    }
  }
  return true;
}

inline int GetThreadNum(int cnt) {
  if (cnt >= 1048576)
    return 1024;
  if (cnt >= 262144)
    return 512;
  if (cnt >= 65536)
    return 256;
  if (cnt >= 16384)
    return 128;
  if (cnt >= 256)
    return 64;
  return 32;
}

inline int64_t get_index(int64_t idx, int64_t ndims, const int64_t* stride, const int64_t* c_shape) {
  int64_t i_idx = 0;
  int64_t t = idx;
  for (int i = ndims - 1; i >= 0; --i) {
    int64_t ratio = t % c_shape[i];
    t /= c_shape[i];
    i_idx += ratio * stride[i];
  }
  return i_idx;
}

// Simplified implementation of std::apply
template <class F, class Tuple, std::size_t... INDEX>
__host__ __device__ constexpr decltype(auto) apply_impl(
    F&& f,
    Tuple&& t,
    std::index_sequence<INDEX...>)
{
  return std::forward<F>(f)(std::get<INDEX>(std::forward<Tuple>(t))...);
}

template <class F, class Tuple>
__host__ __device__ constexpr decltype(auto) apply(F&& f, Tuple&& t) {
  return apply_impl(
      std::forward<F>(f),
      std::forward<Tuple>(t),
      std::make_index_sequence<
          std::tuple_size<std::remove_reference_t<Tuple>>::value>{});
}

template <class F, class Tuple, std::size_t... INDEX>
__host__ __device__ constexpr decltype(auto) apply_with_idx_impl(
    F&& f,
    int idx,
    Tuple&& t,
    std::index_sequence<INDEX...>)
{
  return std::forward<F>(f)(idx, std::get<INDEX>(std::forward<Tuple>(t))...);
}

template <class F, class Tuple>
__host__ __device__ constexpr decltype(auto) apply_with_idx(F&& f, int idx, Tuple&& t) {
  return apply_with_idx_impl(
      std::forward<F>(f),
      idx,
      std::forward<Tuple>(t),
      std::make_index_sequence<
          std::tuple_size<std::remove_reference_t<Tuple>>::value>{});
}

} // namespace impl
} // namespace hetu
