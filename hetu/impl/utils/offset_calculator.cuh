#pragma once

#include "hetu/core/ndarray.h"
#include "hetu/core/memory_pool.h"
#include "hetu/utils/robin_hood_hashing.h"
#include "hetu/utils/unordered_dense.h"

#include <tuple>
#include <cassert>
#include <unordered_map>
#include <list>

namespace hetu {
namespace impl {

class OffsetCalculator {
 public:
  __device__ OffsetCalculator() = default;
  __device__ ~OffsetCalculator() = default;
  __host__ __device__ virtual inline size_t get(size_t linear_idx) const {
    return linear_idx;
  }
};

class StridedOffsetCalculator : public OffsetCalculator {
 public:
  __device__ StridedOffsetCalculator(int dims, const int64_t* shape, const int64_t* stride)
    : OffsetCalculator(),
    _dims(dims) {
    assert(dims <= HT_MAX_NDIM);

    for (int i = 0; i < dims; i++) {
      _shape[i] = shape[i];
      _stride[i] = stride[i];
    }
  }

  StridedOffsetCalculator(int dims, HTShape shape, HTShape stride)
    : OffsetCalculator(),
    _dims(dims) {
    HT_ASSERT(dims <= HT_MAX_NDIM)
      << "Currently we only support shape up to " << HT_MAX_NDIM
      << " dimensions. Got" << dims << ".";

    for (int i = 0; i < dims; i++) {
      _shape[i] = shape[i];
      _stride[i] = stride[i];
    }
  }
  __device__ ~StridedOffsetCalculator() = default;

  __host__ __device__ inline size_t get(size_t linear_idx) const override {
    size_t offset = 0;
    #pragma unroll
    for (int i = _dims - 1; i >= 1; i--) {
      int64_t shape_i = _shape[i];
      auto div_idx = linear_idx / shape_i;
      auto mod_idx = linear_idx - div_idx * shape_i;
      offset += mod_idx * _stride[i];
      linear_idx = div_idx;
    }
    return offset + linear_idx * _stride[0];
  }

 protected:
  int _dims;
  int64_t _shape[HT_MAX_NDIM];
  int64_t _stride[HT_MAX_NDIM];
};

__global__ static void trivial_constructor(OffsetCalculator* dst) {
  new(dst) OffsetCalculator();
}

__global__ static void strided_constructor(StridedOffsetCalculator* dst, int dims,
                                           const int64_t* shape, const int64_t* stride) {
  new(dst) StridedOffsetCalculator(dims, shape, stride);
}

std::tuple<NDArray, OffsetCalculator*>
AllocOffsetCalculator(const NDArray& arr, const Stream& stream);

std::tuple<NDArrayList, std::vector<OffsetCalculator*>>
AllocOffsetCalculator(const NDArrayList& arr_list, const Stream& stream);

// TODO: Find a suitable capacity for LFU cache
constexpr size_t HT_LFU_CAPACITY = 32;

struct CacheKey {
  HTShape shape;
  HTStride stride;

  CacheKey() = default;
  CacheKey(const HTShape& shape, const HTStride& stride) : shape(shape), stride(stride) {}

  bool operator==(const CacheKey& other) const {
    return shape == other.shape && stride == other.stride;
  }
};

struct CacheKeyHash {
  size_t operator()(const CacheKey& key) const {
    size_t seed = 0;
    auto hash_combine = [](size_t& seed, int64_t value) {
      // Following boost::hash_combine
      std::hash<int64_t> hasher;
      seed ^= hasher(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    };
    for (auto i : key.shape) {
      hash_combine(seed, i);
    }
    for (auto i : key.stride) {
      hash_combine(seed, i);
    }
    return seed;
  }
};

struct CacheValue {
  NDArray offset_calculator_arr;
  StridedOffsetCalculator* offset_calculator;

  CacheValue() = default;

  CacheValue(const NDArray& offset_calculator_arr, StridedOffsetCalculator* offset_calculator)
    : offset_calculator_arr(offset_calculator_arr), offset_calculator(offset_calculator) {}
};

struct CacheNode {
  CacheKey key;
  CacheValue value;
  size_t frequency;

  CacheNode() = default;
  CacheNode(const CacheKey& key, const CacheValue& value, size_t frequency)
    : key(key), value(value), frequency(frequency) {}
};

class LFUCache {
 public:
  LFUCache() = default;
  LFUCache(size_t capacity) : _capacity(capacity), _minFreq(0) {
    _cache.reserve(capacity);
    _freqList.reserve(capacity);
  }

  std::tuple<NDArray, StridedOffsetCalculator*> get(const HTShape& shape, const HTStride& stride) {
    return get(CacheKey{shape, stride});
  }

  void put(const HTShape& shape, const HTStride& stride, const NDArray& offset_calculator_arr,
           StridedOffsetCalculator* offset_calculator) {
    put(CacheKey{shape, stride}, CacheValue{offset_calculator_arr, offset_calculator});
  }

 protected:
  std::tuple<NDArray, StridedOffsetCalculator*> get(const CacheKey& key) {
    if (_capacity == 0) return {NDArray(), nullptr};
    auto it = _cache.find(key);
    if (it == _cache.end()) return {NDArray(), nullptr};
    auto node = it->second;
    auto val = node->value;
    auto freq = node->frequency;
    // remove from `freq` list and push to `freq + 1` list
    _freqList[freq].erase(node);
    if (_freqList[freq].size() == 0) {
      _freqList.erase(freq);
      if (_minFreq == freq) _minFreq++;
    }
    _freqList[freq + 1].push_front(CacheNode(key, val, freq + 1));
    _cache[key] = _freqList[freq + 1].begin();
    return {val.offset_calculator_arr, val.offset_calculator};
  }

  void put(const CacheKey& key, const CacheValue& val) {
    if (_capacity == 0) return;
    auto it = _cache.find(key);
    if (it == _cache.end()) {
      // evict when cache is full
      if (_cache.size() == _capacity) {
        auto evict_it = _freqList[_minFreq].back();
        _cache.erase(evict_it.key);
        _freqList[_minFreq].pop_back();
        if (_freqList[_minFreq].size() == 0) {
          _freqList.erase(_minFreq);
        }
      }
      _freqList[1].push_front(CacheNode(key, val, 1));
      _cache[key] = _freqList[1].begin();
      _minFreq = 1;
    } else if (!val.offset_calculator_arr.is_defined()){
      // update value
      auto node = it->second;
      auto freq = node->frequency;
      _freqList[freq].erase(node);
      if (_freqList[freq].size() == 0) {
        _freqList.erase(freq);
        if (_minFreq == freq) _minFreq++;
      }
      _freqList[freq + 1].push_front(CacheNode(key, val, freq + 1));
      _cache[key] = _freqList[freq + 1].begin();
    } else {
      HT_RUNTIME_ERROR << "OffsetCalculator already exists in cache.";
    }
  }

  size_t _capacity;
  size_t _minFreq;
  ankerl::unordered_dense::map<CacheKey, std::list<CacheNode>::iterator, CacheKeyHash> _cache;
  ankerl::unordered_dense::map<size_t, std::list<CacheNode>, ankerl::unordered_dense::hash<size_t>> _freqList;
};

} // namespace impl
} // namespace hetu