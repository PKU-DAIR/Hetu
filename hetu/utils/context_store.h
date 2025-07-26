#pragma once

#include "hetu/common/macros.h"
#include <any>
#include <cstdint>
#include "hetu/utils/json/json.hpp"
#include "hetu/graph/tensor.h"

namespace hetu {

using json = nlohmann::json;
using Tensor = hetu::graph::Tensor;

template<typename T>
std::string serialize(T&& obj) {
  json j;
  if constexpr (std::is_rvalue_reference_v<decltype(obj)>) {
    j = std::move(obj);
  } else {
    j = obj;
  }
  return j.dump();
}

template<typename T>
std::decay_t<T> deserialize(const std::string& str) {
    json j = json::parse(str);
    return j.get<std::decay_t<T>>();
}

template<typename T, typename = void>
struct is_json_serializable : std::false_type {};

template<typename T>
struct is_json_serializable<T, std::void_t<
  decltype(std::declval<json&>() = std::declval<T>())
>> : std::true_type {};

class ContextStore {
 public:
  ContextStore() {}

  void put_bool(const std::string& key, bool value) {
    _ctx_any.insert({key, std::make_any<bool>(value)});
  }

  bool get_bool(const std::string& key, bool default_value = false) const {
    auto it = _ctx_any.find(key);
    return it != _ctx_any.end() ? std::any_cast<bool>(it->second) : default_value;
  }

  void put_int32(const std::string& key, int32_t value) {
    _ctx_any.insert({key, std::make_any<int32_t>(value)});
  }

  int32_t get_int32(const std::string& key, int32_t default_value = 0) const {
    auto it = _ctx_any.find(key);
    return it != _ctx_any.end() ? std::any_cast<int32_t>(it->second) : default_value;
  }

  void put_uint32(const std::string& key, uint32_t value) {
    _ctx_any.insert({key, std::make_any<uint32_t>(value)});
  }

  uint32_t get_uint32(const std::string& key,
                      uint32_t default_value = 0) const {
    auto it = _ctx_any.find(key);
    return it != _ctx_any.end() ? std::any_cast<uint32_t>(it->second)
                            : default_value;
  }

  void put_int64(const std::string& key, int64_t value) {
    _ctx_any.insert({key, std::make_any<int64_t>(value)});
  }

  int64_t get_int64(const std::string& key, int64_t default_value = 0) const {
    auto it = _ctx_any.find(key);
    return it != _ctx_any.end() ? std::any_cast<int64_t>(it->second) : default_value;
  }

  void put_uint64(const std::string& key, uint64_t value) {
    _ctx_any.insert({key, std::make_any<uint64_t>(value)});
  }

  uint64_t get_uint64(const std::string& key,
                      uint64_t default_value = 0) const {
    auto it = _ctx_any.find(key);
    return it != _ctx_any.end() ? std::any_cast<uint64_t>(it->second) : default_value;
  }

  void put_float32(const std::string& key, float value) {
    _ctx_any.insert({key, std::make_any<float>(value)});
  }

  float get_float32(const std::string& key, float default_value = 0) const {
    auto it = _ctx_any.find(key);
    return it != _ctx_any.end() ? std::any_cast<float>(it->second) : default_value;
  }

  void put_float64(const std::string& key, double value) {
    _ctx_any.insert({key, std::make_any<double>(value)});
  }

  double get_float64(const std::string& key, double default_value = 0) const {
    auto it = _ctx_any.find(key);
    return it != _ctx_any.end() ? std::any_cast<double>(it->second) : default_value;
  }

  void put_string(const std::string& key, const std::string& value) {
    _ctx_any.insert({key, std::make_any<std::string>(value)});
  }

  const std::string& get_string(const std::string& key,
                                const std::string& default_value = "") const {
    auto it = _ctx_any.find(key);
    return it != _ctx_any.end() ? std::any_cast<std::string>(it->second) : default_value;
  }

  void put_ndarray(const std::string& key, const NDArray& value) {
    _ctx_any.insert({key, std::make_any<NDArray>(value)});
  }

  const NDArray& get_ndarray(const std::string& key) const {
    auto it = _ctx_any.find(key);
    HT_ASSERT(it != _ctx_any.end()) << "NDArray " << key << " not found";
    return std::any_cast<NDArray>(it->second);
  }

  NDArray pop_ndarray(const std::string& key) {
    auto it = _ctx_any.find(key);
    HT_ASSERT(it != _ctx_any.end()) << "NDArray " << key << " not found";
    NDArray result = std::any_cast<NDArray>(it->second);
    _ctx_any.erase(it);
    return result;
  }

  template<typename T>
  void put_param(const std::string& key, const T& value) {
    _ctx_any.insert({key, std::make_any<T>(value)});
  }

  template <typename T>
  void put(const std::string& key, T&& value) {
    using DecayT = std::decay_t<T>;
    if constexpr (std::is_same_v<DecayT, NDArray> ||
                  std::is_same_v<DecayT, NDArrayMeta> ||
                  std::is_same_v<DecayT, Tensor>) {
      if constexpr (std::is_same_v<DecayT, NDArray>) {
        _ctx_ndarray.insert_or_assign(key, std::forward<T>(value));
      } else if constexpr (std::is_same_v<DecayT, NDArrayMeta>) {
        _ctx_ndarray_meta.insert_or_assign(key, std::forward<T>(value));
      } else if constexpr (std::is_same_v<DecayT, Tensor>) {
        _ctx_tensor.insert_or_assign(key, std::forward<T>(value));
      }
    } else if constexpr (is_json_serializable<DecayT>::value) {
      _ctx.insert_or_assign(key, serialize(std::forward<T>(value)));
    } else {
      HT_ASSERT(false) << "Type " << typeid(T).name() << " is not serializable";
    }
  }

  template <typename T>
  T get(const std::string& key) const {
    if constexpr (std::is_same_v<T, NDArray> ||
                  std::is_same_v<T, NDArrayMeta> ||
                  std::is_same_v<T, Tensor>) {
      if constexpr (std::is_same_v<T, NDArray>) {
        auto it = _ctx_ndarray.find(key);
        HT_ASSERT(it != _ctx_ndarray.end()) << "NDArray " << key << " not found";
        return it->second;
      } else if constexpr (std::is_same_v<T, NDArrayMeta>) {
        auto it = _ctx_ndarray_meta.find(key);
        HT_ASSERT(it != _ctx_ndarray_meta.end()) << "NDArrayMeta " << key << " not found";
        return it->second;
      } else if constexpr (std::is_same_v<T, Tensor>) {
        auto it = _ctx_tensor.find(key);
        HT_ASSERT(it != _ctx_tensor.end()) << "Tensor " << key << " not found";
        return it->second;
      }
    } else if constexpr (is_json_serializable<T>::value) {
      auto it = _ctx.find(key);
      HT_ASSERT(it != _ctx.end()) << "key-value pair " << key << " not found";
      return deserialize<std::decay_t<T>>(it->second);
    } else {
      HT_ASSERT(false) << "Type " << typeid(T).name() << " is not serializable";
    }
  }

  template <typename T>
  T pop(const std::string& key) {
    if constexpr (std::is_same_v<T, NDArray> ||
                  std::is_same_v<T, NDArrayMeta> ||
                  std::is_same_v<T, Tensor>) {
      if constexpr (std::is_same_v<T, NDArray>) {
        auto node_handle = _ctx_ndarray.extract(key);
        HT_ASSERT(!node_handle.empty()) << "NDArray " << key << " not found";
        return std::move(node_handle.mapped());
      } else if constexpr (std::is_same_v<T, NDArrayMeta>) {
        auto node_handle = _ctx_ndarray_meta.extract(key);
        HT_ASSERT(!node_handle.empty()) << "NDArrayMeta " << key << " not found";
        return std::move(node_handle.mapped());
      } else if constexpr (std::is_same_v<T, Tensor>) {
        auto node_handle = _ctx_tensor.extract(key);
        HT_ASSERT(!node_handle.empty()) << "Tensor " << key << " not found";
        return std::move(node_handle.mapped());
      }
    } else if constexpr (is_json_serializable<T>::value) {
      auto node_handle = _ctx.extract(key);
      HT_ASSERT(!node_handle.empty()) << "key-value pair " << key << " not found";
      return deserialize<T>(std::move(node_handle.mapped()));
    } else {
      HT_ASSERT(false) << "Type " << typeid(T).name() << " is not serializable";
    }
  }

  template <typename T>
  bool contains(const std::string& key) const {
    if constexpr (std::is_same_v<T, NDArray> ||
                  std::is_same_v<T, NDArrayMeta> ||
                  std::is_same_v<T, Tensor>) {
      return _ctx_ndarray.find(key) != _ctx_ndarray.end() ||
             _ctx_ndarray_meta.find(key) != _ctx_ndarray_meta.end() ||
             _ctx_tensor.find(key) != _ctx_tensor.end();
    } else if constexpr (is_json_serializable<T>::value) {
      return _ctx.find(key) != _ctx.end();
    } else {
      HT_ASSERT(false) << "Type " << typeid(T).name() << " is not serializable";
    }
  }

  template <typename T>
  void migrate_from(ContextStore& src, const std::string& key, const std::string& new_key = "") {
    std::string final_key = new_key.empty() ? key : new_key;
    if (!this->contains<T>(final_key) || (this->contains<T>(final_key) && src.contains<T>(key))) {
      this->put(final_key, src.get<T>(key));
    }
  }


 private:
  std::unordered_map<std::string, std::string> _ctx;
  std::unordered_map<std::string, std::any> _ctx_any;
  std::unordered_map<std::string, NDArray> _ctx_ndarray;
  std::unordered_map<std::string, NDArrayMeta> _ctx_ndarray_meta;
  std::unordered_map<std::string, Tensor> _ctx_tensor;
};

} // namespace hetu
