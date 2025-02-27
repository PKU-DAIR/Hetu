#pragma once

#include "hetu/common/macros.h"
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
  std::unordered_map<std::string, NDArray> _ctx_ndarray;
  std::unordered_map<std::string, NDArrayMeta> _ctx_ndarray_meta;
  std::unordered_map<std::string, Tensor> _ctx_tensor;
};

} // namespace hetu
