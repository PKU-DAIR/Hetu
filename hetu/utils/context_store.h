#pragma once

#include "hetu/common/macros.h"
#include <any>
#include <cstdint>

namespace hetu {

class ContextStore {
 public:
  ContextStore() {}

  void put_bool(const std::string& key, bool value) {
    _ctx.insert({key, std::make_any<bool>(value)});
  }

  bool get_bool(const std::string& key, bool default_value = false) const {
    auto it = _ctx.find(key);
    return it != _ctx.end() ? std::any_cast<bool>(it->second) : default_value;
  }

  void put_int32(const std::string& key, int32_t value) {
    _ctx.insert({key, std::make_any<int32_t>(value)});
  }

  int32_t get_int32(const std::string& key, int32_t default_value = 0) const {
    auto it = _ctx.find(key);
    return it != _ctx.end() ? std::any_cast<int32_t>(it->second) : default_value;
  }

  void put_uint32(const std::string& key, uint32_t value) {
    _ctx.insert({key, std::make_any<uint32_t>(value)});
  }

  uint32_t get_uint32(const std::string& key,
                      uint32_t default_value = 0) const {
    auto it = _ctx.find(key);
    return it != _ctx.end() ? std::any_cast<uint32_t>(it->second)
                            : default_value;
  }

  void put_int64(const std::string& key, int64_t value) {
    _ctx.insert({key, std::make_any<int64_t>(value)});
  }

  int64_t get_int64(const std::string& key, int64_t default_value = 0) const {
    auto it = _ctx.find(key);
    return it != _ctx.end() ? std::any_cast<int64_t>(it->second) : default_value;
  }

  void put_uint64(const std::string& key, uint64_t value) {
    _ctx.insert({key, std::make_any<uint64_t>(value)});
  }

  uint64_t get_uint64(const std::string& key,
                      uint64_t default_value = 0) const {
    auto it = _ctx.find(key);
    return it != _ctx.end() ? std::any_cast<uint64_t>(it->second) : default_value;
  }

  void put_float32(const std::string& key, float value) {
    _ctx.insert({key, std::make_any<float>(value)});
  }

  float get_float32(const std::string& key, float default_value = 0) const {
    auto it = _ctx.find(key);
    return it != _ctx.end() ? std::any_cast<float>(it->second) : default_value;
  }

  void put_float64(const std::string& key, double value) {
    _ctx.insert({key, std::make_any<double>(value)});
  }

  double get_float64(const std::string& key, double default_value = 0) const {
    auto it = _ctx.find(key);
    return it != _ctx.end() ? std::any_cast<double>(it->second) : default_value;
  }

  void put_string(const std::string& key, const std::string& value) {
    _ctx.insert({key, std::make_any<std::string>(value)});
  }

  const std::string& get_string(const std::string& key,
                                const std::string& default_value = "") const {
    auto it = _ctx.find(key);
    return it != _ctx.end() ? std::any_cast<std::string>(it->second) : default_value;
  }

  void put_ndarray(const std::string& key, const NDArray& value) {
    _ctx.insert({key, std::make_any<NDArray>(value)});
  }

  const NDArray& get_ndarray(const std::string& key) const {
    auto it = _ctx.find(key);
    HT_ASSERT(it != _ctx.end()) << "NDArray " << key << " not found";
    return std::any_cast<NDArray>(it->second);
  }

  NDArray pop_ndarray(const std::string& key) {
    auto it = _ctx.find(key);
    HT_ASSERT(it != _ctx.end()) << "NDArray " << key << " not found";
    NDArray result = std::any_cast<NDArray>(it->second);
    _ctx.erase(it);
    return result;
  }

  template<typename T>
  void put_param(const std::string& key, const T& value) {
    _ctx.insert({key, std::make_any<T>(value)});
  }

 private:
  std::unordered_map<std::string, std::any> _ctx;
};

} // namespace hetu
