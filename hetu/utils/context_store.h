#pragma once

#include "hetu/common/macros.h"

namespace hetu {

class ContextStore {
 public:
  ContextStore() {}

  void put_bool(const std::string& key, bool value) {
    _ctx.insert({key, value ? "true" : "false"});
  }

  bool get_bool(const std::string& key, bool default_value = false) const {
    auto it = _ctx.find(key);
    return it != _ctx.end() ? (it->second == "true") : default_value;
  }

  void put_int32(const std::string& key, int32_t value) {
    _ctx.insert({key, std::to_string(value)});
  }

  int32_t get_int32(const std::string& key, int32_t default_value = 0) const {
    auto it = _ctx.find(key);
    return it != _ctx.end() ? std::stoi(it->second) : default_value;
  }

  void put_uint32(const std::string& key, uint32_t value) {
    _ctx.insert({key, std::to_string(value)});
  }

  uint32_t get_uint32(const std::string& key,
                      uint32_t default_value = 0) const {
    auto it = _ctx.find(key);
    return it != _ctx.end() ? static_cast<uint32_t>(std::stoul(it->second))
                            : default_value;
  }

  void put_int64(const std::string& key, int64_t value) {
    _ctx.insert({key, std::to_string(value)});
  }

  int64_t get_int64(const std::string& key, int64_t default_value = 0) const {
    auto it = _ctx.find(key);
    return it != _ctx.end() ? std::stoll(it->second) : default_value;
  }

  void put_uint64(const std::string& key, uint64_t value) {
    _ctx.insert({key, std::to_string(value)});
  }

  std::vector<int64_t> get_int64_list(const std::string& key) const {
    auto it = _ctx.find(key);
    std::vector<int64_t> ret;
    if (it == _ctx.end()) {
      return ret;
    }
    std::stringstream ss(it->second);
    std::string item;
    while (std::getline(ss, item, ',')) {
      ret.push_back(std::stoll(item));
    }
    return ret;
  }

  void put_int64_list(const std::string& key, const std::vector<int64_t>& value) {
    std::stringstream ss;
    for (size_t i = 0; i < value.size(); ++i) {
      if (i > 0) {
        ss << ",";
      }
      ss << value[i];
    }
    _ctx.insert({key, ss.str()});
  }

  uint64_t get_uint64(const std::string& key,
                      uint64_t default_value = 0) const {
    auto it = _ctx.find(key);
    return it != _ctx.end() ? std::stoull(it->second) : default_value;
  }

  void put_float32(const std::string& key, float value) {
    _ctx.insert({key, std::to_string(value)});
  }

  float get_float32(const std::string& key, float default_value = 0) const {
    auto it = _ctx.find(key);
    return it != _ctx.end() ? std::stof(it->second) : default_value;
  }

  void put_float64(const std::string& key, double value) {
    _ctx.insert({key, std::to_string(value)});
  }

  double get_float64(const std::string& key, double default_value = 0) const {
    auto it = _ctx.find(key);
    return it != _ctx.end() ? std::stod(it->second) : default_value;
  }

  void put_string(const std::string& key, const std::string& value) {
    _ctx.insert({key, value});
  }

  const std::string& get_string(const std::string& key,
                                const std::string& default_value = "") const {
    auto it = _ctx.find(key);
    return it != _ctx.end() ? it->second : default_value;
  }

  void put_ndarray(const std::string& key, const NDArray& value) {
    _ctx_ndarray.insert({key, value});
  }

  const NDArray& get_ndarray(const std::string& key) const {
    auto it = _ctx_ndarray.find(key);
    HT_ASSERT(it != _ctx_ndarray.end()) << "NDArray " << key << " not found";
    return it->second;
  }

  NDArray pop_ndarray(const std::string& key) {
    auto it = _ctx_ndarray.find(key);
    HT_ASSERT(it != _ctx_ndarray.end()) << "NDArray " << key << " not found";
    NDArray result = it->second;
    _ctx_ndarray.erase(it);
    return result;
  }  

 private:
  std::unordered_map<std::string, std::string> _ctx;
  std::unordered_map<std::string, NDArray> _ctx_ndarray;
};

} // namespace hetu
