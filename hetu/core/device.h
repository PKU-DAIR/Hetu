#pragma once

#include "hetu/common/macros.h"
#include <tuple>
#include <cuda_runtime.h>

namespace hetu {

struct DeviceProp {
  int major;
  int minor;
  int multiProcessorCount;
  int maxThreadsPerMultiProcessor;
  int maxGridSize[3];

  DeviceProp() {
    major = 0;
    minor = 0;
    multiProcessorCount = 0;
    maxThreadsPerMultiProcessor = 0;
  }

  DeviceProp(cudaDeviceProp& dprop) {
    major = dprop.major;
    minor = dprop.minor;
    multiProcessorCount = dprop.multiProcessorCount;
    maxThreadsPerMultiProcessor = dprop.maxThreadsPerMultiProcessor;
    for (int i = 0; i < 3; ++i) {
      maxGridSize[i] = dprop.maxGridSize[i];
    }
  }
};

enum class DeviceType : int8_t {
  CPU = 0,
  CUDA,
  NUM_DEVICE_TYPES,
  UNDETERMINED
};

constexpr DeviceType kCPU = DeviceType::CPU;
constexpr DeviceType kCUDA = DeviceType::CUDA;
constexpr DeviceType kUndeterminedDevice = DeviceType::UNDETERMINED;
constexpr int16_t NUM_DEVICE_TYPES =
  static_cast<int16_t>(DeviceType::NUM_DEVICE_TYPES);

std::string DeviceType2Str(const DeviceType&);
std::ostream& operator<<(std::ostream&, const DeviceType&);

using DeviceIndex = uint8_t;

#define HT_MAX_DEVICE_INDEX (16)
#define HT_MAX_HOSTNAME_LENGTH (256)
#define HT_MAX_DEVICE_MULTIPLEX (16)

class Device {
 public:
  static constexpr char BACK_SLASH = '/';
  static constexpr char COLON = ':';

  Device(DeviceType type = kUndeterminedDevice, DeviceIndex index = 0U,
         const std::string& hostname = "", uint8_t multiplex = 0U,
         DeviceIndex global_index = 0U) {
    _init(type, index, hostname, multiplex, global_index);
  }

  Device(const std::string& device, uint8_t multiplex = 0U);

  Device(const Device&) = default;
  Device(Device&&) = default;
  Device& operator=(const Device& device) = default;
  Device& operator=(Device&& device) = default;

  ~Device() = default;

  inline bool operator==(const Device& device) const {
    return type() == device.type() && index() == device.index() &&
      Device::compare_hostname(*this, device) == 0 &&
      multiplex() == device.multiplex();
  }

  inline bool operator!=(const Device& device) const {
    return !operator==(device);
  }

  inline bool operator<(const Device& device) const {
    auto tmp = Device::compare_hostname(*this, device);
    if (tmp != 0)
      return tmp < 0;
    if (type() != device.type())
      return type() < device.type();
    if (index() != device.index())
      return index() < device.index();
    if (multiplex() != device.multiplex())
      return multiplex() < device.multiplex();
    return false;
  }

  inline DeviceType type() const noexcept {
    return _type;
  }

  inline DeviceIndex index() const noexcept {
    return _index;
  }

  inline DeviceIndex global_index() const noexcept {
    return _global_index;
  }

  inline bool is_cpu() const noexcept {
    return _type == kCPU;
  }

  inline bool is_cuda() const noexcept {
    return _type == kCUDA;
  }

  inline bool is_undetermined() const noexcept {
    return _type == kUndeterminedDevice;
  }

  inline bool local() const noexcept {
    return _hostname.empty();
  }

  inline const std::string& hostname() const noexcept {
    return _hostname;
  }

  inline uint8_t multiplex() const noexcept {
    return _multiplex;
  }

  static cudaDeviceProp dprop(int idx) {
    return _dprops[idx].second;
  }

  cudaDeviceProp cuda_dprop() {
    HT_ASSERT(is_cuda())
      << "cuda_dprop() is only available for CUDA devices";
    auto& dprop = _dprops[index()].second;
    auto cuda_dprop = cudaDeviceProp();
    // copy all fields from DeviceProp to cudaDeviceProp
    cuda_dprop.major = dprop.major;
    cuda_dprop.minor = dprop.minor;
    cuda_dprop.multiProcessorCount = dprop.multiProcessorCount;
    cuda_dprop.maxThreadsPerMultiProcessor = dprop.maxThreadsPerMultiProcessor;
    for (int i = 0; i < 3; ++i) {
      cuda_dprop.maxGridSize[i] = dprop.maxGridSize[i];
    }
    return std::move(cuda_dprop);
  }

  std::string compat_string() const;

  static inline std::string GetLocalHostname() {
    char* env = std::getenv("HETU_LOCAL_HOSTNAME");
    if (env == nullptr) {
      return "";
    } else {
      std::string ret(env);
      HT_ASSERT(ret.length() <= HT_MAX_HOSTNAME_LENGTH)
        << "Hostname \"" << ret << "\" exceeds max length "
        << HT_MAX_HOSTNAME_LENGTH;
      ;
      return ret;
    }
  }

  static int compare_hostname(const Device& d1, const Device& d2) {
    if (d1.local() && d2.local()) {
      return 0;
    } else if (d1.local()) {
      return GetLocalHostname().compare(d2.hostname());
    } else if (d2.local()) {
      return d1.hostname().compare(GetLocalHostname());
    } else {
      return d1.hostname().compare(d2.hostname());
    }
  }

 protected: 
  static std::vector<std::pair<bool, cudaDeviceProp>> _dprops;

  static std::vector<std::pair<bool, cudaDeviceProp>> CUDAInit();

 private:
  void _init(DeviceType type, DeviceIndex index, const std::string& hostname,
             uint8_t multiplex, DeviceIndex global_index = 0U) {
    _type = type;
    _index = _type == kCUDA ? index : 0U;
    _global_index = _type == kCUDA ? global_index : 0U;
    HT_ASSERT(_index < HT_MAX_DEVICE_INDEX)
      << "Device index " << _index << " exceeds maximum allowed value "
      << HT_MAX_DEVICE_INDEX;
    if (!hostname.empty() && hostname != "localhost" &&
        hostname != GetLocalHostname()) {
      HT_ASSERT(hostname.length() <= HT_MAX_HOSTNAME_LENGTH)
        << "Hostname \"" << hostname << "\" exceeds max length "
        << HT_MAX_HOSTNAME_LENGTH;
      _hostname = hostname;
    }
    HT_ASSERT(multiplex < HT_MAX_DEVICE_MULTIPLEX)
      << "Multiplex " << multiplex << " exceeds maximum allowed value "
      << HT_MAX_HOSTNAME_LENGTH;
    _multiplex = multiplex;
    if (_type == kCUDA && _dprops[_index].first == false) {
      _dprops[_index].first = true;
      cudaDeviceProp dprop;
      cudaGetDeviceProperties(&dprop, _index);
      _dprops[_index].second = std::move(dprop);
    }
  }

  DeviceType _type;
  DeviceIndex _index;
  std::string _hostname;
  uint8_t _multiplex;
  DeviceIndex _global_index;
};

std::ostream& operator<<(std::ostream&, const Device&);

class DeviceGroup;
using DeviceGroupList = std::vector<DeviceGroup>;

class DeviceGroup {
 public:
  DeviceGroup(const std::vector<Device>& devices) : _devices(devices) {
    // note the order of device group is important to recognize the heterogenous pipelines
    // std::sort(_devices.begin(), _devices.end());
    for (const auto& device : devices) {
      if (device.is_undetermined()) {
        HT_LOG_WARN << "Currently shouldn't use dummy DeviceGroup"
          << ", please use local ds or local dg"
          << ", rather than create a dummy one"
          << ", because it is dangerous";
        _dummy = true;
      }
    }
  }

  DeviceGroup(const std::vector<std::string>& devices) {
    _devices.reserve(devices.size());
    for (const auto& device : devices) {
      _devices.emplace_back(device);
      if (device == "") {
        _dummy = true;
      }
    }
  }

  DeviceGroup() : DeviceGroup(std::vector<Device>()) {}

  DeviceGroup(const DeviceGroup&) = default;
  DeviceGroup& operator=(const DeviceGroup&) = default;
  DeviceGroup(DeviceGroup&&) = default;
  DeviceGroup& operator=(DeviceGroup&&) = default;

  inline bool operator==(const DeviceGroup& device_group) const {
    return _devices == device_group._devices;
  }

  inline bool operator!=(const DeviceGroup& device_group) const {
    return !operator==(device_group);
  }

  inline bool operator<(const DeviceGroup& device_group) const {
    return _devices < device_group._devices;
  }

  inline size_t get_index(const Device& device) const {
    auto it = std::find(_devices.begin(), _devices.end(), device);
    HT_ASSERT_NE(it, _devices.end()) << "Device not found: " << device;
    return it - _devices.begin();
  }

  inline const std::vector<Device>& devices() const {
    return _devices;
  }

  inline size_t num_devices() const {
    return _devices.size();
  }

  inline bool empty() const {
    return _devices.empty();
  }

  inline bool contains(const Device& device) const {
    auto it = std::find(_devices.begin(), _devices.end(), device);
    return it != _devices.end();
  }

  inline bool is_subset(const DeviceGroup& device_group) const {
    for (auto& device : device_group.devices()) {
      if (!contains(device)) {
        return false;
      }
    }
    return true;
  }

  inline const Device& get(size_t i) const {
    return _devices[i];
  }

  inline void set(size_t i, Device device) {
    _devices[i] = device;
  }

 private:
  std::vector<Device> _devices;
  bool _dummy{false};
};

std::ostream& operator<<(std::ostream&, const DeviceGroup&);

} // namespace hetu

namespace std {

template <>
struct hash<hetu::Device> {
  std::size_t operator()(const hetu::Device& device) const noexcept {
    auto hash = std::hash<int>()((static_cast<int>(device.type()) << 16) |
                                 (static_cast<int>(device.index()) << 8) |
                                 static_cast<int>(device.multiplex()));
    if (!device.local()) {
      // Following boost::hash_combine
      hash ^= (std::hash<std::string>()(device.hostname()) + 0x9e3779b9 +
               (hash << 6) + (hash >> 2));
    }
    return hash;
  }
};

template <>
struct hash<std::pair<hetu::Device, hetu::Device>> {
  std::size_t operator()(const std::pair<hetu::Device, hetu::Device>& device_pair) const noexcept {
    // devices in pair need to sort
    auto hash_op = std::hash<hetu::Device>();
    size_t seed = 2;
    if (device_pair.first < device_pair.second) {
      seed ^= hash_op(device_pair.first) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
      seed ^= hash_op(device_pair.second) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    } else {
      seed ^= hash_op(device_pair.second) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
      seed ^= hash_op(device_pair.first) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
    return seed;
  }
};

template <>
struct hash<hetu::DeviceGroup> {
  std::size_t operator()(const hetu::DeviceGroup& group) const noexcept {
    // devices in group are sorted, so we can hash them without re-ordering
    const auto& devices = group.devices();
    auto hash_op = std::hash<hetu::Device>();
    auto seed = devices.size();
    for (const auto& device : devices)
      seed ^= hash_op(device) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    return seed;
  }
};

inline std::string to_string(const hetu::Device& device) {
  std::ostringstream os;
  os << device;
  return os.str();
}

inline std::string to_string(const hetu::DeviceGroup& group) {
  std::ostringstream os;
  os << group;
  return os.str();
}

} // namespace std