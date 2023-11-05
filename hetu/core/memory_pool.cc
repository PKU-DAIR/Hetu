#include "hetu/core/memory_pool.h"

#include <unordered_map>
#include <mutex>

namespace hetu {

namespace {
static std::vector<std::vector<std::shared_ptr<MemoryPool>>> device_mem_pools(
  static_cast<std::underlying_type_t<DeviceType>>(NUM_DEVICE_TYPES),
  std::vector<std::shared_ptr<MemoryPool>>(HT_MAX_DEVICE_INDEX));
static std::mutex pool_register_mutex;
static std::once_flag memory_pool_exit_handler_register_flag;
static std::once_flag error_suppression_flag;
} // namespace

void RegisterMemoryPool(std::shared_ptr<MemoryPool> memory_pool) {
  // register exit handler
  std::call_once(memory_pool_exit_handler_register_flag, []() {
    auto status = std::atexit([]() {
      std::lock_guard<std::mutex> lock(pool_register_mutex);
      HT_LOG_DEBUG << "Destructing all memory pools...";
      device_mem_pools.clear();
      HT_LOG_DEBUG << "Destructed all memory pools";
    });
    HT_ASSERT(status == 0)
      << "Failed to register the exit function for memory pools.";
  });

  std::lock_guard<std::mutex> lock(pool_register_mutex);
  Device device = memory_pool->device();
  auto device_type_id =
    static_cast<std::underlying_type_t<DeviceType>>(device.type());
  HT_ASSERT(device_mem_pools[device_type_id][device.index()] == nullptr)
    << "Memory pool for device " << device << " has been registered";
  device_mem_pools[device_type_id][device.index()] = memory_pool;
  HT_LOG_DEBUG << "Registered memory pool for device " << device;
}

std::shared_ptr<MemoryPool> GetMemoryPool(const Device& device) {
  auto device_type_id =
    static_cast<std::underlying_type_t<DeviceType>>(device.type());
  if (device_mem_pools.empty())
    return nullptr;
  auto& ret = device_mem_pools[device_type_id][device.index()];
  HT_ASSERT(ret != nullptr)
    << "Memory pool for device " << device << " does not exist";
  return ret;
}

DataPtr AllocFromMemoryPool(const Device& device, size_t num_bytes,
                            const Stream& stream) {
  if (stream.device().is_undetermined()) {
    HT_LOG_WARN << "Allocation stream not provided (" << device << ", "
                << stream << ", " << num_bytes << " bytes)";
    return GetMemoryPool(device)->AllocDataSpace(
      num_bytes, Stream(device, kComputingStream));
  } else {
    return GetMemoryPool(device)->AllocDataSpace(num_bytes, stream);
  }
}

void FreeToMemoryPool(DataPtr ptr) {
  auto memory_pool = GetMemoryPool(ptr.device);
  if (memory_pool) {
    memory_pool->FreeDataSpace(ptr);
  } else {
    // TODO: The memory pools may be deconstructed earlier than
    // other static objects pointing to storages when the program terminates.
    // In that case, we cannot free the storages via the memory pool.
    // Currently we simply let the system to collect the memory
    // but side effects remain. We shall find a better solution in the future.
    std::call_once(error_suppression_flag, []() {
      HT_LOG_WARN << "It seems the memory pools have been deconstructed before "
                  << "the storage. Will simply let the system to "
                  << "collect the storage, which is not elegant though.";
    });
  }
}

std::ostream& operator<<(std::ostream& os, const DataPtr& data_ptr) {
  os << "DataPtr(address=" << data_ptr.ptr << ", size=" << data_ptr.size
     << ", device=" << data_ptr.device << ")";
  return os;
}

} // namespace hetu
