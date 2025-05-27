#pragma once

#include "hetu/common/macros.h"
#include "hetu/core/device.h"
#include "hetu/core/stream.h"
#include <memory>
#include <mutex>
#include <future>

namespace hetu {

using DataPtrId = uint64_t;
using mempool_clock_t = uint64_t;

struct DataPtr {
  void* ptr;
  size_t size;
  Device device;
  DataPtrId id; // id provided by the memory pool
  bool shared_memory;
  int64_t shmoffset;

  // 2024.9.17更新data_ptr的辅助信息
  // debug use
  bool is_new_malloc = false; 
  DataPtrId split_from_id = -1;

  DataPtr() = default;
  DataPtr(const DataPtr &a) = default;

  // construct a search key
  DataPtr(size_t s, void* _ptr): size(s), ptr(_ptr) {}

  DataPtr(void* _ptr, size_t _size, const Device& _device, DataPtrId _id)
  : ptr(_ptr), size(_size), device(_device), id(_id), split_from_id(_id), shared_memory(false) {}
};

using DataPtrList = std::vector<DataPtr>;

std::ostream& operator<<(std::ostream&, const DataPtr&);

using DataPtrDeleter = std::function<void(DataPtr)>;

class MemoryPool {
 public:
  MemoryPool(Device device, std::string name)
  : _device{std::move(device)}, _name{std::move(name)} {}

  virtual DataPtr AllocDataSpace(size_t num_bytes,
                                 const Stream& stream = Stream(),
                                 bool shared_memory = false) = 0;
  
  virtual void AllocShareMemory(size_t num_bytes,
                                const Stream& stream = Stream(),
                                bool realloc = false) = 0;
  
  virtual bool ShareMemoryReady() = 0;

  virtual DataPtr BorrowDataSpace(void* ptr, size_t num_bytes,
                                  DataPtrDeleter deleter,
                                  const Stream& stream = Stream()) = 0;

  virtual void FreeDataSpace(DataPtr data_ptr) = 0;

  virtual void EmptyCache() {}

  virtual void MarkDataSpaceUsedByStream(DataPtr data_ptr,
                                         const Stream& stream) = 0;

  virtual void MarkDataSpacesUsedByStream(DataPtrList& data_ptrs,
                                          const Stream& stream) = 0;

  virtual std::future<void> WaitDataSpace(DataPtr data_ptr,
                                          bool async = true) = 0;

  virtual void PrintSummary() = 0;
  
  const Device& device() const {
    return _device;
  }

  const std::string& name() const {
    return _name;
  }

protected:
  inline DataPtrId next_id() {
    // the caller should hold the mutex
    return _next_id++;
  }

  inline mempool_clock_t next_clock() {
    // the caller should hold the mutex
    // spare clock = 0
    return ++_clock;
  }
  
  const Device _device;
  const std::string _name;
  mempool_clock_t _clock{0};
  DataPtrId _next_id{0};
  std::mutex _mtx;
};

void RegisterMemoryPoolCtor(const Device& device,
                            std::function<std::shared_ptr<MemoryPool>()> ctor);

std::shared_ptr<MemoryPool> GetMemoryPool(const Device& device);

DataPtr AllocFromMemoryPool(const Device& device, size_t num_bytes,
                            const Stream& stream = Stream(),
                            bool shared_memory = false);

void AllocShareMemoryFromMemoryPool(const Device& device, size_t num_bytes,
                                    const Stream& stream = Stream());

bool ShareMomoryReadyOfMemoryPool(const Device& device);

DataPtr BorrowToMemoryPool(const Device& device, void* ptr, size_t num_bytes,
                           DataPtrDeleter deleter);

void FreeToMemoryPool(DataPtr ptr);

} // namespace hetu
