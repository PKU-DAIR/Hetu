#include "hetu/impl/memory/CPUMemoryPool.h"
#include "hetu/common/except.h"
#include "hetu/core/memory_pool.h"
#include "hetu/impl/stream/CPUStream.h"
#include "hetu/impl/stream/CUDAStream.h"
#include <mutex>
#include <string>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/stat.h> 
#include "hetu/impl/communication/comm_group.h"

namespace hetu {
namespace impl {

namespace {

inline static void batch_sync_dependent_events(
  std::unordered_map<Stream, std::shared_ptr<Event>>& events) {
  for (auto& kv : events)
    kv.second->Sync();
}

} // namespace

CPUMemoryPool::CPUMemoryPool() : MemoryPool(Device(kCPU), "CPUMemPool") {
  _data_ptr_info.reserve(8192);
  _free_on_alloc_stream_fn =
    std::bind(CPUMemoryPool::_FreeOnAllocStream, this, std::placeholders::_1);
  _free_on_join_stream_fn =
    std::bind(CPUMemoryPool::_FreeOnJoinStream, this, std::placeholders::_1);
}

CPUMemoryPool::~CPUMemoryPool() {
  std::lock_guard<std::mutex> lock(_mtx);
  if (_shm_maxium_size > 0) {
    std::string name_str = "/hetu_shared_memory" +  
                           std::to_string(hetu::impl::comm::GetWorldRank());
    shm_unlink(name_str.c_str());
  }
  CPUStream(Stream(Device(kCPU), kJoinStream)).Sync();
}

DataPtr CPUMemoryPool::AllocDataSpace(size_t num_bytes, const Stream& stream,
                                      bool shared_memory) {
  if (num_bytes == 0)
    return DataPtr{nullptr, 0, Device(kCPU), static_cast<uint64_t>(-1)};

  std::lock_guard<std::mutex> lock(_mtx);
  auto alignment = get_data_alignment();
  size_t aligned_num_bytes = DIVUP(num_bytes, alignment) * alignment;
  
  // Currently the allocation on host memory is blocking. Remember to check for
  // the synchronization of allocation stream when freeing or waiting 
  // if the allocation becomes non-blocking.
  void* ptr;
  DataPtrId nid = next_id();
  DataPtr data_ptr;
  if (shared_memory) {
    HT_ASSERT(ShareMemoryReady())
      << "shared_memory is not ready.";
    std::string name_str = "/hetu_shared_memory" +  
                           std::to_string(hetu::impl::comm::GetWorldRank());
    int shm_fd = shm_open(name_str.c_str(), O_RDWR | O_CREAT, 0666);
    ptr = (void*)((char*)base_ptr + _shm_allocated);
    data_ptr = DataPtr(ptr,aligned_num_bytes, 
                       Device(kCPU), nid);
    data_ptr.shmoffset = _shm_allocated;
    data_ptr.shared_memory = shared_memory;
    _shm_allocated += aligned_num_bytes;
    // HT_LOG_WARN << "Alloc shm at ptr:" << ptr << ", cur_size:" << aligned_num_bytes
    //             << "\ntotal_size:" << _shm_allocated << ", max_size:" << _shm_maxium_size;
    close(shm_fd);
  }
  else {
    int err = posix_memalign(&ptr, alignment, aligned_num_bytes);
    HT_BAD_ALLOC_IF(err != 0)
      << "Failed to allocate " << aligned_num_bytes
      << " bytes of host memory. Error: " << strerror(err);
    data_ptr = DataPtr(ptr, aligned_num_bytes, Device(kCPU), nid);
  }
  _allocated += aligned_num_bytes;
  _peak_allocated = MAX(_peak_allocated, _allocated);
  _alloc_cnt++;

  // Note: The `stream` argument might be a non-CPU stream
  // (e.g., allocated for device to host copy).
  // Since we would mark the data space used by that non-CPU stream later,
  // we simply set the `alloc_stream` as join stream here.
  // During the deallocation, the non-CPU stream would be synchronized
  // before the join stream can deallocate the memory.
  Stream alloc_stream =
    stream.device().is_cpu() ? stream : Stream(Device(kCPU), kJoinStream);
  auto insertion = _data_ptr_info.emplace(
    data_ptr.id, 
    CPUDataPtrInfo(data_ptr.ptr, aligned_num_bytes, alloc_stream));
  HT_RUNTIME_ERROR_IF(!insertion.second)
    << "Failed to insert data " << data_ptr << " to info";

  return data_ptr;
}

void CPUMemoryPool::AllocShareMemory(size_t num_bytes, const Stream& stream,
                                     bool realloc) {
  if (num_bytes == 0)
    return;

  //TODO:We should support more elegantly shm manager in the future.
  if ((!realloc) && (_shm_maxium_size != 0)) {
    HT_LOG_WARN << "Ignore Alloc Shm.";
    return;
  }

  std::lock_guard<std::mutex> lock(_mtx);
  auto alignment = get_data_alignment();
  size_t aligned_num_bytes = DIVUP(num_bytes, alignment) * alignment;
  
  std::string name_str = "/hetu_shared_memory" +  
                          std::to_string(hetu::impl::comm::GetWorldRank());
  int shm_fd = shm_open(name_str.c_str(), O_RDWR | O_CREAT, 0666);
  if (_shm_maxium_size == 0) { 
    int64_t allocated_size = aligned_num_bytes;
    if (ftruncate(shm_fd, allocated_size) == -1) {
        close(shm_fd);
        HT_BAD_ALLOC << "Failed to resize shared memory to:" 
                    << allocated_size;
    }
    base_ptr = mmap(NULL, allocated_size, PROT_READ | PROT_WRITE,
              MAP_SHARED, shm_fd, 0);
    _shm_maxium_size = allocated_size;
    HT_LOG_WARN << "Alloc " << allocated_size << " bytes in shm.";
    memset(base_ptr, 0, allocated_size);
    close(shm_fd);
  }
}

bool CPUMemoryPool::ShareMemoryReady() {
  if (_shm_maxium_size > 0) {
    return true;
  }
  return false;
}

DataPtr CPUMemoryPool::BorrowDataSpace(void* ptr, size_t num_bytes,
                                       DataPtrDeleter deleter,
                                       const Stream& stream) {
  HT_VALUE_ERROR_IF(ptr == nullptr)
    << "Borrowing an empty storage is not allowed";
  HT_VALUE_ERROR_IF(!deleter)
    << "Deleter must not be empty when borrowing storages";
  HT_VALUE_ERROR_IF(stream.is_defined() && !stream.is_blocking())
    << "Stream must be blocking if provided";
  if (num_bytes == 0)
    return DataPtr{nullptr, 0, Device(kCPU), static_cast<DataPtrId>(-1)};
  
  std::lock_guard<std::mutex> lock(_mtx);
  // Note: The borrowed memory must be ready, so we use blocking stream here
  DataPtr data_ptr{ptr, num_bytes, Device(kCPU), next_id()};
  Stream alloc_stream = Stream(Device(kCPU), kBlockingStream);
  auto insertion =
    _data_ptr_info.emplace(data_ptr.id,
                           CPUDataPtrInfo(data_ptr.ptr, data_ptr.size,
                                          alloc_stream, std::move(deleter)));
  HT_RUNTIME_ERROR_IF(!insertion.second)
    << "Failed to insert data " << data_ptr << " to info";
  _borrow_cnt++;

  return data_ptr;
}

void CPUMemoryPool::FreeDataSpace(DataPtr data_ptr) {
  if (data_ptr.ptr == nullptr || data_ptr.size == 0)
    return;

  std::lock_guard<std::mutex> lock(_mtx);

  auto it = _data_ptr_info.find(data_ptr.id);
  HT_RUNTIME_ERROR_IF(it == _data_ptr_info.end())
    << "Cannot find data " << data_ptr << " from info";
  auto& alloc_stream = it->second.alloc_stream;
  auto& dependent_events = it->second.dependent_events;

  // Two cases for deallocation:
  // (1) Never used or only used by allocation stream: 
  //     --- enqueue an async task in allocation stream to free directly.
  // (2) Used by streams other than allocation stream:
  //     --- enqueue an async task in join stream to wait for the events.
  // `FreeDataSpace` would first hold the mutex and determine which case to go,
  // and then the free fn holds the mutex later.
  if (dependent_events.empty() ||
      (dependent_events.size() == 1 &&
        dependent_events.begin()->first == alloc_stream)) {
    // If we do not ignore the blocking stream, it will immediately call the 
    // `_free_on_alloc_stream_fn` and try to acquire the mutex, 
    // leading to deadlock.
    if (!alloc_stream.is_blocking()) {
      CPUStream(alloc_stream)
        .EnqueueTask(
          [this, data_ptr]() { this->_free_on_alloc_stream_fn(data_ptr); },
          "FreeOnAllocStream");
    }
  } else {
    CPUStream(Stream(Device(kCPU), kJoinStream))
      .EnqueueTask(
        [this, data_ptr]() { this->_free_on_join_stream_fn(data_ptr); },
        "FreeOnJoinStream");
  }
}

void CPUMemoryPool::_FreeOnAllocStream(CPUMemoryPool* const pool,
                                       DataPtr data_ptr) {
  std::lock_guard<std::mutex> lock(pool->_mtx);
  auto it = pool->_data_ptr_info.find(data_ptr.id);
  HT_RUNTIME_ERROR_IF(it == pool->_data_ptr_info.end())
    << "Cannot find data " << data_ptr << " in from info";
  if (it->second.deleter) {
    it->second.deleter(data_ptr);
  } else {
    if (data_ptr.shared_memory) {
    }
    else
      free(data_ptr.ptr);
  }
  pool->_allocated -= data_ptr.size;
  pool->_data_ptr_info.erase(it);
  pool->_free_cnt++;
}

void CPUMemoryPool::_FreeOnJoinStream(CPUMemoryPool* const pool,
                                      DataPtr data_ptr) {
  // HT_LOG_WARN << "FreeBEGIN:" << data_ptr;
  std::unique_lock<std::mutex> lock(pool->_mtx);
  auto it = pool->_data_ptr_info.find(data_ptr.id);
  HT_RUNTIME_ERROR_IF(it == pool->_data_ptr_info.end())
    << "Cannot find data " << data_ptr << " from info";
  // Note: Currently the allocation on host memory is blocking, 
  // so it is ok if the allocation stream is not marked.
  auto& dependent_events = it->second.dependent_events;
  // We do not need to lock the memory pool when waiting for the events
  lock.unlock();
  batch_sync_dependent_events(dependent_events);
  if (it->second.deleter) {
    it->second.deleter(data_ptr);
  } else {
    if (data_ptr.shared_memory) {
    }
    else
      free(data_ptr.ptr);
  }
  lock.lock();
  pool->_allocated -= data_ptr.size;
  pool->_data_ptr_info.erase(it);
  pool->_free_cnt++;
  // HT_LOG_WARN << "FreeEND:" << data_ptr;
}

void CPUMemoryPool::MarkDataSpaceUsedByStream(DataPtr data_ptr,
                                              const Stream& stream) {
  if (data_ptr.ptr == nullptr || data_ptr.size == 0 || stream.is_blocking())
    return;

  std::lock_guard<std::mutex> lock(_mtx);
  auto it = _data_ptr_info.find(data_ptr.id);
  HT_RUNTIME_ERROR_IF(it == _data_ptr_info.end())
    << "Cannot find data " << data_ptr << " from info";
  auto& dependent_events = it->second.dependent_events;

  if (stream.device().is_cpu()) {
    dependent_events[stream] = std::make_shared<CPUEvent>(false);
    dependent_events[stream]->Record(stream);
  } else if (stream.device().is_cuda()) {
    // CPU data may be used in host to device copy or device to host copy
    dependent_events[stream] =
      std::make_shared<CUDAEvent>(stream.device(), false);
    dependent_events[stream]->Record(stream);
  } else {
    HT_RUNTIME_ERROR << "CPU arrays must be used on cpu or cuda streams. Got "
                     << stream;
    __builtin_unreachable();
  }
  _mark_cnt++;
}

void CPUMemoryPool::MarkDataSpacesUsedByStream(DataPtrList& data_ptrs,
                                               const Stream& stream) {
  if (stream.is_blocking())
    return;

  std::lock_guard<std::mutex> lock(_mtx);
  // share the event
  std::shared_ptr<Event> event = nullptr;
  if (stream.device().is_cpu()) {
    event = std::make_shared<CPUEvent>(false);
    event->Record(stream);
  } else if (stream.device().is_cuda()) {
    event = std::make_shared<CUDAEvent>(stream.device(), false);
    event->Record(stream);
  } else {
    HT_RUNTIME_ERROR << "CPU arrays must be used on cpu or cuda streams. Got "
                     << stream;
    __builtin_unreachable();
  }
  for (auto& data_ptr : data_ptrs) {
    if (data_ptr.ptr == nullptr || data_ptr.size == 0)
      continue;
    auto it = _data_ptr_info.find(data_ptr.id);
    HT_RUNTIME_ERROR_IF(it == _data_ptr_info.end())
      << "Cannot find data " << data_ptr << " from info";
    it->second.dependent_events[stream] = event;
    _mark_cnt++;
  }
}

std::future<void> CPUMemoryPool::WaitDataSpace(DataPtr data_ptr, bool async) {
  if (data_ptr.ptr == nullptr || data_ptr.size == 0)
    return async ? std::async([]() {}) : std::future<void>();

  std::lock_guard<std::mutex> lock(_mtx);
  auto it = _data_ptr_info.find(data_ptr.id);
  HT_RUNTIME_ERROR_IF(it == _data_ptr_info.end())
    << "Cannot find data " << data_ptr << " from info";
  auto& alloc_stream = it->second.alloc_stream;
  auto& dependent_events = it->second.dependent_events;

  std::future<void> future;
  if (dependent_events.empty()) {
    // Note: Currently the allocation on host memory is blocking, 
    // so we can do nothing here.
    if (async) {
      // avoid future error
      future = std::async([]() {});
    }
  } else if (dependent_events.size() == 1 &&
             dependent_events.begin()->first == alloc_stream) {
    std::shared_ptr<Event>& event = dependent_events.begin()->second;
    if (async) {
      future = std::async([event]() mutable { event->Sync(); });
    } else {
      event->Sync();
    }
  } else {
    if (async) {
      // this will make a copy of `dependent_events`
      future = std::async([dependent_events]() mutable {
        batch_sync_dependent_events(dependent_events);
      });
    } else {
      batch_sync_dependent_events(dependent_events);
    }
  }

  return future;
}

void CPUMemoryPool::PrintSummary() {
  HT_LOG_INFO << name() << ": alloc=" << _allocated << " bytes, "
    << "peak_alloc=" << _peak_allocated << " bytes, "
    << "alloc_cnt=" << _alloc_cnt << ", "
    << "borrow_cnt=" << _borrow_cnt << ", "
    << "free_cnt=" << _free_cnt << ", "
    << "mark_cnt=" << _mark_cnt;
}

namespace {

static std::once_flag cpu_memory_pool_register_flag;

struct CPUMemoryPoolRegister {
  CPUMemoryPoolRegister() {
    std::call_once(cpu_memory_pool_register_flag, []() {
      RegisterMemoryPoolCtor(
          Device(kCPU), []() -> std::shared_ptr<MemoryPool> {
            return std::make_shared<CPUMemoryPool>();
          });
    });
  }
};

static CPUMemoryPoolRegister cpu_memory_pool_register;

} // namespace

} // namespace impl
} // namespace hetu
