#include "hetu/common/except.h"
#include "hetu/impl/memory/CUDAStreamOrderedMemoryPool.cuh"
#include "hetu/impl/stream/CUDAStream.h"
#include "hetu/impl/stream/CPUStream.h"
#include "hetu/impl/utils/cuda_utils.h"
#include <mutex>

namespace hetu {
namespace impl {

using namespace hetu::cuda;

namespace {

static std::once_flag mem_pool_init_flags[HT_MAX_GPUS_COMPILE_TIME];

static void InitDeviceMemoryPool(int32_t device_id) {
  hetu::cuda::CUDADeviceGuard guard(device_id);
  cudaMemPool_t cuda_mem_pool;
  CUDA_CALL(cudaDeviceGetDefaultMemPool(&cuda_mem_pool, device_id));
  // do not release
  uint64_t threshold = UINT64_MAX;
  CUDA_CALL(cudaMemPoolSetAttribute(
    cuda_mem_pool, cudaMemPoolAttrReleaseThreshold, &threshold));
  // enable re-use
  int enable = 1;
  CUDA_CALL(cudaMemPoolSetAttribute(
    cuda_mem_pool, cudaMemPoolReuseFollowEventDependencies, &enable));
  CUDA_CALL(cudaMemPoolSetAttribute(
    cuda_mem_pool, cudaMemPoolReuseAllowOpportunistic, &enable));
  CUDA_CALL(cudaMemPoolSetAttribute(
    cuda_mem_pool, cudaMemPoolReuseAllowInternalDependencies, &enable));
}

inline static void InitDeviceMemoryPoolOnce(int32_t device_id) {
  std::call_once(mem_pool_init_flags[device_id], InitDeviceMemoryPool,
                 device_id);
}

inline static std::string _make_name(DeviceIndex device_id) {
  return "CUDAStreamOrderedMemoryPool(" +
    std::to_string(static_cast<int>(device_id)) + ")";
}

} // namespace

CUDAStreamOrderedMemoryPool::CUDAStreamOrderedMemoryPool(DeviceIndex device_id)
: CUDAMemoryPool(device_id, _make_name(device_id)) {
  _data_ptr_info.reserve(8192);
  _free_stream_watcher.reset(
    new TaskQueue("free_stream_watcher_of_" + name(), 8));
  _free_stream_flags.resize(HT_NUM_STREAMS_PER_DEVICE, 0);
}

CUDAStreamOrderedMemoryPool::~CUDAStreamOrderedMemoryPool() {
  _free_stream_watcher->Shutdown();
  _free_stream_watcher = nullptr;
}

DataPtr CUDAStreamOrderedMemoryPool::AllocDataSpace(size_t num_bytes,
                                                    const Stream& stream,
                                                    bool shared_memory) {
  HT_VALUE_ERROR_IF(!stream.device().is_cuda())
    << "Cuda arrays must be allocated on cuda streams. Got " << stream;
  if (num_bytes == 0)
    return DataPtr{nullptr, 0, device(), static_cast<uint64_t>(-1)};
  
  std::lock_guard<std::mutex> lock(_mtx);
  InitDeviceMemoryPoolOnce(device().index());
  mempool_clock_t alloc_at = next_clock();
  
  auto alignment = get_data_alignment();
  size_t aligned_num_bytes = DIVUP(num_bytes, alignment) * alignment;

  CUDADeviceGuard guard(device().index());
  void* ptr;
  CudaMallocAsync(&ptr, aligned_num_bytes, CUDAStream(stream));
  DataPtr data_ptr{ptr, aligned_num_bytes, device(), next_id()};
  _allocated += aligned_num_bytes;
  _peak_allocated = MAX(_peak_allocated, _allocated);
  _alloc_cnt++;
  
  auto insertion = _data_ptr_info.emplace(
    data_ptr.id, 
    CudaDataPtrInfo(data_ptr.ptr, aligned_num_bytes, stream, alloc_at));
  HT_RUNTIME_ERROR_IF(!insertion.second)
    << "Failed to insert data " << data_ptr << " to info";
  
  return data_ptr;
}

void CUDAStreamOrderedMemoryPool::AllocShareMemory(size_t num_bytes, 
                                                   const Stream& stream,
                                                   bool realloc) {
  HT_NOT_IMPLEMENTED << "CUDA MemoryPool doesn't support share memory currently.";
}

bool CUDAStreamOrderedMemoryPool::ShareMemoryReady() {
  // Since GPU currently doesn't support share memory, this is always true; 
  return true;
}

DataPtr CUDAStreamOrderedMemoryPool::BorrowDataSpace(void*, size_t,
                                                     DataPtrDeleter,
                                                     const Stream&) {
  // TODO: support customized deleter when freeing the memory 
  // so that we can support borrowing on CUDA devices
  HT_NOT_IMPLEMENTED << name()
                     << ": Borrowing CUDA memory is not supported yet";
  __builtin_unreachable();
}

void CUDAStreamOrderedMemoryPool::FreeDataSpace(DataPtr data_ptr) {
  if (data_ptr.ptr == nullptr || data_ptr.size == 0)
    return;
  std::lock_guard<std::mutex> lock(_mtx);
  mempool_clock_t free_at = next_clock();

  auto it = _data_ptr_info.find(data_ptr.id);
  HT_RUNTIME_ERROR_IF(it == _data_ptr_info.end())
    << "Cannot find data " << data_ptr << " from info";
  auto& alloc_stream = it->second.alloc_stream;
  auto& used_streams = it->second.used_streams;

  it->second.free_at = free_at;
  // TODO: support customized deleter when freeing the memory
  CUDADeviceGuard guard(device().index());
  Stream free_stream = alloc_stream;
  if (used_streams.empty() ||
      (used_streams.size() == 1 && *used_streams.begin() == alloc_stream)) {
    CudaFreeAsync(data_ptr.ptr, CUDAStream(alloc_stream));
  } else {
    // Note: In case we forget to mark the data used by the allocation stream,
    // which is unlikely though.
    if (!alloc_stream.is_blocking() &&
        used_streams.find(alloc_stream) == used_streams.end()) {
      used_streams.insert(alloc_stream);
    }
    Stream join_stream(data_ptr.device, kJoinStream);
    for (auto& used_stream : used_streams) {
      CUDAEvent event(data_ptr.device, false);
      event.Record(used_stream);
      event.Block(join_stream);
    }
    CUDA_CALL(cudaFreeAsync(data_ptr.ptr, CUDAStream(join_stream)));
    free_stream = join_stream;
  }

  // Note: If the stream to free the memory is never going to be used,
  // the driver may reserve the memory and try to find any possibility
  // to re-use the memory on the same/other stream, leading to memory leakage
  // (in particular on the join stream). Therefore, we use a background watcher
  // to wait for the streams that are used to free.
  _free_stream_flags[free_stream.stream_index()] = 1;
  auto free_stream_idx = free_stream.stream_index();
  _free_stream_watcher->Enqueue([this, free_stream_idx]() {
    if (this->_free_stream_flags[free_stream_idx] != 0) {
      auto d = this->device();
      CUDAEvent sync_event(d, false);
      sync_event.Record(Stream(d, free_stream_idx));
      sync_event.Sync();
      this->_free_stream_flags[free_stream_idx] = 0;
    }
  });

  _allocated -= data_ptr.size;
  _data_ptr_info.erase(it);
  _free_cnt++;
}

void CUDAStreamOrderedMemoryPool::MarkDataSpaceUsedByStream(
  DataPtr data_ptr, const Stream& stream) {
  if (data_ptr.ptr == nullptr || data_ptr.size == 0 || stream.is_blocking())
    return;
  HT_VALUE_ERROR_IF(!stream.device().is_cuda())
    << "Cuda arrays must be used on cuda streams. Got " << stream;

  std::lock_guard<std::mutex> lock(_mtx);
  auto it = _data_ptr_info.find(data_ptr.id);
  HT_RUNTIME_ERROR_IF(it == _data_ptr_info.end())
    << "Cannot find data " << data_ptr << " from info";
  it->second.used_streams.insert(stream);
  _mark_cnt++;
}

void CUDAStreamOrderedMemoryPool::MarkDataSpacesUsedByStream(
  DataPtrList& data_ptrs, const Stream& stream) {
  if (stream.is_blocking())
    return;
  HT_VALUE_ERROR_IF(!stream.device().is_cuda())
    << "Cuda arrays must be used on cuda streams. Got " << stream;

  std::lock_guard<std::mutex> lock(_mtx);
  for (auto& data_ptr : data_ptrs) {
    if (data_ptr.ptr == nullptr || data_ptr.size == 0)
      continue;
    auto it = _data_ptr_info.find(data_ptr.id);
    HT_RUNTIME_ERROR_IF(it == _data_ptr_info.end())
      << "Cannot find data " << data_ptr << " from info";
    it->second.used_streams.insert(stream);
    _mark_cnt++;
  }
}

std::future<void> CUDAStreamOrderedMemoryPool::WaitDataSpace(DataPtr data_ptr,
                                                             bool async) {
  if (data_ptr.ptr == nullptr || data_ptr.size == 0)
    return async ? std::async([]() {}) : std::future<void>();

  std::unique_lock<std::mutex> lock(_mtx);
  auto it = _data_ptr_info.find(data_ptr.id);
  HT_RUNTIME_ERROR_IF(it == _data_ptr_info.end())
    << "Cannot find data " << data_ptr << " from info";
  auto& alloc_stream = it->second.alloc_stream;
  auto& used_streams = it->second.used_streams;
  lock.unlock();

  // TODO: Avoid synchronizing allocation and used streams again 
  // when freeing the memory. However, remember that it necessitates 
  // tracking whether each async waits has completed or not.
  Stream wait_stream;
  if (used_streams.empty() ||
      (used_streams.size() == 1 && *used_streams.begin() == alloc_stream)) {
    if (alloc_stream.is_blocking())
      return async ? std::async([]() {}) : std::future<void>();
    else
      wait_stream = alloc_stream;
  } else {
    // Note: In case we forget to mark the data used by the allocation stream,
    // which is unlikely though.
    if (!alloc_stream.is_blocking() &&
        used_streams.find(alloc_stream) == used_streams.end()) {
      used_streams.insert(alloc_stream);
    }
    Stream join_stream(data_ptr.device, kJoinStream);
    for (auto& used_stream : used_streams) {
      CUDAEvent event(data_ptr.device, false);
      event.Record(used_stream);
      event.Block(join_stream);
    }
    wait_stream = join_stream;
  }

  if (async) {
    return std::async([wait_stream]() { CUDAStream(wait_stream).Sync(); });
  } else {
    CUDAStream(wait_stream).Sync();
    return std::future<void>();
  }
}

void CUDAStreamOrderedMemoryPool::PrintSummary() {
  HT_LOG_INFO << name() << ": alloc=" << _allocated << " bytes, "
    << "peak_alloc=" << _peak_allocated << " bytes, "
    << "alloc_cnt=" << _alloc_cnt << ", "
    << "free_cnt=" << _free_cnt << ", "
    << "mark_cnt=" << _mark_cnt;
}

namespace {

static std::once_flag cuda_memory_pool_register_flag;

struct CUDAStreamOrderedMemoryPoolRegister {
  CUDAStreamOrderedMemoryPoolRegister() {
    std::call_once(cuda_memory_pool_register_flag, []() {
      // Memory pools are lazily constructed, so we do not need to
      // get device count here.
      for (int32_t i = 0; i < HT_MAX_DEVICE_INDEX; i++) {
        RegisterMemoryPoolCtor(
          Device(kCUDA, i), [i]() -> std::shared_ptr<MemoryPool> {
            return std::make_shared<CUDAStreamOrderedMemoryPool>(i);
          });
      }
    });
  }
};

// static CUDAStreamOrderedMemoryPoolRegister cudu_memory_pool_register;

} // namespace

} // namespace impl
} // namespace hetu
