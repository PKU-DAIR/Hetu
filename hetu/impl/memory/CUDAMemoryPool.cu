#include "hetu/impl/memory/CUDAMemoryPool.cuh"
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
  return "CUDAMemPool(" + std::to_string(static_cast<int>(device_id)) + ")";
}

} // namespace

CUDAMemoryPool::CUDAMemoryPool(DeviceIndex device_id)
: MemoryPool(Device(kCUDA, device_id), _make_name(device_id)) {
  _free_stream_watcher.reset(
    new TaskQueue("free_stream_watcher_of_" + name(), 8));
}

CUDAMemoryPool::~CUDAMemoryPool() {
  _free_stream_watcher->Shutdown();
  _free_stream_watcher = nullptr;
}

DataPtr CUDAMemoryPool::AllocDataSpace(size_t num_bytes, const Stream& stream) {
  HT_VALUE_ERROR_IF(!stream.device().is_cuda())
    << "Cuda arrays must be allocated on cuda streams. Got " << stream;
  if (num_bytes == 0)
    return {nullptr, 0, device()};
  
  InitDeviceMemoryPoolOnce(device().index());
  
  auto alignment = get_data_alignment();
  size_t aligned_num_bytes = DIVUP(num_bytes, alignment) * alignment;

  CUDADeviceGuard guard(device().index());
  void* ptr;
  CudaMallocAsync(&ptr, aligned_num_bytes, CUDAStream(stream));
  DataPtr data_ptr{ptr, aligned_num_bytes, device()};
  _allocated += aligned_num_bytes;
  _peak_allocated = MAX(_peak_allocated, _allocated);
  
  std::lock_guard<std::mutex> lock(_mtx);
  auto insertion =
    _data_ptr_info.emplace(ptr, CudaDataPtrInfo(aligned_num_bytes, stream));
  HT_RUNTIME_ERROR_IF(!insertion.second)
    << "Failed to insert data " << data_ptr << " to info";
  
  return data_ptr;
}

void CUDAMemoryPool::BorrowDataSpace(DataPtr, DataPtrDeleter) {
  // TODO: support customized deleter when freeing the memory 
  // so that we can support borrowing on CUDA devices
  HT_NOT_IMPLEMENTED << "Borrowing memory on " << device()
                     << " is not supported yet";
}

void CUDAMemoryPool::FreeDataSpace(DataPtr data_ptr) {
  if (data_ptr.ptr == nullptr || data_ptr.size == 0)
    return;
  std::lock_guard<std::mutex> lock(_mtx);

  auto it = _data_ptr_info.find(data_ptr.ptr);
  HT_RUNTIME_ERROR_IF(it == _data_ptr_info.end())
    << "Cannot find data " << data_ptr << " from info";
  auto& alloc_stream = it->second.alloc_stream;
  auto& used_streams = it->second.used_streams;

  // TODO: support customized deleter when freeing the memory
  CUDADeviceGuard guard(device().index());
  Stream free_stream = alloc_stream;
  if (used_streams.empty() ||
      (used_streams.size() == 1 && *used_streams.begin() == alloc_stream)) {
    CUDA_CALL(cudaFreeAsync(data_ptr.ptr, CUDAStream(alloc_stream)));
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
    CudaFreeAsync(data_ptr.ptr, CUDAStream(join_stream));
    free_stream = join_stream;
  }

  // Note: If the stream to free the memory is never going to be used,
  // the driver may reserve the memory and try to find any possibility
  // to re-use the memory on the same stream, leading to memory leakage
  // (in particular on the join stream). Therefore, we use a background watcher
  // to wait for the streams that are used to free.
  _free_stream_flags[free_stream.stream_index()] = 1;
  _free_stream_watcher->Enqueue([this]() {
    for (StreamIndex stream_id = 0; stream_id < HT_NUM_STREAMS_PER_DEVICE;
         stream_id++) {
      if (this->_free_stream_flags[stream_id] != 0) {
        auto d = this->device();
        CUDAEvent sync_event(d, false);
        sync_event.Record(Stream(d, stream_id));
        sync_event.Sync();
        this->_free_stream_flags[stream_id] = 0;
      }
    }
  });

  _allocated -= data_ptr.size;
  _data_ptr_info.erase(it);
}

void CUDAMemoryPool::MarkDataSpaceUsedByStream(DataPtr data_ptr,
                                               const Stream& stream) {
  if (data_ptr.ptr == nullptr || data_ptr.size == 0 || stream.is_blocking())
    return;
  HT_VALUE_ERROR_IF(!stream.device().is_cuda())
    << "Cuda arrays must be used on cuda streams. Got " << stream;

  std::lock_guard<std::mutex> lock(_mtx);
  auto it = _data_ptr_info.find(data_ptr.ptr);
  HT_RUNTIME_ERROR_IF(it == _data_ptr_info.end())
    << "Cannot find data " << data_ptr << " from info";
  it->second.used_streams.insert(stream);
}

void CUDAMemoryPool::MarkDataSpacesUsedByStream(DataPtrList& data_ptrs,
                                                const Stream& stream) {
  if (stream.is_blocking())
    return;
  HT_VALUE_ERROR_IF(!stream.device().is_cuda())
    << "Cuda arrays must be used on cuda streams. Got " << stream;

  std::lock_guard<std::mutex> lock(_mtx);
  for (auto& data_ptr : data_ptrs) {
    if (data_ptr.ptr == nullptr || data_ptr.size == 0)
      continue;
    auto it = _data_ptr_info.find(data_ptr.ptr);
    HT_RUNTIME_ERROR_IF(it == _data_ptr_info.end())
      << "Cannot find data " << data_ptr << " from info";
    it->second.used_streams.insert(stream);
  }
}

std::future<void> CUDAMemoryPool::WaitDataSpace(DataPtr data_ptr, bool async) {
  if (data_ptr.ptr == nullptr || data_ptr.size == 0)
    return async ? std::async([]() {}) : std::future<void>();

  std::unique_lock<std::mutex> lock(_mtx);
  auto it = _data_ptr_info.find(data_ptr.ptr);
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

namespace {

static std::once_flag cuda_memory_pool_register_flag;

struct CUDAMemoryPoolRegister {
  CUDAMemoryPoolRegister() {
    std::call_once(cuda_memory_pool_register_flag, []() {
      int32_t num_devices;
      CudaGetDeviceCount(&num_devices);
      for (int32_t i = 0; i < num_devices; i++) {
        auto pool =
          std::make_shared<CUDAMemoryPool>(static_cast<DeviceIndex>(i));
        RegisterMemoryPool(pool);
      }
    });
  }
};

static CUDAMemoryPoolRegister cudu_memory_pool_register;

} // namespace

} // namespace impl
} // namespace hetu
