#pragma once

#include "hetu/impl/memory/CUDAMemoryPool.cuh"
#include "hetu/utils/task_queue.h"

namespace hetu {
namespace impl {

class CUDAStreamOrderedMemoryPool final : public CUDAMemoryPool {
 public:
  CUDAStreamOrderedMemoryPool(DeviceIndex device_id);

  ~CUDAStreamOrderedMemoryPool();

  DataPtr AllocDataSpace(size_t num_bytes, const Stream& stream = Stream(),
                         bool shared_memory = false);
  
  void AllocShareMemory(size_t num_bytes, const Stream& stream = Stream(),
                        bool realloc = false);
  
  bool ShareMemoryReady();

  DataPtr BorrowDataSpace(void* ptr, size_t num_bytes, DataPtrDeleter deleter, const Stream& stream = Stream());

  void FreeDataSpace(DataPtr data_ptr);

  void MarkDataSpaceUsedByStream(DataPtr data_ptr, const Stream& stream);

  void MarkDataSpacesUsedByStream(DataPtrList& data_ptrs, const Stream& stream);

  std::future<void> WaitDataSpace(DataPtr data_ptr, bool async = true);

  void PrintSummary();

 private:
  struct CudaDataPtrInfo {
    void* ptr;
    size_t num_bytes;
    Stream alloc_stream;
    mempool_clock_t alloc_at;
    mempool_clock_t free_at;
    std::unordered_set<Stream> used_streams;

    CudaDataPtrInfo(void* ptr_, size_t num_bytes_, Stream alloc_stream_,
                    mempool_clock_t alloc_at_)
    : ptr(ptr_),
      num_bytes(num_bytes_),
      alloc_stream{std::move(alloc_stream_)},
      alloc_at(alloc_at_), 
      free_at(0) {}
  };

  std::unordered_map<uint64_t, CudaDataPtrInfo> _data_ptr_info;
  std::vector<int> _free_stream_flags{HT_NUM_STREAMS_PER_DEVICE, 0};
  std::unique_ptr<TaskQueue> _free_stream_watcher;

  size_t _allocated{0};
  size_t _peak_allocated{0};
  uint64_t _alloc_cnt{0};
  uint64_t _free_cnt{0};
  uint64_t _mark_cnt{0};
};

} // namespace impl
} // namespace hetu
