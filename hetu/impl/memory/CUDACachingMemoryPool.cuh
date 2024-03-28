#pragma once

#include "hetu/impl/memory/CUDAMemoryPool.cuh"
#include "hetu/impl/stream/CUDAStream.h"
#include "hetu/utils/task_queue.h"
#include "hetu/utils/emhash7_hashmap.h"
#include "hetu/utils/robin_hood_hashing.h"
#include <deque>
#include <map>


namespace hetu {
namespace impl {

struct DataPtrLookupTable {
  std::set<DataPtr, bool (*)(const DataPtr& a, const DataPtr& b)> table;

  DataPtrLookupTable(size_t capacity = 1024)
  : table([](const DataPtr& a, const DataPtr& b) -> bool { 
      if(a.size != b.size)
        return a.size < b.size; 
      else
        return a.ptr < b.ptr;
    }) {}
};

class CUDACachingMemoryPool final : public CUDAMemoryPool {
 public:
  CUDACachingMemoryPool(DeviceIndex device_id, size_t _max_split_size);

  ~CUDACachingMemoryPool();

  DataPtr AllocDataSpace(size_t num_bytes,
                         const Stream& stream = Stream()) override;

  DataPtr BorrowDataSpace(void* ptr, size_t num_bytes,
                          DataPtrDeleter deleter,
                          const Stream& stream = Stream()) override;

  void FreeDataSpace(DataPtr data_ptr) override;

  bool EmptyCache() override;

  void MarkDataSpaceUsedByStream(DataPtr data_ptr,
                                 const Stream& stream) override;

  void MarkDataSpacesUsedByStream(DataPtrList& data_ptrs,
                                  const Stream& stream) override;

  std::future<void> WaitDataSpace(DataPtr data_ptr, bool async = true) override;

  void PrintSummary() override;

 private:
  
  // Status after allocation (AllocDataSpace):
  // (1) status = OCCUPIED_BY_ALLOC_STREAM.
  // Status transition of MarkUsedBy (between AllocDataSpace and FreeDataSpace):
  // (1) if only used by the alloc stream, status = OCCUPIED_BY_ALLOC_STREAM;
  // (2) else, status = OCCUPIED_BY_MULTI_STREAMS.
  // Status transition of FreeDataSpace (freed by user):
  // (1) if status == OCCUPIED_BY_ALLOC_STREAM, then status = AVAILABLE_FOR_ALLOC_STREAM;
  // (2) else (status == OCCUPIED_BY_MULTI_STREAMS), then status = UNAVAILABLE_UNTIL_FREE;
  // Status transition of WatchEvent (freed by system):
  // (1) if status == UNAVAILABLE_UNTIL_FREE, then status = AVAILABLE_FOR_ALL_STREAM;
  enum class OccupationStatus : int8_t {
    OCCUPIED_BY_ALLOC_STREAM = 0,
    OCCUPIED_BY_MULTI_STREAMS,
    UNAVAILABLE_UNTIL_FREE,
    AVAILABLE_FOR_ALLOC_STREAM,
    AVAILABLE_FOR_ALL_STREAM
  };

  friend std::ostream& operator<<(std::ostream& os, const OccupationStatus& status) {
    switch (status) {
      case OccupationStatus::OCCUPIED_BY_ALLOC_STREAM:
        os << "OCCUPIED_BY_ALLOC_STREAM";
        break;
      case OccupationStatus::OCCUPIED_BY_MULTI_STREAMS:
        os << "OCCUPIED_BY_MULTI_STREAMS";
        break;
      case OccupationStatus::UNAVAILABLE_UNTIL_FREE:
        os << "UNAVAILABLE_UNTIL_FREE";
        break;
      case OccupationStatus::AVAILABLE_FOR_ALLOC_STREAM:
        os << "AVAILABLE_FOR_ALLOC_STREAM";
        break;
      case OccupationStatus::AVAILABLE_FOR_ALL_STREAM:
        os << "AVAILABLE_FOR_ALL_STREAM";
        break;
    }
    return os;
  }

  // NOTE: 从PyTorch借鉴的20MB"剩余量"限额
  const size_t kMaxInternalFragment = 20971520;
  const size_t kMinSplitRemaining = 1048576; // 1MB

  // Pack malloc requests in buffer, which aims at using "split ptr" feature
  // to reduce cudaMalloc invoke times.
  const size_t kMallocMinBuffer = 2097152; 
  const size_t kMallocRoundUp = 2097152; 
  const size_t kMallocLargeBuffer = 10485760;
  size_t max_split_size{209715200}; // in bytes

  // Record stream info of an allocated pointer.
  struct CudaDataPtrInfo {
    void* ptr;
    size_t num_bytes;
    PackedStreamId alloc_stream;
    std::unordered_set<PackedStreamId> used_streams;
    DataPtrDeleter deleter;
    DataPtrId id;
    
    OccupationStatus status;
    mempool_clock_t alloc_at;
    mempool_clock_t free_at; // free_at should be set only when inserting ptr table
    uint32_t free_event_cnt;

    DataPtrLookupTable* cached_pool{nullptr}; // 只有当其是unallocated的时候才具有
    std::shared_ptr<CudaDataPtrInfo> prev{nullptr};
    std::shared_ptr<CudaDataPtrInfo> next{nullptr};

    CudaDataPtrInfo(void* ptr_, size_t num_bytes_, const Stream& alloc_stream_,
                    mempool_clock_t alloc_at_, DataPtrId id_, DataPtrDeleter deleter_ = {})
    : ptr(ptr_),
      num_bytes(num_bytes_),
      alloc_stream(alloc_stream_.pack()),
      alloc_at(alloc_at_), 
      id(id_),
      deleter(deleter_),
      free_at(0), 
      free_event_cnt(0) {
      if (!alloc_stream_.is_blocking())
        used_streams.insert(alloc_stream);
      status = OccupationStatus::OCCUPIED_BY_ALLOC_STREAM;
    }

    inline void insert_used_stream(PackedStreamId used_stream) {
      if (used_stream != alloc_stream) {
        used_streams.insert(used_stream);
        status = OccupationStatus::OCCUPIED_BY_MULTI_STREAMS;
      }
    }

    inline bool allocated() const noexcept {
      return alloc_at > free_at;
    }

    inline bool is_split() const noexcept {
      return prev != nullptr || next != nullptr;
    }

    // Note: can_free() means all related entries are unallocated
    inline bool can_free() const noexcept {
      if (allocated()) {
        return false;
      }
      if (is_split()) {
        auto tmp = prev;
        while (tmp != nullptr) {
          if (tmp->allocated()) {
            return false;
          }
          tmp = prev->prev;
        }
        tmp = next;
        while (tmp != nullptr) {
          if (tmp->allocated()) {
            return false;
          }
          tmp = next->next;
        }
      }
      return true;
    }
  
    inline void refresh() {
      used_streams.clear();
      status = OccupationStatus::AVAILABLE_FOR_ALL_STREAM;
      alloc_at = 0;
      free_at = 0;
      free_event_cnt = 0;
    }
  };

  bool FindAvailable(size_t num_bytes,
                     DataPtrLookupTable& lookup_table,
                     DataPtr& ret,
                     bool remove_if_find = true);

  void InsertAvailable(const DataPtr& data_ptr,
                       DataPtrLookupTable& lookup_table);

  bool ReleaseOversized(DataPtrLookupTable& lookup_table, 
                        size_t request_size);

  bool ReleaseAll(DataPtrLookupTable& lookup_table,
                  bool maybe_allocated = true);

  size_t GetAlignedMallocSize(size_t request_size);

  bool AllocNewPtr(void* &ptr, size_t size);

  void WatchEvents();

  bool ShouldSplit(size_t allocated_size, size_t request_size);

  DataPtrLookupTable* TryMerge(std::shared_ptr<CudaDataPtrInfo>&, DataPtrLookupTable*);

  // Info of all data pointers
  // If one pointer is cached in the lookup table of peculiar stream, it will also have record in here. 
  // NOTE: 所有allocated & unallocated指针都在这里保存记录
  emhash7::HashMap<DataPtrId, std::shared_ptr<CudaDataPtrInfo>> _data_ptr_info;
  // Cached data pointers that are available for specific streams
  emhash7::HashMap<PackedStreamId, std::unique_ptr<DataPtrLookupTable>> _available_for_single_stream;
  // Cached data pointers that are available for all stream
  std::unique_ptr<DataPtrLookupTable> _available_for_all_streams;
  // Events to indicate whether marked usages have finished
  emhash7::HashMap<PackedStreamId, std::deque<std::tuple<std::unique_ptr<CUDAEvent>, DataPtrId>>> _free_events;

  size_t _allocated{0};
  size_t _reserved{0}; // allocated size + cached size
  size_t _peak_reserved{0};
  uint64_t _alloc_cnt{0};
  uint64_t _cuda_malloc_cnt{0};
  uint64_t _free_cnt{0};
  uint64_t _mark_cnt{0};
};

} // namespace impl
} // namespace hetu
