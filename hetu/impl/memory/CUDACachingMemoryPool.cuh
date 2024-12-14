#pragma once

#include "hetu/impl/memory/CUDAMemoryPool.cuh"
#include "hetu/impl/memory/memory_manager.h"
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

  DataPtrLookupTable()
  : table([](const DataPtr& a, const DataPtr& b) -> bool { 
      if (a.size != b.size)
        return a.size < b.size; 
      else
        return a.ptr < b.ptr;
    }) {}
};

struct AvailableEvent {
  DataPtrId id;
  mempool_clock_t free_at;
  PackedStreamId stream;
  std::shared_ptr<CUDAEvent> event;

  // construct a search key
  AvailableEvent(mempool_clock_t _free_at, DataPtrId _id): id(_id), free_at(_free_at) {}

  AvailableEvent(DataPtrId _id, mempool_clock_t _free_at, PackedStreamId _stream, const std::shared_ptr<CUDAEvent>& _event)
  : id(_id), free_at(_free_at), stream(_stream), event(_event) {}
};

struct AvailableEventLookupTable {
  std::set<AvailableEvent, bool (*)(const AvailableEvent& a, const AvailableEvent& b)> table;

  AvailableEventLookupTable()
  : table([](const AvailableEvent& a, const AvailableEvent& b) -> bool { 
      if (a.free_at != b.free_at)
        return a.free_at < b.free_at; 
      else
        return a.id < b.id;
    }) {}
};

bool AllocAfterFreeFromCUDACache(const Device& device, void*& ptr, size_t size);

void ProfileAfterEmptyAllCUDACache(const Device& device);

class CUDACachingMemoryPool final : public CUDAMemoryPool {
 public:
  CUDACachingMemoryPool(DeviceIndex device_id, size_t _max_split_size, size_t _max_internal_fragment_size, size_t _pre_allocate_size);

  ~CUDACachingMemoryPool();

  size_t GetCurrAllocated() {
    return _allocated;
  }

  size_t GetCurrReserved() {
    return _reserved;
  }

  size_t GetPeakReserved() {
    return _peak_reserved;
  }

  DataPtr AllocDataSpace(size_t num_bytes,
                         const Stream& stream = Stream()) override;

  DataPtr BorrowDataSpace(void* ptr, size_t num_bytes,
                          DataPtrDeleter deleter,
                          const Stream& stream = Stream()) override;

  void FreeDataSpace(DataPtr data_ptr) override;

  void MarkDataSpaceUsedByStream(DataPtr data_ptr,
                                 const Stream& stream) override;

  void MarkDataSpacesUsedByStream(DataPtrList& data_ptrs,
                                  const Stream& stream) override;

  std::future<void> WaitDataSpace(DataPtr data_ptr, bool async = true) override;

  void PrintSummary() override;

  void EmptyCache() override;

  friend bool AllocAfterFreeFromCUDACache(const Device& device, void*& ptr, size_t size);

  friend void ProfileAfterEmptyAllCUDACache(const Device& device);

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
  // (2) else (status == AVAILABLE_FOR_ALLOC_STREAM), then status = AVAILABLE_FOR_ALL_STREAM;
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

  const size_t kMinSplitRemaining = 1048576; // 1MB

  // Pack malloc requests in buffer, which aims at using "split ptr" feature
  // to reduce cudaMalloc invoke times.
  const size_t kMallocMinBuffer = 2097152; 
  const size_t kMallocRoundUp = 2097152; 
  const size_t kMallocLargeBuffer = 10485760;
  size_t max_split_size{209715200}; // in bytes
  size_t max_internal_fragment_size{20971520}; // NOTE: 从PyTorch借鉴的20MiB剩余量限额
  size_t pre_allocate_size{0}; // 提前分配的显存大小

  // Record stream info of an allocated pointer.
  struct CudaDataPtrInfo {
    void* ptr;
    size_t num_bytes;
    PackedStreamId alloc_stream;
    std::unordered_set<PackedStreamId> used_streams;
    DataPtrDeleter deleter;
    DataPtrId id;
    DataPtrId split_from_id;
    
    OccupationStatus status;
    mempool_clock_t alloc_at;
    mempool_clock_t free_at; // free_at should be set only when inserting ptr table
    uint32_t multi_stream_free_event_cnt;

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
      split_from_id(id_),
      free_at(0), 
      multi_stream_free_event_cnt(0) {
      if (!alloc_stream_.is_blocking())
        used_streams.insert(alloc_stream);
      status = OccupationStatus::OCCUPIED_BY_ALLOC_STREAM;
    }

    inline void insert_used_stream(PackedStreamId used_stream) {
      used_streams.insert(used_stream);
      if (used_streams.size() > 1) {
        status = OccupationStatus::OCCUPIED_BY_MULTI_STREAMS;
      }
    }

    inline bool allocated() const noexcept {
      return alloc_at > free_at;
    }

    inline bool is_split() const noexcept {
      return prev != nullptr || next != nullptr;
    }
  
    inline void refresh() {
      used_streams.clear();
      status = OccupationStatus::AVAILABLE_FOR_ALL_STREAM;
      alloc_at = 0;
      free_at = 0;
      multi_stream_free_event_cnt = 0;
    }
  };

  bool FindAvailable(size_t num_bytes,
                     DataPtrLookupTable& lookup_table,
                     DataPtr& ret,
                     bool remove_if_find = true);

  void InsertAvailable(const DataPtr& data_ptr,
                       DataPtrLookupTable& lookup_table);
                      
  void MoveAvailable(std::shared_ptr<CudaDataPtrInfo>& info);

  size_t GetAlignedMallocSize(size_t request_size);

  bool AllocPtr(void*& ptr, size_t size);

  void FreePtr(void* ptr);

  void ReleaseAll();

  bool ReleaseAndAlloc(void*& ptr, size_t request_size);

  bool WaitUntilAlloc(void*& ptr, size_t request_size);

  bool IsEmpty(DataPtrLookupTable* lookup_table, bool ignore_split = true);

  void AddAvailableEvent(DataPtrId data_ptr_id, std::shared_ptr<AvailableEvent>& available_event);

  void DeleteAvailableEvent(DataPtrId data_ptr_id);

  void WatchEvents();

  bool ShouldSplit(size_t allocated_size, size_t request_size);

  DataPtrLookupTable* TryMerge(std::shared_ptr<CudaDataPtrInfo>& data_info, DataPtrLookupTable* table);

  // Info of all data pointers
  // If one pointer is cached in the lookup table of peculiar stream, it will also have record in here. 
  // NOTE: 所有allocated & unallocated指针都在这里保存记录
  emhash7::HashMap<DataPtrId, std::shared_ptr<CudaDataPtrInfo>> _data_ptr_info;
  // Mapping of data ptr id to the single stream event
  // NOTE: 服务于_single_stream_free_events
  emhash7::HashMap<DataPtrId, std::shared_ptr<AvailableEvent>> _available_event_info;
  // Cached data pointers that are available for specific streams
  emhash7::HashMap<PackedStreamId, std::unique_ptr<DataPtrLookupTable>> _available_for_single_stream;
  // Cached data pointers that are available for all stream
  std::unique_ptr<DataPtrLookupTable> _available_for_all_streams;
  // Events to indicate whether marked usages have finished
  emhash7::HashMap<PackedStreamId, std::unique_ptr<std::deque<std::tuple<std::unique_ptr<CUDAEvent>, DataPtrId>>>> _multi_stream_free_events;
  // Events to indicate whether a data ptr is available to all streams
  emhash7::HashMap<PackedStreamId, std::unique_ptr<AvailableEventLookupTable>> _single_stream_free_events;
  // 2024.9.23 Update: add memory manager to avoid cudaMalloc and cudaFree
  std::shared_ptr<MemoryManager> _memory_manager = nullptr;

  size_t _allocated{0};
  size_t _reserved{0}; // allocated size + cached size
  size_t _peak_reserved{0};
  uint64_t _alloc_cnt{0};
  uint64_t _free_cnt{0};
  uint64_t _mark_cnt{0};
};

} // namespace impl
} // namespace hetu
