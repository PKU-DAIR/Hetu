#include "hetu/impl/memory/CUDACachingMemoryPool.cuh"
#include "hetu/impl/memory/memory_manager.h"
#include "hetu/impl/stream/CUDAStream.h"
#include "hetu/impl/communication/nccl_comm_group.h"
#include "hetu/impl/utils/cuda_utils.h"
#include "hetu/common/timing.h"
#include <mutex>
#include <cstdlib>
#include <string>
#include <stdexcept>

namespace hetu {
namespace impl {

// TODO: Release lock and re-acquire to hide the latency of cudaMalloc
bool CUDACachingMemoryPool::AllocPtr(void*& ptr, size_t size) {
  if (_memory_manager) {
    return _memory_manager->Malloc(&ptr, size);
  } else {
    hetu::cuda::CUDADeviceGuard guard(device().index());
    cudaError_t ret = CudaMallocTry(&ptr, size);
    if (ret == cudaSuccess) {
      return true;
    } else if (ret == cudaErrorMemoryAllocation) {
      // ignore and clear the memory allocation error
      (void) cudaGetLastError();
      return false;
    } else {
      HT_RUNTIME_ERROR << "cudaMalloc failed with rare reason";
    }
  }
}

void CUDACachingMemoryPool::FreePtr(void* ptr) {
  if (_memory_manager) {
    _memory_manager->Free(ptr);
  } else {
    CudaFree(ptr);
  }
}

bool AllocAfterFreeFromCUDACache(const Device& device, void*& ptr, size_t size) {
  auto caching_mempool = std::dynamic_pointer_cast<CUDACachingMemoryPool>(GetMemoryPool(device));
  std::lock_guard<std::mutex> lock(caching_mempool->_mtx);
  return caching_mempool->AllocPtr(ptr, size) || caching_mempool->WaitUntilAlloc(ptr, size);
}

void FreeFromCUDACache(const Device& device, void* ptr) {
  // corresponding to AllocAfterFreeFromCUDACache
  // the caller should hold the mutex
  // should only be used in param buffer
  auto caching_mempool = std::dynamic_pointer_cast<CUDACachingMemoryPool>(GetMemoryPool(device));
  caching_mempool->FreePtr(ptr);
}

void ProfileAfterEmptyAllCUDACache(const Device& device) {
  auto caching_mempool = std::dynamic_pointer_cast<CUDACachingMemoryPool>(GetMemoryPool(device));
  std::lock_guard<std::mutex> lock(caching_mempool->_mtx);
  HT_LOG_INFO << device << "******* [Before Empty] Cuda Caching Mempool ******* ";
  caching_mempool->PrintSummary();
  caching_mempool->EmptyCache();
  // 彻底清空cache后
  // single stream available table一定已经全部清空
  // all stream available table只可能剩下split出来的
  if (caching_mempool->max_split_size > 0) {
    HT_ASSERT(caching_mempool->IsEmpty(caching_mempool->_available_for_all_streams.get()))
      << "The all stream available table shouldn't have any unsplitted entry now";
  } else {
    HT_ASSERT(caching_mempool->IsEmpty(caching_mempool->_available_for_all_streams.get()))
      << "The all stream available table shouldn't have any entry now";
  }
  for (auto& kv : caching_mempool->_available_for_single_stream) {
    HT_ASSERT(caching_mempool->IsEmpty(kv.second.get(), false))
      << "The single stream available table shouldn't have any entry now";
  }
  HT_LOG_INFO << device << "******* [After Empty] Cuda Caching Mempool ******* ";
  caching_mempool->PrintSummary();
}

static std::string _make_name(DeviceIndex device_id) {
  return "CUDACachingMemPool(" + std::to_string(static_cast<int>(device_id)) + ")";
}

CUDACachingMemoryPool::CUDACachingMemoryPool(DeviceIndex device_id, size_t _max_split_size, size_t _max_internal_fragment_size, size_t _pre_allocate_size)
: CUDAMemoryPool(device_id, _make_name(device_id)),
  max_split_size(_max_split_size) ,
  max_internal_fragment_size(_max_internal_fragment_size),
  pre_allocate_size(_pre_allocate_size) {
  _data_ptr_info.reserve(8192);
  _available_for_single_stream.reserve(HT_NUM_STREAMS_PER_DEVICE); 
  _available_for_all_streams.reset(new DataPtrLookupTable());
  if (pre_allocate_size > 0) {
    void* ptr;
    TIK(pre_allocate);
    hetu::cuda::CUDADeviceGuard guard(device_id);
    cudaError_t err = CudaMallocTry(&ptr, pre_allocate_size);
    TOK(pre_allocate);
    if (err != cudaSuccess) {
      HT_RUNTIME_ERROR << "cudaMalloc failed when tring to pre-allocate " << pre_allocate_size << " byte for the CUDACachingMemoryPool on device " << static_cast<int32_t>(device_id)
        << ", the error is: " << cudaGetErrorString(err)
        << ", you may need to adjust HETU_PRE_ALLOCATE_SIZE_MB";
    } else {
      HT_LOG_INFO << "Pre-allocate " << pre_allocate_size << " byte for the CUDACachingMemoryPool on device " << static_cast<int32_t>(device_id)
        << ", the time cost is " << COST_MSEC(pre_allocate) << " ms";
    }
    _memory_manager = std::make_shared<MemoryManager>(ptr, pre_allocate_size);
  }
}

CUDACachingMemoryPool::~CUDACachingMemoryPool() {
  // TODO: free the memory instead of let the OS collect them
}

// 分配num bytes的显存
// 先会从available table中找
// 没有匹配项再AllocPtr
DataPtr CUDACachingMemoryPool::AllocDataSpace(size_t num_bytes,
                                              const Stream& stream,
                                              bool shared_memory) {                         
  HT_VALUE_ERROR_IF(!stream.device().is_cuda())
    << "Cuda arrays must be allocated on cuda streams. Got " << stream;
  if (num_bytes == 0)
    return DataPtr{nullptr, 0, device(), static_cast<DataPtrId>(-1)};

  std::lock_guard<std::mutex> lock(_mtx);
  WatchEvents();

  PackedStreamId packed_stream_id = stream.pack();
  uint64_t alloc_at = next_clock(); // Update the `alloc_at` clock, which is later than the previous `free_at`
  auto alignment = get_data_alignment(); // 256 bytes
  size_t aligned_num_bytes = DIVUP(num_bytes, alignment) * alignment;
  DataPtr data_ptr;
  DataPtr curr_stream_data_ptr;
  DataPtr all_stream_data_ptr;
  bool found_curr_stream_available = false;
  bool found_all_stream_available = false;
  int32_t reuse_flag = -1; // -1 means cannot reuse; 0 means reuse curr stream cache; 1 means reuse all stream cache

  DataPtrLookupTable* target_table = nullptr;
  DataPtrLookupTable* curr_stream_table = nullptr;
  auto info_it = _data_ptr_info.end();

  // Find among data spaces that are available only for this stream.
  // blocking stream并不具有专属于自己的available table
  if (!stream.is_blocking()) {
    auto table_it = _available_for_single_stream.find(packed_stream_id);
    if (table_it == _available_for_single_stream.end()) { 
      auto insertion = _available_for_single_stream.emplace(packed_stream_id, std::make_unique<DataPtrLookupTable>());
      HT_RUNTIME_ERROR_IF(!insertion.second)
        << "Failed to insert lookup table to " << stream;
      table_it = insertion.first;
    } 
    // 已经创建过available table了
    else {
      found_curr_stream_available = FindAvailable(aligned_num_bytes, 
                                                  *(table_it->second), 
                                                  curr_stream_data_ptr,
                                                  false);
    }
    curr_stream_table = table_it->second.get();
  }
  // Find among data spaces that are available for all streams.
  found_all_stream_available = FindAvailable(aligned_num_bytes, 
                                             *_available_for_all_streams, 
                                             all_stream_data_ptr,
                                             false); 
 
  // 如果curr stream和all stream的两个cache中都有可复用项
  if (found_curr_stream_available && found_all_stream_available) {
    // 目前的策略是优先选取best fit的
    // 在同等best fit的条件下再优先选取curr stream的
    // TODO: better strategy
    if (curr_stream_data_ptr.size <= all_stream_data_ptr.size) 
      reuse_flag = 0;
    else
      reuse_flag = 1;
  }
  // 只有curr stream的cache可以复用
  else if (found_curr_stream_available) {
    reuse_flag = 0;
  }
  // 只有all stream的cache可以复用
  else if (found_all_stream_available) {
    reuse_flag = 1;
  }

  if (reuse_flag != -1) {
    if (reuse_flag == 0) {
      data_ptr = curr_stream_data_ptr;
      info_it = _data_ptr_info.find(curr_stream_data_ptr.id);
      HT_RUNTIME_ERROR_IF(info_it == _data_ptr_info.end())
        << "Cannot find curr stream cached data " << curr_stream_data_ptr << " from info";
      HT_ASSERT(info_it->second->cached_pool == curr_stream_table)
        << "Cache pool error";
      HT_ASSERT(info_it->second->status == OccupationStatus::AVAILABLE_FOR_ALLOC_STREAM)
        << "Info status should be " << OccupationStatus::AVAILABLE_FOR_ALLOC_STREAM
        << ", but it is actually " << info_it->second->status;
      target_table = curr_stream_table;
    }
    else if (reuse_flag == 1) {
      data_ptr = all_stream_data_ptr;
      info_it = _data_ptr_info.find(all_stream_data_ptr.id);
      HT_RUNTIME_ERROR_IF(info_it == _data_ptr_info.end())
        << "Cannot find all stream cached data " << all_stream_data_ptr << " from info";
      HT_ASSERT(info_it->second->cached_pool == _available_for_all_streams.get())
        << "Cache pool error";
      HT_ASSERT(info_it->second->status == OccupationStatus::AVAILABLE_FOR_ALL_STREAM)
        << "Info status should be " << OccupationStatus::AVAILABLE_FOR_ALL_STREAM
        << ", but it is actually " << info_it->second->status;
      target_table = _available_for_all_streams.get();
    }
    // reuse要修正info的部分信息
    info_it->second->alloc_stream = packed_stream_id;
    info_it->second->used_streams.clear();
    if (!stream.is_blocking())
      info_it->second->used_streams.insert(packed_stream_id);
    info_it->second->status = OccupationStatus::OCCUPIED_BY_ALLOC_STREAM;
    info_it->second->alloc_at = alloc_at;
    info_it->second->cached_pool = nullptr;
    // 从table中删除该entry
    auto entry_it = target_table->table.find(data_ptr);
    HT_ASSERT(entry_it != target_table->table.end())
      << "Cannot find the entry of " << data_ptr << " in the target table";
    target_table->table.erase(entry_it);
  }
  // Cannot find any avaiable memory to re-use, then AllocPtr from system.
  // *只有这种情况会AllocPtr并将新分配的data ptr放入到info中
  else {
    void* ptr;
    // Now use aligned_num_bytes
    // size_t malloc_size = GetAlignedMallocSize(aligned_num_bytes); 
    size_t malloc_size = aligned_num_bytes;
    // Check whether the memory limitation has been reached. 
    // If yes, we shall free/re-use some cached memories on other streams.
    if (AllocPtr(ptr, malloc_size)
        // Wait until we can release some cached ptrs (maybe all cached ptrs) and accomplish the allocation.
        || WaitUntilAlloc(ptr, malloc_size)) {
      // ------ create new data ptr place 1 ------ 
      data_ptr = DataPtr{ptr, malloc_size, device(), next_id()};
      data_ptr.is_new_malloc = true;
      // mempool debug use    
      // HT_LOG_INFO << "[Create] AllocPtr new " << data_ptr;
      _reserved += malloc_size;
      _peak_reserved = MAX(_peak_reserved, _reserved);
      auto new_info = std::make_shared<CudaDataPtrInfo>(data_ptr.ptr, 
                                                        malloc_size, 
                                                        stream, 
                                                        alloc_at, 
                                                        data_ptr.id);                                               
      // 此时AllocPtr出来的新的(cuda) data ptr还不具有cache的table
      new_info->cached_pool = nullptr; // 默认值其实就是nullptr（这里只是为了强调一下）
      auto insertion = _data_ptr_info.emplace(data_ptr.id, new_info);
      HT_RUNTIME_ERROR_IF(!insertion.second)
        << "Failed to insert data " << data_ptr << " to info";
      info_it = insertion.first;
    } 
    // 清空cache后依然无法分配
    else {
      HT_RUNTIME_ERROR
        << "Try to allocate " << static_cast<double>(malloc_size) / 1024 << " KiB on GPU " << device() 
        << ", but trigger cuda OOM error"
        << ", mempool reserved: " << static_cast<double>(_reserved) / (1024 * 1024 * 1024) << " GiB"
        << ", mempool allocated: " << static_cast<double>(_allocated) / (1024 * 1024 * 1024) << " GiB"
        << ", please set environment variable HETU_MAX_SPLIT_SIZE_MB smaller"
        << ", if you find reserved - allocated is too high"
        << ", and if mempool reserved is much lower than the actual GPU memory"
        << ", that means there are too many external fragments or some other memory consumptions like cuda/nccl context!";
    }
  }

  // Now we have prepared the target ptr. Do splitting if we need and update statistics.
  // 此时的data ptr是一个不在任何available table中的东西
  // 且相应的info已经设置好了alloc_at、alloc_stream
  if (ShouldSplit(data_ptr.size, aligned_num_bytes)) {
    // split的只有可能是刚从table中取出来reuse的条目
    // 因为AllocPtr分配的会不多不少
    // Make the high address part a new cached ptr
    auto cur_info = info_it->second;
    void* remaining_ptr = data_ptr.ptr + aligned_num_bytes;
    size_t remaining_size = data_ptr.size - aligned_num_bytes;
    size_t new_id = next_id();
    // 修正之前的data ptr和info中的(cuda) data ptr 
    data_ptr.size = aligned_num_bytes;
    cur_info->num_bytes = aligned_num_bytes; 
    HT_ASSERT(cur_info->split_from_id == data_ptr.split_from_id)
      << "split_from_id of cur info should remain the same";
    // ------ create new data ptr place 2 ------ 
    auto remaining_data_ptr = DataPtr{remaining_ptr, remaining_size, device(), new_id};
    remaining_data_ptr.split_from_id = data_ptr.split_from_id; // 始终往一个去溯源
    // 将remaining data ptr插入table
    HT_ASSERT(target_table)
      << "Target table is a nullptr";
    InsertAvailable(remaining_data_ptr, *target_table);
    // 将remaining (cuda) data ptr插入info
    auto new_info = std::make_shared<CudaDataPtrInfo>(remaining_ptr, 
                                                      remaining_size, 
                                                      stream, 
                                                      0, 
                                                      new_id);
    new_info->status = target_table == _available_for_all_streams.get() ? 
                       OccupationStatus::AVAILABLE_FOR_ALL_STREAM : OccupationStatus::AVAILABLE_FOR_ALLOC_STREAM;
    new_info->cached_pool = target_table;
    new_info->split_from_id = remaining_data_ptr.split_from_id;
    auto info_insertion = _data_ptr_info.emplace(new_id, new_info);   
    HT_RUNTIME_ERROR_IF(!info_insertion.second)
      << "Failed to insert splitted (cuda) data ptr " << remaining_data_ptr << " to info"; 
    // 用链表连接split的info
    new_info->prev = cur_info;
    new_info->next = cur_info->next;
    if (cur_info->next != nullptr)
      cur_info->next->prev = new_info;
    cur_info->next = new_info;
    // data ptr如果有single stream free event
    // 那么新split出来的event和之前的应该一致
    if (reuse_flag == 0) {
      auto event_info_it = _available_event_info.find(data_ptr.id);
      HT_ASSERT(event_info_it != _available_event_info.end())
        << "Cannot find the info of single stream free event towards data ptr id " << data_ptr.id; 
      auto available_event = event_info_it->second;
      auto new_available_event = std::make_shared<AvailableEvent>(new_id, 
                                                                  available_event->free_at,
                                                                  available_event->stream,
                                                                  available_event->event);
      AddAvailableEvent(new_id, new_available_event);
    }
  }

  // 返回的data ptr如果是复用当前stream的available table的
  // 那么其将会使single stream free event失效
  if (reuse_flag == 0) {
    DeleteAvailableEvent(data_ptr.id);
  }
  _allocated += data_ptr.size;
  _alloc_cnt++;
  HT_LOG_TRACE << "ptr: " << data_ptr.id << ", alloc: " << data_ptr.size << ", stream: " << stream << ", is new malloc: " << data_ptr.is_new_malloc;
  return data_ptr;
}

void CUDACachingMemoryPool::AllocShareMemory(size_t num_bytes, 
                                             const Stream& stream,
                                             bool realloc) {
  HT_NOT_IMPLEMENTED << "CUDA MemoryPool doesn't support share memory currently.";
}

bool CUDACachingMemoryPool::ShareMemoryReady() {
  // Since GPU currently doesn't support share memory, this is always true; 
  return true;
}

// deprecated for now
[[deprecated]]
size_t CUDACachingMemoryPool::GetAlignedMallocSize(size_t request_size) {
  if (request_size < kMallocMinBuffer) {
    return kMallocMinBuffer;
  } else if (request_size < kMallocLargeBuffer) {
    return kMallocLargeBuffer;
  } else {
    return DIVUP(request_size, kMallocRoundUp) * kMallocRoundUp;
  }
  return request_size;
}

bool CUDACachingMemoryPool::ShouldSplit(size_t size, size_t request_size) {
  HT_ASSERT(size >= request_size)
    << "Size error";
  return size <= max_split_size && (size - request_size) >= kMinSplitRemaining;
}

// 从available table中找到某一个满足num bytes的条目
// 如果找不到则返回false
bool CUDACachingMemoryPool::FindAvailable(size_t num_bytes, 
                                          DataPtrLookupTable& lookup_table, 
                                          DataPtr& reuse_data_ptr,
                                          bool remove_if_find) {
  // the caller should hold the mutex
  auto it = lookup_table.table.lower_bound(DataPtr(num_bytes, nullptr)); 
  if (it != lookup_table.table.end()) { 
    auto info_it = _data_ptr_info.find(it->id);
    HT_RUNTIME_ERROR_IF(info_it == _data_ptr_info.end()) 
      << "Cannot find one ptr's info";
    // 目前available table中的应该全是unallocated的条目
    HT_ASSERT(!info_it->second->allocated())
      << "Allocated status error";
    HT_ASSERT(it->size >= num_bytes)
      << "Size error";
    if (it->size != num_bytes) {
      size_t remaining = it->size - num_bytes;
      // 只有两种情况允许使用cache的条目
      // 1、该条目小于等于max_split_size
      // 后续会split该条目
      // 一部分用于新分配的而空余的那部分重新插入
      // 2、想分配的size要比max_split_size还大且剩余的内部碎片较少
      // 后续会直接占用整个条目且不进行split
      if (it->size > max_split_size && remaining > max_internal_fragment_size) {
        // num_bytes > max_split_size, so we will directly allocate this large
        // ptr to request without splitting. But we need to limit the remaining 
        // size to avoid large internal fragment.
        return false;
      }
    }
    reuse_data_ptr = (*it);
    // 删除条目
    if (remove_if_find) {
      lookup_table.table.erase(it);
    }
    return true;
  }
  // 找不到任何一个可以容纳的下的条目 
  return false;
}

// 直接insert即可
// 默认size从小到大排序
// the caller should hold the mutex
void CUDACachingMemoryPool::InsertAvailable(const DataPtr& data_ptr, 
                                            DataPtrLookupTable& lookup_table) {
  auto result = lookup_table.table.emplace(data_ptr);
  HT_RUNTIME_ERROR_IF(!result.second)
    << "Failed to insert key " << data_ptr.size << " to lookup table";
}

// 当某个data ptr在single stream available table但实际已经all stream available时
// 可以考虑移动到all stream available table或其余split出的块儿所在的table
// 调用该函数时需要保证data ptr已经真正意义上all stream available了
// *即其对应的single stream event都已经被释放干净了
// the caller should hold the mutex
void CUDACachingMemoryPool::MoveAvailable(std::shared_ptr<CudaDataPtrInfo>& info) {
  HT_ASSERT(info->status == OccupationStatus::AVAILABLE_FOR_ALLOC_STREAM) 
    << "Info status should be " << OccupationStatus::AVAILABLE_FOR_ALLOC_STREAM
    << ", but found " << info->status;
  HT_ASSERT(info->cached_pool)
    << "Info cached pool shouldn't be nullptr";
  auto data_ptr = DataPtr{info->ptr, info->num_bytes, device(), info->id};
  auto table_it = info->cached_pool->table.find(data_ptr);
  HT_ASSERT(table_it != info->cached_pool->table.end())
    << "Cannot find " << data_ptr << " in the original single stream available table";
  info->cached_pool->table.erase(table_it);
  info->cached_pool = nullptr;
  // 此时info即将插入新的table
  // 需要及时地进行merge操作
  // 优先考虑放到all stream available table中cache住
  DataPtrLookupTable* target_table = _available_for_all_streams.get();
  info->refresh();
  if (info->is_split())
    target_table = TryMerge(info, target_table);
  // Meta information of data_ptr might have change in TryMerge
  data_ptr.ptr = info->ptr;
  data_ptr.size = info->num_bytes;
  data_ptr.split_from_id = info->split_from_id;
  info->cached_pool = target_table;
  InsertAvailable(data_ptr, *target_table);
}

bool CUDACachingMemoryPool::IsEmpty(DataPtrLookupTable* lookup_table, bool ignore_split) {
  for (auto& entry : lookup_table->table) {
    auto info_it = _data_ptr_info.find(entry.id);
    HT_RUNTIME_ERROR_IF(info_it == _data_ptr_info.end())
      << "Cannot find CudaDataPtrInfo for stream's cached ptr";
    HT_ASSERT(!info_it->second->allocated())
      << "Allocate status error";
    if (info_it->second->is_split()) {
      HT_ASSERT(entry.size <= max_split_size)
        << "The splitted size of " << entry.size << " is out-of-range";
      if (ignore_split) {
        continue;
      }
      return false;
    }
    return false;
  }
  return true;
}

// Free all pointers in the all stream available lookup table currently.
// Caller should hold the mutex.
void CUDACachingMemoryPool::ReleaseAll() {
  auto& lookup_table = _available_for_all_streams->table;                                          
  auto it = lookup_table.begin();
  while (it != lookup_table.end()) {
    auto info_it = _data_ptr_info.find(it->id);
    HT_RUNTIME_ERROR_IF(info_it == _data_ptr_info.end())
      << "Cannot find CudaDataPtrInfo for stream's cached ptr";
    HT_ASSERT(!info_it->second->allocated())
      << "Allocate status error";
    // split的无法被释放
    if (info_it->second->is_split()) {
      it++;
      continue;
    }
    FreePtr(it->ptr);
    _reserved -= it->size;
    it = lookup_table.erase(it);
    _data_ptr_info.erase(info_it); 
  }
}

// Free some pointer in the all stream available lookup table to satisfy the request_size.
// It will try to satisfy the request_size with minimum number of ptrs. 
// Caller should hold the mutex.
bool CUDACachingMemoryPool::ReleaseAndAlloc(void*& ptr, size_t request_size) {
  // We only release oversize pointer. If max_split_size_mb is not specified,
  // no pointers will be regarded as oversize.   
  auto& lookup_table = _available_for_all_streams->table;               
  if (lookup_table.empty()) {
    return AllocPtr(ptr, request_size); 
  }                          
  DataPtr tmp_key = {request_size > max_split_size ? request_size : max_split_size, nullptr};
  // Find if there are any ptr larger than request_size
  auto it = lookup_table.lower_bound(tmp_key);
  // 从lower bound的条目往上找最小的可以满足要求的
  // 由于在FindAvailable会对条目进行删除因此都是unallocated的
  // 且由于保证了max_split_size因此其实际上都是可以free的
  if (it != lookup_table.end()) {
    auto info_it = _data_ptr_info.find(it->id);
    HT_RUNTIME_ERROR_IF(info_it == _data_ptr_info.end())
      << "Cannot find CudaDataPtrInfo for stream's cached ptr";
    HT_ASSERT(!info_it->second->allocated())
      << "Allocate status error";
    FreePtr(it->ptr);
    _reserved -= it->size;
    lookup_table.erase(it);
    _data_ptr_info.erase(info_it); 
    HT_ASSERT(AllocPtr(ptr, request_size))
      << "Can't alloc request_size";
    return true;
  }
  // 没有任何能直接满足的条目
  // 那么从大到小释放显存
  it--;
  while (1) {
    auto info_it = _data_ptr_info.find(it->id);
    HT_RUNTIME_ERROR_IF(info_it == _data_ptr_info.end())
      << "Cannot find CudaDataPtrInfo for stream's cached ptr";
    HT_ASSERT(!info_it->second->allocated())
      << "Allocate status error";
    // all stream available table中的碎片
    // 要么就是处于占用状态
    // 要么就是在别的single stream available table中且available event还没结束
    // 因为WatchEvent会在其之前被调用
    // 能够保证当前时钟下可以放到all stream available中的一定都放进去了
    if (info_it->second->is_split()) {
      HT_ASSERT(it->size <= max_split_size)
        << "The splitted size of " << *it << " is out-of-range";
      if (it != lookup_table.begin()) {
        it--;
        continue;
      }
      return AllocPtr(ptr, request_size);
    }
    FreePtr(it->ptr);
    _reserved -= it->size;
    it = lookup_table.erase(it);
    _data_ptr_info.erase(info_it); 
    if (AllocPtr(ptr, request_size)) {
      return true;
    }
    if (it != lookup_table.begin()) {
      it--;
    }
    // 清空all stream available table中所有可以直接FreePtr的
    // 仍然无法AllocPtr
    // 那么只能返回false放弃分配
    else {
      return AllocPtr(ptr, request_size);
    }
  }
}

// *It will hang the whole system!
// Wait and free some pointer in the all stream available lookup table
// until we satisfy the request_size.
// Caller should hold the mutex.
bool CUDACachingMemoryPool::WaitUntilAlloc(void*& ptr, size_t request_size) {
  // Use a thread per stream to async wait is not a good idea.
  // Different stream may have interference because of cache-cache movement.
  while (1) {
    WatchEvents();
    bool successfully_release = ReleaseAndAlloc(ptr, request_size);
    if (successfully_release) {
      return true;
    }
    // mempool debug use
    // HT_LOG_INFO << "Need to wait and free until we can allocate";
    // 停等3毫秒
    std::this_thread::sleep_for(std::chrono::milliseconds(3));
    bool all_free = true;
    for (auto& kv : _multi_stream_free_events) {
      if (!kv.second->empty()) {
        all_free = false;
        break;
      }
    }
    if (all_free) {
      for (auto& kv : _single_stream_free_events) {
        if (!kv.second->table.empty()) {
          all_free = false;
          break;
        }
      }
    }
    // 全部清空了都还不满足request size
    // 那么不得不OOM了
    if (all_free) {
      // 理论上此时应该所有cache的table都被清空了
      // 除了那些split出来无法free的显存碎片
      if (max_split_size > 0) {
        HT_ASSERT(IsEmpty(_available_for_all_streams.get()))
          << "The all stream available table shouldn't have any unsplitted entry now";
        for (auto& kv : _available_for_single_stream) {
          HT_ASSERT(IsEmpty(kv.second.get()))
            << "The single stream available table shouldn't have any unsplitted entry now";
        }
      } 
      // 无split情形
      else {
        HT_ASSERT(IsEmpty(_available_for_all_streams.get(), false))
          << "The all stream available table shouldn't have any entry now";
        for (auto& kv : _available_for_single_stream) {
          HT_ASSERT(IsEmpty(kv.second.get(), false))
            << "The single stream available table shouldn't have any entry now";
        }
      }
      // 考虑清理nccl context
      // hetu::impl::comm::EmptyNCCLCache();
      return AllocPtr(ptr, request_size); 
    }
  }
}

// *Note this will hang the whole system!
// Caller should hold the mutex.
void CUDACachingMemoryPool::EmptyCache() {
  CudaDeviceSynchronize();
  WatchEvents();
  ReleaseAll();
}

// 直接bind外部的内存
// 不会走cache的那一套逻辑
// 即不会被放到任何available table中
DataPtr CUDACachingMemoryPool::BorrowDataSpace(void* ptr, size_t num_bytes,
                                               DataPtrDeleter deleter,
                                               const Stream& stream) {
  HT_VALUE_ERROR_IF(ptr == nullptr)
    << "Borrowing an empty storage is not allowed";
  HT_VALUE_ERROR_IF(!deleter)
    << "Deleter must not be empty when borrowing storages";
  if (num_bytes == 0)
    return DataPtr{nullptr, 0, device(), static_cast<DataPtrId>(-1)};

  std::lock_guard<std::mutex> lock(_mtx);
  WatchEvents();

  // Note: The borrowed memory must be ready, so we use blocking stream here
  DataPtr data_ptr{ptr, num_bytes, device(), next_id()};
  Stream borrow_stream = stream.is_defined() ? stream : Stream(device(), kBlockingStream);
  uint64_t borrow_at = next_clock();
  auto insertion = _data_ptr_info.emplace(data_ptr.id,
                                          std::make_shared<CudaDataPtrInfo>(data_ptr.ptr, data_ptr.size,
                                          borrow_stream, borrow_at, data_ptr.id, std::move(deleter)));
  HT_RUNTIME_ERROR_IF(!insertion.second)
    << "Failed to insert data " << data_ptr << " to info";

  _reserved += num_bytes;
  _peak_reserved = MAX(_peak_reserved, _reserved);
  _allocated += num_bytes;
  _alloc_cnt++;
  return data_ptr;
}

void CUDACachingMemoryPool::FreeDataSpace(DataPtr data_ptr) {
  data_ptr.is_new_malloc = false; // 先将new_malloc属性清空
  if (data_ptr.ptr == nullptr || data_ptr.size == 0)
    return;
  std::lock_guard<std::mutex> lock(_mtx);
  mempool_clock_t free_at = next_clock();

  auto info_it = _data_ptr_info.find(data_ptr.id);
  HT_RUNTIME_ERROR_IF(info_it == _data_ptr_info.end())
    << "Cannot find data " << data_ptr << " from info";
  auto info = info_it->second;

  HT_ASSERT(info->ptr == data_ptr.ptr)
    << "Find info: ptr = " << info->ptr
    << ", but data ptr = " << data_ptr; 
  HT_ASSERT(info->num_bytes == data_ptr.size)
      << "Find info: size = " << info->num_bytes 
      << ", but data ptr = " << data_ptr;

  // for borrow data, we currently adopt method 1
  // the exec graph running & switching memory profiling will be more accurate
  // note we will eventually use method 2

  // method 1: for borrow data we free it directly
  // we should block the used streams here
  /*
  if (info->deleter) {
    // Stream::unpack(info->alloc_stream).Sync();
    auto& used_streams = info->used_streams;
    for (auto s_id : used_streams) {
      Stream::unpack(s_id).Sync();
    }
    info->deleter(data_ptr);
    _data_ptr_info.erase(info_it);
    _reserved -= info->num_bytes;
    _allocated -= info->num_bytes;
    _free_cnt++;
    return;
  }
  */

  // method 2: move borrow data actual free to WatchEvents()
  // we only record the free event here
  if (info->deleter) {
    auto& used_streams = info->used_streams;
    if (used_streams.empty()) {
      info->deleter(data_ptr);
      _data_ptr_info.erase(info_it);
      _reserved -= info->num_bytes;
      _allocated -= info->num_bytes;  
      _free_cnt++;    
      return;
    }
    info->status = OccupationStatus::UNAVAILABLE_UNTIL_FREE;
    for (auto s_id : used_streams) {
      auto event = std::make_unique<CUDAEvent>(data_ptr.device, false);
      event->Record(Stream::unpack(s_id));
      if (_multi_stream_free_events.find(s_id) == _multi_stream_free_events.end())
        _multi_stream_free_events.emplace(s_id, std::make_unique<std::deque<std::tuple<std::unique_ptr<CUDAEvent>, DataPtrId>>>());
      _multi_stream_free_events[s_id]->emplace_back(std::make_tuple(std::move(event), data_ptr.id));
    }
    info->multi_stream_free_event_cnt += used_streams.size();
    return;
  }

  // stream独占
  if (info->status == OccupationStatus::OCCUPIED_BY_ALLOC_STREAM) {
    DataPtrLookupTable* target_table = nullptr;
    auto stream = Stream::unpack(info->alloc_stream);
    info->free_at = free_at;
    _free_cnt++;
    _allocated -= info->num_bytes;
    // blocking stream上释放的东西所有stream之后都能用
    if (stream.is_blocking()) {
      info->status = OccupationStatus::AVAILABLE_FOR_ALL_STREAM;
      target_table = _available_for_all_streams.get();
    }
    // 其余情况只能先放到single stream available table
    else {
      info->status = OccupationStatus::AVAILABLE_FOR_ALLOC_STREAM;
      target_table = _available_for_single_stream[info->alloc_stream].get();
      // 插入event并在之后的WatchEvent中处理
      // 来让AVAILABLE_FOR_ALLOC_STREAM到AVAILABLE_FOR_ALL_STREAM进行转换
      auto event = std::make_shared<CUDAEvent>(data_ptr.device, false);
      event->Record(stream); 
      auto available_event = std::make_shared<AvailableEvent>(info->id, 
                                                              free_at, 
                                                              info->alloc_stream, 
                                                              event);
      AddAvailableEvent(info->id, available_event);
    }
    // 考虑是否可以merge
    if (info->is_split())
      target_table = TryMerge(info, target_table); 
    // Meta information of data_ptr might have change in TryMerge
    data_ptr.size = info->num_bytes;
    data_ptr.ptr = info->ptr;
    data_ptr.split_from_id = info->split_from_id; 
    info->cached_pool = target_table;
    InsertAvailable(data_ptr, *target_table); 
    HT_ASSERT(data_ptr.id == info->id)
      << "The data ptr id and the info id are mismatched";
  } 
  // 当前被很多stream占用
  // 需要插入event等到所有stream都完成
  else if (info->status == OccupationStatus::OCCUPIED_BY_MULTI_STREAMS) {
    info->status = OccupationStatus::UNAVAILABLE_UNTIL_FREE;
    auto& used_streams = info->used_streams;
    for (auto s_id : used_streams) {
      auto event = std::make_unique<CUDAEvent>(data_ptr.device, false);
      event->Record(Stream::unpack(s_id)); 
      if (_multi_stream_free_events.find(s_id) == _multi_stream_free_events.end())
        _multi_stream_free_events.emplace(s_id, std::make_unique<std::deque<std::tuple<std::unique_ptr<CUDAEvent>, DataPtrId>>>());
      _multi_stream_free_events[s_id]->emplace_back(std::make_tuple(std::move(event), data_ptr.id));
    }
    info->multi_stream_free_event_cnt += used_streams.size();
  } 
  else {
    HT_RUNTIME_ERROR << "Unexpected occupation status ("
                     << static_cast<int>(info->status)
                     << ") during call to 'FreeDataSpace' for " << data_ptr;
    __builtin_unreachable();
  }
}

// Try to merge blocks. It assume that data_ptr has not been inserted into any lookup table. 
// It will check if there are adjacent splitted ptr and try to merge them. 
// "data_ptr" refers to newly released ptr, and "table" refers to the lookup 
// table where the ptr was originally intended to be inserted in.
// Return a pointer of the target lookup table which we want to insert the merged ptr in.
// Caller should hold the mutex.
DataPtrLookupTable* CUDACachingMemoryPool::TryMerge(std::shared_ptr<CudaDataPtrInfo>& data_info, 
                                                    DataPtrLookupTable* table) {
  // Decide which lookup table will the merged ptr be inserted in.
  // Only when src and dst are both not global lookup table will it select stream lookup table.
  HT_ASSERT(!data_info->allocated())
    << "TryMerge can only used when data ptr is unallocated";
  HT_ASSERT(data_info->status >= OccupationStatus::AVAILABLE_FOR_ALLOC_STREAM)
    << "TryMerge can only used when info status is available to alloc";
  HT_ASSERT(data_info->cached_pool == nullptr)
    << "TryMerge should guarantee the data ptr is not cached but released just now";
  // 功能函数1
  // 获取最终应该往哪个table去merge
  auto table_selection = [&](DataPtrLookupTable* src, DataPtrLookupTable* dst) -> DataPtrLookupTable* {
    // 在一个table中
    // 可以直接merge
    if (src == dst) {
      return src;
    } 
    // 不在一个table中
    // 必须要求其中一个是all stream available table才可以merge
    // 否则对于两个stream的table其之间的同步关系难以得到保证
    // 因此保守起见我们不进行跨stream的merge
    else {
      if (src == _available_for_all_streams.get())
        return dst;
      else if (dst == _available_for_all_streams.get())
        return src;
      else 
        return nullptr;
    }
  };
  // 功能函数2
  // 将old info往new info上头merge时
  // 处理二者的single stream free event
  auto handle_available_event = [&](std::shared_ptr<CudaDataPtrInfo>& new_info, 
                                    std::shared_ptr<CudaDataPtrInfo>& old_info,
                                    bool new_is_available_for_all) -> void {
    // new info具有available event
    // 这个时候要将new info和old info中最新的event作为最终的event
    if (!new_is_available_for_all) {
      // 因此如果old info没有event
      // 直接返回即可
      if (old_info->status == OccupationStatus::AVAILABLE_FOR_ALL_STREAM) {
        return;
      }
      auto old_event_info_it = _available_event_info.find(old_info->id);
      HT_ASSERT(old_event_info_it != _available_event_info.end())
        << "Cannot find the info of single stream free event towards data ptr id " << old_info->id; 
      auto old_available_event = old_event_info_it->second;
      auto new_event_info_it = _available_event_info.find(new_info->id);
      HT_ASSERT(new_event_info_it != _available_event_info.end())
        << "Cannot find the info of single stream free event towards data ptr id " << new_info->id; 
      auto new_available_event = new_event_info_it->second;
      auto final_available_event = std::make_shared<AvailableEvent>(new_info->id, 
                                                                    new_available_event->free_at, 
                                                                    new_available_event->stream, 
                                                                    new_available_event->event);
      if (old_available_event->free_at > new_available_event->free_at) {
        final_available_event->free_at = old_available_event->free_at;
        final_available_event->event = old_available_event->event;
      }
      DeleteAvailableEvent(old_info->id);
      DeleteAvailableEvent(new_info->id);
      AddAvailableEvent(new_info->id, final_available_event);
    }
    // new info不具有available event
    else {
      auto old_event_info_it = _available_event_info.find(old_info->id);
      HT_ASSERT(old_event_info_it != _available_event_info.end())
        << "Cannot find the info of single stream free event towards data ptr id " << old_info->id; 
      auto old_available_event = old_event_info_it->second;
      auto new_available_event = std::make_shared<AvailableEvent>(new_info->id, 
                                                                  old_available_event->free_at, 
                                                                  old_available_event->stream, 
                                                                  old_available_event->event);
      DeleteAvailableEvent(old_info->id);
      AddAvailableEvent(new_info->id, new_available_event);
    }
  };

  // 实际TryMerge部分
  // 如果是split出来的
  // 我们把剩余部分从
  // 1、available table 以及 2、info
  // 中删去并重新合并（向now靠齐）
  // 同时也要处理三者的single stream free event
  // 注意merge并不会导致split_from_id发生改变
  if (data_info->prev != nullptr && !data_info->prev->allocated()) {
    auto prev_info = data_info->prev; 
    HT_ASSERT(prev_info->split_from_id == data_info->split_from_id)
      << "split_from_id should always be consistent";
    HT_ASSERT(prev_info->status >= OccupationStatus::AVAILABLE_FOR_ALLOC_STREAM)
      << "Prev info status should be available to alloc"
      << ", but found " << prev_info->status;
    HT_ASSERT(prev_info->cached_pool)
      << "Prev info cached pool shouldn't be nullptr";
    auto try_merge_table = table_selection(table, prev_info->cached_pool);
    if (try_merge_table) {
      // 调整available event
      if (try_merge_table != _available_for_all_streams.get())
        handle_available_event(data_info, prev_info, table == _available_for_all_streams.get());
      table = try_merge_table;
      auto prev_data_ptr = DataPtr{prev_info->ptr, prev_info->num_bytes, device(), prev_info->id};
      auto table_it = prev_info->cached_pool->table.find(prev_data_ptr);
      HT_ASSERT(table_it != prev_info->cached_pool->table.end())
        << "Cannot find " << prev_data_ptr << " in the target table";
      prev_info->cached_pool->table.erase(table_it);
      data_info->ptr = prev_info->ptr;
      data_info->num_bytes += prev_info->num_bytes;
      data_info->prev = prev_info->prev;
      if (prev_info->prev != nullptr)
        prev_info->prev->next = data_info;
      auto info_it = _data_ptr_info.find(prev_info->id);
      HT_ASSERT(info_it != _data_ptr_info.end())
        << "Cannot find " << prev_data_ptr << " in the info";
      _data_ptr_info.erase(info_it);
    }
  }
  if (data_info->next != nullptr && !data_info->next->allocated()) {
    auto next_info = data_info->next; 
    HT_ASSERT(next_info->split_from_id == data_info->split_from_id)
      << "split_from_id should always be consistent";
    HT_ASSERT(next_info->status >= OccupationStatus::AVAILABLE_FOR_ALLOC_STREAM)
      << "Next info status should be available to alloc"
      << ", but found " << next_info->status;
    HT_ASSERT(next_info->cached_pool)
      << "Next info cached pool shouldn't be nullptr";
    auto try_merge_table = table_selection(table, next_info->cached_pool);
    if (try_merge_table) {
      // 调整available event
      if (try_merge_table != _available_for_all_streams.get())
        handle_available_event(data_info, next_info, table == _available_for_all_streams.get());
      table = try_merge_table;
      auto next_data_ptr = DataPtr{next_info->ptr, next_info->num_bytes, device(), next_info->id};
      auto table_it = next_info->cached_pool->table.find(next_data_ptr);
      HT_ASSERT(table_it != next_info->cached_pool->table.end())
        << "Cannot find " << next_data_ptr << " in the target table";
      next_info->cached_pool->table.erase(table_it);
      data_info->num_bytes += next_info->num_bytes;
      data_info->next = next_info->next;
      if (next_info->next != nullptr)
        next_info->next->prev = data_info;
      auto info_it = _data_ptr_info.find(next_info->id);
      HT_ASSERT(info_it != _data_ptr_info.end())
        << "Cannot find " << next_data_ptr << " in the info";
      _data_ptr_info.erase(info_it);
    }
  }
  HT_ASSERT(table)
    << "Table shouldn't be nullptr";
  // 修正最终的status
  if (table == _available_for_all_streams.get()) 
    data_info->status = OccupationStatus::AVAILABLE_FOR_ALL_STREAM;
  else
    data_info->status = OccupationStatus::AVAILABLE_FOR_ALLOC_STREAM;
  return table;
}

// the caller should hold the mutex
void CUDACachingMemoryPool::AddAvailableEvent(DataPtrId data_ptr_id, std::shared_ptr<AvailableEvent>& available_event) {
  auto event_info_insertion = _available_event_info.emplace(data_ptr_id, available_event);
  HT_RUNTIME_ERROR_IF(!event_info_insertion.second)
    << "Failed to insert the info of single stream free event towards data ptr id " << data_ptr_id;
  auto event_table_it = _single_stream_free_events.find(available_event->stream);
  if (event_table_it == _single_stream_free_events.end()) {
    auto table_insertion = _single_stream_free_events.emplace(available_event->stream, std::make_unique<AvailableEventLookupTable>());
    HT_RUNTIME_ERROR_IF(!table_insertion.second)
      << "Failed to insert available stream table for stream " << available_event->stream;
    event_table_it = table_insertion.first;
  }
  auto event_insertion = event_table_it->second->table.emplace(*available_event);
  HT_RUNTIME_ERROR_IF(!event_insertion.second)
    << "Failed to insert single stream free event towards data ptr id  " << data_ptr_id;
}

// the caller should hold the mutex
void CUDACachingMemoryPool::DeleteAvailableEvent(DataPtrId data_ptr_id) {
  auto event_info_it = _available_event_info.find(data_ptr_id);
  HT_ASSERT(event_info_it != _available_event_info.end())
    << "Cannot find the info of single stream free event towards data ptr id " << data_ptr_id; 
  auto available_event = event_info_it->second;
  auto event_table_it = _single_stream_free_events.find(available_event->stream);
  HT_ASSERT(event_table_it != _single_stream_free_events.end())
    << "Cannot find the available event table for stream " << available_event->stream;
  auto event_it = event_table_it->second->table.find(*available_event);
  HT_ASSERT(event_it != event_table_it->second->table.end())
    << "Cannot find the single stream free event towards data ptr id " << data_ptr_id;
  // 删除
  event_table_it->second->table.erase(event_it);
  _available_event_info.erase(event_info_it);
}

// the caller should hold the mutex
void CUDACachingMemoryPool::WatchEvents() {
  // 处理single stream上的free
  for (auto& kv : _single_stream_free_events) {
    auto& stream_free_events = kv.second->table;
    while (!stream_free_events.empty()) {
      auto event_it = stream_free_events.begin();
      cudaError_t status = event_it->event->Query();
      if (status == cudaErrorNotReady) {
        // ignore and clear the not-ready error
        (void) cudaGetLastError();
        break;
      } else if (status != cudaSuccess) {
        __HT_FATAL_SILENT(hetu::cuda::cuda_error)
          << "cudaEventQuery failed: " << cudaGetErrorString(status);
        __builtin_unreachable();
      }
      DataPtrId data_ptr_id = event_it->id;
      auto info_it = _data_ptr_info.find(data_ptr_id);
      HT_ASSERT(info_it != _data_ptr_info.end())
        << "Cannot find data ptr with id " << data_ptr_id << " in the info"; 
      auto info = info_it->second;
      // 删除available event
      stream_free_events.erase(event_it);
      auto event_info_it = _available_event_info.find(data_ptr_id);
      HT_ASSERT(event_info_it != _available_event_info.end())
        << "Cannot find the info of single stream free event towards data ptr id " << data_ptr_id;
      _available_event_info.erase(event_info_it);
      // 移动cache entry到合适的table
      MoveAvailable(info);
    }
  }
  // 处理multi stream上的free
  for (auto& kv : _multi_stream_free_events) {
    auto& stream_free_events = *(kv.second);
    while (!stream_free_events.empty()) {
      auto& tuple = stream_free_events.front();
      std::unique_ptr<CUDAEvent>& event = std::get<0>(tuple);
      cudaError_t status = event->Query();
      if (status == cudaErrorNotReady) {
        // ignore and clear the not-ready error
        (void) cudaGetLastError();
        // since events are enqueued in order, we can ignore the rest
        break;
      } else if (status != cudaSuccess) {
        __HT_FATAL_SILENT(hetu::cuda::cuda_error)
          << "cudaEventQuery failed: " << cudaGetErrorString(status);
        __builtin_unreachable();
      }
      // decrement the number of free events for that 
      DataPtrId data_ptr_id = std::get<1>(tuple);
      auto it = _data_ptr_info.find(data_ptr_id);
      HT_RUNTIME_ERROR_IF(it == _data_ptr_info.end())
        << "Cannot find data " << data_ptr_id << " from info";
      auto info = it->second;
      if ((--info->multi_stream_free_event_cnt) == 0) {
        // borrow data
        if (info->deleter) {
          // 直接删除即可
          info->deleter(DataPtr{info->ptr, info->num_bytes, device(), data_ptr_id});
          _data_ptr_info.erase(it);
          _reserved -= info->num_bytes;
          _allocated -= info->num_bytes;
          _free_cnt++;
        } 
        // alloc data
        else {
          HT_ASSERT(info->status == OccupationStatus::UNAVAILABLE_UNTIL_FREE) 
            << "Info status should be " << OccupationStatus::UNAVAILABLE_UNTIL_FREE
            << ", but found " << info->status;
          _allocated -= info->num_bytes;
          _free_cnt++;
          // 优先考虑放到all stream available table中cache住
          DataPtrLookupTable* target_table = _available_for_all_streams.get();
          info->refresh();
          if (info->is_split())
            target_table = TryMerge(info, target_table);
          // Meta information of data_ptr might have change in TryMerge
          auto data_ptr = DataPtr{info->ptr, info->num_bytes, device(), data_ptr_id};
          HT_ASSERT(data_ptr_id == info->id)
            << "The data ptr id and the info id are mismatched"; 
          data_ptr.split_from_id = info->split_from_id;
          info->cached_pool = target_table;
          InsertAvailable(data_ptr, *target_table); 
        }
      }
      stream_free_events.pop_front();
    }
  }
}

void CUDACachingMemoryPool::MarkDataSpaceUsedByStream(DataPtr data_ptr,
                                                      const Stream& stream) {
  if (data_ptr.ptr == nullptr || data_ptr.size == 0 || stream.is_blocking())
    return;
  HT_VALUE_ERROR_IF(!stream.device().is_cuda())
    << "Cuda arrays must be used on cuda streams. Got " << stream;
  PackedStreamId packed_stream_id = stream.pack();
  
  std::lock_guard<std::mutex> lock(_mtx);
  auto it = _data_ptr_info.find(data_ptr.id);
  HT_RUNTIME_ERROR_IF(it == _data_ptr_info.end())
    << "Cannot find data " << data_ptr << " from info";
  auto info = it->second;
  info->insert_used_stream(packed_stream_id);
  _mark_cnt++;
}

void CUDACachingMemoryPool::MarkDataSpacesUsedByStream(DataPtrList& data_ptrs,
                                                       const Stream& stream) {
  if (stream.is_blocking())
    return;
  HT_VALUE_ERROR_IF(!stream.device().is_cuda())
    << "Cuda arrays must be used on cuda streams. Got " << stream;
  PackedStreamId packed_stream_id = stream.pack();
  std::lock_guard<std::mutex> lock(_mtx);
  for (auto& data_ptr : data_ptrs) {
    if (data_ptr.ptr == nullptr || data_ptr.size == 0)
      continue;
    auto it = _data_ptr_info.find(data_ptr.id);
    HT_RUNTIME_ERROR_IF(it == _data_ptr_info.end())
      << "Cannot find data " << data_ptr << " from info";
    auto info = it->second;
    info->insert_used_stream(packed_stream_id);
    _mark_cnt++;
  }
}

std::future<void> CUDACachingMemoryPool::WaitDataSpace(DataPtr data_ptr,
                                                       bool async) {
  if (data_ptr.ptr == nullptr || data_ptr.size == 0)
    return async ? std::async([]() {}) : std::future<void>();

  std::unique_lock<std::mutex> lock(_mtx);
  auto it = _data_ptr_info.find(data_ptr.id);
  HT_RUNTIME_ERROR_IF(it == _data_ptr_info.end())
    << "Cannot find data " << data_ptr << " from info";
  PackedStreamId alloc_stream = it->second->alloc_stream;
  auto& used_streams = it->second->used_streams;
  if (used_streams.empty()) {
    // This only happens when alloc_stream and all used_streams are blocking
    return async ? std::async([]() {}) : std::future<void>();
  }

  // TODO: Avoid synchronizing allocation and used streams again 
  // when freeing the memory. However, remember that it necessitates 
  // tracking whether each async waits has completed or not.
  Stream wait_stream;
  if (used_streams.size() == 1 && *used_streams.begin() == alloc_stream) {
    wait_stream = Stream::unpack(alloc_stream);
  } else {
    Stream join_stream(data_ptr.device, kJoinStream);
    for (auto& used_stream : used_streams) {
      CUDAEvent event(data_ptr.device, false);
      event.Record(Stream::unpack(used_stream));
      event.Block(join_stream);
    }
    wait_stream = join_stream;
  }
  lock.unlock();

  if (async) {
    return std::async([wait_stream]() { CUDAStream(wait_stream).Sync(); });
  } else {
    CUDAStream(wait_stream).Sync();
    return std::future<void>();
  }
}

void CUDACachingMemoryPool::PrintSummary() {
  HT_LOG_INFO << name() << ": alloc = " << static_cast<double>(_allocated) / (1024 * 1024) << " MiB"
    << ", reserved = " << static_cast<double>(_reserved) / (1024 * 1024) << " MiB"
    << ", peak_reserved = " << static_cast<double>(_peak_reserved) / (1024 * 1024) << " MiB"
    << ", alloc_cnt = " << _alloc_cnt
    << ", free_cnt = " << _free_cnt
    << ", mark_cnt = " << _mark_cnt;
}

namespace {

static std::once_flag cuda_caching_memory_pool_register_flag;

static size_t ParseMaxSplitSize() {
  const char* max_split_str = std::getenv("HETU_MAX_SPLIT_SIZE_MB");
  size_t max_split_size_mb = 200; // 默认设置为200MiB
  if (max_split_str != NULL) {
    try {
      max_split_size_mb = std::stoi(max_split_str);
      // TODO: 敲定一个最小值....
    } catch (const std::exception& e) {
      HT_LOG_WARN
        << "Invalid HETU_MAX_SPLIT_SIZE_MB: " << max_split_str << " is set" 
        << ", please provide an integer"
        << ", default value will be used in this process.";
    }
  } 
  return max_split_size_mb * 1024 * 1024;
}

static size_t ParseMaxInternalFragmentSize() {
  const char* max_internal_fragment_str = std::getenv("HETU_MAX_INTERNAL_FRAGMENT_SIZE_MB");
  size_t max_internal_fragment_size_mb = 20; // 默认设置为20MiB
  if (max_internal_fragment_str != NULL) {
    try {
      max_internal_fragment_size_mb = std::stoi(max_internal_fragment_str);
      // TODO: 敲定一个最小值....
    } catch (const std::exception& e) {
      HT_LOG_WARN
        << "Invalid HETU_MAX_INTERNAL_FRAGMENT_SIZE_MB: " << max_internal_fragment_str << " is set" 
        << ", please provide an integer"
        << ", default value will be used in this process.";
    }
  } 
  return max_internal_fragment_size_mb * 1024 * 1024;
}

static size_t ParsePreAllocateSize() {
  const char* pre_allocate_str = std::getenv("HETU_PRE_ALLOCATE_SIZE_MB");
  size_t pre_allocate_size_mb = 0;
  if (pre_allocate_str != NULL) {
    try {
      pre_allocate_size_mb = std::stoi(pre_allocate_str);
    } catch (const std::exception& e) {
      HT_LOG_WARN
        << "Invalid HETU_PRE_ALLOCATE_SIZE_MB: " << pre_allocate_str << " is set" 
        << ", please provide an integer"
        << ", default value will be used in this process.";
    }
  } 
  return pre_allocate_size_mb * 1024 * 1024;
}

struct CUDACachingMemoryPoolRegister {
  CUDACachingMemoryPoolRegister() { 
    std::call_once(cuda_caching_memory_pool_register_flag, []() {
      size_t _max_split_size = ParseMaxSplitSize();
      size_t _max_internal_fragment_size = ParseMaxInternalFragmentSize();
      size_t _pre_allocate_size = ParsePreAllocateSize();
      // Memory pools are lazily constructed, so we do not need to
      // get device count here.
      for (int32_t i = 0; i < HT_MAX_DEVICE_INDEX; i++) {
        RegisterMemoryPoolCtor(
          Device(kCUDA, i), [i, _max_split_size, _max_internal_fragment_size, _pre_allocate_size]() -> std::shared_ptr<MemoryPool> {
            return std::make_shared<CUDACachingMemoryPool>(i, _max_split_size, _max_internal_fragment_size, _pre_allocate_size);
          });
      }
    });
  }
};

static CUDACachingMemoryPoolRegister cuda_caching_memory_pool_register;

} // namespace

} // namespace impl
} // namespace hetu
