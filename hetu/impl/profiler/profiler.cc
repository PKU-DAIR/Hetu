#include "hetu/impl/profiler/profiler.h"
#include <unistd.h>

namespace hetu {
namespace impl {


std::once_flag Profile::_init_flag;
std::vector<std::shared_ptr<Profile>> Profile::_global_profile;
thread_local std::stack<ProfileId> Profile::_cur_profile_ctx;
std::vector< export_event > Profile::memory_profiler_info;
int64_t Profile::init_time;

ProfileId Profile::_next_profile_id() {
  static std::atomic<ProfileId> _global_profile_id{0};
  return _global_profile_id++;
}

RecordFunctionHandle next_unique_record_function_handle() {
  static std::atomic<uint64_t> unique_rf_id {0};
  return RecordFunctionHandle(++unique_rf_id);
}


void RecordFunction::before(std::string name) {
  name_ = name;
  auto profiler = Profile::get_cur_profile();
  if(profiler != nullopt){
    MemoryEvent evt{
      MemoryEventKind::PushRange,
      name,
      getCurrentTimeInMicroseconds(),
      op_id(),
      next_unique_record_function_handle(),
    scope()
    };
    getCurEventList().record(std::move(evt));
  }
}

void RecordFunction::end() {
  auto profiler = Profile::get_cur_profile();
  if(profiler != nullopt){
    MemoryEvent evt{
      MemoryEventKind::PopRange,
      name(),
      getCurrentTimeInMicroseconds(),
      op_id(),
      next_unique_record_function_handle(),
    scope()
    };
    getCurEventList().record(std::move(evt));
  }
}

RecordFunction::~RecordFunction(){
  end();
}




void Profile::Init() {
  // exit handler
  auto status = std::atexit([]() {
    HT_LOG_DEBUG << "Clearing and destructing all profiler...";
    auto& local_device = hetu::impl::comm::GetLocalDevice();

    std::string file_path = "logs/device_" + std::to_string((int)hetu::impl::comm::DeviceToWorldRank(local_device)) + "_" +  ".pt.trace.json";

    Profile::generate_memory_json(file_path);    
    sleep(10);
    for (auto& profile : Profile::_global_profile) {
      if (profile == nullptr)
        continue;
      profile->Clear();
    }
    Profile::_global_profile.clear();
    HT_LOG_DEBUG << "Destructed all profiler";
  });
  HT_ASSERT(status == 0)
      << "Failed to register the exit function for profiler.";

  Profile::init_time = getCurrentTimeInMicroseconds();
  auto concurrency = std::thread::hardware_concurrency();
  Profile::_global_profile.reserve(MIN(concurrency, 16) * 2);
}


RangeMemoryEventList& getCurEventList() {
  auto profiler = Profile::get_cur_profile();
  HT_ASSERT(profiler != nullopt);
  return (*profiler)->get_event_list();
}

void reportCudaMemoryToProfiler(void* ptr, int64_t bytes, int64_t total_alloc, int64_t total_reserved, Device dev){
  // printf("reportCudaMemoryToProfiler\n");
  // printf("ptr: %p, bytes: %ld, total_alloc: %ld, total_reserved: %ld\n", ptr, (long)bytes, (long)total_alloc, (long)total_reserved);
  auto profiler = Profile::get_cur_profile();
  if(profiler != nullopt){
    MemoryEvent evt(
        MemoryEventKind::MemoryAlloc,
        std::string(""),
        getCurrentTimeInMicroseconds());
    evt.updateCudaMemoryStats(ptr, bytes, total_alloc, total_reserved, dev);
    (*profiler)->get_event_list().record(std::move(evt));
  }
  else{
    /*profielr已经析构啦，但是还有显存释放的event*/
  }
}


} // namespace impl
} // namespace hetu