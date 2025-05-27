#include "hetu/impl/stream/CPUStream.h"
#include "hetu/impl/utils/ndarray_utils.h"
#include "hetu/impl/communication/rpc_client.h"
#include <numeric>
#include <mutex>

namespace hetu {
namespace impl {
namespace comm {

using hetu::operator<<;

namespace {

static std::once_flag rpc_init_flag;
static int rpc_world_rank = -1; // 当前进程的rank
static int local_device_idx = -1;
std::vector<int> rpc_world_ranks; // [0, 1, 2, ..., rpc_world_size - 1]
static int rpc_world_size = -1; // 一共有多少rank（例如2台A100机器就是16）
static std::string global_server_address;
static std::mutex rpc_call_mutex;
static std::mutex rpc_create_group_mutex;
static std::vector<std::once_flag>
  worldwide_rpc_comm_group_reg_flags((HT_NUM_STREAMS_PER_DEVICE) + 1);
static std::shared_ptr<DeviceClient> local_client;

} // namespace

struct RPCCallGuard {
  // rpc_THREAD_SERIALIZED requires all rpc calls are sequential,
  // so we need to lock on a global mutex here.
  RPCCallGuard() : lock(rpc_call_mutex) {}
  std::lock_guard<std::mutex> lock;
};

static void RPC_Init_Once() {
  std::call_once(rpc_init_flag, []() {
    // init rpc
    HT_LOG_INFO << "HTSVTR: " << global_server_address;
    // 建立当前进程上的stub
    local_client = std::make_shared<DeviceClient>(grpc::CreateChannel(global_server_address, 
                                                                      grpc::InsecureChannelCredentials()));
    // 告知server当前进程的hostname
    // server会统计一共多少host以及每个host上有多少进程
    local_client->Connect(Device::GetLocalHostname());
    std::vector<int> all_ranks(rpc_world_size);
    std::iota(all_ranks.begin(), all_ranks.end(), 0);
    rpc_world_ranks = all_ranks;
    HT_LOG_INFO << "alrank:" << all_ranks;
    // 同步
    // 等所有client进程都向server注册完
    local_client->Barrier(0, all_ranks);
    // 进程获取自己的rank（与注册顺序相关）
    auto rank_pair = local_client->GetRank(Device::GetLocalHostname());
    HT_LOG_INFO << Device::GetLocalHostname();
    rpc_world_rank = rank_pair.first;
    HT_LOG_DEBUG << "GETRANK:" << rpc_world_rank;
    local_device_idx = rank_pair.second;
    // 启动一个后台线程
    // 定期发送heartbeat给server
    local_client->LaunchHeartBeat(rpc_world_rank);
    // register exit handler
    HT_ASSERT(std::atexit([]() {
                HT_LOG_DEBUG << "Destructing rpc comm groups...";
                local_client->Exit(rpc_world_rank);
                HT_LOG_DEBUG << "Destructed rpc comm groups";
              }) == 0)
      << "Failed to register the exit function for rpc.";
  });
}

int GetWorldRank() {
  RPC_Init_Once();
  return rpc_world_rank;
}

int GetWorldSize() {
  RPC_Init_Once();
  return rpc_world_size;
}

std::vector<int> GetWorldRanks() {
  RPC_Init_Once();
  return rpc_world_ranks;
}

std::shared_ptr<DeviceClientImpl> GetLocalClient() {
  RPC_Init_Once();
  return local_client;
}

int GetGroupRank(const std::vector<int>& world_ranks) {
  int my_world_rank = GetWorldRank();
  auto it = std::find(world_ranks.begin(), world_ranks.end(), my_world_rank);
  return it != world_ranks.end() ? std::distance(world_ranks.begin(), it) : -1;
}

/******************************************************
 * Maintaining the mapping between ranks and devices
 ******************************************************/

namespace {
static std::once_flag device_mapping_init_flag;
std::unordered_map<Device, int> device_to_rank_mapping;
std::vector<Device> rank_to_device_mapping;
DeviceGroup global_device_group;

std::vector<std::string> AllGatherHostnames(std::vector<int> ranks) {
  std::string local_hostname = Device::GetLocalHostname();
  local_client->CommitHostName(local_hostname, rpc_world_rank); 
  std::vector<std::string> hostnames;
  hostnames.reserve(ranks.size());
  for (int rank = 0; rank < ranks.size(); rank++) {
    std::string rank_hostname = local_client->GetHostName(rank);
    hostnames.emplace_back(rank_hostname);
  }
  HT_LOG_INFO << "RPC:" << local_hostname << " " << hostnames;
  return hostnames;
}

void SetUpDeviceMappingWithAssignedLocalDevice(const Device& local_device) {
  HT_ASSERT(local_device.local()) << "Device is not local: " << local_device;
  // auto& comm = RPCCommunicationGroup::GetOrCreateWorldwide();
  RPC_Init_Once();
  auto hostnames = AllGatherHostnames(rpc_world_ranks);
  // Walkaround: communication groups handle ndarrays only
  auto world_size = GetWorldSize();

  device_to_rank_mapping.reserve(world_size);
  rank_to_device_mapping.reserve(world_size);
  local_client->CommitDeviceInfo(static_cast<int>(local_device.type()), local_device.index(), 
                                 local_device.multiplex(), rpc_world_rank);
  for (int rank = 0; rank < world_size; rank++) {
    DeviceInfoReply reply = local_client->GetDeviceInfo(rank);
    Device rank_device(static_cast<DeviceType>(reply.type),
                       reply.index, hostnames[rank],
                       reply.multiplex, rank);
    HT_LOG_DEBUG << "Device of rank[" << rank << "]: " << rank_device;
    HT_ASSERT(device_to_rank_mapping.find(rank_device) ==
                device_to_rank_mapping.end() ||
              rank_device.is_cpu())
      << "Device " << rank_device << " is duplicated.";
    device_to_rank_mapping.insert({rank_device, rank});
    rank_to_device_mapping.emplace_back(rank_device);
  }
  HT_LOG_WARN << "world_size:" << world_size;
  HT_LOG_WARN << "rank_to_device_mapping:" << rank_to_device_mapping;
  global_device_group = DeviceGroup(rank_to_device_mapping);
  HT_LOG_DEBUG << "Global devices: " << global_device_group;
}

void SetUpDeviceMappingAndAssignLocalDevice(
  const std::map<DeviceType, int>& resources,
  const std::vector<int64_t>& device_idxs) {
  // auto& comm = RPCCommunicationGroup::GetOrCreateWorldwide();
  RPC_Init_Once();
  auto hostnames = AllGatherHostnames(rpc_world_ranks);
  auto local_hostname = Device::GetLocalHostname();
  HT_LOG_DEBUG << hostnames;
  HT_ASSERT(hostnames[rpc_world_rank] == local_hostname)
    << "Local hostname mismatched after gathering: " << hostnames[rpc_world_rank]
    << " vs. " << local_hostname;
  Device local_device;
  int local_rank = 0;
  // for (int i = 0; i < rpc_world_rank; ++i) {
  //   if (hostnames[i] == local_hostname)
  //     local_rank++;
  // }
  local_rank = local_device_idx;
  HT_LOG_DEBUG << "local host = " << local_hostname << ", rank = " << rpc_world_rank 
               << ", all hosts = " << hostnames << ", world ranks = " << rpc_world_ranks
               << ", world size = " << GetWorldSize() << ", local rank = " << local_rank;
  HT_LOG_WARN << device_idxs;
  if (resources.find(kCUDA) == resources.end() || resources.at(kCUDA) == 0) {
    // Question: do we need to set the multiplex field for CPU?
    local_device = Device(kCPU);
  } else {
    auto device_id = device_idxs.empty() ? local_rank % resources.at(kCUDA)
                                         : device_idxs[local_rank % resources.at(kCUDA)];
    auto multiplex = local_rank / resources.at(kCUDA);
    auto global_device_id = rpc_world_rank;
    // multiplex当rank数大于卡数时出现（比如只有4个GPU但开了8个进程）
    local_device = Device(kCUDA, device_id, local_hostname, 
                          multiplex, global_device_id);
  }
  SetUpDeviceMappingWithAssignedLocalDevice(local_device);
}

} // namespace

void SetUpDeviceMappingWithAssignedLocalDeviceOnce(const Device& local_device) {
  if (!device_to_rank_mapping.empty()) {
    HT_LOG_WARN << "Device mapping has been set up.";
    return;
  }
  std::call_once(device_mapping_init_flag,
                 SetUpDeviceMappingWithAssignedLocalDevice, local_device);
}

Device SetUpDeviceMappingAndAssignLocalDeviceOnce(
  const std::map<DeviceType, int>& resources,
  const std::vector<int64_t>& device_idxs,
  const std::string server_address) {
  rpc_world_size = resources.find(kCUDA)->second;
  HT_LOG_INFO << server_address;
  global_server_address = server_address;
  if (!device_to_rank_mapping.empty()) {
    HT_LOG_WARN << "Device mapping has been set up.";
    return rank_to_device_mapping[GetWorldRank()];
  }
  std::call_once(device_mapping_init_flag,
                 SetUpDeviceMappingAndAssignLocalDevice, resources, device_idxs);
  return rank_to_device_mapping[GetWorldRank()];
}

bool IsGlobalDeviceGroupReady() {
  return !global_device_group.empty();
}

const DeviceGroup& GetGlobalDeviceGroup() {
  HT_ASSERT(!device_to_rank_mapping.empty())
    << "Please set up the device mapping in advance.";
  return global_device_group;
}

const Device& GetLocalDevice() {
  if (rank_to_device_mapping.empty())
    return Device();
  HT_ASSERT(!rank_to_device_mapping.empty())
    << "Please set up the device mapping in advance.";
  return rank_to_device_mapping.at(GetWorldRank());
}

int GetRankOfLocalHost() {
  int world_rank = GetWorldRank(), local_rank = 0;
  for (int i = 0; i < world_rank; i++)
    if (rank_to_device_mapping[i].local())
      local_rank++;
  return local_rank;
}

int DeviceToWorldRank(const Device& device) {
  auto it = device_to_rank_mapping.find(device);
  HT_ASSERT(it != device_to_rank_mapping.end())
    << "Cannot find device " << device << ".";
  return it->second;
}

const Device& WorldRankToDevice(int rank) {
  HT_ASSERT(rank < rank_to_device_mapping.size())
    << "Cannot find rank " << rank << ".";
  return rank_to_device_mapping[rank];
}

std::vector<int> DeviceGroupToWorldRanks(const DeviceGroup& device_group) {
  std::vector<int> ranks;
  ranks.reserve(device_group.num_devices());
  for (const auto& device : device_group.devices())
    ranks.push_back(DeviceToWorldRank(device));
  // std::sort(ranks.begin(), ranks.end());
  return ranks;
}

} // namespace comm
} // namespace impl
} // namespace hetu
