#include "hetu/core/ndarray.h"
#include "hetu/core/stream.h"
#include "hetu/impl/communication/mpi_comm_group.h"
#include "hetu/impl/utils/common_utils.h"

namespace hetu {
namespace impl {

using namespace hetu::impl::comm;

void AllReduceCpu(const NDArray& input, NDArray& output, ReductionType red_type,
                  const DeviceGroup& device_group, bool use_fp32, const Stream& stream) {
  auto ranks = DeviceGroupToWorldRanks(device_group);
  auto& comm_group = MPICommunicationGroup::GetOrCreate(ranks, stream);
  comm_group->AllReduce(input, output, red_type);
  NDArray::MarkUsedBy({input, output}, stream);
}

void AllGatherCpu(const NDArray& input, NDArray& output,
                  const DeviceGroup& device_group, int32_t gather_dim, const Stream& stream) {
  auto ranks = DeviceGroupToWorldRanks(device_group);
  auto& comm_group = MPICommunicationGroup::GetOrCreate(ranks, stream);
  comm_group->AllGather(input, output, gather_dim);      
  NDArray::MarkUsedBy({input, output}, stream);            
}

void ReduceScatterCpu(const NDArray& input, NDArray& output, ReductionType red_type,
                      const DeviceGroup& device_group, int32_t scatter_dim,
                      bool use_fp32, const Stream& stream) {
  auto ranks = DeviceGroupToWorldRanks(device_group);
  auto& comm_group = MPICommunicationGroup::GetOrCreate(ranks, stream);
  comm_group->ReduceScatter(input, output, scatter_dim, red_type);
  NDArray::MarkUsedBy({input, output}, stream);
}

void P2PSendCpu(const NDArray& data, const Device& dst, const std::vector<int>& comm_group_ranks, const Stream& stream) {
  auto dst_rank = DeviceToWorldRank(dst);
  auto& comm_group = MPICommunicationGroup::GetOrCreate(comm_group_ranks, stream);
  comm_group->Send(data, dst_rank);
  NDArray::MarkUsedBy({data}, stream);
}

void P2PRecvCpu(NDArray& data, const Device& src, const std::vector<int>& comm_group_ranks, const Stream& stream) {
  auto src_rank = DeviceToWorldRank(src);
  auto& comm_group = MPICommunicationGroup::GetOrCreate(comm_group_ranks, stream);
  comm_group->Recv(data, src_rank);
  NDArray::MarkUsedBy({data}, stream);
}

void BatchedISendIRecvCpu(const NDArrayList& send_datas, 
  const std::vector<Device>& dsts, NDArrayList& recv_datas, 
  const std::vector<Device>& srcs, const std::vector<Device>& comm_deivces, 
  const Stream& stream) {
  std::vector<int> ranks(comm_deivces.size());
  std::transform(comm_deivces.begin(), comm_deivces.end(), ranks.begin(), [&](const Device& device) { return DeviceToWorldRank(device); });
  std::sort(ranks.begin(), ranks.end());
  auto& comm_group = MPICommunicationGroup::GetOrCreate(ranks, stream);
  std::vector<CommTask> tasks;
  tasks.reserve(send_datas.size() + recv_datas.size());
  for (int i = 0; i < send_datas.size(); i++) {
    tasks.push_back(comm_group->ISend(send_datas[i], DeviceToWorldRank(dsts[i])));
  }
  for (int i = 0; i < recv_datas.size(); i++) {
    tasks.push_back(comm_group->IRecv(recv_datas[i], DeviceToWorldRank(srcs[i])));
  }
  comm_group->BatchedISendIRecv(tasks);
  NDArray::MarkUsedBy(send_datas, stream);
  NDArray::MarkUsedBy(recv_datas, stream);
}

void BroadcastCommCpu(NDArray& data, int broadcaster,
                      const DeviceGroup& device_group, const Stream& stream) {
  auto ranks = DeviceGroupToWorldRanks(device_group);
  auto& comm_group = MPICommunicationGroup::GetOrCreate(ranks, stream);
  comm_group->Broadcast(data, broadcaster);
  NDArray::MarkUsedBy({data}, stream);
}

void ReduceCommCpu(const NDArray& input, NDArray& output, int reducer,
                const DeviceGroup& device_group, const Stream& stream) {
  auto ranks = DeviceGroupToWorldRanks(device_group);
  auto& comm_group = MPICommunicationGroup::GetOrCreate(ranks, stream);
  comm_group->Reduce(input, output, reducer);
  NDArray::MarkUsedBy({input, output}, stream);    
}

void GatherCpu(const NDArray& input, NDArray& output, int gatherer,
                const DeviceGroup& device_group, const Stream& stream) {
  auto ranks = DeviceGroupToWorldRanks(device_group);
  auto& comm_group = MPICommunicationGroup::GetOrCreate(ranks, stream);
  comm_group->Gather(input, output, gatherer);
  NDArray::MarkUsedBy({input, output}, stream);  
}

void ScatterCpu(const NDArray& input, NDArray& output, int scatterer,
                const DeviceGroup& device_group, const Stream& stream) {
  auto ranks = DeviceGroupToWorldRanks(device_group);
  auto& comm_group = MPICommunicationGroup::GetOrCreate(ranks, stream);
  comm_group->Scatter(input, output, scatterer);
  NDArray::MarkUsedBy({input, output}, stream);  
}

} // namespace impl
} // namespace hetu
