#include "hetu/core/ndarray.h"
#include "hetu/core/stream.h"
#include "hetu/impl/communication/mpi_comm_group.h"
#include "hetu/impl/utils/common_utils.h"

namespace hetu {
namespace impl {

using namespace hetu::impl::comm;

template <typename spec_t>
void memory_copy_cpu(const spec_t* input, spec_t* output, size_t size) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t idx = 0; idx < size; ++idx) {
    output[idx] = input[idx];
  }
}

void BroadcastCommCpu(const NDArray& input, NDArray& output, int broadcaster,
                   const DeviceGroup& device_group, const Stream& stream) {
  auto ranks = DeviceGroupToWorldRanks(device_group);
  auto& comm_group = MPICommunicationGroup::GetOrCreate(ranks, stream);
  size_t size = output->numel();
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input->dtype(), spec_t, "ReshapeCpu", [&]() {
      memory_copy_cpu<spec_t>(input->data_ptr<spec_t>(),
                              output->data_ptr<spec_t>(), size);
    });
  comm_group->Broadcast(output, broadcaster);
}

void AllReduceCpu(const NDArray& input, NDArray& output,
                  const DeviceGroup& device_group, const Stream& stream) {
  auto ranks = DeviceGroupToWorldRanks(device_group);
  auto& comm_group = MPICommunicationGroup::GetOrCreate(ranks, stream);
  comm_group->AllReduce(input, output);
}

void ReduceCommCpu(const NDArray& input, NDArray& output, int reducer,
                const DeviceGroup& device_group, const Stream& stream) {
  auto ranks = DeviceGroupToWorldRanks(device_group);
  auto& comm_group = MPICommunicationGroup::GetOrCreate(ranks, stream);
  comm_group->Reduce(input, output, reducer);
}

void AllGatherCpu(const NDArray& input, NDArray& output,
                  const DeviceGroup& device_group, const Stream& stream) {
  auto ranks = DeviceGroupToWorldRanks(device_group);
  auto& comm_group = MPICommunicationGroup::GetOrCreate(ranks, stream);
  comm_group->AllGather(input, output);
}

void ReduceScatterCpu(const NDArray& input, NDArray& output,
                  const DeviceGroup& device_group, const Stream& stream) {
  auto ranks = DeviceGroupToWorldRanks(device_group);
  auto& comm_group = MPICommunicationGroup::GetOrCreate(ranks, stream);
  comm_group->ReduceScatter(input, output);
}

void GatherCpu(const NDArray& input, NDArray& output, int gatherer,
                const DeviceGroup& device_group, const Stream& stream) {
  auto ranks = DeviceGroupToWorldRanks(device_group);
  auto& comm_group = MPICommunicationGroup::GetOrCreate(ranks, stream);
  comm_group->Gather(input, output, gatherer);
}

void ScatterCpu(const NDArray& input, NDArray& output, int scatterer,
                const DeviceGroup& device_group, const Stream& stream) {
  auto ranks = DeviceGroupToWorldRanks(device_group);
  auto& comm_group = MPICommunicationGroup::GetOrCreate(ranks, stream);
  comm_group->Scatter(input, output, scatterer);
}

void P2PSendCpu(const NDArray& data, const Device& dst, const Stream& stream) {
  auto src_rank = GetWorldRank();
  auto dst_rank = DeviceToWorldRank(dst);
  std::vector<int> ranks(2);
  ranks[0] = std::min(src_rank, dst_rank);
  ranks[1] = std::max(src_rank, dst_rank);
  auto& comm_group = MPICommunicationGroup::GetOrCreate(ranks, stream);
  comm_group->Send(data, dst_rank);
}

void P2PRecvCpu(NDArray& data, const Device& src, const Stream& stream) {
  auto src_rank = DeviceToWorldRank(src);
  auto dst_rank = GetWorldRank();
  std::vector<int> ranks(2);
  ranks[0] = std::min(src_rank, dst_rank);
  ranks[1] = std::max(src_rank, dst_rank);
  auto& comm_group = MPICommunicationGroup::GetOrCreate(ranks, stream);
  comm_group->Recv(data, src_rank);
}

} // namespace impl
} // namespace hetu
