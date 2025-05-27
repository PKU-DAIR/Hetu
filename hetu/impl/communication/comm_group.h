#pragma once

#include "hetu/common/macros.h"
#include "hetu/utils/shared_ptr_wrapper.h"
#include "hetu/core/ndarray.h"
#include "hetu/core/stream.h"
#include "hetu/impl/communication/rpc_client_impl.h"
#include "hetu/utils/json/json.hpp"

using json = nlohmann::json;

namespace hetu {
namespace impl {
namespace comm {

using hetu::operator<<;

struct CommTask {
  std::function<void()> fn;
  NDArrayList data;

  CommTask() = default;
  CommTask(std::function<void()> fn_, NDArrayList data_)
  : fn(std::move(fn_)), data(std::move(data_)) {}
};

class CommunicationGroupDef : public shared_ptr_target {
 protected:
  CommunicationGroupDef(const std::vector<int>& world_ranks,
                        const Stream& stream)
  : _stream(stream) {
    HT_ASSERT(_stream.is_defined()) << "Undefined stream provided.";
    _world_ranks = world_ranks;
    // std::sort(_world_ranks.begin(), _world_ranks.end());
    HT_ASSERT(!_world_ranks.empty())
      << "Passing an empty world ranks is a short-cut and "
      << "should only be used in the \"GetOrCreate\" functions. "
      << "Please explicityly pass the world ranks to the constructor "
      << "of communication groups.";
    HT_ASSERT(IsRanksValid(_world_ranks))
      << "Invalid world ranks: " << _world_ranks;
    for (int i = 0; i < static_cast<int>(_world_ranks.size()); i++)
      _world_to_group_mapping[_world_ranks[i]] = i;
  }

  inline static bool IsRanksValid(const std::vector<int>& ranks) {
    if (ranks.size() < 1)
      return false;
    // for (size_t i = 1; i < ranks.size(); i++)
    //   if (ranks[i - 1] >= ranks[i])
    //     return false;
    return true;
  }

 public:
  ~CommunicationGroupDef() = default;

  virtual void Broadcast(NDArray& data, int broadcaster) {
    HT_NOT_IMPLEMENTED << "Broadcast fn of backend \"" << backend()
                       << "\" is not defined.";
  }

  virtual void AllReduce(const NDArray& input, NDArray& output,
                         ReductionType red_type = kSUM) {
    HT_NOT_IMPLEMENTED << "AllReduce fn of backend \"" << backend()
                       << "\" is not defined.";
  }

  virtual void AllReduceCoalesce(const NDArrayList& inputs,
                                 NDArrayList& outputs,
                                 NDArray contiguous_buffers,
                                 ReductionType red_type = kSUM) {
    HT_NOT_IMPLEMENTED << "AllReduce fn of backend \"" << backend()
                       << "\" is not defined.";
  }

  virtual void AlltoAll(const NDArray& input, NDArray& output) {
    HT_NOT_IMPLEMENTED << "AlltoAll fn of backend \"" << backend()
                       << "\" is not defined.";
  }

  virtual void Reduce(const NDArray& input, NDArray& output, int reducer,
                      ReductionType red_type = kSUM) {
    HT_NOT_IMPLEMENTED << "Reduce fn of backend \"" << backend()
                       << "\" is not defined.";
  }

  virtual void AllGather(const NDArray& input, NDArray& output, int32_t gather_dim = 0) {
    HT_NOT_IMPLEMENTED << "AllGather fn of backend \"" << backend()
                       << "\" is not defined.";
  }

  virtual void ReduceScatter(const NDArray& input, NDArray& output,
                             int32_t scatter_dim = 0, ReductionType red_type = kSUM) {
    HT_NOT_IMPLEMENTED << "ReduceScatter fn of backend \"" << backend()
                       << "\" is not defined.";
  }

  virtual void Gather(const NDArray& input, NDArray& output, int gatherer) {
    HT_NOT_IMPLEMENTED << "Gather fn of backend \"" << backend()
                       << "\" is not defined.";
  }

  virtual void Scatter(const NDArray& input, NDArray& output, int scatterer) {
    HT_NOT_IMPLEMENTED << "Scatter fn of backend \"" << backend()
                       << "\" is not defined.";
  }

  virtual void AllToAll(const NDArrayList& inputs, NDArrayList& outputs) {
    HT_NOT_IMPLEMENTED << "AllToAll fn of backend \"" << backend()
                       << "\" is not defined.";
  }

  virtual void Send(const NDArray& data, int receiver) {
    HT_NOT_IMPLEMENTED << "Send fn of backend \"" << backend()
                       << "\" is not defined.";
  }

  virtual void Recv(NDArray& data, int sender) {
    HT_NOT_IMPLEMENTED << "Recv fn of backend \"" << backend()
                       << "\" is not defined.";
  }

  virtual CommTask ISend(const NDArray& data, int receiver) {
    HT_NOT_IMPLEMENTED << "ISend fn of backend \"" << backend()
                       << "\" is not defined.";
  }

  virtual CommTask IRecv(NDArray& data, int sender) {
    HT_NOT_IMPLEMENTED << "IRecv fn of backend \"" << backend()
                       << "\" is not defined.";
  }

  virtual void BatchedISendIRecv(const std::vector<CommTask>& tasks) {
    HT_NOT_IMPLEMENTED << "BatchedISendIRecv fn of backend \"" << backend()
                       << "\" is not defined.";
  }  

  virtual void Barrier(bool sync = false) {
    HT_NOT_IMPLEMENTED << "Barrier fn of backend \"" << backend()
                       << "\" is not defined.";
  }

  virtual void Sync() = 0;

  int rank() const noexcept {
    return _rank;
  }

  int size() const noexcept {
    return _size;
  }

  const std::vector<int>& world_ranks() const noexcept {
    return _world_ranks;
  }

  int group_to_world_rank(int group_rank) const {
    HT_ASSERT(group_rank >= 0 && group_rank < size())
      << "Invalid rank inside group. Expected to be within "
      << "[" << 0 << ", " << size() << "). "
      << "Got " << group_rank << ".";
    return _world_ranks[group_rank];
  }

  int world_to_group_rank(int world_rank) const {
    auto it = _world_to_group_mapping.find(world_rank);
    HT_ASSERT(it != _world_to_group_mapping.end())
      << "Rank " << world_rank << " does not belong to the current group.";
    return it->second;
  }

  const Stream& stream() const {
    return _stream;
  }

  virtual std::string backend() const = 0;

 protected:
  Stream _stream;
  std::vector<int> _world_ranks;
  std::map<int, int> _world_to_group_mapping;
  int _rank;
  int _size;
};

template <typename CommGroupDef>
class CommGroupWrapper : public shared_ptr_wrapper<CommGroupDef> {
 protected:
  template <typename DerivedCommGroupDef>
  CommGroupWrapper(std::shared_ptr<DerivedCommGroupDef> ptr)
  : shared_ptr_wrapper<CommGroupDef>() {
    static_assert(
      std::is_base_of<CommGroupDef, DerivedCommGroupDef>::value,
      "Template DerivedCommGroupDef is not derived from CommGroupDef.");
    this->_ptr = ptr;
  }

 public:
  CommGroupWrapper() = default;
  using shared_ptr_wrapper<CommGroupDef>::operator=;
  template <typename DerivedCommGroupDef>
  friend class CommGroupWrapper;
  template <typename DerivedCommGroupDef>
  CommGroupWrapper(const CommGroupWrapper<DerivedCommGroupDef>& comm_group)
  : shared_ptr_wrapper<CommGroupDef>() {
    static_assert(std::is_base_of<CommGroupDef, DerivedCommGroupDef>::value,
                  "Tempalte DerivedCommGroupDef is not derived from OpDef");
    this->_ptr = comm_group._ptr;
  }
};

using CommunicationGroup = CommGroupWrapper<CommunicationGroupDef>;

// The following helper functions are implemented in `mpi_comm_group.cc`
// since we rely on MPI to get the world rank and size now.
int GetWorldRank();
int GetWorldSize();
std::vector<int> GetWorldRanks();
int GetGroupRank(const std::vector<int>& world_ranks);
std::shared_ptr<DeviceClientImpl> GetLocalClient();
void SetUpDeviceMappingWithAssignedLocalDeviceOnce(const Device& local_device);
Device SetUpDeviceMappingAndAssignLocalDeviceOnce(
  const std::map<DeviceType, int>& resources = {{kCUDA, 8}},
  const std::vector<int64_t>& device_idxs = {},
  const std::string server_address = "127.0.0.1:23457");
bool IsGlobalDeviceGroupReady();
const DeviceGroup& GetGlobalDeviceGroup();
const Device& GetLocalDevice();
int GetRankOfLocalHost();
int DeviceToWorldRank(const Device& device);
const Device& WorldRankToDevice(int rank);
std::vector<int> DeviceGroupToWorldRanks(const DeviceGroup& device_group);

} // namespace comm
} // namespace impl
} // namespace hetu
