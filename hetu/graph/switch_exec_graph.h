#pragma once

#include "hetu/common/macros.h"
#include "hetu/core/ndarray.h"
#include "hetu/core/dtype.h"
#include "hetu/graph/graph.h"
#include "hetu/graph/define_and_run_graph.h"
#include "hetu/graph/executable_graph.h"
#include "hetu/graph/tensor.h"
#include "hetu/graph/operator.h"
#include "hetu/graph/init/initializer.h"
#include "hetu/impl/communication/comm_group.h"
#include "hetu/impl/communication/nccl_comm_group.h"
#include <nccl.h>

namespace hetu {
namespace graph {

using ExecGraphPair = std::pair<std::shared_ptr<ExecutableGraph>, std::shared_ptr<ExecutableGraph>>;
using Device2Val = std::unordered_map<Device, size_t>;
using DevicePair2Val = std::unordered_map<std::pair<Device, Device>, size_t>;
using Device2DTListPairMap = std::unordered_map<Device, std::pair<std::vector<Device>, std::vector<Tensor>>>;

std::ostream& operator<<(std::ostream& os, const SwitchExecGraph& switcher);

enum class SWITCH_ALGORITHM_LEVEL : int8_t {
  FCFS = 0,
  ROUND_ROBIN,
  MULTI_NODE_ROUND_ROBIN,
  GREEDY,
  NEW_GREEDY
};

enum class SWITCH_PROFILE_LEVEL : int8_t {
  TRACE = 0,
  MEMORY,
  NVLINK,
  TIME,
  INFO,
};

enum class SWITCH_MODE : int8_t {
  SWITCH_ORIGIN_PARAM = 0,
  SWITCH_TRANSFER_PARAM,
  SWITCH_ORIGIN_PARAM_AND_OPTIMIZER,
  SWITCH_CURRENT_GRAD,
  SWITCH_ACCUMULATE_GRAD
};

enum class SWITCH_LEVEL : int8_t {
  EXEC = 0,
  TOPO,
};

enum class P2P_ROUTE_LEVEL : int8_t {
  NVLINK = 0,
  PCIE,
  NET
};

class P2PRoute {
  public:
    P2PRoute(const P2P_ROUTE_LEVEL& route_level = P2P_ROUTE_LEVEL::NVLINK):
      _route_level(route_level) {   
      }

    const P2P_ROUTE_LEVEL& route_level() const {
      return _route_level;
    }
  
  protected:
    P2P_ROUTE_LEVEL _route_level;
};

class ParamBuffer {
  protected:
    friend class SwitchExecGraph;

  public:
    ParamBuffer(const std::string name = {}, 
                const TensorList& tensor_list = {}):
      _name(name),
      _tensor_list(tensor_list) {
      for (const auto& tensor : _tensor_list) {
        if (_dtype == DataType::UNDETERMINED) {
          _dtype = tensor->dtype();
        } else {
          HT_ASSERT(_dtype == tensor->dtype())
            << "ParamBuffer " << _name << " dtype should be consistent"
            << ", but tensor " << tensor << " dtype is " << tensor->dtype() 
            << " and buffer dtype is " << _dtype;
        }
        HT_ASSERT(_tensor_offset_mapping.find(tensor->id()) == _tensor_offset_mapping.end())
          << "tensor should be unique in the ParamBuffer" << _name
          << ", but there are multiple " << tensor;
        _tensor_offset_mapping[tensor->id()] = _buffer_size;
        _buffer_size += tensor->numel() * DataType2Size(tensor->dtype());
      }
    }

    Tensor GetTensor(const TensorId tensor_id) {
      HT_ASSERT(HasTensor(tensor_id))
        << "tensor is not in the ParamBuffer " << _name;
      for (auto& tensor : _tensor_list) {
        if (tensor->id() == tensor_id) {
          return tensor;
        }
      }
      __builtin_unreachable();
    }

    bool HasTensor(const TensorId tensor_id) {
      return _tensor_offset_mapping.find(tensor_id) != _tensor_offset_mapping.end();
    }

    bool HasTensor(const Tensor& tensor) {
      return _tensor_offset_mapping.find(tensor->id()) != _tensor_offset_mapping.end();
    }

    void AddTensor(const Tensor& tensor) {
      HT_ASSERT(_tensor_offset_mapping.find(tensor->id()) == _tensor_offset_mapping.end())
        << "tensor should be unique in the ParamBuffer " << _name
        << ", but there are multiple " << tensor;
      if (_dtype == DataType::UNDETERMINED) {
        _dtype = tensor->dtype();
      } else {
        HT_ASSERT(_dtype == tensor->dtype())
          << "ParamBuffer " << _name << " dtype should be consistent"
          << ", but tensor " << tensor << " dtype is " << tensor->dtype() 
          << " and buffer dtype is " << _dtype;
      }
      _tensor_list.push_back(tensor);
      _tensor_offset_mapping[tensor->id()] = _buffer_size;
      _buffer_size += tensor->numel() * DataType2Size(tensor->dtype());
    }

    bool IsEmpty() const {
      return _tensor_list.empty();
    }

    bool IsEqual(const ParamBuffer& param_buffer) {
      if (_dtype != param_buffer._dtype) {
        return false;
      }
      auto len = _tensor_list.size();
      if (len != param_buffer._tensor_list.size()) {
        return false;
      }
      for (size_t i = 0; i < len; i++) {
        if (_tensor_list[i].get() != param_buffer._tensor_list[i].get()) {
          return false;
        }
      }
      HT_ASSERT(_buffer_size == param_buffer._buffer_size)
        << "the total buffer size should be equal";
      return true;
    }

    bool IsAllocated() const {
      return _is_allocated;
    }

    bool IsAuxiliary() const {
      return _is_auxiliary;
    }

    DataType dtype() const {
      return _dtype;
    }

    Stream stream() const {
      HT_ASSERT(_is_allocated == true)
        << "please ensure you've alloc the buffer " << _name << " in advance";
      return _stream;
    }

    size_t size() const {
      return _buffer_size;
    }

    const TensorList& tensor_list() const {
      return _tensor_list;
    }

    void* AsRawPtr() {
      HT_ASSERT(_is_allocated == true)
        << "please ensure you've alloc the buffer " << _name << " in advance";
      return _raw_ptr;
    }

    std::shared_ptr<NDArrayStorage> AsStorage() {
      HT_ASSERT(_is_allocated == true)
        << "please ensure you've alloc the buffer " << _name << " in advance";
      return _storage;
    }

    NDArray AsNDArray() {
      HT_ASSERT(_is_allocated == true)
        << "please ensure you've alloc the buffer " << _name << " in advance";
      // 设置成一个一维的NDArray
      // Question: device可不可能是unlocal device
      auto meta = NDArrayMeta().set_dtype(_dtype)
                               .set_device(hetu::impl::comm::GetLocalDevice())
                               .set_shape({_buffer_size / DataType2Size(_dtype)});
      return NDArray(meta, _storage);
    }

    const size_t GetByteOffest(const Tensor& tensor) const {
      auto it = _tensor_offset_mapping.find(tensor->id());
      HT_ASSERT(it != _tensor_offset_mapping.end())
        << "Can't find tensor " << tensor << " in the ParamBuffer " << _name;
      return it->second;
    }

    const size_t GetElementOffset(const Tensor& tensor) const {
      auto it = _tensor_offset_mapping.find(tensor->id());
      HT_ASSERT(it != _tensor_offset_mapping.end())
        << "Can't find tensor " << tensor << " in the ParamBuffer " << _name;
      return it->second / DataType2Size(_dtype);
    }

    const size_t GetTensorOffest(const Tensor& tensor) const {
      auto len = _tensor_list.size();
      for (size_t i = 0; i < len; ++i) {
        if (_tensor_list[i]->id() == tensor->id()) {
          return i;
        }
      }
      HT_RUNTIME_ERROR << "Can't find tensor " << tensor << " in the ParamBuffer " << _name;
    }

    void Alloc(const Stream& stream, 
               bool use_nccl = false, 
               ncclComm_t comm = nullptr,
               bool use_caching_mempool = true,
               bool use_async = false); // stream is unused actually (cudaMallocAsync is tooooo slow!)

    void Free();
    
    void Bind(const std::shared_ptr<NDArrayStorage>& storage); // bind to a customized storage rather than alloc by itself

  protected:
    // basic attributes
    std::string _name;
    TensorList _tensor_list;
    bool _is_allocated{false};
    bool _is_auxiliary{false};
    DataType _dtype{DataType::UNDETERMINED};
    std::unordered_map<TensorId, size_t> _tensor_offset_mapping; // 这里的offset是字节数
    Stream _stream; // 记录在哪个stream上alloc的（目前deprecated）
    std::shared_ptr<NDArrayStorage> _storage; // high-level hetu mempool api
    void* _raw_ptr; // low-level cuda/nccl mem api
    size_t _buffer_size{0};

    // profile related
    size_t _alloc_time{0};
    size_t _free_time{0};
};

class ParamBuckets {
  public:
    ParamBuckets(const std::string& name = {},
                 size_t buckets_size = 10):
      _name(name),
      _buckets_size(buckets_size) {
      for (size_t i = 0; i <= _buckets_size; i++) {
        _buckets.emplace_back(std::make_shared<ParamBuffer>(_name + "_bucket_" + std::to_string(i)));
      }
    }

    size_t buckets_size() {
      return _buckets_size;
    }

    size_t GetSuggestedBucketId(const Tensor& tensor);

    std::shared_ptr<ParamBuffer> GetBucket(size_t id) {
      HT_ASSERT(id < _buckets_size)
        << "id is out of range";
      return _buckets.at(id);
    }

    void AddTensor(const Tensor& tensor) {
      size_t bucket_num = GetSuggestedBucketId(tensor);
      HT_ASSERT(bucket_num < _buckets_size) 
        << "id is out of range";
      _buckets.at(bucket_num)->AddTensor(tensor);
    }

    bool HasTensor(const Tensor& tensor) {
      for (const auto& bucket : _buckets) {
        if (bucket->HasTensor(tensor)) {
          return true;
        }
      }
      return false;
    }

    size_t GetTensorBucketId(const Tensor& tensor) {
      for (size_t i = 0; i < _buckets_size; i++) {
        if (_buckets.at(i)->HasTensor(tensor)) {
          return i;
        }
      }
      HT_RUNTIME_ERROR << "Can't find tensor " << tensor << " in the ParamBuckets " << _name;
    }

    std::shared_ptr<ParamBuffer> GetTensorBucket(const Tensor& tensor) {
      for (size_t i = 0; i < _buckets_size; i++) {
        if (_buckets.at(i)->HasTensor(tensor)) {
          return _buckets.at(i);
        }
      }
      HT_RUNTIME_ERROR << "Can't find tensor " << tensor << " in the ParamBuckets " << _name;
    }

    void AllocAll(const Stream& stream) {
      for (const auto& bucket : _buckets) {
        bucket->Alloc(stream);
      }
    }

  protected:
    size_t _buckets_size;
    std::string _name;
    std::vector<std::shared_ptr<ParamBuffer>> _buckets;
};

class ParamSlice {
  protected:
    friend class SwitchExecGraph;

  public:
    ParamSlice(const TensorName& block_name, 
               const HTShape& slice_shape,
               const std::vector<int32_t>& slice_num,
               SwitchExecGraph* switcher): 
      _block_name(block_name),
      _slice_shape(slice_shape),
      _slice_num(slice_num),
      _switcher(switcher) {
    }

    const std::string name() const {
      std::string suffix = "_slice";
      for (const auto& x : _slice_num) {
        suffix += "_" + std::to_string(x);
      }
      return _block_name + suffix;
    }

    const size_t numel() const {
      if (_slice_shape.size() == 0) {
        HT_LOG_WARN << "ParamSlice with 0 numel";
        return 0;
      }
      size_t numel = 1;
      for(auto s : _slice_shape) {
        numel *= s;
      }
      return numel;
    }

    const Tensor& OwnedSliceInst(size_t idx) const {
      HT_ASSERT(idx < _owned_slice_instances.size())
        << "idx out of range";
      return _owned_slice_instances[0];
    }

    const Tensor& NeededSliceInst(size_t idx) const {
      HT_ASSERT(idx < _needed_slice_instances.size())
        << "idx out of range";
      return _needed_slice_instances[0];
    }

    void AddOwnedSliceInst(const Device& device, const Tensor& tensor);

    void AddNeededSliceInst(const Device& device, const Tensor& tensor);

    void ParamSliceComm(Device2DTListPairMap& send_mapping, Device2DTListPairMap& recv_mapping);

  protected:
    TensorName _block_name;
    // 在一个block中的slice编号
    // 例如block有3*2*5个slice
    // 那么一个合法的_slice_num就是{2,1,3}
    HTShape _slice_shape;
    std::vector<int32_t> _slice_num; 
    SwitchExecGraph* _switcher;

    TensorList _owned_slice_instances;
    TensorList _needed_slice_instances;
    std::vector<Device> _owned_devices;
    std::vector<Device> _needed_devices;

    // level 0, round-robin alg
    size_t _round_robin = 0;
};

class ParamBlock {
  protected:
    friend class SwitchExecGraph;

  public:
    ParamBlock(const TensorName& block_name, 
               const std::vector<int32_t>& block_shape,
               const HTShape& slice_shape,
               SwitchExecGraph* switcher):
      _block_name(block_name), 
      _block_shape(block_shape),
      _slice_shape(slice_shape),
      _switcher(switcher) {
    }

    const std::string name() const {
      return _block_name;
    }

    const std::vector<int32_t>& BlockShape() const {
      return _block_shape;
    }

    const HTShape& SliceShape() const {
      return _slice_shape;
    }

    std::vector<std::shared_ptr<ParamSlice>>& GetParamSlices() {
      return _param_slices;
    }

    std::shared_ptr<ParamSlice>& GetParamSlice(const std::vector<int32_t>& slice_num) {
      size_t size = slice_num.size();
      HT_ASSERT(size == _block_shape.size() && size > 0) 
        << "size should be equal to block shape size and non-zero";
      size_t cnt = 1, sum = 0;
      for (int32_t i = size - 1; i >= 0; --i) {
        HT_ASSERT(slice_num[i] < _block_shape[i]) 
          << _block_name << ": slice_num dim " << i << " is out of range"
          << ", slice_num = " << slice_num << " and block_shape = " << _block_shape;
        sum += slice_num[i] * cnt;
        cnt *= _block_shape[i];
      }
      HT_ASSERT(sum < _param_slices.size()) 
        << "slice is out of range";
      return _param_slices[sum];
    }

    void ParamBlockComm(Device2DTListPairMap& send_mapping, Device2DTListPairMap& recv_mapping);

  protected:
    std::vector<int32_t> _block_shape; // # of the abstract slices
    HTShape _slice_shape; // # of the actual elements
    TensorName _block_name;
    SwitchExecGraph* _switcher;

    std::vector<std::shared_ptr<ParamSlice>> _param_slices; 
};

class SwitchExecGraph {
  protected:
    friend class DefineAndRunGraph;
    friend class ExecutableGraph;
    friend class ParamBlock;
    friend class ParamSlice;

  public:
    SwitchExecGraph() {}
    SwitchExecGraph(DefineAndRunGraph* define_graph, 
                    size_t plan_before, 
                    size_t plan_after,
                    DataType dtype,
                    int32_t bucket_num = -1,
                    std::unordered_set<Device> comm_set = {},
                    std::unordered_map<DataType, DataType> dtype_to_filter_dtype = {}):
      _define_graph(define_graph),
      _dtype(dtype),
      _bucket_num(bucket_num),
      _comm_set(comm_set) {
      _switch_plan_pair = std::make_pair(plan_before, plan_after);
      _switch_graph_pair = std::make_pair(define_graph->GetPlan(plan_before).exec_graph, 
                                          define_graph->GetPlan(plan_after).exec_graph); 
      const auto& define_graph_params_unfiltered = define_graph->params();
      const auto& define_graph_params_and_opt_vars_unfiltered = define_graph->params_and_opt_vars();
      TensorCRefList define_graph_params, define_graph_params_and_opt_vars;
      // 筛选出dtype类型的
      auto it = dtype_to_filter_dtype.find(_dtype);
      if (it != dtype_to_filter_dtype.end()) {
        auto filter_dtype = it->second;
        for (const auto& param_and_opt_var_unfiltered_ref : define_graph_params_and_opt_vars_unfiltered) {
          if (param_and_opt_var_unfiltered_ref.get()->dtype() == filter_dtype) {
            define_graph_params_and_opt_vars.emplace_back(param_and_opt_var_unfiltered_ref);
          }
        }
        for (const auto& param_unfiltered_ref : define_graph_params_unfiltered) {
          if (param_unfiltered_ref.get()->dtype() == filter_dtype) {
            define_graph_params.emplace_back(param_unfiltered_ref);
          }
        }
      }
      // 分别设置_define_graph_params和_define_graph_params_and_opt_vars
      _define_graph_params = define_graph_params;
      if (_bucket_num == -1) {
        _define_graph_params_and_opt_vars = define_graph_params_and_opt_vars;
      } 
      // 筛选出bucket中含有的param和opt var
      else {
        _define_graph_params_and_opt_vars = TensorCRefList();
        HT_ASSERT(_switch_graph_pair.first->_use_origin_param_and_optimizer_buckets
                  && _switch_graph_pair.second->_use_origin_param_and_optimizer_buckets)
          << "_bucket_num can only used when the _use_origin_param_and_optimizer_buckets is turned on";
        for (const auto& param_and_opt_var_ref : define_graph_params_and_opt_vars) {
          auto before_it = define_graph->GetPlan(plan_before).tensor_to_exec_tensor_mapping.find(param_and_opt_var_ref.get()->id());
          auto after_it = define_graph->GetPlan(plan_after).tensor_to_exec_tensor_mapping.find(param_and_opt_var_ref.get()->id());
          // e.g. lm_head_weight in gpt 
          // defined but not used in exec graph
          if (before_it == define_graph->GetPlan(plan_before).tensor_to_exec_tensor_mapping.end()
              && after_it == define_graph->GetPlan(plan_after).tensor_to_exec_tensor_mapping.end()) {
            continue;
          }
          HT_ASSERT(before_it != define_graph->GetPlan(plan_before).tensor_to_exec_tensor_mapping.end()
                    && after_it != define_graph->GetPlan(plan_after).tensor_to_exec_tensor_mapping.end())
            << "Cannot find " << param_and_opt_var_ref;
          size_t before_bucket_id = _switch_graph_pair.first->_origin_param_and_optimizer_buckets_map[_dtype]->GetSuggestedBucketId(before_it->second);
          size_t after_bucket_id = _switch_graph_pair.second->_origin_param_and_optimizer_buckets_map[_dtype]->GetSuggestedBucketId(after_it->second);
          HT_ASSERT(before_bucket_id == after_bucket_id)
            << "Currently only support same bucket for same define graph tensor";
          if (before_bucket_id == _bucket_num) {
            _define_graph_params_and_opt_vars.emplace_back(param_and_opt_var_ref);
          }
        }
      }
      // 环境变量
      char* algorithm_env = std::getenv("HETU_SWITCH_ALGORITHM");
      if (algorithm_env != nullptr) {
        std::string algorithm_level = algorithm_env;
        std::transform(algorithm_level.begin(), algorithm_level.end(), algorithm_level.begin(), ::toupper);
        if (algorithm_level == "NEW_GREEDY") {
          _algorithm_level = SWITCH_ALGORITHM_LEVEL::NEW_GREEDY;
        } else if (algorithm_level == "GREEDY") {
          _algorithm_level = SWITCH_ALGORITHM_LEVEL::GREEDY;
        } else if (algorithm_level == "MULTI_NODE_ROUND_ROBIN") {
          _algorithm_level = SWITCH_ALGORITHM_LEVEL::MULTI_NODE_ROUND_ROBIN;
        } else if (algorithm_level == "ROUND_ROBIN") {
          _algorithm_level = SWITCH_ALGORITHM_LEVEL::ROUND_ROBIN;
        } else if (algorithm_level == "FCFS") {
          _algorithm_level = SWITCH_ALGORITHM_LEVEL::FCFS;
        } else {
          HT_RUNTIME_ERROR << "NotImplementedError";
        }
      }
      char* profile_env = std::getenv("HETU_SWITCH_PROFILE");
      if (profile_env != nullptr) {
        std::string profile_level = profile_env;
        std::transform(profile_level.begin(), profile_level.end(), profile_level.begin(), ::toupper);
        if (profile_level == "INFO") {
          _profile_level = SWITCH_PROFILE_LEVEL::INFO;
        } else if (profile_level == "TIME") {
          _profile_level = SWITCH_PROFILE_LEVEL::TIME;
        } else if (profile_level == "NVLINK") {
          _profile_level = SWITCH_PROFILE_LEVEL::NVLINK;
        } else if (profile_level == "MEMORY") {
          _profile_level = SWITCH_PROFILE_LEVEL::MEMORY;
        } else if (profile_level == "TRACE") {
          _profile_level = SWITCH_PROFILE_LEVEL::TRACE;
        } else {
          HT_RUNTIME_ERROR << "NotImplementedError";
        }
      }
    }

    void RecordTensorInfo(const Tensor& tensor, const std::string& info) {
      if (_info_mapping.find(tensor->id()) != _info_mapping.end()) {
        HT_ASSERT(_info_mapping[tensor->id()] == info)
          << "tensor " << tensor << " is already existed in the info mapping of the switcher"
          << ", the new info is conflicted with the existed info";
      }
      _info_mapping[tensor->id()] = info;
    }

    const ExecGraphPair& SwitchGraphPair() const {
      return _switch_graph_pair;
    }
    
    void SwitchParams(SWITCH_MODE switch_mode = SWITCH_MODE::SWITCH_ORIGIN_PARAM,
                      SWITCH_LEVEL switch_level = SWITCH_LEVEL::EXEC,
                      std::string switch_name = "switch",
                      const StreamIndex comp_stream_idx = kSwitchComputingStream,
                      const StreamIndex comm_stream_idx = kSwitchCollectiveStream);

  protected:
    void CreateParamBlock(ParamBlock& block,
                          std::vector<int32_t>& slice_num, 
                          const TensorName& block_name,
                          int32_t dim);

    void MakeAllParamSlices(const Tensor& param, ParamBlock& block, 
                            const Device& device, const DeviceGroup& group,
                            std::vector<int32_t>& slice_num, std::vector<int32_t>& slice_relative_num,
                            const std::unordered_map<int32_t, int32_t>& state,
                            const std::vector<int32_t>& multiple, int32_t dim,
                            bool is_uncontiguous, int32_t uncontiguous_ordinal, 
                            int32_t uncontiguous_multiple, int32_t uncontiguous_slice_multiple,
                            const StreamIndex comp_stream_idx);

    Tensor MergeAllParamSlices(const Tensor& param, ParamBlock& block, 
                               const Device& device, const DeviceGroup& group,
                               std::vector<int32_t>& slice_num, std::vector<int32_t>& slice_relative_num,
                               const std::unordered_map<int32_t, int32_t>& state,
                               const std::vector<int32_t>& multiple, int32_t dim,
                               bool is_uncontiguous, int32_t uncontiguous_ordinal, 
                               int32_t uncontiguous_multiple, int32_t uncontiguous_slice_multiple,
                               const StreamIndex comp_stream_idx);
    
    void MakeCommGraph(SWITCH_MODE switch_mode,
                       SWITCH_LEVEL switch_level,
                       const StreamIndex comp_stream_idx,
                       const StreamIndex comm_stream_idx);

    void BufferBatchedIsendIrecvExec(const hetu::impl::comm::NCCLCommunicationGroup& comm_nccl_group);
    
    void BufferBatchedIsendIrecv(const Operator& op,
                                 const hetu::impl::comm::NCCLCommunicationGroup& comm_nccl_group,
                                 Tensor2NDArrayMap& tensor2data,
                                 Tensor2IntMap& tensor2degrees,
                                 const StreamIndex comp_stream_idx);

    void SwitchParam(const DistributedStatesUnion& src_ds_union, const DeviceGroupUnion& src_group_union,
                     const DistributedStatesUnion& dst_ds_union, const DeviceGroupUnion& dst_group_union,
                     const Tensor& comm_input, const Tensor& after_param, const HTShape& global_shape,
                     const StreamIndex comp_stream_idx);

    void ProfileRunningDetails();

  protected:
    // basic attributes
    DataType _dtype; // 要切换的数据类型
    int32_t _bucket_num; // 要切换的bucket编号
    DefineAndRunGraph* _define_graph; // 定义图
    TensorCRefList _define_graph_params; // 定义图的params tensor
    TensorCRefList _define_graph_params_and_opt_vars; // 定义图的params以及optimizer variables的tensor
    std::pair<size_t, size_t> _switch_plan_pair; // 需要切换的两个exec graph plan的编号
    ExecGraphPair _switch_graph_pair; // 需要切换的两个exec graph的指针

    // comm graph related
    std::shared_ptr<ExecutableGraph> _comm_graph; // 为了应对切换过程中的复杂通信情况而建立的执行图 
    std::unordered_set<Device> _comm_set; // 参与通信图的所有devices
    OpRefList _comm_topo; // 该图的local_topo
    Tensor2ShapeMap _comm_shape_plan; // 该图所有tensor的运行时的shape
    FeedDict _comm_feed_dict; // 该图的输入
    Tensor2TensorMap _comm_feed_dict_mapping; // 该图的输入到before graph的映射
    TensorList _comm_results; // 该图通信的结果，与_define_graph_params一一对应
    Tensor2TensorMap _comm_results_mapping; // 该图的输出到after graph的映射
    TensorList _dummy_links; // 只有send没有recv时BatchedISendIRecvOp的输出dummy tensor需要被记录并在之后fetch

    // comm plan related
    SWITCH_ALGORITHM_LEVEL _algorithm_level = SWITCH_ALGORITHM_LEVEL::NEW_GREEDY; // 采用的算法
    DevicePair2Val _p2p_val_mapping; // deprecated: 记录了每两个device之间的p2p通信通路的总value（目前value是指次数）
    // 同一个device的intra和inter的通信可以overlap
    Device2Val _intra_device_val_mapping; // 记录intra node的device通信的value（目前value指发送数据的量除以带宽）（**热切换场景下接收数据的量固定）
    Device2Val _inter_device_val_mapping; // 记录inter node的device通信的value（目前value指发送数据的量除以带宽）（**热切换场景下接收数据的量固定）
    Device2DTListPairMap _send_mapping; // 记录了每个device要send的(device, tensor)的pair
    Device2DTListPairMap _recv_mapping; // 记录了每个device要recv的(device, placeholder的tensor（之后会替换）)的pair
    std::vector<std::shared_ptr<ParamBlock>> _param_blocks; // 记录了graph所包含的所有的抽象ParamBlock

    // memory optimization related
    std::unordered_map<Device, std::shared_ptr<ParamBuffer>> _send_buffers; // 记录通信时给每个device聚合发送时所用的buffer
    std::unordered_map<Device, std::shared_ptr<ParamBuffer>> _recv_buffers; // 记录通信时从每个device聚合接收时所用的buffer
    std::vector<std::unique_ptr<hetu::impl::CUDAEvent>> _buffer_transfer_events; // 记录将原先的param构成一长条buffer的events
    bool _use_concat_buffer = true; // 是否对concat算子的输出做一个buffer
    std::shared_ptr<ParamBuffer> _concat_buffer; // 通信后从_recv_buffers里concat形成的所有param构成的buffer

    // profile related
    SWITCH_PROFILE_LEVEL _profile_level = SWITCH_PROFILE_LEVEL::INFO; // profile的粒度（开启后会进行同步，因此端到端速度可能会变慢）
    Tensor2StringMap _info_mapping; // 记录tensor到相应param slice名称的映射（只针对BatchedISendIRecvOp的send部分的tensor）
};

class ComplexExecComm : public SwitchExecGraph {
  public:
    ComplexExecComm(const Operator& comm_op, const CommOpInfo& comm_info): 
      SwitchExecGraph(),
      _is_instantiated(false),
      _comm_op(comm_op),
      _comm_info(comm_info) {
      _algorithm_level = SWITCH_ALGORITHM_LEVEL::NEW_GREEDY;
    }

    Tensor Instantiate();

  protected:
    bool _is_instantiated;
    Operator _comm_op;
    CommOpInfo _comm_info;
};

} // namespace graph
} // namespace hetu