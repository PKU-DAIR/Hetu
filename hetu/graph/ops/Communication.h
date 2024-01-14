#pragma once

#include "hetu/graph/operator.h"
#include "hetu/graph/graph.h"
#include "hetu/graph/utils/tensor_utils.h"
#include "hetu/graph/distributed_states.h"
#include "hetu/impl/communication/comm_group.h"

namespace hetu {
namespace graph {

class CommOpImpl;
class AllReduceOpImpl;
class P2PSendOpImpl;
class P2PRecvOpImpl;
class BatchedISendIRecvOpImpl;
class AllGatherOpImpl;
class ReduceScatterOpImpl;
class ScatterOpImpl;

class CommOpImpl final: public OpInterface {
 public:
  // dst_group is only for exec graph instantiate, at this time there will only exists one ds strategy
  // also, multi_dst_ds should contains only one dst_ds for comm op which created in exec graph instantiate
  CommOpImpl(DistributedStatesList multi_dst_ds, DeviceGroup dst_group = DeviceGroup(), 
             ReductionType red_type = kSUM) : OpInterface(quote(CommOp)), 
             _multi_dst_ds(std::move(multi_dst_ds)), _dst_group(std::move(dst_group)), _red_type(red_type) {}      

  uint64_t op_indicator() const noexcept override {
    return COMM_OP;
  }  

 protected:
  bool DoMapToParallelDevices(Operator& op,
                              const DeviceGroup& pg) const override;

  bool DoInstantiate(Operator& op, const Device& placement,
                     StreamIndex stream_index) const override;                              

  std::vector<NDArrayMeta> 
  DoInferMeta(const TensorList& inputs) const override;

  void DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                      const OpMeta& op_meta) const override;

  TensorList DoGradient(Operator& op,
                        const TensorList& grad_outputs) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes,
                           RuntimeContext& runtime_ctx) const override;

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& runtime_ctx) const {}

 public: 
  const DistributedStates& get_dst_distributed_states(Operator& op) const {
    auto& graph = op->graph();
    HT_ASSERT(_multi_dst_ds.size() == 1 || _multi_dst_ds.size() == graph.NUM_STRATEGY)
      << "CommOp get dst ds error!";
    if (_multi_dst_ds.size() == 1) { // for comm op created in exec_graph, without multi ds
      return _multi_dst_ds[0];
    } else { // for comm op created in define_and_run_graph, with multi ds
      return _multi_dst_ds[graph.CUR_STRATEGY_ID];
    }
  }

  ReductionType reduction_type() const {
    return _red_type;
  }
  
  // placement group only for exec_graph
  const DeviceGroup& src_group(Operator& op) const {
    return op->input(0)->placement_group();
  }

  const DeviceGroup& dst_group(Operator& op) const {
    if (_dst_group.empty()) {
      return op->input(0)->placement_group();
    } else {
      return _dst_group;
    }
  }

  bool is_intra_group(Operator& op) const {
    return !is_inter_group(op);
  }

  bool is_inter_group(Operator& op) const {
    return src_group(op) != dst_group(op);
  }

  uint64_t get_comm_type(Operator& op);

  DeviceGroup get_devices_by_dim(Operator& op, int32_t dim) const; 

 protected:
  uint64_t _comm_type{UNKNOWN_OP};
  // DistributedStates _dst_ds;
  DistributedStatesList _multi_dst_ds;
  DeviceGroup _dst_group;
  ReductionType _red_type{kNONE}; // only used for AllReduce, ReduceScatter
};

Tensor MakeCommOp(Tensor input, DistributedStatesList multi_dst_ds, 
                  ReductionType red_type, OpMeta op_meta = OpMeta());

Tensor MakeCommOp(Tensor input, DistributedStatesList multi_dst_ds,
                  const std::string& mode, OpMeta op_meta = OpMeta());

Tensor MakeCommOp(Tensor input, DistributedStatesList multi_dst_ds, 
                  DeviceGroup dst_group, OpMeta op_meta = OpMeta());

Tensor MakeCommOp(Tensor input, DistributedStatesList multi_dst_ds, 
                  OpMeta op_meta = OpMeta());

class AllReduceOpImpl final : public OpInterface {
 public:
  AllReduceOpImpl(DeviceGroup comm_group, ReductionType red_type = kSUM, bool inplace = false)
  : OpInterface(quote(AllReduceOp)), _comm_group(comm_group), _red_type(red_type), _inplace(inplace) {
    HT_ASSERT(_comm_group.num_devices() >= 2)
             << "AllReduce requires two or more comm devices. Got " << _comm_group;
  }

  inline uint64_t op_indicator() const noexcept override {
    return _inplace ? ALL_REDUCE_OP | INPLACE_OP : ALL_REDUCE_OP;
  }

  inline bool inplace() const {
    return _inplace;
  }

  ReductionType reduction_type() const {
    return _red_type;
  }

 protected:
  bool DoMapToParallelDevices(Operator& op,
                              const DeviceGroup& pg) const override;

  bool DoInstantiate(Operator& op, const Device& placement,
                     StreamIndex stream_index) const override;

  std::vector<NDArrayMeta> 
  DoInferMeta(const TensorList& inputs) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes,
                           RuntimeContext& runtime_ctx) const override;

  NDArrayList DoCompute(Operator& op,
                        const NDArrayList& inputs,
                        RuntimeContext& ctx) const override;

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) const {};
  
  bool _inplace{false};

 public:
  const DeviceGroup& comm_group() const {
    return _comm_group;
  }

 protected:
  DeviceGroup _comm_group;
  ReductionType _red_type{kNONE};
};

Tensor MakeAllReduceOp(Tensor input, DeviceGroup comm_group, 
                       bool inplace = false, OpMeta op_meta = OpMeta());

Tensor MakeAllReduceOp(Tensor input, DeviceGroup comm_group, ReductionType red_type, 
                       bool inplace = false, OpMeta op_meta = OpMeta());

class P2PSendOpImpl final : public OpInterface {
 public:
  P2PSendOpImpl(DeviceGroup dst_group, int dst_device_index = -1)
  : OpInterface(quote(P2PSendOp)), _dst_group(std::move(dst_group)), 
    _dst_device_index(dst_device_index) {
    HT_ASSERT(!_dst_group.empty())
      << "Please provide the \"dst_group\" argument to indicate "
      << "the destination devices for P2PSend";
  }

  uint64_t op_indicator() const noexcept override {
    return PEER_TO_PEER_SEND_OP;
  }

 protected:
  bool DoMapToParallelDevices(Operator& op,
                              const DeviceGroup& pg) const override;

  bool DoInstantiate(Operator& op, const Device& placement,
                     StreamIndex stream_index) const override;

  std::vector<NDArrayMeta> 
  DoInferMeta(const TensorList& inputs) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes,
                           RuntimeContext& runtime_ctx) const override;  

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& runtime_ctx) const override;
                        
 public:
  const DeviceGroup& dst_group() const {
    return _dst_group;
  }

  int dst_device_index() const {
    return _dst_device_index;
  }  

 protected:
  DeviceGroup _dst_group;
  int _dst_device_index{-1};
};

Tensor MakeP2PSendOp(Tensor input, DeviceGroup dst_group, 
                     int dst_device_index = -1, OpMeta op_meta = OpMeta());

class P2PRecvOpImpl final : public OpInterface {
 public:
  P2PRecvOpImpl(DeviceGroup src_group, DataType dtype,
                HTShape shape, int src_device_index = -1)
  : OpInterface(quote(P2PRecvOp)), _src_group(std::move(src_group)), _dtype(dtype),
                _shape(std::move(shape)), _src_device_index(src_device_index) {
    HT_ASSERT(!_src_group.empty())
      << "Please provide the \"src_group\" argument to indicate "
      << "the source devices for P2PRecv";
    HT_ASSERT(!_shape.empty())
      << "P2P RecvOp require determined tensor shape to recv. Got empty shape param!";
    HT_ASSERT(_dtype != kUndeterminedDataType)
      << "Please specify data type for P2P communication";
  }

  uint64_t op_indicator() const noexcept override {
    return PEER_TO_PEER_RECV_OP;
  }

 protected:
  bool DoMapToParallelDevices(Operator& op,
                              const DeviceGroup& pg) const override;

  bool DoInstantiate(Operator& op, const Device& placement,
                     StreamIndex stream_index) const override;

  std::vector<NDArrayMeta> 
  DoInferMeta(const TensorList& inputs) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes,
                           RuntimeContext& runtime_ctx) const override;

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& runtime_ctx) const override;

 public:
  const DeviceGroup& src_group() const {
    return _src_group;
  }

  int src_device_index() {
    return _src_device_index;
  } 

 protected:
  DeviceGroup _src_group;
  int _src_device_index{-1};
  DataType _dtype;
  HTShape _shape;           
};

Tensor MakeP2PRecvOp(DeviceGroup src_group, DataType dtype,
                     HTShape shape, int src_device_index = -1, 
                     OpMeta op_meta = OpMeta());

class BatchedISendIRecvOpImpl final : public OpInterface {
 public:
  /*
  // symbolic shape constructor
  BatchedISendIRecvOpImpl(const std::vector<Device>& dst_devices, 
                          const SyShapeList& outputs_shape,
                          const std::vector<Device>& src_devices, 
                          const std::vector<Device>& comm_devices,
                          DataType dtype)
  : OpInterface(quote(BatchedISendIRecvOp)), _dst_devices(dst_devices), 
  _outputs_shape({}), _src_devices(src_devices), 
  _comm_devices(comm_devices), _dtype(dtype) {}
  */
  // fixed shape constructor
  BatchedISendIRecvOpImpl(std::vector<Device> dst_devices, 
                          HTShapeList outputs_shape,
                          std::vector<Device> src_devices, 
                          std::vector<Device> comm_devices,
                          DataType dtype)
  : OpInterface(quote(BatchedISendIRecvOp)), _dst_devices(std::move(dst_devices)), 
  _outputs_shape(std::move(outputs_shape)), _src_devices(std::move(src_devices)), 
  _comm_devices(std::move(comm_devices)), _dtype(dtype) {}

  uint64_t op_indicator() const noexcept override {
    return BATCHED_ISEND_IRECV_OP;
  }

 public:
  void print_mesg(Operator& op) {
    std::ostringstream os;
    os << "dst devices =";
    for (auto& d : _dst_devices) {
      os << " device_" << hetu::impl::comm::DeviceToWorldRank(d);
    }
    os << "src devices =";
    for (auto& s : _src_devices) {
      os << " device_" << hetu::impl::comm::DeviceToWorldRank(s);
    }
    HT_LOG_DEBUG << hetu::impl::comm::GetLocalDevice() 
                 << ": BatchedISendIRecvOp definition: " << op->name() << ": " << os.str();    
  }

  const std::vector<Device>& src_devices() const {
    return _src_devices;
  }

  std::vector<Device>& src_devices() {
    return _src_devices;
  }  

  const std::vector<Device>& dst_devices() const {
    return _dst_devices;
  }

  std::vector<Device>& dst_devices() {
    return _dst_devices;
  }

 protected:
  bool DoInstantiate(Operator& op, const Device& placement,
                     StreamIndex stream_index) const override;
                    
  std::vector<NDArrayMeta> 
  DoInferMeta(const TensorList& inputs) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes,
                           RuntimeContext& runtime_ctx) const override; 

  HTShapeList DoInferDynamicShape(Operator& op, const HTShapeList& input_shapes,
                           RuntimeContext& runtime_ctx) const override;   

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& runtime_ctx) const override;

 protected:
  std::vector<Device> _dst_devices; 
  std::vector<Device> _src_devices;
  std::vector<Device> _comm_devices;
  HTShapeList _outputs_shape;
  DataType _dtype;
};

Tensor MakeBatchedISendIRecvOp(TensorList inputs, 
                               std::vector<Device> dst_devices, 
                               HTShapeList outputs_shape, 
                               std::vector<Device> src_devices, 
                               std::vector<Device> comm_devices, 
                               DataType dtype, OpMeta op_meta = OpMeta());

class AllGatherOpImpl final : public OpInterface {
 public:
  AllGatherOpImpl(DeviceGroup comm_group)
  : OpInterface(quote(AllGatherOp)), _comm_group(std::move(comm_group)) {
    HT_ASSERT(_comm_group.num_devices() >= 2)
      << "AllGather requires two or more devices. Got " << _comm_group;
  }

  uint64_t op_indicator() const noexcept override {
    return ALL_GATHER_OP;
  }

 protected:
  bool DoMapToParallelDevices(Operator& op,
                              const DeviceGroup& pg) const override;

  bool DoInstantiate(Operator& op, const Device& placement,
                     StreamIndex stream_index) const override;

  std::vector<NDArrayMeta> 
  DoInferMeta(const TensorList& inputs) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes,
                           RuntimeContext& runtime_ctx) const override;  

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& runtime_ctx) const override;

 protected:
  DeviceGroup _comm_group;
};

Tensor MakeAllGatherOp(Tensor input, DeviceGroup comm_group,
                       OpMeta op_meta = OpMeta());

class ReduceScatterOpImpl final : public OpInterface {
 public:
  ReduceScatterOpImpl(DeviceGroup comm_group, ReductionType red_type = kSUM,
                      bool inplace = false) : OpInterface(quote(ReduceScatterOp)), 
    _comm_group(std::move(comm_group)), _red_type(red_type), _inplace(inplace) {
    HT_ASSERT(_comm_group.num_devices() >= 2)
      << "ReduceScatter requires two or more devices. Got " << _comm_group;          
  }

  inline bool inplace() const {
    return _inplace;
  }

  inline uint64_t inplace_pos() const {
    return 0;
  }

  inline bool inplace_at(size_t input_position) const override {
    return inplace() && input_position == inplace_pos();
  }

  inline uint64_t op_indicator() const noexcept override {
    return _inplace ? REDUCE_SCATTER_OP | INPLACE_OP : REDUCE_SCATTER_OP;
  }

  ReductionType reduction_type() const {
    return _red_type;
  }

  const DeviceGroup& comm_group() const {
    return _comm_group;
  } 

 protected:
  bool DoMapToParallelDevices(Operator& op,
                              const DeviceGroup& pg) const override;

  bool DoInstantiate(Operator& op, const Device& placement,
                     StreamIndex stream_index) const override;
                                                    
  std::vector<NDArrayMeta> 
  DoInferMeta(const TensorList& inputs) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes,
                           RuntimeContext& runtime_ctx) const override;  

  NDArrayList DoCompute(Operator& op,
                        const NDArrayList& inputs,
                        RuntimeContext& ctx) const override;

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) const {};

  bool _inplace;

 protected:
  DeviceGroup _comm_group;
  ReductionType _red_type{kNONE};
};

Tensor MakeReduceScatterOp(Tensor input, DeviceGroup comm_group,  
                           bool inplace = false, OpMeta op_meta = OpMeta());

Tensor MakeReduceScatterOp(Tensor input, DeviceGroup comm_group, ReductionType red_type, 
                           bool inplace = false, OpMeta op_meta = OpMeta());
}
}