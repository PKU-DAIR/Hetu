#include "hetu/graph/ops/Communication.h"
#include "hetu/graph/headers.h"
#include "hetu/graph/ops/kernel_links.h"

namespace hetu {
namespace graph {

// 注: 在OpImpl/OpInterface只提供函数接口, 状态都存在Operator->OpDef里, 因此这里的函数
// 都要接受op来获取状态并做推导; 值得注意的是由于部分函数接口只有当前CommOpImpl有, 因此外界
// 在调用类似get_comm_type(op)的时候, 必须要检查op->OpImpl是comm_op类型才行
uint64_t CommOpImpl::get_comm_type(Operator& op) {
  // input may be inplaced, so comm_type should be updated for each call
  // if (_comm_type != -1) {
  //   return _comm_type;
  // }
  Tensor& input = op->input(0); 
  const auto& src_ds = input->get_distributed_states();
  const auto& dst_ds = _dst_ds;
  if (src_ds.check_pure_duplicate()) {
    // TODO: check data among comm_devices be duplicate
    _comm_type = COMM_SPLIT_OP;
    HT_LOG_DEBUG << "COMM_SPLIT_OP";
  } else if (src_ds.check_allreduce(dst_ds)) {
    _comm_type = ALL_REDUCE_OP;
    HT_LOG_DEBUG << "ALL_REDUCE_OP";
  } else if (src_ds.check_allgather(dst_ds)) {
    _comm_type = ALL_GATHER_OP;
    HT_LOG_DEBUG << "ALL_GATHER_OP";
  } else if (src_ds.check_reducescatter(dst_ds)) {
    _comm_type = REDUCE_SCATTER_OP;
    HT_LOG_DEBUG << "REDUCE_SCATTER_OP";
  } else {
    _comm_type = P2P_OP; // other case: 非集合通信, 部分device之间p2p通信
    HT_LOG_DEBUG << "P2P_OP";
  }
  return _comm_type;
}

// devices by dim for collective communication
DeviceGroup CommOpImpl::get_devices_by_dim(Operator& op, int32_t dim) const {
  const auto& placement_group = op->placement_group();
  const auto& placement = op->placement();
  Tensor& input = op->input(0);
  HT_ASSERT(!placement_group.empty()) 
    << "Placement Group should be assigned before get devices by dim " << dim;
  int32_t local_device_idx = placement_group.get_index(placement);
  const auto& src_ds = input->get_distributed_states();
  const auto& order = src_ds.get_order();
  const auto& states = src_ds.get_states();

  auto idx = std::find(order.begin(), order.end(), dim);
  int32_t interval = 1;
  for (auto cur_order = idx + 1; cur_order != order.end(); cur_order++) {
    interval *= states.at(*cur_order);
  }
  int32_t macro_interval = interval * src_ds.get_dim(dim);
  int32_t start = local_device_idx - local_device_idx % macro_interval + local_device_idx % interval;
  std::vector<Device> comm_group;
  for (auto i = start; i < start + macro_interval; i += interval) {
    comm_group.push_back(placement_group.get(i));
  }
  return std::move(DeviceGroup(comm_group));
}

void CommOpImpl::DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                                const OpMeta& op_meta) const {
  const Tensor& input = inputs.at(0);
  const auto& ds_input = input->get_distributed_states();
  const auto& ds_dst = get_dst_distributed_states();
  // TODO: check states/order between src and dst
  HT_ASSERT(ds_input.is_valid() && ds_dst.is_valid())
           << "distributed states for input and dst tensor must be valid!";
  HT_ASSERT(ds_input.get_device_num() == ds_dst.get_device_num())
           << "cannot convert src distributed states to unpaired dst distributed states!";
  Tensor& output = outputs.at(0);
  output->set_distributed_states(ds_dst);  
}

std::vector<NDArrayMeta> 
CommOpImpl::DoInferMeta(const TensorList& inputs) const {
  const Tensor& input = inputs.at(0);
  const HTShape& input_shape = input->shape();
  const DistributedStates& src_ds = input->get_distributed_states();
  const DistributedStates& dst_ds = get_dst_distributed_states();
  HTShape shape(input_shape.size());
  for (size_t d = 0; d < input_shape.size(); d++) {
    shape[d] = input_shape[d] * src_ds.get_dim(d) / dst_ds.get_dim(d);
  }
  return {NDArrayMeta().set_dtype(input->dtype()).set_device(input->device()).set_shape(shape)};
}


HTShapeList CommOpImpl::DoInferShape(Operator& op, 
                                     const HTShapeList& input_shapes,
                                     RuntimeContext& runtime_ctx) const {
  const HTShape& input_shape = input_shapes.at(0);
  Tensor& input = op->input(0);
  const auto& src_ds = input->get_distributed_states();
  const auto& dst_ds = get_dst_distributed_states();
  HTShape shape; shape.reserve(input_shape.size());
  for (size_t d = 0; d < input_shape.size(); d++) {
    shape[d] = input_shape[d] * src_ds.get_dim(d) / dst_ds.get_dim(d);
  }
  return {shape};
}

TensorList CommOpImpl::DoGradient(Operator& op,
                                  const TensorList& grad_outputs) const {
  Tensor& input = op->input(0);
  const auto& ds_input = input->get_distributed_states();
  Tensor& output = op->output(0);
  const auto& ds_output = output->get_distributed_states();
  const Tensor& grad_output = grad_outputs.at(0);
  const auto& ds_grad_output = grad_output->get_distributed_states();
  HT_ASSERT(ds_input.is_valid() && ds_output.is_valid())
           << "distributed states for input and output tensor must be valid!";
  HT_ASSERT(ds_input.get_device_num() == ds_output.get_device_num())
           << "distributed states for input and output tensor must be matched!";
  DistributedStates ds_grad_input(ds_input);
  if (ds_grad_input.get_dim(-2) > 1) { // partial->duplicate
    std::pair<std::vector<int32_t>, int32_t> src2dst({{-2}, -1});
    auto res_states = ds_grad_input.combine_states(src2dst);
    auto res_order = ds_grad_input.combine_order(src2dst);
    auto device_num = ds_grad_input.get_device_num();
    ds_grad_input.set_distributed_states({device_num, res_states, res_order});
  }
  Tensor grad_input = MakeCommOp(grad_output, ds_grad_input, OpMeta().set_name("grad_" + op->name()));
  return {grad_input};  
}

bool AllReduceOpImpl::DoMapToParallelDevices(Operator& op, 
                                             const DeviceGroup& pg) const {
  for (int i = 0; i < _comm_group.num_devices(); i++) {
    HT_ASSERT(pg.contains(_comm_group.get(i))) 
      << "AllReduceOp: device in comm_group: " << _comm_group.get(i) 
      << " must in palcement_group: " << pg;
  }
  return OpInterface::DoMapToParallelDevices(op, pg);
}

std::vector<NDArrayMeta> 
AllReduceOpImpl::DoInferMeta(const TensorList& inputs) const {
  return {inputs[0]->meta()};
}

HTShapeList AllReduceOpImpl::DoInferShape(Operator& op, 
                                          const HTShapeList& input_shapes,
                                          RuntimeContext& runtime_ctx) const {
  return {input_shapes.at(0)};
}

void AllReduceOpImpl::DoCompute(Operator& op, const NDArrayList& inputs,
                                NDArrayList& outputs, RuntimeContext& runtime_ctx) const {
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(op->instantiation_ctx().placement.type(), type(),
                                  hetu::impl::AllReduce, inputs.at(0),
                                  outputs.at(0), _comm_group, // _comm_group is a subset of placement_group
                                  op->instantiation_ctx().stream());                              
}

bool P2PSendOpImpl::DoMapToParallelDevices(Operator& op,
                                           const DeviceGroup& pg) const {
  HT_ASSERT(pg.num_devices() == _dst_group.num_devices())
    << "Currently we require equal tensor parallelism degree across "
    << "P2P communication. Got " << pg << " vs. " << _dst_group;
  return OpInterface::DoMapToParallelDevices(op, pg);                                          
}

std::vector<NDArrayMeta> 
P2PSendOpImpl::DoInferMeta(const TensorList& inputs) const {
  return {};
}

HTShapeList P2PSendOpImpl::DoInferShape(Operator& op, 
                                        const HTShapeList& input_shapes,
                                        RuntimeContext& runtime_ctx) const {
  return {};
}

NDArrayList P2PSendOpImpl::DoCompute(Operator& op, const NDArrayList& inputs,
                                     RuntimeContext& runtime_ctx) const {
  NDArray input = inputs.at(0);
  HT_ASSERT(input->dtype() == op->input(0)->dtype())
    << "Data type mismatched for P2P communication: " << input->dtype()
    << " vs. " << op->input(0)->dtype();
  size_t dst_device_index = _dst_device_index == -1 ? 
         op->placement_group().get_index(op->placement()) : _dst_device_index;
  // TODO: sending the shape in compute fn is just a walkaround,
  // we shall determine the shape for recv op in executor
  NDArray send_shape = NDArray::empty({HT_MAX_NDIM + 1}, Device(kCPU), kInt64);
  auto* ptr = send_shape->data_ptr<int64_t>();
  ptr[0] = static_cast<int64_t>(input->ndim());
  std::copy(input->shape().begin(), input->shape().end(), ptr + 1);
  hetu::impl::P2PSendCpu(send_shape, _dst_group.get(dst_device_index),
                         Stream(Device(kCPU), kBlockingStream));

  HT_DISPATCH_KERNEL_CPU_AND_CUDA(op->instantiation_ctx().placement.type(), 
                                  type(), hetu::impl::P2PSend, input,
                                  _dst_group.get(dst_device_index), 
                                  op->instantiation_ctx().stream());
  return NDArrayList();                                    
}

bool P2PRecvOpImpl::DoMapToParallelDevices(Operator& op,
                                           const DeviceGroup& pg) const {
  HT_ASSERT(pg.num_devices() == _src_group.num_devices())
    << "Currently we require equal tensor parallelism degree across "
    << "P2P communication. Got " << _src_group << " vs. " << pg;
  return OpInterface::DoMapToParallelDevices(op, pg);                                          
}

std::vector<NDArrayMeta> 
P2PRecvOpImpl::DoInferMeta(const TensorList& inputs) const {
  return {NDArrayMeta().set_dtype(_dtype).set_shape(_shape)};
}

HTShapeList P2PRecvOpImpl::DoInferShape(Operator& op, 
                                        const HTShapeList& input_shapes,
                                        RuntimeContext& runtime_ctx) const {
  return {_shape};
}

NDArrayList P2PRecvOpImpl::DoCompute(Operator& op, const NDArrayList& inputs,
                                     RuntimeContext& runtime_ctx) const {
  size_t src_device_index = _src_device_index == -1 ?
         op->placement_group().get_index(op->placement()) : _src_device_index;
  // TODO: receiving the shape in compute fn is just a walkaround,
  // we shall determine the shape for recv op in executor
  NDArray recv_shape = NDArray::empty({HT_MAX_NDIM + 1}, Device(kCPU), kInt64);
  hetu::impl::P2PRecvCpu(recv_shape, _src_group.get(src_device_index),
                         Stream(Device(kCPU), kBlockingStream));
  auto* ptr = recv_shape->data_ptr<int64_t>();
  HTShape shape(ptr + 1, ptr + 1 + ptr[0]);
  NDArray output = NDArray::empty(shape, op->instantiation_ctx().placement, op->output(0)->dtype());

  HT_DISPATCH_KERNEL_CPU_AND_CUDA(op->instantiation_ctx().placement.type(), 
                                  type(), hetu::impl::P2PRecv, output,
                                  _src_group.get(src_device_index), 
                                  op->instantiation_ctx().stream());
  return {output};  
}

std::vector<NDArrayMeta> 
BatchedISendIRecvOpImpl::DoInferMeta(const TensorList& inputs) const {
  if (_outputs_shape.size() == 0)
    return {};
  std::vector<NDArrayMeta> output_meta_lsit;
  for (auto& output_shape: _outputs_shape) {
    output_meta_lsit.push_back(NDArrayMeta().set_dtype(_dtype).set_shape(output_shape));
  }
  return std::move(output_meta_lsit);
}

HTShapeList BatchedISendIRecvOpImpl::DoInferShape(Operator& op, 
                                                  const HTShapeList& input_shapes,
                                                  RuntimeContext& runtime_ctx) const {
  if (_outputs_shape.size() == 0)
    return {};                                                    
  HTShapeList outputs_shape(_outputs_shape);                                                    
  return std::move(outputs_shape);
}  

void BatchedISendIRecvOpImpl::DoCompute(Operator& op, 
                                        const NDArrayList& inputs,
                                        NDArrayList& outputs, 
                                        RuntimeContext& runtime_ctx) const {
  for (int i = 0; i < op->num_inputs(); i++) {
    const NDArray& input = inputs.at(i);
    HT_ASSERT(input->dtype() == op->input(i)->dtype())
      << "Data type mismatched for ISend communication: " << input->dtype()
      << " vs. " << op->input(i)->dtype();
  }

  HT_DISPATCH_KERNEL_CPU_AND_CUDA(op->instantiation_ctx().placement.type(), type(), 
                                  hetu::impl::BatchedISendIRecv, inputs, _dst_devices, outputs, 
                                  _src_devices, _comm_devices, op->instantiation_ctx().stream());
}

bool AllGatherOpImpl::DoMapToParallelDevices(Operator& op,
                                             const DeviceGroup& pg) const {
  for (int i = 0; i < _comm_group.num_devices(); i++) {
    HT_ASSERT(pg.contains(_comm_group.get(i))) 
      << "Allgather: device in comm_group: " << _comm_group.get(i) 
      << " must in device group: " << pg;
  }
  return OpInterface::DoMapToParallelDevices(op, pg);  
}

std::vector<NDArrayMeta> 
AllGatherOpImpl::DoInferMeta(const TensorList& inputs) const {
  const Tensor& input = inputs.at(0);
  DataType dtype = input->dtype();
  HTShape gather_shape = input->shape();
  gather_shape[0] *= _comm_group.num_devices();
  return {NDArrayMeta().set_dtype(dtype).set_shape(gather_shape)};
}

HTShapeList AllGatherOpImpl::DoInferShape(Operator& op, 
                                          const HTShapeList& input_shapes,
                                          RuntimeContext& runtime_ctx) const {
  HTShape gather_shape = input_shapes.at(0);
  gather_shape[0] *= _comm_group.num_devices();
  return {gather_shape};  
}

void AllGatherOpImpl::DoCompute(Operator& op, 
                                const NDArrayList& inputs,
                                NDArrayList& outputs,
                                RuntimeContext& runtime_ctx) const {
  HT_ASSERT(inputs.at(0)->dtype() == op->input(0)->dtype())
    << "Data type mismatched for AllGather communication: " << inputs.at(0)->dtype()
    << " vs. " << op->input(0)->dtype();

  HT_DISPATCH_KERNEL_CPU_AND_CUDA(op->instantiation_ctx().placement.type(), type(),
                                  hetu::impl::AllGather, inputs.at(0), outputs.at(0), 
                                  _comm_group, op->instantiation_ctx().stream());
}

bool ReduceScatterOpImpl::DoMapToParallelDevices(Operator& op,
                                                 const DeviceGroup& pg) const {
  for (int i = 0; i < _comm_group.num_devices(); i++) {
    HT_ASSERT(pg.contains(_comm_group.get(i))) 
      << "ReduceScatter: device in comm_group: " << _comm_group.get(i) 
      << " must in device group: " << pg;
  }
  return OpInterface::DoMapToParallelDevices(op, pg);  
}

std::vector<NDArrayMeta> 
ReduceScatterOpImpl::DoInferMeta(const TensorList& inputs) const {
  const Tensor& input = inputs.at(0);
  DataType dtype = input->dtype();
  HTShape scatter_shape = input->shape();
  scatter_shape[0] /= _comm_group.num_devices();
  HT_ASSERT(scatter_shape[0] >= 1) << "ReduceScatter: input shape[0]: " 
    << input->shape()[0] << " must >= comm devices num: " << _comm_group.num_devices();  
  return {NDArrayMeta().set_dtype(dtype).set_shape(scatter_shape)};
}

HTShapeList ReduceScatterOpImpl::DoInferShape(Operator& op, 
                                              const HTShapeList& input_shapes,
                                              RuntimeContext& runtime_ctx) const {
  HTShape scatter_shape = input_shapes.at(0);
  scatter_shape[0] /= _comm_group.num_devices();
  HT_ASSERT(scatter_shape[0] >= 1) << "ReduceScatter: input shape[0]: " 
    << input_shapes.at(0)[0] << " must >= comm devices num: " << _comm_group.num_devices();  
  return {scatter_shape};
}

void ReduceScatterOpImpl::DoCompute(Operator& op, 
                                    const NDArrayList& inputs,
                                    NDArrayList& outputs,
                                    RuntimeContext& runtime_ctx) const {
  HT_ASSERT(inputs.at(0)->dtype() == op->input(0)->dtype())
    << "Data type mismatched for ReduceScatter communication: " << inputs.at(0)->dtype()
    << " vs. " << op->input(0)->dtype();

  HT_DISPATCH_KERNEL_CPU_AND_CUDA(op->instantiation_ctx().placement.type(), type(),
                                  hetu::impl::ReduceScatter, inputs.at(0), outputs.at(0), 
                                  _comm_group, op->instantiation_ctx().stream());
}

Tensor MakeCommOp(Tensor input, DistributedStates dst_ds, OpMeta op_meta) {
  return Graph::MakeOp(std::make_shared<CommOpImpl>(dst_ds), 
                      {std::move(input)}, std::move(op_meta))->output(0);
}

Tensor MakeAllReduceOp(Tensor input, const DeviceGroup& comm_group, 
                       OpMeta op_meta) {
  return Graph::MakeOp(std::make_shared<AllReduceOpImpl>(comm_group, op_meta.device_group), 
                      {std::move(input)}, std::move(op_meta))->output(0);
}

// p2p send no output
Tensor MakeP2PSendOp(Tensor input, const DeviceGroup& dst_group, 
                     int dst_device_index, OpMeta op_meta) {
  return Graph::MakeOp(std::make_shared<P2PSendOpImpl>(dst_group, dst_device_index, op_meta.device_group),
                      {std::move(input)}, std::move(op_meta))->output(0);
}

Tensor MakeP2PRecvOp(const DeviceGroup& src_group, DataType dtype,
                     const HTShape& shape, int src_device_index, OpMeta op_meta) {
  return Graph::MakeOp(std::make_shared<P2PRecvOpImpl>(src_group, dtype, shape, 
                       src_device_index, op_meta.device_group), {}, std::move(op_meta))->output(0);
}

Tensor MakeBatchedISendIRecvOp(TensorList inputs, 
                               const std::vector<Device>& dst_devices, 
                               const HTShapeList& outputs_shape, 
                               const std::vector<Device>& src_devices, 
                               const std::vector<Device>& comm_devices, 
                               DataType dtype, OpMeta op_meta) {
  if (src_devices.size() == 0)
    return Graph::MakeOp(std::make_shared<BatchedISendIRecvOpImpl>(dst_devices, outputs_shape,
                        src_devices, comm_devices, dtype), std::move(inputs), std::move(op_meta))->out_dep_linker();
  else
    return Graph::MakeOp(std::make_shared<BatchedISendIRecvOpImpl>(dst_devices, outputs_shape,
                        src_devices, comm_devices, dtype), inputs, std::move(op_meta))->output(0);  
}

Tensor MakeAllGatherOp(Tensor input, const DeviceGroup& comm_group, 
                       const DeviceGroup& device_group, OpMeta op_meta) {
  return Graph::MakeOp(std::make_shared<AllGatherOpImpl>(comm_group, op_meta.device_group), 
                      {std::move(input)}, std::move(op_meta))->output(0);
}

Tensor MakeReduceScatterOp(Tensor input, const DeviceGroup& comm_group, 
                           const DeviceGroup& device_group, OpMeta op_meta) {
  return Graph::MakeOp(std::make_shared<ReduceScatterOpImpl>(comm_group, op_meta.device_group), 
                      {std::move(input)}, std::move(op_meta))->output(0);
}

}
}