#include "hetu/graph/executable_graph.h"
#include "hetu/graph/ops/data_transfer.h"
#include "hetu/graph/ops/group.h"
#include "hetu/graph/ops/Split.h"
#include "hetu/graph/ops/sum.h"
#include "hetu/graph/ops/Concatenate.h"
#include "hetu/graph/ops/Communication.h"
#include "hetu/graph/ops/Arithmetics.h"
#include "hetu/graph/ops/Loss.h"
#include "hetu/impl/communication/comm_group.h"

namespace hetu {
namespace graph {

Operator& ExecutableGraph::MakeOpInner(std::shared_ptr<OpInterface> body,
                                       TensorList inputs, OpMeta op_meta) {
  _check_all_inputs_in_graph(inputs, op_meta.extra_deps);
  return MakeAndAddOp(std::move(body), std::move(inputs), std::move(op_meta));
}

void ExecutableGraph::ResetVariableDataInner(const Tensor& tensor,
                                             const Initializer& init) {
  if (tensor->placement().is_undetermined()) {
    _add_on_inits[tensor->id()] = std::unique_ptr<Initializer>(init.copy());
  } else {
    init.Init(GetVariableDataInner(tensor));
  }
}

NDArray& ExecutableGraph::GetVariableDataInner(const Tensor& tensor) {
  auto it = _preserved_data.find(tensor->id());
  HT_RUNTIME_ERROR_IF(it == _preserved_data.end())
    << "Cannot find data for variable tensor " << tensor;
  return it->second;
}

NDArray& ExecutableGraph::AllocVariableDataInner(const Tensor& tensor,
                                                 const Initializer& init,
                                                 uint64_t seed,
                                                 const HTShape& global_shape) {
  // TODO: check meta is valid
  _preserved_data[tensor->id()] =
    NDArray::empty(tensor->shape(), tensor->placement(), tensor->dtype());
  auto it = _add_on_inits.find(tensor->id());
  if (it != _add_on_inits.end()) {
    it->second->Init(_preserved_data[tensor->id()], seed, global_shape);
  } else if (!init.vodify()) {
    init.Init(_preserved_data[tensor->id()], seed, global_shape);
  }
  return _preserved_data[tensor->id()];
}

void ExecutableGraph::RegisterVariableDataInner(const Tensor& tensor,
                                                NDArray data,
                                                const Initializer& init) {
  _preserved_data[tensor->id()] = std::move(data);
  auto it = _add_on_inits.find(tensor->id());
  if (it != _add_on_inits.end()) {
    it->second->Init(_preserved_data[tensor->id()]);
  } else if (!init.vodify()) {
    init.Init(_preserved_data[tensor->id()]);
  }
}

bool ExecutableGraph::Instantiate(const TensorList& fetches,
                                  const Device& preferred_device) {
  auto is_op_instantiated = [&](const Operator& op) -> bool {
    return !op->placement().is_undetermined();
  };
  OpRefList topo = Graph::TopoSort(fetches, num_ops(), is_op_instantiated);
  HT_LOG_TRACE << "Instantiating ops: " << topo;

  // global info for all devices
  for (auto& op_ref : topo) {
    auto& op = op_ref.get();
    if (!op->placement().is_undetermined())
      continue;

    // remove redundant comm ops
    if (is_comm_op(op)) {
      auto& input_op = op->input(0)->producer();
      // TODO: special case: input_op include pp but op don't 
      if (is_comm_op(input_op)) {
        ReplaceInput(op, 0, input_op->input(0));
        // input changes, update comm_op type
        reinterpret_cast<CommOpImpl&>(op->body()).get_comm_type(op);
      }
    }
    
    // op->placement_group + tensor->placement_group
    if (!op->device_group().empty()) {
      op->MapToParallelDevices(op->device_group());
      // HT_LOG_INFO << hetu::impl::comm::GetLocalDevice() << ": op " << op << " assigned placement group = " << op->placement_group();
    } else {
      DeviceGroup inferred;
      if (is_group_op(op)) {
        std::vector<Device> devices;
        for (auto& input : op->in_dep_linkers()) {
          for (auto& device : input->producer()->placement_group().devices()) {
            devices.push_back(device);
          }
        }
        inferred = DeviceGroup(devices);
      } else {
        HT_ASSERT(op->num_inputs() > 0)
          << "Currently we cannot infer the devices "
          << "for operators with zero in-degree. : " << op;
        // TODO: Tensor add is_grad attribute, and firstly infer non-grad input tensor's placement group
        // if all input tensor are grad tensor, then infer the first input's placement group
        // inferred = op->input(0)->placement_group();
        for (auto input : op->inputs()) {
          if (!input->is_grad()) {
            inferred = input->placement_group();
            break;
          }
        }
        if (inferred.empty())
          inferred = op->input(0)->placement_group();
      }
      op->MapToParallelDevices(inferred);
      // HT_LOG_INFO << hetu::impl::comm::GetLocalDevice() << ": op " << op << " inferred placement group = " << inferred;
    }
    // udpate stages
    DeviceGroup stage_group;
    if (is_comm_op(op)) {
      auto& op_impl = reinterpret_cast<CommOpImpl&>(op->body());
      stage_group = op_impl.src_group(op);
    } else if (!is_group_op(op)) {
      stage_group = op->placement_group();
    }
    if (!stage_group.empty() && std::find(_stages.begin(), _stages.end(), stage_group) == _stages.end()) {
      _stages.push_back(stage_group);
    }

    // loss & grad should div by num_micro_batches when reduction type = MEAN!!! 
    if (is_loss_gradient_op(op) && op->input(0)->has_distributed_states()) {
      int dp = op->input(0)->get_distributed_states().get_dim(0);
      auto& loss_grad_op_impl = reinterpret_cast<LossGradientOpImpl&>(op->body());
      if ((_num_micro_batches > 1 || dp > 1) && loss_grad_op_impl.reduction() == kMEAN) {
        auto& grads = op->outputs();
        for (auto& grad : grads) {
          if (!grad.is_defined()) {
            continue;
          }
          Tensor grad_scale = MakeDivByConstOp(grad, _num_micro_batches * dp, OpMeta().set_name(grad->name() + "_scale"));
          auto& grad_scale_op = grad_scale->producer();
          grad_scale_op->MapToParallelDevices(op->placement_group());
          for (int i = grad->num_consumers() - 1; i >= 0; i--) {
            auto& consumer_i = grad->consumer(i);
            if (consumer_i->id() == grad_scale_op->id()) continue;
            for (int j = 0; j < consumer_i->num_inputs(); j++) {
              if (consumer_i->input(j)->id() == grad->id()) {
                ReplaceInput(consumer_i, j, grad_scale);
              }
            }
          }
        }
      }
    }

    // add p2p comm_op for pipeline parallel
    const auto& dst_group = op->placement_group();
    for (size_t i = 0; i < op->num_inputs(); i++) {
      auto& input = op->input(i);
      auto& input_op = input->producer();
      const auto& src_group = input_op->placement_group();
      if (src_group != dst_group) {
        // TODO: reuse p2p op & remove useless p2p op
        Tensor p2p_input;
        // there is no two linked comm_op, due to the former code to remove redundant comm_op
        // if comm_op need pp, its placement_group contains both src and dst group
        if (is_comm_op(input_op)) {
          auto& input_op_impl = reinterpret_cast<CommOpImpl&>(input_op->body());
          if (input_op_impl.dst_group(input_op) == dst_group) {
            continue;
          } else {
            const auto& src_group_comm = input_op_impl.src_group(input_op);
            HT_ASSERT(src_group_comm.num_devices() == dst_group.num_devices())
              << "DeviceGroup size in different pipeline stage must be same, "
              << "got " << src_group_comm.num_devices()
              << " vs. " << dst_group.num_devices();

            bool reused = false;
            for (auto& consumer_op : input_op->input(0)->consumers()) {
              if (consumer_op.get()->id() != input_op->id() && is_comm_op(consumer_op)) {
                auto& consumer_op_impl = reinterpret_cast<CommOpImpl&>(consumer_op.get()->body());
                const auto& dst_group_comm = consumer_op_impl.dst_group(consumer_op.get());
                if (consumer_op_impl.get_dst_distributed_states().check_equal(
                  input_op_impl.get_dst_distributed_states()) && dst_group_comm == dst_group) {
                  ReplaceInput(op, i, consumer_op.get()->output(0));
                  reused = true;
                  break;
                }
              }
            }
            if (reused)
              continue;

            p2p_input = MakeCommOp(input_op->input(0), input_op_impl.get_dst_distributed_states(), dst_group);
          }
        } else if (is_comm_op(op)) {
          auto& op_impl = reinterpret_cast<CommOpImpl&>(op->body());
          const auto& src_group_comm = op_impl.src_group(op);
          HT_ASSERT(src_group_comm == src_group)
            << "CommOp(with pp) " << op->name() << ": src group " << src_group_comm
            << " must equal to InputOp " << input_op->name() <<": group " << src_group;
          continue;
        } else {
          HT_ASSERT(src_group.num_devices() == dst_group.num_devices())
            << "DeviceGroup size in different pipeline stage must be same, "
            << "got " << src_group.num_devices() 
            << " vs. " << dst_group.num_devices();

          bool reused = false;
          for (auto& consumer_op : input->consumers()) {
            if (consumer_op.get()->id() != op->id() && is_comm_op(consumer_op)) {
              auto& consumer_op_impl = reinterpret_cast<CommOpImpl&>(consumer_op.get()->body());
              const auto& dst_group_comm = consumer_op_impl.dst_group(consumer_op.get());
              if (consumer_op_impl.get_dst_distributed_states().check_equal(
                  input->get_distributed_states()) && dst_group_comm == dst_group) {
                ReplaceInput(op, i, consumer_op.get()->output(0));
                reused = true;
                break;
              }
            }
          }
          if (reused)
            continue;

          p2p_input = MakeCommOp(input, input->get_distributed_states(), dst_group);
        }
        auto& p2p_op = p2p_input->producer();
        // will be splited into intra_comm + p2p_send(src_group) and p2p_recv(dst_group)
        p2p_op->MapToParallelDevices(input_op->placement_group());
        ReplaceInput(op, i, p2p_input);
      }
    }
  }

  // get updated topo
  OpRefList updated_topo = Graph::TopoSort(fetches, num_ops(), is_op_instantiated);

  // HT_LOG_DEBUG << preferred_device << ": updated topo after map placement_group: " << updated_topo; 

  // local info for local_device
  for (auto& op_ref : updated_topo) {
    auto& op = op_ref.get();
    if (!op->placement().is_undetermined())
      continue;  

    // for local compute: op->placement + tensor->placement
    if (!op->placement_group().contains(preferred_device))
      continue;
    Device placement =
      is_device_to_host_op(op) ? Device(kCPU) : preferred_device;
    StreamIndex stream_id = get_suggested_stream_index(op);
    HT_LOG_TRACE << "Instantiating op " << op << " (placement=" << placement
                 << ", stream_index=" << stream_id << ")";
    bool ok = op->Instantiate(placement, stream_id);
    if (!ok && !placement.is_cpu()) {
      HT_LOG_WARN << "Failed to instantiate op " << op << " on " << placement
                  << ". Will try to instantiate it on the host device.";
      placement = Device(kCPU);
      ok = op->Instantiate(placement, stream_id);
    }
    HT_VALUE_ERROR_IF(!ok) << "Failed to instantiate op " << op << " on "
                           << placement;

    // add transfer ops
    for (size_t i = 0; i < op->num_inputs(); i++) {
      auto& input = op->input(i);
      if (input->placement() != placement && !is_comm_op(op)) {
        HT_RUNTIME_ERROR_IF(!input->placement().local())
          << "Please use P2P communication to fetch remote input";

        auto& input_op = input->producer();

        Tensor transferred_input;
        StreamIndex transfer_stream_id;
        if (input->placement().is_cpu()) {
          transferred_input = MakeDataH2DOp(placement, input);
          transfer_stream_id = kH2DStream;
        } else if (placement.is_cpu()) {
          transferred_input = MakeDataD2HOp(placement, input);
          transfer_stream_id = kD2HStream;
        } else {
          // TODO: support cuda memcpy across processes
          HT_NOT_IMPLEMENTED << "We should use NCCL for P2P communication.";
          __builtin_unreachable();
        }
        auto& transfer_op = transferred_input->producer();
        if (!input_op->placement_group().empty())
          transfer_op->MapToParallelDevices(input_op->placement_group());
        transfer_op->Instantiate(placement, transfer_stream_id);
        ReplaceInput(op, i, transferred_input);
      }
    }
  }

  return true;
}

void ExecutableGraph::SubstituteCommOp(const OpRefList& topo_order) {
  auto& local_device = hetu::impl::comm::GetLocalDevice();
  for (auto& op_ref : topo_order) {
    auto& op = op_ref.get();
    // each device only need to substitute local comm_ops
    if (is_comm_op(op) && op->placement_group().contains(local_device)) {
      HT_LOG_DEBUG << local_device << "==============> substitute comm_op begin: " << op << "...";
      auto& comm_op = op;
      auto& comm_op_impl = reinterpret_cast<CommOpImpl&>(comm_op->body());
      uint64_t comm_type = comm_op_impl.get_comm_type(comm_op);
      const auto& src_group = comm_op_impl.src_group(comm_op);
      const auto& dst_group = comm_op_impl.dst_group(comm_op);
      Tensor& input = comm_op->input(0);
      Tensor result;

      if (comm_op_impl.is_intra_group(comm_op) || comm_op_impl.is_inter_group(comm_op) && 
          src_group.contains(local_device)) {
        // tp
        if (comm_type == P2P_OP) {
          result = comm_op->input(0);
        } else if (comm_type == COMM_SPLIT_OP) {
          auto local_device_index = src_group.get_index(local_device);
          const auto& dst_ds = comm_op_impl.get_dst_distributed_states();
          auto cur_state_index = dst_ds.map_device_to_state_index(local_device_index);
          const auto& order = dst_ds.get_order();
          const auto& states = dst_ds.get_states();
          HTAxes keys; 
          HTShape indices, splits;
          for (auto o : order) {
            if (o >= 0) { 
              keys.push_back(o);
              splits.push_back(states.at(o));
              indices.push_back(cur_state_index[o]);
            }
          }
          HT_LOG_DEBUG << local_device << ": keys = " << keys << "; indices = " << indices << "; splits = " << splits;
          Tensor split_output = MakeSplitOp(input, keys, indices, splits, OpMeta().set_is_deduce_states(false));
          auto& split_op = split_output->producer();
          split_op->MapToParallelDevices(src_group);
          split_op->Instantiate(local_device, kComputingStream);
          result = split_output;
        } else if (comm_type == ALL_REDUCE_OP) {
          DeviceGroup comm_group = comm_op_impl.get_devices_by_dim(comm_op, -2); // do allreduce among comm_group
          Tensor all_reduce_output = MakeAllReduceOp(
            input, comm_group, // comm_group is a subset of placement_group
            OpMeta().set_device_group(src_group)
                    .set_is_deduce_states(false)
                    .set_name(input->name() + "_AllReduce"));
          auto& all_reduce_op = all_reduce_output->producer();
          all_reduce_op->MapToParallelDevices(src_group);
          all_reduce_op->Instantiate(local_device, kCollectiveStream);
          result = all_reduce_output;
          HT_LOG_DEBUG << local_device << ": substitute comm_op to all_reduce_op: " << comm_group;        
        } else if (comm_type == ALL_GATHER_OP) {
          DeviceGroup comm_group = comm_op_impl.get_devices_by_dim(comm_op, 0);
          Tensor all_gather_output = MakeAllGatherOp(
            input, comm_group,
            OpMeta().set_device_group(src_group)
                    .set_is_deduce_states(false)
                    .set_name(input->name() + "_AllGather"));
          auto& all_gather_op = all_gather_output->producer();
          all_gather_op->MapToParallelDevices(src_group);
          all_gather_op->Instantiate(local_device, kCollectiveStream);
          result = all_gather_output;
          HT_LOG_DEBUG << local_device << ": substitute comm_op to all_gather_op: " << comm_group;
        } else if (comm_type == REDUCE_SCATTER_OP) {
          DeviceGroup comm_group = comm_op_impl.get_devices_by_dim(comm_op, -2);
          Tensor reduce_scatter_output =  MakeReduceScatterOp(
            input, comm_group,
            OpMeta().set_device_group(src_group)
                    .set_is_deduce_states(false)
                    .set_name(input->name() + "_ReduceScatter"));
          auto& reduce_scatter_op = reduce_scatter_output->producer();
          reduce_scatter_op->MapToParallelDevices(src_group);
          reduce_scatter_op->Instantiate(local_device, kCollectiveStream);
          result = reduce_scatter_output;
          HT_LOG_DEBUG << local_device << ": substitute comm_op to reduce_scatter_op: " << comm_group;
        } else if (comm_type == BATCHED_ISEND_IRECV_OP) {
          // 1. local_device send data to other devices 2. local_device recv data from other devices
          DataType dtype = input->dtype();
          int32_t local_device_index = src_group.get_index(local_device);
          TensorList send_datas_local;
          std::vector<int32_t> dsts_local;
          HTShapeList recv_shapes_local;
          std::vector<int32_t> srcs_local;
          Tensor self_send_data;
          std::vector<std::pair<int32_t, int32_t>> send_pairs;
          for (int32_t used_device_index = 0; used_device_index < src_group.num_devices(); used_device_index++) {     
            HT_LOG_DEBUG << local_device << ": cross send begin!";
            int32_t device_index = 0;
            TensorList send_datas;
            std::vector<int32_t> dsts;
            // execute cross_send for all devices to get the complete recv_shapes
            CrossSend({}, {}, 0, false, device_index, comm_op, send_datas, dsts, used_device_index);
            HT_ASSERT(device_index == src_group.num_devices()) << "cross send error!";
            HT_LOG_DEBUG << local_device << ": cross send end!";
            // for batch send/recv
            for (int i = 0; i < dsts.size(); i++) {
              send_pairs.push_back({used_device_index, dsts[i]}); // for comm_set
              // local_device send to other devices
              if (used_device_index == local_device_index && dsts[i] != local_device_index) {
                send_datas_local.push_back(send_datas[i]);
                dsts_local.push_back(dsts[i]);
              } 
              // local device recv from other devices
              if (used_device_index != local_device_index && dsts[i] == local_device_index) {
                recv_shapes_local.push_back(send_datas[i]->shape());
                srcs_local.push_back(used_device_index);              
              }
              // special case: local device send to self
              if (used_device_index == local_device_index && dsts[i] == local_device_index) {
                self_send_data = send_datas[i];
              }
            }
          }

          // get comm_devices for batch isend/irecv, union set
          std::set<int32_t> comm_set;
          comm_set.insert(local_device_index);
          comm_set.insert(dsts_local.begin(), dsts_local.end());
          comm_set.insert(srcs_local.begin(), srcs_local.end());
          bool keep_search = true;
          while (keep_search) {
            keep_search = false;
            for (auto& pair : send_pairs) {
              bool find_first = (comm_set.find(pair.first) != comm_set.end());
              bool find_second = (comm_set.find(pair.second) != comm_set.end());
              if (find_first && !find_second) {
                comm_set.insert(pair.second);
                keep_search = true;
              } else if (!find_first && find_second) {
                comm_set.insert(pair.first);
                keep_search = true;
              }
            }
          }
          std::vector<Device> comm_devices(comm_set.size());
          std::vector<Device> dst_devices(dsts_local.size());
          std::vector<Device> src_devices(srcs_local.size());
          std::transform(dsts_local.begin(), dsts_local.end(), dst_devices.begin(), [&](int32_t index) { return src_group.get(index); });
          std::transform(srcs_local.begin(), srcs_local.end(), src_devices.begin(), [&](int32_t index) { return src_group.get(index); });        
          std::transform(comm_set.begin(), comm_set.end(), comm_devices.begin(), [&](int32_t index) { return src_group.get(index); });
          // when needn't recv, MakeBatchedISendIRecvOp return out_dep_linker
          Tensor batched_isend_irecv_output = MakeBatchedISendIRecvOp(send_datas_local, dst_devices, recv_shapes_local, src_devices, comm_devices, dtype, 
            OpMeta().set_is_deduce_states(false).set_name("BatchedISendIRecvOp_for_" + comm_op->name()));
          auto& batched_isend_irecv_op = batched_isend_irecv_output->producer();
          batched_isend_irecv_op->MapToParallelDevices(src_group);
          batched_isend_irecv_op->Instantiate(local_device, kP2PStream);
          TensorList recv_datas_local = batched_isend_irecv_op->outputs();

          HT_LOG_DEBUG << local_device << ": cross receive begin!";
          int32_t device_index = 0;
          // already get the recv_datas by batch_send_recv, so just need local device to execute cross_receive
          result = CrossReceive(0, device_index, comm_op, recv_datas_local, srcs_local, self_send_data, local_device_index);
          HT_ASSERT(device_index == src_group.num_devices()) << "cross receive error!";
          HT_LOG_DEBUG << local_device << ": cross receive end!";

          // add dummy link for topo sort
          if (dst_devices.size() == 0) { // connect comm_op->input producer with batchISendIRecvOp when needn't send
            AddInDeps(batched_isend_irecv_op, {input});
          }
          if (src_devices.size() == 0) { // connect batchISendIRecvOp with comm_op->ouput consumers when needn't recv
            AddInDeps(result->producer(), {batched_isend_irecv_op->out_dep_linker()});
          }          
        }
        // add p2p send after tp
        if (comm_op_impl.is_inter_group(comm_op)) {
          HT_LOG_DEBUG << local_device << ": send to stage " << dst_group;
          Tensor send_out_dep_linker = MakeP2PSendOp(result, dst_group, 
            src_group.get_index(local_device), OpMeta().set_is_deduce_states(false));
          auto& send_op = send_out_dep_linker->producer();
          send_op->MapToParallelDevices(src_group);
          send_op->Instantiate(local_device, kP2PStream);
          // add dummy link for topo sort
          for (int i = 0; i < comm_op->output(0)->num_consumers(); i++) {
            AddInDeps(comm_op->output(0)->consumer(i), {send_out_dep_linker});
          }
        }
      } else {
        // p2p recv
        HT_LOG_DEBUG << local_device << ": just recv from stage " << src_group;
        Tensor& output = comm_op->output(0); // output meta was already deduced in DoInferMeta
        Tensor recv_output = MakeP2PRecvOp(src_group, output->dtype(), output->shape(),
          dst_group.get_index(local_device), OpMeta().set_is_deduce_states(false));
        auto& recv_op = recv_output->producer();
        recv_op->MapToParallelDevices(dst_group);
        recv_op->Instantiate(local_device, kP2PStream);
        // add dummy link for topo sort
        AddInDeps(recv_op, {input});
        result = recv_output;
      }
      result->set_distributed_states(comm_op_impl.get_dst_distributed_states()); // assign distributed states for result tensor

      // find all comm_op->output consumers, and replace the correspond input tensor with result tensor
      for (int i = comm_op->output(0)->num_consumers() - 1; i >= 0; i--) {
        auto& consumer_i = comm_op->output(0)->consumer(i);
        for (int j = 0; j < consumer_i->num_inputs(); j++) {
          if (consumer_i->input(j)->id() == comm_op->output(0)->id()) {
            ReplaceInput(consumer_i, j, result);
          }
        }
      }
      HT_LOG_DEBUG << local_device << "==============> substitute comm_op end: " << op << "...";
    }
  }
}

Tensor ExecutableGraph::CrossReceive(int32_t depth, int32_t& device_index, Operator& comm_op, 
                                     TensorList& recv_datas, std::vector<int32_t>& srcs,
                                     Tensor& self_send_data, int32_t& used_device_index) {
  HT_ASSERT(is_comm_op(comm_op)) << comm_op->name() << " must be comm_op!";
  auto& local_device = hetu::impl::comm::GetLocalDevice();
  auto& comm_op_impl = reinterpret_cast<CommOpImpl&>(comm_op->body());
  const auto& src_group = comm_op_impl.src_group(comm_op);
  const auto& prev_distributed_states = comm_op->input(0)->get_distributed_states();
  auto prev_partial = prev_distributed_states.get_dim(-2);
  auto prev_duplicate = prev_distributed_states.get_dim(-1);
  const auto& prev_order = prev_distributed_states.get_order();
  auto loop_sizes = prev_distributed_states.get_loop_sizes();

  const auto& target_distributed_states = comm_op_impl.get_dst_distributed_states();
  auto target_duplicate = target_distributed_states.get_dim(-1);
  auto local_device_index = src_group.get_index(local_device);
  auto cur_state_index = target_distributed_states.map_device_to_state_index(used_device_index); // 指定的device需要的是tensor的哪一部分数据

  auto get_state_index = [&](int32_t dim) -> int32_t {
    if (cur_state_index.find(dim) != cur_state_index.end()) {
      return cur_state_index[dim];
    } else {
      return 0;
    }
  };
  
  Tensor result;
  // cur_state_index存的是used device需要的是哪些数据, 最终的result是从device_index对应的device中concatenate/sum获取而来的
  if (depth == prev_order.size()) {
    // 如果recv的对象就是自己, 则无需send/recv op
    if (device_index == used_device_index) {
      // 判断self_send_data是否已经赋值
      HT_ASSERT(self_send_data.is_defined()) << "Cross Receive: self_send_data must be a valid tensor!";
      result = self_send_data;
      HT_LOG_DEBUG << local_device << ": device " << used_device_index 
                   << ": recv from device " << device_index << " don't need irecv";
    } else {
      for (int i = 0; i < srcs.size(); i++) {
        if (srcs[i] == device_index) {
          result = recv_datas[i];
          break;
        }
      }
      HT_LOG_DEBUG << local_device << ": device " << used_device_index << ": recv from device " << device_index;      
    }
    device_index += 1;            
  } else {
    auto cur_dim = prev_order[depth];
    if (cur_dim == -2) { // partial
      TensorList part_result_list;
      for (size_t i = 0; i < prev_partial; i++) {
        auto part_result = CrossReceive(depth+1, device_index, comm_op, recv_datas, srcs, self_send_data, used_device_index);
        part_result_list.push_back(part_result);
      }
      auto sum_output = MakeSumOp(part_result_list, OpMeta().set_is_deduce_states(false));
      auto& sum_op = sum_output->producer();
      if (used_device_index == local_device_index) {
        sum_op->MapToParallelDevices(src_group);
        sum_op->Instantiate(local_device, kComputingStream);
      }
      result = sum_output;    
    } else if (cur_dim == -1) {
      auto cur_st = get_state_index(cur_dim);
      if (prev_duplicate % target_duplicate == 0) {
        auto multiple = prev_duplicate / target_duplicate;
        device_index += cur_st * multiple * loop_sizes[depth];
        result = CrossReceive(depth+1, device_index, comm_op, recv_datas, srcs, self_send_data, used_device_index);
        device_index += ((target_duplicate - cur_st) * multiple - 1) * loop_sizes[depth];
      } else if (target_duplicate % prev_duplicate == 0) {
        auto multiple = target_duplicate / prev_duplicate;
        device_index += cur_st / multiple * loop_sizes[depth];
        result = CrossReceive(depth+1, device_index, comm_op, recv_datas, srcs, self_send_data, used_device_index);
        device_index += (target_duplicate - 1 - cur_st) / multiple * loop_sizes[depth];
      } else {
        HT_ASSERT(false) << "cannot support!";
      }
    } else {
      auto pre_st = prev_distributed_states.get_states().at(cur_dim);
      auto tar_st = target_distributed_states.get_dim(cur_dim);
      auto cur_st = get_state_index(cur_dim);
      if (pre_st % tar_st == 0) {
        auto multiple = pre_st / tar_st;
        device_index += cur_st * multiple * loop_sizes[depth];
        if (multiple == 1) {
          result = CrossReceive(depth+1, device_index, comm_op, recv_datas, srcs, self_send_data, used_device_index);
        } else {
          TensorList part_result_list;
          for (size_t i = 0; i < multiple; i++) {
            auto part_result = CrossReceive(depth+1, device_index, comm_op, recv_datas, srcs, self_send_data, used_device_index);
            part_result_list.push_back(part_result);
          }
          auto concatenate_output = MakeConcatenateOp(part_result_list, cur_dim, OpMeta().set_is_deduce_states(false));
          auto& concatenate_op = concatenate_output->producer();
          if (used_device_index == local_device_index) {
            concatenate_op->MapToParallelDevices(src_group);
            concatenate_op->Instantiate(local_device, kComputingStream);
          }
          result = concatenate_output;
        }
        device_index += (tar_st - 1 - cur_st) * multiple * loop_sizes[depth];
      } else if (tar_st % pre_st == 0) {
        auto multiple = tar_st / pre_st;
        device_index += cur_st / multiple * loop_sizes[depth];
        result = CrossReceive(depth+1, device_index, comm_op, recv_datas, srcs, self_send_data, used_device_index);
        device_index += (tar_st - 1 - cur_st) / multiple * loop_sizes[depth];
      } else {
        HT_ASSERT(false) << "cannot support!";
      }
    }
  }
  
  return result;  
}

void ExecutableGraph::CrossSend(std::unordered_map<int32_t, int32_t> split_cur_state, 
                                std::unordered_map<int32_t, int32_t> split_target_state,
                                int32_t depth, bool need_split, int32_t& device_index, 
                                Operator& comm_op, TensorList& send_datas, 
                                std::vector<int32_t>& dsts, int32_t& used_device_index) {
  // basic info
  HT_ASSERT(is_comm_op(comm_op)) << comm_op->name() << " must be comm_op!";
  auto& local_device = hetu::impl::comm::GetLocalDevice();
  auto& comm_op_impl = reinterpret_cast<CommOpImpl&>(comm_op->body());
  const auto& src_group = comm_op_impl.src_group(comm_op);
  const auto& prev_distributed_states = comm_op->input(0)->get_distributed_states();
  auto prev_partial = prev_distributed_states.get_dim(-2);
  auto prev_duplicate = prev_distributed_states.get_dim(-1);
  auto local_device_index = src_group.get_index(local_device);  
  auto cur_state_index = prev_distributed_states.map_device_to_state_index(used_device_index);

  const auto& target_distributed_states = comm_op_impl.get_dst_distributed_states();
  auto target_duplicate = target_distributed_states.get_dim(-1);
  const auto& target_order = target_distributed_states.get_order();
  auto loop_sizes = target_distributed_states.get_loop_sizes();                  
  
  auto get_state_index = [&](int32_t dim) -> int32_t {
    if (cur_state_index.find(dim) != cur_state_index.end()) {
      return cur_state_index[dim];
    } else {
      return 0;
    }
  };

  auto get_keys = [](std::unordered_map<int32_t, int32_t> map) -> HTAxes {
    HTAxes keys; 
    keys.reserve(map.size());
    for (auto kv : map) {
      keys.push_back(kv.first);
    }
    return keys;
  };

  // cross send part
  if (prev_partial == 1 && prev_duplicate > target_duplicate && get_state_index(-1) % (prev_duplicate / target_duplicate) != 0) {    
    HT_LOG_DEBUG << local_device << ": device " << used_device_index << " don't need to send to other devices!";
    device_index += src_group.num_devices();
    return;
  }  
  if (depth == target_order.size()) {
    Tensor send_part;
    if (need_split) {
      HTAxes keys = get_keys(split_target_state);
      HTShape indices, splits;
      indices.reserve(keys.size()); splits.reserve(keys.size());
      for (auto key : keys) {
        indices.push_back(split_cur_state[key]);
        splits.push_back(split_target_state[key]);
      }
      // split_op: 把tensor在keys这些dimension上按照splits[key]份数切分, 并取出第indices[key]份, 作为要send的数据切片
      auto split_output = MakeSplitOp(comm_op->input(0), keys, indices, splits, OpMeta().set_is_deduce_states(false));
      auto& split_op = split_output->producer();
      if (used_device_index == local_device_index) { // 其他device上生成的用于替换comm_op不需要map placement_group和placement
        split_op->MapToParallelDevices(src_group);
        split_op->Instantiate(local_device, kComputingStream);
      }
      send_part = split_output;
    } else {
      // 如果不需要split, 则发送整个tensor
      send_part = comm_op->input(0);
      HT_LOG_DEBUG << local_device << ": device " << used_device_index << ": send to device " << device_index << " don't need split";      
    }
    if (device_index == used_device_index) {
      HT_LOG_DEBUG << local_device << ": device " << used_device_index << ": send to device " << device_index << " don't need isend";
    } else {
      HT_LOG_DEBUG << local_device << ": device " << used_device_index << ": send to device " << device_index;
    }    
    send_datas.push_back(send_part);
    dsts.push_back(device_index);

    device_index += 1;
  } else {
    auto cur_dim = target_order[depth];
    if (cur_dim < 0) {
      HT_ASSERT(cur_dim == -1) << "Target distributed states must not enable partial!";
      auto cur_st = get_state_index(cur_dim);
      if (prev_duplicate % target_duplicate == 0) {
        auto multiple = prev_duplicate / target_duplicate;
        if (cur_st % multiple != 0) {
          HT_LOG_DEBUG << local_device << ": device " << used_device_index << ": don't need to send to other devices!";
          return;
        }
        device_index += cur_st / multiple * loop_sizes[depth];
        CrossSend(split_cur_state, split_target_state, depth+1, need_split, device_index, comm_op, send_datas, dsts, used_device_index);
        device_index += (prev_duplicate - 1 - cur_st) / multiple * loop_sizes[depth];
      } else if (target_duplicate % prev_duplicate == 0) {
        auto multiple = target_duplicate / prev_duplicate;
        device_index += cur_st * multiple * loop_sizes[depth];
        for (size_t i = 0; i < multiple; i++) {
          CrossSend(split_cur_state, split_target_state, depth+1, true, device_index, comm_op, send_datas, dsts, used_device_index);
        }
        device_index += (prev_duplicate - 1 - cur_st) * multiple * loop_sizes[depth];
      } else {
        HT_ASSERT(false) << "cannot support!";
      }
    } else {
      auto pre_st = prev_distributed_states.get_dim(cur_dim);
      auto cur_st = get_state_index(cur_dim);
      auto tar_st = target_distributed_states.get_states().at(cur_dim);
      if (pre_st % tar_st == 0) {
        auto multiple = pre_st / tar_st;
        device_index += cur_st / multiple * loop_sizes[depth];
        split_cur_state[cur_dim] = 0;
        split_target_state[cur_dim] = 1;
        CrossSend(split_cur_state, split_target_state, depth+1, need_split, device_index, comm_op, send_datas, dsts, used_device_index);
        device_index += (pre_st - 1 - cur_st) / multiple * loop_sizes[depth];
      } else if (tar_st % pre_st == 0) {
        auto multiple = tar_st / pre_st;
        device_index += cur_st * multiple * loop_sizes[depth];
        for (size_t i = 0; i < multiple; i++) {
          split_cur_state[cur_dim] = i;
          split_target_state[cur_dim] = multiple; 
          CrossSend(split_cur_state, split_target_state, depth+1, true, device_index, comm_op, send_datas, dsts, used_device_index);
        }
        device_index += (pre_st - 1 - cur_st) * multiple * loop_sizes[depth];
      } else {
        HT_ASSERT(false) << "cannot support!";
      }
    }
  }
}

// schedule: {stage_id: [<is_forward, micro_batch_id>, <is_forward,
// micro_batch_id>, ...], ...}
std::unordered_map<size_t, std::vector<std::pair<bool, size_t>>>
ExecutableGraph::GenerateGpipeSchedule(
  size_t num_stages, size_t num_micro_batches, bool is_inference) {
  std::unordered_map<size_t, std::vector<std::pair<bool, size_t>>> schedule;
  // inference time: for only forward
  if (is_inference) {
    for (size_t stage_id = 0; stage_id < num_stages; stage_id++) {
      std::vector<std::pair<bool, size_t>> tasks;
      tasks.reserve(num_micro_batches);
      for (size_t step_id = 0; step_id < num_micro_batches; step_id++) {
        tasks.push_back({true, step_id});
      }
      schedule[stage_id] = tasks;
    }
    return schedule;
  }
  // traininig time: for forward and backward
  for (size_t stage_id = 0; stage_id < num_stages; stage_id++) {
    std::vector<std::pair<bool, size_t>> tasks;
    tasks.reserve(2 * num_micro_batches);
    for (size_t step_id = 0; step_id < num_micro_batches; step_id++) {
      tasks.push_back({true, step_id});
    }
    for (size_t step_id = 0; step_id < num_micro_batches; step_id++) {
      tasks.push_back({false, step_id});
    }
    schedule[stage_id] = tasks;
  }
  return schedule;
}

// schedule: {stage_id: [<is_forward, micro_batch_id>, <is_forward,
// micro_batch_id>, ...], ...}
std::unordered_map<size_t, std::vector<std::pair<bool, size_t>>>
ExecutableGraph::GeneratePipedreamFlushSchedule(
  size_t num_stages, size_t num_micro_batches, bool is_inference) {
  HT_ASSERT(num_micro_batches >= num_stages)
    << "num_micro_batches must bigger than num_stages in pipedream-flush!";
  std::unordered_map<size_t, std::vector<std::pair<bool, size_t>>> schedule;
  // inference time: for only forward
  if (is_inference) {
    for (size_t stage_id = 0; stage_id < num_stages; stage_id++) {
      std::vector<std::pair<bool, size_t>> tasks;
      tasks.reserve(num_micro_batches);
      for (size_t step_id = 0; step_id < num_micro_batches; step_id++) {
        tasks.push_back({true, step_id});
      }
      schedule[stage_id] = tasks;
    }
    return schedule;
  }
  // traininig time: for forward and backward
  for (size_t stage_id = 0; stage_id < num_stages; stage_id++) {
    std::vector<std::pair<bool, size_t>> tasks;
    tasks.reserve(2 * num_micro_batches);
    size_t num_warmup_microbatches = num_stages - stage_id - 1;
    size_t num_microbatches_remaining =
      num_micro_batches - num_warmup_microbatches;
    // 1. warmup
    for (size_t step_id = 0; step_id < num_warmup_microbatches; step_id++) {
      tasks.push_back({true, step_id});
    }
    // 2. 1F1B
    for (size_t step_id = 0; step_id < num_microbatches_remaining; step_id++) {
      tasks.push_back({true, num_warmup_microbatches + step_id});
      tasks.push_back({false, step_id});
    }
    // 3. cooldown
    for (size_t step_id = 0; step_id < num_warmup_microbatches; step_id++) {
      tasks.push_back({false, num_microbatches_remaining + step_id});
    }
    schedule[stage_id] = tasks;
  }
  return schedule;
}

void ExecutableGraph::ComputeFunc(size_t& micro_batch_id, const OpRefList& topo, RuntimeContext& runtime_ctx, 
                                  Tensor2NDArrayMap& tensor2data, Tensor2IntMap& tensor2degrees, 
                                  Tensor2NDArrayMap& grad_accumulation, bool grad_accumulation_finished,
                                  const TensorIdSet& accumulated_tensor, const OpIdSet& accumulated_ops,
                                  const FeedDict& feed_dict, const TensorList& fetches,
                                  const std::unordered_map<TensorId, size_t>& fetch_indices) {
  for (auto& op_ref : topo) {
    auto& op = op_ref.get();
    bool computed = Operator::all_output_tensors_of(op, [&](Tensor& tensor) {
      return feed_dict.find(tensor->id()) != feed_dict.end();
    });
    if (computed)
      continue;
    
    if (!grad_accumulation_finished && accumulated_ops.find(op->id()) != accumulated_ops.end()) {
      continue;
    }

    NDArrayList input_vals;
    input_vals.reserve(op->num_inputs());
    for (const auto& input : op->inputs()) {
      auto it = tensor2data.find(input->id());
      HT_ASSERT(it != tensor2data.end() && it->second.is_defined())
        << "Failed to execute the \"" << op->type() << "\" operation "
        << "(with name \"" << op->name() << "\"): "
        << "Cannot find input " << input;
      auto& data = it->second;
      if (data->device() != input->placement() ||
          data->dtype() != input->dtype()) {
        tensor2data[input->id()] =
          NDArray::to(data, input->placement(), input->dtype(),
                      kBlockingStream);
      }
      input_vals.push_back(tensor2data[input->id()]);
      // should free memory until op aync compute complete!!!
      // workaround: erase when the stream of input_op is the same as cur_op 
      if ((--tensor2degrees[input->id()]) == 0 && fetch_indices.find(input->id()) == fetch_indices.end()) {
      //   tensor2data.erase(input->id());
      }
    }
    NDArrayList output_vals = op->Compute(input_vals, runtime_ctx, micro_batch_id);
    for (size_t i = 0; i < op->num_outputs(); i++) {
      const auto& output = op->output(i);
      if (accumulated_tensor.find(output->id()) != accumulated_tensor.end()) {
        if (grad_accumulation.find(output->id()) == grad_accumulation.end()) {
          grad_accumulation[output->id()] = output_vals[i];
        } else {
          grad_accumulation[output->id()] = NDArray::add(grad_accumulation[output->id()], output_vals[i]);
        }
        if (grad_accumulation_finished) {
          tensor2data[output->id()] = grad_accumulation[output->id()];
        }
      } else if (tensor2degrees[output->id()] > 0 || fetch_indices.find(output->id()) != fetch_indices.end()) {
        tensor2data[output->id()] = output_vals[i];
      }
    }
  }
}

NDArrayList ExecutableGraph::Run(const Tensor& loss, const TensorList& fetches, 
                                 const FeedDict& feed_dict, const int num_micro_batches) {                        
  auto& local_device = hetu::impl::comm::GetLocalDevice();
  HT_LOG_DEBUG << local_device << ": exec graph run begin .............";
  _num_micro_batches = num_micro_batches;

  auto is_op_computed = [&](const Operator& op) -> bool {
    return Operator::all_output_tensors_of(op, [&](const Tensor& tensor) {
      return feed_dict.find(tensor->id()) != feed_dict.end();
    });
  };
  // TODO: For each pair of `fetches` and `feed_dict`,
  // deduce the optimal execution plan, and cache it.
  for (auto& fetch : fetches) {
    if (fetch->placement_group().empty() || 
        (fetch->placement_group().contains(local_device) && 
         fetch->placement().is_undetermined())) {
      // instantiate ops
      Instantiate(fetches, local_device);
      // init topo contains comm_op
      OpRefList topo = Graph::TopoSort(fetches, num_ops(), is_op_computed);
      HT_LOG_DEBUG << local_device << ": global topo before substitute comm_op: " << topo;

      // substitute comm_op
      HT_LOG_DEBUG << local_device << ": substitute comm_op begin...";
      Graph::push_graph_ctx(id()); // ensure the new ops created in execute_graph
      SubstituteCommOp(topo);
      Graph::pop_graph_ctx();
      HT_LOG_DEBUG << local_device << ": substitute comm_op end...";      
      break;
    }
  }

  // update topo
  OpRefList updated_topo = Graph::TopoSort(fetches, -1, is_op_computed);
  HT_LOG_DEBUG << local_device << ": updated global topo after substitute comm_op: " << updated_topo;

  // split into fw_topo and bw_topo
  OpRefList fw_topo, bw_topo;
  std::tie(fw_topo, bw_topo) = disentangle_forward_and_backward_ops_by_loss(updated_topo, {loss});
  // OpRefList fw_topo, bw_topo;
  // std::tie(fw_topo, bw_topo) = disentangle_forward_and_backward_ops(updated_topo);

  // get local_fw_topo and local_bw_topo
  // ops to substitute comm_op is in the same placement_group, but in the different placement
  OpRefList local_fw_topo, local_bw_topo, local_topo;
  std::copy_if(fw_topo.begin(), fw_topo.end(), std::back_inserter(local_fw_topo),
  [&](OpRef& op_ref) { return op_ref.get()->placement() == local_device; });
  std::copy_if(bw_topo.begin(), bw_topo.end(), std::back_inserter(local_bw_topo),
  [&](OpRef& op_ref) { return op_ref.get()->placement() == local_device; });
  local_topo.reserve(local_fw_topo.size() + local_bw_topo.size());
  local_topo.insert(local_topo.end(), local_fw_topo.begin(), local_fw_topo.end());
  local_topo.insert(local_topo.end(), local_bw_topo.begin(), local_bw_topo.end());
  HT_LOG_DEBUG << local_device << ": local fw topo: " << local_fw_topo << "\nlocal bw topo: " << local_bw_topo;

  // // calculate params
  // int64_t params_size = 0;
  // for (auto& op : local_topo) {
  //   if (is_variable_op(op)) {
  //     params_size += op.get()->output(0)->numel();
  //     HT_LOG_INFO << local_device << ": variable op " << op << ", shape = " << op.get()->output(0)->shape();
  //   }
  // }
  // HT_LOG_INFO << local_device << ": params_size = " << params_size;

  HT_LOG_DEBUG << local_device << ": 1. pipeline init[begin]";
  // pipeline compute
  // runtimectx for m micro batches
  std::vector<RuntimeContext> runtime_ctx_list(num_micro_batches, 
    RuntimeContext(local_topo.size()));
  // tensor data for m micro batches
  std::vector<Tensor2NDArrayMap> tensor2data_list(num_micro_batches);
  // tensor degrees for m micro batches, if degree=0 && not in fetches, free memory for this tensor
  std::vector<Tensor2IntMap> tensor2degrees_list(num_micro_batches);
  // flush update once for m micro batches
  Tensor2NDArrayMap grad_accumulation;
  
  std::unordered_map<TensorId, size_t> fetch_indices;
  for (size_t i = 0; i < fetches.size(); i++)
    fetch_indices[fetches.at(i)->id()] = i;
    
  // get feed in dict & split into m micro batches
  for (const auto& kv : feed_dict) {
    if (!kv.second.is_defined()) continue; // only feed placeholder_op in local device group
    auto micro_batches = NDArray::split(kv.second, num_micro_batches);
    // 加一个pipeline split的tensor状态
    for (int i = 0; i < num_micro_batches; i++) {
      // tensor2data_list[i][kv.first] = NDArray::squeeze(micro_batches[i], 0);
      tensor2data_list[i][kv.first] = micro_batches[i];
    }
  }

  // get consume times for each tensor
  Tensor2IntMap tensor2degrees;
  for (auto& op_ref : local_topo) {
    for (auto& input : op_ref.get()->inputs()) {
      tensor2degrees[input->id()]++;
    }
  }
  for (int i = 0; i < num_micro_batches; i++) {
    tensor2degrees_list[i] = tensor2degrees;
  }

  // some special ops shouldn't be updated before grad accumulation finished
  TensorIdSet accumulated_tensor;
  OpRefDeque accumulated_ops_deque;
  for (auto& op_ref : local_bw_topo) {
    auto& op = op_ref.get();
    // 1. compute_op -(local_grad)-> update_op (local_group)
    // 2. compute_op -(local_grad)-> allreduce -> update_op (local_group)
    // 3. compute_op -(grad_in_group2)-> p2p_send (group1)  p2p_recv -> update_op (group2)
    // 4. compute_op -(grad_in_group2)-> allreduce -> p2p_send (goup1)  p2p_recv -> update_op (group2)
    if (is_optimizer_update_op(op)) {
      Tensor& grad = op->input(1);
      Operator& grad_op = grad->producer();
      if (is_all_reduce_op(grad_op)) {
        accumulated_tensor.insert(grad_op->input(0)->id());
        accumulated_ops_deque.push_back(std::ref(grad_op));
      } else if (is_peer_to_peer_recv_op(grad_op)) {
        accumulated_ops_deque.push_back(std::ref(grad_op));
      } else {
        accumulated_tensor.insert(grad->id());
        accumulated_ops_deque.push_back(op_ref);
      }
    } else if (is_peer_to_peer_send_op(op)) {
      for (auto& consumer_op : op->out_dep_linker()->consumers()) {
        if (is_optimizer_update_op(consumer_op)) {
          Tensor& grad = op->input(0);
          Operator& grad_op = grad->producer();
          if (is_all_reduce_op(grad_op)) {
            accumulated_tensor.insert(grad_op->input(0)->id());
            accumulated_ops_deque.push_back(std::ref(grad_op));
          } else {
            accumulated_tensor.insert(grad->id());
            accumulated_ops_deque.push_back(op_ref);
          }
          break;
        }
      }
    }
  }
  OpIdSet accumulated_ops;
  while (!accumulated_ops_deque.empty()) {
    auto& op_ref = accumulated_ops_deque.front();
    accumulated_ops_deque.pop_front();
    accumulated_ops.insert(op_ref.get()->id());
    Operator::for_each_output_tensor(op_ref.get(), [&](const Tensor& output) {
      for (auto& consumer_op : output->consumers()) {
        if (consumer_op.get()->placement() == local_device) {
          accumulated_ops_deque.push_back(consumer_op);
        }
      }
    });
  }
  HT_LOG_DEBUG << local_device << ": 1. pipeline init[end]";

  HT_LOG_DEBUG << local_device << ": 2. compute[begin]";
  int num_stages = _stages.size();
  bool is_inference = (bw_topo.size() == 0);
  HT_LOG_DEBUG << local_device << ": num_stages = " << num_stages 
    << ", num_micro_batches = " << num_micro_batches << ", is_inference = " 
    << is_inference;
  // get task schedule table for pipedream-flush
  auto schedule = GeneratePipedreamFlushSchedule(
    num_stages, num_micro_batches, is_inference);
  // // get task schedule table for gpipe    
  // auto schedule = generate_gpipe_schedule(num_stages, num_micro_batches);
  // get tasks for current stage
  int stage_id = local_device.index() / _stages.at(0).num_devices();
  // int stage_id = -1;
  // for (int i = 0; i < _stages.size(); i++) {
  //   if (_stages[i].contains(local_device)) {
  //     stage_id = i;
  //   }
  // }
  // HT_LOG_DEBUG << local_device << ": stages = " << _stages << "; stage id = " << stage_id;
  auto& tasks = schedule[stage_id];
  HT_LOG_DEBUG << local_device << ": stage id = " << stage_id;
  for (size_t i = 0; i < tasks.size(); i++) {
    auto& task = tasks[i];
    bool is_forward = task.first;
    size_t& micro_batch_id = task.second;
    auto& tensor2data = tensor2data_list[micro_batch_id];
    auto& tensor2degrees = tensor2degrees_list[micro_batch_id];
    auto& runtime_ctx = runtime_ctx_list[micro_batch_id];
    if (is_forward) {
      ComputeFunc(micro_batch_id, local_fw_topo, runtime_ctx, tensor2data, tensor2degrees, grad_accumulation,
                  false, accumulated_tensor, accumulated_ops, feed_dict, fetches, fetch_indices);
    } else {
      bool grad_accumulation_finished = (i == tasks.size() - 1);
      ComputeFunc(micro_batch_id, local_bw_topo, runtime_ctx, tensor2data, tensor2degrees, grad_accumulation,
                  grad_accumulation_finished, accumulated_tensor, accumulated_ops, feed_dict, 
                  fetches, fetch_indices);
    }
    if (is_forward) {
      HT_LOG_DEBUG << local_device << ": [micro batch " << micro_batch_id << ": forward]";
    } else {
      HT_LOG_DEBUG << local_device << ": [micro batch " << micro_batch_id << ": backward]";
    }
  }
  HT_LOG_DEBUG << local_device << ": 2. compute[end]";

  HT_LOG_DEBUG << local_device << ": 3. get results[begin]";
  NDArrayList results(fetches.size(), NDArray());
  std::unordered_set<OpId> to_sync_op_ids;
  to_sync_op_ids.reserve(fetches.size());
  for (auto& op_ref : local_topo) {
    auto& op = op_ref.get();
    Operator::for_each_output_tensor(op, [&](const Tensor& output) {
      auto it = fetch_indices.find(output->id());
      if (it != fetch_indices.end()) {
        if (output->output_id() >= 0) {
          if (is_variable_op(op) || accumulated_ops.find(op) != accumulated_ops.end() 
            || accumulated_tensor.find(output->id()) != accumulated_tensor.end()) {
            results[it->second] = tensor2data_list[num_micro_batches - 1][output->id()];
          } else if (is_placeholder_op(op)) {
            auto feed_it = feed_dict.find(output->id());
            if (feed_it != feed_dict.end()) {
              results[it->second] = feed_it->second;
            }
          } else {
            NDArrayList result;
            result.reserve(num_micro_batches);
            for (auto& tensor2data : tensor2data_list) {
              result.push_back(tensor2data[output->id()]);
            }
            results[it->second] = NDArray::cat(result);
          }
        }
        to_sync_op_ids.insert(op->id());
      }
    });
  }
  // OpList sync_ops;
  for (auto op_id : to_sync_op_ids) {
    _op_indexing[op_id]->Sync(num_micro_batches - 1);
    // sync_ops.push_back(_op_indexing[op_id]);
  }
  // HT_LOG_DEBUG << local_device << ": sync ops = " << sync_ops;
  HT_LOG_DEBUG << local_device << ": 3. get results[end]";

  // get op execute time, sort and analysis
  bool is_analysis_perf = false;
  if (local_device.index() == 0 && is_analysis_perf) {
    std::vector<std::pair<OpId, int64_t>> op_execute_time;
    // HT_LOG_INFO << local_device << ": local_topo = " << local_topo;
    for (auto& op_ref : local_topo) {
      auto& op = op_ref.get();
      if (is_placeholder_op(op) || is_variable_op(op)) {
        continue;
      }
      // get time cost for all micro batches
      int64_t time_cost = 0;
      for (int i = 0; i < num_micro_batches; i++) {
        time_cost += op->TimeCost(i);
      }
      op_execute_time.push_back({op->id(), time_cost});
    }
    std::sort(op_execute_time.begin(), op_execute_time.end(), [](
      std::pair<OpId, int64_t>& op_t1, std::pair<OpId, int64_t>& op_t2) {
        return op_t1.second > op_t2.second;
      });
    double compute_time = 0;
    double p2p_time = 0;
    double collective_time = 0;
    double blocking_time = 0;
    double other_time = 0;
    std::ostringstream out;
    out << "Op Execute Time: ";
    for (auto& op_time : op_execute_time) {
      auto op = _op_indexing[op_time.first];
      if (is_all_reduce_op(op)) {
        auto allreduce_op = op;
        auto& allreduce_impl = reinterpret_cast<AllReduceOpImpl&>(allreduce_op->body());
        out << std::endl << local_device << ": " 
            << allreduce_op->input(0) << ", shape = "
            << allreduce_op->input(0)->shape() << ", type = "
            << allreduce_op->input(0)->dtype() << ", comm group = [";
        auto comm_group = allreduce_impl.comm_group();
        for (auto device : comm_group.devices()) {
          out << comm_group.get_index(device) << ", ";
        }
        out << "]";
      } else {
        if (op->num_inputs() > 0)
        out << std::endl << local_device << ": " 
            << op->input(0) << ", shape = "
            << op->input(0)->shape() << ", type = "
            << op->input(0)->dtype();        
      }
      out << std::endl << local_device << ": " << op << ": " << op_time.second * 1.0 / 1e6 << " ms";

      if (op->stream_index() == kComputingStream) {
        compute_time += op_time.second * 1.0 / 1e6;
      } else if (op->stream_index() == kP2PStream) {
        p2p_time += op_time.second * 1.0 / 1e6;
      } else if (op->stream_index() == kCollectiveStream) {
        collective_time += op_time.second * 1.0 / 1e6;
      } else if (op->stream_index() == kBlockingStream) {
        blocking_time += op_time.second * 1.0 / 1e6;
      } else {
        other_time += op_time.second * 1.0 / 1e6;
      }
    }
    HT_LOG_INFO << local_device << ": " 
                << "compute time: " << compute_time << " ms, "
                << "p2p time: " << p2p_time << " ms, "
                << "collective time: " << collective_time << " ms, "
                << "blocking time: " << blocking_time << " ms, "
                << "other time: " << other_time << " ms" << std::endl
                << out.str();
  }
  return results;
}

// TODO: merge two `Run` func
NDArrayList ExecutableGraph::Run(const TensorList& fetches,
                                 const FeedDict& feed_dict) {
  // TODO: For each pair of `fetches` and `feed_dict`,
  // deduce the optimal execution plan, and cache it.
  for (auto& fetch : fetches) {
    if (fetch->placement().is_undetermined()) {
      Instantiate(fetches, kCUDA);
      break;
    }
  }

  auto is_op_computed = [&](const Operator& op) -> bool {
    return Operator::all_output_tensors_of(op, [&](const Tensor& tensor) {
      return feed_dict.find(tensor->id()) != feed_dict.end();
    });
  };
  OpRefList topo = Graph::TopoSort(fetches, num_ops(), is_op_computed);

  RuntimeContext runtime_ctx(topo.size());
  Tensor2NDArrayMap tensor2data;
  tensor2data.reserve(topo.size());
  tensor2data.insert(feed_dict.begin(), feed_dict.end());
  NDArrayList results(fetches.size());
  std::unordered_map<TensorId, size_t> fetch_indices;
  for (size_t i = 0; i < fetches.size(); i++)
    fetch_indices[fetches.at(i)->id()] = i;
  std::unordered_set<OpId> to_sync_op_ids;
  to_sync_op_ids.reserve(fetches.size());

  for (auto& op_ref : topo) {
    auto& op = op_ref.get();
    // Question: Is it possible that some outputs are fed in
    // while the rest are not?
    bool computed = Operator::all_output_tensors_of(op, [&](Tensor& tensor) {
      return feed_dict.find(tensor->id()) != feed_dict.end();
    });
    if (computed)
      continue;

    NDArrayList inputs;
    inputs.reserve(op->num_inputs());
    for (size_t i = 0; i < op->num_inputs(); i++) {
      // TODO: Support async transfer. And this could be checked for once.
      auto& data = tensor2data[op->input(i)->id()];
      if (data->device() != op->input(i)->placement() ||
          data->dtype() != op->input(i)->dtype()) {
        tensor2data[op->input(i)->id()] =
          NDArray::to(data, op->input(i)->placement(), op->input(i)->dtype(),
                      op->stream_index());
      }
      inputs.push_back(tensor2data[op->input(i)->id()]);
    }
    auto outputs = op->Compute(inputs, runtime_ctx);

    for (size_t i = 0; i < outputs.size(); i++) {
      tensor2data.insert({op->output(i)->id(), outputs[i]});
      auto it = fetch_indices.find(op->output(i)->id());
      if (it != fetch_indices.end()) {
        results[it->second] = outputs[i];
        to_sync_op_ids.insert(op->id());
      }
    }
    // TODO: remove inputs that are no longer used
  }
  for (auto op_id : to_sync_op_ids) {
    _op_indexing[op_id]->Sync();
  }
  return results;
}

} // namespace graph
} // namespace hetu
