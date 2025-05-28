#include "hetu/graph/executable_graph.h"
#include "hetu/common/logging.h"
#include "hetu/core/device.h"
#include "hetu/core/dtype.h"
#include "hetu/core/memory_pool.h"
#include "hetu/core/ndarray.h"
#include "hetu/core/stream.h"
#include "hetu/graph/common.h"
#include "hetu/graph/operator.h"
#include "hetu/graph/switch_exec_graph.h"
#include "hetu/graph/ops/op_headers.h"
#include "hetu/graph/autocast/autocast.h"
#include "hetu/graph/recompute/recompute.h"
#include "hetu/graph/offload/activation_cpu_offload.h"
#include "hetu/impl/communication/comm_group.h"
#include "hetu/impl/communication/nccl_comm_group.h"
#include "hetu/impl/profiler/profiler.h"
#include "hetu/impl/utils/cuda_utils.h"
#include "hetu/core/symbol.h"
#include "hetu/core/ndarray_storage.h"
#include <any>
#include <cstdint>
#include <memory>
#include <nccl.h>
#include <ctime>
#include <iostream>
#include <fstream>

namespace hetu {
namespace graph {

/*
static size_t GetLayerId(const Tensor& tensor) {
  std::string sub_str = "block";
  std::string name = tensor->name();
  // 将name转换为小写进行匹配
  std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) {
      return std::tolower(c);
  });
  size_t pos = name.find(sub_str);
  HT_ASSERT (pos != std::string::npos) 
    << "Can't find block num in the tensor name " << tensor->name();
  size_t next_char_pos = pos + sub_str.length();
  HT_ASSERT (next_char_pos < tensor->name().length())
    << "Can't find block num in the tensor name " << tensor->name();
  std::string layer_num_str = "";
  while (tensor->name()[next_char_pos] != std::string::npos
         && tensor->name()[next_char_pos] >= '0' 
         && tensor->name()[next_char_pos] <= '9') {
    layer_num_str += tensor->name()[next_char_pos];
    next_char_pos += 1;
  }
  HT_ASSERT(layer_num_str != "")
    << "Cannot fetch the number after 'Block' for " << tensor->name();
  size_t layer_num = std::stoi(layer_num_str);
  return layer_num;
}
*/

static bool is_comm_without_reduce_op(const uint64_t comm_type) {
  return comm_type & (PEER_TO_PEER_SEND_OP | PEER_TO_PEER_RECV_OP |
                      ALL_TO_ALL_OP | ALL_GATHER_OP | BROADCAST_OP |
                      P2P_OP | BATCHED_ISEND_IRECV_OP |
                      GATHER_OP | SCATTER_OP) != 0;
}

bool ExecutableGraph::is_pipeline_stage_send_op(Operator& op) {
  if (op->graph().GetSubGraph(op) == nullptr || op->graph().GetSubGraph(op)->subgraph_type() != SubGraphType::PIPELINE) {
    return false;
  }
  if (is_peer_to_peer_send_op(op)) {
    return true;
  }
  if (is_batched_isend_irecv_op(op)) {
    const auto& batched_isend_irecv_op_impl = dynamic_cast<BatchedISendIRecvOpImpl&>(op->body());
    // 只发不收
    if (batched_isend_irecv_op_impl.src_devices().empty()) {
      HT_ASSERT(!batched_isend_irecv_op_impl.dst_devices().empty())
        << "only one side could be empty";
      return true;
    }
  }
  return false;
}

bool ExecutableGraph::is_pipeline_stage_recv_op(Operator& op) {
  if (op->graph().GetSubGraph(op) == nullptr || op->graph().GetSubGraph(op)->subgraph_type() != SubGraphType::PIPELINE) {
    return false;
  }
  if (is_peer_to_peer_recv_op(op)) {
    return true;
  }
  if (is_batched_isend_irecv_op(op)) {
    const auto& batched_isend_irecv_op_impl = dynamic_cast<BatchedISendIRecvOpImpl&>(op->body());
    // 只收不发
    if (batched_isend_irecv_op_impl.dst_devices().empty()) {
      HT_ASSERT(!batched_isend_irecv_op_impl.src_devices().empty())
        << "only one side could be empty";
      return true;
    }
  }
  return false;
}

Operator& ExecutableGraph::MakeOpInner(std::shared_ptr<OpInterface> body,
                                       TensorList inputs, OpMeta op_meta) {
  _check_all_inputs_in_graph(inputs, op_meta.extra_deps);
  // Use in DoInferMeta
  // Some ops need a inferred device
  // so that the output tensor shape is determined
  if (op_meta.device_group_hierarchy.size() != 0) {
    DeviceGroupUnion device_group_union;
    if (op_meta.device_group_hierarchy.size() == 1) {
      device_group_union = op_meta.device_group_hierarchy.get(0);
    } else {
      device_group_union = op_meta.device_group_hierarchy.get(CUR_STRATEGY_ID);
    }
    auto inferred = hetu::impl::comm::GetLocalDevice();
    if (device_group_union.has(inferred)) {
      CUR_HETERO_ID = device_group_union.get_index(inferred);
    } else {
      CUR_HETERO_ID = 0;
    }
  }
  OpRef op_ref = MakeAndAddOp(std::move(body), std::move(inputs), std::move(op_meta));
  CUR_HETERO_ID = 0;
  // set suggested hetero id
  // 默认直接使用deduce pipeline时得到的第COMPUTE_SUGGESTED_HETERO_ID
  op_ref.get()->set_suggested_hetero_id(COMPUTE_SUGGESTED_HETERO_ID);
  // record the subgraph
  if (get_cur_subgraph_global_name() != "") {
    AddOpToSubGraph(op_ref, get_cur_subgraph_global_name(), get_cur_subgraph_op_type());
  }
  return op_ref;
}

bool ExecutableGraph::Instantiate(const TensorList& fetches,
                                  const Device& preferred_device) {
  auto is_op_instantiated = [&](const Operator& op) -> bool {
    return !op->placement().is_undetermined();
  };
  OpRefList topo = Graph::TopoSort(fetches, num_ops(), is_op_instantiated);
  HT_LOG_TRACE << "Instantiating ops: " << topo;

  HT_LOG_DEBUG << "global info for all devices begin...";
  // global info for all devices
  for (auto& op_ref : topo) {
    auto& op = op_ref.get();
    if (!op->placement().is_undetermined())
      continue;
    // 处理1
    // handle unused or redundant comm ops
    if (is_comm_op(op) && op->placement_group().contains(preferred_device)) {
      auto& comm_op_impl = dynamic_cast<CommOpImpl&>(op->body());
      // 1. remove unused comm ops
      // HT_LOG_INFO << preferred_device << ": " << op << " comm info is " << comm_op_impl.get_comm_info(op, preferred_device);
      if (comm_op_impl.get_comm_type(op, preferred_device) == UNUSED_OP) {
        // HT_LOG_INFO << op << " is an unused comm op and will be removed";
        // 将op从exec graph中删除
        DeleteExecOp(op);
        // the former op of the unused comm op should have the same recompute setting
        op->input(0)->producer()->op_meta().set_multi_recompute(op->op_meta().multi_is_recompute);
        // should remove consumer of unused comm_op from end to begin
        for (int i = op->output(0)->num_consumers() - 1; i >= 0; i--) {
          auto& consumer_i = op->output(0)->consumer(i);
          for (int j = 0; j < consumer_i->num_inputs(); j++) {
            if (consumer_i->input(j)->id() == op->output(0)->id()) {
              Graph::ReplaceInput(consumer_i, j, op->input(0));
            }
          }
          for (int j = 0; j < consumer_i->num_in_dep_linkers(); j++) {
            if (consumer_i->in_dep_linker(j)->id() == op->output(0)->id()) {
              Graph::ReplaceInDepLinker(consumer_i, j, op->input(0));
            }
          }
        }
        continue;
      }
      // 2. fuse redundant comm ops
      auto& input_op = op->input(0)->producer();
      if (is_comm_op(input_op)) {
        // 尝试融合input op和op两个comm算子
        // *目前只支持将不含partial的两个算子融合成一个BatchedIsendIrecv
        auto& input_comm_op_impl = dynamic_cast<CommOpImpl&>(input_op->body());
        if (is_comm_without_reduce_op(input_comm_op_impl.get_comm_type(input_op, preferred_device))
            && is_comm_without_reduce_op(comm_op_impl.get_comm_type(op, preferred_device))) {
          HT_LOG_WARN << "Fuse " << input_op << " with type " << input_comm_op_impl.get_comm_type(input_op, preferred_device)
            << " and " << op << " with type " << comm_op_impl.get_comm_type(op, preferred_device);
          // 将op从exec graph中删除
          DeleteExecOp(input_op);
          Graph::ReplaceInput(op, 0, input_op->input(0));
          // input changes, update comm_op type
          comm_op_impl.get_comm_type(op, preferred_device);
        }
      }
    }
    // 处理2
    // loss & grad should div by num_micro_batches when reduction type = MEAN
    if (is_loss_gradient_op(op) && op->input(0)->has_distributed_states()) {
      int dp = op->input(0)->get_distributed_states().get_dim(0);
      auto& loss_grad_op_impl = dynamic_cast<LossGradientOpImpl&>(op->body());
      if ((_num_micro_batches > 1 || dp > 1) && loss_grad_op_impl.reduction() == kMEAN) {
        auto& grads = op->outputs();
        for (auto& grad : grads) {
          if (!grad.is_defined()) {
            continue;
          }
          // TODO: use symbolic shape, replace _num_micro_batches * dp with global_token / local_token
          Tensor grad_scale = MakeDivByConstOp(grad, _num_micro_batches * dp, OpMeta().set_name(grad->name() + "_scale"));
          auto cur_subgraph = GetSubGraph(grad->producer());
          if (cur_subgraph != nullptr) {
            AddOpToSubGraph(grad_scale->producer(), cur_subgraph->global_name(), SubGraphOpType::BACKWARD);
            HT_LOG_WARN << "find the subgraph of ultimate grad op " << grad->producer() << " " << grad; 
          } else {
            HT_LOG_WARN << "cannot find the subgraph of ultimate grad op " << grad->producer() << " " << grad
              << ", though it is ok if we do not use memory plan"; 
          }
          RecordExecTensor(grad_scale);
          auto& grad_scale_op = grad_scale->producer();
          grad_scale_op->MapToParallelDevices(grad->placement_group_union());
          for (int i = grad->num_consumers() - 1; i >= 0; i--) {
            auto& consumer_i = grad->consumer(i);
            if (consumer_i->id() == grad_scale_op->id()) continue;
            for (int j = 0; j < consumer_i->num_inputs(); j++) {
              if (consumer_i->input(j)->id() == grad->id()) {
                Graph::ReplaceInput(consumer_i, j, grad_scale);
              }
            }
            for (int j = 0; j < consumer_i->num_in_dep_linkers(); j++) {
              if (consumer_i->in_dep_linker(j)->id() == grad->id()) {
                Graph::ReplaceInDepLinker(consumer_i, j, grad_scale);
              }
            }
          }
        }
      }
    }
    // TODO: 处理3
    // if consecutive ops have different placement groups
    // need to insert comm op automatically
    // 目前已在py端手动插入 
  }
  // HT_LOG_WARN << "global info for all devices end...";
  
  // get updated topo
  OpRefList updated_topo = Graph::TopoSort(fetches, num_ops(), is_op_instantiated);
  HT_LOG_DEBUG << "local info for local_device begin, topo is " << updated_topo;
  // local info for local_device
  for (auto& op_ref : updated_topo) {
    auto& op = op_ref.get();
    // HT_LOG_WARN << op << " placement group is " << op->placement_group();
    if (!op->placement().is_undetermined())
      continue;  
    
    Device preferred_device_ = preferred_device;
    if (op->op_meta().is_cpu)
      preferred_device_ = kCPU;
    else if (!op->placement_group().contains(preferred_device_)) // for local compute: op->placement + tensor->placement
      continue;
    Device placement = is_device_to_host_op(op) ? Device(kCPU) : preferred_device_;
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
      if (op->type() == "AdamOp" && i == 4 || input->producer()->op_meta().is_cpu)
        continue;
      if (input->placement() != placement && !is_comm_op(op)) {
        HT_LOG_WARN << op << " placement is " << placement
          << ", but input " << input << " placement is " << input->placement()
          << ", so needs to add data transfer op";
        HT_RUNTIME_ERROR_IF(input->placement().is_undetermined())
          << input << " placement is undetermined";
        HT_RUNTIME_ERROR_IF(!input->placement().local())
          << "Please use P2P communication to fetch remote input";
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
          HT_NOT_IMPLEMENTED << "We should use NCCL for P2P communication: " << op->type();
          __builtin_unreachable();
        }
        auto cur_subgraph = GetSubGraph(input->producer());
        if (cur_subgraph != nullptr) {
          AddOpToSubGraph(transferred_input->producer(), cur_subgraph->global_name(), GetSubGraphOpType(input->producer()));
        } else {
          HT_LOG_WARN << "cannot find the subgraph of device transfer input op " << input->producer()
            << ", though it is ok if we do not use memory plan"; 
        }
        RecordExecTensor(transferred_input);
        auto& transfer_op = transferred_input->producer();
        if (!input->placement_group_union().size() == 0)
          transfer_op->MapToParallelDevices(input->placement_group_union());
        transfer_op->Instantiate(placement, transfer_stream_id);
        Graph::ReplaceInput(op, i, transferred_input);
      }
    }
  }
  HT_LOG_DEBUG << "local info for local_device end...";
  return true;
}

void ExecutableGraph::InsertContiguousOp(const OpRefList& topo_order) {
  auto& local_device = hetu::impl::comm::GetLocalDevice();
  for (auto& op_ref : topo_order) {
    auto& op = op_ref.get();
    if (op->body().require_contig_inputs()) {
      for (size_t i = 0; i < op->num_inputs(); i++) {
        auto& input = op->input(i);
        if (!input->placement_group_union().has(local_device))
          continue;
        if (!input->is_contiguous()) {
          auto op_id = input->get_contiguous_op_id();
          if (op_id.has_value() &&
              _op_indexing[op_id.value()]->placement() == local_device) {
            HT_LOG_TRACE << "Tensor " << input->name()
                         << " is not contiguous for op " << op->body().type()
                         << ". But it may have a contiguous copy, use it instead";
            auto contig_op = _op_indexing[op_id.value()];
            Graph::ReplaceInput(op, i, contig_op->output(0));
          } else {
            HT_LOG_TRACE << hetu::impl::comm::GetLocalDevice() << ": Make Contiguous op for tensor " << input->name()
                         << " while making " << op->body().type() << " op.";
            Tensor contig_input = MakeContiguousOp(
              input, OpMeta().set_name(input->name() + "_contig")
                             .set_is_deduce_states(false));
            HT_LOG_TRACE << "Insert contiguous op for " << input
              << ", shape is " << input->shape()
              << ", stride is " << input->stride();
            auto cur_subgraph = GetSubGraph(input->producer());
            if (cur_subgraph != nullptr) {
              AddOpToSubGraph(contig_input->producer(), cur_subgraph->global_name(), GetSubGraphOpType(input->producer()));
            } else {
              HT_LOG_WARN << "cannot find the subgraph of contiguous input op " << input->producer()
                << ", though it is ok if we do not use memory plan"; 
            }
            RecordExecTensor(contig_input);
            auto& contig_op = contig_input->producer();
            contig_op->MapToParallelDevices(input->placement_group_union());
            contig_op->Instantiate(local_device, kComputingStream);
            Graph::ReplaceInput(op, i, contig_input);
          }
        }
      }
    }
  }
}

void ExecutableGraph::SubstituteCommOp(const OpRefList& topo_order) {
  auto& local_device = hetu::impl::comm::GetLocalDevice();
  // std::unordered_map<OpId, OpId> old_comm_to_new;
  for (auto& op_ref : topo_order) {
    auto& op = op_ref.get();
    // each device only need to substitute local comm_ops
    if (is_comm_op(op) && op->placement_group().contains(local_device)) {
      HT_LOG_DEBUG << local_device << "==============> substitute comm_op begin: " << op << "...";
      auto& comm_op = op;
      // 获取其所在的subgraph使得后续替换的op都出现在同样的subgraph中
      // 有如下几种可能
      // 1、在optimize-compute bridge或compute-optimize bridge的subgraph中
      // 例如zero与grad相关的(split)-all-gather/-all-reduce/-reduce-scatter或batched-send-recv
      // 2、在pipeline layer的subgraph中
      // 例如pp相关的(batched)-send-recv
      // 3、在正常的module中的subgraph中
      // 例如tp与sp相关的一些all-gather与reduce-scatter
      auto comm_subgraph = GetSubGraph(comm_op);
      HT_ASSERT(comm_subgraph != nullptr)
        << "cannot find the subgraph of the comm op to be substituted " << comm_op; 
      HT_ASSERT(get_cur_subgraph_global_name() == "" && get_cur_subgraph_op_type() == SubGraphOpType::UNKNOWN)
        << "please ensure the ctx is empty before substituting " << comm_op;
      push_subgraph_op_type_ctx(GetSubGraphOpType(comm_op));
      push_subgraph_ctx(comm_subgraph->global_name());
      StreamIndex suggested_comm_stream_idx;
      switch (comm_subgraph->subgraph_type()) {
        case SubGraphType::MODULE:
          suggested_comm_stream_idx = kCollectiveStream;
          break;
        case SubGraphType::PIPELINE:
          suggested_comm_stream_idx = kP2PStream;
          break;
        case SubGraphType::OPTIMIZE_COMPUTE_BRIDGE:
        case SubGraphType::COMPUTE_OPTIMIZE_BRIDGE:
          suggested_comm_stream_idx = kBridgeStream;
          break;
        default:
          HT_NOT_IMPLEMENTED << comm_subgraph->global_name() << " subgraph type not implemented";
          break;
      }
      auto& comm_op_impl = dynamic_cast<CommOpImpl&>(comm_op->body());
      const auto& info = comm_op_impl.get_comm_info(comm_op, local_device);
      // HT_LOG_INFO << comm_op << ": " << info;
      uint64_t comm_type = comm_op_impl.get_comm_type(comm_op, local_device, info);
      Tensor& input = comm_op->input(0);
      // *标记通信算子的输入具有symbolic shape
      if (!input->symbolic()) {
        input->init_symbolic_shape();
        AddLeafSymbolicTensor(input);
      }
      bool ignore_flag = false, local_comm_flag = false, determine_flag = false;
      Tensor result = input;
      // TODO: may use input to replace output if it is a no-recv comm op, this is not a good idea

      if ((comm_type & UNUSED_OP) != 0) {
        HT_RUNTIME_ERROR << "Unused comm op should already be deleted when instantiating";
      }
      if ((comm_type & BATCHED_ISEND_IRECV_OP) != 0) {
        // 1. local_device send data to other devices 
        // 2. local_device recv data from other devices
        // use derived method from switch exec graph
        std::unordered_set<Device> comm_set = {};
        if (_bridge_single_communicator) {
          for (const auto rank : _used_ranks) {
            comm_set.insert(hetu::impl::comm::WorldRankToDevice(rank));
          }
        }
        auto complex_exec_comm = ComplexExecComm(comm_op, info, comm_set);
        // workaround: bridge subgraph may allow mismatched global shape due to non-integer reduce-scatter problem
        if (_shape_mismatch_flag > 0 && (comm_subgraph->subgraph_type() == SubGraphType::OPTIMIZE_COMPUTE_BRIDGE || comm_subgraph->subgraph_type() == SubGraphType::COMPUTE_OPTIMIZE_BRIDGE)) {
          ignore_flag = true;
        }
        // HT_LOG_INFO << comm_op << " ignore shape mismatch flag is " << ignore_flag;
        result = complex_exec_comm.Instantiate(suggested_comm_stream_idx, ignore_flag);
        result->set_cur_ds_union(info.dst_ds_union); 
        HT_LOG_DEBUG << local_device << ": substitute " << comm_op << " to batched_isend_irecv_op";
        determine_flag = true;
      }
      if ((comm_type & P2P_OP) != 0) {
        Tensor& output = comm_op->output(0); // output meta was already deduced in DoInferMeta
        HT_ASSERT(output->shape() == result->shape())
          << "p2p shape should be equal";
        // p2p send
        if (info.src_group.contains(local_device)) {
          // 自己发给自己
          if (info.dst_group.get(info.src_group.get_index(local_device)) == local_device) {
            HT_LOG_DEBUG << local_device << ": redundant p2p send from " 
              << info.src_group << " to " << info.dst_group;
          } 
          // 发给别人
          else {
            HT_LOG_DEBUG << local_device << ": send from stage " << info.src_group << " to " << info.dst_group;
            Tensor send_out_dep_linker = MakeP2PSendOp(
              result, info.dst_group, info.src_group.get_index(local_device), 
              _used_ranks, OpMeta().set_is_deduce_states(false));
            // since send_out_dep_linker has an empty shape and is useless, recording its shape is unnecessary
            // but here we still do it to make the code looks more consistent
            RecordExecTensor(send_out_dep_linker);
            auto& send_op = send_out_dep_linker->producer();
            send_op->set_fw_op_id(result->producer()->fw_op_id());
            send_op->MapToParallelDevices(info.src_group_union);
            send_op->Instantiate(local_device, suggested_comm_stream_idx);
            // add dummy link for topo sort
            for (int i = 0; i < comm_op->output(0)->num_consumers(); i++) {
              Graph::AddInDeps(comm_op->output(0)->consumer(i), {send_out_dep_linker});
            }
            HT_LOG_DEBUG << local_device << ": substitute " << comm_op << " to p2p_send_op";        
          }
        }
        // p2p recv
        else {
          HT_ASSERT(info.dst_group.contains(local_device))
            << "dst group must contain local device";
          // 自己收自己
          if (info.src_group.get(info.dst_group.get_index(local_device)) == local_device) {
            HT_LOG_DEBUG << local_device << ": redundant p2p recv from " 
              << info.src_group << " to " << info.dst_group;
          } 
          // 自己收别人
          else {
            HT_LOG_DEBUG << local_device << ": just recv from stage " << info.src_group << " to " << info.dst_group;
            Tensor recv_output = MakeP2PRecvOp(
              info.src_group, output->dtype(), result->symbolic_shape(),
              info.dst_group.get_index(local_device), _used_ranks,
              OpMeta().set_is_deduce_states(false));
            RecordExecTensor(recv_output);
            auto& recv_op = recv_output->producer();
            recv_op->set_fw_op_id(result->producer()->fw_op_id());
            recv_op->MapToParallelDevices(info.dst_group_union);
            recv_op->Instantiate(local_device, suggested_comm_stream_idx);
            // add dummy link for topo sort
            Graph::AddInDeps(recv_op, {result});
            result = recv_output;
            HT_LOG_DEBUG << local_device << ": substitute " << comm_op << " to p2p_recv_op";        
          }
        }
        determine_flag = true;
      }
      if ((comm_type & COMM_SPLIT_OP) != 0) {
        HT_ASSERT(info.src_group == info.dst_group)
          << "wrong src and dst group relationship for " << comm_op
          << ", src_group = " << info.src_group << " and dst group = " << info.dst_group;
        auto local_device_index = info.src_group.get_index(local_device);
        auto cur_state_index = info.local_dst_ds.map_device_to_state_index(local_device_index);
        const auto& order = info.local_dst_ds.get_order();
        HTAxes keys; 
        HTShape indices, splits;
        for (auto o : order) {
          if (o >= 0 && info.local_dst_ds.get_dim(o) != info.local_src_ds.get_dim(o)) { 
            keys.push_back(o);
            auto split_num = info.local_dst_ds.get_dim(o) / info.local_src_ds.get_dim(o);
            splits.push_back(split_num);
            indices.push_back(cur_state_index[o] % split_num);
          }
        }
        HT_LOG_DEBUG << local_device << ": keys = " << keys << "; indices = " << indices << "; splits = " << splits;
        Tensor split_output = MakeSplitOp(
          result, keys, indices, splits, true,
          OpMeta().set_is_deduce_states(false)
                  .set_name("Split_for_" + comm_op->output(0)->consumer(0)->name()));
        RecordExecTensor(split_output);
        auto& split_op = split_output->producer();
        split_op->set_fw_op_id(result->producer()->fw_op_id());
        split_op->MapToParallelDevices(info.src_group_union);
        split_op->Instantiate(local_device, kComputingStream);
        result = split_output;
        HT_LOG_DEBUG << local_device << ": substitute " << comm_op << " to split_op";        
        determine_flag = true;
        local_comm_flag = true;
      }
      if ((comm_type & SCATTER_OP) != 0) {
        HT_ASSERT(info.src_group == info.dst_group)
          << "wrong src and dst group relationship!";
        auto local_device_index = info.src_group.get_index(local_device);
        auto cur_state_index = info.local_dst_ds.map_device_to_state_index(local_device_index);
        const auto& order = info.local_dst_ds.get_order();
        HTAxes keys; 
        HTShape indices, splits;
        for (auto o : order) {
          if (o >= 0 && info.local_dst_ds.get_dim(o) != info.local_src_ds.get_dim(o)) { 
            keys.push_back(o);
            auto split_num = info.local_dst_ds.get_dim(o) / info.local_src_ds.get_dim(o);
            splits.push_back(split_num);
            indices.push_back(cur_state_index[o] % split_num);
          }
        }
        Tensor scatter_output = MakeSplitOp(
          result, keys, indices, splits, true,
          OpMeta().set_is_deduce_states(false)
                  .set_name(result->name() + "_Scatter"));
        RecordExecTensor(scatter_output);
        auto& scatter_op = scatter_output->producer();
        scatter_op->set_fw_op_id(result->producer()->fw_op_id());
        scatter_op->MapToParallelDevices(info.src_group_union);
        scatter_op->Instantiate(local_device, kComputingStream);
        result = scatter_output;
        HT_LOG_DEBUG << local_device << ": substitute " << comm_op << " to scatter_op, input = " << comm_op->input(0) << ", output consumers = " << comm_op->output(0)->consumers();    
        determine_flag = true;
        local_comm_flag = true;
      }
      if ((comm_type & ALL_REDUCE_OP) != 0) {
        HT_ASSERT(info.src_group == info.dst_group)
          << "wrong src and dst group relationship!";
        DeviceGroup comm_group = comm_op_impl.get_devices_by_dim(comm_op, -2); // do allreduce among comm_group
        Tensor all_reduce_output = MakeAllReduceOp(
          result, comm_group, // comm_group is a subset of placement_group
          comm_op_impl.reduction_type(), false,
          OpMeta().set_is_deduce_states(false)
                  .set_name(result->name() + "_AllReduce"));
        RecordExecTensor(all_reduce_output);
        auto& all_reduce_op = all_reduce_output->producer();
        all_reduce_op->set_fw_op_id(result->producer()->fw_op_id());
        all_reduce_op->MapToParallelDevices(info.src_group_union);
        all_reduce_op->Instantiate(local_device, suggested_comm_stream_idx);
        result = all_reduce_output;
        HT_LOG_DEBUG << local_device << ": substitute " << comm_op << " to all_reduce_op: " << comm_group; 
        /*   
        HT_LOG_WARN << local_device << ": " << all_reduce_output << " src ds " << info.src_ds_union.ds_union_info()
          << ", and dst ds is " << info.dst_ds_union.ds_union_info(); 
        */
        determine_flag = true;
        local_comm_flag = true;
      } 
      if ((comm_type & ALL_GATHER_OP) != 0) {
        HT_ASSERT(info.src_group == info.dst_group)
          << "wrong src and dst group relationship!";
        // DeviceGroup comm_group = comm_op_impl.get_devices_by_dim(comm_op, 0);
        int32_t local_device_idx = info.dst_group.get_index(local_device);
        DeviceGroup comm_group = info.local_dst_ds.get_devices_by_dim(-1, local_device_idx, info.dst_group);
        int32_t gather_dim = info.src_ds.get_split_dim(info.dst_ds);
        // workaround for multi allgather.
        // TODO: fix this
        bool already_allgather = false;
        Tensor all_gather_output;
        for (int i = 0; i < result->num_consumers(); ++i) {
          if (is_all_gather_op(result->consumer(i))) {
            already_allgather = true;
            all_gather_output = result->consumer(i)->output(0);
            break;
          }
        }

        if (!already_allgather) {
          all_gather_output = MakeAllGatherOp(result, comm_group, gather_dim,
                                              OpMeta().set_is_deduce_states(false)
                                              .set_name(result->name() + "_AllGather"));
          RecordExecTensor(all_gather_output);
          auto& all_gather_op = all_gather_output->producer();
          all_gather_op->set_fw_op_id(result->producer()->fw_op_id());
          all_gather_op->MapToParallelDevices(info.src_group_union);
          all_gather_op->Instantiate(local_device, suggested_comm_stream_idx);
        }
        
        RecordExecTensor(all_gather_output);
        auto& all_gather_op = all_gather_output->producer();
        all_gather_op->set_fw_op_id(result->producer()->fw_op_id());
        all_gather_op->MapToParallelDevices(info.src_group_union);
        all_gather_op->Instantiate(local_device, suggested_comm_stream_idx);
        result = all_gather_output;
        HT_LOG_DEBUG << local_device << ": substitute " << comm_op << " to all_gather_op: " << comm_group;
        determine_flag = true;
        local_comm_flag = true;
      }
      if ((comm_type & REDUCE_SCATTER_OP) != 0) {
        HT_ASSERT(info.src_group == info.dst_group)
          << "wrong src and dst group relationship!";
        DeviceGroup comm_group = comm_op_impl.get_devices_by_dim(comm_op, -2);
        int32_t scatter_dim = info.dst_ds.get_split_dim(info.src_ds);
        Tensor reduce_scatter_output = MakeReduceScatterOp(
          result, comm_group, comm_op_impl.reduction_type(), 
          scatter_dim, false,
          OpMeta().set_is_deduce_states(false)
                  .set_name(result->name() + "_ReduceScatter"));
        RecordExecTensor(reduce_scatter_output);
        auto& reduce_scatter_op = reduce_scatter_output->producer();
        reduce_scatter_op->set_fw_op_id(result->producer()->fw_op_id());
        reduce_scatter_op->MapToParallelDevices(info.src_group_union);
        reduce_scatter_op->Instantiate(local_device, suggested_comm_stream_idx);
        result = reduce_scatter_output;
        HT_LOG_DEBUG << local_device << ": substitute " << comm_op << " to reduce_scatter_op: " << comm_group;
        determine_flag = true;
        local_comm_flag = true;
      }
      if ((comm_type & SPLIT_ALL_REDUCE_OP) != 0) {
        HT_ASSERT(info.src_group_union.check_equal(info.dst_group_union))
          << "wrong src and dst group relationship!";
        // 先前进行了局部通信
        // 对齐了所有的local ds
        DistributedStatesUnion intermediate_ds_union(info.dst_ds_union);
        if (local_comm_flag) {
          intermediate_ds_union.change_hetero_dim(info.src_ds_union.hetero_dim());
          result->set_cur_ds_union(intermediate_ds_union); 
        }
        size_t split_num = 0;
        std::vector<DeviceGroupList> comm_groups_list;
        std::tie(split_num, comm_groups_list) = comm_op_impl.get_split_comm_groups_list(comm_op, info.src_group_union, intermediate_ds_union);
        Tensor split_all_reduce_output = MakeSplitAllReduceOp(
          result, comm_groups_list, split_num, 
          comm_op_impl.reduction_type(), false,
          OpMeta().set_is_deduce_states(false)
                  .set_name(result->name() + "_SplitAllReduce"));
        RecordExecTensor(split_all_reduce_output);
        auto& split_all_reduce_op = split_all_reduce_output->producer();
        split_all_reduce_op->set_fw_op_id(result->producer()->fw_op_id());
        split_all_reduce_op->MapToParallelDevices(info.src_group_union);
        split_all_reduce_op->Instantiate(local_device, suggested_comm_stream_idx);
        result = split_all_reduce_output;
        HT_LOG_DEBUG << local_device << ": substitute " << comm_op << " to split_all_reduce_op: " << comm_groups_list;         
        determine_flag = true;
      }
      if ((comm_type & SPLIT_REDUCE_SCATTER_OP) != 0) {
        HT_ASSERT(info.src_group_union.check_equal(info.dst_group_union))
          << "wrong src and dst group relationship!";
        // 先前进行了局部通信
        // 对齐了所有的local ds
        DistributedStatesUnion intermediate_ds_union(info.dst_ds_union);
        if (local_comm_flag) {
          intermediate_ds_union.change_hetero_dim(info.src_ds_union.hetero_dim());
          result->set_cur_ds_union(intermediate_ds_union); 
        }
        size_t split_num = 0;
        std::vector<DeviceGroupList> comm_groups_list;
        std::tie(split_num, comm_groups_list) = comm_op_impl.get_split_comm_groups_list(comm_op, info.src_group_union, intermediate_ds_union);
        Tensor split_reduce_scatter_output = MakeSplitReduceScatterOp(
          result, comm_groups_list, split_num, 
          comm_op_impl.reduction_type(), false,
          OpMeta().set_is_deduce_states(false)
                  .set_name(result->name() + "_SplitReduceScatter"));
        RecordExecTensor(split_reduce_scatter_output);
        auto& split_reduce_scatter_op = split_reduce_scatter_output->producer();
        split_reduce_scatter_op->set_fw_op_id(result->producer()->fw_op_id());
        split_reduce_scatter_op->MapToParallelDevices(info.src_group_union);
        split_reduce_scatter_op->Instantiate(local_device, suggested_comm_stream_idx);
        result = split_reduce_scatter_output;
        HT_LOG_DEBUG << local_device << ": substitute " << comm_op << " to split_reduce_scatter_op: " << comm_groups_list;         
        determine_flag = true;
      }
      if ((comm_type & SPLIT_ALL_GATHER_OP) != 0) {
        HT_ASSERT(info.src_group_union.check_equal(info.dst_group_union))
          << "wrong src and dst group relationship!";
        HT_ASSERT(!local_comm_flag)
          << "SPLIT_ALL_GATHER_OP shouldn't have a local comm op before";
        size_t split_num = 0;
        std::vector<DeviceGroupList> comm_groups_list;
        std::tie(split_num, comm_groups_list) = comm_op_impl.get_split_comm_groups_list(comm_op, info.src_group_union, info.dst_ds_union);
        Tensor split_all_gather_output = MakeSplitAllGatherOp(
          result, comm_groups_list, split_num, false,
          OpMeta().set_is_deduce_states(false)
                  .set_name(result->name() + "_SplitAllGather"));
        RecordExecTensor(split_all_gather_output);
        auto& split_all_gather_op = split_all_gather_output->producer();
        split_all_gather_op->MapToParallelDevices(info.src_group_union);
        split_all_gather_op->Instantiate(local_device, suggested_comm_stream_idx);
        result = split_all_gather_output;
        HT_LOG_DEBUG << local_device << ": substitute " << comm_op << " to split_all_gather_op: " << comm_groups_list;         
        determine_flag = true;
      }
      if (!determine_flag) {
        HT_RUNTIME_ERROR << local_device << ": " << comm_op << " type is not supported yet"
          << ", src ds union is " << info.src_ds_union.ds_union_info()
          << ", and dst ds union is " << info.dst_ds_union.ds_union_info()
          << ", src group is " << info.src_group_union
          << ", dst group is " << info.dst_group_union;
      }

      // only send, then need to ignore the shape & ds when replacing the input 
      if (result.get() == input.get()) {
        ignore_flag = true;
      }
      // assign distributed states union for result tensor
      if (!ignore_flag) {
        result->set_cur_ds_union(info.dst_ds_union); 
      }
      // find all comm_op->output consumers, and replace the correspond input tensor with result tensor
      for (int i = comm_op->output(0)->num_consumers() - 1; i >= 0; i--) {
        auto& consumer_i = comm_op->output(0)->consumer(i);
        for (int j = 0; j < consumer_i->num_inputs(); j++) {
          if (consumer_i->input(j)->id() == comm_op->output(0)->id()) {
            Graph::ReplaceInput(consumer_i, j, result, ignore_flag);
          }
        }
        for (int j = 0; j < consumer_i->num_in_dep_linkers(); j++) {
          if (consumer_i->in_dep_linker(j)->id() == comm_op->output(0)->id()) {
            Graph::ReplaceInDepLinker(consumer_i, j, result, ignore_flag);
          }
        }
      }
      // old_comm_to_new[comm_op->id()] = result->producer()->id();
      // bool is_comm_recompute = comm_op->op_meta().is_recompute();
      // result->producer()->op_meta().set_is_recompute(is_comm_recompute);
      pop_subgraph_ctx();
      pop_subgraph_op_type_ctx();
      DeleteExecOp(comm_op);
      HT_LOG_DEBUG << local_device << "==============> substitute comm_op end: " << op << "...";
    }
  }
}

DeviceGroup ExecutableGraph::GetPrevStage() {
  auto local_device = hetu::impl::comm::GetLocalDevice();
  HT_ASSERT(_pipeline_map.find(local_device) != _pipeline_map.end())
    << "something wrong, can't figure out which pipeline the local device belongs to";
  auto& pipeline = _pipeline_map[local_device];
  int32_t stage_id = -1;
  for (int i = 0; i < pipeline.size(); i++) {
    if (pipeline[i].contains(local_device)) {
      stage_id = i;
    }
  }
  HT_ASSERT(stage_id != -1)
    << "something wrong, can't figure out which stage the local device belongs to";
  HT_ASSERT(stage_id != 0)
    << "the first stage doesn't have any former stage";
  return pipeline.at(stage_id - 1);
}

DeviceGroup ExecutableGraph::GetNextStage() {
  auto local_device = hetu::impl::comm::GetLocalDevice();
  HT_ASSERT(_pipeline_map.find(local_device) != _pipeline_map.end())
    << "something wrong, can't figure out which pipeline the local device belongs to";
  auto& pipeline = _pipeline_map[local_device];
  int32_t stage_id = -1;
  for (int i = 0; i < pipeline.size(); i++) {
    if (pipeline[i].contains(local_device)) {
      stage_id = i;
    }
  }
  HT_ASSERT(stage_id != -1)
    << "something wrong, can't figure out which stage the local device belongs to";
  HT_ASSERT(stage_id != pipeline.size() - 1)
    << "the last stage doesn't have any next stage";
  return pipeline.at(stage_id + 1);
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
std::unordered_map<size_t, std::vector<std::pair<int32_t, size_t>>>
ExecutableGraph::GeneratePipedreamFlushSchedule(
  size_t num_stages, size_t num_micro_batches, bool is_inference) {
  std::unordered_map<size_t, std::vector<std::pair<int32_t, size_t>>> schedule;
  // inference time: for only forward
  if (is_inference) {
    for (size_t stage_id = 0; stage_id < num_stages; stage_id++) {
      std::vector<std::pair<int32_t, size_t>> tasks;
      tasks.reserve(num_micro_batches);
      for (size_t step_id = 0; step_id < num_micro_batches; step_id++) {
        tasks.push_back({0, step_id});
      }
      schedule[stage_id] = tasks;
    }
    return schedule;
  }
  // traininig time: for forward and backward
  for (size_t stage_id = 0; stage_id < num_stages; stage_id++) {
    std::vector<std::pair<int32_t, size_t>> tasks;
    // Task type:
    // -1 -> bubble
    // 0 -> forward
    // 1 -> backward
    tasks.reserve(2 * num_micro_batches);
    size_t num_warmup_microbatches = std::min(num_micro_batches, num_stages - stage_id - 1);
    size_t num_microbatches_remaining =
      num_micro_batches - num_warmup_microbatches;
    // 1. warmup
    for (size_t step_id = 0; step_id < num_warmup_microbatches; step_id++) {
      tasks.push_back({0, step_id});
    }
    // 2. 1F1B
    for (size_t step_id = 0; step_id < num_microbatches_remaining; step_id++) {
      tasks.push_back({0, num_warmup_microbatches + step_id});
      tasks.push_back({1, step_id});
    }
    if (num_microbatches_remaining == 0) {
      tasks.push_back({-1, num_microbatches_remaining});
    }
    // 3. cooldown
    for (size_t step_id = 0; step_id < num_warmup_microbatches; step_id++) {
      tasks.push_back({1, num_microbatches_remaining + step_id});
    }
    schedule[stage_id] = tasks;
  }
  return schedule;
}

void ExecutableGraph::ComputeFunc(size_t& micro_batch_id, const OpRefList& topo, RuntimeContext& runtime_ctx,
                                  Tensor2NDArrayMap& tensor2data, Tensor2IntMap& tensor2degrees, 
                                  Tensor2NDArrayMap& grad_accumulation, bool grad_accumulation_finished,
                                  const FeedDict& feed_dict, const TensorList& fetches,
                                  const std::unordered_map<TensorId, size_t>& fetch_indices, bool& is_continuous_p2p) {
  const TensorIdSet& dtype_transfer_tensor = _execute_plan.dtype_transfer_tensor;
  const TensorIdSet& shared_weight_tensor = _execute_plan.shared_weight_tensor;
  const OpIdSet& shared_weight_p2p = _execute_plan.shared_weight_p2p;
  const OpIdSet& shared_weight_grad_p2p = _execute_plan.shared_weight_grad_p2p;
  const TensorIdSet& accumulated_tensor = _execute_plan.accumulated_tensor;
  const OpIdSet& accumulated_ops = _execute_plan.accumulated_ops;

  auto is_shared_weight_or_grad_p2p = [&](const Operator& op) -> bool {
    bool is_shared_weight = (shared_weight_p2p.find(op->id()) != shared_weight_p2p.end());
    bool is_shared_weight_grad = (shared_weight_grad_p2p.find(op->id()) != shared_weight_grad_p2p.end());
    return is_shared_weight || is_shared_weight_grad;
  };

  auto local_device = hetu::impl::comm::GetLocalDevice();

  // HT_LOG_DEBUG << local_device << ": computeFunc topo is" << topo;
  int64_t total_count = 0;
  int64_t max_share_memory = 0;
  for (auto& op_ref : topo) {
    auto& op = op_ref.get();

    // HT_LOG_INFO << local_device << ": computeFunc op " << op;
    HT_ASSERT(!is_placeholder_op(op) && !is_variable_op(op))
      << "Placeholder & Variable ops should not appear in ComputeFunc!";
    bool is_feed_dict_op = Operator::all_output_tensors_of(op, [&](Tensor& tensor) {
      return feed_dict.find(tensor->id()) != feed_dict.end();
    });
    
    if (runtime_ctx.has_runtime_skipped(op->id())) {
      continue; 
    }
    if (is_feed_dict_op) {
      continue;
    }
    // just convert fp32 -> bf16, fp16 in micro batch 0
    // though most of it is actually put in runtime_skipped already
    // but some of it (rotary sin or cos, mask...) is not in runtime_skipped
    if (op->num_outputs() > 0 && dtype_transfer_tensor.find(op->output(0)->id()) != dtype_transfer_tensor.end() && micro_batch_id > 0) {
      // HT_RUNTIME_ERROR << "unreachable";
      continue;
    }
    // in pipeline(shared_weight_p2p not empty), shared weight p2p ops only execute in micro batch 0
    if (!shared_weight_p2p.empty() && shared_weight_p2p.find(op->id()) != shared_weight_p2p.end() && micro_batch_id > 0) {
      // HT_LOG_INFO << local_device << ": skip execute shared weight p2p: " << op;
      continue;
    }

    bool enable_async_param = std::any_cast<bool>(runtime_ctx.get_param("get_cpu_states"));
    enable_async_param = enable_async_param && 
                         micro_batch_id == 0 &&
                         is_optimizer_update_op(op);

    if (enable_async_param && is_adam_op(op)) {
      TIK(cpu_offload);
      int alignment = 16;
      NDArray& gpu_param = _preserved_data[op->input(0)->id()];
      NDArray& gpu_mean = _preserved_data[op->input(2)->id()];
      NDArray& gpu_variance = _preserved_data[op->input(3)->id()];
      NDArray& gpu_step = _preserved_data[op->input(4)->id()];
      max_share_memory += DIVUP(gpu_param->numel() * DataType2Size(gpu_param->dtype()), 
                                alignment) * alignment;
      max_share_memory += DIVUP(gpu_mean->numel() * DataType2Size(gpu_mean->dtype()),
                                alignment) * alignment;
      max_share_memory += DIVUP(gpu_variance->numel() * DataType2Size(gpu_variance->dtype()),
                                alignment) * alignment;
      max_share_memory += DIVUP(gpu_step->numel() * DataType2Size(gpu_step->dtype()),
                                alignment) * alignment;
      if (ShareMomoryReadyOfMemoryPool(kCPU)) {
        NDArray cpu_param = GetCPUParam(gpu_param, op->input(0));
        NDArray cpu_mean = GetCPUParam(gpu_mean, op->input(2));
        NDArray cpu_variance = GetCPUParam(gpu_variance, op->input(3));
        NDArray cpu_step = GetCPUParam(gpu_step, op->input(4));

        NDArray::to(gpu_param, kCPU, 
                    gpu_param->dtype(), kOffloadStream, cpu_param);
        NDArray::to(gpu_mean, kCPU, 
                    gpu_mean->dtype(), kOffloadStream, cpu_mean);
        NDArray::to(gpu_variance, kCPU, 
                    gpu_variance->dtype(), kOffloadStream, cpu_variance);
        NDArray::to(gpu_step, kCPU, 
                    gpu_step->dtype(), kComputingStream, cpu_step);
        HT_LOG_DEBUG << op << "ready for offloading, cpu_step" << gpu_step << " " << cpu_step;
        TOK(cpu_offload);
        total_count += COST_MSEC(cpu_offload);
        // HT_LOG_WARN<< "offload time = " << COST_MSEC(cpu_offload) << " ms."; 
      }
    }

    // accumulated_ops now all execute in PostRun()
    // including shared weight grad p2p ops
    if (accumulated_ops.find(op->id()) != accumulated_ops.end()) {
      continue;
    }

    // debug stuck bug use
    // HT_LOG_INFO << local_device << ": micro batch " << micro_batch_id << " op execute " << op << " on stream " << op->stream_index() << " begin...";
    // batched p2p send & recv
    // 跨hetero stage的batchedIsendIrecv已经包了一层ncclGroupStart和ncclGroupEnd
    // 但参考nccl文档可知最终取决于最外层的ncclGroupStart和ncclGroupEnd
    if ((is_pipeline_stage_send_op(op) || is_pipeline_stage_recv_op(op)) 
        && !is_shared_weight_or_grad_p2p(op)) {
      if (!is_continuous_p2p) {
        is_continuous_p2p = true;
        auto event = std::make_unique<hetu::impl::CUDAEvent>(op->placement());
        event->Record(Stream(op->placement(), kComputingStream));
        event->Record(Stream(op->placement(), kSwitchComputingStream));
        event->Block(Stream(op->placement(), kP2PStream));
        _p2p_events.emplace_back(std::move(event));
        ncclGroupStart_safe();
        // HT_LOG_INFO << local_device << ": nccl group start";
      }
    } else if (is_continuous_p2p) {
      is_continuous_p2p = false;
      ncclGroupEnd_safe();
      auto event = std::make_unique<hetu::impl::CUDAEvent>(op->placement());
      event->Record(Stream(op->placement(), kP2PStream));
      event->Block(Stream(op->placement(), kComputingStream));
      event->Block(Stream(op->placement(), kSwitchComputingStream));
      _p2p_events.emplace_back(std::move(event));
      // HT_LOG_INFO << local_device << ": nccl group end";
    }

    // 2025.2.6 Update: parallel attn op内部会自动根据当前graph的CUR_MICRO_BATCH_ID去选择ctx
    /*
    // parallel attn op算子手动实现且比较复杂
    // 目前单独维护attn ctx
    // 这里需要从外部传入micro batch id来确定 fwd存/bwd取 哪个attn ctx
    if (is_parallel_attn_op(op) || is_parallel_attn_grad_op(op)) {
      if (is_parallel_attn_op(op)) {
        dynamic_cast<ParallelAttentionOpImpl&>(op->body()).set_attn_ctx_num(micro_batch_id);
      } else {
        dynamic_cast<ParallelAttentionGradientOpImpl&>(op->body()).set_attn_ctx_num(micro_batch_id);
      }
    }
    */

    // variable can be directly fetched, needn't save in tensor2data
    // AMP data transfer can be directly fetched, needn't save in tensor2data
    NDArrayList input_vals;
    input_vals.reserve(op->num_inputs());
    for (const auto& input : op->inputs()) {
      HT_ASSERT(input.is_defined())
        << op << " has an undefined input, it cannot run";
      NDArray input_val;
      if (_preserved_data.find(input->id()) != _preserved_data.end()) {
        input_val = _preserved_data[input->id()];
        // HT_LOG_INFO << "fetch " << input << " from _preserved_data, sum is " << NDArray::sum(input_val);
        // 如果有一些_preserved_data是switch过来的
        // 那么我们这里进行实际的sync
        auto event_it = _switch_param_events.find(input->id());
        if (event_it != _switch_param_events.end()) {
          event_it->second->Block(op->instantiation_ctx().stream());
        }     
      } 
      // 其余情况从tensor2data中fetch
      else {
        auto it = tensor2data.find(input->id());
        HT_ASSERT(it != tensor2data.end() && it->second.is_defined())
          << "Failed to execute the \"" << op->type() << "\" operation "
          << "(with name \"" << op->name() << "\"): "
          << "Cannot find input " << input;
        auto& data = it->second;
        if (data->device() != input->placement() ||
            data->dtype() != input->dtype()) {
          if (data->device().is_cpu() && input->placement().is_cuda()) {
            tensor2data[input->id()] = NDArray::to(data, input->placement(), input->dtype(),
                                                   kH2DStream);
            auto event = std::make_unique<hetu::impl::CUDAEvent>(input->placement());
            event->Record(Stream(input->placement(), kH2DStream));
            event->Block(op->instantiation_ctx().stream());
          } else {
            // TODO: use another stream for async data transfer
            tensor2data[input->id()] = NDArray::to(data, input->placement(), input->dtype(),
                                                   op->instantiation_ctx().stream_index);
          }
        }
        input_val = tensor2data[input->id()];
        // should free memory until op async compute complete!!!
        // recved shared weight should not be erased in first micro batch. but can be multi copied and erased in later micro batches
        if ((--tensor2degrees[input->id()]) == 0 
            && fetch_indices.find(input->id()) == fetch_indices.end() 
            && ((micro_batch_id == 0 && shared_weight_tensor.find(input->id()) == shared_weight_tensor.end() 
                 && dtype_transfer_tensor.find(input->id()) == dtype_transfer_tensor.end())
                || micro_batch_id > 0)) {
          tensor2data.erase(input->id());
        }
      }
      input_vals.push_back(input_val);
    }
    if (is_shared_weight_or_grad_p2p(op)) {
      auto event = std::make_unique<hetu::impl::CUDAEvent>(op->placement());
      event->Record(Stream(op->placement(), kComputingStream));
      event->Block(Stream(op->placement(), kP2PStream));
      // HT_LOG_INFO << local_device << ": wte nccl group start";
      ncclGroupStart_safe();
    }

    // **** 调用op计算 ****
    NDArrayList output_vals;
    try {
      output_vals = op->Compute(input_vals, runtime_ctx, micro_batch_id);
    } catch (const std::exception& e) {
      HT_RUNTIME_ERROR << "During computing exec op " << op << " of micro batch " << micro_batch_id
        << " with inputs " << op->inputs() << ", an error occurs: " << e.what();
    }
    // checkOutputsMemory(op, micro_batch_id, input_vals, output_vals);

    if (is_shared_weight_or_grad_p2p(op)) {
      // HT_LOG_INFO << local_device << ": wte nccl group end";
      ncclGroupEnd_safe();
    }
    // HT_LOG_INFO << "micro batch[" << micro_batch_id << "] Running op " << op << " (type: " << op->type() << ") mid...";    
    // Note: The usage should be marked inside kernels, 
    // but we still mark here in case we forget to do so in some kernels. 
    NDArray::MarkUsedBy(input_vals, op->instantiation_ctx().stream());
    NDArray::MarkUsedBy(output_vals, op->instantiation_ctx().stream());
    bool use_fp32_grad_accumulation = std::any_cast<bool>(runtime_ctx.get_param("fp32_grad_accumulation"));
    // HT_LOG_INFO << local_device << ": op execute " << op;
    for (size_t i = 0; i < op->num_outputs(); i++) {
      const auto& output = op->output(i);
      if (accumulated_tensor.find(output->id()) != accumulated_tensor.end()) {
        if (grad_accumulation.find(output->id()) == grad_accumulation.end()) {
          // Using FP32 for Grad Accumulation
          grad_accumulation[output->id()] = NDArray::zeros(output_vals[i]->shape(),
                                                           output_vals[i]->device(),
                                                           output_vals[i]->dtype());
        } 
        NDArray::add(grad_accumulation[output->id()], output_vals[i], 
                     op->instantiation_ctx().stream_index, 
                     grad_accumulation[output->id()]);  
        if (grad_accumulation_finished) {
          tensor2data[output->id()] = grad_accumulation[output->id()];
        }
      } else if (fetch_indices.find(output->id()) != fetch_indices.end()) {
        tensor2data[output->id()] = NDArray::zeros(output_vals[i]->shape(),
                                                   output_vals[i]->device(),
                                                   output_vals[i]->dtype());
        NDArray::add(tensor2data[output->id()], output_vals[i], op->instantiation_ctx().stream_index, tensor2data[output->id()]);    
      } else if (tensor2degrees[output->id()] > 0) {
        tensor2data[output->id()] = output_vals[i];
      } 
    }
    // debug stuck bug use
    // op->instantiation_ctx().stream().Sync();
    // HT_LOG_INFO << local_device << ": micro batch " << micro_batch_id << " op execute " << op << " end...";

    // 提前执行PostRun中的grad reduce
    // workaround: 这部分逻辑is out of topo sort
    for (auto& output : op->outputs()) {
      auto grad_reduce_subgraph_it = _grad_reduce_subgraph_map.find(output->id());
      if (grad_accumulation_finished && _overlap_grad_reduce && grad_reduce_subgraph_it != _grad_reduce_subgraph_map.end()) {
        // HT_LOG_INFO << "execute grad reduce for " << output << " begin...";
        grad_reduce_subgraph_it->second->run(tensor2data, _preserved_data, runtime_ctx, micro_batch_id, SubGraphOpType::UPDATE, false,
          [this](Operator& op, Tensor2NDArrayMap& tensor2data, size_t micro_batch_id) {
            OpHandlerStatus status; 
            if (!is_grad_reduce_op(op)) {
              status.need_skip = true;
            }
            return status; 
          }
        );
        // HT_LOG_INFO << "execute grad reduce for " << op << " end...";
      }
    }
  }
  if (micro_batch_id == 0)
    HT_LOG_DEBUG << "micro batch id = " << micro_batch_id 
                <<", offload time = " << total_count << " ms.";
  if (max_share_memory > 0) {
      HT_LOG_WARN << "max_share_memory should be allocated:" << max_share_memory << "bytes.";
      AllocShareMemoryFromMemoryPool(kCPU, max_share_memory,
                                     Stream(kCPU, kBlockingStream));
  }
}

void ExecutableGraph::GetExecEnvs() {
  char* env = std::getenv("HETU_P2P");
  if (env != nullptr) {
    if (std::string(env) == "SINGLE_COMMUNICATOR") {
      _p2p_single_communicator = true;
      // 这里直接提前把communicator给create好
      // HT_LOG_INFO << "get or create single communicator for P2P ops";
      hetu::impl::comm::NCCLCommunicationGroup::GetOrCreate(_used_ranks, Stream(hetu::impl::comm::GetLocalDevice(), kP2PStream));
    } else {
      HT_RUNTIME_ERROR << "Unknown hetu p2p setting: " + std::string(env);
    }
  } else {
    // 默认不使用single communicator
    _p2p_single_communicator = false;
  }

  env = std::getenv("HETU_BRIDGE");
  if (env != nullptr) {
    if (std::string(env) == "SINGLE_COMMUNICATOR") {
      _bridge_single_communicator = true;
      // 这里直接提前把communicator给create好
      // HT_LOG_INFO << "get or create single communicator for BRIDGE ops";
      hetu::impl::comm::NCCLCommunicationGroup::GetOrCreate(_used_ranks, Stream(hetu::impl::comm::GetLocalDevice(), kBridgeStream));
    } else {
      HT_RUNTIME_ERROR << "Unknown hetu p2p setting: " + std::string(env);
    }
  } else {
    // 默认不使用single communicator
    _bridge_single_communicator = false;
  }

  env = std::getenv("HETU_OVERLAP_GRAD_REDUCE");
  _overlap_grad_reduce = false;
  if (env != nullptr) {
    if (std::string(env) == "FIRST_STAGE") {
      auto local_device = hetu::impl::comm::GetLocalDevice();
      if (_pipeline_map.find(local_device) != _pipeline_map.end()) {
        HT_ASSERT(_pipeline_map[local_device].size() >= 1)
          << "pipeline should have at least one stage";
        if (_pipeline_map[local_device].front().contains(local_device)) {
          // 只有第一个stage去overlap
          _overlap_grad_reduce = true;
        }
      } 
    } else if (std::string(env) == "ALL_STAGES") {
      _overlap_grad_reduce = true;
    } else {
      HT_RUNTIME_ERROR << "Unknown hetu dp overlap setting: " + std::string(env);
    }
  }

  env = std::getenv("HETU_SHAPE_MISMATCH");
  if (env != nullptr) {
    if (std::string(env) == "NO_MISMATCH") {
      _shape_mismatch_flag = 0;
    } else if (std::string(env) == "BRIDGE_SUBGRAPH") {
      _shape_mismatch_flag = 1;
    } else {
      HT_RUNTIME_ERROR << "Unknown hetu shape mismatch setting: " + std::string(env);
    }
  } else {
    // 默认没有shape mismatch
    _shape_mismatch_flag = 0;
  }

  env = std::getenv("HETU_STRAGGLER");
  if (env != nullptr) {
    if (std::string(env) == "ANALYSIS") {
      _straggler_flag = 1;
    } else if (std::string(env) == "EXP") {
      // 每个GPU都会profile
      _straggler_flag = 2;
    } else if (std::string(env) == "EXP_NEW") {
      // 只在0号GPU上profile
      // 并且信息更全
      // 还包含memory信息
      _straggler_flag = 3;
    } else {
      HT_RUNTIME_ERROR << "Unknown hetu straggler level: " + std::string(env);
    }
  } else {
    // 默认不分析straggler
    _straggler_flag = 0;
  }

  env = std::getenv("HETU_STRAGGLER_LOG_FILE");
  if (env != nullptr) {
    _straggler_log_file_path = std::string(env);
  } else {
    // 默认不对straggler打log
    _straggler_log_file_path = "";
  }

  env = std::getenv("HETU_MEMORY_PROFILE");
  if (env != nullptr) {
    if (std::string(env) == "MICRO_BATCH") {
      _memory_profile_level = MEMORY_PROFILE_LEVEL::MICRO_BATCH;
      _all_micro_batches_memory_info.clear();
    } else if (std::string(env) == "INFO") {
      _memory_profile_level = MEMORY_PROFILE_LEVEL::INFO;
    } else if (std::string(env) == "WARN") {
      _memory_profile_level = MEMORY_PROFILE_LEVEL::WARN;
    } else {
      HT_RUNTIME_ERROR << "Unknown hetu memory profile level: " + std::string(env);
    }
  } else {
    // 默认不profile
    _memory_profile_level = MEMORY_PROFILE_LEVEL::WARN;
  }

  env = std::getenv("HETU_MEMORY_LOG_FILE");
  if (env != nullptr) {
    _memory_log_file_path = std::string(env);
  } else {
    // 默认不对memory打log
    _memory_log_file_path = "";
  }

  env = std::getenv("HETU_PARALLEL_ATTN");
  if (env != nullptr) {
    if (std::string(env) == "ANALYSIS") {
      _parallel_attn_flag = 1;
    } else if (std::string(env) == "EXP") {
      _parallel_attn_flag = 2;
    } else {
      HT_RUNTIME_ERROR << "Unknown hetu parallel attn level: " + std::string(env);
    }
  } else {
    // 默认不分析parallel attn
    _parallel_attn_flag = 0;
  }

  env = std::getenv("HETU_PARALLEL_ATTN_LOG_FILE");
  if (env != nullptr) {
    _parallel_attn_log_file_path = std::string(env);
  } else {
    // 默认不对parallel attn打log
    _parallel_attn_log_file_path = "";
  }

  env = std::getenv("HETU_EVENT_TIMING");
  if (env != nullptr) {
    if (std::string(env) == "ON") {
      EVENT_TIMING = true;
    } else if (std::string(env) == "OFF") {
      EVENT_TIMING = false;
    } else {
      HT_RUNTIME_ERROR << "Unknown hetu event timing level: " + std::string(env);
    }
  }
}

// 每次run都会经过的核心部分
// 我们将这一部分单独提取出来做成一个函数来增加代码的可读性
NDArrayList ExecutableGraph::CrucialRun(const TensorList& fetches, 
                                        const FeedDict& feed_dict, 
                                        const IntSymbolDict& int_symbol_dict,
                                        const int num_micro_batches,
                                        const RuntimeContext& global_ctx) {
  auto local_device = hetu::impl::comm::GetLocalDevice();
  // calculate params
  bool is_calculate_params = false;
  if (is_calculate_params) {
    int64_t params_size = 0;
    for (auto& op : _execute_plan.local_topo) {
      if (is_variable_op(op)) {
        params_size += op.get()->output(0)->numel();
        // HT_LOG_INFO << local_device << ": variable op " << op << ", shape = " << op.get()->output(0)->shape();
      }
    }
    HT_LOG_INFO << local_device << ": params_size = " << params_size;
  }

  HT_LOG_DEBUG << local_device << ": 0. Create Execution Plan [end]";

  // ********************** Run Level Check Point **********************
  if (_run_level == RunLevel::TOPO) {
    return {};
  }
  // ********************** Run Level Check Point **********************

  HT_LOG_DEBUG << local_device << ": 1. pipeline init[begin]";
  // runtime ctx for m micro batches
  std::vector<RuntimeContext> runtime_ctx_list(num_micro_batches);
  // tensor data for m micro batches
  std::vector<Tensor2NDArrayMap> tensor2data_list(num_micro_batches);
  // tensor degrees for m micro batches, if degree=0 && not in fetches, free memory for this tensor
  std::vector<Tensor2IntMap> tensor2degrees_list(num_micro_batches);
  // flush update once for m micro batches
  Tensor2NDArrayMap grad_accumulation;

  for (int i = 0; i < num_micro_batches; i++) {
    runtime_ctx_list[i] = RuntimeContext(_execute_plan.local_topo.size(), _shape_plan_pool.at(_active_shape_plan_list[i]));
    runtime_ctx_list[i].copy_param_dict(global_ctx);
  } 

  // placeholder ops: get feed in dict & split into m micro batches
  for (const auto& kv : feed_dict) {
    if (kv.second.size() == 0) // only feed placeholder_op in local device group
      continue;
    if (kv.second.size() == 1) {
      auto micro_batches = NDArray::split(kv.second[0], num_micro_batches);
      for (int i = 0; i < num_micro_batches; i++) {
        tensor2data_list[i][kv.first] = micro_batches[i];
      }
    } else {
      HT_ASSERT(kv.second.size() == num_micro_batches);
      for (int i = 0; i < num_micro_batches; i++) {
        tensor2data_list[i][kv.first] = kv.second[i];
      }
    }
  }

  std::unordered_map<TensorId, size_t> fetch_indices;
  for (size_t i = 0; i < fetches.size(); i++)
    fetch_indices[fetches.at(i)->id()] = i;
  // get consume times for each tensor
  Tensor2IntMap tensor2degrees;
  for (auto& op_ref : _execute_plan.local_topo) {
    for (auto& input : op_ref.get()->inputs()) {
      tensor2degrees[input->id()]++;
    }
  }
  for (int i = 0; i < num_micro_batches; i++) {
    tensor2degrees_list[i] = tensor2degrees;
  }

  if (_pipeline_map.find(local_device) == _pipeline_map.end()) {
    HT_LOG_WARN << local_device << ": can't figure out which pipeline the local device belongs to"
      << ", so we just return";
    return {};
  }
  auto& pipeline = _pipeline_map[local_device];
  int num_stages = pipeline.size();
  bool is_inference = (_execute_plan.local_bw_topo.size() == 0);
  HT_LOG_DEBUG << local_device << ": num_stages = " << num_stages << ", stages = " << pipeline 
    << ", num_micro_batches = " << num_micro_batches << ", is_inference = " << is_inference;
  // get task schedule table for pipedream-flush, also suitable for non-pipeline cases
  auto schedule = GeneratePipedreamFlushSchedule(
    num_stages, num_micro_batches, is_inference);
  // get task schedule table for gpipe    
  // auto schedule = generate_gpipe_schedule(num_stages, num_micro_batches);
  // get tasks for current stage
  // int stage_id = local_device.index() / _stages.at(0).num_devices();
  int stage_id = -1;
  for (int i = 0; i < pipeline.size(); i++) {
    if (pipeline[i].contains(local_device)) {
      stage_id = i;
    }
  }
  // HT_LOG_DEBUG << local_device << ": stages = " << _stages << "; stage id = " << stage_id;
  auto& tasks = schedule[stage_id];
  // NOTE: revert memory plan for now and may be used in the future
  HT_LOG_DEBUG << local_device << ": stage id = " << stage_id;
  HT_LOG_DEBUG << local_device << ": 1. pipeline init[end]";

  HT_LOG_DEBUG << local_device << ": 2. alloc and compute buffer[begin]";
  // alloc origin/transfer params and pre-compute, alloc grads
  PreRun(runtime_ctx_list);
  HT_LOG_DEBUG << local_device << ": 2. alloc and compute buffer[end]";

  // ********************** Run Level Check Point **********************
  if (_run_level == RunLevel::ALLOC) {
    SynchronizeAllStreams();
    // memory debug use
    // hetu::impl::comm::EmptyNCCLCache();
    if (_memory_profile_level == MEMORY_PROFILE_LEVEL::INFO)
      GetCUDAProfiler(local_device)->PrintCurrMemoryInfo(name() + " run ALLOC end");
    return {};
  }
  // ********************** Run Level Check Point **********************

  /*
  HT_LOG_DEBUG << local_device << ": 2-plus. memory plan[begin]";
  // TODO: cache memory plan
  size_t memory_size = 0;
  auto memory_plan = GenerateMemoryPlan(memory_size, tasks, tensor2degrees_list, feed_dict);
  auto memory_space = NDArray::empty({memory_size}, local_device, kInt64, kComputingStream);
  HT_LOG_DEBUG << "Memory plan is generated and allocated";
  if (_memory_profile_level == MEMORY_PROFILE_LEVEL::INFO)
    GetCUDAProfiler(local_device)->PrintCurrMemoryInfo(name() + " alloc memory according to plan end");
  for (auto& op_ref : _execute_plan.local_topo) {
    auto& op = op_ref.get();
    for (auto& tensor : op->outputs()) {
      for (size_t micro_batch_id = 0; micro_batch_id < num_micro_batches; micro_batch_id++) {
        auto it = memory_plan.find({micro_batch_id, tensor->id()});
        if (it == memory_plan.end()) {
          break;
        }
        auto begin_pos = it->second.first;
        auto block_size = it->second.second;
        auto raw_memory = NDArray::slice(memory_space, {begin_pos}, {block_size});
        auto memory = NDArray(NDArrayMeta()
                              .set_shape(GetTensorShape(tensor))
                              .set_dtype(tensor->dtype())
                              .set_device(tensor->producer()->instantiation_ctx().placement), 
                              raw_memory->storage(), raw_memory->storage_offset() * DataType2Size(kInt64) / DataType2Size(tensor->dtype()));
        runtime_ctx_list.at(micro_batch_id).add_runtime_allocation(tensor->id(), memory);
      }
    }
  }
  HT_LOG_DEBUG << local_device << ": 2-plus. memory plan[end]";
  */
  
  HT_LOG_DEBUG << local_device << ": 3. compute[begin]";
  bool is_continuous_p2p = false;
  for (size_t i = 0; i < tasks.size(); i++) {
    auto& task = tasks[i];
    int32_t task_type = task.first;
    // bubble
    if (task_type == -1) {
      ncclGroupEnd_safe();
      ncclGroupStart_safe();
      continue;
    }
    bool is_forward = (task_type == 0);
    size_t& micro_batch_id = task.second;
    auto& tensor2data = tensor2data_list[micro_batch_id];
    auto& tensor2degrees = tensor2degrees_list[micro_batch_id];
    auto& runtime_ctx = runtime_ctx_list[micro_batch_id];
    // set micro batch ctx
    // int_symbol_dict now consists of seqlens needed in parallel attn op
    SetMicroBatchCtx(micro_batch_id, int_symbol_dict);
    // set arithmetic shape
    SetShapePlan(_active_shape_plan_list[micro_batch_id]);
    // set symbolic shape
    // extra shape deduction in UpdateExecShapePlan() may need it
    for (auto& tensor: _leaf_symbolic_tensor_list) {
      if (HasTensorShape(tensor)) {
        tensor->set_symbolic_shape(GetTensorShape(tensor));
      }
    }
    // some tensor (inserted just now) may need to infer shape again
    UpdateExecShapePlan(runtime_ctx);
    // set symbolic shape again for safety
    for (auto& tensor: _leaf_symbolic_tensor_list) {
      tensor->set_symbolic_shape(GetTensorShape(tensor));
    }
    // micro batch i>0 reuse: 
    // 0. shared weight which was recved in micro batch 0
    // 1. f32 -> fp16, bf16 weight which was transfered in micro batch 0
    if (micro_batch_id > 0) {
      if (!_execute_plan.shared_weight_tensor.empty()) {
        for (auto& shared_weight_id : _execute_plan.shared_weight_tensor) {
          if (tensor2data.find(shared_weight_id) != tensor2data.end()) break; // avoid assign twice by fw, bw
          tensor2data[shared_weight_id] = tensor2data_list[0][shared_weight_id];
        }
      }
      if (!_execute_plan.dtype_transfer_tensor.empty()) {
        for (auto& dtype_transfer_id : _execute_plan.dtype_transfer_tensor) {
          if (tensor2data.find(dtype_transfer_id) != tensor2data.end()) break; // avoid assign twice by fw, bw
          tensor2data[dtype_transfer_id] = tensor2data_list[0][dtype_transfer_id];
        }
      }
    }
    if (is_forward) {
      HT_LOG_DEBUG << local_device << ": [micro batch " << micro_batch_id << ": forward begin]";
    } else {
      HT_LOG_DEBUG << local_device << ": [micro batch " << micro_batch_id << ": backward begin]";
    }
    // micro batch i: profile memory begin
    if (_memory_profile_level == MEMORY_PROFILE_LEVEL::MICRO_BATCH) {
      auto micro_batch_memory_info = std::make_shared<MicroBatchMemoryInfo>();
      micro_batch_memory_info->is_forward = is_forward;
      micro_batch_memory_info->stage_id = stage_id;
      micro_batch_memory_info->micro_batch_id = micro_batch_id;
      micro_batch_memory_info->begin_memory_info = GetCUDAProfiler(local_device)->GetCurrMemoryInfo();
      _all_micro_batches_memory_info.emplace_back(micro_batch_memory_info);
    }
    // micro batch i: execute fw/bw
    if (is_forward) {
      // HT_LOG_INFO << "fw topo: " << _execute_plan.local_fw_topo;
      ComputeFunc(micro_batch_id, _execute_plan.local_fw_topo, runtime_ctx,
                  tensor2data, tensor2degrees, grad_accumulation, false, 
                  feed_dict, fetches, fetch_indices, is_continuous_p2p);
    } else {
      bool grad_accumulation_finished = (i == tasks.size() - 1);
      // HT_LOG_INFO << "bw topo: " << _execute_plan.local_bw_topo;
      ComputeFunc(micro_batch_id, _execute_plan.local_bw_topo, runtime_ctx, 
                  tensor2data, tensor2degrees, grad_accumulation, grad_accumulation_finished, 
                  feed_dict, fetches, fetch_indices, is_continuous_p2p);
    }
    // micro batch i: profile memory end
    if (_memory_profile_level == MEMORY_PROFILE_LEVEL::MICRO_BATCH) {
      _all_micro_batches_memory_info.back()->end_memory_info = GetCUDAProfiler(local_device)->GetCurrMemoryInfo();
      // HT_LOG_INFO << *_all_micro_batches_memory_info.back();
    }
    if (is_forward) {
      HT_LOG_DEBUG << local_device << ": [micro batch " << micro_batch_id << ": forward end]";
    } else {
      HT_LOG_DEBUG << local_device << ": [micro batch " << micro_batch_id << ": backward end]";
    }
  }
  if (is_continuous_p2p) {
    ncclGroupEnd_safe();
    auto event = std::make_unique<hetu::impl::CUDAEvent>(local_device);
    event->Record(Stream(local_device, kP2PStream));
    event->Block(Stream(local_device, kComputingStream));
    // event->Block(Stream(local_device, kOptimizerStream));
    _p2p_events.emplace_back(std::move(event));
  }
  HT_LOG_DEBUG << local_device << ": 3. compute[end]";

  // ********************** Run Level Check Point **********************
  // 仅仅是进行了local的计算而不涉及任何grad的reduce
  if (_run_level == RunLevel::COMPUTE_ONLY) {
    SynchronizeAllStreams();
    if (_memory_profile_level == MEMORY_PROFILE_LEVEL::INFO)
      GetCUDAProfiler(local_device)->PrintCurrMemoryInfo(name() + " run COMPUTE_ONLY end");
    return {};
  }
  // ********************** Run Level Check Point **********************

  HT_LOG_DEBUG << local_device << ": 4. reduce grad and update[begin]";
  PostRun(runtime_ctx_list, tensor2data_list[num_micro_batches - 1]);
  HT_LOG_DEBUG << local_device << ": 4. reduce grad and update[end]";

  // ********************** Run Level Check Point **********************
  // 仅仅是算出grad但不更新
  // 这里需要先对grad op进行sync
  if (_run_level == RunLevel::GRAD) {
    // 理论上这里我们可以不让_run_grad_events同步
    // 假如之后切换到别的exec graph的话再在切换grad的时候再进行同步
    for (const auto& event_it : _run_grad_events) {
      event_it.second->Sync();
    }
    _run_grad_events.clear();
    // Question: 可能单独设计接口指定dst exec graph去切换能更快更省显存
    // 即，当前current_grad_buffer切换到dst exec graph后再加到accumulate_grad_buffer上
    // 但dp8逐渐切换到tp8的例子里，逐一切换和直接切换到dst并无明显区别
    // 因此目前grad也用两两间的热切换来弄
    if (_use_current_grad_buffer) {
      // 在define graph中自动切换accumulate_grad_buffer
      // 然后将当前的current_grad_buffer加到当前的accumulate_grad_buffer后清空即可
      for (auto it = _current_grad_buffer_map.begin();
           it != _current_grad_buffer_map.end(); ++it) {
        if (!it->second->IsEmpty() && !_accumulate_grad_buffer_map[it->first]->IsEmpty()) {
          DataType dtype = it->first;
          if (!_accumulate_grad_buffer_map[dtype]->IsAllocated()) {
            // 说明是第一次算grad，之前没有累积grad
            // 直接bind即可
            _accumulate_grad_buffer_map[dtype]->Bind(_current_grad_buffer_map[dtype]->AsStorage());
          } else {
            // 用kBlockingStream集中对整个buffer进行一次add
            // 相比于算出来某一个grad后进行局部的async的add
            // 虽然并发程度降低，但是写法上会简单许多
            auto current_grad_buffer_data = _current_grad_buffer_map[dtype]->AsNDArray();
            auto accumulate_grad_buffer_data = _accumulate_grad_buffer_map[dtype]->AsNDArray();
            if (_grad_scale != 1) {
              NDArray::mul(current_grad_buffer_data,
                           _grad_scale,
                           kBlockingStream,
                           current_grad_buffer_data);
            }
            // 如果有一些累计梯度是switch过来的
            // 那么我们这里进行实际的sync
            for(const auto& event_it : _switch_grad_events) {
              event_it.second->Sync();
            } 
            // 当前的计算的梯度也需要sync
            for(const auto& event_it : _run_grad_events) {
              event_it.second->Sync();
            } 
            NDArray::add(current_grad_buffer_data, 
                         accumulate_grad_buffer_data, 
                         kBlockingStream,
                         accumulate_grad_buffer_data);
          }          
        }
      }
    } 
    // 为节省显存峰值，可以不使用current_grad_buffer
    else {
      // 什么都不用操作
      // 已经在ComputeFunc中将grad加到了accumulate_grad_buffer中
    }
    _p2p_events.clear();
    if (_memory_profile_level == MEMORY_PROFILE_LEVEL::INFO)
      GetCUDAProfiler(local_device)->PrintCurrMemoryInfo(name() + " run GRAD end");
    return {};
  }
  // 说明是RunLevel::UPDATE了
  // 提前进行一些固有map的清空（sync结果前）
  // 这样CPU和GPU可以异步进行
  _run_grad_events.clear();
  bool transfer_not_empty = false;
  for (auto it = _transfer_param_buffer_map.begin();
       it != _transfer_param_buffer_map.end(); ++it) {
    if (!it->second->IsEmpty()) {
      HT_ASSERT(it->second->IsAllocated()) 
        << "transfer param buffer should be allocated";
      transfer_not_empty = true;
    }
  }
  if (transfer_not_empty) {
    for (const auto& [param_id, cur_subgraph] : _optimize_compute_bridge_subgraph_sorted) {
      auto& param_op = _op_indexing[param_id];
      auto it = _transfer_map.find(param_op->output(0)->id());
      HT_ASSERT(it != _transfer_map.end())
        << "The transfer map does not consist of " << param_op->output(0) << " " << param_op->output(0)->dtype();
      auto& transfer_param = it->second;
      // HT_LOG_INFO << "try to remove " << transfer_param << " from preserved data";
      if (transfer_param->placement() == local_device) {
        auto data_it = _preserved_data.find(transfer_param->id());
        HT_ASSERT(data_it != _preserved_data.end())
          << "The preserved data does not consist of " << transfer_param;
        _preserved_data.erase(data_it);
      }
    }
    // _transfer_param_buffer->Free();
  }
  // ********************** Run Level Check Point **********************

  HT_LOG_DEBUG << local_device << ": 5. get results[begin]";
  NDArrayList results(fetches.size(), NDArray());
  std::unordered_set<OpId> to_sync_op_ids;
  to_sync_op_ids.reserve(fetches.size());
  for (auto& op_ref : _execute_plan.local_topo) {
    auto& op = op_ref.get();
    Operator::for_each_output_tensor(op, [&](const Tensor& output) {
      auto it = fetch_indices.find(output->id());
      if (it != fetch_indices.end()) {
        if (output->output_id() >= 0) {
          if (is_variable_op(op) || _execute_plan.accumulated_ops.find(op) != _execute_plan.accumulated_ops.end() 
              || _execute_plan.accumulated_tensor.find(output->id()) != _execute_plan.accumulated_tensor.end()) {
            results[it->second] = tensor2data_list[num_micro_batches - 1][output->id()];
          } else if (is_placeholder_op(op)) {
            auto feed_it = feed_dict.find(output->id());
            if (feed_it != feed_dict.end()) {
              results[it->second] = feed_it->second[num_micro_batches - 1];
            }
          } else {
            NDArrayList result;
            result.reserve(num_micro_batches);
            for (auto& tensor2data : tensor2data_list) {
              auto it = tensor2data.find(output->id());
              HT_ASSERT (it != tensor2data.end()) 
                << "Something wrong! Can't find the data to fetch.";
              result.push_back(tensor2data[output->id()]);
            }
            // HT_LOG_INFO << "concat results of " << output << " fot all micro-batches: " << result;
            results[it->second] = NDArray::cat(result);
          }
        }
        to_sync_op_ids.insert(op->id());
      }
    });
  }
  // SynchronizeAllStreams(local_device);
  // OpList sync_ops;
  for (auto op_id : to_sync_op_ids) {
    _op_indexing[op_id]->Sync(num_micro_batches - 1);
    // sync_ops.push_back(_op_indexing[op_id]);
  }
  
  // HT_LOG_DEBUG << local_device << ": sync ops = " << sync_ops;
  for (size_t i = 0; i < results.size(); i++)
    HT_LOG_TRACE << "fetch " << fetches.at(i) << " (" << i << "-th result): " << results[i];
  HT_LOG_DEBUG << local_device << ": 5. get results[end]";

  // ********************** Run Level Check Point **********************
  // 一次完整的optimizer update发生了
  // transfer param buffer如果存在需要被清理掉
  // origin param buffer不能被清理掉
  // accumulate grad buffer如果存在需要被清理掉
  // current grad buffer需要被清理掉
  // 2024.3.3 update
  // 考虑到单策略alloc和free具有一定耗时
  // 因此把transfer param buffer和current grad buffer的清理放在需要热切换的时候
  if (_run_level == RunLevel::UPDATE) {
    for (auto it = _accumulate_grad_buffer_map.begin();
         it != _accumulate_grad_buffer_map.end(); ++it) {
      if (it->second->IsAllocated()) {
        // 已经对fetches sync过了
        // 这里直接free即可
        it->second->Free();
      }
    }
    if (_use_current_grad_buffer) {
      for (auto it = _current_grad_buffer_map.begin();
           it != _current_grad_buffer_map.end(); ++it) {
        HT_ASSERT(it->second->IsAllocated())
        << "current grad buffer should be allocated in RunLevel::UPDATE";
      }
      // _current_grad_buffer->Free();
    }
    if (_memory_profile_level == MEMORY_PROFILE_LEVEL::INFO)
      GetCUDAProfiler(local_device)->PrintCurrMemoryInfo(name() + " run UPDATE end");
    return results;
  }
  // ********************** Run Level Check Point **********************
}

NDArrayList ExecutableGraph::Run(const Tensor& loss, const TensorList& fetches, 
                                 const FeedDict& feed_dict, const IntSymbolDict& int_symbol_dict, 
                                 const int num_micro_batches,
                                 RunLevel run_level, const double grad_scale,
                                 const RuntimeContext& ctx) {
  
  GetExecEnvs();
  TIK(prepare_run);
  _run_level = run_level;
  _grad_scale = grad_scale;
  auto& local_device = hetu::impl::comm::GetLocalDevice();
  HT_LOG_DEBUG << local_device << ": exec graph run begin .............";
  if (_memory_profile_level == MEMORY_PROFILE_LEVEL::INFO)
    GetCUDAProfiler(local_device)->PrintCurrMemoryInfo(name() + " run begin");

  // TODO: For each pair of `fetches` and `feed_dict`,
  // deduce the optimal execution plan, and cache it.
  _num_micro_batches = num_micro_batches;
  HT_LOG_DEBUG << local_device << ": 0. Create Execution Plan [begin]";
  auto is_op_computed = [&](const Operator& op) -> bool {
    return Operator::all_output_tensors_of(op, [&](const Tensor& tensor) {
      return feed_dict.find(tensor->id()) != feed_dict.end();
    });
  };

  bool is_execute_plan_changed = false;
  for (auto& fetch : fetches) {
    HT_ASSERT(fetch->has_placement_group())
      << "fetch " << fetch << " must have placement group";
    // HT_LOG_INFO << fetch << " placement group union is " << fetch->placement_group_union();
    if (fetch->placement_group_union().has(local_device) && fetch->placement().is_undetermined()) {
      /*
      // topo
      OpRefList topo_before_instantiate = Graph::TopoSort(fetches, num_ops(), is_op_computed);
      HT_LOG_DEBUG << local_device << ": global topo before instantiate: " << topo_before_instantiate;
      */
     
      // instantiate ops
      HT_LOG_DEBUG << local_device << ": [Execution Plan] Instantiate begin...";
      Instantiate(fetches, local_device);
      HT_LOG_DEBUG << local_device << ": [Execution Plan] Instantiate end...";

      // init instantiated topo
      OpRefList topo_before_recompute = Graph::TopoSort(fetches, num_ops(), is_op_computed);
      HT_LOG_DEBUG << local_device << ": global topo before recompute pass: " << topo_before_recompute;

      // add recompute pass
      HT_LOG_DEBUG << local_device << ": [Execution Plan] recompute pass begin...";
      Graph::push_graph_ctx(id());
      Recompute::InsertRecomputedOps(topo_before_recompute);
      Graph::pop_graph_ctx();
      HT_LOG_DEBUG << local_device << ": [Execution Plan] recompute pass end...";

      // init topo with recomputed ops
      OpRefList topo_before_activation_offload = Graph::TopoSort(fetches, num_ops(), is_op_computed);
      HT_LOG_DEBUG << local_device << ": global topo before activation offload pass: " << topo_before_activation_offload;

      // insert activation offload ops
      // TODO: need code review, offload may have bugs
      HT_LOG_DEBUG << local_device << ": [Execution Plan] activation offload pass begin...";
      Graph::push_graph_ctx(id());
      ActivationCPUOffload::OffloadToCPU(topo_before_activation_offload);
      Graph::pop_graph_ctx();
      HT_LOG_INFO << local_device << ": [Execution Plan] activation offload pass end...";

      // init topo contains comm_op
      OpRefList topo_before_substitute_comm = Graph::TopoSort(fetches, num_ops(), is_op_computed);
      HT_LOG_DEBUG << local_device << ": global topo before substitute comm_op: " << topo_before_substitute_comm;

      // substitute comm_op
      HT_LOG_DEBUG << local_device << ": [Execution Plan] substitute comm_op begin...";
      Graph::push_graph_ctx(id()); // ensure the new ops created in execute_graph
      SubstituteCommOp(topo_before_substitute_comm);
      Graph::pop_graph_ctx();
      HT_LOG_DEBUG << local_device << ": [Execution Plan] substitute comm_op end...";

      // update topo with substituted comm_ops
      OpRefList topo_before_contiguous = Graph::TopoSort(fetches, num_ops(), is_op_computed);
      HT_LOG_DEBUG << local_device << ": global topo before add contiguous op: " << topo_before_contiguous;

      // insert contiguous ops
      HT_LOG_DEBUG << local_device << ": [Execution Plan] insert contiguous op begin...";
      Graph::push_graph_ctx(id()); // ensure the new ops created in execute_graph
      InsertContiguousOp(topo_before_contiguous);
      Graph::pop_graph_ctx();
      HT_LOG_DEBUG << local_device << ": [Execution Plan] insert contiguous op end...";
      is_execute_plan_changed = true;
      break;
    }
  }

  if (is_execute_plan_changed) {
    // TODO: replace the fetches to the new substitued results after SubstituteCommOp
    for (auto& fetch : fetches) {
      auto& fetch_op = fetch->producer();
      HT_ASSERT(!is_comm_op(fetch_op)) << fetch << ": is substitued already, don't try to fetch it.";
    }

    // execute in each iteration, should be cached 
    HT_LOG_DEBUG << local_device << ": [Execution Plan] get local fw/bw topo begin...";
    // update topo
    OpRefList updated_topo = Graph::TopoSort(fetches, -1, is_op_computed);
    // HT_LOG_DEBUG << local_device << ": updated global topo after substitute comm_op: " << updated_topo;

    // split into fw_topo and bw_topo
    OpRefList fw_topo, bw_topo;
    std::tie(fw_topo, bw_topo) = disentangle_forward_and_backward_ops_by_loss(updated_topo, {loss});
    // OpRefList fw_topo, bw_topo;
    // std::tie(fw_topo, bw_topo) = disentangle_forward_and_backward_ops(updated_topo);

    // judge whether is shared weight p2p in fw/bw.
    // TODO: leverage subgraph, refactor all these
    auto is_fw_share_weight_p2p_send = [&](const OpRef& op_ref) -> bool {
      // HT_LOG_WARN << "call is_fw_share_weight_p2p_send";
      if (is_pipeline_stage_send_op(op_ref.get())) {
        Operator input_op = op_ref.get()->input(0)->producer();
        while (true) {
          if (GetSubGraph(input_op) != nullptr && GetSubGraph(input_op)->subgraph_type() == SubGraphType::PIPELINE) {
            input_op = input_op->input(0)->producer();
            continue;
          }
          if (GetSubGraph(input_op) != nullptr && GetSubGraph(input_op)->subgraph_type() == SubGraphType::OPTIMIZE_COMPUTE_BRIDGE) {
            return true;
          }
          break;
        }
      }
      return false;
    };
    auto is_fw_share_weight_p2p_recv = [&](const OpRef& op_ref) -> bool {
      // HT_LOG_WARN << "call is_fw_share_weight_p2p_recv";
      if (is_pipeline_stage_recv_op(op_ref.get())) {
        Operator input_op = op_ref.get()->in_dep_linker(0)->producer();
        if (GetSubGraph(input_op) != nullptr && GetSubGraph(input_op)->subgraph_type() == SubGraphType::OPTIMIZE_COMPUTE_BRIDGE) {
          // HT_LOG_INFO << local_device << ": shared weight p2p fw recv: " << op_ref;
          return true;
        }
      }
      return false;
    };
    auto is_bw_share_weight_grad_p2p_send = [&](const OpRef& op_ref) -> bool {
      // HT_LOG_WARN << "call is_bw_share_weight_p2p_send";
      if (is_pipeline_stage_send_op(op_ref.get())) {
        Operator output_op = op_ref.get()->out_dep_linker()->consumer(0);
        if (GetSubGraph(output_op) != nullptr && GetSubGraph(output_op)->subgraph_type() == SubGraphType::COMPUTE_OPTIMIZE_BRIDGE) {
          return true;
        }
      }
      return false;    
    };
    auto is_bw_share_weight_grad_p2p_recv = [&](const OpRef& op_ref) -> bool {
      // HT_LOG_WARN << "call is_bw_share_weight_p2p_recv";
      if (is_pipeline_stage_recv_op(op_ref.get())) {
        auto output_op = op_ref.get()->output(0)->consumer(0);
        while (true) {
          if (GetSubGraph(output_op) != nullptr && GetSubGraph(output_op)->subgraph_type() == SubGraphType::PIPELINE) {
            output_op = output_op->output(0)->consumer(0);
            continue;
          }
          if (GetSubGraph(output_op) != nullptr && GetSubGraph(output_op)->subgraph_type() == SubGraphType::COMPUTE_OPTIMIZE_BRIDGE) {
            return true;
          }
          break;
        }
      }
      return false;
    };

    // get local_fw_topo and local_bw_topo, not contains placeholder & varivale ops
    // ops to substitute comm_op is in the same placement_group, but in the different placement
    OpRefList local_fw_topo, local_bw_topo, local_placeholder_variable_ops, local_topo;
    auto get_local_topo = [&](OpRefList& _topo, OpRefList& _local_topo, OpRefList& _placeholder_variable_ops) {
      // move p2p send op to topo tail
      OpRefList send_op_list;
      OpRefList recv_op_list;
      OpRefList compute_op_list;
      OpRefList update_op_list;
      OpRefList optimizer_op_list;
      OpRefList group_op_list;
      OpRefList share_weight_recv_op_list;
      OpRefList share_weight_grad_recv_op_list;
      // todo: assume pp stages = [0,1,2,3]->[4,5,6,7], then 0 send pre-half of wte to 4, 1 send last-half of wte to 5; 
      // 2 send pre-half of wte to 6, 3 send last-half of wte to 7; notice that 0 and 2 are send the same, 1 and 3 are send the same
      // so 0 can send half of pre-half to 4, 2 can send another half of pre-half to 6, then 4 and 6 do gather(at this time, 4 and 6
      // are waiting for pp bubbles, the time will be reused)
      // todo2: in pipeline last micro batch, stage id > 0 can move grad_reduce & update & group after pipeline p2p and use bubble
      // to do later update, but stage id = 0 can do aync grad_reduce immediately after weight grad was computed, which can be 
      // overlapped with backward compute(no overhead for pure dp, but may make tp backward allreduce slower)
      for (auto& op_ref : _topo) {
        if (op_ref.get()->placement() == local_device || op_ref.get()->op_meta().is_cpu || op_ref.get()->op_meta().is_offload) {
          // share weight p2p send op will not block anything! so treat it as commom compute op
          // fw weight share only in micro batch 0, bw weight grad share only in last micro batch
          // HT_LOG_DEBUG << "get op type for " << op_ref.get();
          if (is_fw_share_weight_p2p_send(op_ref) || is_bw_share_weight_grad_p2p_send(op_ref)) {
            compute_op_list.push_back(op_ref);
            // HT_LOG_WARN << "compute_op";
          } else if (is_fw_share_weight_p2p_recv(op_ref)) {
            share_weight_recv_op_list.push_back(op_ref);
            // HT_LOG_WARN << "share_weight_recv_op";
          } else if (is_bw_share_weight_grad_p2p_recv(op_ref)) {
            share_weight_grad_recv_op_list.push_back(op_ref);
            // HT_LOG_WARN << "share_weight_grad_recv_op";
          } else if (is_pipeline_stage_send_op(op_ref.get())) {          
            send_op_list.push_back(op_ref);
            // HT_LOG_WARN << "send_op";
          } else if (is_pipeline_stage_recv_op(op_ref.get())) {
            recv_op_list.push_back(op_ref);
            // HT_LOG_WARN << "recv_op";
          } else {
            if (is_placeholder_op(op_ref) || is_variable_op(op_ref)) {
              _placeholder_variable_ops.push_back(op_ref);
            } else if (GetSubGraph(op_ref) != nullptr && GetSubGraphOpType(op_ref) == SubGraphOpType::UPDATE) {
              update_op_list.push_back(op_ref);
            } else if (is_optimizer_update_op(op_ref)) {
              optimizer_op_list.push_back(op_ref);
            } else if (is_group_op(op_ref)) {
              group_op_list.push_back(op_ref);
            } else {
              compute_op_list.push_back(op_ref);
            }
          }
        }
      }
      /*
      // workaround: sort grad reduce op & update op for consistent order across heterogeneous strategies
      std::sort(update_op_list.begin(), update_op_list.end(), 
                [](const OpRef& op1, const OpRef& op2) {
                    auto layer_idx1 = GetLayerId(op1.get()->output(0));
                    auto layer_idx2 = GetLayerId(op2.get()->output(0));
                    if (layer_idx1 != layer_idx2) {
                      return layer_idx1 > layer_idx2;
                    } else {
                      return op1.get()->name() < op2.get()->name();
                    }
                  });
      std::sort(optimizer_op_list.begin(), optimizer_op_list.end(), 
                [](const OpRef& op1, const OpRef& op2) {
                    auto layer_idx1 = GetLayerId(op1.get()->input(1));
                    auto layer_idx2 = GetLayerId(op2.get()->input(1));
                    if (layer_idx1 != layer_idx2) {
                      return layer_idx1 > layer_idx2;
                    } else {
                      return op1.get()->input(1)->name() < op2.get()->input(1)->name();
                    }
                  });
      */
      _local_topo.insert(_local_topo.end(), share_weight_grad_recv_op_list.begin(), share_weight_grad_recv_op_list.end()); // first stage
      _local_topo.insert(_local_topo.end(), share_weight_recv_op_list.begin(), share_weight_recv_op_list.end()); // last stage
      _local_topo.insert(_local_topo.end(), recv_op_list.begin(), recv_op_list.end());
      _local_topo.insert(_local_topo.end(), compute_op_list.begin(), compute_op_list.end());
      _local_topo.insert(_local_topo.end(), send_op_list.begin(), send_op_list.end());
      // move allreduce/reduce-scatter & udpate & group op after pipeline p2p, to make p2p & allreduce/reduce-scatter overlap
      _local_topo.insert(_local_topo.end(), update_op_list.begin(), update_op_list.end());
      _local_topo.insert(_local_topo.end(), optimizer_op_list.begin(), optimizer_op_list.end());
      _local_topo.insert(_local_topo.end(), group_op_list.begin(), group_op_list.end());
    };
    get_local_topo(fw_topo, local_fw_topo, local_placeholder_variable_ops);
    get_local_topo(bw_topo, local_bw_topo, local_placeholder_variable_ops); 

    local_topo.reserve(local_placeholder_variable_ops.size() + local_fw_topo.size() + local_bw_topo.size());
    local_topo.insert(local_topo.end(), local_placeholder_variable_ops.begin(), local_placeholder_variable_ops.end());
    local_topo.insert(local_topo.end(), local_fw_topo.begin(), local_fw_topo.end());
    local_topo.insert(local_topo.end(), local_bw_topo.begin(), local_bw_topo.end());
    HT_LOG_DEBUG << local_device  << ": local placeholder & variable ops: " << local_placeholder_variable_ops;
    HT_LOG_DEBUG << local_device << ": local fw topo: " << local_fw_topo << "\nlocal bw topo: " << local_bw_topo;
    HT_LOG_DEBUG << local_device << ": [Execution Plan] get local fw/bw topo end...";

    HT_LOG_DEBUG << local_device << ": [Execution Plan] get leaf symbolic tensor list begin...";
    for (auto& op_ref : updated_topo) {
      for (auto& output : op_ref.get()->outputs()) {
        if (output->symbolic() && is_SyShape_leaf(output->symbolic_shape())) {
          AddLeafSymbolicTensor(output);
        }
      }
    }
    HT_LOG_DEBUG << local_device << ": [Execution Plan] get leaf symbolic tensor list end...";

    HT_LOG_DEBUG << local_device << ": [Execution Plan] get transfer & grad map and running-once tensor & op begin...";
    // some special ops only run at the begining
    // some special ops shouldn't be updated before grad accumulation finished
    TensorIdSet dtype_transfer_tensor;
    TensorIdSet shared_weight_tensor;
    TensorIdSet accumulated_tensor;
    OpIdSet shared_weight_p2p;
    OpIdSet shared_weight_grad_p2p;
    OpIdSet accumulated_ops;
    // 修正op映射后的buffer
    std::unordered_map<DataType, std::shared_ptr<ParamBuffer>> substitute_transfer_param_buffer_map;
    std::unordered_map<DataType, std::shared_ptr<ParamBuffer>> substitute_current_grad_buffer_map;
    std::unordered_map<DataType, std::shared_ptr<ParamBuffer>> substitute_accumulate_grad_buffer_map;
    for (int i = 0; i < static_cast<int>(DataType::NUM_DATA_TYPES); i++) {
      DataType dtype = static_cast<DataType>(i);
      substitute_transfer_param_buffer_map[dtype] = std::make_shared<ParamBuffer>("substitute_transfer_param_buffer_" + DataType2Str(dtype));
      substitute_current_grad_buffer_map[dtype] = std::make_shared<ParamBuffer>("substitute_current_grad_buffer_" + DataType2Str(dtype));
      substitute_accumulate_grad_buffer_map[dtype] = std::make_shared<ParamBuffer>("substitute_accumulate_grad_buffer_" + DataType2Str(dtype));
    }  
    if (_terminate_subgraph != nullptr) {
      _terminate_subgraph->topo_sort();
    }
    for (auto& op_ref : bw_topo) {
      if (is_group_op(op_ref)) {
        accumulated_ops.insert(op_ref.get()->id());
        continue;
      }
      else if (!is_optimizer_update_op(op_ref)) {
        continue;
      }
      auto& param = op_ref.get()->input(0);
      HT_ASSERT(_optimize_compute_bridge_subgraph_map.find(param->producer()->id()) != _optimize_compute_bridge_subgraph_map.end()
                && _compute_optimize_bridge_subgraph_map.find(param->producer()->id()) != _compute_optimize_bridge_subgraph_map.end())
        << "cannot find the corresponding optimize-compute or compute-optimize bridge subgraph for " << param;
      _optimize_compute_bridge_subgraph_map[param->producer()->id()]->topo_sort();
      _compute_optimize_bridge_subgraph_map[param->producer()->id()]->topo_sort();
      const auto& transfer_topo = _optimize_compute_bridge_subgraph_map[param->producer()->id()]->ops_topo();
      const auto& update_topo = _compute_optimize_bridge_subgraph_map[param->producer()->id()]->update_ops_topo();
      // HT_LOG_INFO << param << " corresponding transfer subgraph " << _optimize_compute_bridge_subgraph_map[param->producer()->id()]->global_name() << " topo is " << transfer_topo;
      // HT_LOG_INFO << param << " corresponding update subgraph " << _compute_optimize_bridge_subgraph_map[param->producer()->id()]->global_name() << " topo is " << update_topo;
      /*
      if (_dp_overlap) {
        auto parent_transfer_subgraph = _optimize_compute_bridge_subgraph_map[param->producer()->id()]->parent_graph();
        auto parent_update_subgraph = _compute_optimize_bridge_subgraph_map[param->producer()->id()]->parent_graph();
        HT_ASSERT(parent_transfer_subgraph != nullptr && parent_transfer_subgraph->subgraph_type() == SubGraphType::MODULE
                  && parent_update_subgraph != nullptr && parent_update_subgraph->subgraph_type() == SubGraphType::MODULE
                  && parent_transfer_subgraph == parent_update_subgraph)
          << "the parent subgraph shoud be exactly the same";
        auto parent_module = parent_transfer_subgraph;
        parent_module->topo_sort();
        HT_LOG_INFO << parent_module->global_name() << " fwd topo is " << parent_module->ops_topo();
        HT_LOG_INFO << parent_module->global_name() << " bwd topo is " << parent_module->bwd_ops_topo();
      }
      */
      // 把每个grad reduce算子的上一个算子取出来
      if (_overlap_grad_reduce) {
        if (update_topo.size() >= 1) {
          if (is_grad_reduce_op(update_topo.front())) {
            _grad_reduce_subgraph_map[update_topo.front().get()->input(0)->id()] = _compute_optimize_bridge_subgraph_map[param->producer()->id()];
          }
        }
      }
      for (auto& transfer_op_ref : transfer_topo) {
        // batched send recv可能没有output
        if (transfer_op_ref.get()->num_outputs() >= 1) {
          dtype_transfer_tensor.insert(transfer_op_ref.get()->output(0)->id());
        }
      }
      for (auto& update_op_ref : update_topo) {
        accumulated_ops.insert(update_op_ref.get()->id());
        // 如果update topo中某个input不是该topo中任何一个的output那么就是需要累积梯度的tensor
        for (auto& input : update_op_ref.get()->inputs()) {
          bool is_accumulated_tensor = true;
          for (auto& check_op_ref : update_topo) {
            // 再往后的op肯定不会
            if (check_op_ref.get()->id() == update_op_ref.get()->id()) {
              break;
            }
            for (auto& check_output : check_op_ref.get()->outputs()) {
              if (input->id() == check_output->id()) {
                is_accumulated_tensor = false;
                break;
              }
            }
            if (is_accumulated_tensor == false) {
              break;
            }
          }
          if (is_accumulated_tensor) {
            accumulated_tensor.insert(input->id());
          }
        }
      }    
      // 有该param对应的local transfer param
      if (transfer_topo.size() >= 1 && transfer_topo.back().get()->num_outputs() >= 1 
          && transfer_topo.back().get()->output(0)->num_consumers() >= 1) {
        auto& final_transfer = transfer_topo.back().get()->output(0);
        auto transfer_it = _transfer_map.find(param->id());
        HT_ASSERT(transfer_it != _transfer_map.end())
          << "cannot find the mapping of " << param << " in the transfer map";
        auto& transfer_in_buffer = transfer_it->second;
        if (_shape_mismatch_flag == 0) {
          HT_ASSERT(transfer_in_buffer->shape() == final_transfer->shape()
                    && transfer_in_buffer->dtype() == final_transfer->dtype()
                    && transfer_in_buffer->stride() == final_transfer->stride())
            << "the meta (except device) of the transfer before/after substitute comm op should be equal"
            << ", but meta of transfer in buffer is " << transfer_in_buffer->meta()
            << ", and meta of transfer is " << final_transfer->meta();
        }
        HT_ASSERT(transfer_in_buffer->cur_ds_union().check_equal(final_transfer->cur_ds_union()))
          << "the ds union of the transfer before/after substitute comm op should be equal"
          << ", but found " << transfer_in_buffer << ": " << transfer_in_buffer->cur_ds_union().ds_union_info()
          <<  " vs " << final_transfer << ": " << final_transfer->cur_ds_union().ds_union_info();
        HT_ASSERT(transfer_in_buffer->placement_group_union().check_equal(final_transfer->placement_group_union()))
          << "the placement group union of the transfer before/after substitute comm op should be equal"
          << ", but found " << transfer_in_buffer << ": " << transfer_in_buffer->placement_group_union() 
          << " vs " << final_transfer << ": " << final_transfer->placement_group_union();
        substitute_transfer_param_buffer_map[final_transfer->dtype()]->AddTensor(final_transfer);
        _transfer_map[param->id()] = final_transfer;  
      } else {
        // HT_LOG_INFO << param << " corresponding final transfer is nonlocal";
      }
      // HT_LOG_INFO << param << " substitute final transfer done";
      // 有该param对应的local grad
      if (update_topo.size() >= 1 && update_topo.back().get()->num_outputs() >= 1
          && update_topo.back().get()->output(0)->num_consumers() >= 1) {
        HT_ASSERT(is_optimizer_update_op(update_topo.back()))
          << "subgraph " << _compute_optimize_bridge_subgraph_map[param->producer()->id()]->global_name() << " last op must be an optimizer op";
        auto& final_grad = update_topo.back().get()->input(1);
        HT_ASSERT(final_grad->id() == op_ref.get()->input(1)->id())
          << "final grad in compute-optimize bridge subgraph should be the same as the update op input 1";    
        auto grad_it = _grad_map.find(param->id());
        HT_ASSERT(grad_it != _grad_map.end())
          << "cannot find the mapping of " << param << " in the grad map";
        auto& grad_in_buffer = grad_it->second;
        if (_shape_mismatch_flag == 0) {
          HT_ASSERT(grad_in_buffer->shape() == final_grad->shape()
                    && grad_in_buffer->dtype() == final_grad->dtype()
                    && grad_in_buffer->stride() == final_grad->stride())
            << "the meta (except device) of the grad before/after substitute comm op should be equal"
            << ", but meta of grad in buffer is " << grad_in_buffer->meta()
            << ", and meta of grad is " << final_grad->meta();
        }
        HT_ASSERT(grad_in_buffer->cur_ds_union().check_equal(final_grad->cur_ds_union()))
          << "the ds union of the grad before/after substitute comm op should be equal"
          << ", but found " << grad_in_buffer << ": " << grad_in_buffer->cur_ds_union().ds_union_info()
          <<  " vs " << final_grad << ": " << final_grad->cur_ds_union().ds_union_info();
        HT_ASSERT(grad_in_buffer->placement_group_union().check_equal(final_grad->placement_group_union()))
          << "the placement group union of the grad before/after substitute comm op should be equal"
          << ", but found " << grad_in_buffer << ": " << grad_in_buffer->placement_group_union() 
          << " vs " << final_grad << ": " << final_grad->placement_group_union();
        substitute_current_grad_buffer_map[final_grad->dtype()]->AddTensor(final_grad);
        substitute_accumulate_grad_buffer_map[final_grad->dtype()]->AddTensor(final_grad);
        _grad_map[param->id()] = final_grad;   
      } else {
        HT_ASSERT(param->placement() != local_device)
          << param << " should have a corresponding final grad";
        // HT_LOG_INFO << param << " corresponding final grad is nonlocal";
      }
      // HT_LOG_INFO << param << " substitute final grad done";
    }
    // 这里直接将substitute后的param buffer替换掉define graph中生成的
    if (!bw_topo.empty()){
      for (int i = 0; i < static_cast<int>(DataType::NUM_DATA_TYPES); i++) {
        DataType dtype = static_cast<DataType>(i);
        if (_shape_mismatch_flag == 0) {
        HT_ASSERT(_transfer_param_buffer_map[dtype]->size() == substitute_transfer_param_buffer_map[dtype]->size()
                  && _current_grad_buffer_map[dtype]->size() == substitute_current_grad_buffer_map[dtype]->size()
                    && _accumulate_grad_buffer_map[dtype]->size() == substitute_accumulate_grad_buffer_map[dtype]->size())
            << "buffer size should be equal";
        }
        _transfer_param_buffer_map[dtype] = substitute_transfer_param_buffer_map[dtype];
        _current_grad_buffer_map[dtype] = substitute_current_grad_buffer_map[dtype];
        _accumulate_grad_buffer_map[dtype] = substitute_accumulate_grad_buffer_map[dtype];
      }  
    }
    // dtype_transfer_tensor还有别的（非param的）需要再插入一轮
    // (group1) variable op -> send -> (group2) recv -> other ops
    for (auto& op_ref : local_fw_topo) {
      if (is_fw_share_weight_p2p_send(op_ref)) {
        shared_weight_p2p.insert(op_ref.get()->id());
      }
      if (is_fw_share_weight_p2p_recv(op_ref)) {
        shared_weight_p2p.insert(op_ref.get()->id());
        shared_weight_tensor.insert(op_ref.get()->output(0)->id());
      }
      if (is_data_transfer_op(op_ref) && is_variable_op(op_ref.get()->input(0)->producer())) {
        dtype_transfer_tensor.insert(op_ref.get()->output(0)->id());
      }
    }
    for (auto& op_ref : local_bw_topo) {
      if (is_bw_share_weight_grad_p2p_send(op_ref) || is_bw_share_weight_grad_p2p_recv(op_ref)) {
        shared_weight_grad_p2p.insert(op_ref.get()->id());
      }
      if (is_data_transfer_op(op_ref) && is_variable_op(op_ref.get()->input(0)->producer())) {
        dtype_transfer_tensor.insert(op_ref.get()->output(0)->id());
      }
    }
    HT_LOG_DEBUG << local_device << ": [Execution Plan] get transfer & grad map and running-once tensor & op end...";
    
    HT_ASSERT(shared_weight_tensor.empty() && shared_weight_p2p.empty() && shared_weight_grad_p2p.empty())
      << "currently subgraph & ds hierarchy may not be compatible with the share weight, please don't use the share weight at this moment";
    // update & cached execute plan 
    _execute_plan.update(local_placeholder_variable_ops, local_fw_topo, local_bw_topo, local_topo, dtype_transfer_tensor,
                         shared_weight_tensor, shared_weight_p2p, shared_weight_grad_p2p, accumulated_tensor, accumulated_ops);
  }
  TOK(prepare_run);
  HT_LOG_DEBUG << local_device << ": prepare execution plan cost time = " << COST_MSEC(prepare_run) << " ms."; 
  
  if (_used_ranks.size() >= 2) {
    auto& comm_group = hetu::impl::comm::NCCLCommunicationGroup::GetOrCreate(_used_ranks, local_device);
    comm_group->Barrier(true);
  }
  // sync partially
  /*
  std::vector<int> ranks;
  for (const auto& stage : _pipeline_map[hetu::impl::comm::GetLocalDevice()]) {
    for (const auto& device : stage.devices()) {
      auto rank = hetu::impl::comm::DeviceToWorldRank(device);
      if (std::find(ranks.begin(), ranks.end(), rank) == ranks.end()) {
        ranks.push_back(rank);
      }
    }
  }
  if (ranks.size() >= 2) {
    std::sort(ranks.begin(), ranks.end());
    // hetu::impl::comm::Barrier(ranks);
    auto& comm_group = hetu::impl::comm::NCCLCommunicationGroup::GetOrCreate(ranks, local_device);
    comm_group->Barrier(true);
  }
  */

  // mempool test
  /*
  TIK(free_mempool);
  hetu::impl::ProfileAfterEmptyAllCUDACache(local_device);
  TOK(free_mempool);
  HT_LOG_INFO << local_device << ": free mempool time = " << COST_MSEC(free_mempool) << " ms";
  */

  TIK(crucial_run);
  // ****核心的exec graph执行部分****
  auto results = CrucialRun(fetches, feed_dict, int_symbol_dict, num_micro_batches, ctx);
  auto profiler_optional = hetu::impl::Profile::get_cur_profile();
  bool is_analysis_perf = false;
  if (is_analysis_perf || _straggler_flag || profiler_optional) {
    if (_used_ranks.size() >= 2) {
      auto& comm_group = hetu::impl::comm::NCCLCommunicationGroup::GetOrCreate(_used_ranks, local_device);
      comm_group->Barrier(true);
    }
  }
  TOK(crucial_run);
  HT_LOG_DEBUG << local_device << ": crucial run time = " << COST_MSEC(crucial_run) << " ms";
  if (_run_level == RunLevel::TOPO) {
    return results;
  }
  
  // get all micro batches memory consumption
  if (_memory_profile_level == MEMORY_PROFILE_LEVEL::MICRO_BATCH && _memory_log_file_path != "") {
    std::ofstream file;
    std::string suffix = "_" + std::to_string(hetu::impl::comm::GetWorldRank()) + ".txt";
    file.open(_memory_log_file_path + suffix, std::ios_base::app);
    if (file.is_open()) {
      file << "[" << std::endl;
    } else {
      HT_RUNTIME_ERROR << "Error opening the file";
    }
    auto size = _all_micro_batches_memory_info.size();
    for (size_t i = 0; i < size; i++) {
      if (i != size - 1) {
        file << *_all_micro_batches_memory_info[i] << "," << std::endl;
      } else {
        file << *_all_micro_batches_memory_info[i] << std::endl;
      }
    }
    file << "]";
    file.close();
  }

  // get op execute time, sort and analysis
  if (is_analysis_perf || _straggler_flag) {
    std::vector<std::pair<int64_t, int64_t>> op_execute_time;
    for (auto& op_ref : _execute_plan.local_topo) {
      auto& op = op_ref.get();
      if (is_placeholder_op(op) || is_variable_op(op)) {
        continue;
      }
      if (is_pipeline_stage_send_op(op) || is_pipeline_stage_recv_op(op)) {
        continue;
      }
      // get time cost for all micro batches
      int64_t time_cost = 0;
      for (int i = 0; i < num_micro_batches; i++) {
        time_cost += op->TimeCost(i);
      }
      op_execute_time.push_back({op->id(), time_cost});
    }
    // p2p events
    for (int i = 0; i < _p2p_events.size() / 2; i++) {
      auto& start = _p2p_events[2 * i];
      auto& end = _p2p_events[2 * i + 1];
      // record the time of p2p for each pipeline micro-batch
      op_execute_time.push_back({-(i+1), end->TimeSince(*start)});
    }
    std::sort(op_execute_time.begin(), op_execute_time.end(), [](
      std::pair<int64_t, int64_t>& op_t1, std::pair<int64_t, int64_t>& op_t2) {
        return op_t1.second > op_t2.second;
      });
    double attn_fwd_time = 0;
    double attn_bwd_time = 0;
    double optimizer_time = 0;
    double other_compute_time = 0;
    double pp_p2p_time = 0;
    double tp_collective_time = 0;
    double allgather_time = 0;
    double allreduce_time = 0;
    double reducescatter_time = 0;
    double allgather_comm = 0;
    double allreduce_comm = 0;
    double reducescatter_comm = 0;
    double dp_param_gather_time = 0;
    double dp_grad_reduce_time = 0;
    double blocking_time = 0;
    double other_time = 0;
    std::ostringstream out;
    out << "Op Execute Time: ";
    int print_num = 10000;
    for (auto& op_time : op_execute_time) {
      if (op_time.first >= 0) {
        auto op = _op_indexing[op_time.first];
        // print top 10 op
        if (print_num-- > 0) {
          out << std::endl << local_device << ": " << op << "(type = " << op->type() << "), " << "time = " << op_time.second * 1.0 / 1e6 << " ms";
          if (op->num_inputs() > 0) {
            out << "; input shapes = ";
            for (auto& input : op->inputs()) {
              out << input->shape() << ", ";
            }
          }
          out << "; inputs = " << op->inputs();
        }
        if (op->stream_index() == kComputingStream) {
          if (is_optimizer_update_op(op)) {
            optimizer_time += op_time.second * 1.0 / 1e6;
          } else if (is_parallel_attn_op(op)) {
            attn_fwd_time += op_time.second * 1.0 / 1e6;
          } else if (is_parallel_attn_grad_op(op)) {
            attn_bwd_time += op_time.second * 1.0 / 1e6;
          } else {
            other_compute_time += op_time.second * 1.0 / 1e6;
          }
        } else if (op->stream_index() == kCollectiveStream) {
          tp_collective_time += op_time.second * 1.0 / 1e6;
          if (op->type() == "AllGatherOp") {
            allgather_time += op_time.second * 1.0 / 1e6;
            allgather_comm += op->output(0)->numel();
          }
          else if (op->type() == "AllReduceOp" || op->type() == "SplitAllReduceOp") {
            allreduce_time += op_time.second * 1.0 / 1e6;
            allreduce_comm += op->output(0)->numel();
          }
          else if (op->type() == "ReduceScatterOp" || op->type() == "SplitReduceScatterOp") {
            reducescatter_time += op_time.second * 1.0 / 1e6;
            reducescatter_comm += op->input(0)->numel();
          }
        } else if (op->stream_index() == kBridgeStream) {
          auto subgraph = op->graph().GetSubGraph(op);
          HT_ASSERT(subgraph != nullptr)
            << "kBridgeStream should only be used in bridge subgraph";
          if (subgraph->subgraph_type() == SubGraphType::OPTIMIZE_COMPUTE_BRIDGE) {
            dp_param_gather_time += op_time.second * 1.0 / 1e6;
          } else if (subgraph->subgraph_type() == SubGraphType::COMPUTE_OPTIMIZE_BRIDGE) {
            dp_grad_reduce_time += op_time.second * 1.0 / 1e6;
          }
        } else if (op->stream_index() == kBlockingStream) {
          blocking_time += op_time.second * 1.0 / 1e6;
        } else {
          other_time += op_time.second * 1.0 / 1e6;
        }        
      } else {
        out << std::endl << local_device << ": batch p2p " << -op_time.first << " : " << op_time.second * 1.0 / 1e6 << " ms";
        pp_p2p_time += op_time.second * 1.0 / 1e6;
      }
    }
    if (is_analysis_perf) {
      HT_LOG_INFO << local_device << ": " 
                  << "\ntotal run time: " << COST_MSEC(crucial_run) << " ms, "
                  << "attn fwd time: " << attn_fwd_time << " ms, "
                  << "attn bwd time: " << attn_bwd_time << " ms, "
                  << "optimizer time: " << optimizer_time << " ms, "
                  << "other compute time: " << other_compute_time << " ms, "
                  << "tp collective time: " << tp_collective_time << " ms, "
                  << "allgather time: " << allgather_time << " ms, "
                  << "allreduce time: " << allreduce_time << " ms, "
                  << "reducescatter time: " << reducescatter_time << " ms, "
                  << "allgather comm: " << allgather_comm / 1e9 << " GB, "
                  << "allreduce comm: " << allreduce_comm / 1e9 << " GB, "
                  << "reducescatter comm: " << reducescatter_comm / 1e9 << " GB, "
                  << "pp p2p time (include bubble): " << pp_p2p_time << " ms, "
                  << "dp param gather time: " << dp_param_gather_time << " ms, "
                  << "dp grad reduce time: " << dp_grad_reduce_time << " ms, "
                  << "blocking time: " << blocking_time << " ms, "
                  << "other time: " << other_time << " ms" << std::endl
                  << out.str();
    }
    if (_straggler_flag) {
      HT_LOG_WARN << local_device << ": " 
                  << "\ntotal run time: " << COST_MSEC(crucial_run) << " ms, "
                  << "attn fwd time: " << attn_fwd_time << " ms, "
                  << "attn bwd time: " << attn_bwd_time << " ms, "
                  << "optimizer time: " << optimizer_time << " ms, "
                  << "other compute time: " << other_compute_time << " ms, "
                  << "tp collective time: " << tp_collective_time << " ms, "
                  << "allgather time: " << allgather_time << " ms, "
                  << "allreduce time: " << allreduce_time << " ms, "
                  << "reducescatter time: " << reducescatter_time << " ms, "
                  << "allgather comm: " << allgather_comm / 1e9 << " GB, "
                  << "allreduce comm: " << allreduce_comm / 1e9 << " GB, "
                  << "reducescatter comm: " << reducescatter_comm / 1e9 << " GB, "
                  << "pp p2p time (include bubble): " << pp_p2p_time << " ms, "
                  << "dp param gather time: " << dp_param_gather_time << " ms, "
                  << "dp grad reduce time: " << dp_grad_reduce_time << " ms, "
                  << "blocking time: " << blocking_time << " ms, "
                  << "other time: " << other_time << " ms";
      if (_straggler_log_file_path != "") {
        if (_straggler_flag == 1) {
          ofstream_sync file(_straggler_log_file_path, std::ios_base::app);
          if (file.is_open()) {
            file << other_compute_time << std::endl;
          } else {
            HT_RUNTIME_ERROR << "Error opening the file";
          }
        } else if (_straggler_flag == 2) {
          ofstream_sync file(_straggler_log_file_path + "_" + std::to_string(hetu::impl::comm::GetWorldRank()) + ".txt", std::ios_base::app);
          if (file.is_open()) {
            file << "total run time: " << COST_MSEC(crucial_run) << " ms" << std::endl;
            file << "compute time: " << other_compute_time << " ms" << std::endl;
          } else {
            HT_RUNTIME_ERROR << "Error opening the file";
          }
        } else if (_straggler_flag == 3) {
          if (hetu::impl::comm::GetWorldRank() == 0) {
            ofstream_sync file(_straggler_log_file_path, std::ios_base::app);
            if (file.is_open()) {
              file << "total run time: " << COST_MSEC(crucial_run) << " ms, "
                << "attn fwd time: " << attn_fwd_time << " ms, "
                << "attn bwd time: " << attn_bwd_time << " ms, "
                << "optimizer time: " << optimizer_time << " ms, "
                << "other compute time: " << other_compute_time << " ms, "
                << "tp collective time: " << tp_collective_time << " ms, "
                << "pp p2p time (include bubble): " << pp_p2p_time << " ms, "
                << "dp param gather time: " << dp_param_gather_time << " ms, "
                << "dp grad reduce time: " << dp_grad_reduce_time << " ms, "
                << "blocking time: " << blocking_time << " ms, "
                << "other time: " << other_time << " ms" << std::endl;
              auto memory_info = GetCUDAProfiler(local_device)->GetCurrMemoryInfo();
              file << "all reserved: " << memory_info.all_reserved << " mb, "
                << "mempool reserved: " << memory_info.mempool_reserved << " mb, "
                << "mempool peak reserved: " << memory_info.mempool_peak_reserved << " mb, "
                << "mempool allocated: " << memory_info.mempool_allocated << " mb" << std::endl;
            } else {
              HT_RUNTIME_ERROR << "Error opening the file";
            }
          }
        }
      }
    }
  }

  // TODO: merge with analysis perf
  if (profiler_optional) {
    auto profiler = *profiler_optional;
    profiler->set_device(local_device);
    std::vector<std::pair<int64_t, int64_t>> op_execute_time;
    std::unordered_map<int64_t, int64_t> is_forward;
    std::unordered_map<std::string, double> summarized_time;
    bool current_forward = true;
    for (auto& op_ref : _execute_plan.local_topo) {
      auto& op = op_ref.get();
      if (is_placeholder_op(op) || is_variable_op(op)) {
        continue;
      }
      if (is_peer_to_peer_send_op(op) || is_peer_to_peer_recv_op(op)) {
        continue;
      }
      // get time cost for all micro batches
      int64_t time_cost = 0;
      for (int i = 0; i < num_micro_batches; i++) {
        time_cost += op->TimeCost(i);
      }
      op_execute_time.push_back({op->id(), time_cost});
      is_forward[op->id()] = current_forward;
      if (op->id() == loss->producer_id()) {
        current_forward = false;
      }
    }
    // p2p events
    for (int i = 0; i < _p2p_events.size() / 2; i++) {
      auto& start = _p2p_events[2 * i];
      auto& end = _p2p_events[2 * i + 1];
      // record the time of p2p for each pipeline micro-batch
      op_execute_time.push_back({-(i+1), end->TimeSince(*start)});
    }
    for (auto [op_id, op_time] : op_execute_time) {
      double time_in_ms = op_time * 1.0 / 1e6;
      if (op_id < 0) {
        summarized_time["pp-p2p"] += time_in_ms;
        continue;
      }
      auto& op = _op_indexing[op_id];
      if (op->name().find("Block1") != op->name().npos &&
          is_forward[op_id] == 1) {
        summarized_time["block-forward"] += time_in_ms;
      } else if (op->name().find("Block1") < op->name().find("grad") &&
          is_forward[op_id] != 1 && !is_optimizer_update_op(op) &&
          !(op->stream_index() == kCollectiveStream && is_optimizer_update_op(op->output(0)->consumer(0)))) {
        // exclude update op and grads-reduce op
        summarized_time["block-backward"] += time_in_ms;
      }
      if (op->stream_index() == kComputingStream) {
        if (is_optimizer_update_op(op)) {
          summarized_time["optimizer-update"] += time_in_ms;
          continue;
        }
        if (is_forward[op_id] == 1) {
          summarized_time["forward-compute"] += time_in_ms;
        } else {
          summarized_time["backward-compute"] += time_in_ms;
        }
        summarized_time["forward-backward-compute"] += time_in_ms;
      } else if (op->stream_index() == kCollectiveStream) {
        summarized_time["tp-collective"] += time_in_ms;
        if (is_forward[op_id] == 1) {
          summarized_time["tp-collective-forward"] += time_in_ms;
        } else {
          summarized_time["tp-collective-backward"] += time_in_ms;
        }
      } else if (op->stream_index() == kBridgeStream) {
        auto subgraph = op->graph().GetSubGraph(op);
        HT_ASSERT(subgraph != nullptr)
          << "kBridgeStream should only be used in bridge subgraph";
        if (subgraph->subgraph_type() == SubGraphType::OPTIMIZE_COMPUTE_BRIDGE) {
          summarized_time["dp-param-gather"] += time_in_ms;
        } else if (subgraph->subgraph_type() == SubGraphType::COMPUTE_OPTIMIZE_BRIDGE) {
          summarized_time["dp-grad-reduce"] += time_in_ms;
        }
      } else if (op->stream_index() == kBlockingStream) {
        summarized_time["blocking"] += time_in_ms;
      } else {
        summarized_time["other"] += time_in_ms;
      }
      HTShapeList inputs_shape;
      Operator::for_each_input_tensor(op, [&](const Tensor& input) {
         inputs_shape.push_back(input->shape());
      });
      profiler->push(op->type(), op->name(), inputs_shape, op_time);
    }

    // total time = forward + backward = forward compute + backward compute + tp-collective + tp-p2p
    profiler->push("total-run-time", COST_MSEC(crucial_run));
    profiler->push("forward-compute", summarized_time["forward-compute"]);
    profiler->push("backward-compute", summarized_time["backward-compute"]);
    profiler->push("forward-backward-compute", summarized_time["forward-backward-compute"]);
    profiler->push("pp-p2p", summarized_time["pp-p2p"]);
    profiler->push("tp-collective", summarized_time["tp-collective"]);
    profiler->push("dp-param-gather", summarized_time["dp-param-gather"]);
    profiler->push("dp-grad-reduce", summarized_time["dp-grad-reduce"]);
    profiler->push("blocking", summarized_time["blocking"]);
    profiler->push("other", summarized_time["other"]);
    profiler->push("total-forward-stream-time", summarized_time["forward-compute"] + summarized_time["tp-collective-forward"]);
    profiler->push("total-backward-stream-time", summarized_time["backward-compute"] + summarized_time["tp-collective-backward"]);
    profiler->push("total-stream-time", summarized_time["forward-backward-compute"] + summarized_time["tp-collective"] + summarized_time["tp-p2p"] + summarized_time["pp-p2p"]);
    profiler->push("block-stream-time", summarized_time["block-forward"] + summarized_time["block-backward"]);
    profiler->push("block-forward", summarized_time["block-forward"]);
    profiler->push("block-backward", summarized_time["block-backward"]);
  }

  _p2p_events.clear();
  return results;
}

// TODO: merge two `Run` func
NDArrayList ExecutableGraph::Run(const TensorList& fetches,
                                 const FeedDict& feed_dict) {
  HT_RUNTIME_ERROR << "NotImplementedError";
  
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
  Tensor2NDArrayListMap tensor2data_list;
  tensor2data_list.reserve(topo.size());
  tensor2data_list.insert(feed_dict.begin(), feed_dict.end());
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
      auto& data = tensor2data_list[op->input(i)->id()][0];
      if (data->device() != op->input(i)->placement() ||
          data->dtype() != op->input(i)->dtype()) {
        tensor2data_list[op->input(i)->id()][0] =
          NDArray::to(data, op->input(i)->placement(), op->input(i)->dtype(),
                      op->stream_index());
      }
      inputs.push_back(tensor2data_list[op->input(i)->id()][0]);
    }
    auto outputs = op->Compute(inputs, runtime_ctx);

    for (size_t i = 0; i < outputs.size(); i++) {
      tensor2data_list.insert({op->output(i)->id(), {outputs[i]}});
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
