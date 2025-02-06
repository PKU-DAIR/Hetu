#include "hetu/graph/ops/data_transfer.h"
#include "hetu/graph/offload/activation_cpu_offload.h"

namespace hetu {
namespace graph {

std::stack<std::vector<std::vector<bool>>> ActivationCPUOffload::_multi_cpu_offload_stack{{std::vector<std::vector<bool>>{{false}}}};

bool ActivationCPUOffload::IsNoOffloadOp(Operator& op) {
  return is_comm_op(op);
}

void ActivationCPUOffload::OffloadTensorToCPU(const OpRefList& topo_order, const Tensor& tensor) {
  auto& cur_exec_graph = reinterpret_cast<ExecutableGraph&>(Graph::GetGraph(Graph::cur_graph_ctx()));
  Op2OpRefMap mapped_grad_ops;
  for (auto& consumer : tensor->consumers()) {
    if (!consumer.get()->is_bw_op()) {
      continue;
    }
    mapped_grad_ops.insert({consumer.get()->id(), consumer});
  }
  if (tensor->placement().is_cpu() || mapped_grad_ops.empty()) {
    return;
  }

  auto& op = tensor->producer();
  HT_LOG_DEBUG << "[Offload] insert offload ops to tensor " << tensor
               << " whose original consumers are\n\t\t" << tensor->consumers();
  auto offload_tensor = MakeDataD2HOp(Device(kCPU), tensor,
                                      OpMeta().set_is_offload(true)
                                              .set_name(op->name() + "_offload"));
  cur_exec_graph.RecordExecTensor(offload_tensor);
  auto& offload_op = offload_tensor->producer();
  if (tensor->placement_group_union().size() != 0)
    offload_op->MapToParallelDevices(tensor->placement_group_union());
  offload_op->Instantiate(Device(kCPU), kD2HStream);

  // Find the first grad consumer of the tensor and build execution dependency
  TensorList in_deps;
  for (auto& op_ref : topo_order) {
    if (mapped_grad_ops.find(op_ref.get()->id()) == mapped_grad_ops.end()) {
      continue;
    }
    for (auto& input : op_ref.get()->inputs()) {
      if (!input->producer()->is_bw_op()) {
        continue;
      }
      in_deps.push_back(input);
    }
    if (!in_deps.empty()) {
      break;
    }
  }
  
  std::unordered_map<DeviceIndex, Tensor> device_ops;
  for (auto& out_consumer_ref : tensor->consumers()) {
    auto& out_consumer = out_consumer_ref.get();
    if (!out_consumer->is_bw_op()) {
      continue;
    }
    if (device_ops.find(out_consumer->placement().index()) != device_ops.end()) {
      HT_LOG_DEBUG << "[Offload] inputs of consumer " << out_consumer
                   << "before replacement: " << out_consumer->inputs();
      for (int j = 0; j < out_consumer->num_inputs(); j++) {
        if (out_consumer->input(j)->id() == tensor->id()) {
          Graph::ReplaceInput(out_consumer, j, device_ops[out_consumer->placement().index()]);
        }
      }
      HT_LOG_DEBUG << "[Offload] inputs of consumer " << out_consumer
                   << "after replacement: " << out_consumer->inputs();
    } else {
      Tensor load_tensor = MakeDataH2DOp(out_consumer->placement(), offload_tensor,
                                         OpMeta().set_extra_deps(in_deps)
                                                 .set_name(op->name() + "_load"));
      HT_LOG_DEBUG << "[Offload] Insert H2D op " << load_tensor->producer()
                   << " for " << tensor->name();
      cur_exec_graph.RecordExecTensor(load_tensor);
      auto& load_op = load_tensor->producer();
      if (offload_tensor->placement_group_union().size() != 0)
        load_op->MapToParallelDevices(offload_tensor->placement_group_union());
      load_op->Instantiate(out_consumer->placement(), kH2DStream);
      device_ops[out_consumer->placement().index()] = load_tensor;
      HT_LOG_DEBUG << "[Offload] inputs of consumer " << out_consumer
                   << "before replacement: " << out_consumer->inputs();
      for (int j = 0; j < out_consumer->num_inputs(); j++) {
        if (out_consumer->input(j)->id() == tensor->id()) {
          Graph::ReplaceInput(out_consumer, j, load_tensor);
        }
      }
      HT_LOG_DEBUG << "[Offload] inputs of consumer " << out_consumer
                   << "after replacement: " << out_consumer->inputs();
    }
  }
  HT_LOG_DEBUG << "[Offload] insert offload ops to output " << tensor
               << " whose new consumers are\n\t\t" << tensor->consumers();
}

void ActivationCPUOffload::OffloadToCPU(const OpRefList& topo_order) {
  auto has_grad_consumer = [](const Tensor& tensor) {
    return Tensor::any_consumer_of(tensor, [](const OpRef& op_ref) -> bool {
      return op_ref.get()->is_bw_op();
    });
  };
  // Find candidate offload ops whose outputs are the inputs of grad ops
  HT_LOG_DEBUG << "[Offload] find candidate offload ops begin...";
  OpRefList candidate_offload_ops;
  for (auto& op_ref : topo_order) {
    auto& op = op_ref.get();
    if (!op->op_meta().get_cpu_offload(op->graph().COMPUTE_STRATEGY_ID, op->suggested_hetero_id()) || IsNoOffloadOp(op)) {
      continue;
    }
    if (Operator::any_output_tensor_of(op, has_grad_consumer)) {
      candidate_offload_ops.push_back(op_ref);
    }
  }
  HT_LOG_DEBUG << "[Offload] found " << candidate_offload_ops.size()
               << " candidate offload ops: " << candidate_offload_ops;
  for (auto& op_ref : candidate_offload_ops) {
    auto& op = op_ref.get();
    HT_LOG_DEBUG << "[Offload] Offload outputs of op " << op << " with outputs: " << op->outputs();
    Operator::for_each_output_tensor(op, [&topo_order](const Tensor& tensor) {
      OffloadTensorToCPU(topo_order, tensor);
    });
  }
}

} // namespace graph
} // namespace hetu