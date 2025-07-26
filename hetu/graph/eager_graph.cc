#include "hetu/graph/eager_graph.h"
#include "hetu/graph/ops/variable.h"
#include "hetu/impl/profiler/profiler.h"

namespace hetu {
namespace graph {

Operator& EagerGraph::MakeOpInner(std::shared_ptr<OpInterface> body,
                                  TensorList inputs, OpMeta op_meta) {
  _check_all_inputs_in_graph(inputs, op_meta.extra_deps);
  auto& op = MakeAndAddOp(std::move(body), std::move(inputs), std::move(op_meta));

  // Eager instantiation and execution
  Device placement = op->eager_device();
  if (placement.is_undetermined()) {
      if (op->op_indicator() == VARIABLE_OP) {
      const auto& opimpl = reinterpret_cast<const VariableOpImpl&>(op->body());
      placement = opimpl.device();
    }
  }
  if (placement.is_undetermined()) {
    if (op->num_inputs() > 0) {
      placement = op->input(0)->device();
    } else {
      placement = Device(kCPU);
    }
  }
  StreamIndex stream_id = get_suggested_stream_index(op);

  HT_LOG_DEBUG << "Instantiating op " << op << " (placement=" << placement
    << ", stream_index=" << stream_id << ")";
  bool ok = op->Instantiate(placement, stream_id);
  HT_RUNTIME_ERROR_IF(!ok) << "Failed to place op " << op->name() << " on device " << placement;
  
  NDArrayList input_arrays;
  input_arrays.reserve(op->num_inputs());
  // Insert Contiguous ops
  if (op->body().require_contig_inputs()) {
    for (auto& input : op->inputs()) {
      if (!input->is_contiguous()) {
        input_arrays.push_back(
          NDArray::contiguous(_preserved_data[input->id()],
                              op->instantiation_ctx().stream_index));
      } else {
        input_arrays.push_back(_preserved_data[input->id()]);
      }
    }
  } else {
    for (auto& input : op->inputs())
      input_arrays.push_back(_preserved_data[input->id()]);
  }

  HT_LOG_DEBUG << op << " inputs: " << input_arrays;
  op->LoadAndSaveCtxForBackward(_runtime_ctxs);
  auto output_arrays = op->Compute(input_arrays, _runtime_ctxs);
  auto profiler_optional = hetu::impl::Profile::get_cur_profile();
  if (profiler_optional) {
    auto profiler = *profiler_optional;
    profiler->push(op);
  }
  // Note: The usage should be marked inside kernels, 
  // but we still mark here in case we forget to do so in some kernels. 
  NDArray::MarkUsedBy(input_arrays, op->instantiation_ctx().stream());
  NDArray::MarkUsedBy(output_arrays, op->instantiation_ctx().stream());
  for (size_t i = 0; i < op->num_outputs(); i++)
    _preserved_data[op->output(i)->id()] = output_arrays[i];
  HT_LOG_DEBUG << op << " outputs: " << output_arrays;
  return _op_indexing[op->id()];
}

void EagerGraph::ResetVariableDataInner(const Tensor& tensor,
                                        const Initializer& init) {
  init.Init(GetVariableDataInner(tensor));
}

NDArray& EagerGraph::GetVariableDataInner(const Tensor& tensor) {
  auto it = _preserved_data.find(tensor->id());
  HT_RUNTIME_ERROR_IF(it == _preserved_data.end())
    << "Cannot find data for variable tensor " << tensor;
  return it->second;
}

NDArray& EagerGraph::AllocVariableDataInner(const Tensor& tensor,
                                            const Initializer& init,
                                            uint64_t seed, 
                                            const HTShape& global_shape) {
  // TODO: check meta is valid & maybe we can use non-blocking stream?
  _preserved_data[tensor->id()] = NDArray::empty(
    tensor->shape(), tensor->placement(), tensor->dtype(), kBlockingStream);
  if (!init.vodify()) {
    init.Init(_preserved_data[tensor->id()], seed, global_shape,
              kBlockingStream);
  }
  return _preserved_data[tensor->id()];
}

void EagerGraph::RegisterVariableDataInner(const Tensor& tensor, NDArray data,
                                           const Initializer& init) {
  _preserved_data[tensor->id()] = std::move(data);
  if (!init.vodify()) {
    init.Init(_preserved_data[tensor->id()]);
  }
}

void EagerGraph::RemoveTensor(const Tensor& tensor) {
  auto& producer = _op_indexing[tensor->producer_id()];
  size_t num_outputs = MAX(producer->num_outputs(), 1);
  if (num_outputs ==
      (++_op_to_num_destructed_outputs[tensor->producer_id()])) {
    RemoveOp(producer);
  }
}

NDArray EagerGraph::GetOrCompute(Tensor& tensor) {
  // This function should only be called from Tensor.
  // So we do not need to check the existence.
  tensor->producer()->Sync();
  return _preserved_data[tensor->id()];
}

NDArrayList EagerGraph::Run(const TensorList& fetches,
                            const FeedDict&) {
  NDArrayList ret;
  ret.reserve(fetches.size());
  std::unordered_set<OpId> to_sync_op_ids;
  to_sync_op_ids.reserve(fetches.size());
  for (auto& fetch : fetches) {
    auto it = _preserved_data.find(fetch->id());
    HT_VALUE_ERROR_IF(it == _preserved_data.end())
      << "Tensor " << fetch->name() << " cannot be found in graph " << id();
    ret.push_back(it->second);
    to_sync_op_ids.insert(fetch->producer_id());
  }
  for (auto op_id : to_sync_op_ids)
    _op_indexing[op_id]->Sync();
  return ret;
}

} // namespace graph
} // namespace hetu
