#include "hetu/graph/ops/Broadcast.h"
#include "hetu/graph/headers.h"
#include "hetu/graph/ops/kernel_links.h"

namespace hetu {
namespace graph {

void BroadcastOpImpl::DoCompute(Operator& op,
                                const NDArrayList& inputs, NDArrayList& outputs,
                                RuntimeContext& ctx) const {
  if (mode() == 0) {
    NDArray::broadcast(inputs.at(0), outputs.at(0)->shape(), 
                       op->instantiation_ctx().stream_index, outputs.at(0));
  } else {
    NDArray::broadcast(inputs.at(0), outputs.at(0)->shape(), get_add_axes(), 
                       op->instantiation_ctx().stream_index, outputs.at(0));
  }
}

TensorList BroadcastOpImpl::DoGradient(Operator& op,
                                       const TensorList& grad_outputs) const {
  auto grad_input = op->requires_grad(0) ? MakeBroadcastGradientOp(grad_outputs.at(0), op->input(0), 
                                          op->output(0), get_add_axes(),
                                          op->grad_op_meta().set_name(op->grad_name()))
                                        : Tensor();
  if (mode() == 0)
    return {grad_input, Tensor()};
  else
    return {grad_input};
}

HTShapeList BroadcastOpImpl::DoInferShape(Operator& op,
                                          const HTShapeList& input_shapes,
                                          RuntimeContext& ctx) const {
  HTShapeList outputlist = {};
  if (mode() == 0) {
    HTShape output_shape = input_shapes.at(1);
    outputlist = {input_shapes.at(1)};
  } else {
    HTShape input_shape = input_shapes.at(0);
    HTShape output_shape = get_shape();
    outputlist = {output_shape};
  }
  return outputlist;
}

void BroadcastOpImpl::DoSaveCtxForBackward(const TensorList& inputs, ContextStore& dst_ctx) const {
  dst_ctx.put("in_meta", inputs.at(0)->meta());
}

void BroadcastOpImpl::DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                                     const OpMeta& op_meta,
                                     const InstantiationContext& inst_ctx) const {
  HT_LOG_INFO << op_meta.name << ": warning: deduce states for broadcast op was not checked! please use carefully!";
  OpInterface::DoDeduceStates(inputs, outputs, op_meta, inst_ctx);
}

void BroadcastGradientOpImpl::DoCompute(Operator& op,
                                        const NDArrayList& inputs,
                                        NDArrayList& outputs,
                                        RuntimeContext& ctx) const {
  NDArray::sum(inputs.at(0), get_axes(), false, 
               op->instantiation_ctx().stream_index, outputs.at(0));
}

HTShapeList
BroadcastGradientOpImpl::DoInferShape(Operator& op,
                                      const HTShapeList& input_shapes,
                                      RuntimeContext& ctx) const {
  return {ctx.get_or_create(op->id()).get<NDArrayMeta>("in_meta").shape};
}

void BroadcastGradientOpImpl::DoLoadCtxForBackward(ContextStore& src_ctx, ContextStore& dst_ctx) const {
  dst_ctx.migrate_from<NDArrayMeta>(src_ctx, "in_meta");
}

void BroadcastGradientOpImpl::DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                                             const OpMeta& op_meta,
                                             const InstantiationContext& inst_ctx) const {
  HT_LOG_INFO << op_meta.name << ": warning: deduce states for broadcast gradient op was not checked! please use carefully!";
  OpInterface::DoDeduceStates(inputs, outputs, op_meta, inst_ctx);
}

Tensor MakeBroadcastOp(Tensor input, Tensor output, OpMeta op_meta) {
  return Graph::MakeOp(
          std::make_shared<BroadcastOpImpl>(),
          {std::move(input), std::move(output)},
          std::move(op_meta))->output(0);
}

Tensor MakeBroadcastOp(Tensor input, const HTShape& shape,
                       const HTShape& add_axes,
                       OpMeta op_meta) {
  if (!add_axes.empty())
    HT_ASSERT(input->ndim() + add_axes.size() == shape.size())
    << "input dims plus add_axes is not equal to output dims."
    << "Input dims:" << input->ndim() << ".Add_axes:" << add_axes.size()
    << ".Output dims:" << shape.size();
  return Graph::MakeOp(
          std::make_shared<BroadcastOpImpl>(shape, add_axes),
          {std::move(input)},
          std::move(op_meta))->output(0);
}

Tensor MakeBroadcastGradientOp(Tensor input, Tensor ori_input, 
                               Tensor ori_output, const HTAxes& axes,
                               OpMeta op_meta) {
  HTShapeList outputlist = {};
  HTShape input_shape = ori_input->shape();
  HTShape output_shape = ori_output->shape();
  size_t ndim = output_shape.size();
  HT_ASSERT(input_shape.size() <= ndim);
  size_t diff = ndim - input_shape.size();
  HTAxes add_axes(diff);
  HTKeepDims keep_dims(diff);
  size_t len = diff + input_shape.size();
  HTShape n_input_shape(len);
  if (axes.empty()) {
    for (size_t i = 0; i < diff; ++i) {
      add_axes[i] = i;
      keep_dims[i] = false;
      n_input_shape[i] = 1;
    }
    for (size_t i = diff; i < len; ++i) {
      n_input_shape[i] = input_shape[i - diff];
    }
    for (size_t i = 0; i < ndim; ++i) {
      // HT_ASSERT(output_shape[i] > 0)
      // << "Invalid output shape:" << output_shape;
      HT_ASSERT(n_input_shape[i] == 1 || n_input_shape[i] == output_shape[i]);
      if (i >= diff && n_input_shape[i] == 1 && output_shape[i] > 1) {
        add_axes.emplace_back(i);
        keep_dims.emplace_back(true);
      }
    }
  }
  else {
    for (size_t i = 0; i < diff; ++i) {
      add_axes[i] = axes[i];
      keep_dims[i] = false;
      n_input_shape[axes[i]] = 1;
    }
    int idx = 0;
    for (size_t i = diff; i < len; ++i) {
      while(n_input_shape[idx] != 0)
        idx++;
      n_input_shape[idx] = input_shape[i - diff];
      HT_ASSERT(output_shape[idx] > 0);
      HT_ASSERT(n_input_shape[idx] == 1 || n_input_shape[idx] == output_shape[idx]);
      if (n_input_shape[idx] == 1 && output_shape[idx] > 1) {
        add_axes.emplace_back(idx);
        keep_dims.emplace_back(true);
      }
    }
  }
  return Graph::MakeOp(
          std::make_shared<BroadcastGradientOpImpl>(add_axes, keep_dims),
          {std::move(input)},
          std::move(op_meta))->output(0);
}

} // namespace graph
} // namespace hetu
