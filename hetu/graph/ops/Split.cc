#include "hetu/graph/ops/Split.h"
#include "hetu/graph/ops/Slice.h"
#include "hetu/graph/headers.h"
#include "hetu/graph/ops/kernel_links.h"
#include "hetu/core/symbol.h"
#include <numeric>

namespace hetu {
namespace graph {

NDArrayList SplitOpImpl::DoCompute(Operator& op,
                                   const NDArrayList& inputs,
                                   RuntimeContext& ctx) const {
  NDArrayList outputs = {};
  auto split_num = get_split_num();
  // if split_num is 0, then the split_num is the number of task
  if (split_num == 0) {
    split_num = inputs.size() - 1;
    HTShapeList input_shapes;
    for (auto& input : inputs) {
      input_shapes.emplace_back(input->shape());
    }
    auto multi_task_begin_pos = get_multi_task_begin_pos(input_shapes);
    auto multi_task_output_shape = get_multi_task_output_shape(input_shapes);
    for (int i = 0; i < split_num; i++) {
      outputs.emplace_back(NDArray::slice(inputs.at(0), multi_task_begin_pos[i], multi_task_output_shape[i],
                          op->instantiation_ctx().stream_index));
    }
  } else {
    for (int i = 0; i < get_split_num(); i++) {
      outputs.emplace_back(NDArray::slice(inputs.at(0), get_begin_pos(i), get_output_shape(i),
                          op->instantiation_ctx().stream_index));
    }
  }
  return std::move(outputs);
}

TensorList SplitOpImpl::DoGradient(Operator& op, const TensorList& grad_outputs) const {
  auto split_num = get_split_num();
  if (split_num == 0) {
    auto task_num = grad_outputs.size();
    TensorList grad_task_batch_idxs(task_num, Tensor());
    TensorList task_batch_idxs = {};
    for (int i = 1; i <= task_num; i++) {
      task_batch_idxs.emplace_back(op->input(i));
    }
    TensorList grad_inputs = {};
    if (symbolic()) {
      grad_inputs = {op->requires_grad(0) ? MakeSplitGradientOp(grad_outputs, op->input(0), std::move(task_batch_idxs),
                                                                _begin_pos_list, _output_shape_list,
                                                                dim(), op->grad_op_meta().set_name(op->grad_name()))
                                          : Tensor()};
    } else {
      grad_inputs = {op->requires_grad(0) ? MakeSplitGradientOp(grad_outputs, op->input(0), std::move(task_batch_idxs),
                                                                get_begin_pos_list(), get_output_shape_list(),
                                                                dim(), op->grad_op_meta().set_name(op->grad_name()))
                                           : Tensor()};
    }
    grad_inputs.insert(grad_inputs.end(), grad_task_batch_idxs.begin(), grad_task_batch_idxs.end());
    return std::move(grad_inputs);
  } else {
    if (symbolic()) {
      return {op->requires_grad(0) ? MakeSplitGradientOp(grad_outputs, op->input(0),
                                                        _begin_pos_list, _output_shape_list,
                                                        op->grad_op_meta().set_name(op->grad_name()))
                                  : Tensor()};
    } else {
      return {op->requires_grad(0) ? MakeSplitGradientOp(grad_outputs, op->input(0), get_begin_pos_list(),
                                                        get_output_shape_list(),
                                                        op->grad_op_meta().set_name(op->grad_name()))
                                  : Tensor()};
    }
  }
}

HTShapeList SplitOpImpl::DoInferShape(Operator& op,
                                      const HTShapeList& input_shapes,
                                      RuntimeContext& ctx) const {
  HTShapeList ret;
  auto split_num = get_split_num();
  if (split_num == 0) {
    split_num = input_shapes.size() - 1;
    ret = get_multi_task_output_shape(input_shapes);
  }
  else {
    for (int i = 0; i < split_num; i++) {
      ret.emplace_back(get_output_shape(i));
    }
  }
  return std::move(ret);
}

void SplitOpImpl::DoDeduceStates(const TensorList& inputs, TensorList& outputs,
                                 const OpMeta& op_meta) const {
  const DistributedStates& ds_input = inputs.at(0)->get_distributed_states();
  HT_ASSERT(ds_input.is_valid()) 
    << "SliceOpDef: distributed states for input must be valid!";
  HT_ASSERT(ds_input.get_dim(-2) == 1)
    << "Input tensor shouldn't be partial!";
  HTShape ori_shape = inputs.at(0)->shape();
  int ndim = ori_shape.size();
  auto split_num = get_split_num();
  if (split_num == 0) {
    split_num = inputs.size() - 1;
    HTShapeList input_shapes;
    for (auto& input : inputs) {
      input_shapes.emplace_back(input->shape());
    }
    auto multi_task_begin_pos = get_multi_task_begin_pos(input_shapes);
    auto multi_task_output_shape = get_multi_task_output_shape(input_shapes);
    for (int i = 0; i < split_num; i++) {
      for (int j = 0; j < ndim; j++) {
        if (!(multi_task_begin_pos[i][j] == 0 && multi_task_begin_pos[i][j] + multi_task_output_shape[i][j] == ori_shape[j])) {
          HT_ASSERT(ds_input.get_dim(j) == 1)
            << "Slice dimension " << j << " shouldn't be splited!"; 
        }
      }
      outputs.at(i)->set_distributed_states(ds_input);
    }
  } else {
    for (int i = 0; i < split_num; i++) {
      const HTShape output_shape = get_output_shape(i);
      const HTShape begin_pos = get_begin_pos(i);
      for (int j = 0; j < ndim; j++) {
        if (!(begin_pos[j] == 0 && begin_pos[j] + output_shape[j] == ori_shape[j])) {
          HT_ASSERT(ds_input.get_dim(j) == 1)
            << "Slice dimension " << j << " shouldn't be splited!"; 
        }
      }
      outputs.at(i)->set_distributed_states(ds_input);
    }
  }
}

void SplitOpImpl::DoDeduceHeterProp(const std::vector<int32_t>& inputs_hetero_dim,
                                    TensorList& outputs, const OpMeta& op_meta) const {
  auto split_num = get_split_num();
  if (split_num == 0) {
    split_num = outputs.size();
    for (int i = 0; i < split_num; i++) {
      outputs.at(i)->cur_ds_union().set_hetero_dim(inputs_hetero_dim.at(0));
    }
  } else {
    for (int i = 0; i < split_num; i++) {
      outputs.at(i)->cur_ds_union().set_hetero_dim(inputs_hetero_dim.at(0));
    }
  }
}

NDArrayList SplitGradientOpImpl::DoCompute(Operator& op,
                                           const NDArrayList& inputs,
                                           RuntimeContext& ctx) const {
  NDArrayList outputs = {};
  auto stream_idx = op->instantiation_ctx().stream_index;
  auto split_num = get_split_num();
  if (split_num == 0) {
    auto dim_ = dim();
    split_num = inputs.size() / 2;
    HTShapeList input_shapes = {inputs.back()->shape()};
    for (int i = 0; i < split_num; i++) {
      input_shapes.emplace_back(inputs[i + split_num]->shape());
    }
    auto multi_task_begin_pos = get_multi_task_begin_pos(input_shapes);
    auto multi_task_output_shape = get_multi_task_output_shape(input_shapes);
    auto task_num = std::accumulate(multi_task_output_shape.begin(), multi_task_output_shape.end(), 0,
                                    [dim_](int sum, const HTShape& shape) {
                                      return sum + int(shape[dim_] > 0);
                                    });
    if (task_num == 1) {
      auto idx = std::distance(multi_task_output_shape.begin(),
                               std::find_if(multi_task_output_shape.begin(), multi_task_output_shape.end(),
                                  [dim_](const HTShape& shape) {
                                    return shape[dim_] > 0;
                                  }));
      outputs = {std::move(inputs.at(idx))};
    } else {
      outputs = DoAllocOutputs(op, inputs, ctx);
      for (int i = 0; i < split_num; i++) {
        auto slice_grad_input = NDArray::slice(outputs.at(0), multi_task_begin_pos[i],
                                             multi_task_output_shape[i], stream_idx);
        NDArray::copy(inputs.at(i), stream_idx, slice_grad_input);
      }
    }
  } else {
    outputs = DoAllocOutputs(op, inputs, ctx);
    for (int i = 0; i < split_num; i++) {
      auto slice_grad_input = NDArray::slice(outputs.at(0), get_begin_pos(i),
                                             get_output_shape(i), stream_idx);
      NDArray::copy(inputs.at(i), stream_idx, slice_grad_input);
    }
  }
  return outputs;
}

void SplitGradientOpImpl::DoCompute(Operator& op, const NDArrayList& inputs,
                                    NDArrayList& outputs, RuntimeContext& ctx) const {
  auto stream_idx = op->instantiation_ctx().stream_index;
  auto split_num = get_split_num();
  if (split_num == 0) {
    split_num = inputs.size() / 2;
    HTShapeList input_shapes = {inputs.back()->shape()};
    for (int i = 0; i < split_num; i++) {
      input_shapes.emplace_back(inputs[i + split_num]->shape());
    }
    auto multi_task_begin_pos = get_multi_task_begin_pos(input_shapes);
    auto multi_task_output_shape = get_multi_task_output_shape(input_shapes);
    for (int i = 0; i < split_num; i++) {
      auto slice_grad_input = NDArray::slice(outputs.at(0), multi_task_begin_pos[i],
                                             multi_task_output_shape[i], stream_idx);
      NDArray::copy(inputs.at(i), stream_idx, slice_grad_input);
    }
  } else {
    for (int i = 0; i < split_num; i++) {
      auto slice_grad_input = NDArray::slice(outputs.at(0), get_begin_pos(i),
                                            get_output_shape(i), stream_idx);
      NDArray::copy(inputs.at(i), stream_idx, slice_grad_input);
    }
  }
}

HTShapeList SplitGradientOpImpl::DoInferShape(Operator& op, 
                                              const HTShapeList& input_shapes, 
                                              RuntimeContext& ctx) const {
  return {input_shapes.back()}; 
}

void SplitGradientOpImpl::DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                                         const OpMeta& op_meta) const {
  outputs.at(0)->set_distributed_states(inputs.at(0)->get_distributed_states());  
}

void SplitGradientOpImpl::DoDeduceHeterProp(const std::vector<int32_t>& inputs_hetero_dim,
                                            TensorList& outputs, const OpMeta& op_meta) const {
  outputs.at(0)->cur_ds_union().set_hetero_dim(inputs_hetero_dim.at(0));
}

// need symbolic shape
Tensor MakeSplitOp(Tensor input, const HTAxes& axes, const HTShape& indices,
                   const HTShape& splits, OpMeta op_meta) {
  HT_ASSERT(input->has_shape());
  // get begin_pos, output_shape
  HT_ASSERT(axes.size() == splits.size());
  int len = axes.size();
  for (int i = 0; i < len; ++i) {
    HT_ASSERT(axes[i] >= 0);
    HT_ASSERT(splits[i] >= 0);
    HT_ASSERT(indices[i] >= 0 && indices[i] < splits[i]);
  }

  // Split算子在make时，要将输入的tensor设置成symbolic的，之后shape发生改变时，
  // 直接overwrite该tensor中的symbolic shape的value即可，
  if (!input->symbolic()) {
    input->init_symbolic_shape(); // leaf
  }
  const SyShape& ori_shape = input->symbolic_shape(); 

  int ndim = ori_shape.size();
  SyShape begin_pos(ndim, 0);
  SyShape output_shape(ori_shape);

  for (int i = 0; i < len; ++i) {
    auto axe = axes[i];
    auto part_size = ori_shape[axe] / splits[i];
    begin_pos[axe] = part_size * indices[i];
    if (indices[i] != splits[i] - 1) {
      output_shape[axe] = part_size;
    } else {
      output_shape[axe] = ori_shape[axe] - begin_pos[axe];
    }
  }

  // 将输出的tensor设置成symbolic的
  auto output = Graph::MakeOp(std::make_shared<SliceOpImpl>(std::move(begin_pos), std::move(output_shape)),
                      {std::move(input)}, std::move(op_meta))->output(0);
  // output->copy_symbolic_shape(std::move(output_shape)); // not leaf
  return output;
}

// need symbolic shape
Tensor MakeSplitOp(Tensor input, const HTShape& indices,
                   const HTShape& splits, OpMeta op_meta) {
  HT_ASSERT(input->has_shape());
  // get begin_pos, output_shape
  auto len = indices.size();
  for (int i = 0; i < len; ++i) {
    HT_ASSERT(splits[i] >= 0);
    HT_ASSERT(indices[i] >= 0 && indices[i] < splits[i]);
  }

  if (!input->symbolic()) {
    input->init_symbolic_shape(); // leaf
  }
  const SyShape& ori_shape = input->symbolic_shape(); 
  HT_ASSERT(len == ori_shape.size()) << "size should be equal";
  SyShape begin_pos(len, 0);
  SyShape output_shape(len);

  for (int i = 0; i < len; ++i) {
    auto part_size = ori_shape[i] / splits[i];
    begin_pos[i] = part_size * indices[i];
    if (indices[i] != splits[i] - 1) {
      output_shape[i] = part_size;
    } else {
      output_shape[i] = ori_shape[i] - begin_pos[i];
    }
  }

  auto output = Graph::MakeOp(std::make_shared<SliceOpImpl>(std::move(begin_pos), std::move(output_shape)),
                      {std::move(input)}, std::move(op_meta))->output(0);
  return output;
}

// 这里只能做到在单一的dim上的切分
TensorList MakeSplitOp(Tensor input, int64_t num_chunks, int64_t dim,
                       OpMeta op_meta) {
  HT_ASSERT(input->has_shape());
  dim = NDArrayMeta::ParseAxis(dim, input->ndim());

  if (!input->symbolic()) {
    input->init_symbolic_shape(); // leaf
  }
  const SyShape& ori_shape = input->symbolic_shape(); 

  auto chunk_size = ori_shape[dim] / num_chunks;
  auto chunk_sum = IntSymbol(0);

  SyShapeList begin_pos_list = {};
  SyShapeList output_shape_list = {};
  for (int i = 0; i < num_chunks; ++i) {
    SyShape begin_pos(input->ndim(), 0);
    SyShape output_shape(ori_shape);
    output_shape[dim] = i == num_chunks - 1 ? (ori_shape[dim] - 1) % chunk_size + 1
                                            : chunk_size;
    begin_pos[dim] = chunk_sum;
    chunk_sum = chunk_sum + chunk_size;
    begin_pos_list.emplace_back(begin_pos);
    output_shape_list.emplace_back(output_shape);
  }
  return Graph::MakeOp(std::make_shared<SplitOpImpl>(std::move(begin_pos_list), std::move(output_shape_list)),
                       {input}, std::move(op_meta))->outputs();
}

TensorList MakeSplitOp(Tensor input, TensorList task_batch_idxs, int64_t dim,
                       OpMeta op_meta) {
  HT_ASSERT(input->has_shape());
  dim = NDArrayMeta::ParseAxis(dim, input->ndim());
  SyShapeList begin_pos_list = {};
  SyShapeList output_shape_list = {};
  TensorList inputs = {input};
  inputs.insert(inputs.end(), task_batch_idxs.begin(), task_batch_idxs.end());
  return Graph::MakeOp(std::make_shared<SplitOpImpl>(begin_pos_list, output_shape_list, dim),
                       std::move(inputs), std::move(op_meta))->outputs();
}

Tensor MakeSplitGradientOp(TensorList grad_outputs, Tensor ori_input,
                           SyShapeList begin_pos_list,
                           SyShapeList output_shape_list,
                           OpMeta op_meta) {
  grad_outputs.emplace_back(ori_input);
  return Graph::MakeOp(std::make_shared<SplitGradientOpImpl>(std::move(begin_pos_list), std::move(output_shape_list)),
                       std::move(grad_outputs), std::move(op_meta))->output(0);
}

Tensor MakeSplitGradientOp(TensorList grad_outputs, Tensor ori_input,
                           const HTShapeList& begin_pos_list,
                           const HTShapeList& output_shape_list,
                           OpMeta op_meta) {
  grad_outputs.emplace_back(ori_input);
  return Graph::MakeOp(std::make_shared<SplitGradientOpImpl>(begin_pos_list, output_shape_list),
                       std::move(grad_outputs), std::move(op_meta))->output(0);
}

Tensor MakeSplitGradientOp(TensorList grad_outputs, Tensor ori_input,
                           TensorList task_batch_idxs,
                           SyShapeList begin_pos_list,
                           SyShapeList output_shape_list,
                           int64_t dim,
                           OpMeta op_meta) {
  grad_outputs.insert(grad_outputs.end(), task_batch_idxs.begin(), task_batch_idxs.end());
  grad_outputs.emplace_back(ori_input);
  return Graph::MakeOp(std::make_shared<SplitGradientOpImpl>(std::move(begin_pos_list), std::move(output_shape_list), dim),
                       std::move(grad_outputs), std::move(op_meta))->output(0);
}

Tensor MakeSplitGradientOp(TensorList grad_outputs, Tensor ori_input,
                           TensorList task_batch_idxs,
                           const HTShapeList& begin_pos_list,
                           const HTShapeList& output_shape_list,
                           int64_t dim,
                           OpMeta op_meta) {
  grad_outputs.insert(grad_outputs.end(), task_batch_idxs.begin(), task_batch_idxs.end());
  grad_outputs.emplace_back(ori_input);
  return Graph::MakeOp(std::make_shared<SplitGradientOpImpl>(begin_pos_list, output_shape_list, dim),
                       std::move(grad_outputs), std::move(op_meta))->output(0);
}

// deprecated: only used in gpt inference, before symbolic shape is realized
TensorList MakeSplitOp(Tensor input, int64_t num_chunks, int64_t dim,
                       int64_t padding_axis, OpMeta op_meta) {
  HT_RUNTIME_ERROR << "deprecated";
  HT_ASSERT(input->has_shape());
  dim = NDArrayMeta::ParseAxis(dim, input->ndim());
  padding_axis = NDArrayMeta::ParseAxis(padding_axis, input->ndim());
  HT_ASSERT(dim != padding_axis) << "Split dim can't be the padding dim.";
  int64_t chunk_sum = 0;
  chunk_sum = 0;
  int64_t chunk_size = DIVUP(input->shape(dim), num_chunks);
  HTShape begin_pos(input->ndim());
  HTShape output_shape = input->shape();
  TensorList outputs = {};
  for (int i = 0; i < num_chunks; ++i) {
    output_shape[dim] = i == num_chunks - 1 ? (input->shape(dim) - 1) % chunk_size + 1
                                            : chunk_size;
    begin_pos[dim] = chunk_sum;
    chunk_sum += chunk_size;
    outputs.emplace_back(Graph::MakeOp(
                         std::make_shared<SliceOpImpl>(begin_pos, output_shape, padding_axis),
                         {input}, op_meta)->output(0));
  }
  return std::move(outputs);
}

// seems deprecated
TensorList MakeSplitOp(Tensor input, const HTShape& chunks, int64_t dim,
                       OpMeta op_meta) {
  HT_RUNTIME_ERROR << "deprecated";
  HT_ASSERT(input->has_shape());
  dim = NDArrayMeta::ParseAxis(dim, input->ndim());
  int64_t chunk_sum = 0;
  int64_t len = chunks.size();
  for (int i = 0; i < len; ++i) {
    chunk_sum += chunks[i];
  }
  HT_ASSERT(chunk_sum == input->shape(dim));
  chunk_sum = 0;
  HTShape begin_pos(input->ndim());
  HTShape output_shape = input->shape();
  TensorList outputs = {};
  for (int i = 0; i < len; ++i) {
    output_shape[dim] = chunks[i];
    begin_pos[dim] = chunk_sum;
    chunk_sum += chunks[i];
    outputs.emplace_back(Graph::MakeOp(
                         std::make_shared<SliceOpImpl>(begin_pos, output_shape),
                         {input}, op_meta)->output(0));
  }
  return std::move(outputs);
}

} // namespace graph
} // namespace hetu
