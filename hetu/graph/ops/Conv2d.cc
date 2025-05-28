#include "hetu/graph/ops/Conv2d.h"
#include "hetu/graph/ops/Reduce.h"
#include "hetu/graph/headers.h"
#include "hetu/graph/ops/kernel_links.h"

namespace hetu {
namespace graph {

DistributedStates conv2d_deduce_states(
  std::unordered_map<int32_t, int32_t>& l2res_map, 
  std::unordered_map<int32_t, int32_t>& r2res_map, 
  const DistributedStates& ds_l, const DistributedStates& ds_r) {
  int32_t device_num = ds_l.get_device_num();
  // deduce states
  std::unordered_map<int32_t, int32_t> res_states;
  auto update_res_states = [&](std::unordered_map<int32_t, int32_t>& res_states, 
    const DistributedStates& ds, std::unordered_map<int32_t, int32_t>& res_map) {
    for (auto d : {0, 1}) {
      auto splits = ds.get_dim(d);
      auto new_d = res_map[d];
      HT_ASSERT(res_states.find(new_d) == res_states.end() || res_states[new_d] == splits)
        << "Result states deduced from tensor input should be equal to filter";
      res_states[new_d] = splits;
    }
  };
  update_res_states(res_states, ds_l, l2res_map);
  update_res_states(res_states, ds_r, r2res_map);
  int32_t total_splits = 1;
  for (auto& state : res_states) {
    total_splits *= state.second;
  }
  res_states[-1] = device_num / total_splits;
  // deduce order
  std::vector<int32_t> lorder = ds_l.get_order();
  std::vector<int32_t> rorder = ds_r.get_order();
  auto get_new_order = [](std::unordered_map<int32_t, int32_t>& _map,
  std::vector<int32_t>& _order) -> std::vector<int32_t> {
    std::vector<int32_t> new_order;
    for (int32_t x : _order) {
      new_order.push_back(_map[x]);
    }
    return new_order;
  };
  auto get_index = [](std::vector<int32_t>& _order, int32_t val) -> int32_t {
    auto it = std::find(_order.begin(), _order.end(), val);
    HT_ASSERT(it != _order.end()) << "dimension " << val << " is not in order!";
    return it - _order.begin();
  };
  auto new_lorder = get_new_order(l2res_map, lorder);
  auto new_rorder = get_new_order(r2res_map, rorder);
  // few cases
  if (new_lorder != new_rorder) {
    new_lorder[get_index(new_lorder, 1)] = -1;
    new_rorder[get_index(new_rorder, 0)] = -1;
    HT_ASSERT(new_lorder == new_rorder) << "new_lorder is not equal to new_rorder!";
  } else if (std::find(new_lorder.begin(), new_lorder.end(), 0) != new_lorder.end()
             && ds_l.get_dim(-1) > 1) {
    int32_t ind0 = get_index(new_lorder, 0);
    int32_t ind1 = get_index(new_lorder, 1);
    if (ind0 > ind1) {
      int32_t tmp = ind0;
      ind0 = ind1;
      ind1 = tmp;
    }
    HT_ASSERT(ind0 + 1 == ind1) << "ind0 + 1 != ind1";
    new_lorder.insert(new_lorder.begin() + ind1, -1);
  }
  std::vector<int32_t> res_order(new_lorder);
  // distributed states for output
  DistributedStates ds_result({device_num, res_states, res_order});
  return std::move(ds_result);  
}

void Conv2dOpImpl::DoCompute(Operator&op,
                             const NDArrayList& inputs, NDArrayList& outputs,
                             RuntimeContext& ctx) const {
  NDArray::conv2d(inputs.at(0), inputs.at(1), get_padding(), get_stride(),
                  op->instantiation_ctx().stream_index, outputs.at(0));
}

TensorList Conv2dOpImpl::DoGradient(Operator&op,
                                    const TensorList& grad_outputs) const {
  auto g_op_meta = op->grad_op_meta();
  auto grad_input = op->requires_grad(0) ? MakeConv2dGradientofDataOp(
                                          op->input(1), grad_outputs.at(0), get_padding(),
                                          get_stride(), g_op_meta.set_name(op->grad_name(0)))
                                        : Tensor();
  auto grad_filter = op->requires_grad(1) ? MakeConv2dGradientofFilterOp(op->input(0), grad_outputs.at(0),
                                           get_padding(), get_stride(),
                                           g_op_meta.set_name(op->grad_name(1)))
                                         : Tensor();
  return {grad_input, grad_filter};
}

HTShapeList Conv2dOpImpl::DoInferShape(Operator&op,
                                       const HTShapeList& input_shapes,
                                       RuntimeContext& ctx) const {
  int64_t N = input_shapes.at(0)[0];
  int64_t H = input_shapes.at(0)[2];
  int64_t W = input_shapes.at(0)[3];
  int64_t f_O = input_shapes.at(1)[0];
  int64_t f_H = input_shapes.at(1)[2];
  int64_t f_W = input_shapes.at(1)[3];
  HTShape padding = get_padding();
  HTShape stride = get_stride();
  int64_t out_H = (H + 2 * padding[0] - f_H) / stride[0] + 1;
  int64_t out_W = (W + 2 * padding[1] - f_W) / stride[1] + 1;
  return {{N, f_O, out_H, out_W}};
}

void Conv2dOpImpl::DoSaveCtxForBackward(const TensorList& inputs, ContextStore& dst_ctx) const {
  dst_ctx.put("in_meta_0", inputs.at(0)->meta());
  dst_ctx.put("in_tensor_0", inputs.at(0));
  dst_ctx.put("in_meta_1", inputs.at(1)->meta());
  dst_ctx.put("in_tensor_1", inputs.at(1));
}

void Conv2dOpImpl::DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                                  const OpMeta& op_meta,
                                  const InstantiationContext& inst_ctx) const {
  const DistributedStates& ds_input = inputs.at(0)->get_distributed_states();
  const DistributedStates& ds_filter = inputs.at(1)->get_distributed_states();
  int32_t device_num = ds_input.get_device_num();
  HT_ASSERT(ds_input.is_valid() && ds_filter.is_valid() && 
            ds_input.get_device_num() == ds_filter.get_device_num()) 
    << "Conv2dOpDef: distributed states for input and filter must be valid!";
  HT_ASSERT(ds_input.get_dim(-2) == 1 && ds_filter.get_dim(-2) == 1) 
    << "Tensor input and filter shouldn't be partial";
  HT_ASSERT(ds_input.check_max_dim(2) && ds_filter.check_max_dim(2))
    << "Conv2d input & filter cannot split in H & W dimension!";
  HT_ASSERT(ds_input.get_dim(1) == ds_filter.get_dim(1))
    << "Split states in dimension C for input and in dimension C_in for filter should be equal!";

  std::unordered_map<int32_t, int32_t> l2res_map = {{-1, 1}, {0, 0}, {1, -2}};
  std::unordered_map<int32_t, int32_t> r2res_map = {{-1, 0}, {0, 1}, {1, -2}};
  outputs.at(0)->set_distributed_states(conv2d_deduce_states(l2res_map, r2res_map, ds_input, ds_filter));
}

void Conv2dGradientofFilterOpImpl::DoCompute(Operator& op,
                                             const NDArrayList& inputs,
                                             NDArrayList& outputs,
                                             RuntimeContext& ctx) const {
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(
    op->instantiation_ctx().placement.type(), type(), hetu::impl::Conv2dGradientofFilter,
    inputs.at(0), inputs.at(1), outputs.at(0), get_padding()[0],
    get_padding()[1], get_stride()[0], get_stride()[1], op->instantiation_ctx().stream());
}

HTShapeList
Conv2dGradientofFilterOpImpl::DoInferShape(Operator&op,
                                           const HTShapeList& input_shapes,
                                           RuntimeContext& ctx) const {
  return {ctx.get_or_create(op->id()).get<Tensor>("in_tensor")->temp_shape()};
}

void Conv2dGradientofFilterOpImpl::DoLoadCtxForBackward(ContextStore& src_ctx, ContextStore& dst_ctx) const {
  dst_ctx.migrate_from<NDArrayMeta>(src_ctx, "in_meta_1", "in_meta");
  dst_ctx.migrate_from<Tensor>(src_ctx, "in_tensor_1", "in_tensor");
}

void Conv2dGradientofFilterOpImpl::DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                                                  const OpMeta& op_meta,
                                                  const InstantiationContext& inst_ctx) const {
  const DistributedStates& ds_input = inputs.at(0)->get_distributed_states();
  const DistributedStates& ds_grad_output = inputs.at(1)->get_distributed_states();  
  const DistributedStates& ds_filter = inst_ctx.get<Tensor>("in_tensor")->get_distributed_states();
  HT_ASSERT(ds_input.is_valid() && ds_grad_output.is_valid() && ds_filter.is_valid()) 
    << "ConcatGradientOpDef: distributed states for input and grad_output and filter must be valid!";  
  HT_ASSERT(ds_input.get_dim(-2) == 1 && ds_grad_output.get_dim(-2) == 1 && ds_filter.get_dim(-2) == 1) 
    << "Tensor input and grad_output and filter shouldn't be partial";

  std::unordered_map<int32_t, int32_t> l2res_map = {{-1, 0}, {0, -2}, {1, 1}};
  std::unordered_map<int32_t, int32_t> r2res_map = {{-1, 1}, {0, -2}, {1, 0}};
  auto ds_filter_grad = conv2d_deduce_states(l2res_map, r2res_map, ds_input, ds_grad_output);
  HT_ASSERT(ds_filter.check_equal(ds_filter_grad)) 
    << "Distributed states for filter_grad should be equal to filter!";
  outputs.at(0)->set_distributed_states(ds_filter_grad);
}

void Conv2dGradientofDataOpImpl::DoCompute(Operator& op,
                                           const NDArrayList& inputs,
                                           NDArrayList& outputs,
                                           RuntimeContext& ctx) const {
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(
    op->instantiation_ctx().placement.type(), type(), hetu::impl::Conv2dGradientofData, inputs.at(0),
    inputs.at(1), outputs.at(0), get_padding()[0], get_padding()[1],
    get_stride()[0], get_stride()[1], op->instantiation_ctx().stream());
}

HTShapeList
Conv2dGradientofDataOpImpl::DoInferShape(Operator&op,
                                         const HTShapeList& input_shapes,
                                         RuntimeContext& ctx) const {
  return {ctx.get_or_create(op->id()).get<Tensor>("in_tensor")->temp_shape()};
}

void Conv2dGradientofDataOpImpl::DoLoadCtxForBackward(ContextStore& src_ctx, ContextStore& dst_ctx) const {
  dst_ctx.migrate_from<NDArrayMeta>(src_ctx, "in_meta_0", "in_meta");
  dst_ctx.migrate_from<Tensor>(src_ctx, "in_tensor_0", "in_tensor");
}

void Conv2dGradientofDataOpImpl::DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                                                const OpMeta& op_meta,
                                                const InstantiationContext& inst_ctx) const {
  const DistributedStates& ds_filter = inputs.at(0)->get_distributed_states();
  const DistributedStates& ds_grad_output = inputs.at(1)->get_distributed_states();  
  const DistributedStates& ds_input = inst_ctx.get<Tensor>("in_tensor")->get_distributed_states();
  HT_ASSERT(ds_input.is_valid() && ds_grad_output.is_valid() && ds_filter.is_valid()) 
    << "ConcatGradientOpDef: distributed states for input and grad_output and filter must be valid!";  
  HT_ASSERT(ds_input.get_dim(-2) == 1 && ds_grad_output.get_dim(-2) == 1 && ds_filter.get_dim(-2) == 1) 
    << "Tensor input and grad_output and filter shouldn't be partial";

  std::unordered_map<int32_t, int32_t> l2res_map = {{-1, 0}, {0, -2}, {1, 1}};
  std::unordered_map<int32_t, int32_t> r2res_map = {{-1, 1}, {0, 0}, {1, -2}};
  auto ds_input_grad = conv2d_deduce_states(l2res_map, r2res_map, ds_filter, ds_grad_output);
  HT_ASSERT(ds_input.check_equal(ds_input_grad))
    << "Distributed states for input_grad should be equal to input!";
  outputs.at(0)->set_distributed_states(ds_input_grad);
}

void Conv2dAddBiasOpImpl::DoCompute(Operator& op,
                                    const NDArrayList& inputs,
                                    NDArrayList& outputs, RuntimeContext& ctx) const {
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(
    op->instantiation_ctx().placement.type(), type(), hetu::impl::Conv2dAddBias, inputs.at(0),
    inputs.at(1), inputs.at(2), outputs.at(0), get_padding()[0],
    get_padding()[1], get_stride()[0], get_stride()[1], op->instantiation_ctx().stream());
  
  // auto input_0 = NDArray::to(inputs.at(0), inputs.at(0)->device(), DataType::FLOAT32, kBlockingStream);
  // auto input_1 = NDArray::to(inputs.at(1), inputs.at(1)->device(), DataType::FLOAT32, kBlockingStream);
  // auto output_0 = NDArray::conv2d(input_0, input_1, get_padding(), get_stride(), kBlockingStream);
  // HT_LOG_INFO << "F32:" << output_0;
}

TensorList Conv2dAddBiasOpImpl::DoGradient(Operator& op,
                                           const TensorList& grad_outputs) const {
  auto g_op_meta = op->grad_op_meta();
  auto grad_input = op->requires_grad(0) ? MakeConv2dGradientofDataOp(
                                           op->input(1), grad_outputs.at(0), get_padding(),
                                           get_stride(), g_op_meta.set_name(op->grad_name(0)))
                                          : Tensor();
  auto grad_filter = op->requires_grad(1) ? MakeConv2dGradientofFilterOp(op->input(0), grad_outputs.at(0),
                                            get_padding(), get_stride(),
                                            g_op_meta.set_name(op->grad_name(1)))
                                          : Tensor();
  auto grad_bias = op->requires_grad(2) ? MakeReduceOp(grad_outputs.at(0), ReductionType::SUM, {0, 2, 3}, {false},
                                          g_op_meta.set_name(op->grad_name(2)))
                                        : Tensor();
  return {grad_input, grad_filter, grad_bias};
}

HTShapeList Conv2dAddBiasOpImpl::DoInferShape(Operator& op,
                                              const HTShapeList& input_shapes,
                                              RuntimeContext& ctx) const {
  int64_t N = input_shapes.at(0)[0];
  int64_t H = input_shapes.at(0)[2];
  int64_t W = input_shapes.at(0)[3];
  int64_t f_O = input_shapes.at(1)[0];
  int64_t f_H = input_shapes.at(1)[2];
  int64_t f_W = input_shapes.at(1)[3];
  HTShape padding = get_padding();
  HTShape stride = get_stride();
  int64_t out_H = (H + 2 * padding[0] - f_H) / stride[0] + 1;
  int64_t out_W = (W + 2 * padding[1] - f_W) / stride[1] + 1;
  return {{N, f_O, out_H, out_W}};
}

void Conv2dAddBiasOpImpl::DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                                         const OpMeta& op_meta,
                                         const InstantiationContext& inst_ctx) const {
  const DistributedStates& ds_input = inputs.at(0)->get_distributed_states();
  const DistributedStates& ds_filter = inputs.at(1)->get_distributed_states();
  const DistributedStates& ds_bias = inputs.at(2)->get_distributed_states();  
  HT_ASSERT(ds_input.is_valid() && ds_filter.is_valid() && ds_bias.is_valid() &&
            ds_input.get_device_num() == ds_filter.get_device_num() && 
            ds_filter.get_device_num() == ds_bias.get_device_num()) 
    << "Conv2dOpDef: distributed states for input and filter and bias must be valid!";
  HT_ASSERT(ds_input.get_dim(-2) == 1 && ds_filter.get_dim(-2) == 1 && ds_bias.get_dim(-2) == 1) 
    << "Tensor input and filter and bias shouldn't be partial!";
  HT_ASSERT(ds_input.check_equal(ds_bias))
    << "Distributed states of bias should be equal to input!";

  outputs.at(0)->set_distributed_states(ds_input);  
}

Tensor MakeConv2dOp(Tensor input, Tensor filter, int64_t padding, int64_t stride,
                    OpMeta op_meta) {
  TensorList inputs = {std::move(input), std::move(filter)};
  return Graph::MakeOp(
          std::make_shared<Conv2dOpImpl>(padding, stride),
          std::move(inputs),
          std::move(op_meta))->output(0);
}

Tensor MakeConv2dGradientofFilterOp(Tensor input, Tensor grad_output,
                                    const HTShape& padding, const HTStride& stride,
                                    OpMeta op_meta) {
  return Graph::MakeOp(
          std::make_shared<Conv2dGradientofFilterOpImpl>(padding, stride),
          {std::move(input), std::move(grad_output)},
          std::move(op_meta))->output(0);
}

Tensor MakeConv2dGradientofDataOp(Tensor filter, Tensor grad_output,
                                  const HTShape& padding, const HTStride& stride,
                                  OpMeta op_meta) {
  return Graph::MakeOp(
          std::make_shared<Conv2dGradientofDataOpImpl>(padding, stride),
          {std::move(filter), std::move(grad_output)},
          std::move(op_meta))->output(0);
}

Tensor MakeConv2dAddBiasOp(Tensor input, Tensor filter, Tensor bias, int64_t padding,
                           int64_t stride, OpMeta op_meta) {
  TensorList inputs = {std::move(input), std::move(filter), std::move(bias)};
  return Graph::MakeOp(
          std::make_shared<Conv2dAddBiasOpImpl>(padding, stride),
          std::move(inputs),
          std::move(op_meta))->output(0);
}

} // namespace graph
} // namespace hetu
