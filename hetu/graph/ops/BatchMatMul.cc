#include "hetu/graph/ops/BatchMatMul.h"
#include "hetu/graph/headers.h"
#include "hetu/graph/ops/kernel_links.h"

namespace hetu {
namespace graph {

void BatchMatMulOpImpl::DoCompute(Operator& op, 
                                  const NDArrayList& inputs, NDArrayList& outputs,
                                  RuntimeContext& runtime_ctx) const {
  NDArray::bmm(inputs.at(0), inputs.at(1), trans_a(), trans_b(),
               op->instantiation_ctx().stream_index, outputs.at(0));
}

TensorList BatchMatMulOpImpl::DoGradient(Operator& op,
                                         const TensorList& grad_outputs) const {
  const Tensor& grad_c = grad_outputs.at(0);
  Tensor& a = op->input(0);
  Tensor& b = op->input(1);
  Tensor grad_a;
  Tensor grad_b;
  auto g_op_meta = op->grad_op_meta();
  if (!trans_a() && !trans_b()) {
    // case 1: c = BatchMatMul(a, b)
    // grad_a = BatchMatMul(grad_c, b^T), grad_b = BatchMatMul(a^T, grad_c)
    grad_a = op->requires_grad(0) ? MakeBatchMatMulOp(grad_c, b, false, true, g_op_meta.set_name(op->grad_name(0)))
                                 : Tensor();
    grad_b = op->requires_grad(1) ? MakeBatchMatMulOp(a, grad_c, true, false, g_op_meta.set_name(op->grad_name(1)))
                                 : Tensor();
  } else if (trans_a() && !trans_b()) {
    // case 2: c = BatchMatMul(a^T, b)
    // grad_a = BatchMatMul(b, grad_c^T), grad_b = BatchMatMul(a, grad_c)
    grad_a = op->requires_grad(0) ? MakeBatchMatMulOp(b, grad_c, false, true, g_op_meta.set_name(op->grad_name(0)))
                                 : Tensor();
    grad_b = op->requires_grad(1) ? MakeBatchMatMulOp(a, grad_c, false, false, g_op_meta.set_name(op->grad_name(1)))
                                 : Tensor();
  } else if (!trans_a() && trans_b()) {
    // case 3: c = BatchMatMul(a, b^T)
    // grad_a = BatchMatMul(grad_c, b), grad_b = BatchMatMul(grad_c^T, a)
    grad_a = op->requires_grad(0) ? MakeBatchMatMulOp(grad_c, b, false, false, g_op_meta.set_name(op->grad_name(0)))
                                 : Tensor();
    grad_b = op->requires_grad(1) ? MakeBatchMatMulOp(grad_c, a, true, false, g_op_meta.set_name(op->grad_name(1)))
                                 : Tensor();
  } else {
    // case 4: c = BatchMatMul(a^T, b^T)
    // grad_a = BatchMatMul(b^T, grad_c^T), grad_b = BatchMatMul(grad_c^T, a^T)
    grad_a = op->requires_grad(0) ? MakeBatchMatMulOp(b, grad_c, true, true, g_op_meta.set_name(op->grad_name(0)))
                                 : Tensor();
    grad_b = op->requires_grad(1) ? MakeBatchMatMulOp(grad_c, a, true, true, g_op_meta.set_name(op->grad_name(1)))
                                 : Tensor();
  }
  return {grad_a, grad_b};
}


HTShapeList BatchMatMulOpImpl::DoInferShape(Operator& op,
                                            const HTShapeList& input_shapes,
                                            RuntimeContext& runtime_ctx) const {
  const HTShape& a = input_shapes.at(0);
  const HTShape& b = input_shapes.at(1);
  int ndims = a.size() - 2;
  HT_ASSERT(a.size() >= 2 && b.size() >= 2 && a.size() == b.size() &&
            a.at(trans_a() ? ndims + 0 : ndims + 1) ==
              b.at(trans_b() ? ndims + 1 : ndims + 0))
    << "Invalid input shapes for " << op << ":"
    << " (shape_a) " << a << " (shape_b) " << b << " (transpose_a) "
    << trans_a() << " (transpose_b) " << trans_b();
  HTShape shape = {};
  for (int i = 0; i < ndims; ++i) {
    HT_ASSERT(a[i] == b[i])
    << op->name() << ",a:" << a << ",b:" << b;
    shape.emplace_back(a[i]);
  }
  shape.emplace_back(a.at(trans_a() ? ndims + 1 : ndims));
  shape.emplace_back(b.at(trans_b() ? ndims : ndims + 1));
  return {shape};
}

void BatchMatMulOpImpl::DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                                       const OpMeta& op_meta,
                                       const InstantiationContext& inst_ctx) const {
  const Tensor& a = inputs.at(0);
  const Tensor& b = inputs.at(1);
  const DistributedStates& ds_a = a->get_distributed_states();
  const DistributedStates& ds_b = b->get_distributed_states();
  int32_t device_num = ds_a.get_device_num();

  HT_ASSERT(ds_a.is_valid() && ds_b.is_valid() && ds_a.get_device_num() == ds_b.get_device_num())
    << "BatchMatMulOpDef: distributed states for input tensor must be valid!";
  HT_ASSERT(ds_a.get_dim(-2) == 1 && ds_b.get_dim(-2) == 1)
    << "Input tensor shouldn't be partial!";

  int ndims = a->shape().size() - 2;
  for (int i = 0; i < ndims; i++) {
    HT_ASSERT(ds_a.get_dim(i) == ds_b.get_dim(i))
      << "Split states in batch dimensions of tensor a and tensor b should be equal!";
  }
  int dim_0 = ndims;
  int dim_1 = ndims + 1;
  std::vector<std::unordered_map<int32_t, int32_t>> l2res_case({
    {{-1, dim_1}, {dim_0, dim_0}, {dim_1, -2}}, // no trans
    {{-1, dim_1}, {dim_1, dim_0}, {dim_0, -2}}  // trans A
  });
  auto& l2res_map = l2res_case[trans_a()];
  std::vector<std::unordered_map<int32_t, int32_t>> r2res_case({
    {{-1, dim_0}, {dim_0, -2}, {dim_1, dim_1}}, // no trans
    {{-1, dim_0}, {dim_0, dim_1}, {dim_1, -2}}  // trans A
  });
  auto& r2res_map = r2res_case[trans_b()];
  // deduce states
  int32_t lrow = ds_a.get_dim(ndims + trans_a());
  int32_t lcol = ds_a.get_dim(ndims + 1 - trans_a());
  int32_t rrow = ds_b.get_dim(ndims + trans_b());
  int32_t rcol = ds_b.get_dim(ndims + 1 - trans_b());
  HT_ASSERT(lcol == rrow) << "MatMul: tensor a col " << lcol 
    << " must be equal to tensor b row " << rrow;
    
  std::unordered_map<int32_t, int32_t> res_states({
    {-2, lcol}, {dim_0, lrow}, {dim_1, rcol}});
  int32_t batch_dimension_n = 1;
  for (int i = 0; i < ndims; i++) {
    int32_t n = ds_a.get_dim(i); 
    if (n > 1) {
      res_states[i] = n;
      batch_dimension_n *= n;
    }
  }
  res_states[-1] = device_num / (batch_dimension_n * lcol * lrow * rcol);
  // deduce order
  std::vector<int32_t> lorder = ds_a.get_order();
  std::vector<int32_t> rorder = ds_b.get_order();
  auto get_new_order = [&](std::unordered_map<int32_t, int32_t>& _map,
  std::vector<int32_t>& _order) -> std::vector<int32_t> {
    std::vector<int32_t> new_order;
    for (int32_t x : _order) {
      if (_map.find(x) != _map.end()) {
        new_order.push_back(_map[x]);
      } else {
        new_order.push_back(x);
      }
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
  if (new_lorder != new_rorder) {
    new_lorder[get_index(new_lorder, dim_1)] = -1;
    new_rorder[get_index(new_rorder, dim_0)] = -1;
    HT_ASSERT(new_lorder == new_rorder) << "new_lorder is not equal to new_rorder!";
  } else if (std::find(new_lorder.begin(), new_lorder.end(), dim_0) != new_lorder.end() 
             && ds_a.get_dim(-1) > 1) {
    int32_t ind0 = get_index(new_lorder, dim_0);
    int32_t ind1 = get_index(new_lorder, dim_1);
    if (ind0 > ind1) {
      int32_t tmp = ind0;
      ind0 = ind1;
      ind1 = tmp;
    }
    HT_ASSERT(ind0 + 1 == ind1) << "ind0 + 1 != ind1";
    new_lorder.insert(new_lorder.begin() + ind1, -1);
  }
  std::vector<int32_t> res_order(new_lorder);
  // set distributed states for result c
  Tensor& c = outputs.at(0);
  c->set_distributed_states({device_num, res_states, res_order});  
}

Tensor MakeBatchMatMulOp(Tensor a, Tensor b, bool trans_a, bool trans_b,
                         OpMeta op_meta) {
  TensorList inputs = {std::move(a), std::move(b)};
  return Graph::MakeOp(
          std::make_shared<BatchMatMulOpImpl>(trans_a, trans_b),
          std::move(inputs),
          std::move(op_meta))->output(0);
}

} // namespace graph
} // namespace hetu
