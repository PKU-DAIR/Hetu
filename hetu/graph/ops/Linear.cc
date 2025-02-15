#include "hetu/graph/ops/Linear.h"
#include "hetu/graph/ops/matmul.h"
#include "hetu/graph/ops/Reduce.h"
#include "hetu/graph/headers.h"
#include "hetu/graph/ops/kernel_links.h"

namespace hetu {
namespace graph {

void LinearOpImpl::DoCompute(Operator& op,const NDArrayList& inputs, NDArrayList& outputs,
                            RuntimeContext& ctx) const {
  if (inputs.size() == 2)
    NDArray::linear(inputs.at(0), inputs.at(1), NDArray(), trans_a(), trans_b(),
                    op->instantiation_ctx().stream_index, outputs.at(0));
  else if (inputs.size() == 3)
    NDArray::linear(inputs.at(0), inputs.at(1), inputs.at(2), trans_a(), trans_b(),
                    op->instantiation_ctx().stream_index, outputs.at(0));
}

TensorList LinearOpImpl::DoGradient(Operator& op,const TensorList& grad_outputs) const {
  const Tensor& grad_c = grad_outputs.at(0);
  Tensor& a = op->input(0);
  Tensor& b = op->input(1);
  Tensor grad_a;
  Tensor grad_b;
  auto g_op_meta = op->grad_op_meta();
  if (!trans_a() && !trans_b()) {
    // case 1: c = Linear(a, b)
    // grad_a = Linear(grad_c, b^T), grad_b = Linear(a^T, grad_c)
    grad_a = op->requires_grad(0) ? MakeLinearOp(grad_c, b, false, true, g_op_meta.set_name(op->grad_name(0)))
                                 : Tensor();
    grad_b = op->requires_grad(1) ? MakeLinearOp(a, grad_c, true, false, g_op_meta.set_name(op->grad_name(1)))
                                 : Tensor();
  } else if (trans_a() && !trans_b()) {
    // case 2: c = Linear(a^T, b)
    // grad_a = Linear(b, grad_c^T), grad_b = Linear(a, grad_c)
    grad_a = op->requires_grad(0) ? MakeLinearOp(b, grad_c, false, true, g_op_meta.set_name(op->grad_name(0)))
                                 : Tensor();
    grad_b = op->requires_grad(1) ? MakeLinearOp(a, grad_c, false, false, g_op_meta.set_name(op->grad_name(1)))
                                 : Tensor();
  } else if (!trans_a() && trans_b()) {
    // case 3: c = Linear(a, b^T)
    // grad_a = Linear(grad_c, b), grad_b = Linear(grad_c^T, a)
    grad_a = op->requires_grad(0) ? MakeLinearOp(grad_c, b, false, false, g_op_meta.set_name(op->grad_name(0)))
                                 : Tensor();
    grad_b = op->requires_grad(1) ? MakeLinearOp(grad_c, a, true, false, g_op_meta.set_name(op->grad_name(1)))
                                 : Tensor();
  } else {
    // case 4: c = Linear(a^T, b^T)
    // grad_a = Linear(b^T, grad_c^T), grad_b = Linear(grad_c^T, a^T)
    grad_a = op->requires_grad(0) ? MakeLinearOp(b, grad_c, true, true, g_op_meta.set_name(op->grad_name(0)))
                                 : Tensor();
    grad_b = op->requires_grad(1) ? MakeLinearOp(grad_c, a, true, true, g_op_meta.set_name(op->grad_name(1)))
                                 : Tensor();
  }
  if (op->num_inputs() == 2) {
    return {grad_a, grad_b};
  } else if (op->num_inputs() == 3) {
    Tensor grad_bias = op->requires_grad(2) ? MakeReduceOp(grad_outputs.at(0), ReductionType::SUM, {0}, {false},
                                           g_op_meta.set_name(op->grad_name(2)))
                                         : Tensor();
    return {grad_a, grad_b, grad_bias};
  }
}

HTShapeList LinearOpImpl::DoInferShape(Operator& op,
                                       const HTShapeList& input_shapes,
                                       RuntimeContext& ctx) const {
  const HTShape& a = input_shapes.at(0);
  const HTShape& b = input_shapes.at(1);
  HT_ASSERT(a.size() == 2 && b.size() == 2 &&
            a.at(trans_a() ? 0 : 1) == b.at(trans_b() ? 1 : 0))
    << "Invalid input shapes for " << op << ":"
    << " (shape_a) " << a << " (shape_b) " << b << " (transpose_a) "
    << trans_a() << " (transpose_b) " << trans_b();
  return {{a.at(trans_a() ? 1 : 0), b.at(trans_b() ? 0 : 1)}};
}

void LinearOpImpl::DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                                  const OpMeta& op_meta,
                                  const InstantiationContext& inst_ctx) const {
  const Tensor& a = inputs.at(0);
  const Tensor& b = inputs.at(1);
  const DistributedStates& ds_a = a->get_distributed_states();
  const DistributedStates& ds_b = b->get_distributed_states();
  int32_t device_num = ds_a.get_device_num();
  HT_ASSERT(ds_a.is_valid() && ds_b.is_valid()
            && ds_a.get_device_num() == ds_b.get_device_num())
            << "distributed states for Tensor a & Tensor b should be valid!";  
  Tensor bias;
  DistributedStates ds_bias;
  if (inputs.size() == 3) {
    bias = inputs.at(2);  
    ds_bias = bias->get_distributed_states();
    // check bias states
    if (trans_b()) { // bias shape = (b.shape[0], )
      HT_ASSERT(ds_b.get_dim(0) == ds_bias.get_dim(0))
        << "LinearOp: bias should split same with dimension 0 of b";
    } else { // bias shape = (b.shape[1], )
      HT_ASSERT(ds_b.get_dim(1) == ds_bias.get_dim(0))
        << "LinearOp: bias should split same with dimension 1 of b";
    }
  }
  // l,r to result states map  
  std::vector<std::unordered_map<int32_t, int32_t>> l2res_case({
    {{-1, 1}, {0, 0}, {1, -2}}, // no trans
    {{-1, 1}, {1, 0}, {0, -2}}  // trans A
  });
  auto& l2res_map = l2res_case[trans_a()];
  std::vector<std::unordered_map<int32_t, int32_t>> r2res_case({
    {{-1, 0}, {0, -2}, {1, 1}}, // no trans
    {{-1, 0}, {0, 1}, {1, -2}}  // trans B
  });
  auto& r2res_map = r2res_case[trans_b()];
  // deduce states
  int32_t lrow = ds_a.get_dim(trans_a());
  int32_t lcol = ds_a.get_dim(1-trans_a());
  int32_t rrow = ds_b.get_dim(trans_b());
  int32_t rcol = ds_b.get_dim(1-trans_b());
  HT_ASSERT(lcol == rrow) << "Linear: tensor a.dimension[1] " << lcol 
    << " must be equal to tensor b.dimension[0] " << rrow;
  // if output states contains partial, then requires bias also should be partial
  HT_ASSERT(inputs.size() == 2 || lcol == ds_bias.get_dim(-2))
    << "Linear: partial in output states = " << lcol << " should be equal to partial of bias = " << ds_bias.get_dim(-2);
  std::unordered_map<int32_t, int32_t> res_states({
    {-2, lcol}, {-1, device_num/(lcol*lrow*rcol)}, {0, lrow}, {1, rcol}
  });
  // deduce order
  std::vector<int32_t> lorder = ds_a.get_order();
  std::vector<int32_t> rorder = ds_b.get_order();
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
  if (new_lorder != new_rorder) {
    new_lorder[get_index(new_lorder, 1)] = -1;
    new_rorder[get_index(new_rorder, 0)] = -1;
    HT_ASSERT(new_lorder == new_rorder) << "new_lorder is not equal to new_rorder!";
  } else if (std::find(new_lorder.begin(), new_lorder.end(), 0) != new_lorder.end()
             && ds_a.get_dim(-1) > 1) {
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
  // set distributed states for result c
  Tensor& c = outputs.at(0);
  c->set_distributed_states({device_num, res_states, res_order});
}

void LinearOpImpl::DoDeduceHeterProp(const std::vector<int32_t>& inputs_hetero_dim,
                                     TensorList& outputs, const OpMeta& op_meta,
                                     const InstantiationContext& inst_ctx) const {
  int32_t hetero_a = inputs_hetero_dim.at(0);
  int32_t hetero_b = inputs_hetero_dim.at(1);  
  if (trans_a() && (hetero_a == 0 || hetero_a == 1)) {
    hetero_a = 1 - hetero_a;
  }
  if (trans_b() && (hetero_b == 0 || hetero_b == 1)) {
    hetero_b = 1 - hetero_b;
  }
  int32_t hetero_res;
  if (hetero_a == NULL_HETERO_DIM) {
    HT_ASSERT(hetero_b == NULL_HETERO_DIM)
      << "Currently not support different union hetero type";
    hetero_res = NULL_HETERO_DIM;
  } else {
    if (hetero_a == -1 || hetero_b == -1) {
      if (hetero_a == -1) {
        HT_RUNTIME_ERROR << "not supported yet";
      }
      if (hetero_b == -1) {
        HT_ASSERT(hetero_a >= 0)
          << "hetero a and hetero b can't simutaneously be -1";
        hetero_res = hetero_a;
      }
    } else {
      HT_ASSERT(hetero_a == 1 - hetero_b)
        << "hetero a and hetero b should be opposite in this situation";
      hetero_res = -2;
    }
  }   
  outputs.at(0)->cur_ds_union().set_hetero_dim(hetero_res);
}

Tensor MakeLinearOp(Tensor a, Tensor b, Tensor bias, bool trans_a,
                    bool trans_b, OpMeta op_meta) {
  TensorList inputs = {std::move(a), std::move(b), std::move(bias)};
  return Graph::MakeOp(
        std::make_shared<LinearOpImpl>(trans_a, trans_b),
        std::move(inputs),
        std::move(op_meta))->output(0);
}

Tensor MakeLinearOp(Tensor a, Tensor b, bool trans_a,
                    bool trans_b, OpMeta op_meta) {
  return Graph::MakeOp(
        std::make_shared<LinearOpImpl>(trans_a, trans_b),
        {std::move(a), std::move(b)},
        std::move(op_meta))->output(0);
}

} // namespace graph
} // namespace hetu
