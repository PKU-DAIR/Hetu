#include "hetu/graph/ops/RMSNorm.h"
#include "hetu/graph/ops/Reshape.h"
#include "hetu/graph/ops/Slice.h"
#include "hetu/graph/ops/Reduce.h"
#include "hetu/graph/headers.h"
#include "hetu/graph/ops/kernel_links.h"

namespace hetu {
namespace graph {

void RMSNormOpImpl::DoCompute(Operator& op, 
                              const NDArrayList& inputs, NDArrayList& outputs,
                              RuntimeContext& ctx) const {
  const NDArray x0 = input_indexs(0) >= 0 ? inputs.at(input_indexs(0)) : NDArray();
  const NDArray residual_ = input_indexs(1) >= 0 ? inputs.at(input_indexs(1)) : NDArray();
  const NDArray gamma = input_indexs(2) >= 0 ? inputs.at(input_indexs(2)) : NDArray();
  const NDArray beta_ = input_indexs(3) >= 0 ? inputs.at(input_indexs(3)) : NDArray();
  const NDArray rowscale_ = input_indexs(4) >= 0 ? inputs.at(input_indexs(4)) : NDArray();
  const NDArray colscale_ = input_indexs(5) >= 0 ? inputs.at(input_indexs(5)) : NDArray();
  const NDArray x0_subset_ = input_indexs(6) >= 0 ? inputs.at(input_indexs(6)) : NDArray();
  const NDArray z_subset_ = input_indexs(7) >= 0 ? inputs.at(input_indexs(7)) : NDArray();
  NDArray z = output_indexs(0) >= 0 ? outputs.at(output_indexs(0)) : NDArray();
  NDArray x = output_indexs(1) >= 0 ? outputs.at(output_indexs(1)) : NDArray();
  NDArray dmask = output_indexs(2) >= 0 ? outputs.at(output_indexs(2)) : NDArray();
  NDArray mu = output_indexs(3) >= 0 ? outputs.at(output_indexs(3)) : NDArray();
  NDArray rsigma = output_indexs(4) >= 0 ? outputs.at(output_indexs(4)) : NDArray();

  int64_t hidden_size = gamma->numel();
  NDArray x0mat = x0.is_defined() ? NDArray::view(x0, {-1, hidden_size}) : x0;
  NDArray residualmat = residual_.is_defined() ? NDArray::view(residual_, {-1, hidden_size}) : residual_;
  NDArray rowscalemat = rowscale_.is_defined() ? NDArray::view(rowscale_, {-1}) : rowscale_;
  NDArray x0_subsetmat = x0_subset_.is_defined() ? NDArray::view(x0_subset_, {-1}) : x0_subset_;
  NDArray out_subsetmet = z_subset_.is_defined() ? NDArray::view(z_subset_, {-1}) : z_subset_;
  HT_DISPATCH_KERNEL_CUDA_ONLY(op->instantiation_ctx().placement.type(), type(), hetu::impl::DropoutAddLnFwd,
                               x0mat, residualmat, gamma, beta_, rowscalemat, colscale_, x0_subsetmat, out_subsetmet,
                               z, x, dmask, mu, rsigma, dropout_p(), epsilon(), rowscale_const(), z_numrows(),
                               residual_in_fp32(), is_rms_norm(), op->instantiation_ctx().stream());
}

TensorList RMSNormOpImpl::DoGradient(Operator& op, const TensorList& grad_outputs) const {
  TensorList grad_tensors(op->num_inputs(), Tensor());
  int64_t x0_numrows = 1;
  for (int i = 0; i < op->input(input_indexs(0))->ndim() - 1; ++i) {
    x0_numrows *= op->input(input_indexs(0))->shape(i);
  } 
  TensorList grads = MakeRMSNormGradientOp(grad_outputs.at(0),
                                           grad_outputs.at(1),
                                           output_indexs(1) >= 0 ? op->output(output_indexs(1)) : Tensor(),
                                           input_indexs(5) >= 0 ? op->input(input_indexs(0)) : Tensor(),
                                           output_indexs(2) >= 0 ? op->output(output_indexs(2)) : Tensor(),
                                           output_indexs(3) >= 0 ? op->output(output_indexs(3)) : Tensor(),
                                           output_indexs(4) >= 0 ? op->output(output_indexs(4)) : Tensor(),
                                           input_indexs(2) >= 0 ? op->input(input_indexs(2)) : Tensor(),
                                           input_indexs(4) >= 0 ? op->input(input_indexs(4)) : Tensor(),
                                           input_indexs(5) >= 0 ? op->input(input_indexs(5)) : Tensor(),
                                           input_indexs(6) >= 0 ? op->input(input_indexs(6)) : Tensor(),
                                           input_indexs(7) >= 0 ? op->input(input_indexs(7)) : Tensor(),
                                           dropout_p(), rowscale_const(), input_indexs(6) >= 0 ? x0_numrows : 0,
                                           input_indexs(1) >= 0 ? true : false, is_rms_norm(),
                                           op->grad_op_meta().set_name(op->grad_name()));

  if (input_indexs(0) >= 0 && op->requires_grad(input_indexs(0)))
    grad_tensors[0] = grads[0];
  if (input_indexs(1) >= 0 && op->requires_grad(input_indexs(1)))
    grad_tensors[1] = grads[1];
  if (input_indexs(2) >= 0 && op->requires_grad(input_indexs(2)))
    grad_tensors[2] = grads[2];
  if (input_indexs(3) >= 0 && op->requires_grad(input_indexs(3)))
    grad_tensors[3] = grads[3];
  if (input_indexs(5) >= 0 && op->requires_grad(input_indexs(5)))
    grad_tensors[5] = grads[5];
  return grad_tensors;
}

HTShapeList RMSNormOpImpl::DoInferShape(Operator& op, 
                                        const HTShapeList& input_shapes, 
                                        RuntimeContext& ctx) const {
  HTShapeList out_shapes = {};
  int ptr = 0;
  HTShape x0 = input_indexs(0) >= 0 ? input_shapes.at(input_indexs(0)) : HTShape();
  HTShape residual_ = input_indexs(1) >= 0 ? input_shapes.at(input_indexs(1)) : HTShape();
  HTShape gamma = input_indexs(2) >= 0 ? input_shapes.at(input_indexs(2)) : HTShape();
  HTShape beta_ = input_indexs(3) >= 0 ? input_shapes.at(input_indexs(3)) : HTShape();
  HTShape rowscale_ = input_indexs(4) >= 0 ? input_shapes.at(input_indexs(4)) : HTShape();
  HTShape colscale_ = input_indexs(5) >= 0 ? input_shapes.at(input_indexs(5)) : HTShape();
  HTShape x0_subset_ = input_indexs(6) >= 0 ? input_shapes.at(input_indexs(6)) : HTShape();
  HTShape z_subset_ = input_indexs(7) >= 0 ? input_shapes.at(input_indexs(7)) : HTShape();
  int64_t hidden_size = NumEl(gamma);
  HTShape x0mat_shape = {NumEl(x0) / hidden_size, hidden_size}, x0_subset_shape = {};
  if (x0_subset_.size() > 0)
    x0_subset_shape = {NumEl(x0_subset_) / hidden_size, hidden_size};
  HTShape sizes {!(x0_subset_.size() > 0) ? x0mat_shape[0] : x0_subset_shape[0], x0mat_shape[1]};
  const int rows = sizes[0];
  const int cols = sizes[1];
  HTShape zmat_shape = z_subset_.size() > 0 ? HTShape{z_numrows(), cols} : sizes;
  int64_t numel = 1;
  for (int i = 1; i < x0.size(); ++i)
    numel *= x0[i];
  HTShape x_shape = {NumEl(sizes) / numel};
  HTShape z_shape = {NumEl(zmat_shape) / numel};
  for (int i = 1; i < x0.size(); ++i) {
    x_shape.emplace_back(x0[i]);
    z_shape.emplace_back(x0[i]);
  }
  out_shapes.emplace_back(z_shape);
  if (output_indexs(1) >= 0)
    out_shapes.emplace_back(x_shape);
  if (output_indexs(2) >= 0)
    out_shapes.emplace_back(x_shape);
  out_shapes.emplace_back(HTShape{ rows });
  out_shapes.emplace_back(HTShape{ rows });
  return out_shapes;
}

void RMSNormOpImpl::DoDeduceStates(const TensorList& inputs, TensorList& outputs,
                                   const OpMeta& op_meta) const {
  for (auto output : outputs)
    output->set_distributed_states(inputs.at(0)->get_distributed_states()); 
}

void RMSNormOpImpl::DoDeduceHeterProp(const std::vector<int32_t>& inputs_hetero_dim,
                                      TensorList& outputs, const OpMeta& op_meta) const {
  for (auto output : outputs)
    output->cur_ds_union().set_hetero_dim(inputs_hetero_dim.at(0));
}

void RMSNormGradientOpImpl::DoCompute(Operator& op,const NDArrayList& inputs,
                                      NDArrayList& outputs, RuntimeContext& ctx) const {
  const NDArray dz = input_indexs(0) >= 0 ? inputs.at(input_indexs(0)) : NDArray();
  const NDArray dx_ = input_indexs(1) >= 0 ? inputs.at(input_indexs(1)) : NDArray();
  const NDArray x = input_indexs(2) >= 0 ? inputs.at(input_indexs(2)) : NDArray();
  const NDArray x0_ = input_indexs(3) >= 0 ? inputs.at(input_indexs(3)) : NDArray();
  const NDArray dmask_ = input_indexs(4) >= 0 ? inputs.at(input_indexs(4)) : NDArray();
  const NDArray mu = input_indexs(5) >= 0 ? inputs.at(input_indexs(5)) : NDArray();
  const NDArray rsigma = input_indexs(6) >= 0 ? inputs.at(input_indexs(6)) : NDArray();
  const NDArray gamma = input_indexs(7) >= 0 ? inputs.at(input_indexs(7)) : NDArray();
  const NDArray rowscale_ = input_indexs(8) >= 0 ? inputs.at(input_indexs(8)) : NDArray();
  const NDArray colscale_ = input_indexs(9) >= 0 ? inputs.at(input_indexs(9)) : NDArray();
  const NDArray x0_subset_ = input_indexs(10) >= 0 ? inputs.at(input_indexs(10)) : NDArray();
  const NDArray z_subset_ = input_indexs(11) >= 0 ? inputs.at(input_indexs(11)) : NDArray();
  NDArray dx0 = output_indexs(0) >= 0 ? outputs.at(output_indexs(0)) : NDArray();
  NDArray dresidual = output_indexs(1) >= 0 ? outputs.at(output_indexs(1)) : NDArray();
  NDArray dgamma = output_indexs(2) >= 0 ? outputs.at(output_indexs(2)) : NDArray();
  NDArray dbeta = output_indexs(3) >= 0 ? outputs.at(output_indexs(3)) : NDArray();
  NDArray dcolscale = output_indexs(4) >= 0 ? outputs.at(output_indexs(4)) : NDArray();

  int64_t hidden_size = gamma->numel();
  NDArray xmat = x.is_defined() ? NDArray::view(x, {-1, hidden_size}) : x;
  NDArray dzmat = dz.is_defined() ? NDArray::view(dz, {-1, hidden_size}) : dz;
  NDArray dxmat = dx_.is_defined() ? NDArray::view(dx_, {-1, hidden_size}) : dx_;
  NDArray x0mat = x0_.is_defined() ? NDArray::view(x0_, {-1, hidden_size}) : x0_;
  NDArray rowscalemat = rowscale_.is_defined() ? NDArray::view(rowscale_, {-1}) : rowscale_;
  NDArray x0_subsetmat = x0_subset_.is_defined() ? NDArray::view(x0_subset_, {-1}) : x0_subset_;
  NDArray out_subsetmat = z_subset_.is_defined() ? NDArray::view(z_subset_, {-1}) : z_subset_;

  NDArray dgamma_part = NDArray();
  NDArray dbeta_part = NDArray(); 
  NDArray dcolscale_part = NDArray();

  HT_DISPATCH_KERNEL_CUDA_ONLY(op->instantiation_ctx().placement.type(), type(),
                               hetu::impl::DropoutAddLnBwd, dzmat, dxmat, xmat, x0mat, dmask_, mu,
                               rsigma, gamma, rowscalemat, colscale_, x0_subsetmat, out_subsetmat,
                               dx0, dresidual, dgamma, dbeta, dgamma_part, dbeta_part, dcolscale,
                               dcolscale_part, dropout_p(), rowscale_const(), x0_numrows(),
                               has_residual(), is_rms_norm(), op->instantiation_ctx().stream());
}

// workaround: need to care about all input cases
void RMSNormGradientOpImpl::DoDeduceStates(const TensorList& inputs, TensorList& outputs,
                                           const OpMeta& op_meta) const {
  const DistributedStates& ds_output_grad = inputs.at(0)->get_distributed_states();
  int reduce_dim = inputs.at(0)->ndim() - 1;
  HTAxes axes(reduce_dim);
  HTKeepDims keepdims(reduce_dim);
  for (int d = 0; d < reduce_dim; d++) {
    axes[d] = d;
    keepdims[d] = false;
  }
  DistributedStates ds_gamma_scale = ReduceOpImpl::StatesForDistributedReduce(inputs.at(0), axes, keepdims);
  //HT_LOG_TRACE << "ds_output_grad = " << ds_output_grad.ds_info() << ", ds_gamma_scale = " << ds_gamma_scale.ds_info() << ", output size = " << outputs.size() << ", indx2 = " << output_indexs(2);
  outputs.at(0)->set_distributed_states(ds_output_grad);
  outputs.at(output_indexs(2))->set_distributed_states(ds_gamma_scale);
  outputs.at(output_indexs(3))->set_distributed_states(ds_gamma_scale);
  //HT_LOG_TRACE << "RMSNormGradientOpImpl do gradient end";
}

void RMSNormGradientOpImpl::DoDeduceHeterProp(const std::vector<int32_t>& inputs_hetero_dim,
                                              TensorList& outputs, const OpMeta& op_meta) const {
  HT_ASSERT(inputs_hetero_dim.at(0) >= 0)
    << "Currently not support complex hetero dim deducing"
    << ", the hetero dim should be spilt and reduced to partial";
  outputs.at(0)->cur_ds_union().set_hetero_dim(inputs_hetero_dim.at(0));
  outputs.at(1)->cur_ds_union().set_hetero_dim(-2);
  outputs.at(2)->cur_ds_union().set_hetero_dim(-2);
}

HTShapeList RMSNormGradientOpImpl::DoInferShape(Operator& op, 
                                                const HTShapeList& input_shapes, 
                                                RuntimeContext& ctx) const {
  HTShapeList out_shapes = {};
  HTShape dz = input_indexs(0) >= 0 ? input_shapes.at(input_indexs(0)) : HTShape();
  HTShape dx_ = input_indexs(1) >= 0 ? input_shapes.at(input_indexs(1)) : HTShape();
  HTShape x = input_indexs(2) >= 0 ? input_shapes.at(input_indexs(2)) : HTShape();
  HTShape x0_ = input_indexs(3) >= 0 ? input_shapes.at(input_indexs(3)) : HTShape();
  HTShape dmask_ = input_indexs(4) >= 0 ? input_shapes.at(input_indexs(4)) : HTShape();
  HTShape mu = input_indexs(5) >= 0 ? input_shapes.at(input_indexs(5)) : HTShape();
  HTShape rsigma = input_indexs(6) >= 0 ? input_shapes.at(input_indexs(6)) : HTShape();
  HTShape gamma = input_indexs(7) >= 0 ? input_shapes.at(input_indexs(7)) : HTShape();
  HTShape rowscale_ = input_indexs(8) >= 0 ? input_shapes.at(input_indexs(8)) : HTShape();
  HTShape colscale_ = input_indexs(9) >= 0 ? input_shapes.at(input_indexs(9)) : HTShape();
  HTShape x0_subset_ = input_indexs(10) >= 0 ? input_shapes.at(input_indexs(10)) : HTShape();
  HTShape z_subset_ = input_indexs(11) >= 0 ? input_shapes.at(input_indexs(11)) : HTShape();

  auto sizes = x;
  auto rows = sizes[0];
  auto cols = sizes[1];
  HTShape x0_sizes {!x0_subset_.size() > 0 ? rows : x0_numrows(), cols};
  out_shapes.emplace_back(x);
  if (has_residual()) {
    out_shapes.emplace_back(x);
  } 
  out_shapes.emplace_back(gamma);
  out_shapes.emplace_back(gamma);
  if (output_indexs(4) >= 0) {
    out_shapes.emplace_back(colscale_);
  }
  return out_shapes;
}

void FusedRMSNormOpImpl::DoCompute(Operator& op,
                                   const NDArrayList& inputs, NDArrayList& outputs,
                                   RuntimeContext& ctx) const {
  NDArray::fused_rmsnorm(inputs.at(0), inputs.at(1), normalized_shape(),
                         get_eps(), op->instantiation_ctx().stream_index, 
                         outputs.at(0), outputs.at(1));
}

TensorList FusedRMSNormOpImpl::DoGradient(Operator& op, const TensorList& grad_outputs) const {
  auto g_op_meta = op->grad_op_meta().set_name(op->grad_name(0));
  TensorList empty = {Tensor(), Tensor(), Tensor()};
  TensorList grad_input;
  if (inplace()) {
    grad_input = op->requires_grad(0) ? MakeFusedRMSNormGradientOp(grad_outputs.at(0), 
                                        op->output(0), op->input(1), op->output(1),
                                        normalized_shape(), get_eps(), inplace(), g_op_meta)
                                      : empty;
  }
  else {
    grad_input = op->requires_grad(0) ? MakeFusedRMSNormGradientOp(grad_outputs.at(0), 
                                        op->input(0), op->input(1), op->output(1),
                                        normalized_shape(), get_eps(), inplace(), g_op_meta)
                                      : empty;
  }
  if (!op->requires_grad(1))
    grad_input[1] = Tensor();
  return grad_input;
}

HTShapeList FusedRMSNormOpImpl::DoInferShape(Operator& op,const HTShapeList& input_shapes,
                                             RuntimeContext& ctx) const {
  size_t dim = normalized_shape().size();
  HTShape output_shape = input_shapes.at(0);
  for (size_t i = 0; i < dim; ++i) {
    HT_ASSERT(normalized_shape()[dim - 1 - i] == input_shapes.at(0)[input_shapes.at(0).size() - 1 - i])
    << "Normalized shape's last dims should equal to input shape's.But we have normalized shape:"
    << normalized_shape() << " and input shape:" << input_shapes.at(0);
    output_shape[input_shapes.at(0).size() - 1 - i] = 1;
  }
  return {input_shapes.at(0), output_shape};
}

void FusedRMSNormOpImpl::DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                                        const OpMeta& op_meta) const {
  size_t dim = normalized_shape().size();
  HTShape local_shape = inputs.at(0)->shape();
  int max_dim = local_shape.size() - dim;
  const DistributedStates& ds_input = inputs.at(0)->get_distributed_states();
  const DistributedStates& ds_scale = inputs.at(1)->get_distributed_states();
  HT_ASSERT(ds_input.is_valid() && ds_scale.is_valid() 
            && ds_input.get_device_num() == ds_scale.get_device_num()) 
    << "RMSNormOpDef: input states must be valid!";
  HT_ASSERT(ds_input.get_dim(-2) == 1 && ds_scale.get_dim(-2) == 1)
    << "Input tensor shouldn't be partial!";
  HT_ASSERT(ds_input.check_max_dim(max_dim))
    << "RMSNormOp only support input split in dimension < " << max_dim;
  // scale and bias shape should be normalized_shape, so keep duplicate
  HT_ASSERT(ds_scale.check_pure_duplicate())
    << "Scale should be duplicate!";
  outputs.at(0)->set_distributed_states(ds_input); // output
  outputs.at(1)->set_distributed_states(ds_input); // save_var for backward
}

void FusedRMSNormOpImpl::DoDeduceHeterProp(const std::vector<int32_t>& inputs_hetero_dim,
                                           TensorList& outputs, const OpMeta& op_meta) const {
  outputs.at(0)->cur_ds_union().set_hetero_dim(inputs_hetero_dim.at(0));
  outputs.at(1)->cur_ds_union().set_hetero_dim(inputs_hetero_dim.at(0));
}

void FusedRMSNormGradientOpImpl::DoCompute(Operator& op,const NDArrayList& inputs,
                                           NDArrayList& outputs,
                                           RuntimeContext& ctx) const {
  HT_DISPATCH_KERNEL_CUDA_ONLY(
    op->instantiation_ctx().placement.type(), type(), hetu::impl::FusedRMSNormGradient, inputs.at(0),
    inputs.at(1), inputs.at(2), outputs.at(0), outputs.at(1),
    inputs.at(3),normalized_shape().size(),
    get_eps(), inplace(), op->instantiation_ctx().stream());
}

HTShapeList
FusedRMSNormGradientOpImpl::DoInferShape(Operator& op,const HTShapeList& input_shapes,
                                         RuntimeContext& ctx) const {
  return {input_shapes.at(1), input_shapes.at(2)};
}

void FusedRMSNormGradientOpImpl::DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                                                const OpMeta& op_meta) const {
  const DistributedStates& ds_output_grad = inputs.at(0)->get_distributed_states();
  int reduce_dim = inputs.at(0)->ndim() - normalized_shape().size();
  HTAxes axes(reduce_dim);
  HTKeepDims keepdims(reduce_dim);
  for (int d = 0; d < reduce_dim; d++) {
    axes[d] = d;
    keepdims[d] = false;
  }
  DistributedStates ds_bias_scale = ReduceOpImpl::StatesForDistributedReduce(inputs.at(0), axes, keepdims);
  outputs.at(0)->set_distributed_states(ds_output_grad);
  outputs.at(1)->set_distributed_states(ds_bias_scale);
}

void FusedRMSNormGradientOpImpl::DoDeduceHeterProp(const std::vector<int32_t>& inputs_hetero_dim,
                                                   TensorList& outputs, const OpMeta& op_meta) const {
  HT_ASSERT(inputs_hetero_dim.at(0) >= 0 && inputs_hetero_dim.at(0) < outputs.at(0)->ndim() - normalized_shape().size())
    << "Currently not support complex hetero dim deducing"
    << ", the hetero dim should be spilt and reduced to partial";
  outputs.at(0)->cur_ds_union().set_hetero_dim(inputs_hetero_dim.at(0));
  outputs.at(1)->cur_ds_union().set_hetero_dim(-2);
}

TensorList MakeRMSNormOp(Tensor x0, Tensor residual_, Tensor gamma,
                         Tensor beta_, Tensor rowscale_, Tensor colscale_,
                         Tensor x0_subset_, Tensor z_subset_, 
                         const float dropout_p, const float epsilon,
                         const float rowscale_const, const int64_t z_numrows,
                         bool residual_in_fp32, bool prenorm, 
                         bool is_rms_norm, bool return_dmask, 
                         OpMeta op_meta) {
  auto itype = x0->dtype();
  auto rtype = residual_.is_defined()
        ? residual_->dtype()
        : (residual_in_fp32 ? kFloat32 : x0->dtype());
  std::vector<int> output_indexs(5, -1);
  int ptr = 0;
  bool save_x = residual_.is_defined() || (dropout_p > 0.f) || rowscale_.is_defined() || 
                colscale_.is_defined() || x0_subset_.is_defined() || (itype != rtype);
  output_indexs[0] = ptr++;
  if (save_x)
    output_indexs[1] = ptr++;
  if (dropout_p > 0.f)
    output_indexs[2] = ptr++;
  output_indexs[3] = ptr++;
  output_indexs[4] = ptr++;

  TensorList inputs = {std::move(x0), std::move(residual_), std::move(gamma),
                       std::move(beta_), std::move(rowscale_), std::move(colscale_),
                       std::move(x0_subset_), std::move(z_subset_)};
  TensorList inputs_ = {};
  std::vector<int> input_indexs(8, -1);
  ptr = 0;
  for (int i = 0; i < inputs.size(); ++i) {
    if (inputs[i].is_defined()) {
      input_indexs[i] = ptr++;
      inputs_.emplace_back(inputs[i]);
    }
  }
  return Graph::MakeOp(
        std::make_shared<RMSNormOpImpl>(dropout_p, epsilon, 
                                        rowscale_const, z_numrows,
                                        residual_in_fp32, prenorm, 
                                        is_rms_norm, return_dmask,
                                        input_indexs, output_indexs),
        std::move(inputs_),
        std::move(op_meta))->outputs();
}

TensorList MakeRMSNormGradientOp(Tensor dz, Tensor dx_, Tensor x, Tensor x0_,     
                                 Tensor dmask_, Tensor mu, Tensor rsigma, Tensor gamma,   
                                 Tensor rowscale_, Tensor colscale_, Tensor x0_subset_,  
                                 Tensor z_subset_, const float dropout_p, const float rowscale_const,
                                 const int64_t x0_numrows, const bool has_residual,
                                 bool is_rms_norm, OpMeta op_meta) {
  std::vector<int> output_indexs(5, -1);
  int ptr = 0;
  output_indexs[0] = ptr++;
  if (has_residual) {
    output_indexs[1] = ptr++;
  } 
  output_indexs[2] = ptr++;
  output_indexs[3] = ptr++;
  if (colscale_.is_defined()) {
    output_indexs[4] = ptr++;
  }
  TensorList inputs = {std::move(dz), std::move(dx_), std::move(x), std::move(x0_),
                       std::move(dmask_), std::move(mu), std::move(rsigma), std::move(gamma),
                       std::move(rowscale_), std::move(colscale_), std::move(x0_subset_), std::move(z_subset_)};
  TensorList inputs_ = {};
  std::vector<int> input_indexs(12, -1);
  ptr = 0;
  for (int i = 0; i < inputs.size(); ++i) {
    if (inputs[i].is_defined()) {
      input_indexs[i] = ptr++;
      inputs_.emplace_back(inputs[i]);
    }
  }
  return Graph::MakeOp(
         std::make_shared<RMSNormGradientOpImpl>(dropout_p, rowscale_const, x0_numrows, 
                                                 has_residual, is_rms_norm, input_indexs,
                                                 output_indexs),
         std::move(inputs_),
         std::move(op_meta))->outputs();
}

TensorList MakeFusedRMSNormOp(Tensor input, Tensor ln_scale, HTShape normalized_shape, 
                              double eps, bool inplace, OpMeta op_meta) {
  TensorList inputs = {std::move(input), std::move(ln_scale)};
  return Graph::MakeOp(
          std::make_shared<FusedRMSNormOpImpl>(normalized_shape, eps, inplace),
          std::move(inputs),
          std::move(op_meta))->outputs();   
}

TensorList MakeFusedRMSNormGradientOp(Tensor output_grad, Tensor input, Tensor ln_scale,
                                      Tensor save_var, HTShape normalized_shape, 
                                      double eps, bool inplace, OpMeta op_meta) {
  return Graph::MakeOp(
          std::make_shared<FusedRMSNormGradientOpImpl>(normalized_shape, eps, inplace),
          {std::move(output_grad), std::move(input), 
          std::move(ln_scale), std::move(save_var)},
          std::move(op_meta))->outputs();  
}

} // namespace graph
} // namespace hetu
