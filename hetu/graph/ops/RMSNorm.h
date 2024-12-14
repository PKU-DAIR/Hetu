#pragma once

#include "hetu/graph/operator.h"
#include "hetu/graph/utils/tensor_utils.h"

namespace hetu {
namespace graph {

class RMSNormOpImpl;
class RMSNormOp;
class RMSNormGradientOpImpl;
class RMSNormGradientOp;
class FusedRMSNormOpImpl;
class FusedRMSNormOp;
class FusedRMSNormGradientOpImpl;
class FusedRMSNormGradientOp;

class RMSNormOpImpl final : public OpInterface {
 private:
  friend class RMSNormOp;
  struct constructor_access_key {};

 public:
  RMSNormOpImpl(const float dropout_p, const float epsilon,
                const float rowscale_const, const int64_t z_numrows,
                bool residual_in_fp32, bool prenorm, 
                bool is_rms_norm, bool return_dmask,
                std::vector<int> input_indexs,
                std::vector<int> output_indexs)
  : OpInterface(quote(RMSNormOp)), _dropout_p(dropout_p), _epsilon(epsilon),
                                   _rowscale_const(rowscale_const), 
                                   _z_numrows(z_numrows),
                                   _residual_in_fp32(residual_in_fp32),
                                   _prenorm(prenorm),
                                   _is_rms_norm(is_rms_norm),
                                   _return_dmask(return_dmask),
                                   _input_indexs(input_indexs),
                                   _output_indexs(output_indexs) {
  }

  inline float dropout_p() const {
    return _dropout_p;
  }

  inline float epsilon() const {
    return _epsilon;
  }

  inline float rowscale_const() const {
    return _rowscale_const;
  }

  inline int64_t z_numrows() const {
    return _z_numrows;
  }

  inline bool residual_in_fp32() const {
    return _residual_in_fp32;
  }

  inline bool prenorm() const {
    return _prenorm;
  }

  inline bool is_rms_norm() const {
    return _is_rms_norm;
  }

  inline bool return_dmask() const {
    return _return_dmask;
  }

  inline std::vector<int> input_indexs() const {
    return _input_indexs;
  }

  inline int input_indexs(int i) const {
    return _input_indexs[i];
  }

  inline std::vector<int> output_indexs() const {
    return _output_indexs;
  }

  inline int output_indexs(int i) const {
    return _output_indexs[i];
  }

 protected:
  std::vector<NDArrayMeta> 
  DoInferMeta(const TensorList& inputs) const override {
    std::vector<NDArrayMeta> out_metas = {};
    Tensor x0 = input_indexs(0) >= 0 ? inputs.at(input_indexs(0)) : Tensor();
    Tensor residual_ = input_indexs(1) >= 0 ? inputs.at(input_indexs(1)) : Tensor();
    Tensor gamma = input_indexs(2) >= 0 ? inputs.at(input_indexs(2)) : Tensor();
    Tensor beta_ = input_indexs(3) >= 0 ? inputs.at(input_indexs(3)) : Tensor();
    Tensor rowscale_ = input_indexs(4) >= 0 ? inputs.at(input_indexs(4)) : Tensor();
    Tensor colscale_ = input_indexs(5) >= 0 ? inputs.at(input_indexs(5)) : Tensor();
    Tensor x0_subset_ = input_indexs(6) >= 0 ? inputs.at(input_indexs(6)) : Tensor();
    Tensor z_subset_ = input_indexs(7) >= 0 ? inputs.at(input_indexs(7)) : Tensor();
    int64_t hidden_size = gamma->numel();

    HT_ASSERT(x0->numel() % hidden_size == 0);
    HTShape x0mat_shape = {x0->numel() / hidden_size, hidden_size}, x0_subset_shape = {};
    auto itype = x0->dtype();
    auto rtype = residual_.is_defined()
        ? residual_->dtype()
        : (residual_in_fp32() ? kFloat32 : x0->dtype());
    auto wtype = gamma->dtype();
    auto otype = itype;
    auto ctype = kFloat32;
    auto mtype = kUInt8;
    NDArrayMeta base = x0->meta();
    if (x0_subset_.is_defined()) {
      HT_ASSERT(x0_subset_->numel() % hidden_size == 0);
      x0_subset_shape = {x0_subset_->numel() / hidden_size, hidden_size};
    }
    HTShape sizes {!x0_subset_.is_defined() ? x0mat_shape[0] : x0_subset_shape[0], x0mat_shape[1]};
    const int rows = sizes[0];
    const int cols = sizes[1];
    int ptr = 0;
    bool save_x = residual_.is_defined() || (dropout_p() > 0.f) || rowscale_.is_defined() || 
                  colscale_.is_defined() || x0_subset_.is_defined() || (itype != rtype);
    HTShape zmat_shape = z_subset_.is_defined() ? HTShape{z_numrows(), cols} : sizes;
    int64_t numel = 1;
    for (int i = 1; i < x0->ndim(); ++i)
      numel *= x0->shape(i);
    HTShape x_shape = {NumEl(sizes) / numel};
    HTShape z_shape = {NumEl(zmat_shape) / numel};
    for (int i = 1; i < x0->ndim(); ++i) {
      x_shape.emplace_back(x0->shape(i));
      z_shape.emplace_back(x0->shape(i));
    }
    out_metas.emplace_back(base.set_shape(z_shape).set_dtype(otype));
    if (output_indexs(1) >= 0) {
      out_metas.emplace_back(base.set_shape(x_shape).set_dtype(itype));
    } 
    if (output_indexs(2) >= 0) {
      out_metas.emplace_back(base.set_shape(x_shape).set_dtype(mtype));
    }
    out_metas.emplace_back(base.set_shape({ rows }).set_dtype(ctype));
    out_metas.emplace_back(base.set_shape({ rows }).set_dtype(ctype));
    HT_LOG_DEBUG << out_metas;
    return out_metas;
  };

  void DoDeduceStates(const TensorList& inputs, TensorList& outputs,
                      const OpMeta& op_meta) const override;

  void DoDeduceHeterProp(const std::vector<int32_t>& inputs_hetero_dim,
                         TensorList& outputs, const OpMeta& op_meta) const override;  

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) const override;

  TensorList DoGradient(Operator& op, const TensorList& grad_outputs) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes, RuntimeContext& ctx) const override;

  float _dropout_p;
  float _epsilon;
  float _rowscale_const;
  int64_t _z_numrows;
  bool _residual_in_fp32;
  bool _prenorm;
  bool _is_rms_norm;
  bool _return_dmask;
  std::vector<int> _input_indexs;
  std::vector<int> _output_indexs;

 public:
  bool operator==(const OpInterface& rhs) const override {
    if (OpInterface::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const RMSNormOpImpl&>(rhs);
      return dropout_p() == rhs_.dropout_p() &&
             epsilon() == rhs_.epsilon() &&
             rowscale_const() == rhs_.rowscale_const() &&
             z_numrows() == rhs_.z_numrows() &&
             residual_in_fp32() == rhs_.residual_in_fp32() &&
             prenorm() == rhs_.prenorm() &&
             is_rms_norm() == rhs_.is_rms_norm() &&
             return_dmask() == rhs_.return_dmask() &&
             input_indexs() == rhs_.input_indexs() &&
             output_indexs() == rhs_.output_indexs();
    } 
    else
      return false;
  }
};

TensorList MakeRMSNormOp(Tensor x0, Tensor residual_, Tensor gamma,
                         Tensor beta_, Tensor rowscale_, Tensor colscale_,
                         Tensor x0_subset_ = Tensor(), Tensor z_subset_ = Tensor(), 
                         const float dropout_p = 0, const float epsilon = 1e-5,
                         const float rowscale_const = 1.0, const int64_t z_numrows = 0, 
                         bool residual_in_fp32 = false, bool prenorm = false, 
                         bool is_rms_norm = false, bool return_dmask = false, 
                         OpMeta op_meta = OpMeta());

class RMSNormGradientOpImpl final : public OpInterface {

 public:
  RMSNormGradientOpImpl(const float dropout_p, const float rowscale_const,
                        const int64_t x0_numrows, const bool has_residual,
                        bool is_rms_norm, std::vector<int> input_indexs,
                        std::vector<int> output_indexs)
  : OpInterface(quote(RMSNormGradientOp)), 
  _dropout_p(dropout_p), _rowscale_const(rowscale_const), _x0_numrows(x0_numrows),
  _has_residual(has_residual), _is_rms_norm(is_rms_norm), _input_indexs(input_indexs),
  _output_indexs(output_indexs) {
  }

  inline float dropout_p() const {
    return _dropout_p;
  }

  inline float rowscale_const() const {
    return _rowscale_const;
  }

  inline int64_t x0_numrows() const {
    return _x0_numrows;
  }

  inline bool has_residual() const {
    return _has_residual;
  }

  inline bool is_rms_norm() const {
    return _is_rms_norm;
  }

  inline std::vector<int> input_indexs() const {
    return _input_indexs;
  }

  inline int input_indexs(int i) const {
    return _input_indexs[i];
  }

  inline std::vector<int> output_indexs() const {
    return _output_indexs;
  }

  inline int output_indexs(int i) const {
    return _output_indexs[i];
  }

 protected:
  std::vector<NDArrayMeta> 
  DoInferMeta(const TensorList& inputs) const override {
    std::vector<NDArrayMeta> out_metas = {};
    Tensor dz = input_indexs(0) >= 0 ? inputs.at(input_indexs(0)) : Tensor();
    Tensor dx_ = input_indexs(1) >= 0 ? inputs.at(input_indexs(1)) : Tensor();
    Tensor x = input_indexs(2) >= 0 ? inputs.at(input_indexs(2)) : Tensor();
    Tensor x0_ = input_indexs(3) >= 0 ? inputs.at(input_indexs(3)) : Tensor();
    Tensor dmask_ = input_indexs(4) >= 0 ? inputs.at(input_indexs(4)) : Tensor();
    Tensor mu = input_indexs(5) >= 0 ? inputs.at(input_indexs(5)) : Tensor();
    Tensor rsigma = input_indexs(6) >= 0 ? inputs.at(input_indexs(6)) : Tensor();
    Tensor gamma = input_indexs(7) >= 0 ? inputs.at(input_indexs(7)) : Tensor();
    Tensor rowscale_ = input_indexs(8) >= 0 ? inputs.at(input_indexs(8)) : Tensor();
    Tensor colscale_ = input_indexs(9) >= 0 ? inputs.at(input_indexs(9)) : Tensor();
    Tensor x0_subset_ = input_indexs(10) >= 0 ? inputs.at(input_indexs(10)) : Tensor();
    Tensor z_subset_ = input_indexs(11) >= 0 ? inputs.at(input_indexs(11)) : Tensor();

    auto itype = dz->dtype();
    auto rtype = x->dtype();
    auto wtype = gamma->dtype();
    auto otype = itype;
    auto ctype = kFloat32;
    auto mtype = kUInt8;

    NDArrayMeta base = x->meta();
    int64_t hidden_size = gamma->numel();
    HT_ASSERT(x->numel() % hidden_size == 0);
    HTShape sizes = {x->numel() / hidden_size, hidden_size};
    auto rows = sizes[0];
    auto cols = sizes[1];
    HTShape x0_sizes {!x0_subset_.is_defined() ? rows : x0_numrows(), cols};
    int ptr = 0;
    out_metas.emplace_back(base.set_shape(x->shape()).set_dtype(itype));
    if (has_residual()) {
      out_metas.emplace_back(base.set_dtype(rtype));
    } 
    out_metas.emplace_back(gamma->meta());
    out_metas.emplace_back(gamma->meta());
    if (colscale_.is_defined()) {
      out_metas.emplace_back(colscale_->meta());
    }
    HT_LOG_DEBUG << out_metas;
    return out_metas;
  };

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes, RuntimeContext& ctx) const override;

  void DoDeduceStates(const TensorList& inputs, TensorList& outputs,
                      const OpMeta& op_meta) const override;

  void DoDeduceHeterProp(const std::vector<int32_t>& inputs_hetero_dim,
                         TensorList& outputs, const OpMeta& op_meta) const override;  

  float _dropout_p;
  float _rowscale_const;
  int64_t _x0_numrows;
  bool _has_residual;
  bool _is_rms_norm;
  std::vector<int> _input_indexs;
  std::vector<int> _output_indexs;

 public:
  bool operator==(const OpInterface& rhs) const override {
    if (OpInterface::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const RMSNormGradientOpImpl&>(rhs);
      return dropout_p() == rhs_.dropout_p() &&
             rowscale_const() == rhs_.rowscale_const() &&
             x0_numrows() == rhs_.x0_numrows() &&
             has_residual() == rhs_.has_residual() &&
             is_rms_norm() == rhs_.is_rms_norm() &&
             input_indexs() == rhs_.input_indexs() &&
             output_indexs() == rhs_.output_indexs();
    } 
    else
      return false;
  }
};

TensorList MakeRMSNormGradientOp(Tensor dz, Tensor dx_, Tensor x, Tensor x0_,     
                                 Tensor dmask_, Tensor mu, Tensor rsigma, Tensor gamma,   
                                 Tensor rowscale_, Tensor colscale_, Tensor x0_subset_,  
                                 Tensor z_subset_, const float dropout_p, const float rowscale_const,
                                 const int64_t x0_numrows, const bool has_residual,
                                 bool is_rms_norm, OpMeta op_meta = OpMeta());

class FusedRMSNormOpImpl final : public OpInterface {
 private:
  friend class FusedRMSNormOp;
  struct constrcutor_access_key {};

 public:
  FusedRMSNormOpImpl(const HTShape& normalized_shape, double eps = 0.01, bool inplace = false)
  : OpInterface(quote(FusedRMSNormOp)),
  _normalized_shape(normalized_shape),
  _eps(eps),
  _inplace(inplace) {
  }

  HTShape normalized_shape() const {
    return _normalized_shape;
  }

  double get_eps() const {
    return _eps;
  }

  bool inplace() const {
    return _inplace;
  }

protected:
  std::vector<NDArrayMeta> 
  DoInferMeta(const TensorList& inputs) const override {
    HT_ASSERT_TENSORS_SAME_DTYPE(inputs);
    size_t dim = normalized_shape().size();
    HTShape output_shape = inputs[0]->shape();
    if (inputs[0]->has_shape()) 
      for (size_t i = 0; i < dim; ++i) {
        HT_ASSERT(normalized_shape()[dim - 1 - i] == inputs[0]->shape(inputs[0]->ndim() - 1 - i))
        << "Normalized shape's last dims should equal to input shape's.But we have normalized shape:"
        << normalized_shape() << " and input shape:" << inputs[0]->shape();
        output_shape[inputs[0]->ndim() - 1 - i] = 1;
      }
    NDArrayMeta output_meta = NDArrayMeta().set_dtype(kFloat32)
                                           .set_shape(output_shape)
                                           .set_device(inputs[0]->device());
    return {inputs[0]->meta(), output_meta};
  }

  void DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                      const OpMeta& op_meta) const override;

  void DoDeduceHeterProp(const std::vector<int32_t>& inputs_hetero_dim,
                         TensorList& outputs, const OpMeta& op_meta) const override;  

  TensorList DoGradient(Operator& op,
                        const TensorList& grad_outputs) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes,
                           RuntimeContext& runtime_ctx) const override;

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& runtime_ctx) const override;

  HTShape _normalized_shape;

  double _eps;

  bool _inplace;

 public:
  inline bool require_contig_inputs() const override {
    return false;
  }

  bool operator==(const OpInterface& rhs) const override {
    if (OpInterface::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const FusedRMSNormOpImpl&>(rhs);
      return (get_eps() == rhs_.get_eps()
              && normalized_shape() == rhs_.normalized_shape()); 
    }
    return false;
  }
};

TensorList MakeFusedRMSNormOp(Tensor input, Tensor ln_scale, HTShape normalized_shape, 
                              double eps = 0.01, bool inplace = false, OpMeta op_meta = OpMeta());

class FusedRMSNormGradientOpImpl final : public OpInterface {
 public:
  FusedRMSNormGradientOpImpl(HTShape normalized_shape, double eps, bool inplace)
  : OpInterface(quote(FusedRMSNormGradientOp)),
  _normalized_shape(normalized_shape),
  _eps(eps),
  _inplace(inplace) {
  }

  HTShape normalized_shape() const {
    return _normalized_shape;
  }

  double get_eps() const {
    return _eps;
  }

  bool inplace() const {
    return _inplace;
  }

protected:
  std::vector<NDArrayMeta> 
  DoInferMeta(const TensorList& inputs) const override {
    return {inputs[0]->meta(), inputs[2]->meta()};
  }

  void DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                      const OpMeta& op_meta) const override;  

  void DoDeduceHeterProp(const std::vector<int32_t>& inputs_hetero_dim,
                         TensorList& outputs, const OpMeta& op_meta) const override;  

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes,
                           RuntimeContext& runtime_ctx) const override;

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& runtime_ctx) const override;

  HTShape _normalized_shape;

  double _eps;

  bool _inplace;

 public:
  inline bool require_contig_inputs() const override {
    return false;
  }

  bool operator==(const OpInterface& rhs) const override {
    if (OpInterface::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const FusedRMSNormGradientOpImpl&>(rhs);
      return (get_eps() == rhs_.get_eps()); 
    }
    return false;
  }
};

TensorList MakeFusedRMSNormGradientOp(Tensor output_grad, Tensor input, Tensor ln_scale,
                                      Tensor save_var, HTShape normalized_shape,
                                      double eps, bool inplace, OpMeta op_meta = OpMeta());

} // namespace graph
} // namespace hetu
