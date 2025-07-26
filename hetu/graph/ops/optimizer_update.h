#pragma once

#include "hetu/graph/operator.h"
#include "hetu/graph/optim/optimizerParamScheduler.h"

namespace hetu {
namespace graph {

class OptimizerUpdateOpInterface : public OpInterface {
 public:
  OptimizerUpdateOpInterface(OpType&& op_type, OptimizerParamScheduler param_scheduler)
  : OpInterface(std::move(op_type)), _param_scheduler(param_scheduler) {
  }

  uint64_t op_indicator() const noexcept override {
    return OPTIMIZER_UPDATE_OP;
  }

  bool inplace_at(size_t input_position) const override {
    // By default, the first input is parameter, the second is gradient,
    // and the rest are optimizer states.
    return input_position != 1;
  }

 protected:
  std::vector<NDArrayMeta>
  DoInferMeta(const TensorList& inputs, const InstantiationContext& inst_ctx) const override {
    // Question: should we check whether the param is trainable?
    HT_VALUE_ERROR_IF(!inputs.front()->producer()->is_parameter())
      << "The first input " << inputs.front() << " is not a parameter";
    return {inputs.front()->meta()};
  }

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes,
                           RuntimeContext& runtime_ctx) const override {
    return {input_shapes.front()};
  }

  NDArrayList DoAllocOutputs(Operator& op, const NDArrayList& inputs,
                             RuntimeContext& runtime_ctx) const override {
    // In place update
    return {inputs.front()};
  }

  bool DoMapToParallelDevices(Operator& op, const DeviceGroupUnion& placement_group_union) const override;

 public:
  inline bool require_contig_inputs() const override {
    return false;
  }

  bool operator==(const OpInterface& rhs) const override {
    if (OpInterface::operator==(rhs)) {
      const auto& rhs_ =
        reinterpret_cast<const OptimizerUpdateOpInterface&>(rhs);
      return param_scheduler() == rhs_.param_scheduler();
    }
    return false;
  }

  float learning_rate(int64_t step = 0) const {
    return _param_scheduler.get_lr(step);
  }

  OptimizerParamScheduler param_scheduler() const{
    return _param_scheduler;
  }


 protected:
  OptimizerParamScheduler _param_scheduler;
};

class SGDUpdateOpImpl final : public OptimizerUpdateOpInterface {
 public:
  SGDUpdateOpImpl(OptimizerParamScheduler param_scheduler)
  : OptimizerUpdateOpInterface(quote(SGDUpdateOp), param_scheduler) {}

 protected:
  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& runtime_ctx) const override;
};

class SGDUpdateWithGradScalerOpImpl final : public OptimizerUpdateOpInterface {
 public:
  SGDUpdateWithGradScalerOpImpl(OptimizerParamScheduler param_scheduler)
  : OptimizerUpdateOpInterface(quote(SGDUpdateWithGradScalerOp), param_scheduler) {}

 protected:
  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& runtime_ctx) const override;
};

class MomentumUpdateOpImpl final : public OptimizerUpdateOpInterface {
 public:
  MomentumUpdateOpImpl(OptimizerParamScheduler param_scheduler, float momentum, bool nesterov)
  : OptimizerUpdateOpInterface(quote(MomemtumUpdateOp), param_scheduler),
    _momentum(momentum),
    _nesterov(nesterov) {
    HT_VALUE_ERROR_IF(momentum < 0 || momentum > 1)
      << "Invalid momemtum: " << momentum;
  }

 protected:
  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& runtime_ctx) const override;

 public:
  bool operator==(const OpInterface& rhs) const override {
    if (OpInterface::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const MomentumUpdateOpImpl&>(rhs);
      return momentum() == rhs_.momentum() && nesterov() == rhs_.nesterov();
    }
    return false;
  }

  float momentum() const {
    return _momentum;
  }

  bool nesterov() const {
    return _nesterov;
  }

 protected:
  float _momentum;
  bool _nesterov;
};

class AdamOpImpl : public OptimizerUpdateOpInterface {
 public:
  AdamOpImpl(OptimizerParamScheduler param_scheduler, const std::vector<bool>& multi_zero = {false},
             float beta1 = 0.9, float beta2 = 0.999, 
             float eps = 1e-8)
  : OptimizerUpdateOpInterface(quote(AdamOp), param_scheduler),
    _multi_zero(multi_zero),
    _beta1(beta1),
    _beta2(beta2),
    _eps(eps){
    HT_VALUE_ERROR_IF(beta1 < 0 || beta1 > 1)
      << "Invalid beta1: " << beta1;
    HT_VALUE_ERROR_IF(beta2 < 0 || beta1 > 2)
      << "Invalid beta2: " << beta2;
  }

  uint64_t op_indicator() const noexcept override {
    return OPTIMIZER_UPDATE_OP | ADAM_OP;
  }  

 protected:
  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& runtime_ctx) const override;

  void DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                      const OpMeta& op_meta,
                      const InstantiationContext& inst_ctx) const override;

  void DoDeduceHeterProp(const std::vector<int32_t>& inputs_hetero_dim,
                         TensorList& outputs, const OpMeta& op_meta,
                         const InstantiationContext& inst_ctx) const override; 

  void DoSpecialMergeStrategy(Operator& op, Operator& another_op) override;

 public:
  bool operator==(const OpInterface& rhs) const override {
    if (OpInterface::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const AdamOpImpl&>(rhs);
      return beta1() == rhs_.beta1() && 
             beta2() == rhs_.beta2() && 
             eps() == rhs_.eps();
    }
    return false;
  }

  const std::vector<bool>& multi_zero() const {
    return _multi_zero;
  }

  float beta1() const {
    return _beta1;
  }

  float beta2() const {
    return _beta2;
  }

  float eps() const {
    return _eps;
  }

  float weight_decay(int64_t step) const{
    return _param_scheduler.get_wd(step);
  }

  const NDArray& adam_step() const {
    return _adam_step;
  }

 protected:
  std::vector<bool> _multi_zero;
  float _beta1;
  float _beta2;
  float _eps;
  NDArray _adam_step;
};

Tensor MakeSGDUpdateOp(Tensor param, Tensor grad, OptimizerParamScheduler param_scheduler,
                       OpMeta op_meta = OpMeta());

Tensor MakeSGDUpdateWithGradScalerOp(Tensor param, Tensor grad, Tensor infinite_count, OptimizerParamScheduler param_scheduler ,
                                     OpMeta op_meta = OpMeta());

Tensor MakeMomentumUpdateOp(Tensor param, Tensor grad, Tensor velocity,
                            OptimizerParamScheduler param_scheduler, float momentum, bool nesterov,
                            OpMeta op_meta = OpMeta());

Tensor MakeAdamOp(Tensor param, Tensor grad, 
                  Tensor mean, Tensor variance,
                  OptimizerParamScheduler param_scheduler, Tensor step, float beta1 = 0.9,
                  float beta2 = 0.999, float eps = 1e-8,
                  OpMeta op_meta = OpMeta());

} // namespace graph
} // namespace hetu