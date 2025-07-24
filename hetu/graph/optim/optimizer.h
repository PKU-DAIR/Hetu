#pragma once

#include "hetu/graph/headers.h"
#include "hetu/graph/ops/variable.h"
#include "hetu/graph/optim/optimizerParamScheduler.h"

namespace hetu {
namespace graph {




class Optimizer {
 public:
  Optimizer() {};

  Optimizer(double init_lr, double max_lr, double min_lr,
        int64_t lr_warmup_steps, int64_t lr_decay_steps, std::string lr_decay_style,
        double start_wd = 0, double end_wd = 0, int wd_incr_steps = -1, std::string wd_incr_style = "constant"){
            _param_scheduler = OptimizerParamScheduler(init_lr, max_lr, min_lr, lr_warmup_steps, lr_decay_steps, 
              lr_decay_style, start_wd, end_wd, wd_incr_steps, wd_incr_style);
        }

  Optimizer(TensorList params, double init_lr, double max_lr, double min_lr,
        int64_t lr_warmup_steps, int64_t lr_decay_steps, std::string lr_decay_style,
        double start_wd = 0, double end_wd = 0, int wd_incr_steps = -1, std::string wd_incr_style = "constant")
  : _params(std::move(params)) {
    _param_scheduler = OptimizerParamScheduler(init_lr, max_lr, min_lr, lr_warmup_steps, lr_decay_steps, 
      lr_decay_style, start_wd, end_wd, wd_incr_steps, wd_incr_style);    
    HT_VALUE_ERROR_IF(_params.empty()) << "No parameters are provided";
    for (auto& param : _params) {
      HT_VALUE_ERROR_IF(!param->is_parameter())
        << "Tensor " << param << " is not a parameter";
    }
  }

  Tensor Minimize(const Tensor& loss, const TensorList& var_list = {},
                  const Tensor& grad_loss = {}, const OpName& name = "");
  
  StateDict GetStates(const Tensor& var);
  
  void SetStates(const Tensor& var, const OpName state_name, const NDArray& value);

  virtual GradAndVarList ComputeGradients(const Tensor& loss,
                                          const TensorList& var_list = {},
                                          const Tensor& grad_loss = {});

  virtual Tensor ApplyGradients(const GradAndVarList& grads_and_vars,
                                const OpName& name = OpName(),
                                const Tensor& infinite_count = Tensor());

  OptimizerParamScheduler param_scheduler() const{
    return _param_scheduler;
  }

  double learning_rate(int64_t step_num = 0) const{
    return _param_scheduler.get_lr(step_num);
  }

 protected:
  virtual Tensor ApplyDense(const GradAndVar& grad_and_var, const Tensor& infinite_count = Tensor()) { return Tensor(); }

  virtual void ApplyZero(const GradAndVarList& grads_and_vars);

  virtual Tensor MakeStates(const Tensor& variable, const Tensor& grad, const OpName& state_name);

  TensorList _params;
  OptimizerParamScheduler _param_scheduler;
  std::unordered_map<TensorId, StateDict> state_dict;
};

class SGDOptimizer : public Optimizer {
 public:
  SGDOptimizer(): Optimizer() {};

  SGDOptimizer(double init_lr, double max_lr, double min_lr,
        int lr_warmup_steps, int lr_decay_steps, std::string lr_decay_style,
        float momentum = 0.9f, bool nesterov = false)
  : Optimizer(init_lr, max_lr, min_lr,
        lr_warmup_steps, lr_decay_steps, lr_decay_style,
        0, 0, -1,  "constant") {
    HT_ASSERT(lr_decay_style == "constant");
    _init(momentum, nesterov);
  }

  SGDOptimizer(TensorList params, double init_lr, double max_lr, double min_lr,
        int lr_warmup_steps, int lr_decay_steps, std::string lr_decay_style,
        float momentum = 0.9f, bool nesterov = false)
  : Optimizer(init_lr, max_lr, min_lr,
        lr_warmup_steps, lr_decay_steps, lr_decay_style,
        0, 0, -1,  "constant")  {
    HT_ASSERT(lr_decay_style == "constant");  
    _init(momentum, nesterov);
  }

  Tensor ApplyDense(const GradAndVar& grad_and_var, const Tensor& infinite_count = Tensor());

  float momentum() const {
    return _momentum;
  }

  bool nesterov() const {
    return _nesterov;
  }

 protected:
  void _init(float momentum, bool nesterov) {
    HT_VALUE_ERROR_IF(momentum < 0 || momentum > 1)
      << "Invalid momemtum: " << momentum;
    _momentum = momentum;
    _nesterov = nesterov;
  }

  float _momentum;
  bool _nesterov;
};

class AdamOptimizer : public Optimizer {
 public:
  AdamOptimizer(): Optimizer() {};

  AdamOptimizer(double init_lr, double max_lr, double min_lr,
                int lr_warmup_steps, int lr_decay_steps,  std::string lr_decay_style,
                double start_wd, double end_wd, int wd_incr_steps,  std::string wd_incr_style,
                float beta1 = 0.9, float beta2 = 0.999, float eps = 1e-8)
  : Optimizer(init_lr, max_lr, min_lr,
        lr_warmup_steps, lr_decay_steps, lr_decay_style,
        start_wd, end_wd, wd_incr_steps,  wd_incr_style)   {
    _init(beta1, beta2, eps);
  }

  AdamOptimizer(TensorList params, double init_lr, double max_lr, double min_lr,
        int lr_warmup_steps, int lr_decay_steps, const std::string& lr_decay_style,
        double start_wd, double end_wd, int wd_incr_steps, const std::string& wd_incr_style,
        float beta1 = 0.9, float beta2 = 0.999, float eps = 1e-8)
  : Optimizer(init_lr, max_lr, min_lr,
        lr_warmup_steps, lr_decay_steps, lr_decay_style,
        start_wd, end_wd, wd_incr_steps,  wd_incr_style)  {
    _init(beta1, beta2, eps);
  }

  Tensor ApplyDense(const GradAndVar& grad_and_var, const Tensor& infinite_count = Tensor());

  float beta1() const {
    return _beta1;
  }

  float beta2() const {
    return _beta2;
  }

  float eps() const {
    return _eps;
  }

 protected:
  void _init(float beta1, float beta2, float eps) {
    HT_VALUE_ERROR_IF(beta1 < 0 || beta1 > 1)
      << "Invalid beta1: " << beta1;
    HT_VALUE_ERROR_IF(beta2 < 0 || beta1 > 2)
      << "Invalid beta2: " << beta2;
    _beta1 = beta1;
    _beta2 = beta2;
    _eps = eps;
  }

  float _beta1;
  float _beta2;
  float _eps;
};

} // namespace graph
} // namespace hetu
