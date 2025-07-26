#pragma once

#include "hetu/graph/operator.h"
#include "hetu/graph/utils/tensor_utils.h"

namespace hetu {
namespace graph {

class AttentionOpImpl;
class AttentionOp;
class AttentionGradientOpImpl;
class AttentionGradientOp;
class AttentionVarlenOpImpl;
class AttentionVarlenOp;
class AttentionVarlenGradientOpImpl;
class AttentionVarlenGradientOp;

class AttentionOpImpl final : public OpInterface {
 private:
  friend class AttentionOp;
  struct constructor_access_key {};

 public:
  AttentionOpImpl(double p_dropout, double softmax_scale, 
                  bool is_causal, bool return_softmax)
  : OpInterface(quote(AttentionOp)), _p_dropout(p_dropout), _softmax_scale(softmax_scale),
                                     _is_causal(is_causal), _return_softmax(return_softmax) {
  }

  inline double p_dropout() const {
    return _p_dropout;
  }

  inline double softmax_scale() const {
    return _softmax_scale;
  }

  inline bool is_causal() const {
    return _is_causal;
  }

  inline bool return_softmax() const {
    return _return_softmax;
  }

 protected:
  std::vector<NDArrayMeta> 
  DoInferMeta(const TensorList& inputs, const InstantiationContext& inst_ctx) const override {
    std::vector<NDArrayMeta> out_metas = {};
    NDArrayMeta base = inputs.at(0)->meta();
    // workaround
    // force ndarray meta to be contiguous
    // here set shape will refresh the stride and make it contiguouse
    base.set_shape(base.shape);
    out_metas.emplace_back(base);
    const int batch_size = inputs.at(0)->shape(0);
    const int seqlen_q = inputs.at(0)->shape(1);
    const int num_heads = inputs.at(0)->shape(2);
    const int head_size_og = inputs.at(0)->shape(3);
    const int seqlen_k = inputs.at(1)->shape(1);
    const int num_heads_k = inputs.at(1)->shape(2);
    HT_ASSERT(batch_size > 0)
    << "batch size must be postive";
    HT_ASSERT(head_size_og <= 256)
    << "FlashAttention forward only supports head dimension at most 256";
    HT_ASSERT(num_heads % num_heads_k == 0)
    << "Number of heads in key/value must divide number of heads in query";
    const int pad_len = head_size_og % 8 == 0 ? 0 : 8 - head_size_og % 8;
    HTShape padded_shape;
    for (int i = 0; i < 3; ++i) {
      padded_shape = inputs.at(i)->shape();
      padded_shape[3] += pad_len;
      out_metas.emplace_back(base.set_shape(padded_shape)); //q_padded, k_padded, v_padded.
    }
    padded_shape = inputs.at(0)->shape();
    padded_shape[3] += pad_len;
    out_metas.emplace_back(base.set_shape(padded_shape)); //out_padded
    out_metas.emplace_back(base.set_shape({batch_size, num_heads, seqlen_q}).set_dtype(kFloat)); //softmax_lse
    out_metas.emplace_back(base.set_shape({batch_size, num_heads, seqlen_q + pad_len, seqlen_k + pad_len})
                               .set_dtype(inputs.at(0)->dtype())); //p
    out_metas.emplace_back(base.set_shape({2}).set_device(kCPU).set_dtype(kInt64)); //rng_state
    return out_metas;
  };

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) const override;

  void DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                      const OpMeta& op_meta,
                      const InstantiationContext& inst_ctx) const override;

  TensorList DoGradient(Operator& op, const TensorList& grad_outputs) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes, RuntimeContext& ctx) const override;

  double _p_dropout;
  double _softmax_scale;
  bool _is_causal;
  bool _return_softmax;

 public:
  inline bool require_contig_inputs() const override {
    return false;
  }

  bool operator==(const OpInterface& rhs) const override {
    if (OpInterface::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const AttentionOpImpl&>(rhs);
      return p_dropout() == rhs_.p_dropout() &&
             softmax_scale() == rhs_.softmax_scale() &&
             is_causal() == rhs_.is_causal() &&
             return_softmax() == rhs_.return_softmax();
    } 
    else
      return false;
  }
};

TensorList MakeAttentionOp(Tensor q, Tensor k, Tensor v, double p_dropout = 0.0, double softmax_scale = -1.0, 
                           bool is_causal = false, bool return_softmax = false, OpMeta op_meta = OpMeta());

class AttentionGradientOpImpl final : public OpInterface {

 public:
  AttentionGradientOpImpl(double p_dropout, double softmax_scale, bool is_causal)
  : OpInterface(quote(AttentionGradientOp)), 
  _p_dropout(p_dropout), _softmax_scale(softmax_scale), _is_causal(is_causal) {
  }

  inline double p_dropout() const {
    return _p_dropout;
  }

  inline double softmax_scale() const {
    return _softmax_scale;
  }

  inline bool is_causal() const {
    return _is_causal;
  }

 protected:
  std::vector<NDArrayMeta> 
  DoInferMeta(const TensorList& inputs, const InstantiationContext& inst_ctx) const override {
    return {inputs[1]->meta(), inputs[2]->meta(), inputs[3]->meta()};
  };

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) const override;

  void DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                      const OpMeta& op_meta,
                      const InstantiationContext& inst_ctx) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes, RuntimeContext& ctx) const override;

  double _p_dropout;
  double _softmax_scale;
  bool _is_causal;

 public:
  inline bool require_contig_inputs() const override {
    return false;
  }

  bool operator==(const OpInterface& rhs) const override {
    if (OpInterface::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const AttentionGradientOpImpl&>(rhs);
      return p_dropout() == rhs_.p_dropout()
             && softmax_scale() == rhs_.softmax_scale()
             && is_causal() == rhs_.is_causal();
    } 
    else
      return false;
  }
};

TensorList MakeAttentionGradientOp(Tensor grad_out, Tensor q, Tensor k, Tensor v,
                                   Tensor out, Tensor softmax_lse, Tensor rng_state,
                                   double p_dropout = 0.0, double softmax_scale = -1.0,
                                   bool is_causal = false, OpMeta op_meta = OpMeta());

class AttentionVarlenOpImpl final : public OpInterface {
 private:
  friend class AttentionVarlenOp;
  struct constructor_access_key {};

 public:
  AttentionVarlenOpImpl(int max_seqlen_q, int max_seqlen_k, 
                        double p_dropout, double softmax_scale, 
                        bool zero_tensors, bool is_causal, 
                        bool return_softmax)
  : OpInterface(quote(AttentionVarlenOp)), _max_seqlen_q(max_seqlen_q), _max_seqlen_k(max_seqlen_k), 
                                           _p_dropout(p_dropout), _softmax_scale(softmax_scale),
                                           _zero_tensors(zero_tensors), _is_causal(is_causal), 
                                           _return_softmax(return_softmax) {
  }

  inline int max_seqlen_q() const {
    return _max_seqlen_q;
  }

  inline int max_seqlen_k() const {
    return _max_seqlen_k;
  }

  inline double p_dropout() const {
    return _p_dropout;
  }

  inline double softmax_scale() const {
    return _softmax_scale;
  }

  inline bool zero_tensors() const {
    return _zero_tensors;
  }

  inline bool is_causal() const {
    return _is_causal;
  }

  inline bool return_softmax() const {
    return _return_softmax;
  }

 protected:
  std::vector<NDArrayMeta> 
  DoInferMeta(const TensorList& inputs, const InstantiationContext& inst_ctx) const override {
    std::vector<NDArrayMeta> out_metas = {};
    NDArrayMeta base = inputs.at(0)->meta();
    out_metas.emplace_back(base);
    const int total_q = inputs.at(0)->shape(0);
    const int batch_size = inputs.at(3)->numel() - 1;
    const int num_heads = inputs.at(0)->shape(1);
    const int head_size_og = inputs.at(0)->shape(2);
    const int total_k = inputs.at(1)->shape(0);
    const int num_heads_k = inputs.at(1)->shape(1);
    HT_ASSERT(batch_size > 0)
    << "batch size must be postive";
    HT_ASSERT(head_size_og <= 256)
    << "FlashAttentionVarlen forward only supports head dimension at most 256";
    HT_ASSERT(num_heads % num_heads_k == 0)
    << "Number of heads in key/value must divide number of heads in query";
    const int pad_len = head_size_og % 8 == 0 ? 0 : 8 - head_size_og % 8;
    HTShape padded_shape;
    for (int i = 0; i < 3; ++i) {
      padded_shape = inputs.at(i)->shape();
      padded_shape[2] += pad_len;
      out_metas.emplace_back(base.set_shape(padded_shape)); //q_padded, k_padded, v_padded.
    }
    padded_shape = inputs.at(0)->shape();
    padded_shape[2] += pad_len;
    out_metas.emplace_back(base.set_shape(padded_shape)); //out_padded
    out_metas.emplace_back(base.set_shape({batch_size, num_heads, _max_seqlen_q}).set_dtype(kFloat)); //softmax_lse
    out_metas.emplace_back(base.set_shape({batch_size, num_heads, _max_seqlen_q + pad_len, _max_seqlen_k + pad_len})
                               .set_dtype(inputs.at(0)->dtype())); //p
    out_metas.emplace_back(base.set_shape({2}).set_device(kCPU).set_dtype(kInt64)); //rng_state
    return out_metas;
  };

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) const override;

  TensorList DoGradient(Operator& op, const TensorList& grad_outputs) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes, RuntimeContext& ctx) const override;

  int _max_seqlen_q;
  int _max_seqlen_k;
  double _p_dropout;
  double _softmax_scale;
  bool _zero_tensors;
  bool _is_causal;
  bool _return_softmax;

 public:
  bool operator==(const OpInterface& rhs) const override {
    if (OpInterface::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const AttentionVarlenOpImpl&>(rhs);
      return max_seqlen_q() == rhs_.max_seqlen_q() &&
             max_seqlen_k() == rhs_.max_seqlen_k() &&
             p_dropout() == rhs_.p_dropout() &&
             softmax_scale() == rhs_.softmax_scale() &&
             zero_tensors() == rhs_.zero_tensors() &&
             is_causal() == rhs_.is_causal() &&
             return_softmax() == rhs_.return_softmax();
    } 
    else
      return false;
  }
};

TensorList MakeAttentionVarlenOp(Tensor q, Tensor k, Tensor v, Tensor cu_seqlens_q, Tensor cu_seqlens_k,
                                 int max_seqlen_q, int max_seqlen_k, double p_dropout = 0.0, 
                                 double softmax_scale = -1.0, bool zero_tensors = false, bool is_causal = false, 
                                 bool return_softmax = false, OpMeta op_meta = OpMeta());

class AttentionVarlenGradientOpImpl final : public OpInterface {

 public:
  AttentionVarlenGradientOpImpl(int max_seqlen_q, int max_seqlen_k, 
                                double p_dropout, double softmax_scale, 
                                bool zero_tensors, bool is_causal)
  : OpInterface(quote(AttentionVarlenGradientOp)), 
  _p_dropout(p_dropout), _softmax_scale(softmax_scale), _is_causal(is_causal) {
  }

  inline int max_seqlen_q() const {
    return _max_seqlen_q;
  }

  inline int max_seqlen_k() const {
    return _max_seqlen_k;
  }

  inline double p_dropout() const {
    return _p_dropout;
  }

  inline double softmax_scale() const {
    return _softmax_scale;
  }

  inline bool zero_tensors() const {
    return _zero_tensors;
  }

  inline bool is_causal() const {
    return _is_causal;
  }

 protected:
  std::vector<NDArrayMeta> 
  DoInferMeta(const TensorList& inputs, const InstantiationContext& inst_ctx) const override {
    return {inputs[1]->meta(), inputs[2]->meta(), inputs[3]->meta()};
  };

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes, RuntimeContext& ctx) const override;

  int _max_seqlen_q;
  int _max_seqlen_k;
  double _p_dropout;
  double _softmax_scale;
  bool _zero_tensors;
  bool _is_causal;

 public:
  bool operator==(const OpInterface& rhs) const override {
    if (OpInterface::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const AttentionVarlenGradientOpImpl&>(rhs);
      return max_seqlen_q() == rhs_.max_seqlen_q() &&
             max_seqlen_k() == rhs_.max_seqlen_k() &&
             p_dropout() == rhs_.p_dropout() &&
             softmax_scale() == rhs_.softmax_scale() &&
             zero_tensors() == rhs_.zero_tensors() &&
             is_causal() == rhs_.is_causal();
    } 
    else
      return false;
  }
};

TensorList MakeAttentionVarlenGradientOp(Tensor grad_out, Tensor q, Tensor k, Tensor v, 
                                         Tensor cu_seqlens_q, Tensor cu_seqlens_k,
                                         Tensor out, Tensor softmax_lse, Tensor rng_state,
                                         int max_seqlen_q, int max_seqlen_k,
                                         double p_dropout = 0.0, double softmax_scale = -1.0,
                                         bool zero_tensors = false, bool is_causal = false, 
                                         OpMeta op_meta = OpMeta());
  
TensorList MakeAttentionPackedOp(Tensor qkv, double p_dropout = 0.0, double softmax_scale = -1.0, 
                                 bool is_causal = false, bool return_softmax = false, OpMeta op_meta = OpMeta());


TensorList MakeAttentionVarlenPackedOp(Tensor qkv, Tensor cu_seqlens_q, Tensor cu_seqlens_k,
                                       int max_seqlen_q, int max_seqlen_k, double p_dropout = 0.0, 
                                       double softmax_scale = -1.0, bool zero_tensors = false, bool is_causal = false, 
                                       bool return_softmax = false, OpMeta op_meta = OpMeta());

} // namespace graph
} // namespace hetu
