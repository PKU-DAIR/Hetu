#pragma once

#include "hetu/graph/operator.h"
#include "hetu/graph/utils/tensor_utils.h"
#include "hetu/graph/ops/Loss.h"

namespace hetu {
namespace graph {

class RotaryOpImpl;
class RotaryOp;
class RotaryGradientOpImpl;
class RotaryGradientOp;

class RotaryOpImpl final : public OpInterface {
 public:
  RotaryOpImpl(int64_t head_dim, int64_t group_query_ratio,
    SyShapeList multi_seq_lens_symbol, SyShapeList multi_cp_group_symbol,
    bool packing, IntSymbol max_seqlen_q, IntSymbol max_seqlen_k,
    bool interleaved, bool inplace)
  : OpInterface(quote(RotaryOp)), 
    _head_dim(head_dim), _group_query_ratio(group_query_ratio), _multi_seq_lens_symbol(std::move(multi_seq_lens_symbol)), _multi_cp_group_symbol(std::move(multi_cp_group_symbol)),
    _packing(packing), _max_seqlen_q(std::move(max_seqlen_q)), _max_seqlen_k(std::move(max_seqlen_k)),
    _interleaved(interleaved), _inplace(inplace) {}

  inline int64_t head_dim() const {
    return _head_dim;
  }

  inline int64_t group_query_ratio() const {
    return _group_query_ratio;
  }

  inline int64_t max_seqlen_q() const {
    return _max_seqlen_q->get_val();
  }

  inline int64_t max_seqlen_k() const {
    return _max_seqlen_k->get_val();
  }

  inline bool interleaved() const {
    return _interleaved;
  }

  inline bool inplace() const {
    return _inplace;
  }

 protected:
  std::vector<NDArrayMeta> 
  DoInferMeta(const TensorList& inputs, const InstantiationContext& inst_ctx) const override {
    HT_ASSERT(inputs.at(0)->ndim() == 2);
    HT_ASSERT(inputs.at(1)->ndim() == 2);
    HT_ASSERT(inputs.at(2)->ndim() == 2);
    return {inputs.at(0)->meta()};
  };

  NDArrayList DoCompute(Operator& op,
                        const NDArrayList& inputs,
                        RuntimeContext& ctx) const override;

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) const override;

  TensorList DoGradient(Operator& op, const TensorList& grad_outputs) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes, RuntimeContext& ctx) const override;

  void DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                      const OpMeta& op_meta, const InstantiationContext& inst_ctx) const override;

  void DoDeduceHeterProp(const std::vector<int32_t>& inputs_hetero_dim,
                         TensorList& outputs, const OpMeta& op_meta, const InstantiationContext& inst_ctx) const override;  

 public:
  bool operator==(const OpInterface& rhs) const override {
    if (OpInterface::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const RotaryOpImpl&>(rhs);
      return  _head_dim == rhs_.head_dim() 
              && _group_query_ratio == rhs_.group_query_ratio() 
              && interleaved() == rhs_.interleaved()
              && inplace() == rhs_.inplace();
    } else
      return false;
  }

  int64_t _head_dim;
  int64_t _group_query_ratio;
  SyShapeList _multi_seq_lens_symbol;
  SyShapeList _multi_cp_group_symbol;
  bool _packing;
  IntSymbol _max_seqlen_q;
  IntSymbol _max_seqlen_k;
  bool _interleaved;
  bool _inplace;
};

Tensor MakeRotaryOp(Tensor x, Tensor cos, Tensor sin, int64_t head_dim, int64_t group_query_ratio,
                    SyShapeList multi_seq_lens_symbol, SyShapeList multi_cp_group_symbol, 
                    bool packing, Tensor cu_seqlens_q, Tensor cu_seqlens_k, IntSymbol max_seqlen_q, IntSymbol max_seqlen_k,
                    bool interleaved = false, bool inplace = false,
                    OpMeta op_meta = OpMeta());

class RotaryGradientOpImpl final : public OpInterface {
 public:
  RotaryGradientOpImpl(int64_t head_dim, int64_t group_query_ratio,
    SyShapeList multi_seq_lens_symbol, SyShapeList multi_cp_group_symbol,
    bool packing, IntSymbol max_seqlen_q, IntSymbol max_seqlen_k,
    bool interleaved, bool inplace)
  : OpInterface(quote(RotaryGradientOp)), 
    _head_dim(head_dim), _group_query_ratio(group_query_ratio), _multi_seq_lens_symbol(std::move(multi_seq_lens_symbol)), _multi_cp_group_symbol(std::move(multi_cp_group_symbol)),
    _packing(packing), _max_seqlen_q(std::move(max_seqlen_q)), _max_seqlen_k(std::move(max_seqlen_k)),
    _interleaved(interleaved), _inplace(inplace) {}

  inline int64_t head_dim() const {
    return _head_dim;
  }

  inline int64_t group_query_ratio() const {
    return _group_query_ratio;
  }

  inline int64_t max_seqlen_q() const {
    return _max_seqlen_q->get_val();
  }

  inline int64_t max_seqlen_k() const {
    return _max_seqlen_k->get_val();
  }

  inline bool interleaved() const {
    return _interleaved;
  }

  inline bool inplace() const {
    return _inplace;
  }

 protected:
  std::vector<NDArrayMeta> 
  DoInferMeta(const TensorList& inputs, const InstantiationContext& inst_ctx) const override {
    return {inputs[0]->meta()};
  };

  NDArrayList DoCompute(Operator& op,
                        const NDArrayList& inputs,
                        RuntimeContext& ctx) const override;

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes, RuntimeContext& ctx) const override;

  void DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                      const OpMeta& op_meta, const InstantiationContext& inst_ctx) const override;

  void DoDeduceHeterProp(const std::vector<int32_t>& inputs_hetero_dim,
                         TensorList& outputs, const OpMeta& op_meta, const InstantiationContext& inst_ctx) const override;  

 public:
  bool operator==(const OpInterface& rhs) const override {
    if (OpInterface::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const RotaryGradientOpImpl&>(rhs);
      return  _head_dim == rhs_.head_dim() 
              && _group_query_ratio == rhs_.group_query_ratio() 
              && interleaved() == rhs_.interleaved()
              && inplace() == rhs_.inplace();
    } else
      return false;
  }

  int64_t _head_dim;
  int64_t _group_query_ratio;
  SyShapeList _multi_seq_lens_symbol;
  SyShapeList _multi_cp_group_symbol;
  bool _packing;
  IntSymbol _max_seqlen_q;
  IntSymbol _max_seqlen_k;
  bool _interleaved;
  bool _inplace;
};

Tensor MakeRotaryGradientOp(Tensor dout, Tensor cos, Tensor sin, int64_t head_dim, int64_t group_query_ratio,
                            SyShapeList multi_seq_lens_symbol, SyShapeList multi_cp_group_symbol, 
                            bool packing, Tensor cu_seqlens_q, Tensor cu_seqlens_k, IntSymbol max_seqlen_q, IntSymbol max_seqlen_k,
                            bool interleaved = false, bool inplace = false, OpMeta op_meta = OpMeta());

} // namespace graph
} // namespace hetu
