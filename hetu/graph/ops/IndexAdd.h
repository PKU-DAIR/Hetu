#pragma once

#include "hetu/graph/operator.h"
#include "hetu/graph/utils/tensor_utils.h"

namespace hetu {
namespace graph {

class IndexAddOpImpl;
class IndexAddOp;
class IndexAddGradientOpImpl;
class IndexAddGradientOp;

class IndexAddOpImpl final : public OpInterface {

 public:
  IndexAddOpImpl(int64_t dim)
  : OpInterface(quote(IndexAddOp)), _dim(dim), _symbolic(false) {
  }

  IndexAddOpImpl(int64_t dim, const SyShape& start_and_end_idx)
  : OpInterface(quote(IndexAddOp)), _dim(dim), _start_and_end_idx(start_and_end_idx), _symbolic(true) {
  }

  IndexAddOpImpl(int64_t dim, const HTShape& start_and_end_idx)
  : OpInterface(quote(IndexAddOp)), _dim(dim), _start_and_end_idx(_start_and_end_idx.begin(), _start_and_end_idx.end()), _symbolic(false) {
  }

  inline uint64_t op_indicator() const noexcept override {
    return INDEX_ADD_OP;
  }

  int64_t dim() const {
    return _dim;
  }

  HTShape start_and_end_idx() const {
    return get_HTShape_from_SyShape(_start_and_end_idx);
  }

  const SyShape& symbolic_start_and_end_idx() const {
    return _start_and_end_idx;
  }

  bool symbolic() const {
    return _symbolic;
  }

protected:
  std::vector<NDArrayMeta>
  DoInferMeta(const TensorList& inputs) const override {
    NDArrayMeta output_meta = inputs[0]->meta();
    return {output_meta};
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
                 RuntimeContext& runtime_ctx) const {};

  NDArrayList DoCompute(Operator& op, const NDArrayList& inputs,
                        RuntimeContext& runtime_ctx) const override;

  int64_t _dim;
  SyShape _start_and_end_idx{};
  bool _symbolic;

 public:
  inline bool require_contig_inputs() const override {
    return false;
  }

  bool operator==(const OpInterface& rhs) const override {
    if (OpInterface::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const IndexAddOpImpl&>(rhs);
      return (dim() == rhs_.dim() &&
              start_and_end_idx() == rhs_.start_and_end_idx() &&
              symbolic() == symbolic());
    }
    return false;
  }
};

Tensor MakeIndexAddOp(Tensor x, Tensor y, Tensor task_batch_idx, int64_t dim, OpMeta op_meta = OpMeta());

Tensor MakeIndexAddOp(Tensor x, Tensor y, int64_t dim, const SyShape& start_and_end_dix, OpMeta op_meta = OpMeta());

Tensor MakeIndexAddOp(Tensor x, Tensor y, int64_t dim, const HTShape& start_and_end_dix, OpMeta op_meta = OpMeta());

class IndexAddGradientOpImpl final : public OpInterface {

 public:
  IndexAddGradientOpImpl(int64_t dim, bool require_slice)
  : OpInterface(quote(IndexAddGradientOp)),
  _dim(dim), _require_slice(require_slice), _symbolic(false) {
  }

  IndexAddGradientOpImpl(int64_t dim, bool require_slice, const SyShape& start_and_end_idx)
  : OpInterface(quote(IndexAddGradientOp)),
  _dim(dim), _require_slice(require_slice), _start_and_end_idx(start_and_end_idx), _symbolic(true) {
  }

  IndexAddGradientOpImpl(int64_t dim, bool require_slice, const HTShape& start_and_end_idx)
  : OpInterface(quote(IndexAddGradientOp)),
  _dim(dim), _require_slice(require_slice), _start_and_end_idx(start_and_end_idx.begin(), start_and_end_idx.end()), _symbolic(false) {
  }

  int64_t dim() const {
    return _dim;
  }

  bool require_slice() const {
    return _require_slice;
  }

  HTShape start_and_end_idx() const {
    return get_HTShape_from_SyShape(_start_and_end_idx);
  }

  const SyShape& symbolic_start_and_end_idx() const {
    return _start_and_end_idx;
  }

  bool symbolic() const {
    return _symbolic;
  }

protected:
  std::vector<NDArrayMeta>
  DoInferMeta(const TensorList& inputs) const override {
    NDArrayMeta output_meta = inputs[1]->meta();
    return {output_meta};
  }

  void DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                      const OpMeta& op_meta) const override;

  void DoDeduceHeterProp(const std::vector<int32_t>& inputs_hetero_dim,
                         TensorList& outputs, const OpMeta& op_meta) const override;  

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes,
                           RuntimeContext& runtime_ctx) const override;

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& runtime_ctx) const override;
  
  NDArrayList DoCompute(Operator& op, const NDArrayList& inputs,
                        RuntimeContext& runtime_ctx) const override;

  int64_t _dim;
  SyShape _start_and_end_idx{};
  bool _require_slice;
  bool _symbolic;

 public:
  inline bool require_contig_inputs() const override {
    return false;
  }

  bool operator==(const OpInterface& rhs) const override {
    if (OpInterface::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const IndexAddGradientOpImpl&>(rhs);
      return (dim() == rhs_.dim() &&
              require_slice() == rhs_.require_slice() &&
              start_and_end_idx() == rhs_.start_and_end_idx());
    }
    return false;
  }
};

Tensor MakeIndexAddGradientOp(Tensor grad_output, Tensor x, Tensor task_batch_idx, int64_t dim, bool require_slice,
                              OpMeta op_meta = OpMeta());

Tensor MakeIndexAddGradientOp(Tensor grad_output, Tensor x, int64_t dim, bool require_slice, const SyShape& start_and_end_idx,
                              OpMeta op_meta = OpMeta());

Tensor MakeIndexAddGradientOp(Tensor grad_output, Tensor x, int64_t dim, bool require_slice, const HTShape& start_and_end_idx,
                              OpMeta op_meta = OpMeta());

} // namespace graph
} // namespace hetu
