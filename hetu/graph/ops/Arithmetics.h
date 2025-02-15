#pragma once

#include "hetu/graph/operator.h"
#include "hetu/graph/utils/tensor_utils.h"
#include "hetu/graph/ops/Unary.h"

namespace hetu {
namespace graph {

class BinaryOpImpl;
class BinaryGradientOpImpl;

class AddElewiseOpImpl;
class AddByConstOpImpl;

class SubElewiseOpImpl;
class SubByConstOpImpl;
class SubFromConstOpImpl;

class NegateOpImpl;

class MulElewiseOpImpl;
class MulByConstOpImpl;

class DivElewiseOpImpl;
class DivByConstOpImpl;
class DivFromConstOpImpl;

class ReciprocalOpImpl;

class AddElewiseGradientOpImpl;
class SubElewiseGradientOpImpl;
class MulElewiseGradientOpImpl;
class DivElewiseGradientOpImpl;

class BinaryOpImpl : public OpInterface {
 protected:
  BinaryOpImpl(OpType&& op_type, bool inplace)
  : OpInterface(std::move(op_type)), _inplace(inplace) {}
 
 public:
  inline bool require_contig_inputs() const override {
    return false;
  }

  inline bool inplace() const {
    return _inplace;
  }

  inline uint64_t inplace_pos() const {
    return 0;
  }

  inline bool inplace_at(size_t input_position) const override {
    return inplace() && input_position == inplace_pos();
  }

  inline uint64_t op_indicator() const noexcept override {
    return (_inplace ? INPLACE_OP : 0) | BINARY_OP;
  }

  bool operator==(const OpInterface& rhs) const override {
    if (OpInterface::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const BinaryOpImpl&>(rhs);
      return inplace() == rhs_.inplace();
    }
    return false;
  }

 protected:
  bool _inplace;
};

class BinaryGradientOpImpl : public OpInterface {
 protected:
  BinaryGradientOpImpl(OpType&& op_type, HTAxes axe,
                       HTKeepDims keep_dims, int index)
  : OpInterface(std::move(op_type)),
    _add_axes(axe),
    _keep_dims(keep_dims),
    _index(index) {}
 
 public:
  inline bool require_contig_inputs() const override {
    return false;
  }

  void set_axes(HTAxes axe) {
    _add_axes = axe;
  }

  void set_keep_dims(HTKeepDims keep_dims) {
    _keep_dims = keep_dims;
  }

  HTAxes axes() const {
    return _add_axes;
  }

  HTKeepDims keep_dims() const{
    return _keep_dims;
  }

  int index() const {
    return _index;
  }

  bool operator==(const OpInterface& rhs) const override {
    if (OpInterface::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const BinaryGradientOpImpl&>(rhs);
      return (index() == rhs_.index() &&
              keep_dims() == rhs_.keep_dims() &&
              axes() == rhs_.axes());
    }
    return false;
  }

 protected:
  std::vector<NDArrayMeta> 
  DoInferMeta(const TensorList& inputs, const InstantiationContext& inst_ctx) const override {
    // HT_ASSERT_TENSORS_SAME_DTYPE(inputs);
    NDArrayMeta output_meta = inputs[2]->meta();
    return {output_meta};
  }

  HTAxes _add_axes;
  HTKeepDims _keep_dims;
  int _index;
};

class AddElewiseOpImpl final : public BinaryOpImpl {
 public:
  AddElewiseOpImpl(bool inplace)
  : BinaryOpImpl(quote(AddElewiseOp), inplace) {
  }

 protected:
  std::vector<NDArrayMeta> 
  DoInferMeta(const TensorList& inputs, const InstantiationContext& inst_ctx) const override {
    HTShape shape = Broadcast(inputs[0]->shape(), inputs[1]->shape());
    if (inplace()) 
      HT_ASSERT(shape == inputs[0]->shape())
      << "inplace operator's output shape should be the same as input shape, but got "
      << shape << " and " << inputs[0]->shape();
    // HT_ASSERT_TENSORS_SAME_DTYPE(inputs);
    NDArrayMeta output_meta = NDArrayMeta().set_dtype(inputs[0]->dtype())
                                           .set_shape(shape)
                                           .set_device(inputs[0]->device());
    return {output_meta};
  }

  void DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                      const OpMeta& op_meta,
                      const InstantiationContext& inst_ctx) const override;

  TensorList DoGradient(Operator& op,
                        const TensorList& grad_outputs) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes,
                           RuntimeContext& runtime_ctx) const override;

  void DoSaveCtxForBackward(const TensorList& inputs, ContextStore& dst_ctx) const override;

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) const override;

  NDArrayList DoCompute(Operator& op,
                        const NDArrayList& inputs,
                        RuntimeContext& ctx) const override;

 public:
  bool operator==(const OpInterface& rhs) const override {
    return BinaryOpImpl::operator==(rhs);
  }
};

class AddByConstOpImpl final : public BinaryOpImpl {
 public:
  AddByConstOpImpl(DoubleSymbol value, bool inplace)
  : BinaryOpImpl(quote(AddByConstOp), inplace), _value(value) {
  }

  inline const DoubleSymbol& const_value_symbol() const {
    return _value;
  }

  inline double const_value() const {
    return _value->get_val();
  }

protected:
  std::vector<NDArrayMeta> 
  DoInferMeta(const TensorList& inputs, const InstantiationContext& inst_ctx) const override {
    NDArrayMeta output_meta = inputs.front()->meta();
    return {output_meta};
  }

  TensorList DoGradient(Operator& op,
                        const TensorList& grad_outputs) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes,
                           RuntimeContext& runtime_ctx) const override;

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) const override;
  
  DoubleSymbol _value;

 public:
  bool operator==(const OpInterface& rhs) const override {
    if (BinaryOpImpl::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const AddByConstOpImpl&>(rhs);
      return const_value() == rhs_.const_value();
    }
    return false;
  }
};

class SubElewiseOpImpl final : public BinaryOpImpl {
 public:
  SubElewiseOpImpl(bool inplace)
  : BinaryOpImpl(quote(SubElewiseOp), inplace) {
  }

 protected:
  std::vector<NDArrayMeta> 
  DoInferMeta(const TensorList& inputs, const InstantiationContext& inst_ctx) const override {
    HTShape shape = Broadcast(inputs[0]->shape(), inputs[1]->shape());
    if (inplace()) 
      HT_ASSERT(shape == inputs[0]->shape())
      << "inplace operator's output shape should be the same as input shape, but got "
      << shape << " and " << inputs[0]->shape();
    NDArrayMeta output_meta = NDArrayMeta().set_dtype(inputs[0]->dtype())
                                           .set_shape(shape)
                                           .set_device(inputs[0]->device());
    return {output_meta};
  }

  void DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                      const OpMeta& op_meta,
                      const InstantiationContext& inst_ctx) const override;  

  TensorList DoGradient(Operator& op,
                        const TensorList& grad_outputs) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes,
                           RuntimeContext& runtime_ctx) const override;

  void DoSaveCtxForBackward(const TensorList& inputs, ContextStore& dst_ctx) const override;

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) const override;

 public:
  bool operator==(const OpInterface& rhs) const override {
    return BinaryOpImpl::operator==(rhs);
  }
};

class SubByConstOpImpl final : public BinaryOpImpl {
 public:
  SubByConstOpImpl(DoubleSymbol value, bool inplace)
  : BinaryOpImpl(quote(SubByConstOp), inplace), _value(value) {
  }

  inline const DoubleSymbol& const_value_symbol() const {
    return _value;
  }

  inline double const_value() const {
    return _value->get_val();
  }

protected:
  std::vector<NDArrayMeta> 
  DoInferMeta(const TensorList& inputs, const InstantiationContext& inst_ctx) const override {
    NDArrayMeta output_meta = inputs.front()->meta();
    return {output_meta};
  }

  TensorList DoGradient(Operator& op,
                        const TensorList& grad_outputs) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes,
                           RuntimeContext& runtime_ctx) const override;

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) const override;
  
  DoubleSymbol _value;

 public:
  bool operator==(const OpInterface& rhs) const override {
    if (BinaryOpImpl::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const SubByConstOpImpl&>(rhs);
      return const_value() == rhs_.const_value();
    }
    return false;
  }
};

class SubFromConstOpImpl final : public BinaryOpImpl {
  public:
  SubFromConstOpImpl(DoubleSymbol value, bool inplace)
  : BinaryOpImpl(quote(SubFromConstOp), inplace), _value(value) {
  }

  inline const DoubleSymbol& const_value_symbol() const {
    return _value;
  }

  inline double const_value() const {
    return _value->get_val();
  }

protected:
  std::vector<NDArrayMeta> 
  DoInferMeta(const TensorList& inputs, const InstantiationContext& inst_ctx) const override {
    NDArrayMeta output_meta = inputs.front()->meta();
    return {output_meta};
  }

  TensorList DoGradient(Operator& op,
                        const TensorList& grad_outputs) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes,
                           RuntimeContext& runtime_ctx) const override;

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) const override;
  
  DoubleSymbol _value;

 public:
  bool operator==(const OpInterface& rhs) const override {
    if (BinaryOpImpl::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const SubFromConstOpImpl&>(rhs);
      return const_value() == rhs_.const_value();
    }
    return false;
  }
};

class NegateOpImpl final : public UnaryOpImpl {
 public:
  NegateOpImpl(bool inplace)
  : UnaryOpImpl(quote(NegateOp), inplace) {
  }

protected:
  TensorList DoGradient(Operator& op,
                        const TensorList& grad_outputs) const override;

  NDArrayList DoCompute(Operator& op,
                        const NDArrayList& inputs,
                        RuntimeContext& ctx) const override;

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& runtime_ctx) const override;

 public:
  bool operator==(const OpInterface& rhs) const override {
    return UnaryOpImpl::operator==(rhs);
  }
};

class MulElewiseOpImpl final : public BinaryOpImpl {
public:
  MulElewiseOpImpl(bool inplace)
  : BinaryOpImpl(quote(MulElewiseOp), inplace) {
  }

 protected:
  std::vector<NDArrayMeta> 
  DoInferMeta(const TensorList& inputs, const InstantiationContext& inst_ctx) const override {
    HTShape shape = Broadcast(inputs[0]->shape(), inputs[1]->shape());
    if (inplace()) 
      HT_ASSERT(shape == inputs[0]->shape())
      << "inplace operator's output shape should be the same as input shape, but got "
      << shape << " and " << inputs[0]->shape();
    NDArrayMeta output_meta = NDArrayMeta().set_dtype(inputs[0]->dtype())
                                           .set_shape(shape)
                                           .set_device(inputs[0]->device());
    return {output_meta};
  }

  void DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                      const OpMeta& op_meta,
                      const InstantiationContext& inst_ctx) const override;

  TensorList DoGradient(Operator& op,
                        const TensorList& grad_outputs) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes,
                           RuntimeContext& runtime_ctx) const override;

  void DoSaveCtxForBackward(const TensorList& inputs, ContextStore& dst_ctx) const override;

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) const override;

 public:
  bool operator==(const OpInterface& rhs) const override {
    return BinaryOpImpl::operator==(rhs);
  }
};

class MulByConstOpImpl final : public BinaryOpImpl {
 public:
  MulByConstOpImpl(DoubleSymbol value, bool inplace)
  : BinaryOpImpl(quote(MulByConstOp), inplace), _value(value) {
  }

  inline const DoubleSymbol& const_value_symbol() const {
    return _value;
  }

  inline double const_value() const {
    return _value->get_val();
  }

protected:
  std::vector<NDArrayMeta> 
  DoInferMeta(const TensorList& inputs, const InstantiationContext& inst_ctx) const override {
    NDArrayMeta output_meta = inputs.front()->meta();
    return {output_meta};
  }

  TensorList DoGradient(Operator& op,
                        const TensorList& grad_outputs) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes,
                           RuntimeContext& runtime_ctx) const override;

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) const override;
  
  DoubleSymbol _value;

 public:
  bool operator==(const OpInterface& rhs) const override {
    if (BinaryOpImpl::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const MulByConstOpImpl&>(rhs);
      return const_value() == rhs_.const_value();
    }
    return false;
  }
};

class DivElewiseOpImpl final : public BinaryOpImpl {
 public:
  DivElewiseOpImpl(bool inplace)
  : BinaryOpImpl(quote(DivElewiseOp), inplace) {
  }

 protected:
  std::vector<NDArrayMeta> 
  DoInferMeta(const TensorList& inputs, const InstantiationContext& inst_ctx) const override {
    HTShape shape = Broadcast(inputs[0]->shape(), inputs[1]->shape());
    if (inplace()) 
      HT_ASSERT(shape == inputs[0]->shape())
      << "inplace operator's output shape should be the same as input shape, but got "
      << shape << " and " << inputs[0]->shape();
    NDArrayMeta output_meta = NDArrayMeta().set_dtype(inputs[0]->dtype())
                                           .set_shape(shape)
                                           .set_device(inputs[0]->device());
    return {output_meta};
  }

  void DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                      const OpMeta& op_meta,
                      const InstantiationContext& inst_ctx) const override;

  TensorList DoGradient(Operator& op,
                        const TensorList& grad_outputs) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes,
                           RuntimeContext& runtime_ctx) const override;

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) const override;

  bool _inplace;
 public:
  bool operator==(const OpInterface& rhs) const override {
    return BinaryOpImpl::operator==(rhs);
  }
};

class DivByConstOpImpl final : public BinaryOpImpl {
 public:
  DivByConstOpImpl(DoubleSymbol value, bool inplace)
  : BinaryOpImpl(quote(DivByConstOp), inplace), _value(value) {
  }

  inline const DoubleSymbol& const_value_symbol() const {
    return _value;
  }

  inline double const_value() const {
    return _value->get_val();
  }

protected:
  std::vector<NDArrayMeta> 
  DoInferMeta(const TensorList& inputs, const InstantiationContext& inst_ctx) const override {
    NDArrayMeta output_meta = inputs.front()->meta();
    return {output_meta};
  }

  TensorList DoGradient(Operator& op,
                        const TensorList& grad_outputs) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes,
                           RuntimeContext& runtime_ctx) const override;

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) const override;
  
  DoubleSymbol _value;

 public:
  bool operator==(const OpInterface& rhs) const override {
    if (BinaryOpImpl::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const DivByConstOpImpl&>(rhs);
      return const_value() == rhs_.const_value();
    }
    return false;
  }
};


class DivFromConstOpImpl final : public BinaryOpImpl {
 public:
  DivFromConstOpImpl(DoubleSymbol value, bool inplace)
  : BinaryOpImpl(quote(DivFromConstOp), inplace), _value(value) {
  }

  inline const DoubleSymbol& const_value_symbol() const {
    return _value;
  }

  inline double const_value() const {
    return _value->get_val();
  }

protected:
  std::vector<NDArrayMeta> 
  DoInferMeta(const TensorList& inputs, const InstantiationContext& inst_ctx) const override {
    NDArrayMeta output_meta = inputs.front()->meta();
    return {output_meta};
  }

  TensorList DoGradient(Operator& op,
                        const TensorList& grad_outputs) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes,
                           RuntimeContext& runtime_ctx) const override;

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) const override;
  
  DoubleSymbol _value;

 public:
  bool operator==(const OpInterface& rhs) const override {
    if (BinaryOpImpl::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const DivFromConstOpImpl&>(rhs);
      return const_value() == rhs_.const_value();
    }
    return false;
  }
};

class ReciprocalOpImpl final : public UnaryOpImpl {
 public:
  ReciprocalOpImpl(bool inplace)
  : UnaryOpImpl(quote(ReciprocalOp), inplace) {
  }

protected:
  TensorList DoGradient(Operator& op,
                        const TensorList& grad_outputs) const override;

  NDArrayList DoCompute(Operator& op,
                        const NDArrayList& inputs,
                        RuntimeContext& ctx) const override;

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& runtime_ctx) const override;

 public:
  bool operator==(const OpInterface& rhs) const override {
    return UnaryOpImpl::operator==(rhs);
  }
};

class AddElewiseGradientOpImpl final : public BinaryGradientOpImpl {
 public:
  AddElewiseGradientOpImpl(HTAxes axe, HTKeepDims keep_dims, int index)
  : BinaryGradientOpImpl(quote(AddElewiseGradientOp), axe, keep_dims, index) {}

 protected:
  void DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                      const OpMeta& op_meta,
                      const InstantiationContext& inst_ctx) const override;  

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes,
                           RuntimeContext& runtime_ctx) const override;

  void DoLoadCtxForBackward(ContextStore& src_ctx, ContextStore& dst_ctx) const override;

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& runtime_ctx) const override;
  
  std::vector<NDArrayMeta> 
  DoInferMeta(const TensorList& inputs, const InstantiationContext& inst_ctx) const override {
    // HT_ASSERT_TENSORS_SAME_DTYPE(inputs);
    return {inst_ctx.get<NDArrayMeta>("in_meta")};
  }

 public:
  bool operator==(const OpInterface& rhs) const override {
    return BinaryGradientOpImpl::operator==(rhs);
  }
};

class SubElewiseGradientOpImpl final : public BinaryGradientOpImpl {
 public:
  SubElewiseGradientOpImpl(HTAxes axe, HTKeepDims keep_dims, int index)
  : BinaryGradientOpImpl(quote(SubElewiseGradientOp), axe, keep_dims, index) {}

 protected:
  void DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                      const OpMeta& op_meta,
                      const InstantiationContext& inst_ctx) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes,
                           RuntimeContext& runtime_ctx) const override;

  void DoLoadCtxForBackward(ContextStore& src_ctx, ContextStore& dst_ctx) const override;

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& runtime_ctx) const override;

  std::vector<NDArrayMeta> 
  DoInferMeta(const TensorList& inputs, const InstantiationContext& inst_ctx) const override {
    // HT_ASSERT_TENSORS_SAME_DTYPE(inputs);
    return {inst_ctx.get<NDArrayMeta>("in_meta")};
  }

 public:
  bool operator==(const OpInterface& rhs) const override {
    return BinaryGradientOpImpl::operator==(rhs);
  }
};

class MulElewiseGradientOpImpl final : public BinaryGradientOpImpl {
  public:
  MulElewiseGradientOpImpl(HTAxes axe, HTKeepDims keep_dims, int index)
  : BinaryGradientOpImpl(quote(MulElewiseGradientOp), axe, keep_dims, index) {}

 protected:
  void DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                      const OpMeta& op_meta,
                      const InstantiationContext& inst_ctx) const override;
  
  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes,
                           RuntimeContext& runtime_ctx) const override;

  void DoLoadCtxForBackward(ContextStore& src_ctx, ContextStore& dst_ctx) const override;

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& runtime_ctx) const override;

  std::vector<NDArrayMeta> 
  DoInferMeta(const TensorList& inputs, const InstantiationContext& inst_ctx) const override {
    // HT_ASSERT_TENSORS_SAME_DTYPE(inputs);
    return {inst_ctx.get<NDArrayMeta>("in_meta")};
  }

 public:
  bool operator==(const OpInterface& rhs) const override {
    return BinaryGradientOpImpl::operator==(rhs);
  }
};

class DivElewiseGradientOpImpl final : public BinaryGradientOpImpl {
  public:
  DivElewiseGradientOpImpl(HTAxes axe, HTKeepDims keep_dims, int index)
  : BinaryGradientOpImpl(quote(DivElewiseGradientOp), axe, keep_dims, index) {}

 protected:
  void DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                      const OpMeta& op_meta,
                      const InstantiationContext& inst_ctx) const override;
  
  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes,
                           RuntimeContext& runtime_ctx) const override;

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& runtime_ctx) const override;

  std::vector<NDArrayMeta> 
  DoInferMeta(const TensorList& inputs, const InstantiationContext& inst_ctx) const override {
    // HT_ASSERT_TENSORS_SAME_DTYPE(inputs);
    return {inputs.at(2)->meta()};
  }

 public:
  bool operator==(const OpInterface& rhs) const override {
    return BinaryGradientOpImpl::operator==(rhs);
  }
};

Tensor MakeAddElewiseOp(Tensor a, Tensor b,
                        OpMeta op_meta = OpMeta());

Tensor MakeSubElewiseOp(Tensor a, Tensor b,
                        OpMeta op_meta = OpMeta());

Tensor MakeMulElewiseOp(Tensor a, Tensor b,
                        OpMeta op_meta = OpMeta());

Tensor MakeDivElewiseOp(Tensor a, Tensor b,
                        OpMeta op_meta = OpMeta());

Tensor MakeAddByConstOp(Tensor input, double value,
                        OpMeta op_meta = OpMeta());

Tensor MakeAddByConstOp(Tensor input, DoubleSymbol value,
                        OpMeta op_meta = OpMeta());

Tensor MakeAddByConstOp(double value, Tensor input,
                        OpMeta op_meta = OpMeta());

Tensor MakeAddByConstOp(DoubleSymbol value, Tensor input,
                        OpMeta op_meta = OpMeta());

Tensor MakeSubByConstOp(Tensor input, double value,
                        OpMeta op_meta = OpMeta());

Tensor MakeSubByConstOp(Tensor input, DoubleSymbol value,
                        OpMeta op_meta = OpMeta());

Tensor MakeSubFromConstOp(double value, Tensor input,
                          OpMeta op_meta = OpMeta());

Tensor MakeSubFromConstOp(DoubleSymbol value, Tensor input,
                          OpMeta op_meta = OpMeta());

Tensor MakeMulByConstOp(Tensor input, double value,
                        OpMeta op_meta = OpMeta());

Tensor MakeMulByConstOp(Tensor input, DoubleSymbol value,
                        OpMeta op_meta = OpMeta());

Tensor MakeMulByConstOp(double value, Tensor input,
                        OpMeta op_meta = OpMeta());

Tensor MakeMulByConstOp(DoubleSymbol value, Tensor input,
                        OpMeta op_meta = OpMeta());

Tensor MakeDivByConstOp(Tensor input, double value,
                          OpMeta op_meta = OpMeta());

Tensor MakeDivByConstOp(Tensor input, DoubleSymbol value,
                          OpMeta op_meta = OpMeta());

Tensor MakeDivFromConstOp(double value, Tensor input,
                          OpMeta op_meta = OpMeta());

Tensor MakeDivFromConstOp(DoubleSymbol value, Tensor input,
                          OpMeta op_meta = OpMeta());

Tensor MakeAddElewiseInplaceOp(Tensor a, Tensor b,
                        OpMeta op_meta = OpMeta());

Tensor MakeSubElewiseInplaceOp(Tensor a, Tensor b,
                        OpMeta op_meta = OpMeta());

Tensor MakeMulElewiseInplaceOp(Tensor a, Tensor b,
                        OpMeta op_meta = OpMeta());

Tensor MakeDivElewiseInplaceOp(Tensor a, Tensor b,
                        OpMeta op_meta = OpMeta());

Tensor MakeAddByConstInplaceOp(Tensor input, double value,
                        OpMeta op_meta = OpMeta());

Tensor MakeAddByConstInplaceOp(Tensor input, DoubleSymbol value,
                        OpMeta op_meta = OpMeta());

Tensor MakeAddByConstInplaceOp(double value, Tensor input,
                        OpMeta op_meta = OpMeta());

Tensor MakeAddByConstInplaceOp(DoubleSymbol value, Tensor input,
                        OpMeta op_meta = OpMeta());

Tensor MakeSubByConstInplaceOp(Tensor input, double value,
                        OpMeta op_meta = OpMeta());

Tensor MakeSubByConstInplaceOp(Tensor input, DoubleSymbol value,
                        OpMeta op_meta = OpMeta());

Tensor MakeSubFromConstInplaceOp(double value, Tensor input,
                          OpMeta op_meta = OpMeta());

Tensor MakeSubFromConstInplaceOp(DoubleSymbol value, Tensor input,
                          OpMeta op_meta = OpMeta());

Tensor MakeMulByConstInplaceOp(Tensor input, double value,
                        OpMeta op_meta = OpMeta());

Tensor MakeMulByConstInplaceOp(Tensor input, DoubleSymbol value,
                        OpMeta op_meta = OpMeta());

Tensor MakeMulByConstInplaceOp(double value, Tensor input,
                        OpMeta op_meta = OpMeta());

Tensor MakeMulByConstInplaceOp(DoubleSymbol value, Tensor input,
                        OpMeta op_meta = OpMeta());

Tensor MakeDivByConstInplaceOp(Tensor input, double value,
                          OpMeta op_meta = OpMeta());

Tensor MakeDivByConstInplaceOp(Tensor input, DoubleSymbol value,
                          OpMeta op_meta = OpMeta());

Tensor MakeDivFromConstInplaceOp(double value, Tensor input,
                          OpMeta op_meta = OpMeta());

Tensor MakeDivFromConstInplaceOp(DoubleSymbol value, Tensor input,
                          OpMeta op_meta = OpMeta());

Tensor MakeAddElewiseGradientOp(Tensor a, Tensor input, Tensor output, int index, 
                                OpMeta op_meta = OpMeta());

Tensor MakeSubElewiseGradientOp(Tensor a, Tensor input, Tensor output, int index, 
                                OpMeta op_meta = OpMeta());

Tensor MakeMulElewiseGradientOp(Tensor a, Tensor b, Tensor input, Tensor output, int index, 
                                OpMeta op_meta = OpMeta());

Tensor MakeDivElewiseGradientOp(Tensor a, Tensor b, Tensor input, Tensor output, int index, 
                                OpMeta op_meta = OpMeta());

Tensor MakeNegateOp(Tensor input, OpMeta op_meta = OpMeta());

Tensor MakeNegateInplaceOp(Tensor input, OpMeta op_meta = OpMeta());

Tensor MakeReciprocalOp(Tensor input, OpMeta op_meta = OpMeta());

Tensor MakeReciprocalInplaceOp(Tensor input, OpMeta op_meta = OpMeta());

} // namespace graph
} // namespace hetu
