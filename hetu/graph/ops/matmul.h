#pragma once

#include "hetu/graph/operator.h"
#include "hetu/graph/utils/tensor_utils.h"
#include <iterator>

namespace hetu {
namespace graph {

class MatMulOpImpl;
class MatMulGradientOpImpl;

class MatMulOpImpl final : public OpInterface {
 public:
  MatMulOpImpl(bool trans_a, bool trans_b)
  : OpInterface(quote(MatMulOp)), _trans_a(trans_a), _trans_b(trans_b) {}

 protected:
  std::vector<NDArrayMeta>
  DoInferMeta(const TensorList& inputs, const InstantiationContext& inst_ctx) const override {
    const Tensor& a = inputs.at(0);
    const Tensor& b = inputs.at(1);
    const auto dim_a = a->ndim();
    const auto dim_b = b->ndim();
    int64_t m_a = -1;
    int64_t m_b = -1;
    HTShape shape;

    if (dim_a == 1 && dim_b == 1) {
      m_a = a->shape(0);
      m_b = b->shape(0);
      HT_ASSERT(m_a == m_b)
        << "Failed to construct the \"MatMul\" op: "
        << "Dimensions must be compatible. "
        << "Got " << m_a << " vs. " << m_b << ". "
        << "Input shapes: " << a->shape() << " vs. " << b->shape() << ".";
      shape = HTShape();
    } else if (dim_a == 1 && dim_b == 2) {
      m_a = a->shape(0);
      m_b = b->shape(trans_b() ? 1 : 0);
      HT_ASSERT(m_a == m_b)
        << "Failed to construct the \"MatMul\" op: "
        << "Dimensions must be compatible. "
        << "Got " << m_a << " vs. " << m_b << ". "
        << "Input shapes: " << a->shape() << " vs. " << b->shape() << ".";
      shape = HTShape({b->shape(trans_b() ? 0 : 1)});
    } else if (dim_a == 2 && dim_b == 1) {
      m_a = a->shape(trans_a() ? 0 : 1);
      m_b = b->shape(0);
      HT_ASSERT(m_a == m_b)
        << "Failed to construct the \"MatMul\" op: "
        << "Dimensions must be compatible. "
        << "Got " << m_a << " vs. " << m_b << ". "
        << "Input shape: " << a->shape() << " vs. " << b->shape() << ". ";
      shape = HTShape({a->shape(trans_a() ? 1 : 0)});
    } else if (dim_a == 2 && dim_b == 2) {
      m_a = a->shape(trans_a() ? 0 : 1);
      m_b = b->shape(trans_b() ? 1 : 0);
      HT_ASSERT(m_a == - 1 || m_b == -1 || m_a == m_b)
        << "Failed to construct the \"MatMul\" op: "
        << "Dimensions must be compatible. "
        << "Got " << m_a << " vs. " << m_b << ". "
        << "Input shape: " << a->shape() << " vs. " << b->shape() << ". ";
      shape = HTShape({a->shape(trans_a() ? 1 : 0), b->shape(trans_b() ? 0 : 1)});
    } else if ((dim_a >= 3 && (dim_b == 1 || dim_b == 2))
            || ((dim_a == 1 || dim_a == 2) && dim_b >= 3)) {
      const auto transpose = dim_b > dim_a;
      const auto a_ = transpose ? b : a;
      const auto b_ = transpose ? a : b;
      const auto dim_a_ = transpose ? dim_b : dim_a;
      const auto dim_b_ = transpose ? dim_a : dim_b;
      const auto a_trans = transpose ? !trans_a() : trans_a();
      const auto b_trans = transpose ? !trans_b() : trans_b();

      m_a = a_->shape(a_trans ? dim_a_ - 2 : dim_a_ - 1);
      m_b = b_->shape(!b_trans ? 0
                               : dim_b_ == 2 ? 1 : 0);
      HT_ASSERT(m_a == - 1 || m_b == -1 || m_a == m_b)
        << "Failed to construct the \"MatMul\" op: "
        << "Dimensions must be compatible. "
        << "Got " << m_a << " vs. " << m_b << ". "
        << "Input shape: " << a->shape() << " vs. " << b->shape() << ". ";
      auto output_shape = HTShape(a_->shape().begin(), a_->shape().end() - 2);
      if (a_trans) {
        output_shape.emplace_back(a_->shape(dim_a_ - 1));
      } else {
        output_shape.emplace_back(a_->shape(dim_a_ - 2));
      }
      if (dim_b_ == 2) {
        if (b_trans) {
          output_shape.emplace_back(b_->shape(dim_b_ - 2));
        } else {
          output_shape.emplace_back(b_->shape(dim_b_ - 1));
        }
      }
      if (dim_b_ == 2 && transpose) {
        std::iter_swap(output_shape.end() - 2, output_shape.end() - 1);
      }
      shape = output_shape;
    } else {
      m_a = a->shape(trans_a() ? dim_a - 2 : dim_a - 1);
      m_b = b->shape(trans_b() ? dim_b - 1 : dim_b - 2);
      HT_ASSERT(m_a == - 1 || m_b == -1 || m_a == m_b)
        << "Failed to construct the \"MatMul\" op: "
        << "Dimensions must be compatible. "
        << "Got " << m_a << " vs. " << m_b << ". "
        << "Input shape: " << a->shape() << " vs. " << b->shape() << ". ";
      const auto a_shape = a->shape();
      const auto b_shape = b->shape();
      const auto batch_shape_a = HTShape(a_shape.begin(), a_shape.end() - 2);
      const auto batch_shape_b = HTShape(b_shape.begin(), b_shape.end() - 2);
      auto output_shape = NDArrayMeta::Broadcast(batch_shape_a, batch_shape_b);
      output_shape.emplace_back(trans_a() ? a_shape.back() : a_shape.cend()[-2]);
      output_shape.emplace_back(trans_b() ? b_shape.cend()[-2] : b_shape.back());
      shape = output_shape;
    }
    auto dst_dtype = (a->dtype() == kFloat4 || a->dtype() == kNFloat4)
                     ? b->dtype()
                     : a->dtype();
    return {NDArrayMeta().set_dtype(dst_dtype).set_shape(shape)};
  }

  void DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                      const OpMeta& op_meta,
                      const InstantiationContext& inst_ctx) const override;
  
  void DoDeduceHeterProp(const std::vector<int32_t>& inputs_hetero_dim,
                         TensorList& outputs, const OpMeta& op_meta,
                         const InstantiationContext& inst_ctx) const override;  

  TensorList DoGradient(Operator& op,
                        const TensorList& grad_outputs) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes,
                           RuntimeContext& runtime_ctx) const override {
    const HTShape& a = input_shapes.at(0);
    const HTShape& b = input_shapes.at(1);
    const auto dim_a = a.size();
    const auto dim_b = b.size();
    int64_t m_a = -1;
    int64_t m_b = -1;
    HTShape shape;

    if (dim_a == 1 && dim_b == 1) {
      m_a = a[0];
      m_b = b[0];
      HT_ASSERT(m_a == m_b)
        << "Failed to construct the \"MatMul\" op: "
        << "Dimensions must be compatible. "
        << "Got " << m_a << " vs. " << m_b << ". "
        << "Input shapes: " << a << " vs. " << b << ".";
      shape = HTShape();
    } else if (dim_a == 1 && dim_b == 2) {
      m_a = a[0];
      m_b = b[trans_b() ? 1 : 0];
      HT_ASSERT(m_a == m_b)
        << "Failed to construct the \"MatMul\" op: "
        << "Dimensions must be compatible. "
        << "Got " << m_a << " vs. " << m_b << ". "
        << "Input shapes: " << a << " vs. " << b << ".";
      shape = HTShape({b[trans_b() ? 0 : 1]});
    } else if (dim_a == 2 && dim_b == 1) {
      m_a = a[trans_a() ? 0 : 1];
      m_b = b[0];
      HT_ASSERT(m_a == m_b)
        << "Failed to construct the \"MatMul\" op: "
        << "Dimensions must be compatible. "
        << "Got " << m_a << " vs. " << m_b << ". "
        << "Input shape: " << a << " vs. " << b << ". ";
      shape = HTShape({a[trans_a() ? 1 : 0]});
    } else if (dim_a == 2 && dim_b == 2) {
      m_a = a[trans_a() ? 0 : 1];
      m_b = b[trans_b() ? 1 : 0];
      HT_ASSERT(m_a == - 1 || m_b == -1 || m_a == m_b)
        << "Failed to construct the \"MatMul\" op: "
        << "Dimensions must be compatible. "
        << "Got " << m_a << " vs. " << m_b << ". "
        << "Input shape: " << a << " vs. " << b << ". ";
      shape = HTShape({a[trans_a() ? 1 : 0], b[trans_b() ? 0 : 1]});
    } else if ((dim_a >= 3 && (dim_b == 1 || dim_b == 2))
            || ((dim_a == 1 || dim_a == 2) && dim_b >= 3)) {
      const auto transpose = dim_b > dim_a;
      const auto a_ = transpose ? b : a;
      const auto b_ = transpose ? a : b;
      const auto dim_a_ = transpose ? dim_b : dim_a;
      const auto dim_b_ = transpose ? dim_a : dim_b;
      const auto a_trans = transpose ? !trans_a() : trans_a();
      const auto b_trans = transpose ? !trans_b() : trans_b();

      m_a = a_[a_trans ? dim_a_ - 2 : dim_a_ - 1];
      m_b = b_[!b_trans ? 0
                        : dim_b_ == 2 ? 1 : 0];
      HT_ASSERT(m_a == - 1 || m_b == -1 || m_a == m_b)
        << "Failed to construct the \"MatMul\" op: "
        << "Dimensions must be compatible. "
        << "Got " << m_a << " vs. " << m_b << ". "
        << "Input shape: " << a << " vs. " << b << ". ";
      auto output_shape = HTShape(a_.begin(), a_.end() - 2);
      if (a_trans) {
        output_shape.emplace_back(a_[dim_a_ - 1]);
      } else {
        output_shape.emplace_back(a_[dim_a_ - 2]);
      }
      if (dim_b_ == 2) {
        if (b_trans) {
          output_shape.emplace_back(b_[dim_b_ - 2]);
        } else {
          output_shape.emplace_back(b_[dim_b_ - 1]);
        }
      }
      if (dim_b_ == 2 && transpose) {
        std::iter_swap(output_shape.end() - 2, output_shape.end() - 1);
      }
      shape = output_shape;
    } else {
      m_a = a[trans_a() ? dim_a - 2 : dim_a - 1];
      m_b = b[trans_b() ? dim_b - 1 : dim_b - 2];
      HT_ASSERT(m_a == - 1 || m_b == -1 || m_a == m_b)
        << "Failed to construct the \"MatMul\" op: "
        << "Dimensions must be compatible. "
        << "Got " << m_a << " vs. " << m_b << ". "
        << "Input shape: " << a << " vs. " << b << ". ";
      const auto batch_shape_a = HTShape(a.begin(), a.end() - 2);
      const auto batch_shape_b = HTShape(b.begin(), b.end() - 2);
      auto output_shape = NDArrayMeta::Broadcast(batch_shape_a, batch_shape_b);
      output_shape.emplace_back(trans_a() ? a.back() : a.cend()[-2]);
      output_shape.emplace_back(trans_b() ? b.cend()[-2] : b.back());
      shape = output_shape;
    }
    return {shape};
  }

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& runtime_ctx) const override {
    NDArray::matmul(inputs.at(0), inputs.at(1), trans_a(), trans_b(),
                    op->instantiation_ctx().stream_index, outputs.front());
  }

 public:
  inline bool require_contig_inputs() const override {
    return false;
  }

  bool operator==(const OpInterface& rhs) const override {
    if (OpInterface::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const MatMulOpImpl&>(rhs);
      return trans_a() == rhs_.trans_a() && trans_b() == rhs_.trans_b();
    }
    return false;
  }

  bool trans_a() const {
    return _trans_a;
  }

  bool trans_b() const {
    return _trans_b;
  }

 protected:
  bool _trans_a;
  bool _trans_b;
};

Tensor MakeMatMulOp(Tensor a, Tensor b, bool trans_a = false,
                    bool trans_b = false, OpMeta op_meta = OpMeta());

class MatMulGradientOpImpl final : public OpInterface {
  public:
   MatMulGradientOpImpl(bool trans_a, bool trans_b, int grad_idx)
   : OpInterface(quote(MatMulGradientOp)),
     _grad_idx(grad_idx), _trans_a(trans_a), _trans_b(trans_b) {}
  
  protected:
   std::vector<NDArrayMeta>
   DoInferMeta(const TensorList& inputs, const InstantiationContext& inst_ctx) const override {
     return {inputs.at(2)->meta()};
   }

   HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes,
                            RuntimeContext& runtime_ctx) const override {
    return {input_shapes.at(2)};
   }

   void DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                       const OpMeta& op_meta,
                       const InstantiationContext& inst_ctx) const override;   

   void DoDeduceHeterProp(const std::vector<int32_t>& inputs_hetero_dim,
                          TensorList& outputs, const OpMeta& op_meta,
                          const InstantiationContext& inst_ctx) const override;  

   void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                  RuntimeContext& runtime_ctx) const override;
  
  public:
   inline bool require_contig_inputs() const override {
     return false;
   }

   bool operator==(const OpInterface& rhs) const override {
     if (OpInterface::operator==(rhs)) {
       const auto& rhs_ = reinterpret_cast<const MatMulGradientOpImpl&>(rhs);
       return trans_a() == rhs_.trans_a() && trans_b() == rhs_.trans_b()
           && grad_idx() == rhs_.grad_idx();
     }
     return false;
   }

   int grad_idx() const {
     return _grad_idx;
   }

   bool trans_a() const {
     return _trans_a;
   }

   bool trans_b() const {
     return _trans_b;
   }

  protected:
   int _grad_idx;
   bool _trans_a;
   bool _trans_b;
};

Tensor MakeMatMulGradientOp(Tensor a, Tensor b, Tensor output, int grad_idx,
                            bool trans_a = false, bool trans_b = false,
                            OpMeta op_meta = OpMeta());

class MatMul4BitOpImpl final : public OpInterface {
 public:
  MatMul4BitOpImpl(bool trans_a, bool trans_b, int blocksize)
  : OpInterface(quote(MatMul4BitOp)), _trans_a(trans_a), _trans_b(trans_b),
                                      _blocksize(blocksize) {}

 protected:
  std::vector<NDArrayMeta>
  DoInferMeta(const TensorList& inputs, const InstantiationContext& inst_ctx) const override {
    const Tensor& a = inputs.at(0);
    const Tensor& b = inputs.at(1);
    const auto dim_a = a->ndim();
    const auto dim_b = b->ndim();
    int64_t m_a = -1;
    int64_t m_b = -1;
    HTShape shape;
    m_a = a->shape(trans_a() ? 0 : 1);
    m_b = b->shape(trans_b() ? 1 : 0);
    HT_ASSERT(m_a == - 1 || m_b == -1 || m_a == m_b)
      << "Failed to construct the \"MatMul4Bit\" op: "
      << "Dimensions must be compatible. "
      << "Got " << m_a << " vs. " << m_b << ". "
      << "Input shape: " << a->shape() << " vs. " << b->shape() << ". ";
    shape = HTShape({a->shape(trans_a() ? 1 : 0), b->shape(trans_b() ? 0 : 1)});
    auto out_meta = a->meta(); 
    out_meta.set_shape(shape);
    return {out_meta};
  }

  void DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                      const OpMeta& op_meta,
                      const InstantiationContext& inst_ctx) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes,
                           RuntimeContext& runtime_ctx) const override {
    const HTShape& a = input_shapes.at(0);
    const HTShape& b = input_shapes.at(1);
    const auto dim_a = a.size();
    const auto dim_b = b.size();
    int64_t m_a = -1;
    int64_t m_b = -1;
    HTShape shape;
    m_a = a[trans_a() ? 0 : 1];
    m_b = b[trans_b() ? 1 : 0];
    HT_ASSERT(m_a == - 1 || m_b == -1 || m_a == m_b)
      << "Failed to construct the \"MatMul4Bit\" op: "
      << "Dimensions must be compatible. "
      << "Got " << m_a << " vs. " << m_b << ". "
      << "Input shape: " << a << " vs. " << b << ". ";
    shape = HTShape({a[trans_a() ? 1 : 0], b[trans_b() ? 0 : 1]});
    return {shape};
  }

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& runtime_ctx) const override {
    NDArray::matmul4bit(inputs.at(0), inputs.at(1), inputs.at(2), inputs.at(3),
                        trans_a(), trans_b(), blocksize(),
                        op->instantiation_ctx().stream_index, outputs.front());
  }

 public:
  inline bool require_contig_inputs() const override {
    return false;
  }

  bool operator==(const OpInterface& rhs) const override {
    if (OpInterface::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const MatMul4BitOpImpl&>(rhs);
      return trans_a() == rhs_.trans_a() 
          && trans_b() == rhs_.trans_b()
          && blocksize() == rhs_.blocksize();
    }
    return false;
  }

  bool trans_a() const {
    return _trans_a;
  }

  bool trans_b() const {
    return _trans_b;
  }

  int blocksize() const {
    return _blocksize;
  }

 protected:
  bool _trans_a;
  bool _trans_b;
  int _blocksize;
};

Tensor MakeMatMul4BitOp(Tensor a, Tensor b, Tensor absmax, Tensor datatype, 
                        bool trans_a = false, bool trans_b = false, 
                        int blocksize = 4096, OpMeta op_meta = OpMeta());

} // namespace graph
} // namespace hetu
