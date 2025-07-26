#pragma once

#include "hetu/graph/operator.h"
#include "hetu/graph/utils/tensor_utils.h"
#include "hetu/core/symbol.h"
#include "hetu/graph/ops/Views.h"

namespace hetu {
namespace graph {

class SplitOpImpl;
class SplitOp;
class SplitGradientOpImpl;
class SplitGradientOp;

class SplitOpImpl final : public ViewsOpImpl {
 public:
  SplitOpImpl(const SyShapeList& begin_pos_list, const SyShapeList& output_shape_list, int64_t dim = 0)
  : ViewsOpImpl(quote(SplitOp)),
    _begin_pos_list(begin_pos_list),
    _output_shape_list(output_shape_list),
    _dim(dim),
    _symbolic(true) {}

  SplitOpImpl(const HTShapeList& begin_pos_list, const HTShapeList& output_shape_list, int64_t dim = 0)
  : ViewsOpImpl(quote(SplitOp)),
    _dim(dim),
    _symbolic(false) {
      auto split_num = begin_pos_list.size();
      for (int i = 0; i < split_num; i++) {
        set_HTShape_to_SyShape(begin_pos_list[i], _begin_pos_list[i]);
        set_HTShape_to_SyShape(output_shape_list[i], _output_shape_list[i]);
      }
    }
  
  inline uint64_t op_indicator() const noexcept override {
    return SLICE_OP;
  }

  inline int get_split_num() const {
    return _begin_pos_list.size();
  }

  const SyShapeList& get_symbolic_begin_pos_list() const {
    return _begin_pos_list;
  }

  const SyShapeList& get_symbolic_output_shape_list() const {
    return _output_shape_list;
  }

  HTShapeList get_begin_pos_list() const {
    HTShapeList ret;
    for (auto& sy_shape : _begin_pos_list) {
      ret.push_back(get_HTShape_from_SyShape(sy_shape));
    }
    return std::move(ret);
  }

  HTShapeList get_output_shape_list() const {
    HTShapeList ret;
    for (auto& sy_shape : _output_shape_list) {
      ret.push_back(get_HTShape_from_SyShape(sy_shape));
    }
    return std::move(ret);
  }

  HTShapeList get_multi_task_begin_pos(const HTShapeList& input_shapes) const {
    HTShapeList begin_pos_list = {};
    HT_ASSERT(input_shapes.size() > 1)
      << "please provide task_batch_idxs for multi-task split";
    int task_num = input_shapes.size() - 1;
    for (int i = 0; i < task_num; i++) {
      HTShape begin_pos(input_shapes[0].size(), 0);
      begin_pos[_dim] = input_shapes[0][_dim] / input_shapes[i + 1][2] * input_shapes[i + 1][0];
      begin_pos_list.push_back(begin_pos);
    }
    return begin_pos_list;
  }

  HTShapeList get_multi_task_output_shape(const HTShapeList& input_shapes) const {
    HTShapeList output_shape_list = {};
    HT_ASSERT(input_shapes.size() > 1)
      << "please provide task_batch_idxs for multi-task split";
    int task_num = input_shapes.size() - 1;
    for (int i = 0; i < task_num; i++) {
      HTShape output_shape(input_shapes[0]);
      output_shape[_dim] = input_shapes[0][_dim] / input_shapes[i + 1][2] * input_shapes[i + 1][1];
      output_shape_list.push_back(output_shape);
    }
    return output_shape_list;
  }
  
  HTShape get_begin_pos(int64_t idx) const {
    return get_HTShape_from_SyShape(_begin_pos_list[idx]);
  }

  HTShape get_output_shape(int64_t idx) const {
    return get_HTShape_from_SyShape(_output_shape_list[idx]);
  }

  const SyShape& get_symbolic_begin_pos(int64_t idx) const {
    return _begin_pos_list[idx];
  }

  const SyShape& get_symbolic_output_shape(int64_t idx) const {
    return _output_shape_list[idx];
  }

  bool symbolic() const {
    return _symbolic;
  }
  
  int64_t dim() const {
    return _dim;
  }

 protected:
  std::vector<NDArrayMeta>
  DoInferMeta(const TensorList& inputs, const InstantiationContext& inst_ctx) const override {
    std::vector<NDArrayMeta> meta_list;
    auto split_num = get_split_num();
    if (split_num == 0) {
      split_num = inputs.size() - 1;
      HTShapeList input_shapes;
      for (auto& input : inputs) {
        input_shapes.emplace_back(input->shape());
      }
      auto multi_task_begin_pos = get_multi_task_begin_pos(input_shapes);
      auto multi_task_output_shape = get_multi_task_output_shape(input_shapes);
      for (int i = 0; i < split_num; i++) {
        HT_ASSERT(inputs[0]->ndim() == multi_task_begin_pos[i].size() &&
                  inputs[0]->ndim() == multi_task_output_shape[i].size());
        int len = multi_task_begin_pos[i].size();
        for (int j = 0; j < len; ++j) {
          HT_ASSERT(multi_task_begin_pos[i][j] + multi_task_output_shape[i][j] <= inputs[0]->shape(j))
            << "dim " << j << " begin pos is " << multi_task_begin_pos[i][j] << " and output len is " << multi_task_output_shape[i][j]
            << ", but input len is " << inputs[0]->shape(j);
        }
        meta_list.push_back(NDArrayMeta().set_dtype(inputs[0]->dtype())
                                        .set_shape(multi_task_output_shape[i])
                                        .set_stride(inputs[0]->stride())
                                        .set_device(inputs[0]->device()));
      }
    } else {
      for (int i = 0; i < split_num; i++) {
        HT_ASSERT(inputs[0]->ndim() == _begin_pos_list[i].size() &&
                  inputs[0]->ndim() == _output_shape_list[i].size());
        int len = _begin_pos_list[i].size();
        for (int j = 0; j < len; ++j) {
          HT_ASSERT(_begin_pos_list[i][j]->get_val() + _output_shape_list[i][j]->get_val() <= inputs[0]->shape(j))
            << "dim " << j << " begin pos is " << _begin_pos_list[i][j]->get_val() << " and output len is " << _output_shape_list[i][j]->get_val()
            << ", but input len is " << inputs[0]->shape(j);
        }
        meta_list.push_back(NDArrayMeta().set_dtype(inputs[0]->dtype())
                                        .set_shape(get_output_shape(i))
                                        .set_stride(inputs[0]->stride())
                                        .set_device(inputs[0]->device()));
      }
    }
    return std::move(meta_list);
  }

  void DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                      const OpMeta& op_meta,
                      const InstantiationContext& inst_ctx) const override;

  void DoDeduceHeterProp(const std::vector<int32_t>& inputs_hetero_dim,
                         TensorList& outputs, const OpMeta& op_meta,
                         const InstantiationContext& inst_ctx) const override;  

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) const {};
  
  NDArrayList DoCompute(Operator& op, const NDArrayList& inputs,
                        RuntimeContext& ctx) const override;

  TensorList DoGradient(Operator& op, const TensorList& grad_outputs) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes, RuntimeContext& ctx) const override;

  void DoSaveCtxForBackward(const TensorList& inputs, ContextStore& dst_ctx) const override;

  SyShapeList _begin_pos_list;
  SyShapeList _output_shape_list;
  bool _symbolic;
  int64_t _dim;

 public:
  bool operator==(const OpInterface& rhs) const override {
    if (ViewsOpImpl::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const SplitOpImpl&>(rhs);
      auto split_num = _begin_pos_list.size();
      for (int i = 0; i < split_num; i++) {
        if (get_begin_pos(i) != rhs_.get_begin_pos(i) || get_output_shape(i) != rhs_.get_output_shape(i) || dim() != rhs_.dim())
          return false;
      }
      return true;
    }
    return false;
  }
};

class SplitGradientOpImpl : public ViewsOpImpl {
 public:
  SplitGradientOpImpl(SyShapeList begin_pos_list,
                      SyShapeList output_shape_list,
                      int64_t dim = 0)
  : ViewsOpImpl(quote(SplitGradientOp)),
    _begin_pos_list(std::move(begin_pos_list)),
    _output_shape_list(std::move(output_shape_list)),
    _dim(dim),
    _symbolic(true) {}
  SplitGradientOpImpl(const HTShapeList& begin_pos_list,
                      const HTShapeList& output_shape_list,
                      int64_t dim = 0)
  : ViewsOpImpl(quote(SplitGradientOp)),
    _dim(dim),
    _symbolic(false) {
      auto split_num = begin_pos_list.size();
      for (int i = 0; i < split_num; i++) {
        set_HTShape_to_SyShape(begin_pos_list[i], _begin_pos_list[i]);
        set_HTShape_to_SyShape(output_shape_list[i], _output_shape_list[i]);
      }
    }

  inline int get_split_num() const {
    return _begin_pos_list.size();
  }
  
  HTShape get_begin_pos(int64_t idx) const {
    return get_HTShape_from_SyShape(_begin_pos_list[idx]);
  }

  HTShape get_output_shape(int64_t idx) const {
    return get_HTShape_from_SyShape(_output_shape_list[idx]);
  }

  HTShapeList get_multi_task_begin_pos(const HTShapeList& input_shapes) const {
    HTShapeList begin_pos_list = {};
    HT_ASSERT(input_shapes.size() > 1)
      << "please provide task_batch_idxs for multi-task split";
    int task_num = input_shapes.size() - 1;
    for (int i = 0; i < task_num; i++) {
      HTShape begin_pos(input_shapes[0].size(), 0);
      begin_pos[_dim] = input_shapes[0][_dim] / input_shapes[i + 1][2] * input_shapes[i + 1][0];
      begin_pos_list.push_back(begin_pos);
    }
    return begin_pos_list;
  }

  HTShapeList get_multi_task_output_shape(const HTShapeList& input_shapes) const {
    HTShapeList output_shape_list = {};
    HT_ASSERT(input_shapes.size() > 1)
      << "please provide task_batch_idxs for multi-task split";
    int task_num = input_shapes.size() - 1;
    for (int i = 0; i < task_num; i++) {
      HTShape output_shape(input_shapes[0]);
      output_shape[_dim] = input_shapes[0][_dim] / input_shapes[i + 1][2] * input_shapes[i + 1][1];
      output_shape_list.push_back(output_shape);
    }
    return output_shape_list;
  }

  const SyShape& get_symbolic_begin_pos(int64_t idx) const {
    return _begin_pos_list[idx];
  }

  const SyShape& get_symbolic_output_shape(int64_t idx) const {
    return _output_shape_list[idx];
  }

  const SyShapeList& get_symbolic_begin_pos_list() const {
    return _begin_pos_list;
  }

  const SyShapeList& get_symbolic_output_shape_list() const {
    return _output_shape_list;
  }

  HTShapeList get_begin_pos_list() const {
    HTShapeList ret;
    for (auto& sy_shape : _begin_pos_list) {
      ret.push_back(get_HTShape_from_SyShape(sy_shape));
    }
    return std::move(ret);
  }

  HTShapeList get_output_shape_list() const {
    HTShapeList ret;
    for (auto& sy_shape : _output_shape_list) {
      ret.push_back(get_HTShape_from_SyShape(sy_shape));
    }
    return std::move(ret);
  }

  bool symbolic() const {
    return _symbolic;
  }

  int64_t dim() const {
    return _dim;
  }

 protected:
  std::vector<NDArrayMeta>
  DoInferMeta(const TensorList& inputs, const InstantiationContext& inst_ctx) const override {
    return {inst_ctx.get<NDArrayMeta>("in_meta")};
  };

  void DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                      const OpMeta& op_meta,
                      const InstantiationContext& inst_ctx) const override;

  void DoDeduceHeterProp(const std::vector<int32_t>& inputs_hetero_dim,
                         TensorList& outputs, const OpMeta& op_meta,
                         const InstantiationContext& inst_ctx) const override; 

  NDArrayList DoCompute(Operator& op, const NDArrayList& inputs,
                        RuntimeContext& ctx) const override;
  
  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                  RuntimeContext& ctx) const override;
  
  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes, RuntimeContext& ctx) const override;

  void DoLoadCtxForBackward(ContextStore& src_ctx, ContextStore& dst_ctx) const override;

  SyShapeList _begin_pos_list;
  SyShapeList _output_shape_list;
  bool _symbolic;
  int64_t _dim;

 public:
  bool operator==(const OpInterface& rhs) const override {
    if (ViewsOpImpl::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const SplitGradientOpImpl&>(rhs);
      auto split_num = _begin_pos_list.size();
      for (int i = 0; i < split_num; i++) {
        if (get_begin_pos(i) != rhs_.get_begin_pos(i) || get_output_shape(i) != rhs_.get_output_shape(i) || dim() != rhs_.dim())
          return false;
      }
      return true;
    }
    return false;
  }
};

// seems deprecated
TensorList MakeSplitOp(Tensor input, int64_t num_chunks, int64_t dim,
                       bool remain = false, OpMeta op_meta = OpMeta());

// deprecated: only used in gpt inference, before symbolic shape is realized
TensorList MakeSplitOp(Tensor input, int64_t num_chunks, int64_t dim,
                       int64_t padding_axis, bool remain = false, OpMeta op_meta = OpMeta());

// 这里只能做到在单一的dim上的切分
// 主要用于qkv.split(3)
TensorList MakeSplitOp(Tensor input, const HTShape& chunks, int64_t dim,
                       bool remain = false, OpMeta op_meta = OpMeta());

TensorList MakeSplitOp(Tensor input, TensorList task_batch_idxs, int64_t dim,
                       OpMeta op_meta = OpMeta());

Tensor MakeSplitGradientOp(TensorList grad_outputs,
                           SyShapeList begin_pos_list,
                           SyShapeList output_shape_list,
                           OpMeta op_meta = OpMeta());

Tensor MakeSplitGradientOp(TensorList grad_outputs,
                           const HTShapeList& begin_pos_list,
                           const HTShapeList& output_shape_list,
                           OpMeta op_meta = OpMeta());

Tensor MakeSplitGradientOp(TensorList grad_outputs,
                           TensorList task_batch_idxs,
                           SyShapeList begin_pos_list,
                           SyShapeList output_shape_list,
                           int64_t dim,
                           OpMeta op_meta = OpMeta());

Tensor MakeSplitGradientOp(TensorList grad_outputs,
                           TensorList task_batch_idxs,
                           const HTShapeList& begin_pos_list,
                           const HTShapeList& output_shape_list,
                           int64_t dim,
                           OpMeta op_meta = OpMeta());

// 可以缺省
// 只在axes部分维度上切分
// 主要用于替换exec graph中的通信算子
Tensor MakeSplitOp(Tensor input, const HTAxes& axes, const HTShape& indices,
                   const HTShape& splits, bool remain = false, OpMeta op_meta = OpMeta());

// 不可缺省   
// 主要用于exec graph witch时通信图的建立          
Tensor MakeSplitOp(Tensor input, const HTShape& indices, const HTShape& splits, 
                   bool remain = false, OpMeta op_meta = OpMeta());

} // namespace graph
} // namespace hetu
