#pragma once

#include "hetu/graph/operator.h"
#include "hetu/graph/utils/tensor_utils.h"

namespace hetu {
namespace graph {

using OpDim = std::vector<std::string>;
using OpDimList = std::vector<OpDim>;
using LabelMap = std::unordered_map<std::string, int>;

class EinsumOpImpl;
class EinsumOp;
class EinsumGradientOpImpl;
class EinsumGradientOp;

struct EinsumParameters {
  std::string _msg;

  std::vector<std::string> _input_msgs;

  std::string _output_msg;

  OpDimList input_dims;

  OpDimList output_dims;

  LabelMap num_labels;

  LabelMap output_labels_idx;

  LabelMap undefined_labels;  // only used in gradient

  int output_size;

  int num_output_labels;

  int elli_len;

  std::vector<int> input_elli_len;

  int elli_pos;

  friend bool operator==(const EinsumParameters& l, const EinsumParameters& r);
};

bool operator==(const EinsumParameters& l, const EinsumParameters& r);

class EinsumOpImpl final : public OpInterface {
 public:
  EinsumOpImpl(const EinsumParameters& params)
  : OpInterface(quote(EinsumOp)),
    _params(params) {
  }

  inline std::string fetch_msg() const {
    return _params._msg;
  }

  inline EinsumParameters params() const {
    return _params;
  }

 protected:
  EinsumParameters ParseMsg();

  std::vector<NDArrayMeta>
  DoInferMeta(const TensorList& inputs, const InstantiationContext& inst_ctx) const override {
    EinsumParameters para = params();
    LabelMap label_to_size;
    HTShape output_shape(para.output_size);
    std::vector<int> elli_size(para.elli_len, -1);
    for (size_t i = 0; i < inputs.size(); ++i) {
      int input_idx = 0;
      HTShape perm_shape(para.num_output_labels, 0);
      OpDim input_labels = para.input_dims[i];
      HTShape input_shape = inputs[i]->shape();
      for (const auto& label : input_labels) {
        if (label == "...") {
          if (para.input_elli_len[i] == para.elli_len) {
            for (int k = 0; k < para.elli_len; ++k) {
              if (elli_size[k] == -1) {
                elli_size[k] = input_shape[input_idx + k];
              } else {
                // HT_ASSERT(elli_size[k] == input_shape[input_idx + k]);
              }
            }
          }
          input_idx += para.elli_len;
        } else {
          if (label_to_size.find(label) == label_to_size.end()) {
            label_to_size[label] = input_shape[input_idx];
          } else {
            HT_ASSERT(label_to_size[label] == input_shape[input_idx])
              << label << ":" << label_to_size[label] << ","
              << input_shape[input_idx] << std::endl;
          }
          input_idx += 1;
        }
      }
    }
    if (para.output_dims.empty()) {
      output_shape = {1};
    } else {
      int output_idx = 0;
      for (const auto& label : para.output_dims.at(0)) {
        if (label == "...") {
          for (int k = 0; k < para.elli_len; ++k) {
            output_shape[para.elli_pos + k] = elli_size[k];
          }
          output_idx += para.elli_len;
        } else {
          output_shape[output_idx] = label_to_size[label];
          output_idx += 1;
        }
      }
      if (output_shape.size() == 0) {
        output_shape = {1};
      }
    }
    NDArrayMeta output_meta = NDArrayMeta().set_dtype(inputs[0]->dtype())
                                           .set_shape(output_shape)
                                           .set_device(inputs[0]->device());    
    return {output_meta};
  }

  TensorList DoGradient(Operator& op,
                        const TensorList& grad_outputs) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes,
                           RuntimeContext& runtime_ctx) const override;

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& runtime_ctx) const override;

  EinsumParameters _params;

 public:
  inline bool require_contig_inputs() const override {
    return false;
  }

  bool operator==(const OpInterface& rhs) const override {
    if (OpInterface::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const EinsumOpImpl&>(rhs);
      return (params() == rhs_.params());
    }
    return false;
  }

};

EinsumParameters EinsumParseMsg(const TensorList& inputs,
                                std::string cmd);

Tensor MakeEinsumOp(const std::string& msg, TensorList inputs,
                    OpMeta op_meta = OpMeta());

class EinsumGradientOpImpl final : public OpInterface {

 public:
  EinsumGradientOpImpl(EinsumParameters params)
  : OpInterface(quote(EinsumGradientOp)),
    _params(params) {
  }

  inline std::string fetch_msg() const {
    return _params._msg;
  }

  inline EinsumParameters params() const {
    return _params;
  }

  Tensor pred;

  Tensor pred_in;

 protected:

  std::vector<NDArrayMeta>
  DoInferMeta(const TensorList& inputs, const InstantiationContext& inst_ctx) const override {
    HT_ASSERT_TENSORS_SAME_DTYPE(inputs);
    return {inputs[inputs.size() - 1]->meta()};
  }

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes,
                           RuntimeContext& ctx) const override;

  EinsumParameters _params;

 public:
  inline bool require_contig_inputs() const override {
    return false;
  }

  bool operator==(const OpInterface& rhs) const override {
    if (OpInterface::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const EinsumGradientOpImpl&>(rhs);
      return (params() == rhs_.params());
    }
    return false;
  }

};

EinsumParameters EinsumGradientParseMsg(const TensorList& inputs,
                                        std::string cmd, size_t outdim);

Tensor MakeEinsumGradientOp(const std::string& msg, TensorList inputs,
                            Tensor ori_output, Tensor ori_input,
                            OpMeta op_meta = OpMeta());

} // namespace graph
} // namespace hetu
