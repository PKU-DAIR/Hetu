#pragma once

#include "hetu/graph/operator.h"
#include "hetu/graph/utils/tensor_utils.h"

namespace hetu {
namespace graph {

class DataH2DOpImpl final : public OpInterface {
 public:
  DataH2DOpImpl(Device device)
  : OpInterface(quote(DataH2D)), _device(std::move(device)) {}

  uint64_t op_indicator() const noexcept override {
    return HOST_TO_DEVICE_OP;
  }

 protected:
  std::vector<NDArrayMeta>
  DoInferMeta(const TensorList& inputs, const InstantiationContext& inst_ctx) const override {
    return {NDArrayMeta().set(inputs.front()->meta()).set_device(device())};
  }

  TensorList DoGradient(Operator& op,
                        const TensorList& grad_outputs) const override;

  bool DoInstantiate(Operator& op, const Device& placement,
                     StreamIndex stream_id) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes,
                           RuntimeContext& runtime_ctx) const override {
    return {input_shapes.front()};
  }

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& runtime_ctx) const override;

 public:
  inline bool require_contig_inputs() const override {
    return false;
  }

  bool operator==(const OpInterface& rhs) const override {
    if (OpInterface::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const DataH2DOpImpl&>(rhs);
      return device() == rhs_.device();
    }
    return false;
  }

  const Device& device() const {
    return _device;
  }

 protected:
  Device _device;
};

class DataD2HOpImpl final : public OpInterface {
 public:
  DataD2HOpImpl(Device device)
  : OpInterface(quote(DataD2H)), _device(std::move(device)) {}

  uint64_t op_indicator() const noexcept override {
    return DEVICE_TO_HOST_OP;
  }

 protected:
  std::vector<NDArrayMeta>
  DoInferMeta(const TensorList& inputs, const InstantiationContext& inst_ctx) const override {
    return {NDArrayMeta().set(inputs.front()->meta()).set_device(device())};
  }

  TensorList DoGradient(Operator& op,
                        const TensorList& grad_outputs) const override;

  bool DoInstantiate(Operator& op, const Device& placement,
                     StreamIndex stream_id) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes,
                           RuntimeContext& runtime_ctx) const override {
    return {input_shapes.front()};
  }

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& runtime_ctx) const override;

 public:
  inline bool require_contig_inputs() const override {
    return false;
  }

  bool operator==(const OpInterface& rhs) const override {
    if (OpInterface::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const DataD2HOpImpl&>(rhs);
      return device() == rhs_.device();
    }
    return false;
  }

  const Device& device() const {
    return _device;
  }

 protected:
  Device _device;
};

class DataTransferOpImpl final : public OpInterface {
 public:
  DataTransferOpImpl(DataType datatype, Device dev = kUndeterminedDevice)
  : OpInterface(quote(DataTransfer)),
  _dtype(datatype),
  _dev(dev) {}

  uint64_t op_indicator() const noexcept {
    return DATA_TRANSFER_OP;
  }

 protected:
  std::vector<NDArrayMeta>
  DoInferMeta(const TensorList& inputs, const InstantiationContext& inst_ctx) const override {
    NDArrayMeta out_meta = inputs.front()->meta();
    out_meta.set_dtype(_dtype);
    if (_dev != kUndeterminedDevice)
      out_meta.set_device(_dev);
    return {out_meta};
  }

  TensorList DoGradient(Operator& op,
                        const TensorList& grad_outputs) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes,
                           RuntimeContext& runtime_ctx) const override {
    return {input_shapes.front()};
  }

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& runtime_ctx) const override;

  NDArrayList DoCompute(Operator& op, const NDArrayList& inputs,
                        RuntimeContext& runtime_ctx) const override;

 public:
  inline bool require_contig_inputs() const override {
    return false;
  }

  bool operator==(const OpInterface& rhs) const override {
    if (OpInterface::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const DataTransferOpImpl&>(rhs);
      return datatype() == rhs_.datatype() &&
             dev() == rhs_.dev();
    }
    return false;
  }

  const DataType& datatype() const {
    return _dtype;
  }

  const Device& dev() const {
    return _dev;
  }

 protected:
  DataType _dtype;
  Device _dev;
};

Tensor MakeDataH2DOp(Device device, Tensor input, OpMeta op_meta = OpMeta());

Tensor MakeDataD2HOp(Device device, Tensor input, OpMeta op_meta = OpMeta());

Tensor MakeDataTransferOp(DataType datatype, Tensor input, Device dev = kUndeterminedDevice, OpMeta op_meta = OpMeta());

} // namespace graph
} // namespace hetu
