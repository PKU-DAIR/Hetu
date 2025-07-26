#include "hetu/graph/ops/scalars_like.h"
#include "hetu/graph/headers.h"

namespace hetu {
namespace graph {

Tensor MakeScalarsLikeOp(Tensor input, double scalar_value, OpMeta op_meta) {
  return Graph::MakeOp(std::make_shared<ScalarsLikeOpImpl>(scalar_value),
                       {std::move(input)}, std::move(op_meta))
    ->output(0);
}

void ScalarsLikeOpImpl::DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                                       const OpMeta& op_meta,
                                       const InstantiationContext& inst_ctx) const {
  const DistributedStates& ds_input = inputs.at(0)->get_distributed_states();
  HT_ASSERT(ds_input.is_valid()) 
    << "ScalarLikeOpDef: distributed states for input must be valid!";  
  int32_t device_num = ds_input.get_device_num();
  std::pair<std::vector<int32_t>, int32_t> src2dst({{-2}, -1}); // partial -> duplicate
  std::unordered_map<int32_t, int32_t> res_states = ds_input.combine_states(src2dst);
  std::vector<int32_t> res_order = ds_input.combine_order(src2dst);
  Tensor& output = outputs.at(0);
  output->set_distributed_states({device_num, res_states, res_order});
}

} // namespace graph
} // namespace hetu
