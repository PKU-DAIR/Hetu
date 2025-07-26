#pragma once

#include "hetu/graph/common.h"
#include "hetu/graph/executable_graph.h"
#include "hetu/graph/graph.h"
#include <functional>

namespace hetu {
namespace graph {

class ActivationCPUOffload {
 public:
  static const std::vector<std::vector<bool>>& multi_cpu_offload() {
    return _multi_cpu_offload_stack.top();
  }

  static void push_cpu_offload_enabled(const std::vector<std::vector<bool>>& multi_cpu_offload) {
    _multi_cpu_offload_stack.push(multi_cpu_offload);
  }

  static void pop_cpu_offload_enabled() {
    _multi_cpu_offload_stack.pop();
  }

  static void OffloadToCPU(const OpRefList& topo_order);

 protected:

  static bool IsNoOffloadOp(Operator& op);

  static void OffloadTensorToCPU(const OpRefList& topo_order, const Tensor& tensor);

  static std::stack<std::vector<std::vector<bool>>> _multi_cpu_offload_stack;
};

} // namespace graph
} // namespace hetu