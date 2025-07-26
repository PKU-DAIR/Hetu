#include "hetu/_binding/module.h"
#include "hetu/_binding/constants.h"
#include "hetu/_binding/utils/pybind_common.h"
#include "hetu/_binding/core/device.h"
#include "hetu/_binding/core/dtype.h"
#include "hetu/_binding/core/stream.h"
#include "hetu/_binding/core/ndarray.h"
#include "hetu/_binding/core/symbol.h"
#include "hetu/_binding/graph/operator.h"
#include "hetu/_binding/graph/tensor.h"
#include "hetu/_binding/graph/distributed_states.h"
#include "hetu/_binding/graph/graph.h"
#include "hetu/_binding/graph/autocast.h"
#include "hetu/_binding/graph/recompute.h"
#include "hetu/_binding/graph/cpu_offload.h"
#include "hetu/_binding/graph/gradscaler.h"
#include "hetu/_binding/graph/sgdoptimizer.h"
#include "hetu/_binding/graph/subgraph.h"
#include "hetu/_binding/graph/adamoptimizer.h"
#include "hetu/_binding/graph/runcontext.h"
#include "hetu/_binding/graph/dataloader.h"
#include "hetu/_binding/graph/init/initializer.h"
#include "hetu/_binding/distributed/comm_group.h"
#include "hetu/_binding/graph/profiler.h"

PYBIND11_MODULE(HT_CORE_PY_MODULE, m) {
  hetu::AddPyDeviceTypeToModule(m);
  hetu::AddPyDeviceGroupTypeToModule(m);
  hetu::AddPyDataTypeTypeToModule(m);
  hetu::AddPyStreamTypeToModule(m);
  hetu::AddPyNDArrayTypeToModule(m);
  hetu::AddPyIntSymbolTypeToModule(m);
  hetu::AddPyCommGroupTypeToModule(m);
  hetu::graph::AddPyOperatorTypeToModule(m);
  hetu::graph::AddPyTensorTypeToModule(m);
  hetu::graph::AddPyDistributedStatesTypeToModule(m);
  hetu::graph::AddPyDistributedStatesUnionTypeToModule(m);
  hetu::graph::AddPyGraphTypeToModule(m);
  hetu::graph::AddPyAutoCastTypeToModule(m);
  hetu::graph::AddPyGradScalerTypeToModule(m);
  hetu::graph::AddPySGDOptimizerTypeToModule(m);
  hetu::graph::AddPySubGraphTypeToModule(m);
  hetu::graph::AddPyAdamOptimizerTypeToModule(m);
  hetu::graph::AddPyRunContextTypeToModule(m);
  hetu::graph::AddPyDataloaderTypeToModule(m);
  hetu::graph::AddPyInitializerTypeToModule(m);
  auto internal_sub_module = m.def_submodule("_internal_context");
  hetu::graph::AddOpContextManagingFunctionsToModule(internal_sub_module);
  hetu::graph::AddGraphContextManagingFunctionsToModule(internal_sub_module);
  hetu::graph::AddAutoCastContextManagingFunctionsToModule(internal_sub_module);
  hetu::graph::AddSubGraphContextManagingFunctionsToModule(internal_sub_module);
  hetu::graph::AddRecomputeContextManagingFunctionsToModule(internal_sub_module);
  hetu::graph::AddCPUOffloadContextManagingFunctionsToModule(internal_sub_module);
  hetu::impl::AddPyProfileTypeToModule(m);
  hetu::impl::AddProfileContextManagingFunctionsToModule(internal_sub_module);
}
