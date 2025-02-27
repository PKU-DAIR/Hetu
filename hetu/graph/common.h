#pragma once

#include "hetu/common/macros.h"
#include "hetu/core/ndarray.h"
#include "hetu/core/symbol.h"
#include <vector>
#include <queue>
#include <deque>
#include <type_traits>
#include <functional>

namespace hetu {
namespace graph {

// Somethong went wrong if we remove this line...
using hetu::operator<<;

class OpInterface;
class OpDef;
class Operator;
using OpRef = std::reference_wrapper<Operator>;
using OpCRef = std::reference_wrapper<const Operator>;
using OpId = uint64_t;
using OpType = std::string;
using OpName = std::string;
using OpList = std::vector<Operator>;
using OpDeque = std::deque<Operator>;
using OpRefList = std::vector<OpRef>;
using OpCRefList = std::vector<OpCRef>;
using OpRefQueue = std::queue<OpRef>;
using OpRefDeque = std::deque<OpRef>;
using OpCRefDeque = std::deque<OpCRef>;
using OpIdList = std::vector<OpId>;
using OpIdSet = std::unordered_set<OpId>;
using Op2OpMap = std::unordered_map<OpId, Operator>;
using Op2OpRefMap = std::unordered_map<OpId, OpRef>;
using Op2OpCRefMap = std::unordered_map<OpId, OpCRef>;


template <typename T>
struct is_op_list : std::false_type {};
template <>
struct is_op_list<OpList> : std::true_type {};

class TensorDef;
class Tensor;
using TensorRef = std::reference_wrapper<Tensor>;
using TensorCRef = std::reference_wrapper<const Tensor>;
using TensorId = uint64_t;
using TensorName = std::string;
using TensorList = std::vector<Tensor>;
using TensorRefList = std::vector<TensorRef>;
using TensorCRefList = std::vector<TensorCRef>;
using TensorIdList = std::vector<TensorId>;
using TensorIdSet = std::unordered_set<TensorId>;
using Tensor2TensorMap = std::unordered_map<TensorId, Tensor>;
using Tensor2TensorListMap = std::unordered_map<TensorId, TensorList>;
using Tensor2NDArrayMap = std::unordered_map<TensorId, NDArray>;
using Tensor2NDArrayListMap = std::unordered_map<TensorId, NDArrayList>;
using Tensor2IntMap = std::unordered_map<TensorId, int>;
using Tensor2StringMap = std::unordered_map<TensorId, std::string>;
using Tensor2ShapeMap = std::unordered_map<TensorId, HTShape>;
using StateDict = std::unordered_map<OpName, Tensor>;
using Tensor2ShapeMapList = std::vector<Tensor2ShapeMap>;

using GradAndVar = std::pair<Tensor, Tensor>;
using GradAndVarList = std::vector<GradAndVar>;

template <typename T>
struct is_tensor_list : std::false_type {};
template <>
struct is_tensor_list<TensorList> : std::true_type {};

using GraphId = uint64_t;
using GraphName = std::string;
using FeedDict = Tensor2NDArrayListMap;
using IntSymbolDict = std::unordered_map<IntSymbol, std::vector<int64_t>>;
using ParameterDict = std::unordered_map<std::string, int64_t>;
using MemoryBlock = std::pair<size_t, size_t>;
using MemoryBlockList = std::vector<MemoryBlock>;
using MicroBatchTensorId = std::pair<size_t, TensorId>;
using MemoryPlan = std::map<MicroBatchTensorId, MemoryBlock>;
using Device2PipelineMap = std::unordered_map<Device, DeviceGroupList>;
class Graph;
class EagerGraph;
class DefineByRunGraph;
class DefineAndRunGraph;
class ExecutableGraph;
class SwitchExecGraph;

class ParamBuffer;
class ParamBuckets;

// TODO: larger nmb will slow down the whole system 
#define HT_MAX_NUM_MICRO_BATCHES (64)
} // namespace graph
} // namespace hetu
