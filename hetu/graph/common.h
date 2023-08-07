#pragma once

#include "hetu/common/macros.h"
#include "hetu/core/ndarray.h"
#include <vector>
// #include <queue>
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
using OpRefList = std::vector<OpRef>;
using OpCRefList = std::vector<OpCRef>;
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
using Tensor2IntMap = std::unordered_map<TensorId, int>;

using GradAndVar = std::pair<Tensor, Tensor>;
using GradAndVarList = std::vector<GradAndVar>;

template <typename T>
struct is_tensor_list : std::false_type {};
template <>
struct is_tensor_list<TensorList> : std::true_type {};

using GraphId = uint64_t;
using AutoCastId = uint64_t;
using GraphName = std::string;
using FeedDict = Tensor2NDArrayMap;
class Graph;
class EagerGraph;
class DefineByRunGraph;
class DefineAndRunGraph;
class ExecutableGraph;

#define HT_MAX_NUM_MICRO_BATCHES (128)
} // namespace graph
} // namespace hetu
