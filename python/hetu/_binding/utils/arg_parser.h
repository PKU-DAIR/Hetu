#pragma once

#include <Python.h>
#include "hetu/_binding/utils/pybind_common.h"
#include "hetu/_binding/utils/python_primitives.h"
#include "hetu/_binding/utils/numpy.h"
#include "hetu/_binding/core/dtype.h"
#include "hetu/_binding/core/device.h"
#include "hetu/_binding/core/stream.h"
#include "hetu/_binding/core/ndarray.h"
#include "hetu/_binding/core/symbol.h"
#include "hetu/_binding/graph/operator.h"
#include "hetu/_binding/graph/tensor.h"
#include "hetu/_binding/graph/graph.h"
#include "hetu/_binding/graph/sgdoptimizer.h"
#include "hetu/_binding/graph/adamoptimizer.h"
#include "hetu/_binding/graph/distributed_states.h"
#include "hetu/_binding/graph/init/initializer.h"

namespace hetu {

using namespace hetu::graph;

enum class ArgType : uint8_t {
  /* Python primitives */
  BOOL = 0, 
  INT64, 
  FLOAT64, 
  STRING, 
  DICT,
  
  /* Python sequences of primitives */
  BOOL_LIST,
  BOOL_LIST_LIST,
  INT64_LIST, 
  FLOAT64_LIST, 
  STRING_LIST, 
  
  /* NumPy arrays (NumPy scalars will be treated as primitives) */
  PY_ARRAY, 
  PY_ARRAY_LIST,

  /* Python objects (for new tensors from lists or tuples, 
                     please use sparingly) */
  PY_OBJECT, 
  
  /* Hetu types */
  DATA_TYPE, 
  DEVICE, 
  DEVICE_GROUP, 
  DEVICE_GROUP_LIST, 
  DG_HIERARCHY,
  STREAM, 
  ND_ARRAY, 
  ND_ARRAY_LIST, 
  TENSOR, 
  TENSOR_LIST,
  OPERATOR,
  OPERATOR_LIST,
  FEED_DICT,
  INT_SYMBOL_DICT,
  PARAMETER_DICT,
  STATE_DICT,
  DISTRIBUTED_STATES,
  DISTRIBUTED_STATES_LIST,
  DS_HIERARCHY,
  INT_SYMBOL,
  SYMBOLIC_SHAPE,
  SYMBOLIC_SHAPE_LIST,
  INITIALIZER,
  SGDOPTIMIZER,
  ADAMOPTIMIZER,
};

std::string ArgType2Str(ArgType);
ArgType Str2ArgType(const std::string&);
std::ostream& operator<<(std::ostream&, const ArgType&);

class FnArg;
class FnSignature;
class PyArgParser;
class ParsedPyArgs;

std::ostream& operator<<(std::ostream&, const FnArg&);
std::ostream& operator<<(std::ostream&, const FnSignature&);

class FnArg {
 public:
  FnArg(const std::string& fmt, size_t equal_sign_hint = std::string::npos);

  bool check_arg(PyObject* obj) const;

  inline ArgType type() const { return _arg_type; }
  inline const std::string& name() const { return _arg_name; }
  inline bool optional() const { return _optional; }
  inline bool allow_none() const { return _allow_none; }
  inline const std::string default_repr() const { return _default_repr; }

 private:
  friend class FnSignature;
  friend class ParsedPyArgs;

  ArgType _arg_type;
  std::string _arg_name;
  PyObject* _py_arg_name;
  bool _optional;
  bool _allow_none;
  bool _default_as_none;

  // default values
  std::string _default_repr;
  bool _default_bool;
  int64_t _default_int64;
  double _default_float64;
  std::string _default_string;
  std::vector<int64_t> _default_int64_list;
  std::vector<double> _default_float64_list;
  std::vector<bool> _default_bool_list;
  std::vector<std::vector<bool>> _default_bool_list_list;
};

class FnSignature {
 public:
  FnSignature(const std::string& fmt, size_t index);

  bool parse(PyObject* self, PyObject* args, PyObject* kwargs, 
             std::vector<PyObject*>& parsed, bool is_number_method, 
             std::ostream* const mismatch_reason = nullptr);

  inline const std::string& name() const { return _fn_name; }
  inline size_t index() const { return _index; }
  inline const FnArg& arg(size_t i) const { return _args.at(i); }
  inline size_t num_args() const { return _args.size(); }

 private:
  friend class ParsedPyArgs;
  std::string _fn_name;
  std::string _fn_body;
  size_t _index;
  std::vector<FnArg> _args;
  size_t _max_args;
  size_t _min_args;
};

class ParsedPyArgs {
 public:
  ParsedPyArgs(const FnSignature& signature, std::vector<PyObject*>&& args)
  : _signature(signature), _args(std::move(args)) {}

  inline const FnSignature& signature() const { return _signature; }

  inline const std::string& signature_name() const { 
    return _signature.name();
  }
  
  inline size_t signature_index() const { return _signature.index(); }

  inline const FnArg& signature_arg(size_t i) const {
    return _signature.arg(i);
  }

  inline bool has(size_t i) const { return _args[i] != nullptr; }

  // Explanation of the following getters:
  // - `get_<type>`: 
  //    Get the parsed arg without checking the existence.
  //    This is useful for non-optional arguments.
  // - `get_<type>_or_default`:
  //    Get the parsed arg if exists or return the default value if not.
  //    This is useful for optional arguments with non-"None" default values.
  // - `get_<type>_or_else`:
  //    Get the parsed arg if exists or return the "else" value if not.
  //    This is useful for optional arguments and the function arguments
  //    cannot be determined in advance.
  // - `get_<type>_or_peek`:
  //    Get the parsed arg if exists or peek from context libs. 
  //    If the context libs is empty, then `hetu::nullopt` will be returned.
  //    This is useful for optional arguments with "None" default values 
  //    and the function arguments are provided through context libs, 
  //    such as DataType, Device, and Stream. 
  // - `get_<type>_optional`:
  //    Get the parsed arg if exists or return an optional value. 
  //    It requires the `type` is derived from `hetu::shared_ptr_wrapper` 
  //    so that an empty constructor could work as an optional value. 
  //    This is useful for optional arguments with "None" default values 
  //    and the function arguments cannot be provided through context libs,
  //    such as NDArray and Tensor.
  
  inline bool get_bool(size_t i) const {
    return Bool_FromPyBool(_args[i]);
  }
  
  inline bool get_bool_or_default(size_t i) const {
    return has(i) ? get_bool(i) : signature_arg(i)._default_bool;
  }

  inline bool get_bool_or_else(size_t i, bool default_value) const {
    return has(i) ? get_bool(i) : default_value;
  }

  inline int64_t get_int64(size_t i) const {
    return Int64_FromPyLong(_args[i]);
  }
  
  inline int64_t get_int64_or_default(size_t i) const {
    return has(i) ? get_int64(i) : signature_arg(i)._default_int64;
  }

  inline int64_t get_int64_or_else(size_t i, int64_t default_value) const {
    return has(i) ? get_int64(i) : default_value;
  }

  inline double get_float64(size_t i) const {
    return Float64_FromPyFloat(_args[i]);
  }

  inline double get_float64_or_default(size_t i) const {
    return has(i) ? get_float64(i) : signature_arg(i)._default_float64;
  }

  inline double get_float64_or_else(size_t i, double default_value) const {
    return has(i) ? get_float64(i) : default_value;
  }

  inline std::string get_string(size_t i) const {
    return String_FromPyUnicode(_args[i]);
  }

  inline std::string get_string_or_default(size_t i) const {
    return has(i) ? get_string(i) : signature_arg(i)._default_string;
  }

  inline std::string get_string_or_else(size_t i,
                                        std::string&& default_value) const {
    return has(i) ? get_string(i) : default_value;
  }

  inline std::vector<bool> get_bool_list(size_t i) const {
    return BoolList_FromPyBoolList(_args[i]);
  }

  inline std::vector<bool> get_bool_list_or_default(size_t i) const {
    return has(i) ? get_bool_list(i) : signature_arg(i)._default_bool_list;
  }

  inline std::vector<std::vector<bool>> get_bool_list_list(size_t i) const {
    return BoolListList_FromPyBoolListList(_args[i]);
  }

  inline std::vector<std::vector<bool>> get_bool_list_list_or_default(size_t i) const {
    return has(i) ? get_bool_list_list(i) : signature_arg(i)._default_bool_list_list;
  }

  inline std::vector<int64_t> get_int64_list(size_t i) const {
    return Int64List_FromPyIntList(_args[i]);
  }

  inline std::vector<int64_t> get_int64_list_or_default(size_t i) const {
    return has(i) ? get_int64_list(i) : signature_arg(i)._default_int64_list;
  }

  inline std::vector<double> get_float64_list(size_t i) const {
    return Float64List_FromPyFloatList(_args[i]);
  }

  inline std::vector<double> get_float64_list_or_default(size_t i) const {
    return has(i) ? get_float64_list(i) 
                  : signature_arg(i)._default_float64_list;
  }

  inline std::vector<std::string> get_string_list(size_t i) const {
    return StringList_FromPyStringList(_args[i]);
  }

  inline optional<std::vector<std::string>> 
  get_string_list_optional(size_t i) const {
    return has(i) ? optional<std::vector<std::string>>(get_string_list(i)) 
                  : nullopt;
  }

  inline std::unordered_map<int32_t, int32_t> get_unordered_map(size_t i) const {
    return UnorderedMap_FromPyDict(_args[i]);
  }

  inline PyObject* get_numpy_array(size_t i) const {
    return _args[i];
  }

  inline NDArrayList get_numpy_array_list(size_t i) const {
    return NDArrayListFromNumpyList(_args[i]);
  }

  inline PyObject* get_numpy_array_optional(size_t i) const {
    return has(i) ? get_numpy_array(i) : nullptr;
  }

  inline PyObject* get_py_obj(size_t i) const {
    return _args[i];
  }

  inline PyObject* get_py_obj_optional(size_t i) const {
    return has(i) ? get_py_obj(i) : nullptr;
  }

  inline DataType get_dtype(size_t i) const {
    return DataType_FromPyObject(_args[i]);
  }

  inline DataType get_dtype_or_else(size_t i, DataType default_value) const {
    return has(i) ? get_dtype(i) : default_value;
  }

  inline optional<DataType> get_dtype_or_peek(size_t i) const {
    return has(i) ? optional<DataType>(get_dtype(i)) : get_dtype_ctx().peek();
  }

  inline Device get_device(size_t i) const {
    return Device_FromPyObject(_args[i]);
  }

  inline optional<Device> get_device_or_peek(size_t i) const {
    return has(i) ? optional<Device>(get_device(i)) : get_eager_device_ctx().peek();
  }

  inline DeviceGroup get_device_group(size_t i) const {
    return DeviceGroup_FromPyObject(_args[i]);
  }

  inline DeviceGroupHierarchy get_dg_hierarchy(size_t i) const {
    return DeviceGroupHierarchy_FromPyObject(_args[i]);
  }

  inline optional<DeviceGroupHierarchy> get_dg_hierarchy_or_peek(size_t i) const {
    return has(i) ? optional<DeviceGroupHierarchy>(get_dg_hierarchy(i)) \
                  : get_dg_hierarchy_ctx().peek();
  }

  inline StreamIndex get_stream_index(size_t i) const {
    return static_cast<StreamIndex>(get_int64(i));
  }

  inline optional<StreamIndex> get_stream_index_or_peek(size_t i) const {
    return has(i) ? optional<StreamIndex>(get_stream_index(i))
                  : get_stream_index_ctx().peek();
  }

  inline Stream get_stream(size_t i) const {
    return Stream_FromPyObject(_args[i]);
  }

  inline optional<Stream> get_stream_or_peek(size_t i) const {
    return has(i) ? optional<Stream>(get_stream(i)) : get_stream_ctx().peek();
  }

  inline NDArray get_ndarray(size_t i) const {
    return NDArray_FromPyObject(_args[i]);
  }

  inline NDArray get_ndarray_optional(size_t i) const {
    return has(i) ? get_ndarray(i) : NDArray();
  }

  inline NDArrayList get_ndarray_list(size_t i) const {
    return NDArrayList_FromPyObject(_args[i]);
  }

  inline NDArrayList get_ndarray_list_or_empty(size_t i) const {
    return has(i) ? NDArrayList_FromPyObject(_args[i]) : NDArrayList();
  }

  inline Tensor get_tensor(size_t i) const {
    return Tensor_FromPyObject(_args[i]);
  }

  inline Tensor get_tensor_optional(size_t i) const {
    return has(i) ? get_tensor(i) : Tensor();
  }

  inline TensorList get_tensor_list(size_t i) const {
    return TensorList_FromPyObject(_args[i]);
  }

  inline TensorList get_tensor_list_or_empty(size_t i) const {
    return has(i) ? TensorList_FromPyObject(_args[i]) : TensorList();
  }

  inline Operator get_operator(size_t i) const {
    return Operator_FromPyObject(_args[i]);
  }

  inline Operator get_operator_optional(size_t i) const {
    return has(i) ? get_operator(i) : Operator();
  }

  inline OpList get_operator_list(size_t i) const {
    return OperatorList_FromPyObject(_args[i]);
  }

  inline OpList get_operator_list_or_empty(size_t i) const {
    return has(i) ? OperatorList_FromPyObject(_args[i]) : OpList();
  }

  inline FeedDict get_feed_dict(size_t i) const {
    return FeedDict_FromPyObject(_args[i]);
  }

  inline FeedDict get_feed_dict_or_empty(size_t i) const {
    return has(i) ? get_feed_dict(i) : FeedDict();
  }

  inline IntSymbolDict get_int_symbol_dict(size_t i) const {
    return IntSymbolDict_FromPyObject(_args[i]);
  }

  inline IntSymbolDict get_int_symbol_dict_or_empty(size_t i) const {
    return has(i) ? get_int_symbol_dict(i) : IntSymbolDict();
  }

  inline ParameterDict get_parameter_dict(size_t i) const {
    return ParameterDict_FromPyObject(_args[i]);
  }

  inline ParameterDict get_parameter_dict_or_empty(size_t i) const {
    return has(i) ? get_parameter_dict(i) : ParameterDict();
  }

  inline StateDict get_state_dict(size_t i) const {
    return StateDict_FromPyObject(_args[i]);
  }

  inline StateDict get_state_dict_or_empty(size_t i) const {
    return has(i) ? get_state_dict(i) : StateDict();
  }

  inline SGDOptimizer get_sgdoptimizer(size_t i) const {
    return SGDOptimizer_FromPyObject(_args[i]);
  }  

  inline AdamOptimizer get_adamoptimizer(size_t i) const {
    return AdamOptimizer_FromPyObject(_args[i]);
  }  

  inline DistributedStates get_distributed_states_or_empty(size_t i) {
    return has(i) ? DistributedStates_FromPyObject(_args[i]) : DistributedStates();
  }

  inline DistributedStates get_distributed_states(size_t i) {
    return DistributedStates_FromPyObject(_args[i]);
  }

  inline DistributedStatesList get_distributed_states_list_or_empty(size_t i) {
    return has(i) ? DistributedStatesList_FromPyObject(_args[i]) : DistributedStatesList();
  }

  inline DistributedStatesList get_distributed_states_list(size_t i) {
    return DistributedStatesList_FromPyObject(_args[i]);
  }

  inline DistributedStatesHierarchy get_ds_hierarchy_or_empty(size_t i) {
    return has(i) ? DistributedStatesHierarchy_FromPyObject(_args[i]) : DistributedStatesHierarchy();
  }

  inline DistributedStatesHierarchy get_ds_hierarchy(size_t i) {
    return DistributedStatesHierarchy_FromPyObject(_args[i]);
  }

  inline IntSymbol get_int_symbol_or_empty(size_t i) {
    return has(i) ? IntSymbol_FromPyObject(_args[i]) : IntSymbol();
  }

  inline IntSymbol get_int_symbol(size_t i) {
    return IntSymbol_FromPyObject(_args[i]);
  }

  inline SyShape get_symbolic_shape_or_empty(size_t i) {
    return has(i) ? SyShape_FromPyObject(_args[i]) : SyShape();
  }

  inline SyShape get_symbolic_shape(size_t i) {
    return SyShape_FromPyObject(_args[i]);
  }

  inline SyShapeList get_symbolic_shape_list_or_empty(size_t i) {
    return has(i) ? SyShapeList_FromPyObject(_args[i]) : SyShapeList();
  }

  inline SyShapeList get_symbolic_shape_list(size_t i) {
    return SyShapeList_FromPyObject(_args[i]);
  }

  inline std::shared_ptr<Initializer> get_initializer(size_t i) {
    return Initializer_FromPyObject(_args[i]);
  }
 private:
  const FnSignature& _signature;
  std::vector<PyObject*> _args;
};

class PyArgParser {
 public:
  PyArgParser(const std::vector<std::string>& fmts);
  
  ParsedPyArgs parse(PyObject* self, PyObject* args, PyObject* kwargs, 
                     bool is_number_method);

  inline ParsedPyArgs parse(PyObject* args, PyObject* kwargs) {
    return parse(nullptr, args, kwargs, false);
  }

  const std::vector<FnSignature>& signatures() const { return _signatures; }

  inline size_t num_signatures() const { return _signatures.size(); }

  const std::string& name() const { return _signatures.front().name(); }

 private:
  std::vector<FnSignature> _signatures;
};

#define OP_META_ARGS                                                           \
  "int stream_index=None, "                                                    \
  "List[List[DeviceGroup]] device_group_hierarchy=None, "                                     \
  "List[Tensor] extra_deps=None, "                                             \
  "OpName name=None, "                                                         \
  "bool is_cpu=False"

inline OpMeta parse_op_meta(const ParsedPyArgs& parsed_args, size_t offset) {
  OpMeta ret = CurrentOpMetaCtx();
  if (parsed_args.has(offset + 1))
    ret.set_device_group_hierarchy(parsed_args.get_dg_hierarchy(offset + 1));
  if (parsed_args.has(offset + 2))
    ret.set_extra_deps(parsed_args.get_tensor_list(offset + 2));
  if (parsed_args.has(offset + 3))
    ret.set_name(parsed_args.get_string(offset + 3));
  if (parsed_args.has(offset + 4))
    ret.set_is_cpu(parsed_args.get_bool(offset + 4));
  return ret;
}

} // namespace
