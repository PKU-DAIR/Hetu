#include "hetu/_binding/utils/arg_parser.h"
#include "hetu/_binding/utils/except.h"
#include "hetu/_binding/utils/python_primitives.h"
#include "hetu/_binding/utils/numpy.h"
#include "hetu/utils/string_utils.h"

namespace hetu {

std::string ArgType2Str(ArgType type) {
  switch (type) {
    case ArgType::BOOL:
      return "bool";
    case ArgType::INT64:
      return "int";
    case ArgType::FLOAT64:
      return "float";
    case ArgType::STRING:
      return "string";
    case ArgType::BOOL_LIST:
      return "List[bool]";
    case ArgType::BOOL_LIST_LIST:
      return "List[List[bool]]";
    case ArgType::INT64_LIST:
      return "List[int]";
    case ArgType::FLOAT64_LIST:
      return "List[float]";
    case ArgType::STRING_LIST:
      return "List[string]";
    case ArgType::PY_ARRAY:
      return "numpy.array";
    case ArgType::PY_ARRAY_LIST:
      return "List[numpy.array]";
    case ArgType::PY_OBJECT:
      return "py_obj";
    case ArgType::DATA_TYPE:
      return "hetu.dtype";
    case ArgType::DEVICE:
      return "hetu.device";
    case ArgType::DEVICE_GROUP:
      return "hetu.DeviceGroup";
    case ArgType::DEVICE_GROUP_LIST:
      return "List[hetu.DeviceGroup]";
    case ArgType::DG_HIERARCHY:
      return "List[List[hetu.DeviceGroup]]";
    case ArgType::STREAM:
      return "hetu.stream";
    case ArgType::ND_ARRAY:
      return "hetu.NDArray";
    case ArgType::ND_ARRAY_LIST:
      return "List[hetu.NDArray]";
    case ArgType::TENSOR:
      return "hetu.Tensor";
    case ArgType::TENSOR_LIST:
      return "List[hetu.Tensor]";
    case ArgType::OPERATOR:
      return "hetu.Operator";
    case ArgType::OPERATOR_LIST:
      return "List[hetu.Operator]";
    case ArgType::FEED_DICT:
      return "FeedDict";
    case ArgType::INT_SYMBOL_DICT:
      return "IntSymbolDict";
    case ArgType::PARAMETER_DICT:
      return "ParameterDict";
    case ArgType::STATE_DICT:
      return "StateDict";
    case ArgType::SGDOPTIMIZER:
      return "SGDOptimizer";
    case ArgType::ADAMOPTIMIZER:
      return "AdamOptimizer";
    case ArgType::DISTRIBUTED_STATES:
      return "hetu.DistributedStates";
    case ArgType::DISTRIBUTED_STATES_LIST:
      return "List[hetu.DistributedStates]";
    case ArgType::DS_HIERARCHY:
      return "List[hetu.DistributedStatesUnion]";
    case ArgType::INT_SYMBOL:
      return "hetu.IntSymbol";
    case ArgType::SYMBOLIC_SHAPE:
      return "List[hetu.IntSymbol]";
    case ArgType::SYMBOLIC_SHAPE_LIST:
      return "List[List[hetu.IntSymbol]]";
    case ArgType::INITIALIZER:
      return "hetu.Initializer";
    default:
      HT_VALUE_ERROR << "Unknown argument type: " << static_cast<int>(type);
      __builtin_unreachable();
  }
}

ArgType Str2ArgType(const std::string& type) {
  if (type == "bool") 
    return ArgType::BOOL;
  if (type == "int" || type == "int64_t" || type == "int64")
    return ArgType::INT64;
  if (type == "float" || type == "double" || type == "float64")
    return ArgType::FLOAT64;
  if (type == "str" || type == "std::string" || type == "string" ||
      type == "OpName" || type == "TensorName")
    return ArgType::STRING;
  if (type == "List[bool]" || type == "BoolList" ||
      type == "std::vector<bool>" || type == "vector<bool>" || 
      type == "HTKeepDims")
    return ArgType::BOOL_LIST;
  if (type == "List[List[bool]]" || type == "BoolListList" ||
      type == "std::vector<std::vector<bool>>" || type == "vector<vector<bool>>")
    return ArgType::BOOL_LIST_LIST;
  if (type == "std::unordered_map<int,int>")
    return ArgType::DICT;
  if (type == "List[int]" || type == "IntList" ||
      type == "std::vector<int64_t>" || type == "vector<int64_t>" || 
      type == "HTShape" || type == "HTStride" || type == "HTAxes")
    return ArgType::INT64_LIST;
  if (type == "List[float]" || type == "FloatList" || 
      type == "std::vector<double>" || type == "vector<double>")
    return ArgType::FLOAT64_LIST;
  if (type == "List[string]" || type == "List[str]" || type == "StringList" || 
      type == "std::vector<std::string>" || type == "vector<string>")
    return ArgType::STRING_LIST;
  if (type == "numpy.array" || type == "numpy.ndarray" || 
      type == "PyArray" || type == "NumpyArray")
    return ArgType::PY_ARRAY;
  if (type == "List[numpy.array]" || type == "List[numpy.ndarray]" || 
      type == "List[PyArray]" || type == "List[NumpyArray]")
    return ArgType::PY_ARRAY_LIST;
  if (type == "PyObject*" || type == "py_object" || type == "py_obj") 
    return ArgType::PY_OBJECT;
  if (type == "hetu.dtype" || type == "dtype" || type == "DataType")
    return ArgType::DATA_TYPE;
  if (type == "hetu.device" || type == "device" || type == "Device") 
    return ArgType::DEVICE;
  if (type == "hetu.DeviceGroup" || type == "DeviceGroup")
    return ArgType::DEVICE_GROUP;
  if (type == "List[hetu.DeviceGroup]" || type == "List[DeviceGroup]" ||
      type == "hetu.DeviceGroupList" || type == "DeviceGroupList")
    return ArgType::DEVICE_GROUP_LIST;
  if (type == "List[List[hetu.DeviceGroup]]" || type == "List[List[DeviceGroup]]" || 
      type == "DeviceGroupHierarchy")
    return ArgType::DG_HIERARCHY;
  if (type == "hetu.stream" || type == "stream" || type == "Stream") 
    return ArgType::STREAM;
  if (type == "hetu.NDArray" || type == "NDArray" || 
      type == "hetu.ndarray" || type == "ndarray") 
    return ArgType::ND_ARRAY;
  if (type == "List[hetu.NDArray]" || type == "List[NDArray]" || 
      type == "List[hetu.ndarray]" || type == "List[ndarray]" || 
      type == "NDArrayList") 
    return ArgType::ND_ARRAY_LIST;
  if (type == "hetu.Tensor" || type == "Tensor" || 
      type == "hetu.tensor" || type == "tensor") 
    return ArgType::TENSOR;
  if (type == "List[hetu.Tensor]" || type == "List[Tensor]" ||
      type == "List[hetu.tensor]" || type == "List[tensor]" || 
      type == "TensorList") 
    return ArgType::TENSOR_LIST;
  if (type == "hetu.Operator" || type == "Operator" || 
      type == "hetu.operator" || type == "operator")
      return ArgType::OPERATOR;
  if (type == "List[hetu.Operator]" || type == "List[Operator]" ||
      type == "List[hetu.operator]" || type == "List[operator]" || 
      type == "OperatorList" || type == "OpList") 
    return ArgType::OPERATOR_LIST;
  if (type == "FeedDict" || type == "feed_dict")
    return ArgType::FEED_DICT;
  if (type == "IntSymbolDict" || type == "int_symbol_dict")
    return ArgType::INT_SYMBOL_DICT;
  if (type == "ParameterDict" || type == "parameter_dict")
    return ArgType::PARAMETER_DICT;
  if (type == "StateDict" || type == "state_dict")
    return ArgType::STATE_DICT;
  if (type == "Optimizer" || type == "SGDOptimizer")
    return ArgType::SGDOPTIMIZER;
  if (type == "AdamOptimizer")
    return ArgType::ADAMOPTIMIZER;
  if (type == "hetu.DistributedStates" || type == "DistributedStates")
    return ArgType::DISTRIBUTED_STATES;
  if (type == "List[hetu.DistributedStates]" || type == "DistributedStatesList" ||
      type == "List[DistributedStates]")
    return ArgType::DISTRIBUTED_STATES_LIST; 
  if (type == "List[hetu.DistributedStatesUnion]" || type == "List[DistributedStatesUnion]" ||
      type == "DistributedStatesHierarchy")
    return ArgType::DS_HIERARCHY;
  if (type == "hetu.IntSymbol" || type == "IntSymbol")
    return ArgType::INT_SYMBOL;
  if (type == "List[hetu.IntSymbol]" || type == "List[IntSymbol]" || type == "SyShape")
    return ArgType::SYMBOLIC_SHAPE;
  if (type == "List[List[hetu.IntSymbol]]" || type == "List[List[IntSymbol]]" || type == "List[SyShape]")
    return ArgType::SYMBOLIC_SHAPE_LIST;
  if (type == "hetu.Initializer" || type == "Initializer")
    return ArgType::INITIALIZER;
  HT_VALUE_ERROR << "Unknown argument type: " << type;
  __builtin_unreachable();
}

std::ostream& operator<<(std::ostream& os, const ArgType& type) {
  os << ArgType2Str(type);
  return os;
}
// TODO: FnArg & Check Dict for unordered_map
FnArg::FnArg(const std::string& fmt, size_t equal_sign_hint) {
  auto space = fmt.find(' ');
  HT_ASSERT(space != std::string::npos)
    << "Invalid argument \'" << fmt << "\'";
  auto arg_type_str = fmt.substr(0, space);
  auto arg_name_str = fmt.substr(space + 1);

  _arg_type = Str2ArgType(arg_type_str);

  auto equal_sign = equal_sign_hint;
  if (equal_sign == std::string::npos) {
    equal_sign = arg_name_str.find('=');
  } else {
    HT_ASSERT(equal_sign < fmt.length() && fmt.at(equal_sign) == '=')
      << "Invalid hint of equality sign " << equal_sign 
      << " for argument \'" << fmt << "\'";
    equal_sign -= space + 1;
  }
  if (equal_sign == std::string::npos) {
    _arg_name = arg_name_str;
    _optional = false;
    _default_as_none = false;
  } else {
    _arg_name = arg_name_str.substr(0, equal_sign);
    _optional = true;
    auto default_str = arg_name_str.substr(equal_sign + 1);
    HT_ASSERT(!default_str.empty()) << "Invalid argument \'" << fmt << "\': " 
      << "default value not provided.";
    // parse default value
    if (default_str == "None") {
      _allow_none = true;
      _default_as_none = true;
      _default_repr = default_str;
    } else {
      _default_as_none = false;
    }
    
    switch (_arg_type) {
      case ArgType::BOOL:
        if (!_default_as_none) {
          if (!parse_bool_slow_but_safe(default_str, _default_bool))
            HT_VALUE_ERROR << "Cannot parsed default value to bool: "
                           << default_str;
          _default_repr = _default_bool ? "True" : "False";
        }
        break;
      case ArgType::INT64:
        if (!_default_as_none) {
          if (!parse_int64_slow_but_safe(default_str, _default_int64))
            HT_VALUE_ERROR << "Cannot parsed default value to int: "
                           << default_str;
          _default_repr = std::to_string(_default_int64);
        }
        break;
      case ArgType::FLOAT64:
        if (!_default_as_none) {
          if (!parse_float64_slow_but_safe(default_str, _default_float64))
            HT_VALUE_ERROR << "Cannot parsed default value to float: "
                           << default_str;
          _default_repr = std::to_string(_default_float64);
        }
        break;
      case ArgType::STRING:
        if (!_default_as_none) {
          auto err_msg = parse_string_literal(default_str, _default_string);
          if (!err_msg.empty())
            HT_VALUE_ERROR << "Cannot parse default string: " << err_msg;
          _default_repr = '\'' + _default_string + '\'';
        }
        break;
      case ArgType::BOOL_LIST:
        if (!_default_as_none) {
          auto err_msg = parse_bool_list_slow_but_safe(
            default_str, _default_bool_list);
          if (!err_msg.empty())
            HT_VALUE_ERROR << "Cannot parse default List[bool]: " << err_msg;
          _default_repr = default_str;
        }
        break;
      case ArgType::BOOL_LIST_LIST:
        if (!_default_as_none) {
          auto err_msg = parse_bool_list_list_slow_but_safe(
            default_str, _default_bool_list_list);
          if (!err_msg.empty())
            HT_VALUE_ERROR << "Cannot parse default List[bool]: " << err_msg;
          _default_repr = default_str;
        }
        break;
      case ArgType::INT64_LIST:
        if (!_default_as_none) {
          auto err_msg = parse_int64_list_slow_but_safe(
            default_str, _default_int64_list);
          if (!err_msg.empty())
            HT_VALUE_ERROR << "Cannot parse default List[int]: " << err_msg;
          _default_repr = default_str;
        }
        break;
      case ArgType::FLOAT64_LIST:
        if (!_default_as_none) {
          auto err_msg = parse_float64_list_slow_but_safe(
            default_str, _default_float64_list);
          if (!err_msg.empty())
            HT_VALUE_ERROR << "Cannot parse default List[float]: " << err_msg;
          _default_repr = default_str;
        }
        break;
      case ArgType::STRING_LIST:
      case ArgType::PY_ARRAY:
      case ArgType::PY_ARRAY_LIST:
      case ArgType::PY_OBJECT:
      case ArgType::DATA_TYPE:
      case ArgType::DEVICE:
      case ArgType::DEVICE_GROUP:
      case ArgType::DEVICE_GROUP_LIST:
      case ArgType::DG_HIERARCHY:
      case ArgType::STREAM:
      case ArgType::ND_ARRAY:
      case ArgType::ND_ARRAY_LIST:
      case ArgType::TENSOR:
      case ArgType::TENSOR_LIST:
      case ArgType::OPERATOR:
      case ArgType::OPERATOR_LIST:
      case ArgType::FEED_DICT:
      case ArgType::INT_SYMBOL_DICT:
      case ArgType::PARAMETER_DICT:
      case ArgType::DISTRIBUTED_STATES:
      case ArgType::DISTRIBUTED_STATES_LIST:
      case ArgType::DS_HIERARCHY:
      case ArgType::INT_SYMBOL:
      case ArgType::SYMBOLIC_SHAPE:
      case ArgType::SYMBOLIC_SHAPE_LIST:
      case ArgType::INITIALIZER:
        if (!_default_as_none) {
          HT_VALUE_ERROR << "Default " << _arg_type << " can only be None";
        }
        break;
      default:
        HT_VALUE_ERROR << "Unknown argument type: " << arg_type_str;
    }
  } 
  _py_arg_name = PyUnicode_FromString(_arg_name);
}

bool FnArg::check_arg(PyObject* obj) const {
  switch (_arg_type) {
    case ArgType::BOOL:
      return CheckPyBool(obj);
    case ArgType::INT64:
      return CheckPyLong(obj);
    case ArgType::FLOAT64:
      return CheckPyFloat(obj);
    case ArgType::STRING:
      return CheckPyString(obj);
    case ArgType::DICT:
      return true;
      // TODO:
      // return CheckPyDict(obj);
    case ArgType::BOOL_LIST:
      return CheckPyBoolList(obj);
    case ArgType::BOOL_LIST_LIST:
      return CheckPyBoolListList(obj);
    case ArgType::INT64_LIST:
      return CheckPyIntList(obj);
    case ArgType::FLOAT64_LIST:
      return CheckPyFloatList(obj);
    case ArgType::STRING_LIST:
      return CheckPyStringList(obj);
    case ArgType::DATA_TYPE:
      return CheckPyDataType(obj);
    case ArgType::PY_ARRAY:
      return CheckNumpyArray(obj);
    case ArgType::PY_ARRAY_LIST:
      return CheckNumpyArrayList(obj);
    case ArgType::PY_OBJECT:
      return true;
    case ArgType::DEVICE:
      return CheckPyDevice(obj);
    case ArgType::DEVICE_GROUP:
      return CheckPyDeviceGroup(obj);
    case ArgType::DEVICE_GROUP_LIST:
      return CheckPyDeviceGroupList(obj);
    case ArgType::DG_HIERARCHY:
      return CheckPyDeviceGroupHierarchy(obj);
    case ArgType::STREAM:
      return CheckPyStream(obj);
    case ArgType::ND_ARRAY:
      return CheckPyNDArray(obj);
    case ArgType::ND_ARRAY_LIST:
      return CheckPyNDArrayList(obj);
    case ArgType::TENSOR:
      return CheckPyTensor(obj);
    case ArgType::TENSOR_LIST:
      return CheckPyTensorList(obj);
    case ArgType::OPERATOR:
      return CheckPyOperator(obj);
    case ArgType::OPERATOR_LIST:
      return CheckPyOperatorList(obj);
    case ArgType::FEED_DICT:
      return CheckPyFeedDict(obj);
    case ArgType::INT_SYMBOL_DICT:
      return CheckPyIntSymbolDict(obj);
    case ArgType::PARAMETER_DICT:
      return CheckPyParameterDict(obj);
    case ArgType::SGDOPTIMIZER:
      return CheckPySGDOptimizer(obj);
    case ArgType::ADAMOPTIMIZER:
      return CheckPyAdamOptimizer(obj);
    case ArgType::DISTRIBUTED_STATES:
      return CheckPyDistributedStates(obj);
    case ArgType::DISTRIBUTED_STATES_LIST:
      return CheckPyDistributedStatesList(obj);
    case ArgType::DS_HIERARCHY:
      return CheckPyDistributedStatesHierarchy(obj);
    case ArgType::INT_SYMBOL:
      return CheckPyIntSymbol(obj);
    case ArgType::SYMBOLIC_SHAPE:
      return CheckPySyShape(obj);
    case ArgType::SYMBOLIC_SHAPE_LIST:
      return CheckPySyShapeList(obj);
    case ArgType::INITIALIZER:
      return CheckPyInitializer(obj);
    default:
      HT_VALUE_ERROR << "Unknown argument type: " 
        << static_cast<int>(_arg_type);
      __builtin_unreachable();
  }
}

std::ostream& operator<<(std::ostream& os, const FnArg& fn_args) {
  os << fn_args.type() << ' ' << fn_args.name();
  if (fn_args.optional()) {
    os << '=' << fn_args.default_repr();
  }
  return os;
}

FnSignature::FnSignature(const std::string& fmt, size_t index): _index(index) {
  auto left_parentheses = fmt.find('(');
  auto right_parentheses = fmt.rfind(')');
  HT_ASSERT(left_parentheses != std::string::npos && 
            right_parentheses != std::string::npos)
    << "Invalid function argument: " << fmt;
  _fn_name = fmt.substr(0, left_parentheses);
  _fn_body = fmt.substr(left_parentheses + 1, 
                        right_parentheses - left_parentheses - 1);
  trim(_fn_body);

  std::unordered_set<std::string> arg_names;
  if (!_fn_body.empty()) {
    decltype(left_parentheses) ll = 0;
    bool prev_has_default = false;
    do {
      // Note: If a default string must contain commas (e.g., IntList), 
      // please make sure the commas are not followed by spaces.
      // For instance, [1,2,3] is valid but [1, 2, 3] is not.
      auto rr = _fn_body.find(", ", ll);
      if (rr == std::string::npos) 
        rr = _fn_body.length() + 1;
      auto arg_str = _fn_body.substr(ll, rr - ll);
      trim(arg_str);
      auto equal_sign = arg_str.find('=');
      if (equal_sign != std::string::npos) {
        prev_has_default = true;
      } else {
        HT_ASSERT(!prev_has_default) << "Invalid format \'" << fmt << "\': " 
          << "Non-optional arguments cannot follow optional arguments.";
      }
      // TODO: allow number as tensor?
      _args.emplace_back(arg_str, equal_sign);
      HT_ASSERT(arg_names.find(_args.back().name()) == arg_names.end())
        << "Duplicated name for arguments: " << fmt;
      arg_names.insert(_args.back().name());
      ll = rr + 2;
    } while (ll < _fn_body.length());
  }

  _max_args = _args.size();
  _min_args = 0;
  for (auto& arg : _args) 
    if (!arg._optional) 
      _min_args++;
}

bool FnSignature::parse(PyObject* self, PyObject* args, PyObject* kwargs, 
                        std::vector<PyObject*>& parsed, bool is_number_method, 
                        std::ostream* const mismatch_reason) {
  size_t num_args = 0, num_kw_args = 0;
  if (is_number_method) {
    if (self) num_args++;
    if (args) num_args++;
    if (kwargs) num_args++;
  } else {
    num_args = args ? PyTuple_GET_SIZE(args) : 0;
    num_kw_args = kwargs ? PyDict_Size(kwargs) : 0;
  }
  size_t num_provided = num_args + num_kw_args;
  if (num_provided < _min_args) {
    if (mismatch_reason) {
      (*mismatch_reason) << "Too few arguments (at least " << _min_args
        << " arguments should be provided, got " << num_provided << ")";
    }
    return false;
  }
  if (num_provided > _max_args) {
    if (mismatch_reason) {
      (*mismatch_reason) << "Too many arguments (at most " << _max_args
        << " arguments should be provided, got " << num_provided << ")";
    }
    return false;
  }

  parsed.resize(_max_args, nullptr);
  size_t used_kwargs = 0;
  for (size_t i = 0; i < _max_args; i++) {
    PyObject* obj = nullptr;
    auto& fn_arg = _args[i];
    if (is_number_method) {
      obj = i == 0 ? self : (i == 1 ? args : kwargs);
    } else {
      if (i < num_args) {
        // positional arguments
        obj = PyTuple_GET_ITEM(args, i);
      } else if (i < _min_args && num_kw_args > 0) {
        // positional arguments provided in kwargs
        // TODO: support alias
        obj = PyDict_GetItem(kwargs, fn_arg._py_arg_name);
        if (obj)
          used_kwargs++;
      } else if (num_kw_args > 0) {
        // keyword arguments
        // TODO: support alias
        obj = PyDict_GetItem(kwargs, fn_arg._py_arg_name);
        if (obj)
          used_kwargs++;
      }
    }

    if (!obj) {
      if (!fn_arg.optional()) {
        if (mismatch_reason) {
          (*mismatch_reason) << "Argument \'" << fn_arg.name() << "\' "
            << "is missing";
        }
        return false;
      }
    } else if (obj == Py_None) {
      if (!fn_arg.allow_none()) {
        if (mismatch_reason) {
          (*mismatch_reason) << "Argument \'" << fn_arg.name() << "\' "
            << "cannot be none";
        }
        return false;
      }
    } else {
      if (!fn_arg.check_arg(obj)) {
        if (mismatch_reason) {
          (*mismatch_reason) << "Type \'" << Py_TYPE(obj)->tp_name << "\' "
            << "cannot be cast to type \'" << fn_arg.type() << "\' "
            << "for argument \'" << fn_arg.name() << "\'";
        }
        return false;
      }
      parsed[i] = obj;
    }
  }

  if (used_kwargs != num_kw_args) {
    if (mismatch_reason) {
      (*mismatch_reason) << "Provided " << num_kw_args << " keyword arguments "
        << "but only used " << used_kwargs << " of them";
    }
    return false;
  } else {
    return true;
  }
}

std::ostream& operator<<(std::ostream& os, const FnSignature& fn_signature) {
  os << fn_signature.name() << '(';
  if (fn_signature.num_args() > 0) {
    os << fn_signature.arg(0);
    for (size_t i = 1; i < fn_signature.num_args(); i++)
      os << ", " << fn_signature.arg(i);
  }
  os << ')';
  return os;
}

PyArgParser::PyArgParser(const std::vector<std::string>& fmts) {
  HT_ASSERT(!fmts.empty()) << "No signatures are provided.";
  for (size_t i = 0; i < fmts.size(); i++) {
    _signatures.emplace_back(fmts[i], i);
    HT_ASSERT(_signatures.back().name() == _signatures.front().name())
      << "Names of signatures are inconsistent: " << fmts;
  }
}

ParsedPyArgs PyArgParser::parse(PyObject* self, PyObject* args, 
                                PyObject* kwargs, bool is_number_method) {
  std::vector<PyObject*> parsed;
  for (auto& signature : _signatures) {
    bool ok = signature.parse(self, args, kwargs, parsed, is_number_method);
    if (ok)
      return ParsedPyArgs(signature, std::move(parsed));
  }
  
  // run the parse fns again to print why they failed
  std::ostringstream os;
  os << "Invalid Arguments for function \'" << name() << "\':";
  for (auto& signature : _signatures) {
    os << "\n    " << signature << " ==> ";
    signature.parse(self, args, kwargs, parsed, is_number_method, &os);
  }
  HT_TYPE_ERROR << os.str();
  __builtin_unreachable();
}

} // namespace
