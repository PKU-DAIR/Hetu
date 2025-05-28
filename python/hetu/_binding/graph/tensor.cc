#include "hetu/_binding/graph/tensor.h"
#include "hetu/_binding/graph/tensor_ctor.h"
#include "hetu/_binding/graph/graph.h"
#include "hetu/_binding/core/ndarray.h"
#include "hetu/_binding/core/symbol.h"
#include "hetu/_binding/constants.h"
#include "hetu/_binding/utils/function_registry.h"
#include "hetu/_binding/utils/pybind_common.h"
#include "hetu/_binding/utils/python_primitives.h"
#include "hetu/_binding/utils/except.h"
#include "hetu/_binding/utils/decl_utils.h"
#include "hetu/_binding/utils/arg_parser.h"
#include "hetu/graph/ops/variable.h"
#include "hetu/graph/ops/data_transfer.h"

namespace hetu {
namespace graph {

PyObject* PyTensor_New(Tensor&& tensor, bool return_none_if_undefined) {
  HT_PY_FUNC_BEGIN
  if (return_none_if_undefined && !tensor.is_defined()) {
    Py_RETURN_NONE;
  } else {
    auto* unsafe_self = PyTensor_Type->tp_alloc(PyTensor_Type, 0);
    HT_RUNTIME_ERROR_IF(!unsafe_self) << "Failed to alloc PyTensor";
    auto* self = reinterpret_cast<PyTensor*>(unsafe_self);
    new(&self->tensor) Tensor();
    self->tensor = std::move(tensor);
    return reinterpret_cast<PyObject*>(self);
  }
  HT_PY_FUNC_END
}

PyObject* PyTensorList_New(TensorList&& tensors,
                           bool return_none_if_undefined) {
  HT_PY_FUNC_BEGIN
  PyObject* ret = PyList_New(tensors.size());
  HT_RUNTIME_ERROR_IF(!ret) << "Failed to alloc list";
  for (size_t i = 0; i < tensors.size(); i++) {
    auto* tensor_obj = PyTensor_New(std::move(tensors[i]), return_none_if_undefined);
    PyList_SET_ITEM(ret, i, tensor_obj);
  }
  return ret;
  HT_PY_FUNC_END
}

PyObject* PyDict_FromStateDict(StateDict map) {
  PyObject* dict_obj = PyDict_New();
  for (auto& item : map) {
    PyObject* key = PyUnicode_FromString(item.first);
    Tensor val = item.second;
    PyObject* value = PyTensor_New(std::move(val));
    PyDict_SetItem(dict_obj, key, value);
  }
  return dict_obj;
}

inline PyObject* PyTensor_pynew(PyTypeObject* type, PyObject* args, 
                                PyObject* kwargs) {
  return TensorCopyCtor(type, args, kwargs);
}

PyObject* PyTensor_make_subclass(PyObject*, PyObject* args, PyObject* kwargs) {
  HT_PY_FUNC_BEGIN
  static PyArgParser parser({
    "_make_subclass(PyObject* cls, Tensor data, bool requires_grad=false)"
  });
  auto parsed_args = parser.parse(args, kwargs);
  if (parsed_args.signature_index() == 0) {
    PyObject* cls = parsed_args.get_py_obj(0);
    HT_TYPE_ERROR_IF(!PyType_Check(cls))
      << "Expected argument \"cls\" to be a type (got " 
      << Py_TYPE(cls)->tp_name << ")";
    PyTypeObject* cls_type = reinterpret_cast<PyTypeObject*>(cls);
    HT_TYPE_ERROR_IF(!PyType_IsSubtype(cls_type, PyTensor_Type))
      << "Type " << cls_type->tp_name << " is not derived from hetu.Tensor";

    auto* unsafe_self = cls_type->tp_alloc(cls_type, 0);
    HT_RUNTIME_ERROR_IF(!unsafe_self) << "Failed to alloc PyTensor";
    // Question: Is the casting safe?
    auto* self = reinterpret_cast<PyTensor*>(unsafe_self);
    new(&self->tensor) Tensor();

    self->tensor = parsed_args.get_tensor(1);
    HT_TYPE_ERROR_IF(!self->tensor->is_variable())
      << "Subclass of hetu.Tensor must be created from a variable. "
      << "Please detach the tensor first.";
    HT_NOT_IMPLEMENTED;
    // self->tensor->set_trainable(parsed_args.get_bool_or_default(2));
    
    return reinterpret_cast<PyObject*>(self);
  } else {
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();
  }
  HT_PY_FUNC_END
}

PyObject* PyTensor_make_recompute(PyTensor* self, PyObject* args, PyObject* kwargs) {
  HT_PY_FUNC_BEGIN
  static PyArgParser parser({
    "_make_recompute(List[bool] is_recompute=[True])",
    "_make_recompute(List[List[bool]] is_recompute=[[True]])"
  });
  auto parsed_args = parser.parse(args, kwargs);
  if (parsed_args.signature_index() == 0) {
    std::vector<bool> is_recompute = parsed_args.get_bool_list_or_default(0);
    self->tensor->producer()->op_meta().set_multi_recompute({is_recompute});
  } else if (parsed_args.signature_index() == 1) {
    std::vector<std::vector<bool>> multi_is_recompute = parsed_args.get_bool_list_list_or_default(0);
    self->tensor->producer()->op_meta().set_multi_recompute(multi_is_recompute);
  } else {
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();
  }
  Py_RETURN_NONE;
  HT_PY_FUNC_END
}

void PyTensor_dealloc(PyTensor* self) {
  (&self->tensor)->~Tensor();
  Py_TYPE(self)->tp_free(self);
}

PyObject* PyTensor_str(PyTensor* self) {
  HT_PY_FUNC_BEGIN
  // Compute?
  return PyUnicode_FromString(self->tensor->name());
  HT_PY_FUNC_END
}

PyObject* PyTensor_repr(PyTensor* self) {
  return PyTensor_str(self);
}

Py_hash_t PyTensor_hash(PyTensor* self) {
  return self->tensor->id();
}

PyObject* PyTensor_richcompare(PyObject* a, PyObject* b, int op) {
  if (!PyObject_TypeCheck(a, PyTensor_Type) || !PyObject_TypeCheck(b, PyTensor_Type)) {
    Py_RETURN_NOTIMPLEMENTED;
  }
  PyTensor* ta = (PyTensor*)a;
  PyTensor* tb = (PyTensor*)b;
  switch (op) {
    case Py_EQ:
      if (ta->tensor->id() == tb->tensor->id()) {
        Py_RETURN_TRUE;
      } else {
        Py_RETURN_FALSE;
      }
    case Py_NE:
      if (ta->tensor->id() != tb->tensor->id()) {
        Py_RETURN_TRUE;
      } else {
        Py_RETURN_FALSE;
      }
    default:
      Py_RETURN_NOTIMPLEMENTED;
  }
}

PyObject* PyTensor_id(PyTensor* self) {
  HT_PY_FUNC_BEGIN
  return PyLong_FromInteger(self->tensor->id());
  HT_PY_FUNC_END
}

PyObject* PyTensor_name(PyTensor* self) {
  HT_PY_FUNC_BEGIN
  return PyUnicode_FromString(self->tensor->name());
  HT_PY_FUNC_END
}

PyObject* PyTensor_ndim(PyTensor* self) {
  HT_PY_FUNC_BEGIN
  return PyLong_FromInteger(self->tensor->ndim());
  HT_PY_FUNC_END
}

PyObject* PyTensor_dim(PyTensor* self, PyObject* args, PyObject* kwargs) {
  HT_PY_FUNC_BEGIN
  static PyArgParser parser({
    "dim()"
  });
  auto parsed_args = parser.parse(args, kwargs);
  if (parsed_args.signature_index() == 0) {
    return PyLong_FromInteger(self->tensor->ndim());
  } else {
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();
  }
  HT_PY_FUNC_END
}

PyObject* PyTensor_shape(PyTensor* self) {
  HT_PY_FUNC_BEGIN
  return PyLongList_FromIntegerList(self->tensor->shape());
  HT_PY_FUNC_END
}

PyObject* PyTensor_numel(PyTensor* self) {
  HT_PY_FUNC_BEGIN
  return PyLong_FromInteger(self->tensor->numel());
  HT_PY_FUNC_END
}

PyObject* PyTensor_global_numel(PyTensor* self) {
  HT_PY_FUNC_BEGIN
  return PyLong_FromInteger(NumEl(self->tensor->global_shape()));
  HT_PY_FUNC_END
}

PyObject* PyTensor_timecost(PyTensor* self) {
  HT_PY_FUNC_BEGIN
  return PyLong_FromInteger(self->tensor->producer()->TimeCost());
  HT_PY_FUNC_END
}

PyObject* PyTensor_global_shape(PyTensor* self) {
  HT_PY_FUNC_BEGIN
  return PyLongList_FromIntegerList(self->tensor->global_shape());
  HT_PY_FUNC_END
}

PyObject* PyTensor_symbolic_shape(PyTensor* self) {
  HT_PY_FUNC_BEGIN
  if (!self->tensor->symbolic()) {
    HT_LOG_DEBUG << "You are using symbolic_shape attribute of " << self->tensor
      << ", so we turn this tensor to a symbolic tensor on the fly!";
    self->tensor->init_symbolic_shape(); 
  }
  return PySyShape_New(self->tensor->symbolic_shape());
  HT_PY_FUNC_END
}

PyObject* PyTensor_is_contiguous(PyTensor* self) {
  HT_PY_FUNC_BEGIN
  Py_RETURN_BOOLEAN_COND(self->tensor->is_contiguous());
  HT_PY_FUNC_END
}

PyObject* PyTensor_size(PyTensor* self, PyObject* args, PyObject* kwargs) {
  HT_PY_FUNC_BEGIN
  static PyArgParser parser({
    "size(int dim=None)"
  });
  auto parsed_args = parser.parse(args, kwargs);
  if (parsed_args.signature_index() == 0) {
    if (parsed_args.has(0))
      return PyLong_FromInteger(self->tensor->shape(parsed_args.get_int64(0)));
    else
      return PyLongList_FromIntegerList(self->tensor->shape());
  } else {
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();
  }
  HT_PY_FUNC_END
}

PyObject* PyTensor_stride(PyTensor* self, PyObject* args, PyObject* kwargs) {
  HT_PY_FUNC_BEGIN
  static PyArgParser parser({
    "stride(int dim=None)"
  });
  auto parsed_args = parser.parse(args, kwargs);
  if (parsed_args.signature_index() == 0) {
    if (parsed_args.has(0))
      return PyLong_FromInteger(self->tensor->stride(parsed_args.get_int64(0)));
    else
      return PyLongList_FromIntegerList(self->tensor->stride());
  } else {
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();
  }
  HT_PY_FUNC_END
}

PyObject* PyTensor_device(PyTensor* self) {
  HT_PY_FUNC_BEGIN
  return PyDevice_New(self->tensor->device());
  HT_PY_FUNC_END
}

PyObject* PyTensor_dtype(PyTensor* self) {
  HT_PY_FUNC_BEGIN
  return PyDataType_New(self->tensor->dtype());
  HT_PY_FUNC_END
}

PyObject* PyTensor_is_variable(PyTensor* self) {
  HT_PY_FUNC_BEGIN
  Py_RETURN_BOOLEAN_COND(self->tensor->is_variable());
  HT_PY_FUNC_END
}

PyObject* PyTensor_is_parameter(PyTensor* self) {
  HT_PY_FUNC_BEGIN
  Py_RETURN_BOOLEAN_COND(self->tensor->is_parameter());
  HT_PY_FUNC_END
}

PyObject* PyTensor_requires_grad(PyTensor* self) {
  HT_PY_FUNC_BEGIN
  // TODO: rename ``requires_grad'' to ``requires_grad''
  Py_RETURN_BOOLEAN_COND(self->tensor->requires_grad());
  HT_PY_FUNC_END
}

PyObject* PyTensor_distributed_states(PyTensor* self) {
  HT_PY_FUNC_BEGIN
  return PyDistributedStates_New(self->tensor->get_distributed_states());  
  HT_PY_FUNC_END
}

PyObject* PyTensor_ds_hierarchy(PyTensor* self) {
  HT_PY_FUNC_BEGIN
  return PyDistributedStatesHierarchy_New(self->tensor->ds_hierarchy());   
  HT_PY_FUNC_END
}

PyObject* PyTensor_dg_hierarchy(PyTensor* self) {
  HT_PY_FUNC_BEGIN
  return PyDeviceGroupHierarchy_New(self->tensor->producer()->device_group_hierarchy());   
  HT_PY_FUNC_END
}

PyObject* PyTensor_has_placement_group(PyTensor* self) {
  HT_PY_FUNC_BEGIN
  Py_RETURN_BOOLEAN_COND(self->tensor->has_placement_group());   
  HT_PY_FUNC_END
}

PyObject* PyTensor_placement_group_union(PyTensor* self) {
  HT_PY_FUNC_BEGIN
  if (self->tensor->has_placement_group())
    return PyDeviceGroupList_New(self->tensor->placement_group_union().raw_data());
  else
    return PyDeviceGroupList_New({}); 
  HT_PY_FUNC_END
}

PyObject* PyTensor_device_group(PyTensor* self) {
  HT_PY_FUNC_BEGIN
  return PyDeviceGroup_New(self->tensor->producer()->device_group());  
  HT_PY_FUNC_END
}

PyObject* PyTensor_data(PyTensor* self) {
  HT_PY_FUNC_BEGIN
  return PyNDArray_New(GetDetachedVariableData(self->tensor), true);
  HT_PY_FUNC_END
}

PyObject* PyTensor_gpu_data(PyTensor* self) {
  HT_PY_FUNC_BEGIN
  return PyNDArray_New(GetDetachedVariableData(self->tensor, true), true);
  HT_PY_FUNC_END
}

PyObject* PyTensor_raw_data_ptr(PyTensor* self) {
  HT_PY_FUNC_BEGIN
  return PyLong_FromInteger(int64_t(self->tensor->get_raw_data_ptr()));
  HT_PY_FUNC_END
}

PyObject* PyTensor_is_floating_point(PyTensor* self) {
  HT_PY_FUNC_BEGIN
  Py_RETURN_BOOLEAN_COND(IsFloatingPoint(self->tensor->dtype()));
  HT_PY_FUNC_END
}

PyObject* PyTensor_graph(PyTensor* self) {
  HT_PY_FUNC_BEGIN
  return PyGraph_New(self->tensor->graph_id());
  HT_PY_FUNC_END
}

PyObject* PyTensor_get_or_compute(PyTensor* self) {
  HT_PY_FUNC_BEGIN
  return PyNDArray_New(self->tensor->get_or_compute());
  HT_PY_FUNC_END
}

PyObject* PyTensor_symbolic(PyTensor* self) {
  HT_PY_FUNC_BEGIN
  self->tensor->init_symbolic_shape();
  Py_RETURN_NONE;
  HT_PY_FUNC_END
}

PyObject* PyTensor_check_ds_hierarchy_equal(PyTensor* self, PyObject* args, PyObject* kwargs) {
  HT_PY_FUNC_BEGIN
  static PyArgParser parser({
    "check_ds_hierarchy_equal(List[DistributedStatesUnion] ds_hierarchy)"
  });
  auto parsed_args = parser.parse(args, kwargs);
  if (parsed_args.signature_index() == 0) {
    Py_RETURN_BOOLEAN_COND(self->tensor->check_ds_hierarchy_equal(parsed_args.get_ds_hierarchy(0)));
  } else {
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();
  }  
  HT_PY_FUNC_END
}

PyObject* PyTensor_check_ds_hierarchy_equal_except_split(PyTensor* self, PyObject* args, PyObject* kwargs) {
  HT_PY_FUNC_BEGIN
  static PyArgParser parser({
    "check_ds_hierarchy_equal_except_split(List[DistributedStatesUnion] ds_hierarchy)"
  });
  auto parsed_args = parser.parse(args, kwargs);
  if (parsed_args.signature_index() == 0) {
    Py_RETURN_BOOLEAN_COND(self->tensor->check_ds_hierarchy_equal_except_split(parsed_args.get_ds_hierarchy(0)));
  } else {
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();
  }  
  HT_PY_FUNC_END
}

PyObject* PyTensor_from_numpy(PyObject*, PyObject* args, PyObject* kwargs) {
  HT_PY_FUNC_BEGIN
  auto* unsafe_self = PyTensor_Type->tp_alloc(PyTensor_Type, 0);
  HT_RUNTIME_ERROR_IF(!unsafe_self) << "Failed to alloc PyTensor";
  auto* self = reinterpret_cast<PyTensor*>(unsafe_self);
  
  static PyArgParser parser({
    "from_numpy(numpy.array data, bool requires_grad=false, DistributedStatesHierarchy ds_hierarchy=None, " OP_META_ARGS ")", 
  });
  auto parsed_args = parser.parse(args, kwargs);

  if (parsed_args.signature_index() == 0) {
    auto* array_obj = parsed_args.get_numpy_array(0);
    bool requires_grad = parsed_args.get_bool_or_default(1);
    DistributedStatesHierarchy ds_hierarchy = parsed_args.get_ds_hierarchy_or_empty(2);
    new(&self->tensor) Tensor();
    self->tensor = MakeParameterOp(NDArrayFromNumpy(array_obj), false, kUndeterminedDataType, requires_grad, ds_hierarchy, parse_op_meta(parsed_args, 3));
  } else {
    Py_TYPE(self)->tp_free(self);
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();
  }

  return reinterpret_cast<PyObject*>(self);
  HT_PY_FUNC_END
}

PyObject* PyTensor_from_buffer(PyObject*, PyObject* args, PyObject* kwargs) {
  HT_PY_FUNC_BEGIN
  auto* unsafe_self = PyTensor_Type->tp_alloc(PyTensor_Type, 0);
  HT_RUNTIME_ERROR_IF(!unsafe_self) << "Failed to alloc PyTensor";
  auto* self = reinterpret_cast<PyTensor*>(unsafe_self);
  
  static PyArgParser parser({
    "from_buffer(bytes data, DataType dtype, List[int] shape, bool requires_grad=false, \
     DistributedStatesHierarchy ds_hierarchy=None, " OP_META_ARGS ")",
    "from_buffer(bytearray data, DataType dtype, List[int] shape, bool requires_grad=false, \
     DistributedStatesHierarchy ds_hierarchy=None, " OP_META_ARGS ")",
  });
  auto parsed_args = parser.parse(args, kwargs);

  if (parsed_args.signature_index() == 0) {
    // Get buffer interface from Python object
    Py_buffer view;
    PyObject* buffer_obj = parsed_args.get_bytes(0);
    if (PyObject_GetBuffer(buffer_obj, &view, PyBUF_SIMPLE) < 0) {
      PyErr_SetString(PyExc_TypeError, "Unable to get buffer from object");
      Py_TYPE(self)->tp_free(self);
      return nullptr;
    }

    DataType dtype = parsed_args.get_dtype(1);
    std::vector<int64_t> shape = parsed_args.get_int64_list(2);
    bool requires_grad = parsed_args.get_bool_or_default(3);
    DistributedStatesHierarchy ds_hierarchy = parsed_args.get_ds_hierarchy_or_empty(4);

    // Create NDArray from buffer
    NDArray array = NDArrayFromBuffer(view.buf, dtype, shape);
    PyBuffer_Release(&view);

    new(&self->tensor) Tensor();
    self->tensor = MakeParameterOp(array, false, dtype, requires_grad, ds_hierarchy, parse_op_meta(parsed_args, 5));
    
    return reinterpret_cast<PyObject*>(self);
  } else if (parsed_args.signature_index() == 1) {
    // Get buffer interface from Python object
    Py_buffer view;
    PyObject* buffer_obj = parsed_args.get_bytearray(0);
    if (PyObject_GetBuffer(buffer_obj, &view, PyBUF_SIMPLE) < 0) {
      PyErr_SetString(PyExc_TypeError, "Unable to get buffer from object");
      Py_TYPE(self)->tp_free(self);
      return nullptr;
    }

    DataType dtype = parsed_args.get_dtype(1);
    std::vector<int64_t> shape = parsed_args.get_int64_list(2);
    bool requires_grad = parsed_args.get_bool_or_default(3);
    DistributedStatesHierarchy ds_hierarchy = parsed_args.get_ds_hierarchy_or_empty(4);

    // Create NDArray from buffer
    NDArray array = NDArrayFromBuffer(view.buf, dtype, shape);
    PyBuffer_Release(&view);

    new(&self->tensor) Tensor();
    self->tensor = MakeParameterOp(array, false, dtype, requires_grad, ds_hierarchy, parse_op_meta(parsed_args, 5));
    
    return reinterpret_cast<PyObject*>(self);
  } else {
    Py_TYPE(self)->tp_free(self);
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();
  }
  HT_PY_FUNC_END
}

PyObject* PyTensor_data_transfer(PyTensor* self, PyObject* args, PyObject* kwargs) {
  HT_PY_FUNC_BEGIN
  auto* unsafe_self = PyTensor_Type->tp_alloc(PyTensor_Type, 0);
  HT_RUNTIME_ERROR_IF(!unsafe_self) << "Failed to alloc PyTensor";
  auto* new_self = reinterpret_cast<PyTensor*>(unsafe_self);
  
  static PyArgParser parser({
    "to(DataType dtype, device dev=None, " OP_META_ARGS ")"
  });
  auto parsed_args = parser.parse(args, kwargs);

  if (parsed_args.signature_index() == 0) {
    new(&new_self->tensor) Tensor();
    auto dev = parsed_args.get_device_or_peek(1);
    new_self->tensor = MakeDataTransferOp(parsed_args.get_dtype(0), 
                                          self->tensor, 
                                          dev.value_or(kUndeterminedDevice),
                                          parse_op_meta(parsed_args, 2));
    return reinterpret_cast<PyObject*>(new_self);
  } else {
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();
  }
  HT_PY_FUNC_END
}

PyObject* PyTensor_reset_data(PyTensor* self, PyObject* args, PyObject* kwargs) {
  HT_PY_FUNC_BEGIN
  static PyArgParser parser({
    "reset_data(numpy.array provided_data)",
    "reset_data(NDArray provided_data)",
  });
  auto parsed_args = parser.parse(args, kwargs);
  if (parsed_args.signature_index() == 0) {
    auto* array_obj = parsed_args.get_numpy_array(0);
    ResetVariableData(self->tensor, NDArrayFromNumpy(array_obj));
    // HT_LOG_TRACE << "ResetVariableData successfully.";
  }
  else if (parsed_args.signature_index() == 1) {
    auto array_obj = parsed_args.get_ndarray(0);
    // HT_ASSERT(self->tensor->shape() == array_obj->shape())
    //   << "The shape of provided data does not match the shape of the tensor: "
    //   << self->tensor->shape() << " vs " << array_obj->shape();
    ResetVariableData(self->tensor, array_obj);
    // HT_LOG_TRACE << "ResetVariableData successfully.";
  } else {
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();
  }
  Py_RETURN_NONE;
  HT_PY_FUNC_END
}

PyObject* PyTensor_reset_data_from_splits(PyTensor* self, PyObject* args, PyObject* kwargs) {
  HT_PY_FUNC_BEGIN
  static PyArgParser parser({
    "reset_data_from_splits(List[numpy.array] provided_datas)",
  });
  auto parsed_args = parser.parse(args, kwargs);
  if (parsed_args.signature_index() == 0) {
    ResetVariableDataFromSplits(self->tensor, parsed_args.get_numpy_array_list(0));
    // HT_LOG_TRACE << "ResetVariableData successfully.";
  }
  else {
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();
  }
  Py_RETURN_NONE;
  HT_PY_FUNC_END
}

PyObject* PyTensor_get_data(PyTensor* self, PyObject* args, PyObject* kwargs) {
  HT_PY_FUNC_BEGIN
  static PyArgParser parser({
    "get_data()"
  });
  auto parsed_args = parser.parse(args, kwargs);
  if (parsed_args.signature_index() == 0) {
    // HT_LOG_INFO << self->tensor << " " << Graph::GetGraph(self->tensor).type();
    auto ret = GetDetachedVariableData(self->tensor);
    // HT_LOG_TRACE << "GetDetachedVariableData successfully.";
    return NDArrayToNumpy(ret, false, true);
  } else {
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();
  }
  HT_PY_FUNC_END
}

PyObject* PyTensor_get_device_group_union(PyTensor* self, PyObject* args, PyObject* kwargs) {
  HT_PY_FUNC_BEGIN
  static PyArgParser parser({
    "get_device_group_union()"
  });
  auto parsed_args = parser.parse(args, kwargs);
  if (parsed_args.signature_index() == 0) {
    auto ret = GetVariableDeviceGroupUnion(self->tensor);
    // HT_LOG_TRACE << "GetVariableDeviceGroupUnion successfully.";
    return PyDeviceGroupList_New(ret.raw_data());
  } else {
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();
  }
  HT_PY_FUNC_END
}

PyObject* PyTensor_get_ds_union(PyTensor* self, PyObject* args, PyObject* kwargs) {
  HT_PY_FUNC_BEGIN
  static PyArgParser parser({
    "get_ds_union()"
  });
  auto parsed_args = parser.parse(args, kwargs);
  if (parsed_args.signature_index() == 0) {
    return PyDistributedStatesUnion_New(self->tensor->cur_ds_union());
  } else {
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();
  }
  HT_PY_FUNC_END
}

PyObject* PyTensor_from_numpy_parallel(PyObject*, PyObject* args, PyObject* kwargs) {
  HT_PY_FUNC_BEGIN
  auto* unsafe_self = PyTensor_Type->tp_alloc(PyTensor_Type, 0);
  HT_RUNTIME_ERROR_IF(!unsafe_self) << "Failed to alloc PyTensor";
  auto* self = reinterpret_cast<PyTensor*>(unsafe_self);
  
  static PyArgParser parser({
    "from_numpy_parallel(numpy.array data, List[DistributedStatesUnion] ds_hierarchy, bool requires_grad=false, \
     ParameterDict parameter_dict=None, " OP_META_ARGS ")", 
    "from_numpy_parallel(numpy.array data, DataType dtype, List[DistributedStatesUnion] ds_hierarchy, bool requires_grad=false, \
     ParameterDict parameter_dict=None, " OP_META_ARGS ")", 
    "from_numpy_parallel(List[numpy.array] multi_data, List[DistributedStatesUnion] ds_hierarchy, bool requires_grad=false, \
     ParameterDict parameter_dict=None, " OP_META_ARGS ")", 
  });
  auto parsed_args = parser.parse(args, kwargs);

  if (parsed_args.signature_index() == 0) {
    auto* array_obj = parsed_args.get_numpy_array(0);
    DistributedStatesHierarchy ds_hierarchy = parsed_args.get_ds_hierarchy(1);
    bool requires_grad = parsed_args.get_bool_or_default(2);
    ParameterDict parameter_dict = parsed_args.get_parameter_dict_or_empty(3);
    new(&self->tensor) Tensor();
    self->tensor = MakeParallelVariableOp(NDArrayFromNumpy(array_obj), ds_hierarchy, false, kUndeterminedDataType, 
                                          requires_grad, parameter_dict, parse_op_meta(parsed_args, 4));
  } else if (parsed_args.signature_index() == 1) {
    auto* array_obj = parsed_args.get_numpy_array(0);
    DataType dtype = parsed_args.get_dtype(1);
    DistributedStatesHierarchy ds_hierarchy = parsed_args.get_ds_hierarchy(2);
    bool requires_grad = parsed_args.get_bool_or_default(3);  
    ParameterDict parameter_dict = parsed_args.get_parameter_dict_or_empty(4);
    new(&self->tensor) Tensor();
    self->tensor = MakeParallelVariableOp(NDArrayFromNumpy(array_obj, {}, dtype), ds_hierarchy, false, kUndeterminedDataType, 
                                          requires_grad, parameter_dict, parse_op_meta(parsed_args, 5));
  } else if (parsed_args.signature_index() == 2) {
    NDArrayList multi_data = parsed_args.get_numpy_array_list(0);
    DistributedStatesHierarchy ds_hierarchy = parsed_args.get_ds_hierarchy(1);
    bool requires_grad = parsed_args.get_bool_or_default(2);
    ParameterDict parameter_dict = parsed_args.get_parameter_dict_or_empty(3);
    new(&self->tensor) Tensor();
    self->tensor = MakeParallelVariableOp(multi_data, ds_hierarchy, false, kUndeterminedDataType, 
                                          requires_grad, parameter_dict, parse_op_meta(parsed_args, 4));
  } else {
    Py_TYPE(self)->tp_free(self);
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();
  }

  return reinterpret_cast<PyObject*>(self);
  HT_PY_FUNC_END
}

PyObject* PyTensor_to_numpy(PyTensor* self, PyObject* args, PyObject* kwargs) {
  HT_PY_FUNC_BEGIN
  static PyArgParser parser({
    "numpy(bool force=false, bool save=false)"
  });
  auto parsed_args = parser.parse(args, kwargs);
  if (parsed_args.signature_index() == 0) {
    bool force = parsed_args.get_bool_or_default(0);
    bool save = parsed_args.get_bool_or_default(1);
    return NDArrayToNumpy(self->tensor->get_or_compute(), force, save);
  } else {
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();
  }
  HT_PY_FUNC_END
}

PyObject* PyTensor_set_requires_grad(PyTensor* self, PyObject* args, PyObject* kwargs) {
  HT_PY_FUNC_BEGIN
  static PyArgParser parser({
    "set_requires_grad(bool requires_grad=false)"
  });
  auto parsed_args = parser.parse(args, kwargs);
  if (parsed_args.signature_index() == 0) {
    bool requires_grad = parsed_args.get_bool_or_default(0);
    self->tensor->set_requires_grad(requires_grad);
    auto producer_op = self->tensor->producer();
    if (is_variable_op(producer_op)) {
      if (producer_op->type() == "VariableOp")
        reinterpret_cast<VariableOpImpl&>(producer_op->body()).set_requires_grad(requires_grad);
      else
        reinterpret_cast<ParallelVariableOpImpl&>(producer_op->body()).set_requires_grad(requires_grad);
    }
  } else {
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();
  }
  Py_RETURN_NONE;
  HT_PY_FUNC_END
}

PyObject* PyTensor_copy(PyTensor* self, PyObject* args, PyObject* kwargs) {
  HT_PY_FUNC_BEGIN
  auto* unsafe_self = PyTensor_Type->tp_alloc(PyTensor_Type, 0);
  HT_RUNTIME_ERROR_IF(!unsafe_self) << "Failed to alloc PyTensor";
  auto* new_tensor = reinterpret_cast<PyTensor*>(unsafe_self);
    
  new(&new_tensor->tensor) Tensor();
    
  NDArray data = self->tensor->get_or_compute();
  NDArray data_copy = NDArray::copy(data, kBlockingStream);
  DataType dtype = self->tensor->dtype();
  bool requires_grad = self->tensor->requires_grad();
  DistributedStatesHierarchy ds_hierarchy = self->tensor->ds_hierarchy();
  
  new_tensor->tensor = MakeParameterOp(
    data_copy,
    false,
    dtype,
    requires_grad,
    ds_hierarchy
  );
    
  return reinterpret_cast<PyObject*>(new_tensor);
  HT_PY_FUNC_END
}

// NOLINTNEXTLINE
PyGetSetDef PyTensor_properties[] = {
  {PY_GET_SET_DEF_NAME("id"), (getter) PyTensor_id, nullptr, nullptr, nullptr}, 
  {PY_GET_SET_DEF_NAME("name"), (getter) PyTensor_name, nullptr, nullptr, nullptr}, 
  {PY_GET_SET_DEF_NAME("ndim"), (getter) PyTensor_ndim, nullptr, nullptr, nullptr}, 
  {PY_GET_SET_DEF_NAME("shape"), (getter) PyTensor_shape, nullptr, nullptr, nullptr}, 
  {PY_GET_SET_DEF_NAME("device"), (getter) PyTensor_device, nullptr, nullptr, nullptr}, 
  {PY_GET_SET_DEF_NAME("dtype"), (getter) PyTensor_dtype, nullptr, nullptr, nullptr},
  {PY_GET_SET_DEF_NAME("global_shape"), (getter) PyTensor_global_shape, nullptr, nullptr, nullptr},  
  {PY_GET_SET_DEF_NAME("symbolic_shape"), (getter) PyTensor_symbolic_shape, nullptr, nullptr, nullptr},  
  {PY_GET_SET_DEF_NAME("is_variable"), (getter) PyTensor_is_variable, nullptr, nullptr, nullptr}, 
  {PY_GET_SET_DEF_NAME("is_parameter"), (getter) PyTensor_is_parameter, nullptr, nullptr, nullptr}, 
  {PY_GET_SET_DEF_NAME("requires_grad"), (getter) PyTensor_requires_grad, nullptr, nullptr, nullptr},
  {PY_GET_SET_DEF_NAME("distributed_states"), (getter) PyTensor_distributed_states, nullptr, nullptr, nullptr},
  {PY_GET_SET_DEF_NAME("ds_hierarchy"), (getter) PyTensor_ds_hierarchy, nullptr, nullptr, nullptr},
  {PY_GET_SET_DEF_NAME("dg_hierarchy"), (getter) PyTensor_dg_hierarchy, nullptr, nullptr, nullptr},
  {PY_GET_SET_DEF_NAME("has_placement_group"), (getter) PyTensor_has_placement_group, nullptr, nullptr, nullptr},
  {PY_GET_SET_DEF_NAME("placement_group_union"), (getter) PyTensor_placement_group_union, nullptr, nullptr, nullptr},
  {PY_GET_SET_DEF_NAME("data"), (getter) PyTensor_data, nullptr, nullptr, nullptr}, 
  {PY_GET_SET_DEF_NAME("gpu_data"), (getter) PyTensor_gpu_data, nullptr, nullptr, nullptr}, 
  {PY_GET_SET_DEF_NAME("graph"), (getter) PyTensor_graph, nullptr, nullptr, nullptr}, 
  {nullptr}
};

PyTypeObject PyTensor_Type_obj = {
  PyVarObject_HEAD_INIT(nullptr, 0) 
  "hetu.Tensor", /* tp_name */
  sizeof(PyTensor), /* tp_basicsize */
  0, /* tp_itemsize */
  (destructor) PyTensor_dealloc, /* tp_dealloc */
  0, /* tp_vectorcall_offset */
  nullptr, /* tp_getattr */
  nullptr, /* tp_setattr */
  nullptr, /* tp_reserved */
  (reprfunc) PyTensor_repr, /* tp_repr */
  nullptr, /* tp_as_number */
  nullptr, /* tp_as_sequence */
  nullptr, /* tp_as_mapping */
  (hashfunc) PyTensor_hash, /* tp_hash  */
  nullptr, /* tp_call */
  (reprfunc) PyTensor_str, /* tp_str */
  nullptr, /* tp_getattro */
  nullptr, /* tp_setattro */
  nullptr, /* tp_as_buffer */
  Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /* tp_flags */
  nullptr, /* tp_doc */
  nullptr, /* tp_traverse */
  nullptr, /* tp_clear */
  (richcmpfunc) PyTensor_richcompare, /* tp_richcompare */
  0, /* tp_weaklistoffset */
  nullptr, /* tp_iter */
  nullptr, /* tp_iternext */
  nullptr, /* tp_methods */
  nullptr, /* tp_members */
  PyTensor_properties, /* tp_getset */
  nullptr, /* tp_base */
  nullptr, /* tp_dict */
  nullptr, /* tp_descr_get */
  nullptr, /* tp_descr_set */
  0, /* tp_dictoffset */
  nullptr, /* tp_init */
  nullptr, /* tp_alloc */
  PyTensor_pynew, /* tp_new */
};
PyTypeObject* PyTensor_Type = &PyTensor_Type_obj;

std::vector<PyMethodDef> InitTensorPyMethodDefs() {
  std::vector<PyMethodDef> ret = {{nullptr}};
  AddPyMethodDefs(ret, {
    {"dim", (PyCFunction) PyTensor_dim, METH_VARARGS | METH_KEYWORDS, nullptr }, 
    {"size", (PyCFunction) PyTensor_shape, METH_VARARGS | METH_KEYWORDS, nullptr }, 
    {"numel", (PyCFunction) PyTensor_numel, METH_VARARGS | METH_KEYWORDS, nullptr }, 
    {"global_numel", (PyCFunction) PyTensor_global_numel, METH_VARARGS | METH_KEYWORDS, nullptr }, 
    {"timecost", (PyCFunction) PyTensor_timecost, METH_VARARGS | METH_KEYWORDS, nullptr }, 
    {"stride", (PyCFunction) PyTensor_stride, METH_VARARGS | METH_KEYWORDS, nullptr }, 
    {"numpy", (PyCFunction) PyTensor_to_numpy, METH_VARARGS | METH_KEYWORDS, nullptr }, 
    {"to", (PyCFunction) PyTensor_data_transfer, METH_VARARGS | METH_KEYWORDS, nullptr },
    {"copy", (PyCFunction) PyTensor_copy, METH_NOARGS, nullptr},
    {"reset_data", (PyCFunction) PyTensor_reset_data, METH_VARARGS | METH_KEYWORDS, nullptr },
    {"reset_data_from_splits", (PyCFunction) PyTensor_reset_data_from_splits, METH_VARARGS | METH_KEYWORDS, nullptr },
    {"get_data", (PyCFunction) PyTensor_get_data, METH_VARARGS | METH_KEYWORDS, nullptr },
    {"raw_data_ptr", (PyCFunction) PyTensor_raw_data_ptr, METH_VARARGS | METH_KEYWORDS, nullptr },
    {"is_floating_point", (PyCFunction) PyTensor_is_floating_point, METH_NOARGS, nullptr },
    {"get_device_group_union", (PyCFunction) PyTensor_get_device_group_union, METH_VARARGS | METH_KEYWORDS, nullptr },
    {"get_ds_union", (PyCFunction) PyTensor_get_ds_union, METH_VARARGS | METH_KEYWORDS, nullptr },
    {"check_ds_hierarchy_equal", (PyCFunction) PyTensor_check_ds_hierarchy_equal, METH_VARARGS | METH_KEYWORDS, nullptr },
    {"check_ds_hierarchy_equal_except_split", (PyCFunction) PyTensor_check_ds_hierarchy_equal_except_split, METH_VARARGS | METH_KEYWORDS, nullptr },
    {"get_or_compute", (PyCFunction) PyTensor_get_or_compute, METH_NOARGS, nullptr }, 
    {"symbolic", (PyCFunction) PyTensor_symbolic, METH_NOARGS, nullptr }, 
    {"set_requires_grad", (PyCFunction) PyTensor_set_requires_grad, METH_VARARGS | METH_KEYWORDS, nullptr },
    {"is_contiguous", (PyCFunction) PyTensor_is_contiguous, METH_NOARGS, nullptr },
    {"_make_subclass", (PyCFunction) PyTensor_make_subclass, METH_CLASS | METH_VARARGS | METH_KEYWORDS, nullptr }, 
    {"_make_recompute", (PyCFunction) PyTensor_make_recompute, METH_VARARGS | METH_KEYWORDS, nullptr },
    {nullptr}
  });
  AddPyMethodDefs(ret, hetu::graph::get_registered_tensor_methods());
  return ret;
}

std::vector<PyMethodDef> InitTensorPyClassMethodDefs() {
  std::vector<PyMethodDef> ret = {{nullptr}};
  AddPyMethodDefs(ret, {
    {"from_numpy", (PyCFunction) PyTensor_from_numpy, METH_VARARGS | METH_KEYWORDS, nullptr }, 
    {"from_numpy_parallel", (PyCFunction) PyTensor_from_numpy_parallel, METH_VARARGS | METH_KEYWORDS, nullptr },     
    {"from_buffer", (PyCFunction) PyTensor_from_buffer, METH_VARARGS | METH_KEYWORDS, nullptr },
    {nullptr}
  });
  AddPyMethodDefs(ret, hetu::graph::get_registered_tensor_class_methods());
  return ret;
}

void AddPyTensorTypeToModule(py::module_& module) {
  PyTensor_Type->tp_as_number = &(get_registered_tensor_number_methods());
  static auto tensor_methods = InitTensorPyMethodDefs();
  PyTensor_Type->tp_methods = tensor_methods.data();
  HT_RUNTIME_ERROR_IF(PyType_Ready(PyTensor_Type) < 0) 
    << "PyTensor_Type not ready";
  Py_INCREF(PyTensor_Type);
  HT_RUNTIME_ERROR_IF(0 != PyModule_AddObject(
      module.ptr(), "Tensor", reinterpret_cast<PyObject*>(PyTensor_Type)))
    << "Failed to add PyTensor_Type";
  
  static auto tensor_class_methods = InitTensorPyClassMethodDefs();
  HT_RUNTIME_ERROR_IF(0 != PyModule_AddFunctions(
      module.ptr(), tensor_class_methods.data()))
    << "Failed to add Tensor class methods";
}

} // namespace graph
} // namespace hetu
