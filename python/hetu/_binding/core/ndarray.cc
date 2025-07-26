#include "hetu/_binding/core/ndarray.h"
#include "hetu/_binding/constants.h"
#include "hetu/_binding/utils/function_registry.h"
#include "hetu/_binding/utils/pybind_common.h"
#include "hetu/_binding/utils/except.h"
#include "hetu/_binding/utils/decl_utils.h"
#include "hetu/_binding/utils/arg_parser.h"
#include "hetu/graph/ops/kernel_links.h"

namespace hetu {

NDArray NDArrayCopyFromNDArrayCtor(const NDArray& ndarray,
                                   optional<DataType>&& dtype,
                                   optional<Device>&& device) {
  // `NDArray::to` does not trigger data transmission
  // when data types and devices are the same.
  // Use `NDArray::copy` instead.
  bool same_dtype = dtype == nullopt || dtype == ndarray->dtype();
  bool same_device = device == nullopt || device == ndarray->device();
  if (same_dtype && same_device) {
    return NDArray::copy(ndarray, kBlockingStream);
  } else {
    return NDArray::to(
      ndarray, std::move(device.value_or(Device(kUndeterminedDevice))),
      std::move(dtype.value_or(kUndeterminedDataType)), kBlockingStream);
  }
}

NDArray NDArrayCopyFromNumpyCtor(PyObject* obj, optional<DataType>&& dtype,
                                 optional<Device>&& device) {
  return NDArrayCopyFromNDArrayCtor(NDArrayFromNumpy(obj), std::move(dtype),
                                    std::move(device));
}

NDArray NDArrayCopyFromSequenceCtor(PyObject* obj, optional<DataType>&& dtype,
                                    optional<Device>&& device) {
  auto* numpy_array = NumpyFromSequences(obj);
  // For sequences of floating points, Numpy uses np.float64 by default.
  // We should cast it to float32 here.
  // In fact, Numpy uses np.int64 for sequences of integers, too.
  // However, we follow PyTorch to keep it and do not cast it to int32. =
  NDArray ret;
  if (!dtype && GetNumpyArrayDataType(numpy_array) == kFloat64) {
    ret = NDArrayCopyFromNumpyCtor(numpy_array, kFloat32, std::move(device));
  } else {
    ret = NDArrayCopyFromNumpyCtor(numpy_array, std::move(dtype),
                                   std::move(device));
  }
  Py_DECREF(numpy_array);
  return ret;
}

NDArray NDArrayFromBuffer(void* data, DataType dtype, const HTShape& shape) {
  size_t size = NumEl(shape) * DataType2Size(dtype);
  NDArray array = NDArray::empty(shape, Device(kCPU), dtype);
  void* dst = array->raw_data_ptr();
  memcpy(dst, data, size);
  return array;
}

PyObject* PyNDArray_New(const NDArray& ndarray, bool return_none_if_undefined) {
  HT_PY_FUNC_BEGIN
  if (return_none_if_undefined && !ndarray.is_defined()) {
    Py_RETURN_NONE;
  } else {
    auto* unsafe_self = PyNDArray_Type->tp_alloc(PyNDArray_Type, 0);
    HT_RUNTIME_ERROR_IF(!unsafe_self) << "Failed to alloc PyNDArray";
    auto* self = reinterpret_cast<PyNDArray*>(unsafe_self);
    new(&self->ndarray) NDArray();
    self->ndarray = ndarray;
    return reinterpret_cast<PyObject*>(self);
  }
  HT_PY_FUNC_END
}

PyObject* PyNDArrayList_New(const NDArrayList& ndarray_list) {
  HT_PY_FUNC_BEGIN
  auto* py_ndarray_list = PyList_New(ndarray_list.size());
  HT_RUNTIME_ERROR_IF(!py_ndarray_list) << "Failed to alloc list";
  for (size_t i = 0; i < ndarray_list.size(); i++) {
    auto* ndarray_obj = PyNDArray_New(ndarray_list[i]);
    PyList_SET_ITEM(py_ndarray_list, i, ndarray_obj);
  }
  return py_ndarray_list;
  HT_PY_FUNC_END
}

PyObject* PyNDArray_pynew(PyTypeObject* type, PyObject* args, 
                          PyObject* kwargs) {
  HT_PY_FUNC_BEGIN
  auto* unsafe_self = PyNDArray_Type->tp_alloc(PyNDArray_Type, 0);
  HT_RUNTIME_ERROR_IF(!unsafe_self) << "Failed to alloc PyNDArray";
  auto* self = reinterpret_cast<PyNDArray*>(unsafe_self);
  
  static PyArgParser parser({
    "NDArray(numpy.array data, DataType dtype=None, Device device=None)", 
    "NDArray(NDArray data, DataType dtype=None, Device device=None)", 
    "NDArray(PyObject* data, DataType dtype=None, Device device=None)"
  });
  auto parsed_args = parser.parse(args, kwargs);
  
  if (parsed_args.signature_index() == 0) {
    new(&self->ndarray) NDArray();
    self->ndarray = NDArrayCopyFromNumpyCtor(
      parsed_args.get_numpy_array(0), 
      parsed_args.get_dtype_or_peek(1), 
      parsed_args.get_device_or_peek(2));
  } else if (parsed_args.signature_index() == 1) {
    new(&self->ndarray) NDArray();
    self->ndarray = NDArrayCopyFromNDArrayCtor(
      parsed_args.get_ndarray(0), 
      parsed_args.get_dtype_or_peek(1), 
      parsed_args.get_device_or_peek(2));
  } else if (parsed_args.signature_index() == 2) {
    new(&self->ndarray) NDArray();
    self->ndarray = NDArrayCopyFromSequenceCtor(
      parsed_args.get_py_obj(0), 
      parsed_args.get_dtype_or_peek(1), 
      parsed_args.get_device_or_peek(2));
  } else {
    Py_TYPE(self)->tp_free(self);
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();
  }
  
  return reinterpret_cast<PyObject*>(self);
  HT_PY_FUNC_END
}

void PyNDArray_dealloc(PyNDArray* self) {
  (&self->ndarray)->~NDArray();
  Py_TYPE(self)->tp_free(self);
}

PyObject* PyNDArray_str(PyNDArray* self) {
  HT_PY_FUNC_BEGIN
  Py_RETURN_UNICODE_FROM_OSS(self->ndarray);
  HT_PY_FUNC_END
}

PyObject* PyNDArray_repr(PyNDArray* self) {
  return PyNDArray_str(self);
}

// TODO: shape and size

PyObject* PyNDArray_device(PyNDArray* self) {
  HT_PY_FUNC_BEGIN
  return PyDevice_New(self->ndarray->device());
  HT_PY_FUNC_END
}

PyObject* PyNDArray_dtype(PyNDArray* self) {
  HT_PY_FUNC_BEGIN
  return PyDataType_New(self->ndarray->dtype());
  HT_PY_FUNC_END
}

PyObject* PyNDArray_shape(PyNDArray* self) {
  HT_PY_FUNC_BEGIN
  return PyLongList_FromIntegerList(self->ndarray->shape());
  HT_PY_FUNC_END
}

PyObject* PyNDArray_data_ptr(PyNDArray* self) {
  HT_PY_FUNC_BEGIN
  return PyLong_FromInteger(int64_t(self->ndarray->raw_data_ptr()));
  HT_PY_FUNC_END
}

PyObject* PyNDArray_shm_offset(PyNDArray* self) {
  HT_PY_FUNC_BEGIN
  return PyLong_FromInteger(int64_t(self->ndarray->shm_offset()));
  HT_PY_FUNC_END
}

PyObject* PyNDArray_numel(PyNDArray* self) {
  HT_PY_FUNC_BEGIN
  return PyLong_FromInteger(self->ndarray->numel());
  HT_PY_FUNC_END
}

PyObject* PyNDArray_from_numpy(PyObject*, PyObject* args, PyObject* kwargs) {
  HT_PY_FUNC_BEGIN
  auto* unsafe_self = PyNDArray_Type->tp_alloc(PyNDArray_Type, 0);
  HT_RUNTIME_ERROR_IF(!unsafe_self) << "Failed to alloc PyNDArray";
  auto* self = reinterpret_cast<PyNDArray*>(unsafe_self);
  
  static PyArgParser parser({
    "numpy_to_NDArray(numpy.array data)",
    "numpy_to_NDArray(numpy.array data, HTShape dynamic_shape)",
    "numpy_to_NDArray(numpy.array data, DataType dtype)",
  });
  auto parsed_args = parser.parse(args, kwargs);

  if (parsed_args.signature_index() == 0) {
    auto* array_obj = parsed_args.get_numpy_array(0);
    new(&self->ndarray) NDArray();
    self->ndarray = NDArrayFromNumpy(array_obj);
  } else if (parsed_args.signature_index() == 1) {
    auto* array_obj1 = parsed_args.get_numpy_array(0);
    HTShape dynamic_shape = parsed_args.get_int64_list(1);
    new(&self->ndarray) NDArray();
    self->ndarray = NDArrayFromNumpy(array_obj1, dynamic_shape);
  } else if (parsed_args.signature_index() == 2) {
    auto* array_obj1 = parsed_args.get_numpy_array(0);
    DataType dtype = parsed_args.get_dtype(1);
    new(&self->ndarray) NDArray();
    self->ndarray = NDArrayFromNumpy(array_obj1, {}, dtype);
  } else {
    Py_TYPE(self)->tp_free(self);
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();
  }

  return reinterpret_cast<PyObject*>(self);
  HT_PY_FUNC_END
}

PyObject* PyNDArray_from_buffer(PyObject*, PyObject* args, PyObject* kwargs) {
  HT_PY_FUNC_BEGIN
  auto* unsafe_self = PyNDArray_Type->tp_alloc(PyNDArray_Type, 0);
  HT_RUNTIME_ERROR_IF(!unsafe_self) << "Failed to alloc PyNDArray";
  auto* self = reinterpret_cast<PyNDArray*>(unsafe_self);
  
  static PyArgParser parser({
    "buffer_to_NDArray(bytes data, DataType dtype, List[int] shape)",
    "buffer_to_NDArray(bytearray data, DataType dtype, List[int] shape)",
  });
  auto parsed_args = parser.parse(args, kwargs);

  if (parsed_args.signature_index() == 0) {
    Py_buffer view;
    PyObject* buffer_obj = parsed_args.get_bytes(0);
    if (PyObject_GetBuffer(buffer_obj, &view, PyBUF_SIMPLE) < 0) {
      PyErr_SetString(PyExc_TypeError, "Unable to get buffer from object");
      Py_TYPE(self)->tp_free(self);
      return nullptr;
    }

    DataType dtype = parsed_args.get_dtype(1);
    HTShape shape = parsed_args.get_int64_list(2);

    size_t expected_size = NumEl(shape) * DataType2Size(dtype);
    if (view.len < expected_size) {
      PyBuffer_Release(&view);
      PyErr_SetString(PyExc_ValueError, 
        "buffer size does not match the expected size");
      Py_TYPE(self)->tp_free(self);
      return nullptr;
    }
    new(&self->ndarray) NDArray();
    self->ndarray = NDArrayFromBuffer(view.buf, dtype, shape);

    PyBuffer_Release(&view);
    return reinterpret_cast<PyObject*>(self);
  } else if (parsed_args.signature_index() == 1) {
    Py_buffer view;
    PyObject* buffer_obj = parsed_args.get_bytearray(0);
    if (PyObject_GetBuffer(buffer_obj, &view, PyBUF_SIMPLE) < 0) {
      PyErr_SetString(PyExc_TypeError, "Unable to get buffer from object");
      Py_TYPE(self)->tp_free(self);
      return nullptr;
    }

    DataType dtype = parsed_args.get_dtype(1);
    HTShape shape = parsed_args.get_int64_list(2);

    size_t expected_size = NumEl(shape) * DataType2Size(dtype);
    if (view.len < expected_size) {
      PyBuffer_Release(&view);
      PyErr_SetString(PyExc_ValueError, 
        "buffer size does not match the expected size");
      Py_TYPE(self)->tp_free(self);
      return nullptr;
    }
    new(&self->ndarray) NDArray();
    self->ndarray = NDArrayFromBuffer(view.buf, dtype, shape);

    PyBuffer_Release(&view);
    return reinterpret_cast<PyObject*>(self);
  } else {
    Py_TYPE(self)->tp_free(self);
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();
  }
  HT_PY_FUNC_END
}

PyObject* PyNDArray_to_numpy(PyNDArray* self, PyObject* args, 
                             PyObject* kwargs) {
  HT_PY_FUNC_BEGIN
  HT_VALUE_ERROR_IF(!self->ndarray.is_defined()) 
    << "NDArray is not defined";
  static PyArgParser parser({
    "numpy(bool force=false, bool save=false)"
  });
  auto parsed_args = parser.parse(args, kwargs);
  if (parsed_args.signature_index() == 0) {
    bool force = parsed_args.get_bool_or_default(0);
    bool save = parsed_args.get_bool_or_default(1);
    return NDArrayToNumpy(self->ndarray, force, save);
  } else {
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();
  }
  HT_PY_FUNC_END
}

PyObject* PyNDArray_to(PyNDArray* self, PyObject* args, PyObject* kwargs) {
  HT_PY_FUNC_BEGIN
  static PyArgParser parser({
    "to(DataType dtype)", 
    "to(Device device=None, DataType dtype=None)", 
    "to(NDArray other)"
  });
  auto parsed_args = parser.parse(args, kwargs);

  if (parsed_args.signature_index() == 0) {
    auto dtype = parsed_args.get_dtype_or_peek(0);
    if (dtype == nullopt || dtype == self->ndarray->dtype()) {
      return reinterpret_cast<PyObject*>(self);
    } else {
      auto out = NDArray::to(self->ndarray, Device(), *dtype, kBlockingStream);
      return PyNDArray_New(out);
    }
  } else if (parsed_args.signature_index() == 1) {
    auto device = parsed_args.get_device_or_peek(0);
    auto dtype = parsed_args.get_dtype_or_peek(1);
    bool same_device = device == nullopt || device == self->ndarray->device();
    bool same_dtype = dtype == nullopt || dtype == self->ndarray->dtype();
    if (same_device && same_dtype) {
      return reinterpret_cast<PyObject*>(self);
    } else {
      auto out = NDArray::to(
        self->ndarray, 
        device.value_or(Device()), 
        dtype.value_or(kUndeterminedDataType), 
        kBlockingStream);
      return PyNDArray_New(out);
    }
  } else if (parsed_args.signature_index() == 2) {
    auto other = parsed_args.get_ndarray(0);
    bool same_device = other->device() == self->ndarray->device();
    bool same_dtype = other->dtype() == self->ndarray->dtype();
    if (same_device && same_dtype) {
      return reinterpret_cast<PyObject*>(self);
    } else {
      auto out = NDArray::to(
        self->ndarray, 
        other->device(), 
        other->dtype(), 
        kBlockingStream);
      return PyNDArray_New(out);
    }
  } else {
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();
  }
  HT_PY_FUNC_END
}

PyObject* PyNDArray_slice(PyNDArray* self, PyObject* args, PyObject* kwargs) {
  HT_PY_FUNC_BEGIN
  static PyArgParser parser({
    "slice(HTShape begin_pos, HTShape output_shape)",
  });
  auto parsed_args = parser.parse(args, kwargs);

  if (parsed_args.signature_index() == 0) {
    auto input = self->ndarray;
    auto out = NDArray::empty(parsed_args.get_int64_list(1), input->device(),
                              input->dtype(), kBlockingStream,
                              parsed_args.get_int64_list(1));
    Stream stream(input->device(), kBlockingStream);
    HT_DISPATCH_KERNEL_CPU_AND_CUDA(input->device().type(), __FUNCTION__,
                                    hetu::impl::Slice, input, out, 
                                    parsed_args.get_int64_list(0), stream);
    return PyNDArray_New(out);
  } else {
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();
  }
  HT_PY_FUNC_END
}

PyObject* PyNDArray_transpose(PyNDArray* self, PyObject* args, PyObject* kwargs) {
  HT_PY_FUNC_BEGIN
  static PyArgParser parser({
    "transpose(HTShape perms)",
  });
  auto parsed_args = parser.parse(args, kwargs);

  if (parsed_args.signature_index() == 0) {
    auto out = NDArray::permute(self->ndarray, parsed_args.get_int64_list(0), kBlockingStream);
    return PyNDArray_New(out);
  } else {
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();
  }
  HT_PY_FUNC_END
}

PyObject* PyNDArray_copy(PyNDArray* self, PyObject* args, PyObject* kwargs) {
  HT_PY_FUNC_BEGIN
  auto out = NDArray::copy(self->ndarray, kBlockingStream);
  return PyNDArray_New(out);
  HT_PY_FUNC_END
}

PyObject* PyNDArray_contiguous(PyNDArray* self, PyObject* args, PyObject* kwargs) {
  HT_PY_FUNC_BEGIN
  auto out = NDArray::contiguous(self->ndarray, kBlockingStream);
  return PyNDArray_New(out);
  HT_PY_FUNC_END
}

PyObject* PyNDArray_view(PyNDArray* self, PyObject* args, PyObject* kwargs) {
  HT_PY_FUNC_BEGIN
  static PyArgParser parser({
    "view(HTShape shape)",
  });
  auto parsed_args = parser.parse(args, kwargs);

  if (parsed_args.signature_index() == 0) {
    auto out = NDArray::view(self->ndarray, parsed_args.get_int64_list(0));
    return PyNDArray_New(out);
  } else {
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();
  }
  HT_PY_FUNC_END
}

// NOLINTNEXTLINE
PyGetSetDef PyNDArray_properties[] = {
  {PY_GET_SET_DEF_NAME("device"), (getter) PyNDArray_device, nullptr, nullptr, nullptr}, 
  {PY_GET_SET_DEF_NAME("dtype"), (getter) PyNDArray_dtype, nullptr, nullptr, nullptr},
  {PY_GET_SET_DEF_NAME("shape"), (getter) PyNDArray_shape, nullptr, nullptr, nullptr}, 
  {PY_GET_SET_DEF_NAME("data_ptr"), (getter) PyNDArray_data_ptr, nullptr, nullptr, nullptr}, 
  {PY_GET_SET_DEF_NAME("shm_offset"), (getter) PyNDArray_shm_offset, nullptr, nullptr, nullptr}, 
  {nullptr}
};

// NOLINTNEXTLINE
PyTypeObject PyNDArray_Type_obj = {
  PyVarObject_HEAD_INIT(nullptr, 0) 
  "hetu.NDArray", /* tp_name */
  sizeof(PyNDArray), /* tp_basicsize */
  0, /* tp_itemsize */
  (destructor) PyNDArray_dealloc, /* tp_dealloc */
  0, /* tp_vectorcall_offset */
  nullptr, /* tp_getattr */
  nullptr, /* tp_setattr */
  nullptr, /* tp_reserved */
  (reprfunc) PyNDArray_repr, /* tp_repr */
  nullptr, /* tp_as_number */
  nullptr, /* tp_as_sequence */
  nullptr, /* tp_as_mapping */
  nullptr, /* tp_hash  */
  nullptr, /* tp_call */
  (reprfunc) PyNDArray_str, /* tp_str */
  nullptr, /* tp_getattro */
  nullptr, /* tp_setattro */
  nullptr, /* tp_as_buffer */
  Py_TPFLAGS_DEFAULT, /* tp_flags */
  nullptr, /* tp_doc */
  nullptr, /* tp_traverse */
  nullptr, /* tp_clear */
  nullptr, /* tp_richcompare */
  0, /* tp_weaklistoffset */
  nullptr, /* tp_iter */
  nullptr, /* tp_iternext */
  nullptr, /* tp_methods */
  nullptr, /* tp_members */
  PyNDArray_properties, /* tp_getset */
  nullptr, /* tp_base */
  nullptr, /* tp_dict */
  nullptr, /* tp_descr_get */
  nullptr, /* tp_descr_set */
  0, /* tp_dictoffset */
  nullptr, /* tp_init */
  nullptr, /* tp_alloc */
  PyNDArray_pynew, /* tp_new */
};
PyTypeObject* PyNDArray_Type = &PyNDArray_Type_obj;

std::vector<PyMethodDef> InitNDArrayPyMethodDefs() {
  std::vector<PyMethodDef> ret = {{nullptr}};
  AddPyMethodDefs(ret, {
    {"numpy", (PyCFunction) PyNDArray_to_numpy, METH_VARARGS | METH_KEYWORDS, nullptr }, 
    {"numel", (PyCFunction) PyNDArray_numel, METH_NOARGS, nullptr }, 
    {"to", (PyCFunction) PyNDArray_to, METH_VARARGS | METH_KEYWORDS, nullptr }, 
    {"slice", (PyCFunction) PyNDArray_slice, METH_VARARGS | METH_KEYWORDS, nullptr }, 
    {"copy", (PyCFunction) PyNDArray_copy, METH_NOARGS, nullptr }, 
    {"contiguous", (PyCFunction) PyNDArray_contiguous, METH_NOARGS, nullptr }, 
    {"transpose", (PyCFunction) PyNDArray_transpose, METH_VARARGS | METH_KEYWORDS, nullptr }, 
    {"view", (PyCFunction) PyNDArray_view, METH_VARARGS | METH_KEYWORDS, nullptr }, 
    {nullptr}
  });
  AddPyMethodDefs(ret, hetu::impl::get_registered_ndarray_methods());
  return ret;
}

std::vector<PyMethodDef> InitNDArrayPyClassMethodDefs() {
  std::vector<PyMethodDef> ret = {{nullptr}};
  AddPyMethodDefs(ret, {
    // TODO: wrap from_numpy of NDArray in a capsule
    {"numpy_to_NDArray", (PyCFunction) PyNDArray_from_numpy, METH_VARARGS | METH_KEYWORDS, nullptr }, 
    {"buffer_to_NDArray", (PyCFunction) PyNDArray_from_buffer, METH_VARARGS | METH_KEYWORDS, nullptr },
    {nullptr}
  });
  AddPyMethodDefs(ret, hetu::impl::get_registered_ndarray_class_methods());
  return ret;
}

void AddPyNDArrayTypeToModule(py::module_& module) {
  static auto ndarray_methods = InitNDArrayPyMethodDefs();
  PyNDArray_Type->tp_methods = ndarray_methods.data();
  HT_RUNTIME_ERROR_IF(PyType_Ready(PyNDArray_Type) < 0) 
    << "PyNDArray_Type not ready";
  Py_INCREF(PyNDArray_Type);
  HT_RUNTIME_ERROR_IF(0 != PyModule_AddObject(
      module.ptr(), "NDArray", reinterpret_cast<PyObject*>(PyNDArray_Type)))
    << "Failed to add PyNDArray_Type";
  
  static auto ndarray_class_methods = InitNDArrayPyClassMethodDefs();
  // TODO: wrap into a submodule?
  HT_RUNTIME_ERROR_IF(0 != PyModule_AddFunctions(
      module.ptr(), ndarray_class_methods.data()))
    << "Failed to add NDArray class methods";
}

} // namespace hetu
