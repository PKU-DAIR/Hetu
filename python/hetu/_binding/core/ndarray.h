#pragma once

#include <Python.h>
#include "hetu/core/ndarray.h"
#include "hetu/utils/optional.h"
#include "hetu/_binding/utils/pybind_common.h"

namespace hetu {

struct PyNDArray {
  PyObject_HEAD;
  NDArray ndarray;
};

extern PyTypeObject* PyNDArray_Type;

inline bool PyNDArray_Check(PyObject* obj) {
  return PyNDArray_Type && PyObject_TypeCheck(obj, PyNDArray_Type);
}

inline bool PyNDArray_CheckExact(PyObject* obj) {
  return PyNDArray_Type && obj->ob_type == PyNDArray_Type;
}

PyObject* PyNDArray_New(const NDArray& ndarray,
                        bool return_none_if_undefined = true);

PyObject* PyNDArrayList_New(const NDArrayList& ndarray_list);

void AddPyNDArrayTypeToModule(py::module_& module);

/******************************************************
 * Constructor Utils
 ******************************************************/

NDArray NDArrayCopyFromNDArrayCtor(const NDArray& ndarray,
                                   optional<DataType>&& dtype,
                                   optional<Device>&& device);

NDArray NDArrayCopyFromNumpyCtor(PyObject* obj, optional<DataType>&& dtype,
                                 optional<Device>&& device);

NDArray NDArrayCopyFromSequenceCtor(PyObject* obj, optional<DataType>&& dtype,
                                    optional<Device>&& device);

NDArray NDArrayFromBuffer(void* data, DataType dtype, const HTShape& shape);

/******************************************************
 * ArgParser Utils
 ******************************************************/

inline bool CheckPyNDArray(PyObject* obj) {
  return PyNDArray_Check(obj);
}

inline NDArray NDArray_FromPyObject(PyObject* obj) {
  return reinterpret_cast<PyNDArray*>(obj)->ndarray;
}

inline bool CheckPyNDArrayList(PyObject* obj) {
  bool is_tuple = PyTuple_Check(obj);
  if (is_tuple || PyList_Check(obj)) {
    size_t size = is_tuple ? PyTuple_GET_SIZE(obj) : PyList_GET_SIZE(obj);
    if (size > 0) {
      // only check for the first item for efficiency
      auto* item = is_tuple ? PyTuple_GET_ITEM(obj, 0) \
                            : PyList_GET_ITEM(obj, 0);
      if (!CheckPyNDArray(item))
        return false;
    }
    return true;
  }
  return false;
}

inline NDArrayList NDArrayList_FromPyObject(PyObject* obj) {
  bool is_tuple = PyTuple_Check(obj);
  size_t size = is_tuple ? PyTuple_GET_SIZE(obj) : PyList_GET_SIZE(obj);
  NDArrayList ret(size);
  for (size_t i = 0; i < size; i++) {
    auto* item = is_tuple ? PyTuple_GET_ITEM(obj, i) : PyList_GET_ITEM(obj, i);
    ret[i] = NDArray_FromPyObject(item);
  }
  return ret;
}

} // namespace hetu
