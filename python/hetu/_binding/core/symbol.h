#pragma once

#include <Python.h>
#include <type_traits>
#include "hetu/core/symbol.h"
#include "hetu/_binding/utils/pybind_common.h"

namespace hetu {

struct PyIntSymbol {
  PyObject_HEAD;
  IntSymbol int_symbol;
};

extern PyTypeObject* PyIntSymbol_Type;

inline bool PyIntSymbol_Check(PyObject* obj) {
  return PyIntSymbol_Type && PyObject_TypeCheck(obj, PyIntSymbol_Type);
}

inline bool PyIntSymbol_CheckExact(PyObject* obj) {
  return PyIntSymbol_Type && obj->ob_type == PyIntSymbol_Type;
}

PyObject* PyIntSymbol_New(const IntSymbol& int_symbol);

PyObject* PySyShape_New(const SyShape& symbol_shape);

void AddPyIntSymbolTypeToModule(py::module_& module);

/******************************************************
 * ArgParser Utils
 ******************************************************/

inline bool CheckPyIntSymbol(PyObject* obj) {
  return PyIntSymbol_Check(obj);
}

inline IntSymbol IntSymbol_FromPyObject(PyObject* obj) {
  return reinterpret_cast<PyIntSymbol*>(obj)->int_symbol;
}

inline bool CheckPySyShape(PyObject* obj) {
  bool is_tuple = PyTuple_Check(obj);
  if (is_tuple || PyList_Check(obj)) {
    size_t size = is_tuple ? PyTuple_GET_SIZE(obj) : PyList_GET_SIZE(obj);
    if (size > 0) {
      // only check for the first item for efficiency
      auto* item = is_tuple ? PyTuple_GET_ITEM(obj, 0) \
                            : PyList_GET_ITEM(obj, 0);
      if (!CheckPyIntSymbol(item))
        return false;
    }
    return true;
  }
  return false;
}

inline SyShape SyShape_FromPyObject(PyObject* obj) {
  bool is_tuple = PyTuple_Check(obj);
  size_t size = is_tuple ? PyTuple_GET_SIZE(obj) : PyList_GET_SIZE(obj);
  SyShape ret(size);
  for (size_t i = 0; i < size; i++) {
    auto* item = is_tuple ? PyTuple_GET_ITEM(obj, i) : PyList_GET_ITEM(obj, i);
    ret[i] = IntSymbol_FromPyObject(item);
  }
  return ret;
}

inline bool CheckPySyShapeList(PyObject* obj) {
  bool is_tuple = PyTuple_Check(obj);
  if (is_tuple || PyList_Check(obj)) {
    size_t size = is_tuple ? PyTuple_GET_SIZE(obj) : PyList_GET_SIZE(obj);
    if (size > 0) {
      // only check for the first item for efficiency
      auto* item = is_tuple ? PyTuple_GET_ITEM(obj, 0) \
                            : PyList_GET_ITEM(obj, 0);
      if (!CheckPySyShape(item))
        return false;
    }
    return true;
  }
  return false;
}

inline SyShapeList SyShapeList_FromPyObject(PyObject* obj) {
  bool is_tuple = PyTuple_Check(obj);
  size_t size = is_tuple ? PyTuple_GET_SIZE(obj) : PyList_GET_SIZE(obj);
  SyShapeList ret(size);
  for (size_t i = 0; i < size; i++) {
    auto* item = is_tuple ? PyTuple_GET_ITEM(obj, i) : PyList_GET_ITEM(obj, i);
    ret[i] = SyShape_FromPyObject(item);
  }
  return ret;
}

inline bool CheckPySyShapeListList(PyObject* obj) {
  bool is_tuple = PyTuple_Check(obj);
  if (is_tuple || PyList_Check(obj)) {
    size_t size = is_tuple ? PyTuple_GET_SIZE(obj) : PyList_GET_SIZE(obj);
    if (size > 0) {
      // only check for the first item for efficiency
      auto* item = is_tuple ? PyTuple_GET_ITEM(obj, 0) \
                            : PyList_GET_ITEM(obj, 0);
      if (!CheckPySyShapeList(item))
        return false;
    }
    return true;
  }
  return false;
}

inline SyShapeListList SyShapeListList_FromPyObject(PyObject* obj) {
  bool is_tuple = PyTuple_Check(obj);
  size_t size = is_tuple ? PyTuple_GET_SIZE(obj) : PyList_GET_SIZE(obj);
  SyShapeListList ret(size);
  for (size_t i = 0; i < size; i++) {
    auto* item = is_tuple ? PyTuple_GET_ITEM(obj, i) : PyList_GET_ITEM(obj, i);
    ret[i] = SyShapeList_FromPyObject(item);
  }
  return ret;
}

} // namespace hetu
