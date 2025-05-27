#pragma once

#include <Python.h>
#include <any>
#include <boolobject.h>
#include <cstdint>
#include <floatobject.h>
#include <longobject.h>
#include "hetu/_binding/utils/python_primitives.h"
#include "hetu/graph/operator.h"
#include "hetu/_binding/core/ndarray.h"
#include "hetu/_binding/graph/tensor.h"
#include "hetu/_binding/utils/numpy.h"
#include "hetu/_binding/utils/pybind_common.h"

namespace hetu {
namespace graph {

struct PyRunContext {
  PyObject_HEAD;
  RuntimeContext ctx;
};

extern PyTypeObject* PyRunContext_Type;

inline bool CheckPyRunContextDict(PyObject* obj) {
  if (PyDict_Check(obj)) {
    return true;
  }
  return false;
}

inline bool PyRunContext_Check(PyObject* obj) {
  return PyRunContext_Type && PyObject_TypeCheck(obj, PyRunContext_Type);
}

inline bool PyRunContext_CheckExact(PyObject* obj) {
  return PyRunContext_Type && obj->ob_type == PyRunContext_Type;
}

PyObject* PyRunContext_New();

void AddPyRunContextTypeToModule(py::module_& module);

/******************************************************
 * ArgParser Utils
 ******************************************************/

inline bool CheckPyRunContext(PyObject* obj) {
  return PyRunContext_Check(obj);
}

inline RuntimeContext RuntimeContext_FromPyObject(PyObject* obj) {
  RuntimeContext ctx;
  PyObject* key;
  PyObject* value;
  Py_ssize_t pos = 0;
  int64_t cnt = 0;
  while (PyDict_Next(obj, &pos, &key, &value)) {
    std::string k = String_FromPyUnicode(key);
    if (PyBool_Check(value)) {
      ctx.set_param(k, std::make_any<bool>(Bool_FromPyBool(value)));
      // HT_LOG_WARN << "Bool PARAM:" << k << ", VALUE:" << Bool_FromPyBool(value);
    } else if (PyLong_Check(value)) {
      ctx.set_param(k, std::make_any<int64_t>(Int64_FromPyLong(value)));
      // HT_LOG_WARN << "Int PARAM:" << k << ", VALUE:" << Int64_FromPyLong(value);
    } else if (PyFloat_Check(value)) {
      ctx.set_param(k, std::make_any<double>(Float64_FromPyFloat(value)));
    } else if (CheckPyString(value)) {
      ctx.set_param(k, std::make_any<std::string>(String_FromPyUnicode(value)));
    }
  }
  return ctx;
}

} // namespace graph
} // namespace hetu
