#include "hetu/_binding/graph/runcontext.h"
#include "hetu/_binding/constants.h"
#include "hetu/_binding/utils/pybind_common.h"
#include "hetu/_binding/utils/except.h"
#include "hetu/_binding/utils/decl_utils.h"
#include "hetu/_binding/utils/arg_parser.h"
#include <boolobject.h>

namespace hetu {
namespace graph {

PyObject* PyRunContext_New() {
  HT_PY_FUNC_BEGIN
  auto* unsafe_self = PyRunContext_Type->tp_alloc(PyRunContext_Type, 0);
  HT_RUNTIME_ERROR_IF(!unsafe_self) << "Failed to alloc PyRunContext";
  auto* self = reinterpret_cast<PyRunContext*>(unsafe_self);
  // new(&self->graph_id) RunContextId();
  return reinterpret_cast<PyObject*>(self);
  HT_PY_FUNC_END
}

void PyRunContext_dealloc(PyDevice* self) {
  // (&self->graph_id)->~RunContextId();
  Py_TYPE(self)->tp_free(self);
}

PyObject* PyRunContext_str(PyRunContext* self) {
  HT_PY_FUNC_BEGIN
  return PyUnicode_FromString("runcontext");
  HT_PY_FUNC_END
}

PyObject* PyRunContext_repr(PyRunContext* self) {
  return PyRunContext_str(self);
}

PyObject* PyRunContext_set_fp32_grad_accumulation(PyRunContext* self, PyObject* args, PyObject* kwargs) {
  HT_PY_FUNC_BEGIN
  static PyArgParser parser({
    "set_fp32_grad_accumulation(bool is_fp32=false)",
  });
  auto parsed_args = parser.parse(args, kwargs);
  if (parsed_args.signature_index() == 0) {
    self->ctx.set_fp32_grad_accumulation(parsed_args.get_bool_or_default(0));
    Py_RETURN_NONE;
  } else {
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();
  }
  HT_PY_FUNC_END
}

PyObject* PyRunContext_set_fp32_comm_reduce(PyRunContext* self, PyObject* args, PyObject* kwargs) {
  HT_PY_FUNC_BEGIN
  static PyArgParser parser({
    "set_fp32_comm_reduce(bool is_fp32=false)",
  });
  auto parsed_args = parser.parse(args, kwargs);
  if (parsed_args.signature_index() == 0) {
    self->ctx.set_fp32_comm_reduce(parsed_args.get_bool_or_default(0));
    Py_RETURN_NONE;
  } else {
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();
  }
  HT_PY_FUNC_END
}

PyObject* PyRunContext_fp32_grad_accumulation(PyRunContext* self) {
  HT_PY_FUNC_BEGIN
  return PyBool_FromLong(int64_t(std::any_cast<bool>(self->ctx.get_param("fp32_grad_accumulation"))));
  HT_PY_FUNC_END
}

PyObject* PyRunContext_fp32_comm_reduce(PyRunContext* self) {
  HT_PY_FUNC_BEGIN
  return PyBool_FromLong(int64_t(std::any_cast<bool>(self->ctx.get_param("fp32_comm_reduce"))));
  HT_PY_FUNC_END
}

// NOLINTNEXTLINE
PyMethodDef PyRunContextCtx_methods[] = { 
  {nullptr}
};

// NOLINTNEXTLINE
PyGetSetDef PyRunContext_properties[] = {
  {PY_GET_SET_DEF_NAME("fp32_grad_accumulation"), (getter) PyRunContext_fp32_grad_accumulation, nullptr, nullptr, nullptr}, 
  {PY_GET_SET_DEF_NAME("fp32_comm_reduce"), (getter) PyRunContext_fp32_comm_reduce, nullptr, nullptr, nullptr}, 
  {nullptr}
};

// NOLINTNEXTLINE
PyMethodDef PyRunContext_methods[] = {
  {"set_fp32_grad_accumulation", (PyCFunction) PyRunContext_set_fp32_grad_accumulation, METH_VARARGS | METH_KEYWORDS, nullptr },
  {nullptr}
};

// NOLINTNEXTLINE
PyTypeObject PyRunContext_Type_obj = {
  PyVarObject_HEAD_INIT(nullptr, 0) 
  "hetu.RunContext", /* tp_name */
  sizeof(PyRunContext), /* tp_basicsize */
  0, /* tp_itemsize */
  (destructor) PyRunContext_dealloc, /* tp_dealloc */
  0, /* tp_vectorcall_offset */
  nullptr, /* tp_getattr */
  nullptr, /* tp_setattr */
  nullptr, /* tp_reserved */
  (reprfunc) PyRunContext_repr, /* tp_repr */
  nullptr, /* tp_as_number */
  nullptr, /* tp_as_sequence */
  nullptr, /* tp_as_mapping */
  nullptr, /* tp_hash  */
  nullptr, /* tp_call */
  (reprfunc) PyRunContext_str, /* tp_str */
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
  PyRunContext_methods, /* tp_methods */
  nullptr, /* tp_members */
  PyRunContext_properties, /* tp_getset */
  nullptr, /* tp_base */
  nullptr, /* tp_dict */
  nullptr, /* tp_descr_get */
  nullptr, /* tp_descr_set */
  0, /* tp_dictoffset */
  nullptr, /* tp_init */
  nullptr, /* tp_alloc */
  nullptr, /* tp_new */
};
PyTypeObject* PyRunContext_Type = &PyRunContext_Type_obj;

void AddPyRunContextTypeToModule(py::module_& module) {
  HT_RUNTIME_ERROR_IF(PyType_Ready(PyRunContext_Type) < 0) 
    << "PyRunContext_Type not ready";
  Py_INCREF(PyRunContext_Type);
  HT_RUNTIME_ERROR_IF(0 != PyModule_AddObject(
      module.ptr(), "RunContext", reinterpret_cast<PyObject*>(PyRunContext_Type)))
    << "Failed to add PyRunContext_Type";
}

} // namespace graph
} // namespace hetu
