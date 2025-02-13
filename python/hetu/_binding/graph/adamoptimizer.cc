#include "hetu/_binding/graph/tensor.h"
#include "hetu/_binding/graph/adamoptimizer.h"
#include "hetu/_binding/graph/tensor_ctor.h"
#include "hetu/_binding/graph/graph.h"
#include "hetu/_binding/core/ndarray.h"
#include "hetu/_binding/constants.h"
#include "hetu/_binding/utils/function_registry.h"
#include "hetu/_binding/utils/pybind_common.h"
#include "hetu/_binding/utils/python_primitives.h"
#include "hetu/_binding/utils/except.h"
#include "hetu/_binding/utils/decl_utils.h"
#include "hetu/_binding/utils/arg_parser.h"
#include "hetu/graph/ops/variable.h"

namespace hetu {
namespace graph {

PyObject* PyAdamOptimizer_New(AdamOptimizer&& optimizer, bool return_none_if_undefined) {
  HT_PY_FUNC_BEGIN
  if (return_none_if_undefined) {
    Py_RETURN_NONE;
  } else {
    auto* unsafe_self = PyAdamOptimizer_Type->tp_alloc(PyAdamOptimizer_Type, 0);
    HT_RUNTIME_ERROR_IF(!unsafe_self) << "Failed to alloc PyAdamOptimizer";
    auto* self = reinterpret_cast<PyAdamOptimizer*>(unsafe_self);
    new(&self->optimizer) AdamOptimizer();
    self->optimizer = std::move(optimizer);
    return reinterpret_cast<PyObject*>(self);
  }
  HT_PY_FUNC_END
}

inline PyObject* PyAdamOptimizer_pynew(PyTypeObject* type, PyObject* args, 
                                      PyObject* kwargs) {
  HT_PY_FUNC_BEGIN
  auto* unsafe_self = PyAdamOptimizer_Type->tp_alloc(PyAdamOptimizer_Type, 0);
  HT_RUNTIME_ERROR_IF(!unsafe_self) << "Failed to alloc PyAdamOptimizer";
  auto* self = reinterpret_cast<PyAdamOptimizer*>(unsafe_self);

  static PyArgParser parser({
    "AdamOptimizer(float lr, float beta1=0.9, float beta2=0.999, float eps=1e-8)",

    "AdamOptimizer(TensorList vars, float lr, float beta1=0.9, float beta2=0.999, float eps=1e-8)",

    "AdamOptimizer(float init_lr, float max_lr, float min_lr, int lr_warmup_steps, int lr_decay_steps, string lr_decay_style, "
                  "float start_wd=0, float end_wd=0, int wd_incr_steps=-1, string wd_incr_style=\"constant\", "
                  "float beta1=0.9, float beta2=0.999, float eps=1e-8)",

    "AdamOptimizer(TensorList vars, float init_lr, float max_lr, float min_lr, int lr_warmup_steps, int lr_decay_steps, string lr_decay_style, "
                  "float start_wd=0, float end_wd=0, int wd_incr_steps=-1, string wd_incr_style=\"constant\", "
                  "float beta1=0.9, float beta2=0.999, float eps=1e-8)",

  });
  auto parsed_args = parser.parse(args, kwargs);


  // Question: During device placement, the ndarray data will be copied 
  // if it does not match the expected dtype or device. 
  // Can we defer the copy to device placement?
  if (parsed_args.signature_index() == 0) {
    new(&self->optimizer) AdamOptimizer();
    self->optimizer = AdamOptimizer(parsed_args.get_float64_or_default(0),
                                    parsed_args.get_float64_or_default(0),
                                    parsed_args.get_float64_or_default(0),
                                    0,
                                    1000,
                                    "constant",
                                    0,
                                    0,
                                    -1,
                                    "constant",
                                    parsed_args.get_float64_or_default(1),
                                    parsed_args.get_float64_or_default(2),
                                    parsed_args.get_float64_or_default(3)
                                    );
  } else if (parsed_args.signature_index() == 1) {
    new(&self->optimizer) AdamOptimizer();
    auto params = parsed_args.get_tensor_list_or_empty(0);
    self->optimizer = AdamOptimizer(params,
                                    parsed_args.get_float64_or_default(1),
                                    parsed_args.get_float64_or_default(1),
                                    parsed_args.get_float64_or_default(1),
                                    0,
                                    1000,
                                    "constant",
                                    0,
                                    0,
                                    -1,
                                    "constant",
                                    parsed_args.get_float64_or_default(2),
                                    parsed_args.get_float64_or_default(3),
                                    parsed_args.get_float64_or_default(4)
                                    );
  } else if (parsed_args.signature_index() == 2) {
    new(&self->optimizer) AdamOptimizer();
    self->optimizer = AdamOptimizer(parsed_args.get_float64_or_default(0), 
                                    parsed_args.get_float64_or_default(1), 
                                    parsed_args.get_float64_or_default(2), 
                                    parsed_args.get_int64_or_default(3),
                                    parsed_args.get_int64_or_default(4),
                                    parsed_args.get_string_or_default(5),
                                    parsed_args.get_float64_or_default(6),
                                    parsed_args.get_float64_or_default(7), 
                                    parsed_args.get_int64_or_default(8), 
                                    parsed_args.get_string_or_default(9),
                                    parsed_args.get_float64_or_default(10),
                                    parsed_args.get_float64_or_default(11),        
                                    parsed_args.get_float64_or_default(12)
                                    );
  } else if (parsed_args.signature_index() == 3) {
    new(&self->optimizer) AdamOptimizer();
    auto params = parsed_args.get_tensor_list_or_empty(0);
    self->optimizer = AdamOptimizer(params,
                                    parsed_args.get_float64_or_default(1), 
                                    parsed_args.get_float64_or_default(2), 
                                    parsed_args.get_float64_or_default(3), 
                                    parsed_args.get_int64_or_default(4),
                                    parsed_args.get_int64_or_default(5),
                                    parsed_args.get_string_or_default(6),
                                    parsed_args.get_float64_or_default(7),
                                    parsed_args.get_float64_or_default(8), 
                                    parsed_args.get_int64_or_default(9), 
                                    parsed_args.get_string_or_default(10),
                                    parsed_args.get_float64_or_default(11),
                                    parsed_args.get_float64_or_default(12),        
                                    parsed_args.get_float64_or_default(13)
                                    );
  } else {
    Py_TYPE(self)->tp_free(self);
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();
  }

  return reinterpret_cast<PyObject*>(self);
  HT_PY_FUNC_END
}

void PyAdamOptimizer_dealloc(PyAdamOptimizer* self) {
  (&self->optimizer)->~AdamOptimizer();
  Py_TYPE(self)->tp_free(self);
}

PyObject* PyAdamOptimizer_str(PyAdamOptimizer* self) {
  HT_PY_FUNC_BEGIN
  // Compute?
  return PyUnicode_FromString("optimizer");
  HT_PY_FUNC_END
}

PyObject* PyAdamOptimizer_repr(PyAdamOptimizer* self) {
  return PyAdamOptimizer_str(self);
}

// PyObject* PyAdamOptimizer_learning_rate(PyAdamOptimizer* self) {
//   HT_PY_FUNC_BEGIN
//   return PyFloat_FromDouble(static_cast<double>(self->optimizer.learning_rate()));
//   HT_PY_FUNC_END
// }

PyObject* PyAdamOptimizer_minimize(PyAdamOptimizer* self, PyObject* args, PyObject* kwargs) {
  HT_PY_FUNC_BEGIN
  static PyArgParser parser({
    "minimize(Tensor loss, TensorList var_list=None, Tensor grad_loss=None, std::string name=\"\")"
  });
  auto parsed_args = parser.parse(args, kwargs);
  if (parsed_args.signature_index() == 0) {
    return PyTensor_New(self->optimizer.Minimize(parsed_args.get_tensor_optional(0),
                         parsed_args.get_tensor_list_or_empty(1),
                         parsed_args.get_tensor_optional(2),
                         parsed_args.get_string_or_default(3)));
  } else {
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();
  }
  HT_PY_FUNC_END
}

PyObject* PyAdamOptimizer_get_states(PyAdamOptimizer* self, PyObject* args, PyObject* kwargs) {
  HT_PY_FUNC_BEGIN
  static PyArgParser parser({
    "get_states(Tensor var)"
  });
  auto parsed_args = parser.parse(args, kwargs);
  if (parsed_args.signature_index() == 0) {
    return PyDict_FromStateDict(self->optimizer.GetStates(parsed_args.get_tensor_optional(0)));
  } else {
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();
  }
  HT_PY_FUNC_END
}

PyObject* PyAdamOptimizer_set_states(PyAdamOptimizer* self, PyObject* args, PyObject* kwargs) {
  HT_PY_FUNC_BEGIN
  static PyArgParser parser({
    "set_states(Tensor var, OpName state_name, numpy.array value)"
  });
  auto parsed_args = parser.parse(args, kwargs);
  if (parsed_args.signature_index() == 0) {
    self->optimizer.SetStates(parsed_args.get_tensor_optional(0), 
        parsed_args.get_string(1), 
        NDArrayFromNumpy(parsed_args.get_numpy_array(2)));
    Py_RETURN_NONE;
  } else {
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();
  }
  HT_PY_FUNC_END
}


// NOLINTNEXTLINE
PyGetSetDef PyAdamOptimizer_properties[] = {
  // {PY_GET_SET_DEF_NAME("learning_rate"), (getter) PyAdamOptimizer_learning_rate, nullptr, nullptr, nullptr}, 
  {nullptr}
};

PyTypeObject PyAdamOptimizer_Type_obj = {
  PyVarObject_HEAD_INIT(nullptr, 0) 
  "hetu.AdamOptimizer", /* tp_name */
  sizeof(PyAdamOptimizer), /* tp_basicsize */
  0, /* tp_itemsize */
  (destructor) PyAdamOptimizer_dealloc, /* tp_dealloc */
  0, /* tp_vectorcall_offset */
  nullptr, /* tp_getattr */
  nullptr, /* tp_setattr */
  nullptr, /* tp_reserved */
  (reprfunc) PyAdamOptimizer_repr, /* tp_repr */
  nullptr, /* tp_as_number */
  nullptr, /* tp_as_sequence */
  nullptr, /* tp_as_mapping */
  nullptr, /* tp_hash  */
  nullptr, /* tp_call */
  (reprfunc) PyAdamOptimizer_str, /* tp_str */
  nullptr, /* tp_getattro */
  nullptr, /* tp_setattro */
  nullptr, /* tp_as_buffer */
  Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /* tp_flags */
  nullptr, /* tp_doc */
  nullptr, /* tp_traverse */
  nullptr, /* tp_clear */
  nullptr, /* tp_richcompare */
  0, /* tp_weaklistoffset */
  nullptr, /* tp_iter */
  nullptr, /* tp_iternext */
  nullptr, /* tp_methods */
  nullptr, /* tp_members */
  PyAdamOptimizer_properties, /* tp_getset */
  nullptr, /* tp_base */
  nullptr, /* tp_dict */
  nullptr, /* tp_descr_get */
  nullptr, /* tp_descr_set */
  0, /* tp_dictoffset */
  nullptr, /* tp_init */
  nullptr, /* tp_alloc */
  PyAdamOptimizer_pynew, /* tp_new */
};
PyTypeObject* PyAdamOptimizer_Type = &PyAdamOptimizer_Type_obj;

std::vector<PyMethodDef> InitAdamOptimizerPyMethodDefs() {
  std::vector<PyMethodDef> ret = {{nullptr}};
  AddPyMethodDefs(ret, {
    {"minimize", (PyCFunction) PyAdamOptimizer_minimize, METH_VARARGS | METH_KEYWORDS, nullptr }, 
    {"get_states", (PyCFunction) PyAdamOptimizer_get_states, METH_VARARGS | METH_KEYWORDS, nullptr }, 
    {"set_states", (PyCFunction) PyAdamOptimizer_set_states, METH_VARARGS | METH_KEYWORDS, nullptr }, 
    // {"makestates", (PyCFunction) PyAdamOptimizer_makestates, METH_VARARGS | METH_KEYWORDS, nullptr }, 
    // {"zerograd", (PyCFunction) PyAdamOptimizer_zerograd, METH_NOARGS, nullptr }, 
    // {"step", (PyCFunction) PyAdamOptimizer_step, METH_NOARGS, nullptr },
    // {"_make_subclass", (PyCFunction) PyAdamOptimizer_make_subclass, METH_CLASS | METH_VARARGS | METH_KEYWORDS, nullptr }, 
    {nullptr}
  });
  AddPyMethodDefs(ret, hetu::graph::get_registered_optimizer_methods());
  return ret;
}

std::vector<PyMethodDef> InitAdamOptimizerPyClassMethodDefs() {
  std::vector<PyMethodDef> ret = {{nullptr}};
  AddPyMethodDefs(ret, {
    // {"from_numpy", (PyCFunction) PyAdamOptimizer_from_numpy, METH_VARARGS | METH_KEYWORDS, nullptr }, 
    {nullptr}
  });
  AddPyMethodDefs(ret, hetu::graph::get_registered_optimizer_class_methods());
  return ret;
}

void AddPyAdamOptimizerTypeToModule(py::module_& module) {
  PyAdamOptimizer_Type->tp_as_number = &(get_registered_optimizer_number_methods());
  static auto optimizer_methods = InitAdamOptimizerPyMethodDefs();
  PyAdamOptimizer_Type->tp_methods = optimizer_methods.data();
  HT_RUNTIME_ERROR_IF(PyType_Ready(PyAdamOptimizer_Type) < 0) 
    << "PyAdamOptimizer_Type not ready";
  Py_INCREF(PyAdamOptimizer_Type);
  HT_RUNTIME_ERROR_IF(0 != PyModule_AddObject(
      module.ptr(), "AdamOptimizer", reinterpret_cast<PyObject*>(PyAdamOptimizer_Type)))
    << "Failed to add PyAdamOptimizer_Type";
  
  static auto optimizer_class_methods = InitAdamOptimizerPyClassMethodDefs();
  HT_RUNTIME_ERROR_IF(0 != PyModule_AddFunctions(
      module.ptr(), optimizer_class_methods.data()))
    << "Failed to add AdamOptimizer class methods";
}

} // namespace graph
} // namespace hetu
