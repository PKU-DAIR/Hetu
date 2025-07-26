#include "hetu/_binding/graph/tensor.h"
#include "hetu/_binding/graph/sgdoptimizer.h"
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

PyObject* PySGDOptimizer_New(SGDOptimizer&& optimizer, bool return_none_if_undefined) {
  HT_PY_FUNC_BEGIN
  if (return_none_if_undefined) {
    Py_RETURN_NONE;
  } else {
    auto* unsafe_self = PySGDOptimizer_Type->tp_alloc(PySGDOptimizer_Type, 0);
    HT_RUNTIME_ERROR_IF(!unsafe_self) << "Failed to alloc PySGDOptimizer";
    auto* self = reinterpret_cast<PySGDOptimizer*>(unsafe_self);
    new(&self->optimizer) SGDOptimizer();
    self->optimizer = std::move(optimizer);
    return reinterpret_cast<PyObject*>(self);
  }
  HT_PY_FUNC_END
}

inline PyObject* PySGDOptimizer_pynew(PyTypeObject* type, PyObject* args, 
                                      PyObject* kwargs) {
  HT_PY_FUNC_BEGIN
  auto* unsafe_self = PySGDOptimizer_Type->tp_alloc(PySGDOptimizer_Type, 0);
  HT_RUNTIME_ERROR_IF(!unsafe_self) << "Failed to alloc PySGDOptimizer";
  auto* self = reinterpret_cast<PySGDOptimizer*>(unsafe_self);

  static PyArgParser parser({
    "SGDOptimizer(float lr, float momentum=0.0, bool nesterov=false)", 

    "SGDOptimizer(TensorList vars, float lr, float momentum=0.0, bool nesterov=false)", 

    "SGDOptimizer(float init_lr, float max_lr, float min_lr, int lr_warmup_steps, int lr_decay_steps, string lr_decay_style, "
                  "float momentum=0.0, bool nesterov=false)", 

    "SGDOptimizer(TensorList vars, float init_lr, float max_lr, float min_lr, int lr_warmup_steps, int lr_decay_steps, "
                  "string lr_decay_style, float momentum=0.0, bool nesterov=false)", 


  });
  auto parsed_args = parser.parse(args, kwargs);
  
  // Question: During device placement, the ndarray data will be copied 
  // if it does not match the expected dtype or device. 
  // Can we defer the copy to device placement?
  if (parsed_args.signature_index() == 0) {
    new(&self->optimizer) SGDOptimizer();
    self->optimizer = SGDOptimizer(parsed_args.get_float64_or_default(0), 
                                   parsed_args.get_float64_or_default(0), 
                                   parsed_args.get_float64_or_default(0), 
                                   0,
                                   1000,
                                   "constant",
                                   parsed_args.get_float64_or_default(1), 
                                   parsed_args.get_bool_or_default(2));
  } else if (parsed_args.signature_index() == 1) {
    new(&self->optimizer) SGDOptimizer();
    auto params = parsed_args.get_tensor_list_or_empty(0);
    self->optimizer = SGDOptimizer(params,
                                   parsed_args.get_float64_or_default(1), 
                                   parsed_args.get_float64_or_default(1), 
                                   parsed_args.get_float64_or_default(1), 
                                   0,
                                   1000,
                                   "constant",
                                   parsed_args.get_float64_or_default(2), 
                                   parsed_args.get_bool_or_default(3));
  } else if (parsed_args.signature_index() == 2) {
    new(&self->optimizer) SGDOptimizer();
    self->optimizer = SGDOptimizer(parsed_args.get_float64_or_default(0), 
                                   parsed_args.get_float64_or_default(1), 
                                   parsed_args.get_float64_or_default(2), 
                                   parsed_args.get_int64_or_default(3), 
                                   parsed_args.get_int64_or_default(4), 
                                   parsed_args.get_string_or_default(5), 
                                   parsed_args.get_float64_or_default(6), 
                                   parsed_args.get_bool_or_default(7));
  } else if (parsed_args.signature_index() == 3) {
    new(&self->optimizer) SGDOptimizer();
    auto params = parsed_args.get_tensor_list_or_empty(0);
    self->optimizer = SGDOptimizer(params,
                                   parsed_args.get_float64_or_default(1), 
                                   parsed_args.get_float64_or_default(2), 
                                   parsed_args.get_float64_or_default(3), 
                                   parsed_args.get_int64_or_default(4), 
                                   parsed_args.get_int64_or_default(5), 
                                   parsed_args.get_string_or_default(6), 
                                   parsed_args.get_float64_or_default(7), 
                                   parsed_args.get_bool_or_default(8));
  } else {
    Py_TYPE(self)->tp_free(self);
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();
  }

  return reinterpret_cast<PyObject*>(self);
  HT_PY_FUNC_END
}

// PyObject* PySGDOptimizer_make_subclass(PyObject*, PyObject* args, PyObject* kwargs) {
//   HT_PY_FUNC_BEGIN
//   static PyArgParser parser({
//     "_make_subclass(PyObject* cls, SGDOptimizer data, bool requires_grad=false)"
//   });
//   auto parsed_args = parser.parse(args, kwargs);
//   if (parsed_args.signature_index() == 0) {
//     PyObject* cls = parsed_args.get_py_obj(0);
//     HT_TYPE_ERROR_IF(!PyType_Check(cls))
//       << "Expected argument \"cls\" to be a type (got " 
//       << Py_TYPE(cls)->tp_name << ")";
//     PyTypeObject* cls_type = reinterpret_cast<PyTypeObject*>(cls);
//     HT_TYPE_ERROR_IF(!PyType_IsSubtype(cls_type, PySGDOptimizer_Type))
//       << "Type " << cls_type->tp_name << " is not derived from hetu.SGDOptimizer";

//     auto* unsafe_self = cls_type->tp_alloc(cls_type, 0);
//     HT_RUNTIME_ERROR_IF(!unsafe_self) << "Failed to alloc PySGDOptimizer";
//     // Question: Is the casting safe?
//     auto* self = reinterpret_cast<PySGDOptimizer*>(unsafe_self);
//     new(&self->tensor) SGDOptimizer();

//     self->tensor = parsed_args.get_tensor(1);
//     HT_TYPE_ERROR_IF(!self->tensor->is_variable())
//       << "Subclass of hetu.SGDOptimizer must be created from a variable. "
//       << "Please detach the tensor first.";
//     HT_NOT_IMPLEMENTED;
//     // self->tensor->set_trainable(parsed_args.get_bool_or_default(2));
    
//     return reinterpret_cast<PyObject*>(self);
//   } else {
//     HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
//     __builtin_unreachable();
//   }
//   HT_PY_FUNC_END
// }

void PySGDOptimizer_dealloc(PySGDOptimizer* self) {
  (&self->optimizer)->~SGDOptimizer();
  Py_TYPE(self)->tp_free(self);
}

PyObject* PySGDOptimizer_str(PySGDOptimizer* self) {
  HT_PY_FUNC_BEGIN
  // Compute?
  return PyUnicode_FromString("optimizer");
  HT_PY_FUNC_END
}

PyObject* PySGDOptimizer_repr(PySGDOptimizer* self) {
  return PySGDOptimizer_str(self);
}

PyObject* PySGDOptimizer_learning_rate(PySGDOptimizer* self) {
  HT_PY_FUNC_BEGIN
  return PyFloat_FromDouble(static_cast<double>(self->optimizer.learning_rate()));
  HT_PY_FUNC_END
}

PyObject* PySGDOptimizer_minimize(PySGDOptimizer* self, PyObject* args, PyObject* kwargs) {
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

// PyObject* PySGDOptimizer_zerograd(PySGDOptimizer* self) {
//   HT_PY_FUNC_BEGIN
//   self->optimizer.ZeroGrad();
//   HT_PY_FUNC_END
// }

// PyObject* PySGDOptimizer_step(PySGDOptimizer* self) {
//   HT_PY_FUNC_BEGIN
//   self->optimizer.Step();
//   HT_PY_FUNC_END
// }

// PyObject* PySGDOptimizer_makestates(PySGDOptimizer* self, PyObject* args, PyObject* kwargs) {
//   HT_PY_FUNC_BEGIN
//   static PyArgParser parser({
//     "makestates(Tensor variable, std::string state_name=\"\")"
//   });
//   auto parsed_args = parser.parse(args, kwargs);
//   if (parsed_args.signature_index() == 0) {
//     self->optimizer.MakeStates(parsed_args.get_tensor_optional(0),
//                                parsed_args.get_string_or_default(1));
//   } else {
//     HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
//     __builtin_unreachable();
//   }
//   HT_PY_FUNC_END
// }

// NOLINTNEXTLINE
PyGetSetDef PySGDOptimizer_properties[] = {
  {PY_GET_SET_DEF_NAME("learning_rate"), (getter) PySGDOptimizer_learning_rate, nullptr, nullptr, nullptr}, 
  {nullptr}
};

PyTypeObject PySGDOptimizer_Type_obj = {
  PyVarObject_HEAD_INIT(nullptr, 0) 
  "hetu.SGDOptimizer", /* tp_name */
  sizeof(PySGDOptimizer), /* tp_basicsize */
  0, /* tp_itemsize */
  (destructor) PySGDOptimizer_dealloc, /* tp_dealloc */
  0, /* tp_vectorcall_offset */
  nullptr, /* tp_getattr */
  nullptr, /* tp_setattr */
  nullptr, /* tp_reserved */
  (reprfunc) PySGDOptimizer_repr, /* tp_repr */
  nullptr, /* tp_as_number */
  nullptr, /* tp_as_sequence */
  nullptr, /* tp_as_mapping */
  nullptr, /* tp_hash  */
  nullptr, /* tp_call */
  (reprfunc) PySGDOptimizer_str, /* tp_str */
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
  PySGDOptimizer_properties, /* tp_getset */
  nullptr, /* tp_base */
  nullptr, /* tp_dict */
  nullptr, /* tp_descr_get */
  nullptr, /* tp_descr_set */
  0, /* tp_dictoffset */
  nullptr, /* tp_init */
  nullptr, /* tp_alloc */
  PySGDOptimizer_pynew, /* tp_new */
};
PyTypeObject* PySGDOptimizer_Type = &PySGDOptimizer_Type_obj;

std::vector<PyMethodDef> InitSGDOptimizerPyMethodDefs() {
  std::vector<PyMethodDef> ret = {{nullptr}};
  AddPyMethodDefs(ret, {
    {"minimize", (PyCFunction) PySGDOptimizer_minimize, METH_VARARGS | METH_KEYWORDS, nullptr }, 
    // {"makestates", (PyCFunction) PySGDOptimizer_makestates, METH_VARARGS | METH_KEYWORDS, nullptr }, 
    // {"zerograd", (PyCFunction) PySGDOptimizer_zerograd, METH_NOARGS, nullptr }, 
    // {"step", (PyCFunction) PySGDOptimizer_step, METH_NOARGS, nullptr },
    // {"_make_subclass", (PyCFunction) PySGDOptimizer_make_subclass, METH_CLASS | METH_VARARGS | METH_KEYWORDS, nullptr }, 
    {nullptr}
  });
  AddPyMethodDefs(ret, hetu::graph::get_registered_optimizer_methods());
  return ret;
}

std::vector<PyMethodDef> InitSGDOptimizerPyClassMethodDefs() {
  std::vector<PyMethodDef> ret = {{nullptr}};
  AddPyMethodDefs(ret, {
    // {"from_numpy", (PyCFunction) PySGDOptimizer_from_numpy, METH_VARARGS | METH_KEYWORDS, nullptr }, 
    {nullptr}
  });
  AddPyMethodDefs(ret, hetu::graph::get_registered_optimizer_class_methods());
  return ret;
}

void AddPySGDOptimizerTypeToModule(py::module_& module) {
  PySGDOptimizer_Type->tp_as_number = &(get_registered_optimizer_number_methods());
  static auto optimizer_methods = InitSGDOptimizerPyMethodDefs();
  PySGDOptimizer_Type->tp_methods = optimizer_methods.data();
  HT_RUNTIME_ERROR_IF(PyType_Ready(PySGDOptimizer_Type) < 0) 
    << "PySGDOptimizer_Type not ready";
  Py_INCREF(PySGDOptimizer_Type);
  HT_RUNTIME_ERROR_IF(0 != PyModule_AddObject(
      module.ptr(), "SGDOptimizer", reinterpret_cast<PyObject*>(PySGDOptimizer_Type)))
    << "Failed to add PySGDOptimizer_Type";
  
  static auto optimizer_class_methods = InitSGDOptimizerPyClassMethodDefs();
  HT_RUNTIME_ERROR_IF(0 != PyModule_AddFunctions(
      module.ptr(), optimizer_class_methods.data()))
    << "Failed to add SGDOptimizer class methods";
}

} // namespace graph
} // namespace hetu
