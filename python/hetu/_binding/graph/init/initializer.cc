#include "hetu/_binding/graph/init/initializer.h"
#include "hetu/_binding/utils/except.h"
#include "hetu/_binding/utils/arg_parser.h"
#include "hetu/_binding/utils/decl_utils.h"
#include "hetu/_binding/utils/function_registry.h"

namespace hetu {
namespace graph {

PyObject* PyInitializer_New(const Initializer& init) {
  HT_PY_FUNC_BEGIN
  auto* unsafe_self = PyInitializer_Type->tp_alloc(PyInitializer_Type, 0);
  HT_RUNTIME_ERROR_IF(!unsafe_self) << "Failed to alloc PyInitializer";
  auto* self = reinterpret_cast<PyInitializer*>(unsafe_self);
  self->init.reset(init.copy());
  return reinterpret_cast<PyObject*>(self);
  HT_PY_FUNC_END  
}

void PyInitializer_dealloc(PyInitializer* self) {
  self->init.reset();
  Py_TYPE(self)->tp_free(self);
}

PyObject* PyInitializer_voidified_initializer(PyTypeObject* type, PyObject* args, PyObject* kwargs) {
  HT_PY_FUNC_BEGIN
  auto* unsafe_self = PyInitializer_Type->tp_alloc(PyInitializer_Type, 0);
  HT_RUNTIME_ERROR_IF(!unsafe_self) << "Failed to alloc PyInitializer";  
  auto* self = reinterpret_cast<PyInitializer*>(unsafe_self);
  static PyArgParser parser({
    "voidified_initializer()",
  });
  auto parsed_args = parser.parse(args, kwargs);
  if (parsed_args.signature_index() == 0) {
    self->init = std::make_shared<VoidifiedInitializer>();
  } else {
    Py_TYPE(self)->tp_free(self);
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();    
  }
  return reinterpret_cast<PyObject*>(self);
  HT_PY_FUNC_END    
}

REGISTER_INITIALIZER_CLASS_METHOD(
  voidified_initializer,
  (PyCFunction) PyInitializer_voidified_initializer,
  METH_VARARGS | METH_KEYWORDS, 
  nullptr);

PyObject* PyInitializer_provided_initializer(PyTypeObject* type, PyObject* args, PyObject* kwargs) {
  HT_PY_FUNC_BEGIN
  auto* unsafe_self = PyInitializer_Type->tp_alloc(PyInitializer_Type, 0);
  HT_RUNTIME_ERROR_IF(!unsafe_self) << "Failed to alloc PyInitializer";  
  auto* self = reinterpret_cast<PyInitializer*>(unsafe_self);
  static PyArgParser parser({
    "provided_initializer(NDArray provided_data)",
    "provided_initializer(numpy.array provided_data)",
  });
  auto parsed_args = parser.parse(args, kwargs);
  if (parsed_args.signature_index() == 0) {
    self->init = std::make_shared<ProvidedInitializer>(parsed_args.get_ndarray(0));
  } else if (parsed_args.signature_index() == 1) {
    auto* array_obj = parsed_args.get_numpy_array(0);
    self->init = std::make_shared<ProvidedInitializer>(NDArrayFromNumpy(array_obj));
  } else {
    Py_TYPE(self)->tp_free(self);
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();    
  }
  return reinterpret_cast<PyObject*>(self);
  HT_PY_FUNC_END    
}

REGISTER_INITIALIZER_CLASS_METHOD(
  provided_initializer,
  (PyCFunction) PyInitializer_provided_initializer,
  METH_VARARGS | METH_KEYWORDS, 
  nullptr);  

PyObject* PyInitializer_constant_initializer(PyTypeObject* type, PyObject* args, PyObject* kwargs) {
  HT_PY_FUNC_BEGIN
  auto* unsafe_self = PyInitializer_Type->tp_alloc(PyInitializer_Type, 0);
  HT_RUNTIME_ERROR_IF(!unsafe_self) << "Failed to alloc PyInitializer";  
  auto* self = reinterpret_cast<PyInitializer*>(unsafe_self);
  static PyArgParser parser({
    "constant_initializer(double value)",
  });
  auto parsed_args = parser.parse(args, kwargs);
  if (parsed_args.signature_index() == 0) {
    self->init = std::make_shared<ConstantInitializer>(parsed_args.get_float64(0));
  } else {
    Py_TYPE(self)->tp_free(self);
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();    
  }
  return reinterpret_cast<PyObject*>(self);
  HT_PY_FUNC_END    
}

REGISTER_INITIALIZER_CLASS_METHOD(
  constant_initializer,
  (PyCFunction) PyInitializer_constant_initializer,
  METH_VARARGS | METH_KEYWORDS, 
  nullptr);

PyObject* PyInitializer_zeros_initializer(PyTypeObject* type, PyObject* args, PyObject* kwargs) {
  HT_PY_FUNC_BEGIN
  auto* unsafe_self = PyInitializer_Type->tp_alloc(PyInitializer_Type, 0);
  HT_RUNTIME_ERROR_IF(!unsafe_self) << "Failed to alloc PyInitializer";  
  auto* self = reinterpret_cast<PyInitializer*>(unsafe_self);
  static PyArgParser parser({
    "zeros_initializer()",
  });
  auto parsed_args = parser.parse(args, kwargs);
  if (parsed_args.signature_index() == 0) {
    self->init = std::make_shared<ZerosInitializer>();
  } else {
    Py_TYPE(self)->tp_free(self);
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();    
  }
  return reinterpret_cast<PyObject*>(self);
  HT_PY_FUNC_END    
}

REGISTER_INITIALIZER_CLASS_METHOD(
  zeros_initializer,
  (PyCFunction) PyInitializer_zeros_initializer,
  METH_VARARGS | METH_KEYWORDS, 
  nullptr);

PyObject* PyInitializer_ones_initializer(PyTypeObject* type, PyObject* args, PyObject* kwargs) {
  HT_PY_FUNC_BEGIN
  auto* unsafe_self = PyInitializer_Type->tp_alloc(PyInitializer_Type, 0);
  HT_RUNTIME_ERROR_IF(!unsafe_self) << "Failed to alloc PyInitializer";  
  auto* self = reinterpret_cast<PyInitializer*>(unsafe_self);
  static PyArgParser parser({
    "ones_initializer()",
  });
  auto parsed_args = parser.parse(args, kwargs);
  if (parsed_args.signature_index() == 0) {
    self->init = std::make_shared<OnesInitializer>();
  } else {
    Py_TYPE(self)->tp_free(self);
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();    
  }
  return reinterpret_cast<PyObject*>(self);
  HT_PY_FUNC_END    
}

REGISTER_INITIALIZER_CLASS_METHOD(
  ones_initializer,
  (PyCFunction) PyInitializer_ones_initializer,
  METH_VARARGS | METH_KEYWORDS, 
  nullptr);  

PyObject* PyInitializer_uniform_initializer(PyTypeObject* type, PyObject* args, PyObject* kwargs) {
  HT_PY_FUNC_BEGIN
  auto* unsafe_self = PyInitializer_Type->tp_alloc(PyInitializer_Type, 0);
  HT_RUNTIME_ERROR_IF(!unsafe_self) << "Failed to alloc PyInitializer";  
  auto* self = reinterpret_cast<PyInitializer*>(unsafe_self);
  static PyArgParser parser({
    "uniform_initializer(double lb=0, double ub=1)",
  });
  auto parsed_args = parser.parse(args, kwargs);
  if (parsed_args.signature_index() == 0) {
    self->init = std::make_shared<UniformInitializer>(parsed_args.get_float64_or_default(0),
                                                      parsed_args.get_float64_or_default(1));
  } else {
    Py_TYPE(self)->tp_free(self);
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();    
  }
  return reinterpret_cast<PyObject*>(self);
  HT_PY_FUNC_END    
}

REGISTER_INITIALIZER_CLASS_METHOD(
  uniform_initializer,
  (PyCFunction) PyInitializer_uniform_initializer,
  METH_VARARGS | METH_KEYWORDS, 
  nullptr);

PyObject* PyInitializer_normal_initializer(PyTypeObject* type, PyObject* args, PyObject* kwargs) {
  HT_PY_FUNC_BEGIN
  auto* unsafe_self = PyInitializer_Type->tp_alloc(PyInitializer_Type, 0);
  HT_RUNTIME_ERROR_IF(!unsafe_self) << "Failed to alloc PyInitializer";  
  auto* self = reinterpret_cast<PyInitializer*>(unsafe_self);
  static PyArgParser parser({
    "normal_initializer(double mean=0, double stddev=1)",
  });
  auto parsed_args = parser.parse(args, kwargs);
  if (parsed_args.signature_index() == 0) {
    self->init = std::make_shared<NormalInitializer>(parsed_args.get_float64_or_default(0),
                                                     parsed_args.get_float64_or_default(1));
  } else {
    Py_TYPE(self)->tp_free(self);
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();    
  }
  return reinterpret_cast<PyObject*>(self);
  HT_PY_FUNC_END    
}

REGISTER_INITIALIZER_CLASS_METHOD(
  normal_initializer,
  (PyCFunction) PyInitializer_normal_initializer,
  METH_VARARGS | METH_KEYWORDS, 
  nullptr);  

PyObject* PyInitializer_truncated_normal_initializer(PyTypeObject* type, PyObject* args, PyObject* kwargs) {
  HT_PY_FUNC_BEGIN
  auto* unsafe_self = PyInitializer_Type->tp_alloc(PyInitializer_Type, 0);
  HT_RUNTIME_ERROR_IF(!unsafe_self) << "Failed to alloc PyInitializer";  
  auto* self = reinterpret_cast<PyInitializer*>(unsafe_self);
  static PyArgParser parser({
    "truncated_normal_initializer(double mean=0, double stddev=1, double lb=-2, double ub=2)",
  });
  auto parsed_args = parser.parse(args, kwargs);
  if (parsed_args.signature_index() == 0) {
    self->init = std::make_shared<TruncatedNormalInitializer>(parsed_args.get_float64_or_default(0),
                                                              parsed_args.get_float64_or_default(1),
                                                              parsed_args.get_float64_or_default(2),
                                                              parsed_args.get_float64_or_default(3));
  } else {
    Py_TYPE(self)->tp_free(self);
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();    
  }
  return reinterpret_cast<PyObject*>(self);
  HT_PY_FUNC_END    
}

REGISTER_INITIALIZER_CLASS_METHOD(
  truncated_normal_initializer,
  (PyCFunction) PyInitializer_truncated_normal_initializer,
  METH_VARARGS | METH_KEYWORDS, 
  nullptr);

PyObject* PyInitializer_xavier_uniform_initializer(PyTypeObject* type, PyObject* args, PyObject* kwargs) {
  HT_PY_FUNC_BEGIN
  auto* unsafe_self = PyInitializer_Type->tp_alloc(PyInitializer_Type, 0);
  HT_RUNTIME_ERROR_IF(!unsafe_self) << "Failed to alloc PyInitializer";  
  auto* self = reinterpret_cast<PyInitializer*>(unsafe_self);
  static PyArgParser parser({
    "xavier_uniform_initializer(double gain=3.0)",
  });
  auto parsed_args = parser.parse(args, kwargs);
  if (parsed_args.signature_index() == 0) {
    self->init = std::make_shared<XavierUniformInitializer>(parsed_args.get_float64_or_default(0));
  } else {
    Py_TYPE(self)->tp_free(self);
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();    
  }
  return reinterpret_cast<PyObject*>(self);
  HT_PY_FUNC_END    
}

REGISTER_INITIALIZER_CLASS_METHOD(
  xavier_uniform_initializer,
  (PyCFunction) PyInitializer_xavier_uniform_initializer,
  METH_VARARGS | METH_KEYWORDS, 
  nullptr);

PyObject* PyInitializer_xavier_normal_initializer(PyTypeObject* type, PyObject* args, PyObject* kwargs) {
  HT_PY_FUNC_BEGIN
  auto* unsafe_self = PyInitializer_Type->tp_alloc(PyInitializer_Type, 0);
  HT_RUNTIME_ERROR_IF(!unsafe_self) << "Failed to alloc PyInitializer";  
  auto* self = reinterpret_cast<PyInitializer*>(unsafe_self);
  static PyArgParser parser({
    "xavier_normal_initializer(double gain=1.0)",
  });
  auto parsed_args = parser.parse(args, kwargs);
  if (parsed_args.signature_index() == 0) {
    self->init = std::make_shared<XavierNormalInitializer>(parsed_args.get_float64_or_default(0));
  } else {
    Py_TYPE(self)->tp_free(self);
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();    
  }
  return reinterpret_cast<PyObject*>(self);
  HT_PY_FUNC_END    
}

REGISTER_INITIALIZER_CLASS_METHOD(
  xavier_normal_initializer,
  (PyCFunction) PyInitializer_xavier_normal_initializer,
  METH_VARARGS | METH_KEYWORDS, 
  nullptr);

PyObject* PyInitializer_he_uniform_initializer(PyTypeObject* type, PyObject* args, PyObject* kwargs) {
  HT_PY_FUNC_BEGIN
  auto* unsafe_self = PyInitializer_Type->tp_alloc(PyInitializer_Type, 0);
  HT_RUNTIME_ERROR_IF(!unsafe_self) << "Failed to alloc PyInitializer";  
  auto* self = reinterpret_cast<PyInitializer*>(unsafe_self);
  static PyArgParser parser({
    "he_uniform_initializer(double gain=6.0)",
  });
  auto parsed_args = parser.parse(args, kwargs);
  if (parsed_args.signature_index() == 0) {
    self->init = std::make_shared<HeUniformInitializer>(parsed_args.get_float64_or_default(0));
  } else {
    Py_TYPE(self)->tp_free(self);
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();    
  }
  return reinterpret_cast<PyObject*>(self);
  HT_PY_FUNC_END    
}

REGISTER_INITIALIZER_CLASS_METHOD(
  he_uniform_initializer,
  (PyCFunction) PyInitializer_he_uniform_initializer,
  METH_VARARGS | METH_KEYWORDS, 
  nullptr);

PyObject* PyInitializer_he_normal_initializer(PyTypeObject* type, PyObject* args, PyObject* kwargs) {
  HT_PY_FUNC_BEGIN
  auto* unsafe_self = PyInitializer_Type->tp_alloc(PyInitializer_Type, 0);
  HT_RUNTIME_ERROR_IF(!unsafe_self) << "Failed to alloc PyInitializer";  
  auto* self = reinterpret_cast<PyInitializer*>(unsafe_self);
  static PyArgParser parser({
    "he_normal_initializer(double gain=2.0)",
  });
  auto parsed_args = parser.parse(args, kwargs);
  if (parsed_args.signature_index() == 0) {
    self->init = std::make_shared<HeNormalInitializer>(parsed_args.get_float64_or_default(0));
  } else {
    Py_TYPE(self)->tp_free(self);
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();    
  }
  return reinterpret_cast<PyObject*>(self);
  HT_PY_FUNC_END    
}

REGISTER_INITIALIZER_CLASS_METHOD(
  he_normal_initializer,
  (PyCFunction) PyInitializer_he_normal_initializer,
  METH_VARARGS | METH_KEYWORDS, 
  nullptr);

PyObject* PyInitializer_lecun_uniform_initializer(PyTypeObject* type, PyObject* args, PyObject* kwargs) {
  HT_PY_FUNC_BEGIN
  auto* unsafe_self = PyInitializer_Type->tp_alloc(PyInitializer_Type, 0);
  HT_RUNTIME_ERROR_IF(!unsafe_self) << "Failed to alloc PyInitializer";  
  auto* self = reinterpret_cast<PyInitializer*>(unsafe_self);
  static PyArgParser parser({
    "lecun_uniform_initializer(double gain=3.0)",
  });
  auto parsed_args = parser.parse(args, kwargs);
  if (parsed_args.signature_index() == 0) {
    self->init = std::make_shared<LecunUniformInitializer>(parsed_args.get_float64_or_default(0));
  } else {
    Py_TYPE(self)->tp_free(self);
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();    
  }
  return reinterpret_cast<PyObject*>(self);
  HT_PY_FUNC_END    
}

REGISTER_INITIALIZER_CLASS_METHOD(
  lecun_uniform_initializer,
  (PyCFunction) PyInitializer_lecun_uniform_initializer,
  METH_VARARGS | METH_KEYWORDS, 
  nullptr);

PyObject* PyInitializer_lecun_normal_initializer(PyTypeObject* type, PyObject* args, PyObject* kwargs) {
  HT_PY_FUNC_BEGIN
  auto* unsafe_self = PyInitializer_Type->tp_alloc(PyInitializer_Type, 0);
  HT_RUNTIME_ERROR_IF(!unsafe_self) << "Failed to alloc PyInitializer";  
  auto* self = reinterpret_cast<PyInitializer*>(unsafe_self);
  static PyArgParser parser({
    "lecun_normal_initializer(double gain=1.0)",
  });
  auto parsed_args = parser.parse(args, kwargs);
  if (parsed_args.signature_index() == 0) {
    self->init = std::make_shared<LecunNormalInitializer>(parsed_args.get_float64_or_default(0));
  } else {
    Py_TYPE(self)->tp_free(self);
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();    
  }
  return reinterpret_cast<PyObject*>(self);
  HT_PY_FUNC_END    
}

REGISTER_INITIALIZER_CLASS_METHOD(
  lecun_normal_initializer,
  (PyCFunction) PyInitializer_lecun_normal_initializer,
  METH_VARARGS | METH_KEYWORDS, 
  nullptr);

// NOLINTNEXTLINE
PyTypeObject PyInitializer_Type_obj = {
  PyVarObject_HEAD_INIT(nullptr, 0) 
  "hetu.Initializer", /* tp_name */
  sizeof(PyInitializer), /* tp_basicsize */
  0, /* tp_itemsize */
  (destructor) PyInitializer_dealloc, /* tp_dealloc */
  0, /* tp_vectorcall_offset */
  nullptr, /* tp_getattr */
  nullptr, /* tp_setattr */
  nullptr, /* tp_reserved */
  nullptr, /* tp_repr */
  nullptr, /* tp_as_number */
  nullptr, /* tp_as_sequence */
  nullptr, /* tp_as_mapping */
  nullptr, /* tp_hash  */
  nullptr, /* tp_call */
  nullptr, /* tp_str */
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
  nullptr, /* tp_getset */
  nullptr, /* tp_base */
  nullptr, /* tp_dict */
  nullptr, /* tp_descr_get */
  nullptr, /* tp_descr_set */
  0, /* tp_dictoffset */
  nullptr, /* tp_init */
  nullptr, /* tp_alloc */
  nullptr, /* tp_new */
};
PyTypeObject* PyInitializer_Type = &PyInitializer_Type_obj;

void AddPyInitializerTypeToModule(py::module_& module) {
  HT_RUNTIME_ERROR_IF(PyType_Ready(PyInitializer_Type) < 0) 
    << "PyInitializer_Type not ready";
  Py_INCREF(PyInitializer_Type);
  HT_RUNTIME_ERROR_IF(0 != PyModule_AddObject(
      module.ptr(), "Initializer", 
      reinterpret_cast<PyObject*>(PyInitializer_Type)))
    << "Failed to add PyInitializer_Type";
  static auto initializer_class_methods = hetu::graph::get_registered_initializer_class_methods();
  HT_RUNTIME_ERROR_IF(0 != PyModule_AddFunctions(
      module.ptr(), initializer_class_methods.data()))
    << "Failed to add Initializer class methods";
}
} // namespace graph
} // namespace hetu
