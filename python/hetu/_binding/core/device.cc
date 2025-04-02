#include "hetu/_binding/core/device.h"
#include "hetu/_binding/constants.h"
#include "hetu/_binding/utils/pybind_common.h"
#include "hetu/_binding/utils/except.h"
#include "hetu/_binding/utils/decl_utils.h"
#include "hetu/_binding/utils/arg_parser.h"

namespace hetu {

PyObject* PyDevice_New(const Device& device) {
  HT_PY_FUNC_BEGIN
  auto* unsafe_self = PyDevice_Type->tp_alloc(PyDevice_Type, 0);
  HT_RUNTIME_ERROR_IF(!unsafe_self) << "Failed to alloc PyDevice";  
  auto* self = reinterpret_cast<PyDevice*>(unsafe_self);
  new (&self->device) Device(device);
  return reinterpret_cast<PyObject*>(self);
  HT_PY_FUNC_END
}

PyObject* PyDevice_pynew(PyTypeObject* type, PyObject* args, 
                         PyObject* kwargs) {
  HT_PY_FUNC_BEGIN
  auto* unsafe_self = PyDevice_Type->tp_alloc(PyDevice_Type, 0);
  HT_RUNTIME_ERROR_IF(!unsafe_self) << "Failed to alloc PyDevice";  
  auto* self = reinterpret_cast<PyDevice*>(unsafe_self);
  static PyArgParser parser({
    "Device(string device, int multiplex=0)", 
    "Device(Device device)"
  });
  auto parsed_args = parser.parse(args, kwargs);
  if (parsed_args.signature_index() == 0) {
    new (&self->device) Device(
      parsed_args.get_string(0), 
      parsed_args.get_int64_or_default(1));
  } else if (parsed_args.signature_index() == 1) {
    new (&self->device) Device(parsed_args.get_device(0));
  } else {
    Py_TYPE(self)->tp_free(self);
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();
  }
  return reinterpret_cast<PyObject*>(self);
  HT_PY_FUNC_END
}

void PyDevice_dealloc(PyDevice* self) {
  (&self->device)->~Device();
  Py_TYPE(self)->tp_free(self);
}

PyObject* PyDevice_str(PyDevice* self) {
  HT_PY_FUNC_BEGIN
  return PyUnicode_FromString(std::to_string(self->device));
  HT_PY_FUNC_END
}

PyObject* PyDevice_repr(PyDevice* self) {
  return PyDevice_str(self);
}

PyObject* PyDevice_type(PyDevice* self) {
  HT_PY_FUNC_BEGIN
  return PyUnicode_FromString(DeviceType2Str(self->device.type()));
  HT_PY_FUNC_END
}

PyObject* PyDevice_index(PyDevice* self) {
  HT_PY_FUNC_BEGIN
  return PyLong_FromInteger(self->device.index());
  HT_PY_FUNC_END
}

PyObject* PyDevice_is_cpu(PyDevice* self) {
  HT_PY_FUNC_BEGIN
  Py_RETURN_BOOLEAN_COND(self->device.is_cpu());
  HT_PY_FUNC_END
}

PyObject* PyDevice_is_cuda(PyDevice* self) {
  HT_PY_FUNC_BEGIN
  Py_RETURN_BOOLEAN_COND(self->device.is_cuda());
  HT_PY_FUNC_END
}

PyObject* PyDevice_is_undetermined(PyDevice* self) {
  HT_PY_FUNC_BEGIN
  Py_RETURN_BOOLEAN_COND(self->device.is_undetermined());
  HT_PY_FUNC_END
}

PyObject* PyDevice_local(PyDevice* self) {
  HT_PY_FUNC_BEGIN
  Py_RETURN_BOOLEAN_COND(self->device.local());
  HT_PY_FUNC_END
}

PyObject* PyDevice_hostname(PyDevice* self) {
  HT_PY_FUNC_BEGIN
  return PyUnicode_FromString(self->device.hostname());
  HT_PY_FUNC_END
}

PyObject* PyDevice_multiplex(PyDevice* self) {
  HT_PY_FUNC_BEGIN
  return PyLong_FromInteger(self->device.multiplex());
  HT_PY_FUNC_END
}

PyObject* PyDevice_GetLocalHostname(PyObject*) {
  HT_PY_FUNC_BEGIN
  return PyUnicode_FromString(Device::GetLocalHostname());
  HT_PY_FUNC_END
}

PyObject* PyDevice_rich_cmp(PyObject* obj_x, PyObject* obj_y, int op) {
  HT_PY_FUNC_BEGIN
  if (!PyDevice_Check(obj_x) || !PyDevice_Check(obj_y)) {
    Py_RETURN_NOTIMPLEMENTED;
  }
  auto& x = reinterpret_cast<PyDevice*>(obj_x)->device;
  auto& y = reinterpret_cast<PyDevice*>(obj_y)->device;
  Py_RETURN_RICH_CMP_EQ_AND_LT_ONLY(x, y, op);
  HT_PY_FUNC_END
}

Py_ssize_t PyDevice_hash(PyDevice* self) {
  HT_PY_FUNC_BEGIN
  return static_cast<Py_ssize_t>(std::hash<Device>()(self->device));
  HT_PY_FUNC_RETURN(-1)
}

PyObject* PyDevice_reduce(PyDevice* self) {
  HT_PY_FUNC_BEGIN
  const auto& device = self->device;
  auto* ret = PyTuple_New(2);
  HT_RUNTIME_ERROR_IF(!ret) << "Failed to alloc tuple";
  
  py::object hetu_module = py::module::import("hetu");
  py::object hetu_device = hetu_module.attr("device");
  PyTuple_SET_ITEM(ret, 0, hetu_device.release().ptr());

  int num_args = (device.multiplex() != 0) ? 2 : 1;
  auto* args = PyTuple_New(num_args);
  HT_RUNTIME_ERROR_IF(!args) << "Failed to alloc tuple";
  auto* device_str_obj = PyUnicode_FromString(device.compat_string());
  PyTuple_SET_ITEM(args, 0, device_str_obj);
  if (num_args >= 2) {
    auto* multiplex_obj = PyLong_FromInteger(device.multiplex());
    PyTuple_SET_ITEM(args, 1, multiplex_obj);
  }
  PyTuple_SET_ITEM(ret, 1, args);

  return ret;
  HT_PY_FUNC_END
}

// NOLINTNEXTLINE
PyGetSetDef PyDevice_properties[] = {
  {PY_GET_SET_DEF_NAME("type"), (getter) PyDevice_type, nullptr, nullptr, nullptr}, 
  {PY_GET_SET_DEF_NAME("index"), (getter) PyDevice_index, nullptr, nullptr, nullptr}, 
  {PY_GET_SET_DEF_NAME("is_cpu"), (getter) PyDevice_is_cpu, nullptr, nullptr, nullptr}, 
  {PY_GET_SET_DEF_NAME("is_cuda"), (getter) PyDevice_is_cuda, nullptr, nullptr, nullptr}, 
  {PY_GET_SET_DEF_NAME("is_undetermined"), (getter) PyDevice_is_undetermined, nullptr, nullptr, nullptr}, 
  {PY_GET_SET_DEF_NAME("local"), (getter) PyDevice_local, nullptr, nullptr, nullptr}, 
  {PY_GET_SET_DEF_NAME("hostname"), (getter) PyDevice_hostname, nullptr, nullptr, nullptr}, 
  {PY_GET_SET_DEF_NAME("multiplex"), (getter) PyDevice_multiplex, nullptr, nullptr, nullptr}, 
  {nullptr}
};

// NOLINTNEXTLINE
PyMethodDef PyDevice_methods[] = {
  {"get_local_hostname", (PyCFunction) PyDevice_GetLocalHostname, METH_CLASS | METH_NOARGS, nullptr }, 
  {"__reduce__", (PyCFunction) PyDevice_reduce, METH_NOARGS, nullptr}, 
  {nullptr}
};

// NOLINTNEXTLINE
PyTypeObject PyDevice_Type_obj = {
  PyVarObject_HEAD_INIT(nullptr, 0) 
  "hetu.device", /* tp_name */
  sizeof(PyDevice), /* tp_basicsize */
  0, /* tp_itemsize */
  (destructor) PyDevice_dealloc, /* tp_dealloc */
  0, /* tp_vectorcall_offset */
  nullptr, /* tp_getattr */
  nullptr, /* tp_setattr */
  nullptr, /* tp_reserved */
  (reprfunc) PyDevice_repr, /* tp_repr */
  nullptr, /* tp_as_number */
  nullptr, /* tp_as_sequence */
  nullptr, /* tp_as_mapping */
  (hashfunc) PyDevice_hash, /* tp_hash  */
  nullptr, /* tp_call */
  (reprfunc) PyDevice_str, /* tp_str */
  nullptr, /* tp_getattro */
  nullptr, /* tp_setattro */
  nullptr, /* tp_as_buffer */
  Py_TPFLAGS_DEFAULT, /* tp_flags */
  nullptr, /* tp_doc */
  nullptr, /* tp_traverse */
  nullptr, /* tp_clear */
  (richcmpfunc) PyDevice_rich_cmp, /* tp_richcompare */
  0, /* tp_weaklistoffset */
  nullptr, /* tp_iter */
  nullptr, /* tp_iternext */
  PyDevice_methods, /* tp_methods */
  nullptr, /* tp_members */
  PyDevice_properties, /* tp_getset */
  nullptr, /* tp_base */
  nullptr, /* tp_dict */
  nullptr, /* tp_descr_get */
  nullptr, /* tp_descr_set */
  0, /* tp_dictoffset */
  nullptr, /* tp_init */
  nullptr, /* tp_alloc */
  PyDevice_pynew, /* tp_new */
};
PyTypeObject* PyDevice_Type = &PyDevice_Type_obj;

void AddPyDeviceTypeToModule(py::module_& module) {
  HT_RUNTIME_ERROR_IF(PyType_Ready(PyDevice_Type) < 0) 
    << "PyDevice_Type not ready";
  Py_INCREF(PyDevice_Type);
  HT_RUNTIME_ERROR_IF(0 != PyModule_AddObject(
      module.ptr(), "device", reinterpret_cast<PyObject*>(PyDevice_Type)))
    << "Failed to add PyDevice_Type";
}

PyObject* PyDeviceGroup_New(const DeviceGroup& device_group) {
  HT_PY_FUNC_BEGIN
  auto* unsafe_self = PyDeviceGroup_Type->tp_alloc(PyDeviceGroup_Type, 0);
  HT_RUNTIME_ERROR_IF(!unsafe_self) << "Failed to alloc PyDeviceGroup";  
  auto* self = reinterpret_cast<PyDeviceGroup*>(unsafe_self);
  new (&self->device_group) DeviceGroup(device_group);
  return reinterpret_cast<PyObject*>(self);
  HT_PY_FUNC_END
}

PyObject* PyDeviceGroupList_New(const DeviceGroupList& dg_list) {
  HT_PY_FUNC_BEGIN
  PyObject* ret = PyList_New(dg_list.size());
  HT_RUNTIME_ERROR_IF(!ret) << "Failed to alloc list";
  for (size_t i = 0; i < dg_list.size(); i++) {
    auto* dg_obj = PyDeviceGroup_New(dg_list[i]);
    PyList_SET_ITEM(ret, i, dg_obj);
  }
  return ret;
  HT_PY_FUNC_END
}

PyObject* PyDeviceGroupHierarchy_New(const DeviceGroupHierarchy& dg_hierarchy) {
  HT_PY_FUNC_BEGIN
  PyObject* ret = PyList_New(dg_hierarchy.size());
  HT_RUNTIME_ERROR_IF(!ret) << "Failed to alloc dict";
  for (size_t i = 0; i < dg_hierarchy.size(); i++) {
    auto* dg_list_obj = PyDeviceGroupList_New(dg_hierarchy.raw_data()[i].raw_data());
    PyList_SET_ITEM(ret, i, dg_list_obj);
  }
  return ret;
  HT_PY_FUNC_END
}

PyObject* PyDeviceGroup_pynew(PyTypeObject* type, PyObject* args, 
                              PyObject* kwargs) {
  HT_PY_FUNC_BEGIN
  auto* unsafe_self = PyDeviceGroup_Type->tp_alloc(PyDeviceGroup_Type, 0);
  HT_RUNTIME_ERROR_IF(!unsafe_self) << "Failed to alloc PyDeviceGroup";  
  auto* self = reinterpret_cast<PyDeviceGroup*>(unsafe_self);
  static PyArgParser parser({
    "DeviceGroup(List[string] device)", 
    "DeviceGroup(DeviceGroup device_group)"
  });
  auto parsed_args = parser.parse(args, kwargs);
  if (parsed_args.signature_index() == 0) {
    new (&self->device_group) DeviceGroup(parsed_args.get_string_list(0));
  } else if (parsed_args.signature_index() == 1) {
    new (&self->device_group) DeviceGroup(parsed_args.get_device_group(0));
  } else {
    Py_TYPE(self)->tp_free(self);
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();
  }
  return reinterpret_cast<PyObject*>(self);
  HT_PY_FUNC_END
}

void PyDeviceGroup_dealloc(PyDeviceGroup* self) {
  (&self->device_group)->~DeviceGroup();
  Py_TYPE(self)->tp_free(self);
}

PyObject* PyDeviceGroup_str(PyDeviceGroup* self) {
  HT_PY_FUNC_BEGIN
  return PyUnicode_FromString(std::to_string(self->device_group));
  HT_PY_FUNC_END
}

PyObject* PyDeviceGroup_repr(PyDeviceGroup* self) {
  return PyDeviceGroup_str(self);
}

PyObject* PyDeviceGroup_empty(PyDeviceGroup* self) {
  HT_PY_FUNC_BEGIN
  Py_RETURN_BOOLEAN_COND(self->device_group.empty());
  HT_PY_FUNC_END
}

PyObject* PyDeviceGroup_num_devices(PyDeviceGroup* self) {
  HT_PY_FUNC_BEGIN
  return PyLong_FromInteger(self->device_group.num_devices());
  HT_PY_FUNC_END
}

PyObject* PyDeviceGroup_device_index(PyDeviceGroup* self) {
  HT_PY_FUNC_BEGIN
  const auto& device_group = self->device_group;
  const auto& devices = device_group.devices();
  
  auto* py_devices = PyList_New(devices.size());
  HT_RUNTIME_ERROR_IF(!py_devices) << "Failed to alloc list";
  for (size_t i = 0; i < devices.size(); i++) {
    auto* device_idx = PyLong_FromInteger(devices.at(i).index());
    PyList_SET_ITEM(py_devices, i, device_idx);
  }

  return py_devices;
  HT_PY_FUNC_END
}

PyObject* PyDeviceGroup_contains(PyDeviceGroup* self, PyObject* args) {
  HT_PY_FUNC_BEGIN
  static PyArgParser parser({
    "contains(Device device)", 
    "contains(string device, int multiplex=0)"
  });
  auto parsed_args = parser.parse(args, nullptr);
  if (parsed_args.signature_index() == 0) {
    Device device = parsed_args.get_device(0);
    Py_RETURN_BOOLEAN_COND(self->device_group.contains(device));
  } else if (parsed_args.signature_index() == 1) {
    Py_RETURN_BOOLEAN_COND(self->device_group.contains(Device(
      parsed_args.get_string(0), 
      parsed_args.get_int64_or_default(1))));
  } else {
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();
  }
  HT_PY_FUNC_END
}

PyObject* PyDeviceGroup_get(PyDeviceGroup* self, PyObject* args) {
  HT_PY_FUNC_BEGIN
  static PyArgParser parser({"get(int index)"});
  auto parsed_args = parser.parse(args, nullptr);
  if (parsed_args.signature_index() == 0) {
    auto index = parsed_args.get_int64(0);
    return PyDevice_New(self->device_group.get(index));
  } else {
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();
  }
  HT_PY_FUNC_END
}

PyObject* PyDeviceGroup_get_index(PyDeviceGroup* self, PyObject* args) {
  HT_PY_FUNC_BEGIN
  static PyArgParser parser({"get_index(Device device)"});
  auto parsed_args = parser.parse(args, nullptr);
  if (parsed_args.signature_index() == 0) {
    Device device = parsed_args.get_device(0);
    return PyLong_FromInteger(self->device_group.get_index(device));
  } else {
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();
  }
  HT_PY_FUNC_END
}

PyObject* PyDeviceGroup_rich_cmp(PyObject* obj_x, PyObject* obj_y, int op) {
  HT_PY_FUNC_BEGIN
  if (!PyDeviceGroup_Check(obj_x) || !PyDeviceGroup_Check(obj_y)) {
    Py_RETURN_NOTIMPLEMENTED;
  }
  auto& x = reinterpret_cast<PyDeviceGroup*>(obj_x)->device_group;
  auto& y = reinterpret_cast<PyDeviceGroup*>(obj_y)->device_group;
  Py_RETURN_RICH_CMP_EQ_AND_LT_ONLY(x, y, op);
  HT_PY_FUNC_END
}

Py_ssize_t PyDeviceGroup_hash(PyDeviceGroup* self) {
  HT_PY_FUNC_BEGIN
  return static_cast<Py_ssize_t>(std::hash<DeviceGroup>()(self->device_group));
  HT_PY_FUNC_RETURN(-1)
}

PyObject* PyDeviceGroup_reduce(PyDeviceGroup* self) {
  HT_PY_FUNC_BEGIN
  const auto& device_group = self->device_group;
  const auto& devices = device_group.devices();
  auto* ret = PyTuple_New(2);
  HT_RUNTIME_ERROR_IF(!ret) << "Failed to alloc tuple";
  
  py::object hetu_module = py::module::import("hetu");
  py::object hetu_device_group = hetu_module.attr("DeviceGroup");
  PyTuple_SET_ITEM(ret, 0, hetu_device_group.release().ptr());

  auto* args = PyTuple_New(1);
  HT_RUNTIME_ERROR_IF(!args) << "Failed to alloc tuple";
  auto* py_devices = PyList_New(devices.size());
  HT_RUNTIME_ERROR_IF(!py_devices) << "Failed to alloc list";
  for (size_t i = 0; i < devices.size(); i++) {
    auto* device_str_obj = PyUnicode_FromString(devices.at(i).compat_string());
    PyList_SET_ITEM(py_devices, i, device_str_obj);
  }
  PyTuple_SET_ITEM(args, 0, py_devices);
  PyTuple_SET_ITEM(ret, 1, args);

  return ret;
  HT_PY_FUNC_END
}

// NOLINTNEXTLINE
PyGetSetDef PyDeviceGroup_properties[] = {
  {PY_GET_SET_DEF_NAME("empty"), (getter) PyDeviceGroup_empty, nullptr, nullptr, nullptr}, 
  {PY_GET_SET_DEF_NAME("num_devices"), (getter) PyDeviceGroup_num_devices, nullptr, nullptr, nullptr}, 
  {PY_GET_SET_DEF_NAME("device_index"), (getter) PyDeviceGroup_device_index, nullptr, nullptr, nullptr},   
  {nullptr}
};

// NOLINTNEXTLINE
PyMethodDef PyDeviceGroup_methods[] = {
  {"contains", (PyCFunction) PyDeviceGroup_contains, METH_VARARGS, nullptr }, 
  {"get", (PyCFunction) PyDeviceGroup_get, METH_VARARGS, nullptr }, 
  {"get_index", (PyCFunction) PyDeviceGroup_get_index, METH_VARARGS, nullptr }, 
  {"__reduce__", (PyCFunction) PyDeviceGroup_reduce, METH_NOARGS, nullptr}, 
  {nullptr}
};

PyTypeObject PyDeviceGroup_Type_obj = {
  PyVarObject_HEAD_INIT(nullptr, 0) 
  "hetu.DeviceGroup", /* tp_name */
  sizeof(PyDeviceGroup), /* tp_basicsize */
  0, /* tp_itemsize */
  (destructor) PyDeviceGroup_dealloc, /* tp_dealloc */
  0, /* tp_vectorcall_offset */
  nullptr, /* tp_getattr */
  nullptr, /* tp_setattr */
  nullptr, /* tp_reserved */
  (reprfunc) PyDeviceGroup_repr, /* tp_repr */
  nullptr, /* tp_as_number */
  nullptr, /* tp_as_sequence */
  nullptr, /* tp_as_mapping */
  (hashfunc) PyDeviceGroup_hash, /* tp_hash  */
  nullptr, /* tp_call */
  (reprfunc) PyDeviceGroup_str, /* tp_str */
  nullptr, /* tp_getattro */
  nullptr, /* tp_setattro */
  nullptr, /* tp_as_buffer */
  Py_TPFLAGS_DEFAULT, /* tp_flags */
  nullptr, /* tp_doc */
  nullptr, /* tp_traverse */
  nullptr, /* tp_clear */
  (richcmpfunc) PyDeviceGroup_rich_cmp, /* tp_richcompare */
  0, /* tp_weaklistoffset */
  nullptr, /* tp_iter */
  nullptr, /* tp_iternext */
  PyDeviceGroup_methods, /* tp_methods */
  nullptr, /* tp_members */
  PyDeviceGroup_properties, /* tp_getset */
  nullptr, /* tp_base */
  nullptr, /* tp_dict */
  nullptr, /* tp_descr_get */
  nullptr, /* tp_descr_set */
  0, /* tp_dictoffset */
  nullptr, /* tp_init */
  nullptr, /* tp_alloc */
  PyDeviceGroup_pynew, /* tp_new */
};
PyTypeObject* PyDeviceGroup_Type = &PyDeviceGroup_Type_obj;

void AddPyDeviceGroupTypeToModule(py::module_& module) {
  HT_RUNTIME_ERROR_IF(PyType_Ready(PyDeviceGroup_Type) < 0) 
    << "PyDeviceGroup_Type not ready";
  Py_INCREF(PyDeviceGroup_Type);
  HT_RUNTIME_ERROR_IF(0 != PyModule_AddObject(
      module.ptr(), "DeviceGroup", 
      reinterpret_cast<PyObject*>(PyDeviceGroup_Type)))
    << "Failed to add PyDeviceGroup_Type";
}

ContextManager<Device>& get_eager_device_ctx() {
  static ContextManager<Device> eager_device_ctx;
  return eager_device_ctx;
}

ContextManager<hetu::graph::DeviceGroupHierarchy>& get_dg_hierarchy_ctx() {
  static ContextManager<hetu::graph::DeviceGroupHierarchy> dg_hierarchy_ctx;
  return dg_hierarchy_ctx;
}

} // namespace hetu
