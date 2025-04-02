#pragma once

#include <Python.h>
#include "hetu/core/device.h"
#include "hetu/graph/distributed_states.h"
#include "hetu/_binding/utils/pybind_common.h"
#include "hetu/_binding/utils/python_primitives.h"
#include "hetu/_binding/utils/context_manager.h"

namespace hetu {

struct PyDevice {
  PyObject_HEAD;
  Device device;
};

extern PyTypeObject* PyDevice_Type;

inline bool PyDevice_Check(PyObject* obj) {
  return PyDevice_Type && PyObject_TypeCheck(obj, PyDevice_Type);
}

inline bool PyDevice_CheckExact(PyObject* obj) {
  return PyDevice_Type && obj->ob_type == PyDevice_Type;
}

PyObject* PyDevice_New(const Device& device);

void AddPyDeviceTypeToModule(py::module_& module);

struct PyDeviceGroup {
  PyObject_HEAD;
  DeviceGroup device_group;
};

extern PyTypeObject* PyDeviceGroup_Type;

inline bool PyDeviceGroup_Check(PyObject* obj) {
  return PyDeviceGroup_Type && PyObject_TypeCheck(obj, PyDeviceGroup_Type);
}

inline bool PyDeviceGroup_CheckExact(PyObject* obj) {
  return PyDeviceGroup_Type && obj->ob_type == PyDeviceGroup_Type;
}

using DeviceGroupHierarchy = graph::DeviceGroupHierarchy;

PyObject* PyDeviceGroup_New(const DeviceGroup& device_group);

PyObject* PyDeviceGroupList_New(const DeviceGroupList& dg_list);

PyObject* PyDeviceGroupHierarchy_New(const DeviceGroupHierarchy& dg_hierarchy);

void AddPyDeviceGroupTypeToModule(py::module_& module);

/******************************************************
 * ArgParser Utils
 ******************************************************/

inline bool CheckPyDevice(PyObject* obj) {
  // string can be cast to device
  return PyDevice_Check(obj) || CheckPyString(obj);
}

inline Device Device_FromPyObject(PyObject* obj) {
  if (PyDevice_Check(obj)) {
    return reinterpret_cast<PyDevice*>(obj)->device;
  } else {
    return Device(String_FromPyUnicode(obj));
  }
}

inline bool CheckPyDeviceList(PyObject* obj) {
  bool is_tuple = PyTuple_Check(obj);
  if (is_tuple || PyList_Check(obj)) {
    size_t size = is_tuple ? PyTuple_GET_SIZE(obj) : PyList_GET_SIZE(obj);
    if (size > 0) {
      // only check for the first item for efficiency
      auto* item = is_tuple ? PyTuple_GET_ITEM(obj, 0) \
                            : PyList_GET_ITEM(obj, 0);
      if (!CheckPyDevice(item))
        return false;
    }
    return true;
  }
  return false;
}

inline std::vector<Device> DeviceList_FromPyObject(PyObject* obj) {
  bool is_tuple = PyTuple_Check(obj);
  size_t size = is_tuple ? PyTuple_GET_SIZE(obj) : PyList_GET_SIZE(obj);
  std::vector<Device> ret(size);
  for (size_t i = 0; i < size; i++) {
    auto* item = is_tuple ? PyTuple_GET_ITEM(obj, i) : PyList_GET_ITEM(obj, i);
    ret[i] = Device_FromPyObject(item);
  }
  return ret;
}

inline bool CheckPyDeviceGroup(PyObject* obj) {
  // list/tuple of devices or strings can be cast to device group
  return PyDeviceGroup_Check(obj) || CheckPyDeviceList(obj);
}

inline DeviceGroup DeviceGroup_FromPyObject(PyObject* obj) {
  if (PyDeviceGroup_Check(obj)) {
    return reinterpret_cast<PyDeviceGroup*>(obj)->device_group;
  } else {
    return DeviceGroup(DeviceList_FromPyObject(obj));
  }
}

inline bool CheckPyDeviceGroupList(PyObject* obj) {
  bool is_tuple = PyTuple_Check(obj);
  if (is_tuple || PyList_Check(obj)) {
    size_t size = is_tuple ? PyTuple_GET_SIZE(obj) : PyList_GET_SIZE(obj);
    if (size > 0) {
      // only check for the first item for efficiency
      auto* item = is_tuple ? PyTuple_GET_ITEM(obj, 0) \
                            : PyList_GET_ITEM(obj, 0);
      if (!CheckPyDeviceGroup(item))
        return false;
    }
    return true;
  }
  return false;
}

inline DeviceGroupList DeviceGroupList_FromPyObject(PyObject* obj) {
  bool is_tuple = PyTuple_Check(obj);
  size_t size = is_tuple ? PyTuple_GET_SIZE(obj) : PyList_GET_SIZE(obj);
  DeviceGroupList ret(size);
  for (size_t i = 0; i < size; i++) {
    auto* item = is_tuple ? PyTuple_GET_ITEM(obj, i) : PyList_GET_ITEM(obj, i);
    ret[i] = DeviceGroup_FromPyObject(item);
  }
  return ret;
}

inline bool CheckPyDeviceGroupHierarchy(PyObject* obj) {
  bool is_tuple = PyTuple_Check(obj);
  if (is_tuple || PyList_Check(obj)) {
    size_t size = is_tuple ? PyTuple_GET_SIZE(obj) : PyList_GET_SIZE(obj);
    if (size > 0) {
      // only check for the first item for efficiency
      auto* item = is_tuple ? PyTuple_GET_ITEM(obj, 0) \
                            : PyList_GET_ITEM(obj, 0);
      if (!CheckPyDeviceGroupList(item))
        return false;
    }
    return true;
  }
  return false;
}

inline hetu::graph::DeviceGroupHierarchy DeviceGroupHierarchy_FromPyObject(PyObject* obj) {
  bool is_tuple = PyTuple_Check(obj);
  size_t size = is_tuple ? PyTuple_GET_SIZE(obj) : PyList_GET_SIZE(obj);
  hetu::graph::DeviceGroupHierarchy ret;
  for (size_t i = 0; i < size; i++) {
    auto* item = is_tuple ? PyTuple_GET_ITEM(obj, i) : PyList_GET_ITEM(obj, i);
    ret.add(hetu::graph::DeviceGroupUnion(DeviceGroupList_FromPyObject(item)));
  }
  return ret;
}

/******************************************************
 * For contextlib usage
 ******************************************************/

ContextManager<Device>& get_eager_device_ctx();
ContextManager<hetu::graph::DeviceGroupHierarchy>& get_dg_hierarchy_ctx();

} // namespace hetu
