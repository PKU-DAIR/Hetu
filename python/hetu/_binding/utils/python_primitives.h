#pragma once

#include "hetu/common/macros.h"
#include <Python.h>
#include "hetu/_binding/utils/except.h"
#include "hetu/_binding/utils/numpy.h"

namespace hetu {

inline bool CheckPyBool(PyObject* obj) {
  return PyBool_Check(obj);
}

inline bool Bool_FromPyBool(PyObject* obj) {
  return obj == Py_True;
}

inline bool CheckPyLong(PyObject* obj) {
  return (PyLong_Check(obj) && !PyBool_Check(obj)) || CheckNumpyInt(obj);
}

inline int32_t Int32_FromPyLong(PyObject* obj) {
  int overflow = 0;
  auto ret = PyLong_AsLongAndOverflow(obj, &overflow);
  HT_VALUE_ERROR_IF(PyErr_Occurred()) 
    << "Error occurred in PyLong_AsLongAndOverflow";
  HT_VALUE_ERROR_IF(overflow != 0) 
    << "Python integer overflowed for int32_t";
  return ret;
}

inline int64_t Int64_FromPyLong(PyObject* obj) {
  int overflow = 0;
  auto ret = PyLong_AsLongLongAndOverflow(obj, &overflow);
  HT_VALUE_ERROR_IF(PyErr_Occurred()) 
    << "Error occurred in PyLong_AsLongLongAndOverflow";
  HT_VALUE_ERROR_IF(overflow != 0) 
    << "Python integer overflowed for int64_t";
  return ret;
}

inline bool CheckPyFloat(PyObject* obj) {
  return PyFloat_Check(obj) || PyLong_Check(obj) || 
    CheckNumpyInt(obj) || CheckNumpyFloat(obj);
}

template <typename T, 
          std::enable_if_t<std::is_unsigned<T>::value>* = nullptr>
inline PyObject* PyLong_FromInteger(T value) {
  return PyLong_FromUnsignedLongLong(static_cast<uint64_t>(value));
}

template <typename T,
          std::enable_if_t<std::is_integral<T>::value &&
                           std::is_signed<T>::value>* = nullptr>
inline PyObject* PyLong_FromInteger(T value) {
  return PyLong_FromLongLong(static_cast<int64_t>(value));
}

inline double Float64_FromPyFloat(PyObject* obj) {
  if (PyFloat_Check(obj)) {
    return PyFloat_AS_DOUBLE(obj);
  }
  auto ret = PyFloat_AsDouble(obj);
  HT_VALUE_ERROR_IF(PyErr_Occurred()) << "Error occurred in PyFloat_AsDouble";
  return ret;
}

inline bool CheckPyString(PyObject* obj) {
  return PyBytes_Check(obj) || PyUnicode_Check(obj);
}

inline bool CheckPyBytes(PyObject* obj) {
  return PyBytes_Check(obj);
}

inline bool CheckPyByteArray(PyObject* obj) {
  return PyByteArray_Check(obj);
}

inline PyObject* PyUnicode_FromString(const std::string& str) {
  return PyUnicode_FromStringAndSize(str.c_str(), str.size());
}

inline std::string String_FromPyUnicode(PyObject* obj) {
  if (PyBytes_Check(obj)) {
    size_t length = PyBytes_GET_SIZE(obj);
    return std::string(PyBytes_AS_STRING(obj), length);
  } else if (PyUnicode_Check(obj)) {
    Py_ssize_t length;
    const char* ret = PyUnicode_AsUTF8AndSize(obj, &length);
    HT_VALUE_ERROR_IF(!ret) << "Error occurred in PyUnicode_AsUTF8AndSize";
    return std::string(ret, length);
  } else {
    HT_VALUE_ERROR << "Cannot cast " << Py_TYPE(obj)->tp_name << " as string";
    __builtin_unreachable();
  }
}

inline bool CheckPyIntList(PyObject* obj) {
  bool is_tuple = PyTuple_Check(obj);
  if (is_tuple || PyList_Check(obj)) {
    size_t size = is_tuple ? PyTuple_GET_SIZE(obj) : PyList_GET_SIZE(obj);
    if (size > 0) {
      // only check for the first item for efficiency
      auto* item = is_tuple ? PyTuple_GET_ITEM(obj, 0) \
                            : PyList_GET_ITEM(obj, 0);
      if (!CheckPyLong(item))
        return false;
    }
    return true;
  }
  return false;
}

inline std::vector<int32_t> Int32List_FromPyIntList(PyObject* obj) {
  bool is_tuple = PyTuple_Check(obj);
  size_t size = is_tuple ? PyTuple_GET_SIZE(obj) : PyList_GET_SIZE(obj);
  std::vector<int32_t> ret(size);
  for (size_t i = 0; i < size; i++) {
    auto* item = is_tuple ? PyTuple_GET_ITEM(obj, i) : PyList_GET_ITEM(obj, i);
    ret[i] = Int32_FromPyLong(item);
  }
  return ret;
}

inline std::vector<int64_t> Int64List_FromPyIntList(PyObject* obj) {
  bool is_tuple = PyTuple_Check(obj);
  size_t size = is_tuple ? PyTuple_GET_SIZE(obj) : PyList_GET_SIZE(obj);
  std::vector<int64_t> ret(size);
  for (size_t i = 0; i < size; i++) {
    auto* item = is_tuple ? PyTuple_GET_ITEM(obj, i) : PyList_GET_ITEM(obj, i);
    ret[i] = Int64_FromPyLong(item);
  }
  return ret;
}

template <typename T,
          std::enable_if_t<std::is_integral<T>::value>* = nullptr>
inline PyObject* PyLongList_FromIntegerList(const std::vector<T>& values) {
  auto* ret = PyList_New(values.size());
  for (size_t i = 0; i < values.size(); i++)
    PyList_SET_ITEM(ret, i, PyLong_FromInteger(values[i]));
  return ret;
}

inline bool CheckPyFloatList(PyObject* obj) {
  bool is_tuple = PyTuple_Check(obj);
  if (is_tuple || PyList_Check(obj)) {
    size_t size = is_tuple ? PyTuple_GET_SIZE(obj) : PyList_GET_SIZE(obj);
    if (size > 0) {
      // only check for the first item for efficiency
      auto* item = is_tuple ? PyTuple_GET_ITEM(obj, 0) \
                            : PyList_GET_ITEM(obj, 0);
      if (!CheckPyFloat(item))
        return false;
    }
    return true;
  }
  return false;
}

inline std::vector<double> Float64List_FromPyFloatList(PyObject* obj) {
  bool is_tuple = PyTuple_Check(obj);
  size_t size = is_tuple ? PyTuple_GET_SIZE(obj) : PyList_GET_SIZE(obj);
  std::vector<double> ret(size);
  for (size_t i = 0; i < size; i++) {
    auto* item = is_tuple ? PyTuple_GET_ITEM(obj, i) : PyList_GET_ITEM(obj, i);
    ret[i] = Float64_FromPyFloat(item);
  }
  return ret;
}

inline bool CheckPyBoolList(PyObject* obj) {
  bool is_tuple = PyTuple_Check(obj);
  if (is_tuple || PyList_Check(obj)) {
    size_t size = is_tuple ? PyTuple_GET_SIZE(obj) : PyList_GET_SIZE(obj);
    if (size > 0) {
      // only check for the first item for efficiency
      auto* item = is_tuple ? PyTuple_GET_ITEM(obj, 0) \
                            : PyList_GET_ITEM(obj, 0);
      if (!CheckPyBool(item))
        return false;
    }
    return true;
  }
  return false;
}

inline std::vector<bool> BoolList_FromPyBoolList(PyObject* obj) {
  bool is_tuple = PyTuple_Check(obj);
  size_t size = is_tuple ? PyTuple_GET_SIZE(obj) : PyList_GET_SIZE(obj);
  std::vector<bool> ret(size);
  for (size_t i = 0; i < size; i++) {
    auto* item = is_tuple ? PyTuple_GET_ITEM(obj, i) : PyList_GET_ITEM(obj, i);
    ret[i] = Bool_FromPyBool(item);
  }
  return ret;
}

inline bool CheckPyBoolListList(PyObject* obj) {
  bool is_tuple = PyTuple_Check(obj);
  if (is_tuple || PyList_Check(obj)) {
    size_t size = is_tuple ? PyTuple_GET_SIZE(obj) : PyList_GET_SIZE(obj);
    if (size > 0) {
      // only check for the first item for efficiency
      auto* item = is_tuple ? PyTuple_GET_ITEM(obj, 0) \
                            : PyList_GET_ITEM(obj, 0);
      if (!CheckPyBoolList(item))
        return false;
    }
    return true;
  }
  return false;
}

inline std::vector<std::vector<bool>> BoolListList_FromPyBoolListList(PyObject* obj) {
  bool is_tuple = PyTuple_Check(obj);
  size_t size = is_tuple ? PyTuple_GET_SIZE(obj) : PyList_GET_SIZE(obj);
  std::vector<std::vector<bool>> ret(size);
  for (size_t i = 0; i < size; i++) {
    auto* item = is_tuple ? PyTuple_GET_ITEM(obj, i) : PyList_GET_ITEM(obj, i);
    ret[i] = BoolList_FromPyBoolList(item);
  }
  return ret;
}

inline bool CheckPyStringList(PyObject* obj) {
  bool is_tuple = PyTuple_Check(obj);
  if (is_tuple || PyList_Check(obj)) {
    size_t size = is_tuple ? PyTuple_GET_SIZE(obj) : PyList_GET_SIZE(obj);
    if (size > 0) {
      // only check for the first item for efficiency
      auto* item = is_tuple ? PyTuple_GET_ITEM(obj, 0) \
                            : PyList_GET_ITEM(obj, 0);
      if (!CheckPyString(item))
        return false;
    }
    return true;
  }
  return false;
}

inline std::vector<std::string> StringList_FromPyStringList(PyObject* obj) {
  bool is_tuple = PyTuple_Check(obj);
  size_t size = is_tuple ? PyTuple_GET_SIZE(obj) : PyList_GET_SIZE(obj);
  std::vector<std::string> ret(size);
  for (size_t i = 0; i < size; i++) {
    auto* item = is_tuple ? PyTuple_GET_ITEM(obj, i) : PyList_GET_ITEM(obj, i);
    ret[i] = String_FromPyUnicode(item);
  }
  return ret;
}

inline std::unordered_map<int32_t, int32_t> UnorderedMap_FromPyDict(PyObject* obj) {
  if(!PyDict_Check(obj)) {
    HT_VALUE_ERROR << "Expected a dictionary object.";
  }
  std::unordered_map<int32_t, int32_t> ret_map;
  Py_ssize_t pos = 0;
  PyObject* key;
  PyObject* value;

  while (PyDict_Next(obj, &pos, &key, &value)) {
    if (!PyLong_Check(key) || !PyLong_Check(value)) {
      HT_VALUE_ERROR << "Expected integer keys and values in the dictionary.";
    }
    int32_t key_ = static_cast<int32_t>(PyLong_AsLong(key));
    int32_t value_ = static_cast<int32_t>(PyLong_AsLong(value));
    ret_map[key_] = value_;
  }

  return ret_map;
}

inline PyObject* PyDict_FromUnorderedMap(std::unordered_map<int32_t, int32_t> map) {
  PyObject* dict_obj = PyDict_New();
  for (auto& item : map) {
    PyObject* key = PyLong_FromInteger(item.first);
    PyObject* value = PyLong_FromInteger(item.second);
    PyDict_SetItem(dict_obj, key, value);
  }
  return dict_obj;
}

} // namespace
