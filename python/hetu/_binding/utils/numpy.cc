#ifndef PY_ARRAY_UNIQUE_SYMBOL
#define PY_ARRAY_UNIQUE_SYMBOL _HETU_NUMPY_ARRAY_API
#endif

#ifndef NPY_NO_DEPRECATED_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#endif

#include <numpy/arrayobject.h>
#include <pybind11/numpy.h>
#include "hetu/_binding/utils/pybind_common.h"
#include "hetu/_binding/utils/numpy.h"
#include "hetu/_binding/utils/except.h"
#include "hetu/impl/utils/dispatch.h"
#include "hetu/utils/optional.h"
#include <mutex>

namespace hetu {

namespace {
static std::once_flag import_np_flag;
static void ImportNumpyOnce() {
  std::call_once(import_np_flag, []() {
    HT_RUNTIME_ERROR_IF(_import_array() < 0) << "Failed to import NumPy";
  });
}

inline HTShape FromNumpyShape(npy_intp* numpy_shape, size_t ndim) {
  HTShape shape(ndim);
  for (size_t i = 0; i < ndim; i++)
    shape[i] = static_cast<HTShape::value_type>(numpy_shape[i]);
  return shape;
}

/*
inline HTShape FromNumpyDynamicShape(void* numpy_shape, size_t ndim, size_t numpy_dsize) {
  HTShape shape(ndim);
  for (size_t i = 0; i < ndim; i++)
    shape[i] = *static_cast<HTShape::value_type *>(numpy_shape + i * numpy_dsize);
  return shape;
}
*/

inline HTStride FromNumpyStride(npy_intp* numpy_stride, size_t ndim, 
                               size_t item_size) {
  HTStride stride(ndim);
  for (size_t i = 0; i < ndim; i++) {
    stride[i] = static_cast<HTStride::value_type>(numpy_stride[i]);
    HT_VALUE_ERROR_IF(stride[i] < 0) << "Negative strides are not supported";
    HT_RUNTIME_ERROR_IF(stride[i] % item_size != 0) << "Stride " << stride[i] 
      << " is not a multiple of item size " << item_size;
    stride[i] /= item_size;
  }
  return stride;
}

inline DataType FromNumpyDataType(int numpy_dtype, size_t numpy_dsize) {
  DataType dtype;
  PyObject* dtype_obj;
  switch (numpy_dtype) {
    case NPY_UINT8: dtype = kUInt8; break;
    case NPY_INT8: dtype = kInt8; break;
    case NPY_INT16: dtype = kInt16; break;
    case NPY_INT32: dtype = kInt32; break;
    case NPY_INT64: dtype = kInt64; break;
    case NPY_FLOAT16: dtype = kFloat16; break;
    case NPY_FLOAT32: dtype = kFloat32; break;
    case NPY_FLOAT64: dtype = kFloat64; break;
    case NPY_BOOL:  dtype = kBool; break;
    default:
      dtype_obj = PyArray_TypeObjectFromType(numpy_dtype);
      HT_RUNTIME_ERROR_IF(!dtype_obj) 
        << "Failed to get numpy type with " << numpy_dtype;
      HT_VALUE_ERROR 
        << "Cannot convert numpy array with type "
        << reinterpret_cast<PyTypeObject*>(dtype_obj)->tp_name 
        << ". Supported types: uint8, int8, int16, int32, int64, "
        << "float16, float32, float64, and bool.";
      __builtin_unreachable();
  }
  auto dsize = DataType2Size(dtype);
  if (dsize != numpy_dsize) {
    dtype_obj = PyArray_TypeObjectFromType(numpy_dtype);
    HT_RUNTIME_ERROR_IF(!dtype_obj) 
      << "Failed to get numpy type with " << numpy_dtype;
    HT_RUNTIME_ERROR 
      << "Data sizes mismatch: " 
      << dsize << " (hetu." << dtype << ") vs. " << numpy_dsize << " (" 
      << reinterpret_cast<PyTypeObject*>(dtype_obj)->tp_name << ").";
  }
  return dtype;
}

class StorageWathcer {
 public:
  StorageWathcer(std::shared_ptr<NDArrayStorage> storage)
  : _storage(storage) {}
 private:
  std::shared_ptr<NDArrayStorage> _storage;
};

} // namespace

bool CheckNumpyInt(PyObject* obj) {
  ImportNumpyOnce();
  return PyArray_IsScalar(obj, Integer);
}

bool CheckNumpyFloat(PyObject* obj) {
  ImportNumpyOnce();
  return PyArray_IsScalar(obj, Floating);
}

bool CheckNumpyBool(PyObject* obj) {
  ImportNumpyOnce();
  return PyArray_IsScalar(obj, Bool);
}

bool CheckNumpyArray(PyObject* obj) {
  ImportNumpyOnce();
  return PyArray_Check(obj);
}

bool CheckNumpyArrayList(PyObject* obj) {
  bool is_tuple = PyTuple_Check(obj);
  if (is_tuple || PyList_Check(obj)) {
    size_t size = is_tuple ? PyTuple_GET_SIZE(obj) : PyList_GET_SIZE(obj);
    if (size > 0) {
      // only check for the first item for efficiency
      auto* item = is_tuple ? PyTuple_GET_ITEM(obj, 0) \
                            : PyList_GET_ITEM(obj, 0);
      if (!CheckNumpyArray(item))
        return false;
    }
    return true;
  }
  return false;
}

DataType GetNumpyArrayDataType(PyObject* obj) {
  auto* numpy_array = reinterpret_cast<PyArrayObject*>(obj);
  auto element_size = static_cast<size_t>(PyArray_ITEMSIZE(numpy_array));
  return FromNumpyDataType(PyArray_TYPE(numpy_array), element_size);
}

NDArray NDArrayFromNumpy(PyObject* obj, const HTShape& dynamic_shape, DataType datatype) {
  auto* numpy_array = reinterpret_cast<PyArrayObject*>(obj);
  HT_VALUE_ERROR_IF(!PyArray_EquivByteorders(
      PyArray_DESCR(numpy_array)->byteorder, NPY_NATIVE))
    << "The provided Numpy array is not in machine byte-order";
  
  bool writable = PyArray_ISWRITEABLE(numpy_array);
  // HT_LOG_WARN_IF(!writable) << "The provided Numpy array is non-writable.";
  HT_VALUE_ERROR_IF(!PyArray_IS_C_CONTIGUOUS(numpy_array))
    << "Non-contiguous arrays are not supported yet.";

  auto ndim = static_cast<size_t>(PyArray_NDIM(numpy_array));
  auto shape = FromNumpyShape(PyArray_DIMS(numpy_array), ndim);
  auto element_size = static_cast<size_t>(PyArray_ITEMSIZE(numpy_array));
  auto stride = FromNumpyStride(PyArray_STRIDES(numpy_array), ndim, element_size);
  HT_VALUE_ERROR_IF(stride != Shape2Stride(shape))
    << "Strided arrays are not supported yet.";
  if (datatype == kFloat4 || datatype == kNFloat4) {
    shape[ndim - 1] *= 2;
    for (int i = 0; i < ndim - 1; ++i) {
      stride[i] *= 2;
    }
  }
  auto dtype = datatype == kUndeterminedDataType ? FromNumpyDataType(PyArray_TYPE(numpy_array), element_size)
                                                 : datatype;
  auto meta = NDArrayMeta().set_dtype(dtype).set_shape(shape).set_device(kCPU);

  if (!dynamic_shape.empty())
    meta.set_dynamic_shape(dynamic_shape);

  void* ptr = PyArray_DATA(numpy_array);
  Py_INCREF(obj);
  // TODO: mark non-writable and lazy copy on writing
  int64_t borrow_size = (datatype == kFloat4 || datatype == kNFloat4) 
                        ? ((meta.numel() + 1) / 2) * element_size 
                        : meta.numel() * element_size;
  auto storage = std::make_shared<NDArrayStorage>(BorrowToMemoryPool(
    Device(kCPU), ptr, borrow_size, [obj](DataPtr ptr) {
      py::call_guard<py::gil_scoped_acquire>();
      // py::gil_scoped_acquire gil;
      Py_DECREF(obj);
    }));

  return NDArray(meta, storage);
}

NDArrayList NDArrayListFromNumpyList(PyObject* obj, const HTShape& dynamic_shape, DataType datatype) {
  bool is_tuple = PyTuple_Check(obj);
  size_t size = is_tuple ? PyTuple_GET_SIZE(obj) : PyList_GET_SIZE(obj);
  NDArrayList ret(size);
  for (size_t i = 0; i < size; i++) {
    auto* item = is_tuple ? PyTuple_GET_ITEM(obj, i) : PyList_GET_ITEM(obj, i);
    ret[i] = NDArrayFromNumpy(item, dynamic_shape, datatype);
  }
  return ret;  
}

PyObject* NDArrayToNumpy(NDArray ndarray, bool force, bool save) {
  if (!ndarray->is_cpu()) {
    HT_VALUE_ERROR_IF(!force) 
      << "Cannot convert data on " << ndarray->device().type() << " "
      << "to numpy array. Please set force=True or "
      << "copy the data to host memory first";
    ndarray = NDArray::cpu(ndarray, kBlockingStream);
  }
  if (ndarray->dtype() == DataType::FLOAT4 || ndarray->dtype() == DataType::NFLOAT4) {
    int element_size = 1;
    HTStride numpy_stride = ndarray->stride();
    numpy_stride[ndarray->ndim() - 1] *= 2;
    for (auto& v : numpy_stride)
      v = (v * element_size / 2);
    auto numpy_shape = ndarray->shape();
    numpy_shape[numpy_shape.size() - 1] /= 2;

    PyObject* ret = nullptr;
    auto* watcher = new StorageWathcer(ndarray->storage());
    py::capsule deref(watcher, [](void* watcher) {
      delete reinterpret_cast<StorageWathcer*>(watcher);
    });
    py::array_t<uint8_t, py::array::c_style> py_arr(
      numpy_shape, 
      numpy_stride, 
      ndarray->data_ptr<uint8_t>(), 
      deref);
    ret = py_arr.release().ptr();
    HT_RUNTIME_ERROR_IF(!ret) << "Failed to create numpy array";  
    return ret;
  }
  if ((ndarray->dtype() == DataType::FLOAT16 || ndarray->dtype() == DataType::BFLOAT16) && save) {
    int element_size = 2;
    HTStride numpy_stride = ndarray->stride();
    for (auto& v : numpy_stride)
      v *= element_size;
    auto numpy_shape = ndarray->shape();

    PyObject* ret = nullptr;
    auto* watcher = new StorageWathcer(ndarray->storage());
    py::capsule deref(watcher, [](void* watcher) {
      delete reinterpret_cast<StorageWathcer*>(watcher);
    });
    py::array_t<int16_t, py::array::c_style> py_arr(
      numpy_shape, 
      numpy_stride, 
      ndarray->data_ptr<int16_t>(), 
      deref);
    ret = py_arr.release().ptr();
    HT_RUNTIME_ERROR_IF(!ret) << "Failed to create numpy array";  
    return ret;
  }
  if (ndarray->dtype() == DataType::FLOAT16 || ndarray->dtype() == DataType::BFLOAT16) {
    ndarray = NDArray::toFloat32(ndarray, kBlockingStream);
  }
  
  auto element_size = DataType2Size(ndarray->dtype());
  HTStride numpy_stride = ndarray->stride();
  for (auto& v : numpy_stride)
    v *= element_size;

  PyObject* ret = nullptr;
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES_EXCEPT_FLOAT16(
    ndarray->dtype(), spec_t, "ToNumpy", [&]() {
      auto* watcher = new StorageWathcer(ndarray->storage());
      py::capsule deref(watcher, [](void* watcher) {
        delete reinterpret_cast<StorageWathcer*>(watcher);
      });
      py::array_t<spec_t, py::array::c_style> py_arr(
        ndarray->shape(), 
        numpy_stride, 
        ndarray->data_ptr<spec_t>(), 
        deref);
      ret = py_arr.release().ptr();
    });
  HT_RUNTIME_ERROR_IF(!ret) << "Failed to create numpy array";  
  return ret;
}

PyObject* NumpyFromSequences(PyObject* obj) {
  auto* numpy_array = PyArray_FromAny(
    obj, nullptr, 0, 0, 
    NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_BEHAVED | \
      NPY_ARRAY_ENSUREARRAY | NPY_ARRAY_FORCECAST, 
    nullptr);
  HT_VALUE_ERROR_IF(!(numpy_array && CheckNumpyArray(numpy_array))) 
    << "The input should be sequences that can be converted to Numpy arrays";
  return numpy_array;
}

} // namespace hetu
