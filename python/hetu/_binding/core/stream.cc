#include "hetu/_binding/core/stream.h"
#include "hetu/_binding/core/device.h"
#include "hetu/_binding/constants.h"
#include "hetu/_binding/utils/pybind_common.h"
#include "hetu/_binding/utils/except.h"
#include "hetu/_binding/utils/decl_utils.h"
#include "hetu/_binding/utils/arg_parser.h"
#include "hetu/impl/stream/CUDAStream.h"

namespace hetu {

PyObject* PyStream_New(const Stream& stream) {
  HT_PY_FUNC_BEGIN
  auto* unsafe_self = PyStream_Type->tp_alloc(PyStream_Type, 0);
  HT_RUNTIME_ERROR_IF(!unsafe_self) << "Failed to alloc PyStream";  
  auto* self = reinterpret_cast<PyStream*>(unsafe_self);
  new (&self->stream) Stream(stream);
  return reinterpret_cast<PyObject*>(self);
  HT_PY_FUNC_END
}

PyObject* PyStream_pynew(PyTypeObject* type, PyObject* args, 
                         PyObject* kwargs) {
  HT_PY_FUNC_BEGIN
  auto* unsafe_self = PyStream_Type->tp_alloc(PyStream_Type, 0);
  HT_RUNTIME_ERROR_IF(!unsafe_self) << "Failed to alloc PyStream";  
  auto* self = reinterpret_cast<PyStream*>(unsafe_self);
  static PyArgParser parser({
    "Stream(string device, int stream_id)", 
    "Stream(Device device, int stream_id)", 
    "Stream(Stream stream)"
  });
  auto parsed_args = parser.parse(args, kwargs);
  if (parsed_args.signature_index() == 0) {
    auto device = Device(parsed_args.get_string(0));
    auto stream_index = parsed_args.get_int64(1);
    new (&self->stream) Stream(device, stream_index); 
  } else if (parsed_args.signature_index() == 1) {
    Device device = parsed_args.get_device(0);
    auto stream_index = parsed_args.get_int64(1);
    new (&self->stream) Stream(device, stream_index);
  } else if (parsed_args.signature_index() == 2) {
    new (&self->stream) Stream(parsed_args.get_stream(0));
  } else {
    Py_TYPE(self)->tp_free(self);
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();
  }
  return reinterpret_cast<PyObject*>(self);
  HT_PY_FUNC_END
}

void PyStream_dealloc(PyStream* self) {
  (&self->stream)->~Stream();
  Py_TYPE(self)->tp_free(self);
}

PyObject* PyStream_str(PyStream* self) {
  HT_PY_FUNC_BEGIN
  return PyUnicode_FromString(std::to_string(self->stream));
  HT_PY_FUNC_END
}

PyObject* PyStream_repr(PyStream* self) {
  return PyStream_str(self);
}

PyObject* PyStream_device(PyStream* self) {
  HT_PY_FUNC_BEGIN
  return PyDevice_New(self->stream.device());
  HT_PY_FUNC_END
}

PyObject* PyStream_device_type(PyStream* self) {
  HT_PY_FUNC_BEGIN
  return PyUnicode_FromString(DeviceType2Str(self->stream.device_type()));
  HT_PY_FUNC_END
}

PyObject* PyStream_device_index(PyStream* self) {
  HT_PY_FUNC_BEGIN
  return PyLong_FromInteger(self->stream.device_index());
  HT_PY_FUNC_END
}

PyObject* PyStream_stream_index(PyStream* self) {
  HT_PY_FUNC_BEGIN
  return PyLong_FromInteger(self->stream.stream_index());
  HT_PY_FUNC_END
}

PyObject* PyStream_is_define(PyStream* self) {
  HT_PY_FUNC_BEGIN
  Py_RETURN_BOOLEAN_COND(self->stream.is_defined());
  HT_PY_FUNC_END
}

PyObject* PyStream_rich_cmp(PyObject* obj_x, PyObject* obj_y, int op) {
  HT_PY_FUNC_BEGIN
  if (!PyStream_Check(obj_x) || !PyStream_Check(obj_y)) {
    Py_RETURN_NOTIMPLEMENTED;
  }
  auto& x = reinterpret_cast<PyStream*>(obj_x)->stream;
  auto& y = reinterpret_cast<PyStream*>(obj_y)->stream;
  Py_RETURN_RICH_CMP_EQ_ONLY(x, y, op, 
                             Py_TYPE(obj_x)->tp_name, 
                             Py_TYPE(obj_y)->tp_name);
  HT_PY_FUNC_END
}

Py_ssize_t PyStream_hash(PyStream* self) {
  HT_PY_FUNC_BEGIN
  return static_cast<Py_ssize_t>(std::hash<Stream>()(self->stream));
  HT_PY_FUNC_RETURN(-1)
}

PyObject* PyStream_ptr(PyStream* self) {
  HT_PY_FUNC_BEGIN
  return PyLong_FromInteger(int64_t((void*)(impl::CUDAStream(self->stream).cuda_stream())));
  HT_PY_FUNC_END
}

PyObject* PyStream_reduce(PyStream* self) {
  HT_PY_FUNC_BEGIN
  const auto& stream = self->stream;
  auto* ret = PyTuple_New(2);
  HT_RUNTIME_ERROR_IF(!ret) << "Failed to alloc tuple";
  
  py::object hetu_module = py::module::import("hetu");
  py::object hetu_stream = hetu_module.attr("stream");
  PyTuple_SET_ITEM(ret, 0, hetu_stream.release().ptr());

  auto* args = PyTuple_New(2);
  HT_RUNTIME_ERROR_IF(!args) << "Failed to alloc tuple";
  auto* device_str_obj = PyUnicode_FromString(stream.device().compat_string());
  PyTuple_SET_ITEM(args, 0, device_str_obj);
  auto* stream_index_obj = PyLong_FromInteger(stream.stream_index());
  PyTuple_SET_ITEM(args, 1, stream_index_obj);
  PyTuple_SET_ITEM(ret, 1, args);

  return ret;
  HT_PY_FUNC_END
}

// NOLINTNEXTLINE
PyGetSetDef PyStream_properties[] = {
  {PY_GET_SET_DEF_NAME("device"), (getter) PyStream_device, nullptr, nullptr, nullptr}, 
  {PY_GET_SET_DEF_NAME("device_type"), (getter) PyStream_device_type, nullptr, nullptr, nullptr}, 
  {PY_GET_SET_DEF_NAME("device_index"), (getter) PyStream_device_index, nullptr, nullptr, nullptr}, 
  {PY_GET_SET_DEF_NAME("stream_index"), (getter) PyStream_stream_index, nullptr, nullptr, nullptr}, 
  {PY_GET_SET_DEF_NAME("is_define"), (getter) PyStream_is_define, nullptr, nullptr, nullptr}, 
  {PY_GET_SET_DEF_NAME("ptr"), (getter) PyStream_ptr, nullptr, nullptr, nullptr}, 
  {nullptr}
};

// NOLINTNEXTLINE
PyMethodDef PyStream_methods[] = {
  {"__reduce__", (PyCFunction) PyStream_reduce, METH_NOARGS, nullptr}, 
  {nullptr}
};

// NOLINTNEXTLINE
PyTypeObject PyStream_Type_obj = {
  PyVarObject_HEAD_INIT(nullptr, 0) 
  "hetu.stream", /* tp_name */
  sizeof(PyStream), /* tp_basicsize */
  0, /* tp_itemsize */
  (destructor) PyStream_dealloc, /* tp_dealloc */
  0, /* tp_vectorcall_offset */
  nullptr, /* tp_getattr */
  nullptr, /* tp_setattr */
  nullptr, /* tp_reserved */
  (reprfunc) PyStream_repr, /* tp_repr */
  nullptr, /* tp_as_number */
  nullptr, /* tp_as_sequence */
  nullptr, /* tp_as_mapping */
  (hashfunc) PyStream_hash, /* tp_hash  */
  nullptr, /* tp_call */
  (reprfunc) PyStream_str, /* tp_str */
  nullptr, /* tp_getattro */
  nullptr, /* tp_setattro */
  nullptr, /* tp_as_buffer */
  Py_TPFLAGS_DEFAULT, /* tp_flags */
  nullptr, /* tp_doc */
  nullptr, /* tp_traverse */
  nullptr, /* tp_clear */
  (richcmpfunc) PyStream_rich_cmp, /* tp_richcompare */
  0, /* tp_weaklistoffset */
  nullptr, /* tp_iter */
  nullptr, /* tp_iternext */
  PyStream_methods, /* tp_methods */
  nullptr, /* tp_members */
  PyStream_properties, /* tp_getset */
  nullptr, /* tp_base */
  nullptr, /* tp_dict */
  nullptr, /* tp_descr_get */
  nullptr, /* tp_descr_set */
  0, /* tp_dictoffset */
  nullptr, /* tp_init */
  nullptr, /* tp_alloc */
  PyStream_pynew, /* tp_new */
};
PyTypeObject* PyStream_Type = &PyStream_Type_obj;

void AddPyStreamTypeToModule(py::module_& module) {
  HT_RUNTIME_ERROR_IF(PyType_Ready(PyStream_Type) < 0) 
    << "PyStream_Type not ready";
  Py_INCREF(PyStream_Type);
  HT_RUNTIME_ERROR_IF(0 != PyModule_AddObject(
      module.ptr(), "stream", reinterpret_cast<PyObject*>(PyStream_Type)))
    << "Failed to add PyStream_Type";
}

ContextManager<StreamIndex>& get_stream_index_ctx() {
  static ContextManager<StreamIndex> stream_index_ctx;
  return stream_index_ctx;
}

ContextManager<Stream>& get_stream_ctx() {
  static ContextManager<Stream> stream_ctx;
  return stream_ctx;
}

} // namespace hetu
