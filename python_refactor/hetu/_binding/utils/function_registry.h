#pragma once

#include <Python.h>
#include "hetu/common/macros.h"
#include "hetu/_binding/utils/decl_utils.h"

namespace hetu {

namespace impl {

std::vector<PyMethodDef>& get_registered_ndarray_methods();

std::vector<PyMethodDef>& get_registered_ndarray_class_methods();

int RegisterNDArrayMethod(const char* name, PyCFunction func, int flags,
                          const char* doc);

int RegisterNDArrayClassMethod(const char* name, PyCFunction func, int flags,
                               const char* doc);

#define REGISTER_NDARRAY_METHOD(name, func, flags, doc)                        \
  static int __ndarray_method_##name##_registry =                              \
    hetu::impl::RegisterNDArrayMethod(quote(name), func, flags, doc)

#define REGISTER_NDARRAY_CLASS_METHOD(name, func, flags, doc)                  \
  static int __ndarray_class_method_##name##_registry =                        \
    hetu::impl::RegisterNDArrayClassMethod(quote(name), func, flags, doc)

} // namespace impl

namespace graph {

PyNumberMethods& get_registered_tensor_number_methods();

std::vector<PyMethodDef>& get_registered_tensor_methods();

std::vector<PyMethodDef>& get_registered_tensor_class_methods();

int RegisterTensorMethod(const char* name, PyCFunction func, int flags,
                         const char* doc);

int RegisterTensorClassMethod(const char* name, PyCFunction func, int flags,
                              const char* doc);

#define REGISTER_TENSOR_NUMBER_METHOD(slot, func)                              \
  static auto __tensor_number_method_##slot##_registry =                       \
    ((hetu::graph::get_registered_tensor_number_methods().slot) = (func))

#define REGISTER_TENSOR_METHOD(name, func, flags, doc)                         \
  static auto __tensor_method_##name##_registry =                              \
    hetu::graph::RegisterTensorMethod(quote(name), func, flags, doc)

#define REGISTER_TENSOR_CLASS_METHOD(name, func, flags, doc)                   \
  static auto __tensor_class_method_##name##_registry =                        \
    hetu::graph::RegisterTensorClassMethod(quote(name), func, flags, doc)

PyNumberMethods& get_registered_optimizer_number_methods();

std::vector<PyMethodDef>& get_registered_optimizer_methods();

std::vector<PyMethodDef>& get_registered_optimizer_class_methods();

int RegisterOptimizerMethod(const char* name, PyCFunction func, int flags,
                            const char* doc);

int RegisterOptimizerClassMethod(const char* name, PyCFunction func, int flags,
                                 const char* doc);

#define REGISTER_OPTIMIZER_NUMBER_METHOD(slot, func)                              \
  static auto __optimizer_number_method_##slot##_registry =                       \
    ((hetu::graph::get_registered_optimizer_number_methods().slot) = (func))

#define REGISTER_OPTIMIZER_METHOD(name, func, flags, doc)                         \
  static auto __optimizer_method_##name##_registry =                              \
    hetu::graph::RegisterOptimizerMethod(quote(name), func, flags, doc)

#define REGISTER_OPTIMIZER_CLASS_METHOD(name, func, flags, doc)                   \
  static auto __optimizer_class_method_##name##_registry =                        \
    hetu::graph::RegisterOptimizerClassMethod(quote(name), func, flags, doc)

} // namespace graph

} // namespace hetu
