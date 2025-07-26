#pragma once

#include "hetu/common/macros.h"
#include "hetu/core/dtype.h"

/******************************************************
 * Dispatch Utils. Learned from PyTorch.
 ******************************************************/

#define HT_DISPATH_CASE(DATA_TYPE, SPEC_TYPE, ...)                             \
  case DATA_TYPE: {                                                            \
    using SPEC_TYPE = hetu::DataType2SpecMeta<DATA_TYPE>::spec_type;           \
    __VA_ARGS__();                                                             \
    break;                                                                     \
  }

#define HT_DISPATH_SWITCH(DTYPE, NAME, ...)                                    \
  do {                                                                         \
    const auto& _dtype = DTYPE;                                                \
    switch (_dtype) {                                                          \
      __VA_ARGS__                                                              \
      default:                                                                 \
        HT_NOT_IMPLEMENTED << "\"" << NAME << "\" is not implemented for "     \
                           << "\"" << _dtype << "\"";                          \
    }                                                                          \
  } while (0)

#define HT_DISPATH_CASE_FLOATING_TYPES(SPEC_TYPE, ...)                         \
  HT_DISPATH_CASE(hetu::DataType::FLOAT16, SPEC_TYPE, __VA_ARGS__)             \
  HT_DISPATH_CASE(hetu::DataType::BFLOAT16, SPEC_TYPE, __VA_ARGS__)            \
  HT_DISPATH_CASE(hetu::DataType::FLOAT32, SPEC_TYPE, __VA_ARGS__)             \
  HT_DISPATH_CASE(hetu::DataType::FLOAT64, SPEC_TYPE, __VA_ARGS__)

#define HT_DISPATH_CASE_FLOATING_TYPES_EXCEPT_FLOAT16(SPEC_TYPE, ...)          \
  HT_DISPATH_CASE(hetu::DataType::FLOAT32, SPEC_TYPE, __VA_ARGS__)             \
  HT_DISPATH_CASE(hetu::DataType::FLOAT64, SPEC_TYPE, __VA_ARGS__)

#define HT_DISPATCH_FLOATING_TYPES(DTYPE, SPEC_TYPE, NAME, ...)                \
  HT_DISPATH_SWITCH(DTYPE, NAME,                                               \
                    HT_DISPATH_CASE_FLOATING_TYPES(SPEC_TYPE, __VA_ARGS__))

#define HT_DISPATH_CASE_INTEGER_TYPES(SPEC_TYPE, ...)                          \
  HT_DISPATH_CASE(hetu::DataType::UINT8, SPEC_TYPE, __VA_ARGS__)               \
  HT_DISPATH_CASE(hetu::DataType::INT8, SPEC_TYPE, __VA_ARGS__)                \
  HT_DISPATH_CASE(hetu::DataType::INT16, SPEC_TYPE, __VA_ARGS__)               \
  HT_DISPATH_CASE(hetu::DataType::INT32, SPEC_TYPE, __VA_ARGS__)               \
  HT_DISPATH_CASE(hetu::DataType::INT64, SPEC_TYPE, __VA_ARGS__)

#define HT_DISPATCH_INTEGER_TYPES(DTYPE, SPEC_TYPE, NAME, ...)                 \
  HT_DISPATH_SWITCH(DTYPE, NAME,                                               \
                    HT_DISPATH_CASE_INTEGER_TYPES(SPEC_TYPE, __VA_ARGS__))

#define HT_DISPATH_CASE_INTEGER_AND_FLOATING_TYPES(SPEC_TYPE, ...)             \
  HT_DISPATH_CASE_FLOATING_TYPES(SPEC_TYPE, __VA_ARGS__)                       \
  HT_DISPATH_CASE_INTEGER_TYPES(SPEC_TYPE, __VA_ARGS__)

#define HT_DISPATH_CASE_INTEGER_AND_FLOATING_TYPES_EXCEPT_FLOAT16(SPEC_TYPE, ...)             \
  HT_DISPATH_CASE_FLOATING_TYPES_EXCEPT_FLOAT16(SPEC_TYPE, __VA_ARGS__)                       \
  HT_DISPATH_CASE_INTEGER_TYPES(SPEC_TYPE, __VA_ARGS__)

#define HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(DTYPE, SPEC_TYPE, NAME, ...)    \
  HT_DISPATH_SWITCH(                                                           \
    DTYPE, NAME,                                                               \
    HT_DISPATH_CASE_INTEGER_AND_FLOATING_TYPES(SPEC_TYPE, __VA_ARGS__))

#define HT_DISPATCH_INTEGER_AND_FLOATING_TYPES_EXCEPT_FLOAT16(DTYPE, SPEC_TYPE, NAME, ...)    \
  HT_DISPATH_SWITCH(                                                                          \
    DTYPE, NAME,                                                                              \
    HT_DISPATH_CASE_INTEGER_AND_FLOATING_TYPES_EXCEPT_FLOAT16(SPEC_TYPE, __VA_ARGS__))

/******************************************************
 * Dispatch Utils for Paired Types.
 * The current implement is tedious and hard to read.
 * Try to make it elegant in the future.
 ******************************************************/

namespace hetu {
namespace __paired_dtype_anon {
constexpr int paired_dtype(DataType t1, DataType t2) {
  return (static_cast<int>(t1) << 8) | static_cast<int>(t2);
};
} // namespace __paired_dtype_anon
} // namespace hetu

#define HT_DISPATH_PAIRED_CASE(DTYPE_A, DTYPE_B, SPEC_A, SPEC_B, ...)          \
  case hetu::__paired_dtype_anon::paired_dtype(DTYPE_A, DTYPE_B): {            \
    using SPEC_A = hetu::DataType2SpecMeta<DTYPE_A>::spec_type;                \
    using SPEC_B = hetu::DataType2SpecMeta<DTYPE_B>::spec_type;                \
    __VA_ARGS__();                                                             \
    break;                                                                     \
  }

#define HT_DISPATH_PAIRED_SWITCH(DTYPE_A, DTYPE_B, NAME, ...)                  \
  do {                                                                         \
    const auto& _dtype_a = DTYPE_A;                                            \
    const auto& _dtype_b = DTYPE_B;                                            \
    switch (hetu::__paired_dtype_anon::paired_dtype(_dtype_a, _dtype_b)) {     \
      __VA_ARGS__                                                              \
      default:                                                                 \
        HT_NOT_IMPLEMENTED << "\"" << NAME << "\" is not implemented for "     \
                           << "\"" << _dtype_a << "\" x \"" << _dtype_b        \
                           << "\"";                                            \
    }                                                                          \
  } while (0)

#define HT_DISPATH_PAIRED_CASE_FLOATING_TYPES(SPEC_A, SPEC_B, ...)             \
  HT_DISPATH_PAIRED_CASE(hetu::DataType::FLOAT16, hetu::DataType::BFLOAT16,    \
                         SPEC_A, SPEC_B, __VA_ARGS__)                          \
  HT_DISPATH_PAIRED_CASE(hetu::DataType::FLOAT16, hetu::DataType::FLOAT16,     \
                         SPEC_A, SPEC_B, __VA_ARGS__)                          \
  HT_DISPATH_PAIRED_CASE(hetu::DataType::FLOAT16, hetu::DataType::FLOAT32,     \
                         SPEC_A, SPEC_B, __VA_ARGS__)                          \
  HT_DISPATH_PAIRED_CASE(hetu::DataType::BFLOAT16, hetu::DataType::FLOAT16,    \
                         SPEC_A, SPEC_B, __VA_ARGS__)                          \
  HT_DISPATH_PAIRED_CASE(hetu::DataType::BFLOAT16, hetu::DataType::BFLOAT16,    \
                         SPEC_A, SPEC_B, __VA_ARGS__)                          \
  HT_DISPATH_PAIRED_CASE(hetu::DataType::BFLOAT16, hetu::DataType::FLOAT32,    \
                         SPEC_A, SPEC_B, __VA_ARGS__)                          \
  HT_DISPATH_PAIRED_CASE(hetu::DataType::FLOAT32, hetu::DataType::BFLOAT16,     \
                         SPEC_A, SPEC_B, __VA_ARGS__)                          \
  HT_DISPATH_PAIRED_CASE(hetu::DataType::FLOAT32, hetu::DataType::FLOAT16,     \
                         SPEC_A, SPEC_B, __VA_ARGS__)                          \
  HT_DISPATH_PAIRED_CASE(hetu::DataType::FLOAT32, hetu::DataType::FLOAT32,     \
                         SPEC_A, SPEC_B, __VA_ARGS__)                          \
  HT_DISPATH_PAIRED_CASE(hetu::DataType::FLOAT32, hetu::DataType::FLOAT64,     \
                         SPEC_A, SPEC_B, __VA_ARGS__)                          \
  HT_DISPATH_PAIRED_CASE(hetu::DataType::FLOAT64, hetu::DataType::FLOAT32,     \
                         SPEC_A, SPEC_B, __VA_ARGS__)                          \
  HT_DISPATH_PAIRED_CASE(hetu::DataType::FLOAT64, hetu::DataType::FLOAT64,     \
                         SPEC_A, SPEC_B, __VA_ARGS__)

#define HT_DISPATCH_PAIRED_FLOATING_TYPES(DTYPE_A, DTYPE_B, SPEC_A, SPEC_B,    \
                                          NAME, ...)                           \
  HT_DISPATH_PAIRED_SWITCH(                                                    \
    DTYPE_A, DTYPE_B, NAME,                                                    \
    HT_DISPATH_PAIRED_CASE_FLOATING_TYPES(SPEC_A, SPEC_B, __VA_ARGS__))

#define HT_DISPATH_PAIRED_CASE_SIGNED_INTEGER_TYPES(SPEC_A, SPEC_B, ...)       \
  HT_DISPATH_PAIRED_CASE(hetu::DataType::INT8, hetu::DataType::INT8, SPEC_A,   \
                         SPEC_B, __VA_ARGS__)                                  \
  HT_DISPATH_PAIRED_CASE(hetu::DataType::INT8, hetu::DataType::INT16, SPEC_A,  \
                         SPEC_B, __VA_ARGS__)                                  \
  HT_DISPATH_PAIRED_CASE(hetu::DataType::INT8, hetu::DataType::INT32, SPEC_A,  \
                         SPEC_B, __VA_ARGS__)                                  \
  HT_DISPATH_PAIRED_CASE(hetu::DataType::INT8, hetu::DataType::INT64, SPEC_A,  \
                         SPEC_B, __VA_ARGS__)                                  \
  HT_DISPATH_PAIRED_CASE(hetu::DataType::INT16, hetu::DataType::INT8, SPEC_A,  \
                         SPEC_B, __VA_ARGS__)                                  \
  HT_DISPATH_PAIRED_CASE(hetu::DataType::INT16, hetu::DataType::INT16, SPEC_A, \
                         SPEC_B, __VA_ARGS__)                                  \
  HT_DISPATH_PAIRED_CASE(hetu::DataType::INT16, hetu::DataType::INT32, SPEC_A, \
                         SPEC_B, __VA_ARGS__)                                  \
  HT_DISPATH_PAIRED_CASE(hetu::DataType::INT16, hetu::DataType::INT64, SPEC_A, \
                         SPEC_B, __VA_ARGS__)                                  \
  HT_DISPATH_PAIRED_CASE(hetu::DataType::INT32, hetu::DataType::INT8, SPEC_A,  \
                         SPEC_B, __VA_ARGS__)                                  \
  HT_DISPATH_PAIRED_CASE(hetu::DataType::INT32, hetu::DataType::INT16, SPEC_A, \
                         SPEC_B, __VA_ARGS__)                                  \
  HT_DISPATH_PAIRED_CASE(hetu::DataType::INT32, hetu::DataType::INT32, SPEC_A, \
                         SPEC_B, __VA_ARGS__)                                  \
  HT_DISPATH_PAIRED_CASE(hetu::DataType::INT32, hetu::DataType::INT64, SPEC_A, \
                         SPEC_B, __VA_ARGS__)                                  \
  HT_DISPATH_PAIRED_CASE(hetu::DataType::INT64, hetu::DataType::INT8, SPEC_A,  \
                         SPEC_B, __VA_ARGS__)                                  \
  HT_DISPATH_PAIRED_CASE(hetu::DataType::INT64, hetu::DataType::INT16, SPEC_A, \
                         SPEC_B, __VA_ARGS__)                                  \
  HT_DISPATH_PAIRED_CASE(hetu::DataType::INT64, hetu::DataType::INT32, SPEC_A, \
                         SPEC_B, __VA_ARGS__)                                  \
  HT_DISPATH_PAIRED_CASE(hetu::DataType::INT64, hetu::DataType::INT64, SPEC_A, \
                         SPEC_B, __VA_ARGS__)

#define HT_DISPATCH_PAIRED_SIGNED_INTEGER_TYPES(DTYPE_A, DTYPE_B, SPEC_A,      \
                                                SPEC_B, NAME, ...)             \
  HT_DISPATH_PAIRED_SWITCH(                                                    \
    DTYPE_A, DTYPE_B, NAME,                                                    \
    HT_DISPATH_PAIRED_CASE_SIGNED_INTEGER_TYPES(SPEC_A, SPEC_B, __VA_ARGS__))

#define HT_DISPATH_PAIRED_CASE_SIGNED_INTEGER_AND_FLOATING_TYPES(SPEC_A,       \
                                                                 SPEC_B, ...)  \
  HT_DISPATH_PAIRED_CASE_FLOATING_TYPES(SPEC_A, SPEC_B, __VA_ARGS__)           \
  HT_DISPATH_PAIRED_CASE_SIGNED_INTEGER_TYPES(SPEC_A, SPEC_B, __VA_ARGS__)     \
  HT_DISPATH_PAIRED_CASE(hetu::DataType::INT8, hetu::DataType::FLOAT32,        \
                         SPEC_A, SPEC_B, __VA_ARGS__)                          \
  HT_DISPATH_PAIRED_CASE(hetu::DataType::INT8, hetu::DataType::FLOAT64,        \
                         SPEC_A, SPEC_B, __VA_ARGS__)                          \
  HT_DISPATH_PAIRED_CASE(hetu::DataType::INT16, hetu::DataType::FLOAT32,       \
                         SPEC_A, SPEC_B, __VA_ARGS__)                          \
  HT_DISPATH_PAIRED_CASE(hetu::DataType::INT16, hetu::DataType::FLOAT64,       \
                         SPEC_A, SPEC_B, __VA_ARGS__)                          \
  HT_DISPATH_PAIRED_CASE(hetu::DataType::INT32, hetu::DataType::FLOAT32,       \
                         SPEC_A, SPEC_B, __VA_ARGS__)                          \
  HT_DISPATH_PAIRED_CASE(hetu::DataType::INT32, hetu::DataType::FLOAT64,       \
                         SPEC_A, SPEC_B, __VA_ARGS__)                          \
  HT_DISPATH_PAIRED_CASE(hetu::DataType::INT64, hetu::DataType::FLOAT32,       \
                         SPEC_A, SPEC_B, __VA_ARGS__)                          \
  HT_DISPATH_PAIRED_CASE(hetu::DataType::INT64, hetu::DataType::FLOAT64,       \
                         SPEC_A, SPEC_B, __VA_ARGS__)                          \
  HT_DISPATH_PAIRED_CASE(hetu::DataType::FLOAT32, hetu::DataType::INT8,        \
                         SPEC_A, SPEC_B, __VA_ARGS__)                          \
  HT_DISPATH_PAIRED_CASE(hetu::DataType::FLOAT32, hetu::DataType::INT16,       \
                         SPEC_A, SPEC_B, __VA_ARGS__)                          \
  HT_DISPATH_PAIRED_CASE(hetu::DataType::FLOAT32, hetu::DataType::INT32,       \
                         SPEC_A, SPEC_B, __VA_ARGS__)                          \
  HT_DISPATH_PAIRED_CASE(hetu::DataType::FLOAT32, hetu::DataType::INT64,       \
                         SPEC_A, SPEC_B, __VA_ARGS__)                          \
  HT_DISPATH_PAIRED_CASE(hetu::DataType::FLOAT64, hetu::DataType::INT8,        \
                         SPEC_A, SPEC_B, __VA_ARGS__)                          \
  HT_DISPATH_PAIRED_CASE(hetu::DataType::FLOAT64, hetu::DataType::INT16,       \
                         SPEC_A, SPEC_B, __VA_ARGS__)                          \
  HT_DISPATH_PAIRED_CASE(hetu::DataType::FLOAT64, hetu::DataType::INT32,       \
                         SPEC_A, SPEC_B, __VA_ARGS__)                          \
  HT_DISPATH_PAIRED_CASE(hetu::DataType::FLOAT64, hetu::DataType::INT64,       \
                         SPEC_A, SPEC_B, __VA_ARGS__)

#define HT_DISPATCH_PAIRED_SIGNED_INTEGER_AND_FLOATING_TYPES(                  \
  DTYPE_A, DTYPE_B, SPEC_A, SPEC_B, NAME, ...)                                 \
  HT_DISPATH_PAIRED_SWITCH(                                                    \
    DTYPE_A, DTYPE_B, NAME,                                                    \
    HT_DISPATH_PAIRED_CASE_SIGNED_INTEGER_AND_FLOATING_TYPES(SPEC_A, SPEC_B,   \
                                                             __VA_ARGS__))

/******************************************************
 * Dispatch Utils for Triple Types.
 * Now Just Support Float Type and Spec_C = FP32.
 * Just for Mix Precision Binary Ops.
 ******************************************************/

namespace hetu {
namespace __triple_dtype_anon {
constexpr int triple_dtype(DataType t1, DataType t2, DataType t3) {
  return (static_cast<int>(t1) << 16) | 
         (static_cast<int>(t2) << 8) |
         static_cast<int>(t3);
};
} // namespace __triple_dtype_anon
} // namespace hetu

#define HT_DISPATH_TRIPLE_CASE(DTYPE_A, DTYPE_B, DTYPE_C, SPEC_A, SPEC_B, SPEC_C, ...)         \
  case hetu::__triple_dtype_anon::triple_dtype(DTYPE_A, DTYPE_B, DTYPE_C): {                   \
    using SPEC_A = hetu::DataType2SpecMeta<DTYPE_A>::spec_type;                                \
    using SPEC_B = hetu::DataType2SpecMeta<DTYPE_B>::spec_type;                                \
    using SPEC_C = hetu::DataType2SpecMeta<DTYPE_C>::spec_type;                                \
    __VA_ARGS__();                                                                             \
    break;                                                                                     \
  }

#define HT_DISPATH_TRIPLE_SWITCH(DTYPE_A, DTYPE_B, DTYPE_C, NAME, ...)                   \
  do {                                                                                   \
    const auto& _dtype_a = DTYPE_A;                                                      \
    const auto& _dtype_b = DTYPE_B;                                                      \
    const auto& _dtype_c = DTYPE_C;                                                      \
    switch (hetu::__triple_dtype_anon::triple_dtype(_dtype_a, _dtype_b, _dtype_c)) {     \
      __VA_ARGS__                                                                        \
      default:                                                                           \
        HT_NOT_IMPLEMENTED << "\"" << NAME << "\" is not implemented for "               \
                           << "\"" << _dtype_a << "\" x \"" << _dtype_b                  \
                           << "\" x \"" << _dtype_c << "\"";                             \
    }                                                                                    \
  } while (0)

#define HT_DISPATH_TRIPLE_CASE_FLOATING_TYPES(SPEC_A, SPEC_B, SPEC_C, ...)                              \
  HT_DISPATH_TRIPLE_CASE(hetu::DataType::FLOAT32, hetu::DataType::BFLOAT16, hetu::DataType::FLOAT32,    \
                         SPEC_A, SPEC_B, SPEC_C, __VA_ARGS__)                                           \
  HT_DISPATH_TRIPLE_CASE(hetu::DataType::FLOAT32, hetu::DataType::FLOAT16, hetu::DataType::FLOAT32,     \
                         SPEC_A, SPEC_B, SPEC_C, __VA_ARGS__)                                           \
  HT_DISPATH_TRIPLE_CASE(hetu::DataType::FLOAT16, hetu::DataType::FLOAT16, hetu::DataType::FLOAT32,     \
                         SPEC_A, SPEC_B, SPEC_C, __VA_ARGS__)                                           \
  HT_DISPATH_TRIPLE_CASE(hetu::DataType::BFLOAT16, hetu::DataType::BFLOAT16, hetu::DataType::FLOAT32,   \
                         SPEC_A, SPEC_B, SPEC_C, __VA_ARGS__)                                           \
  HT_DISPATH_TRIPLE_CASE(hetu::DataType::BFLOAT16, hetu::DataType::FLOAT32, hetu::DataType::FLOAT32,    \
                         SPEC_A, SPEC_B, SPEC_C, __VA_ARGS__)                                           \
  HT_DISPATH_TRIPLE_CASE(hetu::DataType::FLOAT16, hetu::DataType::FLOAT32, hetu::DataType::FLOAT32,     \
                         SPEC_A, SPEC_B, SPEC_C, __VA_ARGS__)                                           

#define HT_DISPATCH_TRIPLE_FLOATING_TYPES(DTYPE_A, DTYPE_B, DTYPE_C, SPEC_A, SPEC_B, SPEC_C, NAME, ...) \
  HT_DISPATH_TRIPLE_SWITCH(                                                                             \
    DTYPE_A, DTYPE_B, DTYPE_C, NAME,                                                                    \
    HT_DISPATH_TRIPLE_CASE_FLOATING_TYPES(SPEC_A, SPEC_B, SPEC_C, __VA_ARGS__))
