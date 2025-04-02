#pragma once

#include "hetu/common/macros.h"
#include "hetu/core/float16.h"
#include "hetu/core/bfloat16.h"

namespace hetu {

enum class DataType : int8_t {
  UINT8 = 0,
  INT8,
  INT16,
  INT32,
  INT64,
  FLOAT16,
  FLOAT32,
  FLOAT64,
  BFLOAT16,
  FLOAT4,
  NFLOAT4,
  BOOL,
  NUM_DATA_TYPES,
  UNDETERMINED
};

template <DataType DTYPE>
struct DataType2SpecMeta;

#define DECLARE_DATA_TYPE_TO_SPECIALIZED_META(DTYPE, SPEC_TYPE, NAME)          \
  template <>                                                                  \
  struct DataType2SpecMeta<DTYPE> {                                            \
    using spec_type = SPEC_TYPE; /* the specialized type */                    \
    static inline std::string str() { return quote(NAME); }                    \
  }

DECLARE_DATA_TYPE_TO_SPECIALIZED_META(DataType::UINT8, uint8_t, uint8);
DECLARE_DATA_TYPE_TO_SPECIALIZED_META(DataType::INT8, int8_t, int8);
DECLARE_DATA_TYPE_TO_SPECIALIZED_META(DataType::INT16, int16_t, int16);
DECLARE_DATA_TYPE_TO_SPECIALIZED_META(DataType::INT32, int32_t, int32);
DECLARE_DATA_TYPE_TO_SPECIALIZED_META(DataType::INT64, int64_t, int64);
// TODO: support half-precision floating points
DECLARE_DATA_TYPE_TO_SPECIALIZED_META(DataType::FLOAT16, float16, float16);
DECLARE_DATA_TYPE_TO_SPECIALIZED_META(DataType::FLOAT32, float, float32);
DECLARE_DATA_TYPE_TO_SPECIALIZED_META(DataType::FLOAT64, double, float64);
DECLARE_DATA_TYPE_TO_SPECIALIZED_META(DataType::BFLOAT16, bfloat16, bfloat16);
DECLARE_DATA_TYPE_TO_SPECIALIZED_META(DataType::FLOAT4, uint8_t, float4);
DECLARE_DATA_TYPE_TO_SPECIALIZED_META(DataType::NFLOAT4, uint8_t, nfloat4);
DECLARE_DATA_TYPE_TO_SPECIALIZED_META(DataType::BOOL, bool, bool);

#define HT_DATA_TYPE_CASE_RETURN(DTYPE, RETURN_VALUE)                          \
  case DTYPE: do { return RETURN_VALUE;                                        \
    } while (0)

constexpr DataType kUInt8 = DataType::UINT8;
constexpr DataType kByte = DataType::UINT8;
constexpr DataType kInt8 = DataType::INT8;
constexpr DataType kChar = DataType::INT8;
constexpr DataType kInt16 = DataType::INT16;
constexpr DataType kShort = DataType::INT16;
constexpr DataType kInt32 = DataType::INT32;
constexpr DataType kInt = DataType::INT32;
constexpr DataType kInt64 = DataType::INT64;
constexpr DataType kLong = DataType::INT64;
constexpr DataType kFloat16 = DataType::FLOAT16;
constexpr DataType kHalf = DataType::FLOAT16;
constexpr DataType kFloat32 = DataType::FLOAT32;
constexpr DataType kFloat = DataType::FLOAT32;
constexpr DataType kFloat64 = DataType::FLOAT64;
constexpr DataType kDouble = DataType::FLOAT64;
constexpr DataType kBFloat16 = DataType::BFLOAT16;
constexpr DataType kFloat4 = DataType::FLOAT4;
constexpr DataType kNFloat4 = DataType::NFLOAT4;
constexpr DataType kBool = DataType::BOOL;
constexpr DataType kUndeterminedDataType = DataType::UNDETERMINED;
constexpr int16_t NUM_DATA_TYPES =
  static_cast<int16_t>(DataType::NUM_DATA_TYPES);

size_t DataType2Size(const DataType&);
std::string DataType2Str(const DataType&);
std::ostream& operator<<(std::ostream&, const DataType&);
bool IsFloatingPoint(const DataType&);

} // namespace hetu

namespace std {
template <>
struct hash<hetu::DataType> {
  std::size_t operator()(hetu::DataType data_type) const noexcept {
    return std::hash<int>()(static_cast<int>(data_type));
  }
};

inline std::string to_string(hetu::DataType data_type) {
  return hetu::DataType2Str(data_type);
}
} // namespace std
