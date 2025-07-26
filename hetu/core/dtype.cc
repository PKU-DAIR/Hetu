#include "hetu/core/dtype.h"

namespace hetu {

size_t DataType2Size(const DataType& t) {
#define __HT_DATA_TYPE_CASE_RETURN_SIZE(TYPE)                                  \
  HT_DATA_TYPE_CASE_RETURN(TYPE, sizeof(DataType2SpecMeta<TYPE>::spec_type))

  switch (t) {
    __HT_DATA_TYPE_CASE_RETURN_SIZE(DataType::UINT8);
    __HT_DATA_TYPE_CASE_RETURN_SIZE(DataType::INT8);
    __HT_DATA_TYPE_CASE_RETURN_SIZE(DataType::INT16);
    __HT_DATA_TYPE_CASE_RETURN_SIZE(DataType::INT32);
    __HT_DATA_TYPE_CASE_RETURN_SIZE(DataType::INT64);
    __HT_DATA_TYPE_CASE_RETURN_SIZE(DataType::FLOAT16);
    __HT_DATA_TYPE_CASE_RETURN_SIZE(DataType::FLOAT32);
    __HT_DATA_TYPE_CASE_RETURN_SIZE(DataType::FLOAT64);
    __HT_DATA_TYPE_CASE_RETURN_SIZE(DataType::BFLOAT16);
    __HT_DATA_TYPE_CASE_RETURN_SIZE(DataType::FLOAT4);
    __HT_DATA_TYPE_CASE_RETURN_SIZE(DataType::NFLOAT4);
    __HT_DATA_TYPE_CASE_RETURN_SIZE(DataType::BOOL); // TODO: handle boolean
    case kUndeterminedDataType:
      HT_RUNTIME_ERROR << "Cannot get size when data type is undetermined";
      __builtin_unreachable();
    default:
      HT_VALUE_ERROR << "Unknown data type: " << static_cast<int16_t>(t);
      __builtin_unreachable();
  }
}

std::string DataType2Str(const DataType& t) {
#define __HT_DATA_TYPE_CASE_RETURN_STR(TYPE)                                   \
  HT_DATA_TYPE_CASE_RETURN(TYPE, DataType2SpecMeta<TYPE>::str())
  switch (t) {
    __HT_DATA_TYPE_CASE_RETURN_STR(DataType::UINT8);
    __HT_DATA_TYPE_CASE_RETURN_STR(DataType::INT8);
    __HT_DATA_TYPE_CASE_RETURN_STR(DataType::INT16);
    __HT_DATA_TYPE_CASE_RETURN_STR(DataType::INT32);
    __HT_DATA_TYPE_CASE_RETURN_STR(DataType::INT64);
    __HT_DATA_TYPE_CASE_RETURN_STR(DataType::FLOAT16);
    __HT_DATA_TYPE_CASE_RETURN_STR(DataType::FLOAT32);
    __HT_DATA_TYPE_CASE_RETURN_STR(DataType::FLOAT64);
    __HT_DATA_TYPE_CASE_RETURN_STR(DataType::BFLOAT16);
    __HT_DATA_TYPE_CASE_RETURN_STR(DataType::FLOAT4);
    __HT_DATA_TYPE_CASE_RETURN_STR(DataType::NFLOAT4);
    __HT_DATA_TYPE_CASE_RETURN_STR(DataType::BOOL);
    case kUndeterminedDataType: return "undertermined";
    default:
      HT_VALUE_ERROR << "Unknown data type: " << static_cast<int16_t>(t);
      __builtin_unreachable();
  }
}

std::ostream& operator<<(std::ostream& os, const DataType& data_type) {
  os << DataType2Str(data_type);
  return os;
}

bool IsFloatingPoint(const DataType& t) {
  switch (t) {
    case DataType::FLOAT16:
    case DataType::FLOAT32:
    case DataType::FLOAT64:
    case DataType::BFLOAT16:
    case DataType::FLOAT4:
    case DataType::NFLOAT4:
      return true;
    default:
      return false;
  }
}

} // namespace hetu
