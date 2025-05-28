#include "hetu/impl/utils/cuda_utils.h"
#include "hetu/impl/stream/CUDAStream.h"
#include "hetu/impl/utils/offset_calculator.cuh"

namespace hetu {
namespace impl {

namespace {

static std::once_flag offset_calculator_init_flag;
static NDArray trivial_offset_calculator_arr;
static OffsetCalculator* trivial_offset_calculator;
static LFUCache lfu_cache;

static void OffCalcInitOnce(const Stream& stream) {
  std::call_once(offset_calculator_init_flag, [](const Stream& stream) {
    lfu_cache = LFUCache(HT_LFU_CAPACITY);
    CUDAStream cuda_stream(stream);
    trivial_offset_calculator_arr =
      NDArray::empty({static_cast<int64_t>(sizeof(OffsetCalculator))},
                      Device(kCUDA, stream.device_index()), kByte, stream.stream_index());
    trivial_offset_calculator =
      trivial_offset_calculator_arr->data_ptr<OffsetCalculator>();
    hetu::cuda::CUDADeviceGuard guard(stream.device_index());
    trivial_constructor<<<1, 1, 0, cuda_stream>>>(trivial_offset_calculator);
  }, stream);
}

} // namespace

std::tuple<NDArray, OffsetCalculator*>
AllocOffsetCalculator(const NDArray& arr, const Stream& stream) {
  OffCalcInitOnce(stream);
  if (arr->is_contiguous()) {
    return {trivial_offset_calculator_arr, trivial_offset_calculator};
  }
  NDArray offset_calculator_arr;
  StridedOffsetCalculator* offset_calculator;
  auto shape_arr_host = arr->shape();
  auto stride_arr_host = arr->stride();
  std::tie(offset_calculator_arr, offset_calculator) =
      lfu_cache.get(shape_arr_host, stride_arr_host);
  if (!offset_calculator) {
    auto device_id = arr->device().index();
    CUDAStream cuda_stream(stream);
    size_t ndim = arr->ndim();
    size_t alloc_size = ndim * sizeof(int64_t);
    HTShape offset_calculator_meta = arr->shape();
    offset_calculator_meta.insert(offset_calculator_meta.end(),
                                  arr->stride().begin(), arr->stride().end());
    auto offset_calculator_meta_arr =
      hetu::cuda::to_int64_ndarray(offset_calculator_meta, device_id);
    int64_t *shape_ptr = offset_calculator_meta_arr->data_ptr<int64_t>();
    int64_t *stride_ptr = shape_ptr + arr->shape().size();
    offset_calculator_arr = NDArray::empty({static_cast<int64_t>(sizeof(StridedOffsetCalculator))},
                                           Device(kCUDA, device_id), kByte, stream.stream_index());
    offset_calculator = offset_calculator_arr->data_ptr<StridedOffsetCalculator>();
    hetu::cuda::CUDADeviceGuard guard(device_id);
    strided_constructor<<<1, 1, 0, cuda_stream>>>(offset_calculator, ndim,
                                                  shape_ptr, stride_ptr);
    NDArray::MarkUsedBy({offset_calculator_meta_arr}, stream);
    lfu_cache.put(shape_arr_host, stride_arr_host, offset_calculator_arr, offset_calculator);
  }
  return {offset_calculator_arr, offset_calculator};
}

// TODO: merge all allocators into once
std::tuple<NDArrayList, std::vector<OffsetCalculator*>>
AllocOffsetCalculator(const NDArrayList& arr_list, const Stream& stream) {
  OffCalcInitOnce(stream);
  size_t arr_num = arr_list.size();
  NDArrayList offset_calculator_arr_list(arr_num);
  std::vector<OffsetCalculator*> offset_calculator_list(arr_num);

  NDArrayList non_contig_arr_list;
  std::vector<size_t> non_contig_arr_idx;
  for (auto i = 0; i < arr_num; i++) {
    auto& arr = arr_list[i];
    if (arr->is_contiguous()) {
      offset_calculator_arr_list[i] = trivial_offset_calculator_arr;
      offset_calculator_list[i] = trivial_offset_calculator;
    } else {
      NDArray offset_calculator_arr;
      StridedOffsetCalculator* offset_calculator;
      auto shape_arr_host = arr->shape();
      auto stride_arr_host = arr->stride();
      std::tie(offset_calculator_arr, offset_calculator) =
          lfu_cache.get(shape_arr_host, stride_arr_host);
      if (offset_calculator) {
        offset_calculator_arr_list[i] = offset_calculator_arr;
        offset_calculator_list[i] = offset_calculator;
      } else {
        non_contig_arr_list.push_back(arr);
        non_contig_arr_idx.push_back(i);
      }
    }
  }
  if (non_contig_arr_list.empty()) {
    return {offset_calculator_arr_list, offset_calculator_list};
  }
  size_t non_contig_arr_num = non_contig_arr_list.size();
  for (auto i = 0; i < non_contig_arr_num; i++) {
    auto& arr = non_contig_arr_list[i];
    auto idx = non_contig_arr_idx[i];
    auto shape_arr_host = arr->shape();
    auto stride_arr_host = arr->stride();
    std::tie(offset_calculator_arr_list[idx], offset_calculator_list[idx]) =
        lfu_cache.get(shape_arr_host, stride_arr_host);
    if (!offset_calculator_list[idx]) {
      NDArray offset_calculator_arr;
      StridedOffsetCalculator* offset_calculator;
      auto device_id = arr->device().index();
      CUDAStream cuda_stream(stream);
      size_t ndim = arr->ndim();
      size_t alloc_size = ndim * sizeof(int64_t);
      HTShape offset_calculator_meta = arr->shape();
      offset_calculator_meta.insert(offset_calculator_meta.end(),
                                    arr->stride().begin(), arr->stride().end());
      auto offset_calculator_meta_arr =
        hetu::cuda::to_int64_ndarray(offset_calculator_meta, device_id);
      int64_t *shape_ptr = offset_calculator_meta_arr->data_ptr<int64_t>();
      int64_t *stride_ptr = shape_ptr + arr->shape().size();
      offset_calculator_arr = NDArray::empty({static_cast<int64_t>(sizeof(StridedOffsetCalculator))},
                                            Device(kCUDA, device_id), kByte, stream.stream_index());
      offset_calculator = offset_calculator_arr->data_ptr<StridedOffsetCalculator>();
      hetu::cuda::CUDADeviceGuard guard(device_id);
      strided_constructor<<<1, 1, 0, cuda_stream>>>(offset_calculator, ndim,
                                                    shape_ptr, stride_ptr);
      NDArray::MarkUsedBy({offset_calculator_meta_arr}, stream);
      lfu_cache.put(shape_arr_host, stride_arr_host, offset_calculator_arr, offset_calculator);
      offset_calculator_arr_list[idx] = offset_calculator_arr;
      offset_calculator_list[idx] = offset_calculator;
    }
  }
  return {offset_calculator_arr_list, offset_calculator_list};
}

} // namespace impl
} // namespace hetu