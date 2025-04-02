#include "hetu/impl/stream/CUDAStream.h"
#include <mutex>

namespace hetu {
namespace impl {

namespace {

static int32_t num_devices;
static std::once_flag global_init_flag;
static std::once_flag device_init_flags[HT_MAX_GPUS_COMPILE_TIME];
static std::vector<int> device_initialized(HT_MAX_GPUS_COMPILE_TIME, 0);
static std::vector<std::vector<cudaStream_t>>
  device_streams(HT_MAX_GPUS_COMPILE_TIME,
                 std::vector<cudaStream_t>(HT_NUM_STREAMS_PER_DEVICE));

static void InitGlobal() {
  CudaGetDeviceCount(&num_devices);
  HT_ASSERT_LE(num_devices, HT_MAX_GPUS_COMPILE_TIME)
    << "Too many devices are provided: " << num_devices << ". "
    << "Expected " << HT_MAX_GPUS_COMPILE_TIME << " or fewer.";
}

inline static void InitGlobalOnce() {
  std::call_once(global_init_flag, InitGlobal);
}

static void InitDevice(int32_t device_id) {
  InitGlobalOnce();
  HT_ASSERT_LT(device_id, num_devices)
    << "Device id " << device_id << " exceeds number of available devices "
    << num_devices;
  hetu::cuda::CUDADeviceGuard guard(device_id);
  for (int32_t i = 0; i < HT_NUM_STREAMS_PER_DEVICE; i++) {
    if (i == 0) {
      device_streams[device_id][i] = cudaStreamLegacy;
    }
    else {
      CudaStreamCreateWithPriority(
        &device_streams[device_id][i], cudaStreamDefault, 0);
    }
  }
  device_initialized[device_id] = 1;
}

inline static void InitDeviceOnce(int32_t device_id) {
  std::call_once(device_init_flags[device_id], InitDevice, device_id);
}

} // namespace

CUDAStream::CUDAStream(const Stream& stream)
: _device_id{stream.device_index()}, _stream_id{stream.stream_index()} {
  HT_ASSERT(stream.device().is_cuda())
    << "Initializing CUDA stream "
    << "for non-cuda device: " << stream.device();
  HT_ASSERT(_stream_id >= kBlockingStream &&
            _stream_id < HT_NUM_STREAMS_PER_DEVICE)
    << "Invalid device stream id: " << _stream_id;
  if (!device_initialized[_device_id]) {
    InitGlobalOnce();
    InitDeviceOnce(_device_id);
  }
}

cudaStream_t CUDAStream::cuda_stream() const noexcept {
  return device_streams[_device_id][_stream_id];
}

int GetCUDADeiceCount() {
  InitGlobalOnce();
  return num_devices;
}

void SynchronizeAllCUDAStreams(const Device& device) {
  InitGlobalOnce();
  if (device.is_undetermined()) {
    for (int32_t device_id = 0; device_id < num_devices; device_id++) {
      if (!device_initialized[device_id])
        continue;
      for (size_t j = 0; j < device_streams[device_id].size(); j++)
        if (device_streams[device_id][j])
          CUDAStream(Stream(Device(kCUDA, device_id), j)).Sync();
    }
  } else {
    auto device_id = device.index();
    if (!device_initialized[device_id])
      return;
    for (size_t j = 0; j < device_streams[device_id].size(); j++)
      if (device_streams[device_id][j])
        CUDAStream(Stream(Device(kCUDA, device_id), j)).Sync();
  }
}

} // namespace impl
} // namespace hetu
