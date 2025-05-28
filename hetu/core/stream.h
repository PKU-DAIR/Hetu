#pragma once

#include "hetu/core/device.h"

namespace hetu {

using StreamIndex = int32_t;
constexpr int32_t HT_NUM_STREAMS_PER_DEVICE = 1 << 4;
constexpr StreamIndex kUndeterminedStream = -1;
constexpr StreamIndex kBlockingStream = 0;
constexpr StreamIndex kComputingStream = 1;
constexpr StreamIndex kSwitchComputingStream = 2;
constexpr StreamIndex kH2DStream = 3;
constexpr StreamIndex kD2HStream = 4;
constexpr StreamIndex kP2PStream = 5;
constexpr StreamIndex kCollectiveStream = 6;
constexpr StreamIndex kSwitchCollectiveStream = 7;
constexpr StreamIndex kBridgeStream = 8;
constexpr StreamIndex kOffloadStream = 9;
constexpr StreamIndex kJoinStream = HT_NUM_STREAMS_PER_DEVICE - 1;

using PackedStreamId = uint16_t;

class Stream {
 public:
  Stream(Device device = Device(), StreamIndex id = kBlockingStream)
  : _device(device), _id(id) {
    HT_ASSERT(_device.local())
      << "Cannot create stream for remote device: " << _device;
    HT_ASSERT(_id >= kBlockingStream && _id < HT_NUM_STREAMS_PER_DEVICE)
      << "Invalid stream index: " << _id;
  }

  Stream(const Stream&) = default;
  Stream(Stream&&) = default;
  Stream& operator=(const Stream&) = default;
  Stream& operator=(Stream&&) = default;

  void Sync() const;

  inline PackedStreamId pack() const noexcept {
    // pack: stream id (8 bits) || device type (4 bits) || device id (4 bits)
    static_assert(HT_NUM_STREAMS_PER_DEVICE <= 256,
                  "Device index cannot be packed into 8 bits");
    static_assert(NUM_DEVICE_TYPES <= 16,
                  "Device type cannot be packed into 4 bits");
    static_assert(HT_MAX_DEVICE_INDEX <= 16,
                  "Device index cannot be packed into 4 bits");
    return ((static_cast<uint16_t>(stream_index()) & 0xFF) << 8) |
      ((static_cast<uint16_t>(device_type()) & 0xF) << 4) |
      ((static_cast<uint16_t>(device_index()) & 0xF));
  }

  static Stream unpack(PackedStreamId packed) {
    auto device_type = static_cast<DeviceType>((packed & 0xF0) >> 4);
    auto device_index = static_cast<DeviceIndex>(packed & 0xF);
    auto stream_index = static_cast<StreamIndex>((packed & 0xFF00) >> 8);
    return Stream(Device(device_type, device_index), stream_index);
  }

  inline const Device& device() const noexcept {
    return _device;
  }

  inline DeviceType device_type() const noexcept {
    return _device.type();
  }

  inline DeviceIndex device_index() const noexcept {
    return _device.index();
  }

  inline StreamIndex stream_index() const noexcept {
    return _id;
  }

  inline bool is_defined() const noexcept {
    return !_device.is_undetermined();
  }

  inline bool is_blocking() const noexcept {
    return is_defined() && stream_index() == kBlockingStream;
  }

  inline bool operator==(const Stream& stream) const {
    return _device == stream._device && _id == stream._id;
  }

  inline bool operator!=(const Stream& stream) const {
    return !operator==(stream);
  }

 private:
  Device _device;
  StreamIndex _id;
};

void SynchronizeAllStreams(const Device& device = Device());

std::ostream& operator<<(std::ostream&, const Stream&);

class Event {
 public:
  Event(Device device, bool enable_timing = true)
  : _device(std::move(device)), _enable_timing(enable_timing) {}

  virtual bool IsRecorded() = 0;

  virtual void Record(const Stream& stream) = 0;

  virtual void Sync() = 0;

  virtual void Block(const Stream& stream) = 0;

  virtual int64_t TimeSince(const Event& event) const = 0;

  inline const Device& device() const {
    return _device;
  }
  
  inline bool enable_timing() const {
    return _enable_timing;
  }

 protected:
  const Device _device;
  const bool _enable_timing;
};

} // namespace hetu

namespace std {
template <>
struct hash<hetu::Stream> {
  std::size_t operator()(const hetu::Stream& stream) const noexcept {
    // Following boost::hash_combine
    auto hash = std::hash<hetu::StreamIndex>()(stream.stream_index());
    hash ^= (std::hash<hetu::Device>()(stream.device()) + 0x9e3779b9 +
             (hash << 6) + (hash >> 2));
    return hash;
  }
};

inline std::string to_string(const hetu::Stream& stream) {
  std::ostringstream os;
  os << stream;
  return os.str();
}

} // namespace std
