#pragma once

#include <iostream>
#include <memory>
#include <string>
#include "hetu/utils/json/json.hpp"

using json = nlohmann::json;

namespace hetu {

struct DeviceInfoReply {
  int type;
  int index;
  int multiplex;
};

class DeviceClientImpl {
 public:
  DeviceClientImpl() {}

  virtual int Connect(const std::string& hostname) {}

  virtual std::pair<int, int> GetRank(const std::string& user) {}

  virtual int CommitHostName(const std::string& hostname, int rank) {}

  virtual std::string GetHostName(int rank) {}

  virtual int CommitDeviceInfo(int type, int index, int multiplex, int rank) {}

  virtual DeviceInfoReply GetDeviceInfo(int rank) {}

  virtual int CommitNcclId(const std::string& nccl_id, const std::vector<int>& world_rank, int stream_id) {}

  virtual std::string GetNcclId(const std::vector<int>& world_rank, int stream_id) {}

  virtual int Exit(int rank) {}

  virtual int PutDouble(const std::string& key, double value) {}

  virtual double GetDouble(const std::string& key) {}

  virtual std::string RemoveDouble(const std::string& key) {}

  virtual int PutInt(const std::string& key, int64_t value) {}

  virtual int64_t GetInt(const std::string& key) {}

  virtual std::string RemoveInt(const std::string& key) {}

  virtual int PutString(const std::string& key, const std::string& value) {}

  virtual std::string GetString(const std::string& key) {}

  virtual std::string RemoveString(const std::string& key) {}

  virtual int PutBytes(const std::string& key, const std::string& value) {}

  virtual std::string GetBytes(const std::string& key) {}

  virtual std::string RemoveBytes(const std::string& key) {}

  virtual int PutJson(const std::string& key, const json& value) {}

  virtual json GetJson(const std::string& key) {}

  virtual std::string RemoveJson(const std::string& key) {}

  virtual int Barrier(int rank, const std::vector<int>& world_rank) {}

  virtual int Consistent(int rank, int value, const std::vector<int>& world_rank) {}

  virtual int HeartBeat(int rank) {}

  virtual void LaunchHeartBeat(int rank) {};
};

} //namespace hetu
