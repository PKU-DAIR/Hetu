#include "hetu/impl/communication/rpc_client.h"
#include "hetu/common/except.h"
#include "hetu/common/logging.h"

namespace hetu {

int DeviceClient::Connect(const std::string& hostname) {
  // Data we are sending to the server.
  ConnectRequest request;
  request.set_hostname(hostname);

  // Container for the data we expect from the server.
  ConnectReply reply;

  // Context for the client. It could be used to convey extra information to
  // the server and/or tweak certain RPC behaviors.
  ClientContext context;

  // The actual RPC.
  Status status = stub_->Connect(&context, request, &reply);

  // Act upon its status.
  if (status.ok()) {
    return reply.status();
  } else {
    HT_RUNTIME_ERROR << status.error_code() << ": " << status.error_message();
    __builtin_unreachable();
  }
}

std::pair<int, int> DeviceClient::GetRank(const std::string& user) {
  // Data we are sending to the server.
  RankRequest request;
  request.set_name(user);

  // Container for the data we expect from the server.
  RankReply reply;

  // Context for the client. It could be used to convey extra information to
  // the server and/or tweak certain RPC behaviors.
  ClientContext context;

  // The actual RPC.
  Status status = stub_->GetRank(&context, request, &reply);

  // Act upon its status.
  if (status.ok()) {
    return std::pair<int, int>(reply.rank(), reply.local_device());
  } else {
    HT_RUNTIME_ERROR << status.error_code() << ": " << status.error_message();
    __builtin_unreachable();
  }
}

int DeviceClient::CommitHostName(const std::string& hostname, int rank) {
  // Data we are sending to the server.
  CommitHostNameRequest request;
  request.set_hostname(hostname);
  request.set_rank(rank);

  // Container for the data we expect from the server.
  CommitHostNameReply reply;

  // Context for the client. It could be used to convey extra information to
  // the server and/or tweak certain RPC behaviors.
  ClientContext context;

  // The actual RPC.
  Status status = stub_->CommitHostName(&context, request, &reply);

  // Act upon its status.
  if (status.ok()) {
    return reply.status();
  } else {
    HT_RUNTIME_ERROR << status.error_code() << ": " << status.error_message();
    __builtin_unreachable();
  }
}

std::string DeviceClient::GetHostName(int rank) {
  // Data we are sending to the server.
  GetHostNameRequest request;
  request.set_rank(rank);

  // Container for the data we expect from the server.
  GetHostNameReply reply;

  // Context for the client. It could be used to convey extra information to
  // the server and/or tweak certain RPC behaviors.
  ClientContext context;

  // The actual RPC.
  Status status = stub_->GetHostName(&context, request, &reply);

  // Act upon its status.
  if (status.ok()) {
    return reply.hostname();
  } else {
    HT_RUNTIME_ERROR << status.error_code() << ": " << status.error_message();
    __builtin_unreachable();
  }
}

int DeviceClient::CommitDeviceInfo(int type, int index, int multiplex, int rank) {
  // Data we are sending to the server.
  CommitDeviceInfoRequest request;
  request.set_type(type);
  request.set_index(index);
  request.set_multiplex(multiplex);
  request.set_rank(rank);

  // Container for the data we expect from the server.
  CommitDeviceInfoReply reply;

  // Context for the client. It could be used to convey extra information to
  // the server and/or tweak certain RPC behaviors.
  ClientContext context;

  // The actual RPC.
  Status status = stub_->CommitDeviceInfo(&context, request, &reply);

  // Act upon its status.
  if (status.ok()) {
    return reply.status();
  } else {
    HT_RUNTIME_ERROR << status.error_code() << ": " << status.error_message();
    __builtin_unreachable();
  }
}

DeviceInfoReply DeviceClient::GetDeviceInfo(int rank) {
  // Data we are sending to the server.
  GetDeviceInfoRequest request;
  request.set_rank(rank);

  // Container for the data we expect from the server.
  GetDeviceInfoReply reply;

  // Context for the client. It could be used to convey extra information to
  // the server and/or tweak certain RPC behaviors.
  ClientContext context;

  // The actual RPC.
  Status status = stub_->GetDeviceInfo(&context, request, &reply);

  // Act upon its status.
  DeviceInfoReply out;
  if (status.ok()) {
    out.type = reply.type();
    out.index = reply.index();
    out.multiplex = reply.multiplex();
    return out;
  } else {
    HT_RUNTIME_ERROR << status.error_code() << ": " << status.error_message();
    __builtin_unreachable();
  }
}

int DeviceClient::CommitNcclId(const std::string& nccl_id, const std::vector<int>& world_rank, int stream_id) {
  // Data we are sending to the server.
  CommitNcclIdRequest request;
  request.set_nccl_id(nccl_id);
  for (auto rank: world_rank)
    request.add_world_rank(rank);
  request.set_stream_id(stream_id);

  // Container for the data we expect from the server.
  CommitNcclIdReply reply;

  // Context for the client. It could be used to convey extra information to
  // the server and/or tweak certain RPC behaviors.
  ClientContext context;

  // The actual RPC.
  Status status = stub_->CommitNcclId(&context, request, &reply);

  // Act upon its status.
  if (status.ok()) {
    return reply.status();
  } else {
    HT_RUNTIME_ERROR << status.error_code() << ": " << status.error_message();
    __builtin_unreachable();
  }
}

std::string DeviceClient::GetNcclId(const std::vector<int>& world_rank, int stream_id) {
  // Data we are sending to the server.
  GetNcclIdRequest request;
  for (auto rank: world_rank)
    request.add_world_rank(rank);
  request.set_stream_id(stream_id);

  // Container for the data we expect from the server.
  GetNcclIdReply reply;

  // Context for the client. It could be used to convey extra information to
  // the server and/or tweak certain RPC behaviors.
  ClientContext context;

  // The actual RPC.
  Status status = stub_->GetNcclId(&context, request, &reply);

  // Act upon its status.
  if (status.ok()) {
    return reply.nccl_id();
  } else {
    HT_RUNTIME_ERROR << status.error_code() << ": " << status.error_message();
    __builtin_unreachable();
  }
}

int DeviceClient::Exit(int rank) {
  // Data we are sending to the server.
  stop();
  ExitRequest request;
  request.set_rank(rank);

  // Container for the data we expect from the server.
  ExitReply reply;

  // Context for the client. It could be used to convey extra information to
  // the server and/or tweak certain RPC behaviors.
  ClientContext context;

  // The actual RPC.
  Status status = stub_->Exit(&context, request, &reply);

  // Act upon its status.
  if (status.ok()) {
    return reply.status();
  } else {
    HT_RUNTIME_ERROR << status.error_code() << ": " << status.error_message();
    __builtin_unreachable();
  }
}

int DeviceClient::PutDouble(const std::string& key, double value) {
  // Data we are sending to the server.
  PutDoubleRequest request;
  request.set_key(key);
  request.set_value(value);


  // Container for the data we expect from the server.
  PutDoubleReply reply;

  // Context for the client. It could be used to convey extra information to
  // the server and/or tweak certain RPC behaviors.
  ClientContext context;

  // The actual RPC.
  Status status = stub_->PutDouble(&context, request, &reply);

  // Act upon its status.
  if (status.ok()) {
    return reply.status();
  } else {
    HT_RUNTIME_ERROR << status.error_code() << ": " << status.error_message();
    __builtin_unreachable();
  }
}


double DeviceClient::GetDouble(const std::string& key) {
  // Data we are sending to the server.
  GetDoubleRequest request;
  request.set_key(key);


  // Container for the data we expect from the server.
  GetDoubleReply reply;

  // Context for the client. It could be used to convey extra information to
  // the server and/or tweak certain RPC behaviors.
  ClientContext context;

  // The actual RPC.
  Status status = stub_->GetDouble(&context, request, &reply);

  // Act upon its status.
  if (status.ok()) {
    return reply.value();
  } else {
    HT_RUNTIME_ERROR << status.error_code() << ": " << status.error_message();
    __builtin_unreachable();
  }
}

std::string DeviceClient::RemoveDouble(const std::string& key) {
  // Data we are sending to the server.
  RemoveDoubleRequest request;
  request.set_key(key);


  // Container for the data we expect from the server.
  RemoveDoubleReply reply;

  // Context for the client. It could be used to convey extra information to
  // the server and/or tweak certain RPC behaviors.
  ClientContext context;

  // The actual RPC.
  Status status = stub_->RemoveDouble(&context, request, &reply);

  // Act upon its status.
  if (status.ok()) {
    return reply.message();
  } else {
    HT_RUNTIME_ERROR << status.error_code() << ": " << status.error_message();
    __builtin_unreachable();
  }
}

int DeviceClient::PutInt(const std::string& key, int64_t value) {
  // Data we are sending to the server.
  PutIntRequest request;
  request.set_key(key);
  request.set_value(value);


  // Container for the data we expect from the server.
  PutIntReply reply;

  // Context for the client. It could be used to convey extra information to
  // the server and/or tweak certain RPC behaviors.
  ClientContext context;

  // The actual RPC.
  Status status = stub_->PutInt(&context, request, &reply);

  // Act upon its status.
  if (status.ok()) {
    return reply.status();
  } else {
    HT_RUNTIME_ERROR << status.error_code() << ": " << status.error_message();
    __builtin_unreachable();
  }
}


int64_t DeviceClient::GetInt(const std::string& key) {
  // Data we are sending to the server.
  GetIntRequest request;
  request.set_key(key);


  // Container for the data we expect from the server.
  GetIntReply reply;

  // Context for the client. It could be used to convey extra information to
  // the server and/or tweak certain RPC behaviors.
  ClientContext context;

  // The actual RPC.
  Status status = stub_->GetInt(&context, request, &reply);

  // Act upon its status.
  if (status.ok()) {
    return reply.value();
  } else {
    HT_RUNTIME_ERROR << status.error_code() << ": " << status.error_message();
    __builtin_unreachable();
  }
}

std::string DeviceClient::RemoveInt(const std::string& key) {
  // Data we are sending to the server.
  RemoveIntRequest request;
  request.set_key(key);


  // Container for the data we expect from the server.
  RemoveIntReply reply;

  // Context for the client. It could be used to convey extra information to
  // the server and/or tweak certain RPC behaviors.
  ClientContext context;

  // The actual RPC.
  Status status = stub_->RemoveInt(&context, request, &reply);

  // Act upon its status.
  if (status.ok()) {
    return reply.message();
  } else {
    HT_RUNTIME_ERROR << status.error_code() << ": " << status.error_message();
    __builtin_unreachable();
  }
}

int DeviceClient::PutString(const std::string& key, const std::string& value) {
  // Data we are sending to the server.
  PutStringRequest request;
  request.set_key(key);
  request.set_value(value);


  // Container for the data we expect from the server.
  PutStringReply reply;

  // Context for the client. It could be used to convey extra information to
  // the server and/or tweak certain RPC behaviors.
  ClientContext context;

  // The actual RPC.
  Status status = stub_->PutString(&context, request, &reply);

  // Act upon its status.
  if (status.ok()) {
    return reply.status();
  } else {
    HT_RUNTIME_ERROR << status.error_code() << ": " << status.error_message();
    __builtin_unreachable();
  }
}


std::string DeviceClient::GetString(const std::string& key) {
  // Data we are sending to the server.
  GetStringRequest request;
  request.set_key(key);


  // Container for the data we expect from the server.
  GetStringReply reply;

  // Context for the client. It could be used to convey extra information to
  // the server and/or tweak certain RPC behaviors.
  ClientContext context;

  // The actual RPC.
  Status status = stub_->GetString(&context, request, &reply);

  // Act upon its status.
  if (status.ok()) {
    return reply.value();
  } else {
    std::cout << status.error_code() << ": " << status.error_message()
              << std::endl;
    return "RPC failed.";
  }
}

std::string DeviceClient::RemoveString(const std::string& key) {
  // Data we are sending to the server.
  RemoveStringRequest request;
  request.set_key(key);


  // Container for the data we expect from the server.
  RemoveStringReply reply;

  // Context for the client. It could be used to convey extra information to
  // the server and/or tweak certain RPC behaviors.
  ClientContext context;

  // The actual RPC.
  Status status = stub_->RemoveString(&context, request, &reply);

  // Act upon its status.
  if (status.ok()) {
    return reply.message();
  } else {
    HT_RUNTIME_ERROR << status.error_code() << ": " << status.error_message();
    __builtin_unreachable();
  }
}

int DeviceClient::PutBytes(const std::string& key, const std::string& value) {
  // Data we are sending to the server.
  PutBytesRequest request;
  request.set_key(key);
  request.set_value(value);


  // Container for the data we expect from the server.
  PutBytesReply reply;

  // Context for the client. It could be used to convey extra information to
  // the server and/or tweak certain RPC behaviors.
  ClientContext context;

  // The actual RPC.
  Status status = stub_->PutBytes(&context, request, &reply);

  // Act upon its status.
  if (status.ok()) {
    return reply.status();
  } else {
    HT_RUNTIME_ERROR << status.error_code() << ": " << status.error_message();
    __builtin_unreachable();
  }
}


std::string DeviceClient::GetBytes(const std::string& key) {
  // Data we are sending to the server.
  GetBytesRequest request;
  request.set_key(key);


  // Container for the data we expect from the server.
  GetBytesReply reply;

  // Context for the client. It could be used to convey extra information to
  // the server and/or tweak certain RPC behaviors.
  ClientContext context;

  // The actual RPC.
  Status status = stub_->GetBytes(&context, request, &reply);

  // Act upon its status.
  if (status.ok()) {
    return reply.value();
  } else {
    HT_RUNTIME_ERROR << status.error_code() << ": " << status.error_message();
    __builtin_unreachable();
  }
}

std::string DeviceClient::RemoveBytes(const std::string& key) {
  // Data we are sending to the server.
  RemoveBytesRequest request;
  request.set_key(key);


  // Container for the data we expect from the server.
  RemoveBytesReply reply;

  // Context for the client. It could be used to convey extra information to
  // the server and/or tweak certain RPC behaviors.
  ClientContext context;

  // The actual RPC.
  Status status = stub_->RemoveBytes(&context, request, &reply);

  // Act upon its status.
  if (status.ok()) {
    return reply.message();
  } else {
    HT_RUNTIME_ERROR << status.error_code() << ": " << status.error_message();
    __builtin_unreachable();
  }
}

int DeviceClient::PutJson(const std::string& key, const json& value) {
  // Data we are sending to the server.
  PutJsonRequest request;
  request.set_key(key);
  request.set_value(value.dump());


  // Container for the data we expect from the server.
  PutJsonReply reply;

  // Context for the client. It could be used to convey extra information to
  // the server and/or tweak certain RPC behaviors.
  ClientContext context;

  // The actual RPC.
  Status status = stub_->PutJson(&context, request, &reply);

  // Act upon its status.
  if (status.ok()) {
    return reply.status();
  } else {
    HT_RUNTIME_ERROR << status.error_code() << ": " << status.error_message();
    __builtin_unreachable();
  }
}


json DeviceClient::GetJson(const std::string& key) {
  // Data we are sending to the server.
  GetJsonRequest request;
  request.set_key(key);


  // Container for the data we expect from the server.
  GetJsonReply reply;

  // Context for the client. It could be used to convey extra information to
  // the server and/or tweak certain RPC behaviors.
  ClientContext context;

  // The actual RPC.
  Status status = stub_->GetJson(&context, request, &reply);

  // Act upon its status.
  if (status.ok()) {
    std::string output_string = reply.value();
    json output = json::parse(output_string);
    return output;
  } else {
    HT_RUNTIME_ERROR << status.error_code() << ": " << status.error_message();
    __builtin_unreachable();
  }
}

std::string DeviceClient::RemoveJson(const std::string& key) {
  // Data we are sending to the server.
  RemoveJsonRequest request;
  request.set_key(key);


  // Container for the data we expect from the server.
  RemoveJsonReply reply;

  // Context for the client. It could be used to convey extra information to
  // the server and/or tweak certain RPC behaviors.
  ClientContext context;

  // The actual RPC.
  Status status = stub_->RemoveJson(&context, request, &reply);

  // Act upon its status.
  if (status.ok()) {
    return reply.message();
  } else {
    HT_RUNTIME_ERROR << status.error_code() << ": " << status.error_message();
    __builtin_unreachable();
  }
}

int DeviceClient::Barrier(int rank, const std::vector<int>& world_rank) {
  // Data we are sending to the server.
  BarrierRequest request;
  request.set_rank(rank);
  for (auto rank_: world_rank)
    request.add_world_rank(rank_);

  // Container for the data we expect from the server.
  BarrierReply reply;

  // Context for the client. It could be used to convey extra information to
  // the server and/or tweak certain RPC behaviors.
  ClientContext context;

  // The actual RPC.
  Status status = stub_->Barrier(&context, request, &reply);

  // Act upon its status.
  if (status.ok()) {
    return reply.status();
  } else {
    HT_RUNTIME_ERROR << status.error_code() << ": " << status.error_message();
    __builtin_unreachable();
  }
}

int DeviceClient::Consistent(int rank, int value, const std::vector<int>& world_rank) {
  // Data we are sending to the server.
  ConsistentRequest request;
  request.set_rank(rank);
  request.set_value(value);
  for (auto rank_: world_rank)
    request.add_world_rank(rank_);

  // Container for the data we expect from the server.
  ConsistentReply reply;

  // Context for the client. It could be used to convey extra information to
  // the server and/or tweak certain RPC behaviors.
  ClientContext context;

  // The actual RPC.
  Status status = stub_->Consistent(&context, request, &reply);

  // Act upon its status.
  if (status.ok()) {
    return reply.status();
  } else {
    HT_RUNTIME_ERROR << status.error_code() << ": " << status.error_message();
    __builtin_unreachable();
  }
}

int DeviceClient::HeartBeat(int rank) {
  // Data we are sending to the server.
  HeartBeatRequest request;
  request.set_rank(rank);

  // Container for the data we expect from the server.
  HeartBeatReply reply;

  // Context for the client. It could be used to convey extra information to
  // the server and/or tweak certain RPC behaviors.
  ClientContext context;

  // The actual RPC.
  Status status = stub_->HeartBeat(&context, request, &reply);

  // Act upon its status.
  if (status.ok()) {
    return reply.status();
  } else {
    HT_RUNTIME_ERROR << status.error_code() << ": " << status.error_message();
    __builtin_unreachable();
  }
}

void DeviceClient::LaunchHeartBeat(int rank) {
  _rank = rank;
  this->start();
}

} //namespace hetu

