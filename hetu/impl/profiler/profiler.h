#pragma once

#include "hetu/core/stream.h"
#include "hetu/common/macros.h"
#include "hetu/core/ndarray.h"
#include "hetu/graph/common.h"
#include "hetu/core/device.h"
#include "hetu/graph/tensor.h"
#include "hetu/graph/operator.h"
#include "hetu/graph/subgraph.h"
#include "hetu/utils/optional.h"
#include <stack>
#include "hetu/graph/graph.h"
#include "hetu/graph/common.h"
#include <set>
#include "hetu/impl/communication/comm_group.h"
#include <fstream>
#include <chrono>

namespace hetu {
namespace impl {

typedef uint64_t RecordFunctionHandle;

inline int64_t getCurrentTimeInMicroseconds() {
  auto now = std::chrono::high_resolution_clock::now();
  auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
  return ns / 1000; // 将纳秒转换为微秒，但保留更高精度
}


struct export_event {
  std::string name;          // 事件的名称
  std::int64_t l;            // 起始位置或时间
  std::int64_t r;            // 结束位置或时间
  uint64_t ptr;
  std::int64_t bytes; // 内存变化量
  std::int64_t total_alloc;
  std::int64_t total_reserved;
  DeviceIndex device_id;             // 设备ID

  // 构造函数，允许从外部直接初始化所有成员变量
  export_event(const std::string& name_ = "", int device_id_ = -1, std::int64_t l_ = std::numeric_limits<std::int64_t>::max(), 
    std::int64_t r_ = std::numeric_limits<std::int64_t>::min(), uint64_t ptr_ = 0, std::int64_t bytes_ = 0, std::int64_t total_alloc_ = 0, std::int64_t total_reserved_ = 0)
    : name(name_), l(l_), r(r_), ptr(ptr_), bytes(bytes_), total_alloc(total_alloc_), total_reserved(total_reserved_), device_id(device_id_) {}
  void update_r(std::int64_t r_new){
    HT_ASSERT(r_new >= l);
    r = r_new;
  }
};

enum class MemoryEventKind : uint16_t {
  Mark,
  PushRange,
  PopRange,
  MemoryAlloc,
};

enum class RecordScope : uint8_t {
    OP = 0,
    CUSTOM_SCOPE,
    NUM_SCOPES,
};


struct MemoryEvent final {
  MemoryEvent(
    MemoryEventKind kind,
    std::string name,
    int64_t ts,
    hetu::graph::OpId op_id = -1,
    RecordFunctionHandle handle = 0,
    RecordScope scope = RecordScope::OP,
    int node_id = -1)
    : name_(std::move(name)),
        kind_(kind),
        ts_(ts),
        handle_(handle),
        scope_(scope),
        op_id_(op_id),
        node_id_(node_id) {}


  MemoryEvent(MemoryEvent&& other)
    : name_(other.name_),
    kind_(other.kind_),
    handle_(other.handle_),
    ts_(other.ts_),
    cuda_memory_usage_(other.cuda_memory_usage_),
    cuda_memory_total_alloc_(other.cuda_memory_total_alloc_),
    cuda_memory_total_reserved_(other.cuda_memory_total_reserved_),
    device_id_(other.device_id_),
    node_id_(other.node_id_),
    scope_(other.scope_),
    op_id_(other.op_id_),
    ptr_(other.ptr_) {}

  MemoryEvent& operator=(MemoryEvent&& other) {
    if (this != &other) {
      name_ = other.name_;
      kind_ = other.kind_;
      handle_ = other.handle_;
      ts_ = other.ts_;
      cuda_memory_usage_ = other.cuda_memory_usage_;
      cuda_memory_total_alloc_ = other.cuda_memory_total_alloc_;
      cuda_memory_total_reserved_ = other.cuda_memory_total_alloc_;
      device_id_ = other.device_id_;
      node_id_ = other.node_id_;
      scope_ = other.scope_;
      ptr_ = other.ptr_;
      op_id_ = other.op_id_;
    }
    return *this;
  }

  std::string kind() const {
    switch(kind_) {
    case MemoryEventKind::Mark: return "mark";
    case MemoryEventKind::PushRange: return "push";
    case MemoryEventKind::PopRange: return "pop";
    case MemoryEventKind::MemoryAlloc: return "memory_alloc";
    }
    throw std::runtime_error("unknown MemoryEventKind");
  }

  const std::string name() const {
    return name_;
  }

  std::vector<std::vector<int64_t>> shapes() const {
    return shapes_;
  }

  hetu::DeviceIndex device_id() const {
    return device_id_;
  }

  void updateCudaMemoryStats(void* ptr, int64_t bytes, int64_t total_alloc, int64_t total_reserved, Device device) {
    HT_ASSERT(device.type() == DeviceType::CUDA);
    cuda_memory_usage_ = bytes;
    cuda_memory_total_alloc_ = total_alloc;
    cuda_memory_total_reserved_ = total_reserved;
    ptr_ = ptr;
  }

  int64_t cudaMemoryUsage() const {
    return cuda_memory_usage_;
  }

  int64_t cudaMemoryTotalAlloc() const {
    return cuda_memory_total_alloc_;
  }

  int64_t cudaMemoryTotalReserved() const {
    return cuda_memory_total_reserved_;
  }

  RecordFunctionHandle handle() const {
    return handle_;
  }

  int nodeId( ) const {
    return node_id_;
  }

  void setNodeId(int node_id) {
    node_id_ = node_id;
  }

  void setName(std::string newName_) {
    name_ = std::move(newName_);
  }


  int64_t getTs(){
    return ts_;
  }

  RecordScope scope() const {
    return scope_;
  }

  void* getPtr(){
    return ptr_;
  }

  hetu::graph::OpId op_id() const{
    return op_id_;
  }

  void* ptr() const{
    return ptr_;
  }

  private:
    // signed to allow for negative intervals, initialized for safety.
    std::string name_;
    MemoryEventKind kind_;
    RecordFunctionHandle handle_ {0};
    std::vector<std::vector<int64_t>> shapes_;
    int64_t cuda_memory_usage_ = 0;
    int64_t cuda_memory_total_alloc_ = 0;
    int64_t cuda_memory_total_reserved_ = 0;
    int64_t ts_ = 0;
    hetu::DeviceIndex device_id_ = -1;
    int node_id_ = 0;
    hetu::graph::OpId op_id_;
    RecordScope scope_;
    void* ptr_;
};


struct RangeMemoryEventList {
  RangeMemoryEventList() {
      events_.reserve(kReservedCapacity);
  }

  template<typename... Args>
  void record(Args&&... args) {
    events_.emplace_back(std::forward<Args>(args)...);
  }

  std::vector<MemoryEvent> consolidate() {
    std::vector<MemoryEvent> result;
    result.insert(
      result.begin(),
      std::make_move_iterator(events_.begin()),
      std::make_move_iterator(events_.end()));
    events_.erase(events_.begin(), events_.end());
    return result;
  }

  size_t size() {
    return events_.size();
  }
  
  void clear(){
    events_.clear();
  }

  private:
    std::vector<MemoryEvent> events_;
    static const size_t kReservedCapacity = 1024;
};


struct RecordFunction{
  RecordFunction(RecordScope scope = RecordScope::OP) : scope_(scope) {}

  virtual ~RecordFunction();

  RecordFunction(const RecordFunction&) = delete;
  RecordFunction& operator=(const RecordFunction&) = delete;

  inline const std::string& name() const{
    return name_;
  } 

  const HTShapeList inputs_shape() const {
    return inputs_shape_;
  }

  inline RecordScope scope() const{
    return scope_;
  }

  hetu::graph::OpId op_id() const{
    return op_id_;
  }

  void before(
    std::string fn,
    hetu::graph::OpId id) {
    op_id_ = id;
    before(fn);
  }
  
  void before(std::string);


  void end();

  inline RecordFunctionHandle handle() const {
    return handle_;
  }

  inline void setHandle(RecordFunctionHandle handle) {
    handle_ = handle;
  }


  bool needs_inputs = false;
  private:

    hetu::graph::OpId op_id_;
    bool called_start_callbacks_ = false;
    std::string name_;
    HTShapeList inputs_shape_;

    const RecordScope scope_;
    RecordFunctionHandle handle_ {0};

};


struct OpProfilerInfo {
  hetu::graph::OpType type;
  hetu::graph::OpName name;
  HTShapeList inputs_shape;
  double cost_time;
};

using ProfileId = uint64_t;




class Profile {
 public:
  Profile(bool enabled = true, bool use_cpu = false, bool use_cuda = false,
          bool record_shapes = false, bool profile_memory = false)
  : _id(_next_profile_id()), _enabled(enabled), _use_cpu(use_cpu), _use_cuda(use_cuda),
    _record_shapes(record_shapes), _profile_memory(profile_memory), _device(Device()) {
      if(profile_memory){
        HT_ASSERT(use_cuda);
        MemoryEvent evt0(
            MemoryEventKind::Mark,
            std::string("__start_profile"), 
            getCurrentTimeInMicroseconds());
        get_event_list().record(std::move(evt0));
      }
    }

  Profile(const Profile&) = delete;
  Profile& operator=(const Profile&) = delete;
  Profile(Profile&&) = delete;
  Profile& operator=(Profile&&) = delete;

  ~Profile() {
    Clear();
  }

  void Clear() {
    _op_record.clear();
    _ops.clear();
    _graph_view_record.clear();
    if(event_list != nullptr){
      event_list->clear();
      event_list = nullptr;
    }
  }

  void push(hetu::graph::OpType type, hetu::graph::OpName name,
            HTShapeList inputs_shape, double cost_time) {
    _op_record.push_back({type, name, inputs_shape, cost_time});
  }

  void push(hetu::graph::Operator& op) {
    HT_VALUE_ERROR_IF(!_enabled) << "The Profiler is not enabled";
    _ops.push_back(op);
  }

  void push(hetu::graph::OpType type, double total_time) {
    HT_VALUE_ERROR_IF(!_enabled) << "The Profiler is not enabled";
    _graph_view_record.push_back({type, total_time});
  }

  void set_device(Device device) {
    _device = device;
  }

  bool enabled() const {
    return _enabled;
  }

  bool profile_memory() const{
    return _profile_memory;
  }

  bool record_shapes() const {
    return _record_shapes;
  }

  void sync_op() {
    if (!enabled())
      return;
    if (!_ops.empty()) {
      std::unordered_set<hetu::Stream> _sync_stream;
      for (auto& op : _ops) {
        _sync_stream.insert(op->instantiation_ctx().stream());
      }
      for (auto& stream : _sync_stream) {
        stream.Sync();
      }
      for (auto& op : _ops) {
        HTShapeList inputs_shape;
	      hetu::graph::Operator::for_each_input_tensor(op, [&](const hetu::graph::Tensor& input) {
          inputs_shape.push_back(input->shape());
        });
        _op_record.push_back({op->type(), op->name(), inputs_shape, op->TimeCost(0) * 1.0 / 1e6});
      }
      _ops.clear();
    }
    return;
  }

  std::vector<std::pair<hetu::graph::OpType, std::pair<double, std::pair<double, int>>>>
  get_optype_view() {
    sync_op();
    std::vector<std::pair<hetu::graph::OpType, std::pair<double, std::pair<double, int>>>> single_optype_total_time;
    std::map<hetu::graph::OpType, std::pair<double, int>> single_optype_total_time_unordered;

    for (auto& record: _op_record) {
      if (single_optype_total_time_unordered.find(record.type) == single_optype_total_time_unordered.end()) {
        single_optype_total_time_unordered[record.type] = {record.cost_time, 1};
      } else {
        single_optype_total_time_unordered[record.type].first += record.cost_time;
        single_optype_total_time_unordered[record.type].second++;
      }
    }
    for(auto& record : single_optype_total_time_unordered) {
      auto type = record.first;
      auto total_time = record.second.first;
      auto cnt = record.second.second;
      single_optype_total_time.push_back({type, {total_time, {total_time / cnt, cnt}}});
    }
    std::sort(single_optype_total_time.begin(), single_optype_total_time.end(),
      [&](std::pair<hetu::graph::OpType, std::pair<double, std::pair<double, int>>> x, std::pair<hetu::graph::OpType, std::pair<double, std::pair<double, int>>> y) {
                return x.second.second.first > y.second.second.first; });
    return single_optype_total_time;
  }

  std::vector<std::pair<std::pair<hetu::graph::OpType, HTShapeList>, std::pair<double, std::pair<double, int>>>>
  get_optype_with_inputs_view() {
    sync_op();

    std::vector<std::pair<std::pair<hetu::graph::OpType, HTShapeList>, std::pair<double, std::pair<double, int>>>> single_optype_with_inputs_total_time;
    std::map<std::pair<hetu::graph::OpType, HTShapeList>, std::pair< double, int>> single_optype_with_inputs_total_time_unordered;
    for (auto& record: _op_record) {
      if (single_optype_with_inputs_total_time_unordered.find({record.type, record.inputs_shape}) == single_optype_with_inputs_total_time_unordered.end()) {
        single_optype_with_inputs_total_time_unordered[{record.type, record.inputs_shape}] = {record.cost_time, 1};
      }
      else {
        single_optype_with_inputs_total_time_unordered[{record.type, record.inputs_shape}].first += record.cost_time;
        single_optype_with_inputs_total_time_unordered[{record.type, record.inputs_shape}].second++;
      }
    }
    for (auto& record : single_optype_with_inputs_total_time_unordered) {
      auto type = record.first.first;
      auto inputs_shape = record.first.second;
      auto total_time = record.second.first;
      auto cnt = record.second.second;
      single_optype_with_inputs_total_time.push_back({{type, inputs_shape}, {total_time, {total_time / cnt, cnt}}});
    }
    std::sort(single_optype_with_inputs_total_time.begin(), single_optype_with_inputs_total_time.end(),
      [&](std::pair<std::pair<hetu::graph::OpType, HTShapeList>, std::pair<double, std::pair<double, int>>> x, std::pair<std::pair<hetu::graph::OpType, HTShapeList>, std::pair<double, std::pair<double, int>>> y) {
            return x.second.second.first > y.second.second.first; });
    return single_optype_with_inputs_total_time;
  }

  std::vector<std::pair<hetu::graph::OpType, double>> get_graph_view() {
    return _graph_view_record;
  }

  std::vector<OpProfilerInfo> get_op_view() {
    sync_op();
    return  _op_record;
  }

  RangeMemoryEventList& get_event_list() {
    if(event_list == nullptr) event_list = std::make_shared<RangeMemoryEventList>();
    return *event_list;
  }


  static std::string print_instant_event(int64_t ts, int deviceId, int64_t addr, int64_t bytes, int64_t totalAllocated, int64_t totalReserved){
      std::ostringstream json;
      json << std::fixed << std::setprecision(0);
      json << "        {\n"
                << "            \"ph\": \"" << "i" << "\",\n"
                << "            \"cat\": \"" << "cpu_instant_event" << "\",\n"
                << "            \"s\": \"" << 't' << "\",\n"
                << "            \"name\": \"" << "[memory]" << "\",\n"
                << "            \"pid\": " << -1 << ",\n"
                << "            \"tid\": " << -1 << ",\n"
                << "            \"ts\": " << ts  << ",\n"
                << "            \"args\": {\n"
                << "                \"Profiler Event Index\": " << -1 << ",\n"
                << "                \"Call stack\": \"" << "None" << "\",\n"
                << "                \"Device Type\": " << "1" << ",\n"
                << "                \"Device Id\": " << deviceId << ",\n"
                << "                \"Addr\": " << addr << ",\n"
                << "                \"Bytes\": " << bytes << ",\n"
                << "                \"Total Allocated\": " << totalAllocated << ",\n"
                << "                \"Total Reserved\": " << totalReserved << "\n"
                << "            }\n"
                << "        }";
      return json.str();
  }

  static std::string print_complete_event(std::string name, int64_t ts, int64_t dur){
      std::ostringstream json;
      json << std::fixed << std::setprecision(0); 
      json << "        {\n"
          << "            \"ph\": \"" << "X" << "\",\n"
          << "            \"cat\": \"" << "cpu_op" << "\",\n"
          << "            \"name\": \"" << name << "\",\n"
          << "            \"pid\": " << -1 << ",\n"
          << "            \"tid\": " << -1 << ",\n"
          << "            \"ts\": " << ts  << ",\n"
          << "            \"dur\": " << dur << ",\n"
          << "            \"args\": {\n"
          << "                \"Trace name\": \"" << "Profiler" << "\",\n"
          << "                \"Trace iteration\": " << 0 << ",\n"
          << "                \"External id\": " << -1 << ",\n"
          << "                \"Profiler Event Index\": " << -1 << ",\n"
          << "                \"Call stack\": \"" << -1 << "\",\n"
          << "                \"Input Dims\": " << "[]" << ",\n"
          << "                \"Input type\": " << "[]" << "\n"
          << "            }\n"
          << "        }";    
      return json.str();
  }

  static void generate_json(std::vector< export_event >& tmp_exports_event, std::string file_path){
    std::ostringstream json;
    auto& local_device = hetu::impl::comm::GetLocalDevice();
    json << std::fixed << std::setprecision(0); 
    json << "{\n";
    json << "    \"traceEvents\": [\n";
    for(int i = 0; i < tmp_exports_event.size(); i ++){
      auto& export_evt = tmp_exports_event[i];
      HT_ASSERT(export_evt.l > 0 && export_evt.r > 0 && export_evt.l <= export_evt.r);
      // if(export_evt.l == export_evt.r && export_evt.bytes == 0) continue;
      if(export_evt.l == export_evt.r) json << print_instant_event(export_evt.l, (int)export_evt.device_id, (int64_t)export_evt.ptr, export_evt.bytes, export_evt.total_alloc, export_evt.total_reserved);
      else if(export_evt.l != export_evt.r) json << print_complete_event(export_evt.name, export_evt.l ,  (export_evt.r -  export_evt.l));
      if(i == (int)tmp_exports_event.size() - 1) json << "\n";
      else json << ",\n";
    }
    json << "    ]\n";
    json << "}\n";    
    std::ofstream json_file(file_path);
    if(json_file) json_file << json.str();
    tmp_exports_event.clear();
    return ;
  }

  void update_subgraph_memory(std::unordered_map<std::string, std::shared_ptr<hetu::graph::SubGraph>>& subgraphs){
    auto& local_device = hetu::impl::comm::GetLocalDevice();
    std::unordered_map<hetu::graph::OpId, std::shared_ptr<hetu::graph::SubGraph> > OpId2SubGraph; // op到subgraph的映射
    std::unordered_set<hetu::graph::OpId> param_ops, fw_ops, bw_ops, update_ops; // 对op分类
    for(auto [name, subgraph] : subgraphs){
      for(auto [name, op] : subgraph->ops()){
        OpId2SubGraph[op->id()] = subgraph;
        if(hetu::graph::is_variable_op(op)){
          param_ops.insert(op->id());
        }
        else fw_ops.insert(op->id());
      }
      for(auto [name, op] : subgraph->bwd_ops()){
        OpId2SubGraph[op->id()] = subgraph;
        bw_ops.insert(op->id());
      }
      for(auto [name, op] : subgraph->update_ops()){
        OpId2SubGraph[op->id()] = subgraph;
        update_ops.insert(op->id());
      }
    }
    

    auto evt_list = get_event_list().consolidate();  // 获得所有记录的事件，包含push，memory_alloc，pop等类型
    std::vector< export_event > tmp_exports_event; 
    // 为subgraph创建对应的区间，例如某个subgraph对应的区间是[x, y], 那么[x,y]内发生的所有事件(op push, memory_alloc,op pop)归这个subgraph所有
    // 一个相同的subgrph，会在前向和反向出现多次，不能使用一个[x, y]的区间表示，不然每个subgrph几乎都涵盖了几乎所有的op
    // 因此，分成前向和反向分别处理
    auto process_stage = [&](int l, int r, std::unordered_set<hetu::graph::OpId>& ops_set, std::string stage_name){
      std::unordered_map<hetu::graph::OpId, std::pair<int64_t, int64_t> > op2LR;
      for(int i = l; i <= r; i ++){
        if(evt_list[i].kind() == "push" && ops_set.find(evt_list[i].op_id()) != ops_set.end()){
          auto op_event = export_event(evt_list[i].name(), local_device.index(), evt_list[i].getTs(), evt_list[i].getTs());
          auto op_id = evt_list[i].op_id();
          while(i + 1 <= r && evt_list[i + 1].kind() == "memory_alloc"){
            i ++;
          }
          HT_ASSERT(i + 1 <= r && evt_list[i + 1].kind() == "pop");
          op_event.update_r(evt_list[i + 1].getTs());
          // tmp_exports_event.push_back(op_event);
          op2LR[op_id] = {op_event.l, op_event.r};
        }
      }
      std::unordered_map<std::shared_ptr<hetu::graph::SubGraph>, std::pair<int64_t, int64_t> > subGraph2LR;
      std::function<std::pair<int64_t, int64_t>(std::string, std::shared_ptr<hetu::graph::SubGraph>)> dfs_push_subgraph_event;
      dfs_push_subgraph_event = [&](std::string subgraph_name, std::shared_ptr<hetu::graph::SubGraph> subgraph) {
        if (subGraph2LR.find(subgraph) != subGraph2LR.end()) {
            return subGraph2LR[subgraph];
        }

        int64_t L = std::numeric_limits<std::int64_t>::max();
        int64_t R = std::numeric_limits<std::int64_t>::min();
        for (auto& [name, subgraph_son] : subgraph->subgraphs()) {
            auto lr = dfs_push_subgraph_event(name, subgraph_son);
            L = std::min(L, lr.first);
            R = std::max(R, lr.second);
        }
        for (auto& [name, op] : subgraph->ops()) {
            if (op2LR.find(op->id()) != op2LR.end()) {
                L = std::min(L, op2LR[op->id()].first);
                R = std::max(R, op2LR[op->id()].second);
            }
        }
        for (auto& [name, op] : subgraph->bwd_ops()) {
            if (op2LR.find(op->id()) != op2LR.end()) {
                L = std::min(L, op2LR[op->id()].first);
                R = std::max(R, op2LR[op->id()].second);
            }
        }
        for (auto& [name, op] : subgraph->update_ops()) {
            if (op2LR.find(op->id()) != op2LR.end()) {
                L = std::min(L, op2LR[op->id()].first);
                R = std::max(R, op2LR[op->id()].second);
            }
        }
        if(L != std::numeric_limits<std::int64_t>::max()){
          auto subgraph_event = export_event(subgraph_name + stage_name, local_device.index(), L, R);
          tmp_exports_event.push_back(subgraph_event);
        }
        subGraph2LR[subgraph] = {L, R};
        return std::make_pair(L, R);
      };
      for(auto [name, subgraph] : subgraphs){
        dfs_push_subgraph_event(name, subgraph);
      }
    };

    std::vector< std::pair<int, int> > task_stage; // 记录forward_step和backward_step的开始和结束
    for(int i = 0; i < evt_list.size(); i ++){
      if((evt_list[i].name() == "forward_step" || evt_list[i].name() == "backward_step") && evt_list[i].kind() == "push"){
        int j = i + 1;
        while(j < evt_list.size() && 
              !((evt_list[j].name() == "forward_step" || evt_list[j].name() == "backward_step") && evt_list[j].kind() == "pop")) j ++;
        HT_ASSERT((evt_list[j].name() == "forward_step" || evt_list[j].name() == "backward_step") && evt_list[j].kind() == "pop");
        task_stage.push_back({i, j});
        i = j;
      }
    }

    for(auto [l, r] : task_stage){
      HT_ASSERT(l + 1 <= r - 1);
      if(evt_list[l].name() == "forward_step"){
        process_stage(l + 1, r - 1, fw_ops,"_fw");
      }
      else if(evt_list[l].name() == "backward_step"){
        process_stage(l + 1, r - 1, bw_ops, "_bw");
        process_stage(l + 1, r - 1, update_ops, "_update");
      }
    }

    for(int i = 0; i < evt_list.size(); i ++){
      if(evt_list[i].kind() == "memory_alloc"){
        auto bytes = evt_list[i].cudaMemoryUsage();
        auto total_alloc = evt_list[i].cudaMemoryTotalAlloc();
        auto total_reserved = evt_list[i].cudaMemoryTotalReserved();
        auto ts = evt_list[i].getTs();
        auto ptr = evt_list[i].getPtr();
        tmp_exports_event.push_back(export_event("", local_device.index(), ts, ts, (uint64_t)ptr, bytes, total_alloc, total_reserved));         
      }
      else if(!(evt_list[i].name() == "forward_step" || evt_list[i].name() == "backward_step") && evt_list[i].kind() == "push" ){
        int j = i + 1;
        while(j < evt_list.size() && evt_list[j].kind() == "memory_alloc") j ++;
        HT_ASSERT(evt_list[j].kind() == "pop");
        tmp_exports_event.push_back(export_event(evt_list[i].name(), local_device.index(), evt_list[i].getTs(), evt_list[j].getTs()));        
      }
    }
    for(auto& exports_event : tmp_exports_event){
      HT_ASSERT(exports_event.l > init_time);
      exports_event.l -= init_time;
      exports_event.r -= init_time;
    }
    sort(tmp_exports_event.begin(), tmp_exports_event.end(), [&](export_event a, export_event b){
      return a.l < b.l;
    });
    

    for(auto &evt : tmp_exports_event){
      HT_ASSERT(evt.l <= evt.r && evt.l >= 0 && evt.r >= 0);
      memory_profiler_info.push_back(evt);
    }

    return ;
  }

  static void generate_memory_json(std::string file_path){
    generate_json(memory_profiler_info, file_path);
  }

 private:
  static ProfileId _next_profile_id();


 protected:
  ProfileId _id;
  bool _enabled;
  bool _record_shapes;
  bool _use_cpu;
  bool _use_cuda;
  bool _profile_memory;
  Device _device;
  std::vector<std::pair<std::string, double>> _graph_view_record;
  std::vector<OpProfilerInfo> _op_record;
  std::vector<hetu::graph::Operator> _ops;
  std::shared_ptr<RangeMemoryEventList> event_list = nullptr;
  static std::vector< export_event > memory_profiler_info;
  static int64_t init_time;

  static void InitOnce() {
    std::call_once(Profile::_init_flag, Profile::Init);
  }

  static void Init();

 public:
  static Profile& make_new_profile(bool enabled = true, bool use_cpu = false,
                                   bool use_cuda = false, bool record_shapes = false,
                                   bool profile_memory = false) {
    InitOnce();
    auto res = std::make_shared<Profile>(enabled, use_cpu, use_cuda, record_shapes, profile_memory);
    Profile::_global_profile.push_back(res);
    return *Profile::_global_profile.back();
  }

  ProfileId id() {
    return _id;
  }

  static inline optional<std::shared_ptr<Profile>> get_cur_profile() {
    if (Profile::_cur_profile_ctx.empty() ||
        !Profile::_global_profile[Profile::_cur_profile_ctx.top()]->enabled())
      return std::nullopt;
    return Profile::_global_profile[Profile::_cur_profile_ctx.top()];
  }

  static inline std::shared_ptr<Profile> get_profile(ProfileId profile_id) {
    HT_VALUE_ERROR_IF(profile_id >= Profile::_global_profile.size())
      << "Profile with id " << profile_id << " does not exist";
    return Profile::_global_profile[profile_id];
  }

  static void push_profile_ctx(ProfileId id) {
    HT_VALUE_ERROR_IF(id >= Profile::_global_profile.size())
      << "Profile with id " << id << " does not exist";
    Profile::_cur_profile_ctx.push(id);
  }

  static void pop_profile_ctx() {
    Profile::_cur_profile_ctx.pop();
  }



 protected:
  static std::once_flag _init_flag;
  static std::vector<std::shared_ptr<Profile>> _global_profile;
  static thread_local std::stack<ProfileId> _cur_profile_ctx;
};

RangeMemoryEventList& getCurEventList();

void reportCudaMemoryToProfiler(void* ptr, int64_t bytes, int64_t total_alloc, int64_t total_reserved, Device dev);


#define RECORD_OP(op_name, op_id, ...) \
  hetu::impl::RecordFunction guard(hetu::impl::RecordScope::OP); \
guard.before(op_name, op_id, ##__VA_ARGS__);



#define RECORD_CUSTOM_SCOPE(op_name, ...) \
  hetu::impl::RecordFunction guard(hetu::impl::RecordScope::CUSTOM_SCOPE); \
guard.before(op_name, -1, ##__VA_ARGS__);

} // namespace impl
} // namespace hetu