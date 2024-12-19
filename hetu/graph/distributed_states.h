#pragma once

#include "hetu/graph/common.h"
#include <functional>
#include <algorithm>

namespace hetu {
namespace graph {

class DistributedStates;
using DistributedStatesList = std::vector<DistributedStates>;

class DistributedStates {
 public:
  DistributedStates() : _device_num(-1), _states({}), _order({}), _zero(false) {};
  DistributedStates(DeviceGroup placement_group, std::unordered_map<int32_t, int32_t> states, 
                    std::vector<int32_t> order = {}, bool zero = false) {
    _placement_group = placement_group;
    _device_num = placement_group.num_devices();
    _zero = zero;
    set_states(states);
    set_order(order); 
  }
  // independent distributed states, the placement_group should be assigned when binding with tensor
  DistributedStates(int32_t device_num, std::unordered_map<int32_t, int32_t> states, 
                    std::vector<int32_t> order = {}, bool zero = false) {
    _placement_group = DeviceGroup();
    _device_num = device_num;
    _zero = zero;
    set_states(states);
    set_order(order); 
  }
  DistributedStates(const DistributedStates& ds) {
    _device_num = -1;
    if (ds._device_num == -1) return; // do nothing if ds is invalid
    set_distributed_states(ds);
  }

  void set_placement_group(const DeviceGroup& placement_group);

  void set_placement(const Device& placement);

  void set_distributed_states(const DistributedStates& dst_distributed_states);

  bool is_none() const;
  
  bool is_valid() const;

  const std::unordered_map<int32_t, int32_t>& get_states() const {
    return _states;
  }

  int32_t states(int32_t key) const {
    return get_dim(key);
  } 

  const std::vector<int32_t>& get_order() const {
    return _order;
  }

  int32_t order(int32_t i) const {
    return _order.at(i);
  }

  bool has_placement_group() const {
    return !_placement_group.empty();
  }
  
  bool zero() const {
    return _zero;
  }

  void set_zero(bool zero) {
    _zero = zero;
  }

  const DeviceGroup& get_placement_group() const {
    return _placement_group;
  }

  const Device& get_placement() const {
    return _placement;
  }

  int32_t get_placement_index() const {
    return _placement_group.get_index(_placement);
  }

  int32_t get_device_num() const {
    return _device_num;
  }

  std::unordered_map<int32_t, int32_t> combine_states(const std::pair<std::vector<int32_t>, int32_t>& src2dst) const;
  static std::unordered_map<int32_t, int32_t> combine_states(const std::pair<std::vector<int32_t>, int32_t>& src2dst, 
                                                             const std::unordered_map<int32_t, int32_t>& ori_states);
  std::vector<int32_t> combine_order(const std::pair<std::vector<int32_t>, int32_t>& src2dst) const;
  static std::vector<int32_t> combine_order(const std::pair<std::vector<int32_t>, int32_t>& src2dst, 
                                            const std::vector<int32_t>& ori_order);
  bool equal_states_and_order(const std::unordered_map<int32_t, int32_t>& states1, const std::vector<int32_t>& order1,
                              const std::unordered_map<int32_t, int32_t>& states2, const std::vector<int32_t>& order2) const;
  bool check_equal(const DistributedStates& dst_distributed_states) const;
  bool check_max_dim(int32_t max_dim) const;
  bool check_pure_duplicate() const;    
  bool check_combine(const DistributedStates& dst_distributed_states, const std::pair<std::vector<int32_t>, int32_t>& src2dst) const;

  std::unordered_map<int32_t, int32_t> reduce_states(int dim) const;
  std::vector<int32_t> reduce_order(int dim) const;
  bool check_reduce_dim(const DistributedStates& dst_distributed_states, int dim) const;

  bool check_split(const DistributedStates& dst_distributed_states) const;
  bool check_allreduce(const DistributedStates& dst_distributed_states) const;
  bool check_scatter(const DistributedStates& dst_distributed_states) const;
  bool check_allgather(const DistributedStates& dst_distributed_states) const;
  bool check_reducescatter(const DistributedStates& dst_distributed_states) const;
  bool check_boradcast(const DistributedStates& dst_distributed_states) const;
  bool check_reduce(const DistributedStates& dst_distributed_states) const;  

  int32_t get_dim(int32_t index) const;
  int32_t get_split_dim(const DistributedStates& merged_distributed_states) const;
  std::vector<int32_t> get_loop_sizes() const;
  std::unordered_map<int32_t, int32_t> map_device_to_state_index(int32_t device_index) const;
  int32_t get_dup_group_index(int32_t device_index) const;
  DeviceGroup get_devices_by_dim(int32_t dim, int32_t local_device_idx, DeviceGroup group) const;
  std::string ds_info() const;

 protected:
  // should call set_states and set_order at the same time
  void set_states(const std::unordered_map<int32_t, int32_t>& states);

  void set_order(const std::vector<int32_t>& order);

  bool _zero;
  std::unordered_map<int32_t, int32_t> _states; // {dimension: split_num}, {-2: partial, -1: duplicate, 0~n-1: dimension}
  std::vector<int32_t> _order; // for device mapping
  DeviceGroup _placement_group; // must be assigned when binding with tensor
  int32_t _device_num; // if placement_group is null, the device_num must be assigned
  Device _placement;
};

class SplitPattern {
 public:
  SplitPattern(bool contiguous) : _contiguous(contiguous) {};

  bool is_contiguous() const {
    return _contiguous;
  }

  bool check_equal(const SplitPattern& another) const {
    return _contiguous == another.is_contiguous();
  }

 protected:
  bool _contiguous;
};

#define NULL_HETERO_DIM -3

class DistributedStatesUnion {
 public: 
  DistributedStatesUnion() {}
  DistributedStatesUnion(const DistributedStatesList& ds_list, int32_t hetero_dim = NULL_HETERO_DIM) 
  : _union(ds_list),
    _hetero_dim(hetero_dim) {
    if (ds_list.size() > 1) {
      HT_ASSERT(hetero_dim != NULL_HETERO_DIM)
        << "hetero size must be 1 if the hetero dim is NULL";
    }
  }

  const SplitPattern& split_pattern() const {
    return _split_pattern;
  }

  void set_split_pattern(const SplitPattern& split_pattern) {
    _split_pattern = split_pattern;
  }

  bool is_hetero() const {
    bool judgement_1 = (_hetero_dim != NULL_HETERO_DIM);
    bool judgement_2 = (_union.size() > 1);
    HT_ASSERT(!(judgement_1 ^ judgement_2))
        << "hetero means union size is greater than 1"
        << ", but found hetero dim is " << _hetero_dim
        << " and union size is " << _union.size();
    return judgement_1;
  }

  int32_t hetero_dim() const {
    return _hetero_dim;
  }

  void set_hetero_dim(int32_t hetero_dim) {
    _hetero_dim = hetero_dim;
  }

  void change_hetero_dim(int32_t hetero_dim) {
    size_t union_size = _union.size();
    HT_ASSERT(_hetero_dim != NULL_HETERO_DIM && union_size > 1)
      << "must be hetero already";
    for (size_t i = 0; i < union_size; i++) {
      auto& ds = _union.at(i);
      HT_ASSERT(ds.order(0) == _hetero_dim)
        << "now only support hetero change in its first order";
      auto local_ds = get_local(i);
      auto new_states = local_ds.get_states();
      auto new_order = local_ds.get_order();
      if (new_states.find(hetero_dim) == new_states.end()) {
        new_states[hetero_dim] = union_size;
      } else {
        new_states[hetero_dim] *= union_size;
      }
      if (std::find(new_order.begin(), new_order.end(), hetero_dim) == new_order.end()) {
        new_order.insert(new_order.begin(), hetero_dim);
      }
      HT_ASSERT(new_order.at(0) == hetero_dim)
        << "now only support hetero change in its first order";
      _union.at(i) = DistributedStates(ds.get_device_num(), new_states, new_order, ds.zero());
    }
    _hetero_dim = hetero_dim;
  }

  const DistributedStatesUnion to_hetero(int32_t dim, int32_t num) const {
    HT_ASSERT(is_hetero())
      << "The union is already hetero";
    HT_ASSERT(_union.size() == 1)
      << "Double check, the union is already hetero";
    const auto& ds = _union.at(0);
    const auto& states = ds.get_states();
    auto it = states.find(dim);
    HT_ASSERT(it != states.end())
      << "The ds must consists of the dim";
    HT_ASSERT(it->second % num == 0)
      << "Dim of ds need to be divided by the num (size of union)";
    DistributedStatesList ds_list;
    for (int32_t i = 0; i < num; i++) {
      ds_list.emplace_back(ds);
    }
    DistributedStatesUnion ret{ds_list};
    ret.set_hetero_dim(dim);
    return ret;
  }

  const DistributedStatesList& raw_data() const {
    return _union;
  }

  DistributedStates& get(size_t i) {
    return _union.at(i);
  }

  const DistributedStates& get(size_t i) const {
    return _union.at(i);
  }

  const DistributedStates get_local(size_t i) const {
    // 需要从hetero dim剔除掉hetero size
    if (is_hetero() == false) {
      HT_ASSERT(_union.size() == 1)
        << "Homo means only one ds in the union";
      return _union.at(0);
    }
    auto ds = _union.at(i);
    HT_ASSERT(ds.states(_hetero_dim) % _union.size() == 0)
      << "Hetero dim of ds need to be divided by the size of union";
    auto new_device_num = ds.get_device_num() / _union.size();
    auto new_states = ds.get_states();
    auto new_order = ds.get_order();
    auto new_zero = ds.zero();
    HT_ASSERT(new_states.find(_hetero_dim) != new_states.end())
      << "The ds must consists of the hetero dim";
    new_states[_hetero_dim] /= _union.size();
    if (new_states[_hetero_dim] == 1 && _hetero_dim >= 0) {
      new_states = ds.reduce_states(_hetero_dim);
      new_order = ds.reduce_order(_hetero_dim);
    }
    return DistributedStates(new_device_num, new_states, new_order, new_zero);
  }

  DistributedStates& get_default_ds() {
    return _union.at(0);
  }

  const DistributedStates& get_default_ds() const {
    return _union.at(0);
  }

  void add(const DistributedStates& ds) {
    _union.emplace_back(ds);
  }

  size_t size() const {
    return _union.size();
  }

  bool is_empty() const {
    return (_union.size() == 0);
  }

  bool check_equal(const DistributedStatesUnion& another) {
    if (is_hetero() != another.is_hetero() 
        || _hetero_dim != another.hetero_dim()
        || _union.size() != another.size()) {
      return false;
    }
    size_t size = _union.size();
    for (size_t i = 0; i < size; i++) {
      if (!_union.at(i).check_equal(another.get(i))) {
        return false;
      }
    }
    return true;
  }

  std::string ds_union_info() const;

 protected:
  DistributedStatesList _union;
  int32_t _hetero_dim{NULL_HETERO_DIM};
  SplitPattern _split_pattern{SplitPattern(true)};
};

class DistributedStatesHierarchy {
 public:
  DistributedStatesHierarchy() {}
  DistributedStatesHierarchy(const std::vector<DistributedStatesUnion>& ds_union_list) : _hierarchy(ds_union_list) {}

  const std::vector<DistributedStatesUnion>& raw_data() const {
    return _hierarchy;
  }

  DistributedStatesUnion& get(size_t i) {
    return _hierarchy.at(i);
  }

  const DistributedStatesUnion& get(size_t i) const {
    return _hierarchy.at(i);
  }

  DistributedStates& get_default_ds() {
    return _hierarchy.at(0).get(0);
  }

  const DistributedStates& get_default_ds() const {
    return _hierarchy.at(0).get(0);
  }

  void add(const DistributedStatesUnion& ds_union) {
    _hierarchy.emplace_back(ds_union);
  }

  size_t size() const {
    return _hierarchy.size();
  }

 protected:
  std::vector<DistributedStatesUnion> _hierarchy;
};

class DeviceGroupUnion {
 public:
  DeviceGroupUnion() {}
  DeviceGroupUnion(const DeviceGroupList& dg_list) : _union(dg_list) {
    std::unordered_map<Device, bool> device_hash;
    for (const auto& device_group : dg_list) {
      for (const auto& device : device_group.devices()) {
        if (device_hash[device] == true) {
          HT_RUNTIME_ERROR << "DeviceGroupUnion have duplicate " << device << " in different units"
            << ", the device group list is " << dg_list;
        }
        device_hash[device] = true;
      }
    }
  }

  inline const DeviceGroupList& raw_data() const {
    return _union;
  }

  inline DeviceGroup& get(size_t i) {
    return _union.at(i);
  }

  inline const DeviceGroup& get(size_t i) const {
    return _union.at(i);
  }

  inline const DeviceGroup& get(const Device& device) const {
    for (const auto& device_group : _union) {
      if (device_group.contains(device)) {
        return device_group;
      }
    }
    HT_RUNTIME_ERROR << "Can't find device " << device << " in the DeviceGroupUnion";
  }

  inline size_t get_index(const Device& device) const {
    auto size = _union.size();
    for (size_t i = 0; i < size; i++) {
      if (_union.at(i).contains(device)) {
        return i;
      }
    }
    HT_RUNTIME_ERROR << "Can't find device " << device << " in the DeviceGroupUnion";
  }

  inline bool has(const Device& device) const {
    auto size = _union.size();
    for (size_t i = 0; i < size; i++) {
      if (_union.at(i).contains(device)) {
        return true;
      }
    }
    return false;
  }

  inline void add(const DeviceGroup& dg) {
    _union.emplace_back(dg);
    std::unordered_map<Device, bool> device_hash;
    for (const auto& device_group : _union) {
      for (const auto& device : device_group.devices()) {
        if (device_hash[device] == true) {
          HT_RUNTIME_ERROR << "DeviceGroupUnion have duplicate " << device << " in different units";
        }
        device_hash[device] = true;
      }
    }
  }

  inline size_t size() const {
    return _union.size();
  }

  inline DeviceGroup all() const {
    std::vector<Device> all_devices;
    for (const auto& device_group : _union) {
      for (const auto& device : device_group.devices()) {
        all_devices.emplace_back(device);
      }
    }
    return DeviceGroup(all_devices);
  }

  inline DeviceGroupUnion dummy(size_t idx) const {
    std::vector<DeviceGroup> ret;
    auto size = _union.size();
    HT_ASSERT(idx < size)
      << "DeviceGroupUnion dummy " << idx << " is out of range";
    auto span = _union.at(idx).num_devices();
    for (size_t i = 0; i < size; i++) {
      std::vector<Device> cur_dup;
      // add real devices
      if (i == idx) {
        for (size_t j = 0; j < span; j++) {
          cur_dup.emplace_back(_union.at(idx).get(j));
        }
      }
      // add dummy devices
      else {
        for (size_t j = 0; j < span; j++) {
          cur_dup.emplace_back(Device());
        }
      }
      ret.emplace_back(cur_dup);
    }
    return DeviceGroupUnion(ret);
  }

  inline bool check_equal(const DeviceGroupUnion& another) const {
    if (_union.size() != another.size()) {
      return false;
    }
    auto size = _union.size();
    for (size_t i = 0; i < size; i++) {
      if (_union.at(i) != another.get(i)) {
        return false;
      }
    }
    return true;
  }

  // 目前认为两个union完全一样才是intra的
  inline bool is_intra_group(const DeviceGroupUnion& another) const {
    return check_equal(another);
  }

  static inline DeviceGroupUnion device_group_to_union(const DeviceGroup& dg, const hetu::graph::DistributedStates& ds, int32_t hetero_dim, int32_t num) {
    auto num_devices = dg.num_devices();
    HT_ASSERT(ds.states(hetero_dim) % num == 0)
      << "Hetero dim size of ds should be divided by the num (union size)";
    auto span = ds.states(hetero_dim) / num;
    std::vector<std::vector<Device>> union_devices_list(num);
    for (size_t i = 0; i < num_devices; i++) {
      auto state_index = ds.map_device_to_state_index(i);
      int32_t span_idx = state_index[hetero_dim] / span;
      union_devices_list[span_idx].emplace_back(dg.get(i));
    }
    DeviceGroupList ret;
    for (size_t i = 0; i < num; i++) {
      ret.emplace_back(union_devices_list[i]);
    }
    return DeviceGroupUnion(ret);
  }

  // Currently unused
  /*
  static inline DeviceGroup union_to_device_group(const DeviceGroupUnion& dg_union, const hetu::graph::DistributedStates& ds) {
    int32_t union_size = dg_union.size();
    std::vector<size_t> dup_cnt_list(union_size, 0);
    std::vector<Device> ret;
    HT_ASSERT(ds.states(-1) == union_size)
      << "Dup degree of ds should be equal to union size";
    HT_LOG_WARN_IF(ds.order(0) != -1)
      << "It is suggested to put dup at the first position in the ds order sequence"
      << ", but now the ds is: " << ds.ds_info();
    int32_t dup_num_devices = -1;
    for (size_t i = 0; i < union_size; i++) {
      DeviceGroup dup = dg_union.get(i);
      if (dup_num_devices == -1) {
        dup_num_devices = dup.num_devices();
      }
      HT_ASSERT(dup_num_devices == dup.num_devices())
        << "All dup units in the union should have the equal size";
    }
    auto num_devices = dup_num_devices * union_size;
    for (size_t i = 0; i < num_devices; i++) {
      auto state_index = ds.map_device_to_state_index(i);
      int32_t dup_idx = ds.states(-1) == 1 ? 0 : state_index[-1];
      ret.emplace_back(dg_union.get(dup_idx).get(dup_cnt_list[dup_idx]++));
    }
    // no remains
    for (auto cnt : dup_cnt_list) {
      HT_ASSERT(cnt == dup_num_devices)
        << "Assumption wrong";
    }
    return DeviceGroup(ret);
  }
  */

  // merge操作不保序
  // 只会用在comm op的placement group初始化中
  static inline DeviceGroupUnion merge(const DeviceGroupUnion& x, const DeviceGroupUnion& y) {
    HT_ASSERT(x.size() == y.size())
      << "Size of the two union to merge should be equal";
    auto size = x.size();
    DeviceGroupList ret;
    for (size_t i = 0; i < size; i++) {
      // 2024.8.16: fix a bug
      // 注意这里不能使用set而是要用vector
      // device的顺序很重要
      std::vector<Device> merged_devices;
      for (const auto& device : x.get(i).devices()) {
        merged_devices.emplace_back(device);
      }
      for (const auto& device : y.get(i).devices()) {
        if (std::find(merged_devices.begin(), merged_devices.end(), device) == merged_devices.end()) {
          merged_devices.emplace_back(device);
        }
      }
      ret.emplace_back(merged_devices);
    }
    return DeviceGroupUnion(ret);
  }

 private:
  DeviceGroupList _union; // 按hetero dp的顺序
};

using Op2DGUnionMap = std::unordered_map<OpId, DeviceGroupUnion>;

std::ostream& operator<<(std::ostream& os, const DeviceGroupUnion& dg_union);

class DeviceGroupHierarchy {
 public:
  DeviceGroupHierarchy() {}
  DeviceGroupHierarchy(const std::vector<DeviceGroupList>& dg_lists) : _hierarchy(dg_lists.begin(), dg_lists.end()) {}
  DeviceGroupHierarchy(const std::vector<DeviceGroupUnion>& dg_union_list) : _hierarchy(dg_union_list) {}

  inline const std::vector<DeviceGroupUnion>& raw_data() const {
    return _hierarchy;
  }

  inline DeviceGroupUnion& get(size_t i) {
    return _hierarchy.at(i);
  }

  inline const DeviceGroupUnion& get(size_t i) const {
    return _hierarchy.at(i);
  }

  inline void add(const DeviceGroupUnion& dg_union) {
    _hierarchy.emplace_back(dg_union);
  }

  inline size_t size() const {
    return _hierarchy.size();
  }

 private:
  std::vector<DeviceGroupUnion> _hierarchy;
};

std::ostream& operator<<(std::ostream& os, const DeviceGroupHierarchy& dg_hierarchy);

} // namespace graph
} // namespace hetu