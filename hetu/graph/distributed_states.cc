#include "hetu/graph/distributed_states.h"

namespace hetu {
namespace graph {

// for distributed states

void DistributedStates::set_placement_group(const DeviceGroup& placement_group) {
  HT_ASSERT(_device_num == -1 || placement_group.num_devices() == _device_num) 
            << "devices num in placement_group " << placement_group.num_devices() 
            << " must be equal to distributed requirement " << _device_num << "!";
  _placement_group = placement_group;
  _device_num = placement_group.num_devices();
}

void DistributedStates::set_placement(const Device& placement) {
  HT_ASSERT(_placement_group.num_devices() > 0 && _placement_group.contains(placement))
            << "the placement device " << placement << " must in placement group " << _placement_group;    
  _placement = placement;
}

void DistributedStates::set_distributed_states(const DistributedStates& dst_distributed_states) {
  HT_ASSERT(_device_num == -1 || _device_num == dst_distributed_states._device_num)
            << "device num in dst_distributed_states: " << dst_distributed_states._device_num
            << " must equal to tensor requirement: " << _device_num << "!";
  if (_device_num == -1) {
    _device_num = dst_distributed_states._device_num;
  }
  set_states(dst_distributed_states._states); // set_states会检查是否和device_num相匹配
  set_order(dst_distributed_states._order); // set_order会检查是否和states相匹配
}

bool DistributedStates::is_valid() const {
  return _device_num == 1 || (_device_num > 1 && _states.size() > 0 && _order.size() > 0); 
}

void DistributedStates::set_states(const std::unordered_map<int32_t, int32_t>& states) {
  // 必须先确定device_num再赋值states
  HT_ASSERT(_device_num != -1) << "must assign device num before set states!";
  // HT_ASSERT(states.find(-1) != states.end() && states.find(-2) != states.end()) << "partial states[-2] or duplicate states[-1] must be assigned!";
  // check states & set max dimension
  int32_t device_num = 1;
  std::unordered_map<int32_t, int32_t> res_states;
  for (auto& kv : states) {
    if (kv.second > 1) {
      res_states[kv.first] = kv.second;
      device_num *= kv.second;
    }
  }
  if (res_states.find(-2) == res_states.end()) {
    res_states[-2] = 1;
  }
  if (res_states.find(-1) == res_states.end()) {
    res_states[-1] = 1;
  }

  HT_ASSERT(device_num == _device_num) << "the devices num " << device_num 
            <<" used in states is not equal to distributed requirement " << _device_num << "!";
  _states = res_states;
}

void DistributedStates::set_order(const std::vector<int32_t>& order) {
  HT_ASSERT(_states.size() > 0) << "must assign states before set order!";
  // check order, must match states
  if (order.size() == 0) {
    // get default order
    std::vector<int32_t> keys; 
    for (auto kv : _states) {
      if (kv.second > 1) { // partial/duplicate必须大于1才能分到order
        keys.push_back(kv.first);
      }
    }
    std::sort(keys.begin(), keys.end());
    _order = keys;      
  } else {
    for (auto kv : _states) {
      if (kv.second > 1) {
        HT_ASSERT(std::find(order.begin(), order.end(), kv.first) != order.end())
                  << "order is not matched with states!";
      }
    }
    std::vector<int32_t> res_order;
    for (auto o : order) {
      if (_states[o] > 1) {
        res_order.push_back(o);
      }
    }
    _order = res_order;
  }
}

std::unordered_map<int32_t, int32_t> 
DistributedStates::combine_states(const std::pair<std::vector<int32_t>, int32_t>& src2dst) const {
  auto states = std::unordered_map<int32_t, int32_t>(_states);
  auto src = src2dst.first;
  auto dst = src2dst.second;
  int32_t value = 1;
  // 1. erase src
  for (auto s : src) {
    // HT_ASSERT(states.find(s) != states.end()) << "src " << s << " must in the states.keys !";
    HT_ASSERT(s != dst) << "cannot combine src to the same dst dimension " << s;
    if (s == -2 || s == -1) { // partial or duplicate
      value *= states[s];
      states[s] = 1;
    } else { // normal dimension
      if (states.find(s) != states.end()) {
        value *= states[s];
        states.erase(s); // 普通的dimension, value=1时就从map里删除
      }
      // dimension after s: move forward 1 offset. why?
      std::vector<int32_t> keys; 
      keys.reserve(states.size());
      for (auto kv : states) {
        if (kv.first >= 0) {
          keys.push_back(kv.first);
        }
      }
      std::sort(keys.begin(), keys.end());
      for (auto key : keys) {
        if (key > s) {
          auto val = states[key];
          states.erase(key);
          states[key - 1] = val;
        }
      }
    }
  }
  // 2. merge to dst
  if (dst == -2 || dst == -1) {
    states[dst] *= value;
  } else {
    for (auto s : src) { // 和前面往前挪一位的操作相对应
      if (s >= 0 && dst > s) {
        dst -= 1;
      }
    }
    if (states.find(dst) == states.end()) {
      states[dst] = value;
    } else {
      states[dst] *= value;
    }
  }

  return std::move(states);
}

// 合并必须保证[src]+dst的dimension是连续的
std::vector<int32_t>
DistributedStates::combine_order(const std::pair<std::vector<int32_t>, int32_t>& src2dst) const {
  auto order = std::vector<int32_t>(_order);
  auto src = src2dst.first;
  auto dst = src2dst.second;
  std::vector<int32_t> inds;
  auto collect_safe_index = [&](int32_t dim) {
    auto it = std::find(order.begin(), order.end(), dim);
    if (it != order.end()) {
      auto ind = std::distance(order.begin(), it);
      inds.push_back(ind);
    } 
    // else {
    //   HT_LOG_DEBUG << "dimension " << dim << " is not in order !";
    // }
  };
  for (auto s : src) {
    collect_safe_index(s);
  }
  collect_safe_index(dst);
  std::sort(inds.begin(), inds.end());
  if (inds.size() > 0) {
    for (size_t i = 1; i < inds.size(); i++) {
      HT_ASSERT(inds[i] == inds[0] + i) << "Cannot combine dimensions not adjacent!"; // 要combine的inds必须连续?
    }
    for (auto it = inds.rbegin(); it != inds.rend(); ++it) {
      if (it + 1 != inds.rend()) {
        order.erase(order.begin() + *it);
      } else {
        order[*it] = dst;
      }
    }
    for (size_t i = 0; i < order.size(); i++) {
      if (order[i] > 0) {
        for (auto s : src) { // 通过src消除掉多少dimension, 在src后面的dimension就往前挪多少位
          if (s >= 0 && order[i] > s) {
            order[i] -= 1;
          }
        }
      }
    }
  }

  return std::move(order);
}
  
bool DistributedStates::equal_states_and_order(const std::unordered_map<int32_t, int32_t>& states1, 
                                               const std::vector<int32_t>& order1,
                                               const std::unordered_map<int32_t, int32_t>& states2, 
                                               const std::vector<int32_t>& order2) const {
  return (states1 == states2) && (order1 == order2);                              
}

bool DistributedStates::check_equal(const DistributedStates& dst_distributed_states) const {
  auto dst_device_num = dst_distributed_states.get_device_num();
  const auto& src_states = get_states();
  const auto& src_order = get_order();
  const auto& dst_states = dst_distributed_states.get_states();
  const auto& dst_order = dst_distributed_states.get_order();
  return (_device_num == dst_device_num) && equal_states_and_order(src_states, src_order, dst_states, dst_order);
}

bool DistributedStates::check_max_dim(int32_t max_dim) const {
  if (_device_num == 1 && _order.size() == 0) {
    return true;
  }
  for (auto s : _states) {
    if (s.first >= max_dim) {
      return false;
    }
  }
  for (auto o : _order) {
    if (o >= max_dim) {
      return false;
    }
  }
  return true;
}

bool DistributedStates::check_pure_duplicate() const {
  if (_device_num == get_dim(-1)) {
    return true;
  } else {
    return false;
  }
}

bool DistributedStates::check_combine(const DistributedStates& dst_distributed_states,
                        const std::pair<std::vector<int32_t>, int32_t>& src2dst) const {
  const auto& states = combine_states(src2dst);
  const auto& order = combine_order(src2dst);

  const auto& dst_states = dst_distributed_states.get_states();
  const auto& dst_order = dst_distributed_states.get_order();
  return equal_states_and_order(states, order, dst_states, dst_order);
}

std::unordered_map<int32_t, int32_t> DistributedStates::reduce_states(int dim) const {
  auto states = std::unordered_map<int32_t, int32_t>(_states);
  if (dim == -2 || dim == -1) {
    states[dim] = 1;
  } else if (states.find(dim) != states.end()) {
    states.erase(dim);
  }
  return std::move(states);
}

std::vector<int32_t> DistributedStates::reduce_order(int dim) const {
  auto order = std::vector<int32_t>(_order);
  auto it = std::find(order.begin(), order.end(), dim); 
  if (it != order.end()) {
    order.erase(it);
  }
  return std::move(order);
}

bool DistributedStates::check_reduce_dim(const DistributedStates& dst_distributed_states, int dim) const {
  const auto& states = reduce_states(dim);
  const auto& order = reduce_order(dim);

  const auto& dst_states = dst_distributed_states.get_states();
  const auto& dst_order = dst_distributed_states.get_order();
  return equal_states_and_order(states, order, dst_states, dst_order);                                 
}

bool DistributedStates::check_allreduce(const DistributedStates& dst_distributed_states) const {
  std::pair<std::vector<int32_t>, int32_t> src2dst = {{-2}, -1};
  return states(-2) > 1 && check_combine(dst_distributed_states, src2dst);
}

// 判断逻辑有待验证
// TODO: map不要用at，因为并不存在这个key，会报异常
bool DistributedStates::check_allgather(const DistributedStates& dst_distributed_states) const {
  std::pair<std::vector<int32_t>, int32_t> src2dst = {{0}, -1};
  return states(0) > 1 && check_combine(dst_distributed_states, src2dst);
}

bool DistributedStates::check_reducescatter(const DistributedStates& dst_distributed_states) const {
  std::pair<std::vector<int32_t>, int32_t> src2dst = {{-2}, 0};
  return states(-2) > 1 && check_combine(dst_distributed_states, src2dst);
}

// 单个device上的一份, 分到多个device上每人一份duplicate
bool DistributedStates::check_boradcast(const DistributedStates& dst_distributed_states) const {
  return dst_distributed_states.states(-1) > 1 && dst_distributed_states.check_reduce_dim(*this, -1);
}

// 多个device上每人一份partial, 合并之后交给其中某个device上
bool DistributedStates::check_reduce(const DistributedStates& dst_distributed_states) const {
  return states(-2) > 1 && check_reduce_dim(dst_distributed_states, -2);
}

int32_t DistributedStates::get_dim(int32_t index) const {
  if (index == -2 || index == -1 || _states.find(index) != _states.end()) {
    return _states.at(index);
  } else {
    return 1;
  }
}

std::vector<int32_t> DistributedStates::get_loop_sizes() const {
  std::vector<int32_t> loop_sizes = {1};
  for (auto it = _order.rbegin(); it != _order.rend(); it++) {
    auto tmp_size = loop_sizes[0] * get_dim(*it);
    loop_sizes.insert(loop_sizes.begin(), tmp_size);
  }
  loop_sizes.erase(loop_sizes.begin());
  return std::move(loop_sizes);
}

std::unordered_map<int32_t, int32_t> 
DistributedStates::map_device_to_state_index(int32_t device_index) const {
  std::unordered_map<int32_t, int32_t> state_index;
  for (auto it = _order.rbegin(); it != _order.rend(); it++) {
    int32_t cur_order = *it;
    int32_t cur_state = _states.at(cur_order);
    state_index[cur_order] = device_index % cur_state;
    device_index /= cur_state;
  }
  return std::move(state_index);
}

std::string DistributedStates::ds_info() const {
  std::ostringstream os;
  os << "device num = " << _device_num << ", order = [";
  std::vector<int32_t> order(_order);
  for (auto o = order.begin(); o != order.end(); o++) {
    os << *o;
    if (o + 1 != order.end()) {
      os << ", ";
    } else {
      os << "]";
    }
  }

  std::sort(order.begin(), order.end());
  os << ", states = {";
  for (auto d = order.begin(); d != order.end(); d++) {
    os << *d << ": " << _states.at(*d);
    if (d + 1 != order.end()) {
      os << ", ";
    } else {
      os << "}";
    }
  }
  return os.str();    
}
}
}