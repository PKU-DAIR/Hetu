#include "hetu/graph/headers.h"
#include "hetu/graph/executable_graph.h"
#include "hetu/graph/profiler.h"
#include "hetu/graph/ops/kernel_links.h"

namespace hetu {
namespace graph {

/****************************************************************
 ---------------------- Parallel Attn Impl ----------------------
*****************************************************************/

static int64_t get_local_seq_len(const Tensor& input, const SyShapeList& multi_seq_lens_symbol) {
  auto& graph = input->producer()->graph();
  if (graph.type() == GraphType::EAGER) {
    HT_ASSERT(multi_seq_lens_symbol.size() == 1 && multi_seq_lens_symbol.at(0).size() == 1)
      << "eager graph should only have one strategy"
      << ", and currently not support cp";
    return multi_seq_lens_symbol.at(0).at(0)->get_val();
  }
  HT_ASSERT(multi_seq_lens_symbol.size() == graph.NUM_STRATEGY 
            && graph.COMPUTE_STRATEGY_ID < multi_seq_lens_symbol.size())
    << "strategy num should be matched";
  HT_ASSERT(!input->cur_ds_union().is_hetero() 
            || input->cur_ds_union().size() == multi_seq_lens_symbol.at(graph.COMPUTE_STRATEGY_ID).size())
    << "ds union size and seq lens symbol size should be matched";
  auto& seq_lens_symbol = multi_seq_lens_symbol[graph.COMPUTE_STRATEGY_ID];
  if (graph.USE_HETERO_ID) {
    return seq_lens_symbol.at(graph.CUR_HETERO_ID)->get_val();
  } else {
    if (input->cur_ds_union().is_hetero()) {
      auto idx = input->inferred_local_placement_group_idx();
      return seq_lens_symbol.at(idx)->get_val();
    } else {
      for (auto& symbol : seq_lens_symbol) {
        HT_ASSERT(symbol->get_val() == seq_lens_symbol.at(0)->get_val())
          << "all seq lens should be equal in homo setting";
      }
      return seq_lens_symbol.at(0)->get_val();
    }
  }
}

static int64_t get_local_dcp_idx(const Tensor& input) {
  auto local_device = hetu::impl::comm::GetLocalDevice();
  if (input->cur_ds_union().is_hetero()) {
    HT_ASSERT(input->placement_group_union().has(local_device))
      << "ParallelAttnOp input " << input << " should already deduced the pg union and be local";
    return input->placement_group_union().get_index(local_device);
  } 
  const auto& ds = input->cur_ds_union().get(0);
  const auto& pg = input->placement_group_union().get(0);
  auto it = std::find(pg.devices().begin(), pg.devices().end(), local_device);
  HT_ASSERT(it != pg.devices().end())
    << "get_local_dcp_idx should ensure local device is in the placement group";
  auto device_idx = std::distance(pg.devices().begin(), it);
  // 查找device idx对应到ds中是第几个split 0
  auto state = ds.map_device_to_state_index(device_idx);
  return state[0];
}

static std::tuple<int64_t, DeviceGroupList, std::vector<int64_t>> get_local_ring(const Tensor& input, const SyShapeList& multi_seq_lens_symbol, const SyShapeList& multi_cp_group_symbol) {
  auto& graph = input->producer()->graph();
  if (graph.type() == GraphType::EAGER) {
    HT_ASSERT(multi_seq_lens_symbol.size() == 1 && multi_seq_lens_symbol.at(0).size() == 1
              && multi_cp_group_symbol.size() == 1 && multi_cp_group_symbol.at(0).size() == 1)
      << "eager graph should only have one strategy"
      << ", and currently not support cp";
    return std::make_tuple(0, DeviceGroupList{DeviceGroup{{input->producer()->eager_device()}}}, std::vector<int64_t>{multi_seq_lens_symbol.at(0).at(0)->get_val()});
  }
  auto local_device = hetu::impl::comm::GetLocalDevice();
  HT_ASSERT(multi_seq_lens_symbol.size() == multi_cp_group_symbol.size() 
            && multi_seq_lens_symbol.size() == graph.NUM_STRATEGY 
            && graph.COMPUTE_STRATEGY_ID < multi_seq_lens_symbol.size())
    << "strategy num should be matched";
  auto& seq_lens_symbol = multi_seq_lens_symbol[graph.COMPUTE_STRATEGY_ID];
  auto& cp_group_symbol = multi_cp_group_symbol[graph.COMPUTE_STRATEGY_ID];
  auto dcp_size = cp_group_symbol.size();
  HT_ASSERT(dcp_size == seq_lens_symbol.size())
    << "dcp size should be matched";
  int64_t ring_idx;
  DeviceGroupList tp_group_list;
  std::vector<int64_t> seq_len_list;
  const auto& ds_union = input->cur_ds_union();
  DeviceGroupUnion pg_union = input->placement_group_union();
  if (!ds_union.is_hetero()) {
    // 在第0维将placement group切成dcp size份
    // 每一份即是一个tp
    pg_union = DeviceGroupUnion::device_group_to_union(pg_union.get(0), ds_union.get(0), 0, dcp_size);
  }
  auto cur_dcp_idx = get_local_dcp_idx(input);
  auto cur_cp_group_idx = cp_group_symbol.at(cur_dcp_idx)->get_val();
  for (size_t i = 0; i < dcp_size; i++) {
    auto cp_group_idx = cp_group_symbol[i]->get_val();
    if (cp_group_idx == cur_cp_group_idx) {
      tp_group_list.emplace_back(pg_union.get(i));
      seq_len_list.emplace_back(seq_lens_symbol[i]->get_val());
    }
    if (i == cur_dcp_idx) {
      ring_idx = tp_group_list.size() - 1;
    }
  }
  return std::make_tuple(ring_idx, tp_group_list, seq_len_list);
}

AttnCommRing::AttnCommRing(const Operator& op, const hetu::impl::comm::NCCLCommunicationGroup& nccl_comm_group, StreamIndex stream_idx, 
                           int64_t ring_idx, const DeviceGroupList& tp_group_list, const std::vector<int64_t>& seq_len_list,
                           int64_t batch_size, int64_t q_num_heads, int64_t kv_num_heads, int64_t head_dim,
                           double softmax_scale, double p_dropout, size_t kv_storage_size)
  : _nccl_comm_group(nccl_comm_group), _stream_idx(stream_idx), _ring_idx(ring_idx), _tp_group_list(tp_group_list), _seq_len_list(seq_len_list),
    _batch_size(batch_size), _q_num_heads(q_num_heads), _kv_num_heads(kv_num_heads), _head_dim(head_dim),
    _softmax_scale(softmax_scale), _p_dropout(p_dropout), _kv_storage_size(kv_storage_size) {
  _ring_size = _tp_group_list.size();
  HT_ASSERT(_ring_size == _seq_len_list.size() && _ring_idx < _ring_size)
    << "ring size should be aligned";
  _attn_info_list.reserve(_ring_size * _ring_size);
  _local_device = hetu::impl::comm::GetLocalDevice();
  HT_ASSERT(_tp_group_list[_ring_idx].contains(_local_device))
    << "ring idx should be aligned";
  if (dynamic_cast<ExecutableGraph&>(op->graph())._parallel_attn_flag > 0) {
    _need_profile = true;
    for (size_t i = 0; i < _ring_size; i++) {
      _comm_profile_start_event_list.emplace_back(std::make_shared<hetu::impl::CUDAEvent>(_local_device));
      _comm_profile_end_event_list.emplace_back(std::make_shared<hetu::impl::CUDAEvent>(_local_device));
      _attn_profile_start_event_list.emplace_back(std::make_shared<hetu::impl::CUDAEvent>(_local_device));
      _attn_profile_end_event_list.emplace_back(std::make_shared<hetu::impl::CUDAEvent>(_local_device));
      _corr_profile_start_event_list.emplace_back(std::make_shared<hetu::impl::CUDAEvent>(_local_device));
      _corr_profile_end_event_list.emplace_back(std::make_shared<hetu::impl::CUDAEvent>(_local_device));
      _grad_profile_start_event_list.emplace_back(std::make_shared<hetu::impl::CUDAEvent>(_local_device));
      _grad_profile_end_event_list.emplace_back(std::make_shared<hetu::impl::CUDAEvent>(_local_device));
    }
  }  
  auto env = std::getenv("HETU_PARALLEL_ATTN_SPLIT_PATTERN");
  if (env != nullptr) {
    if (std::string(env) == "NORMAL") {
      _attn_split_pattern = AttnSplitPattern::NORMAL;
    } else if (std::string(env) == "STRIPE") {
      _attn_split_pattern = AttnSplitPattern::STRIPE;
      HT_RUNTIME_ERROR << "currently stripe attn is not implemented";
    } else if (std::string(env) == "SYM") {
      _attn_split_pattern = AttnSplitPattern::SYM;
    } else {
      HT_RUNTIME_ERROR << "Unknown hetu parallel attn split pattern: " + std::string(env);
    }
  } else {
    // 默认使用NORMAL
    _attn_split_pattern = AttnSplitPattern::NORMAL;
  }
}

void AttnCommRing::GenerateAttnInfo() {
  HT_ASSERT(_attn_info_list.empty())
    << "GenerateAttnInfo can only run once";
  // q所在的rank是i
  for (size_t i = 0; i < _ring_size; i++) {
    // k和v所在的rank是j
    for (size_t j = 0; j < _ring_size; j++) {
      AttnMask attn_mask = AttnMask::EMPTY;
      int64_t valid_len = -1;
      // 自己rank上的qkv
      // 直接使用causal mask即可
      if (i == j) {
        // 什么都不用做
        attn_mask = AttnMask::CAUSAL;
      } 
      else {
        if (_attn_split_pattern == AttnSplitPattern::NORMAL) {
          if (i < j) {
            attn_mask = AttnMask::EMPTY;
          } else {
            attn_mask = AttnMask::FULL;
          }
        } else if (_attn_split_pattern == AttnSplitPattern::SYM) {
          // TODO: 目前默认前一半后一半均匀切
          // 后一半的q才是valid的
          auto determine_rank = std::min(i, j);
          // 这里的alg需要保证所有视角下的valid len一致
          // 除2具有这个性质
          valid_len = _seq_len_list[determine_rank] / 2; 
          if (i < j) {
            // mask -> valid len row at bottom
            attn_mask = AttnMask::ROW; 
          } else {
            // mask -> valid len col at left
            attn_mask = AttnMask::COL;
          }
        } 
        // 其余baseline有待实现
        else {
          HT_RUNTIME_ERROR << "NotImplementedError";
        }
      }
      // 建立rank-i上q与rank-j上的k和v之间的info
      _attn_info_list.emplace_back(std::make_shared<AttnInfo>(attn_mask, 
                                                              valid_len,
                                                              _seq_len_list[i],
                                                              _seq_len_list[j]));
    }
  }
}

void AttnCommRing::PrepareKVBlocks(const NDArray& local_k, const NDArray& local_v, 
                                   bool reuse_local_kv_storage, bool piggyback_grad) {
  HT_ASSERT(!_is_storage_prepared && _kv_block_to_storage_map.empty())
    << "storage of a ring should only prepared once";
  if (piggyback_grad) {
    HT_LOG_WARN_IF(_kv_storage_size > 2)
      << "more than 2 storage will only raise memory peak consumption but have no benefit on the efficiency"
      << ", if you piggyback the gradients in the KV blocks (now only support this way)";
  }
  // TODO: 更好的ring alg
  // numel取所有seq_len中的最大值来分配storage
  int64_t skip_seq_len_numel = 1;
  DataType dtype = local_k->dtype();
  for (size_t i = 0; i < 4; i++) {
    // skip seq_len dim
    if (i == 1) {
      continue;
    }
    skip_seq_len_numel *= local_k->shape(i);
  }
  // 堆叠k和v
  skip_seq_len_numel *= 2;
  // 表示每个storage上要存的kv的最长的seq_len以及piggyback时dkv的最长的seq_len
  std::vector<int64_t> kv_max_seq_len_list, dkv_max_seq_len_list;
  for (size_t i = 0; i < _kv_storage_size; i++) {
    kv_max_seq_len_list.emplace_back(-1);
    dkv_max_seq_len_list.emplace_back(-1);
  }
  // 建立kv block到kv storage的映射
  // 最一开始拥有的local的kv映射到0号storage上
  // 下一轮即将获得的映射到1号storage上
  // 依次类推
  int64_t cur_storage_idx = 0;
  int64_t cur_block_idx = _ring_idx;
  HT_ASSERT(_kv_block_to_storage_map.empty())
    << "_kv_block_to_storage_map should be empty";
  for (size_t i = 0; i < _ring_size; i++) {
    HT_ASSERT(_kv_block_to_storage_map.find(cur_block_idx) == _kv_block_to_storage_map.end())
      << "block idx duplicate in _kv_block_to_storage_map";
    _kv_block_to_storage_map[cur_block_idx] = cur_storage_idx;
    cur_block_idx = (cur_block_idx + _ring_size - 1) % _ring_size;
    cur_storage_idx = (cur_storage_idx + 1) % _kv_storage_size;
  }
  auto final_storage_idx = cur_storage_idx;
  auto final_next_storage_idx = (final_storage_idx + 1) % _kv_storage_size;
  // 注意piggyback的dkv是上一轮的(+1)而不是下一轮的(-1)
  // 这个与storage的顺序正好相反
  for (size_t i = 0; i < _ring_size; i++) {
    auto storage_idx = _kv_block_to_storage_map[i];
    auto kv_seq_len = _seq_len_list.at(i);
    auto dkv_seq_len = _seq_len_list.at((i + 1) % _ring_size);
    kv_max_seq_len_list.at(storage_idx) = std::max(kv_max_seq_len_list.at(storage_idx), kv_seq_len);
    dkv_max_seq_len_list.at(storage_idx) = std::max(dkv_max_seq_len_list.at(storage_idx), dkv_seq_len);
  }
  kv_max_seq_len_list.at(final_storage_idx) = std::max(kv_max_seq_len_list.at(final_storage_idx), _seq_len_list.at(_ring_idx));
  dkv_max_seq_len_list.at(final_storage_idx) = std::max(dkv_max_seq_len_list.at(final_storage_idx), _seq_len_list.at((_ring_idx + 1) % _ring_size));
  // 捎带grad的情形还需要再通信一轮才可以得到local dkv
  if (piggyback_grad) {
    kv_max_seq_len_list.at(final_next_storage_idx) = std::max(kv_max_seq_len_list.at(final_next_storage_idx), _seq_len_list.at((_ring_idx + _ring_size - 1) % _ring_size));
    dkv_max_seq_len_list.at(final_next_storage_idx) = std::max(dkv_max_seq_len_list.at(final_next_storage_idx), _seq_len_list.at(_ring_idx));
  }
  HT_ASSERT(_kv_storage_list.empty())
    << "_kv_storage_list should be empty";
  // already support support cp degree not divided by num of storage
  /*
  // TODO: let KVBlock 0 and KVBlock _ring_size (on rank 0) be two KVBlocks
  // and we should use KVBlock _ring_size to SaveCtx 
  HT_ASSERT(_ring_size % _kv_storage_size == 0)
    << "Currently only support that cp degree could be divided by num of storage"
    << ", otherwise one KVBlock may have two possible storage when doing the ring attn";
  */
  for (size_t i = 0; i < _kv_storage_size; i++) {
    auto kv_max_seq_len = kv_max_seq_len_list.at(i);
    auto dkv_max_seq_len = dkv_max_seq_len_list.at(i);
    // HT_LOG_DEBUG << "[ParallelAttn]: attn storage " << i << " kv_max_seq_len is " << kv_max_seq_len << " and dkv_max_seq_len (if piggyback grad) is " << dkv_max_seq_len;
    auto numel = (kv_max_seq_len + (piggyback_grad ? dkv_max_seq_len : 0)) * skip_seq_len_numel;
    auto size = (dtype == kFloat4 || dtype == kNFloat4) ? ((numel + 1) / 2) * DataType2Size(dtype) : numel * DataType2Size(dtype);
    if (reuse_local_kv_storage && (_ring_idx % _kv_storage_size == i)) {
      HT_ASSERT(local_k->storage() == local_v->storage() && local_k->storage()->size() == size)
        << "reuse_local_kv_storage should only be set when it really could be reused";
      _kv_storage_list.emplace_back(std::make_shared<AttnStorage>(local_k->storage())); // local_kv的storage拿来充公用作ring中的buffer
      continue;
    }
    // 实际分配显存
    _kv_storage_list.emplace_back(std::make_shared<AttnStorage>(std::make_shared<NDArrayStorage>(AllocFromMemoryPool(_local_device, 
                                                                                                                     size, 
                                                                                                                     Stream(_local_device, _stream_idx)))));
  }
  HT_ASSERT(_kv_block_list.empty())
    << "_kv_block_list should be empty";
  // _kv_block_list = std::vector<std::shared_ptr<AttnBlock>>(_ring_size, nullptr);
  for (size_t i = 0; i < _ring_size; i++) {
    auto storage_idx = _kv_block_to_storage_map[i];
    auto kv_seq_len = _seq_len_list.at(i);
    auto dkv_seq_len = _seq_len_list.at((i + 1) % _ring_size);
    auto kv_block = std::make_shared<AttnBlock>(_local_device,
                                                HTShape{_batch_size * 2, kv_seq_len, _kv_num_heads, _head_dim},
                                                dtype, 
                                                "KVBlock_" + std::to_string(i),
                                                piggyback_grad,
                                                HTShape{_batch_size * 2, dkv_seq_len, _kv_num_heads, _head_dim});
    // HT_LOG_DEBUG << "[ParallelAttn]: " << _local_device << " KVBlock " << i << " has kv_seq_len = " << kv_seq_len << " and dkv_seq_len = " << dkv_seq_len << ", bind to storage " << storage_idx;
    kv_block->bind_attn_storage(_kv_storage_list.at(storage_idx));
    _kv_block_list.emplace_back(kv_block);
  }
  // workaround: 最后一次通信（piggyback grad甚至还有额外一次）单独处理
  // 原因是当cp不能整除storage数目时
  // local kv block所bind的storage与初始的storage不是一个
  _final_kv_block = std::make_shared<AttnBlock>(_local_device,
                                                HTShape{_batch_size * 2, _seq_len_list.at(_ring_idx), _kv_num_heads, _head_dim},
                                                dtype, 
                                                "KVBlock_final",
                                                piggyback_grad,
                                                HTShape{_batch_size * 2, _seq_len_list.at((_ring_idx + 1) % _ring_size), _kv_num_heads, _head_dim});
  _final_kv_block->bind_attn_storage(_kv_storage_list.at(final_storage_idx));
  if (piggyback_grad) {
    _final_next_kv_block = std::make_shared<AttnBlock>(_local_device,
                                                       HTShape{_batch_size * 2, _seq_len_list.at((_ring_idx + _ring_size - 1) % _ring_size), _kv_num_heads, _head_dim},
                                                       dtype, 
                                                       "KVBlock_final_next",
                                                       piggyback_grad,
                                                       HTShape{_batch_size * 2, _seq_len_list.at(_ring_idx), _kv_num_heads, _head_dim});
    _final_next_kv_block->bind_attn_storage(_kv_storage_list.at(final_next_storage_idx));
  }
  // 将初始时的local kv放入到对应的ring buffer中
  auto block_local_k = _kv_block_list[_ring_idx]->get_4d_k();
  auto block_local_v = _kv_block_list[_ring_idx]->get_4d_v();
  NDArray::contiguous(local_k, _stream_idx, block_local_k);
  NDArray::contiguous(local_v, _stream_idx, block_local_v);
}

void AttnCommRing::PrepareStorageFwd(const NDArray& local_q, const NDArray& local_k, const NDArray& local_v, const NDArray& local_out) {
  // HT_LOG_DEBUG << "[ParallelAttn]: PrepareStorageFwd begin";
  HT_ASSERT(local_q.is_defined() && local_k.is_defined() && local_v.is_defined() && local_out.is_defined())
    << "PrepareStorageFwd inputs should be allocated in advance";
  HT_ASSERT(!_is_storage_prepared)
    << "storage of a ring should only prepared once";
  HT_ASSERT(local_q->shape().size() == 4
            && local_k->shape().size() == 4
            && local_k->shape() == local_v->shape()
            && local_q->shape() == local_out->shape()
            && local_q->dtype() == local_k->dtype() 
            && local_k->dtype() == local_v->dtype()
            && local_q->dtype() == local_out->dtype())
    << "local_q, local_k and local_v should be [batch_size, seq_len, num_heads, head_dim] format" 
    << ", and dtype should be equal";
  DataType dtype = local_k->dtype();
  // 1、分配ring过程中所需的所有kv block的storage
  PrepareKVBlocks(local_k, local_v);
  // 2、分配local q
  // local_q直接使用即可
  // _local_q = local_q;
  // TODO: 验证q连续或非连续情形下FlashAttn的性能差异
  _local_q = NDArray::contiguous(local_q, _stream_idx);
  // 3、分配ring过程中所有的rng state
  for (size_t i = 0; i < _ring_size; i++) {
    _rng_state_list.emplace_back(NDArray::empty(HTShape{2},
                                                Device(kCPU),
                                                kInt64,
                                                _stream_idx));
  }
  // 4、分配临时的_softmax_lse和_out以及最终累积的_acc_softmax_lse_transposed和_acc_out
  // 注意lse强制要求是fp32的
  _softmax_lse = NDArray::empty(HTShape{_batch_size, _q_num_heads, _seq_len_list[_ring_idx]},
                                _local_device, 
                                kFloat, 
                                _stream_idx);
  _out = NDArray::empty(HTShape{_batch_size, _seq_len_list[_ring_idx], _q_num_heads, _head_dim},
                        _local_device, 
                        dtype, 
                        _stream_idx);
  // 只有fwd需要ExecCorr
  // 使用transpose好的
  // 在SaveCtx时再transpose回来
  _acc_softmax_lse_transposed = NDArray::empty(HTShape{_batch_size, _seq_len_list[_ring_idx], _q_num_heads, 1},
                                               _local_device,
                                               kFloat,
                                               _stream_idx);
  _acc_out = local_out;
  // 结束
  _is_storage_prepared = true;
  // HT_LOG_DEBUG << "[ParallelAttn]: PrepareStorageFwd end";
}

void AttnCommRing::PrepareStorageBwd(const std::shared_ptr<AttnCtx>& attn_ctx, const NDArray& grad_output, const NDArray& local_dq) {
  // HT_LOG_DEBUG << "[ParallelAttn]: PrepareStorageBwd begin";
  HT_ASSERT(grad_output.is_defined() && local_dq.is_defined())
    << "PrepareStorageBwd inputs should be allocated in advance";
  HT_ASSERT(!_is_storage_prepared)
    << "storage of a ring should only prepared once";
  // 1、直接拿到ctx中的东西
  // 这一部分早就在fwd时分配了不过一直没有被释放
  _local_q = attn_ctx->q;
  const auto& local_k = attn_ctx->k;
  const auto& local_v = attn_ctx->v;
  _acc_out = attn_ctx->acc_out;
  _acc_softmax_lse = attn_ctx->acc_softmax_lse;
  _rng_state_list = attn_ctx->rng_state_list;
  // 2、分配ring过程中所需的所有kv block的storage
  // 同时piggyback下一轮的acc_dk与acc_dv
  PrepareKVBlocks(local_k, local_v, false, true);
  // 3、分配grad_output和_acc_dq
  _grad_output = grad_output;
  _acc_dq = local_dq;
  // 4、分配临时的dq、dk和dv的storage
  // 这里按最长的seq_len去分配
  // 目前走mempool临时去分配
  // 结束
  _is_storage_prepared = true;
  // HT_LOG_DEBUG << "[ParallelAttn]: PrepareStorageBwd end";
}

void AttnCommRing::SaveCtx(const std::shared_ptr<AttnCtx>& attn_ctx) {
  attn_ctx->q = _local_q;
  // 注意k和v共有一个storage而q则单独有一个storage
  // 需要保证cur的storage上的通讯已经结束
  _final_kv_block->wait_until_comm_done(Stream(_local_device, _stream_idx));
  attn_ctx->k = _final_kv_block->get_4d_k();
  attn_ctx->v = _final_kv_block->get_4d_v();
  attn_ctx->acc_out = _acc_out;
  HTShape temp_shape = {_acc_softmax_lse_transposed->shape(0), _acc_softmax_lse_transposed->shape(1), _acc_softmax_lse_transposed->shape(2)};
  // [batch_size, seq_len, num_heads, 1] -> [batch_size, seq_len, num_heads] -> [batch_size, num_heads, seq_len]
  attn_ctx->acc_softmax_lse = NDArray::contiguous(NDArray::permute(NDArray::reshape(_acc_softmax_lse_transposed, temp_shape, _stream_idx), HTAxes{0, 2, 1}, _stream_idx), _stream_idx);
  attn_ctx->rng_state_list = _rng_state_list;
}

void AttnCommRing::SaveGradient(NDArray& local_dq, NDArray& local_dk, NDArray& local_dv) {
  HT_ASSERT(local_dq.is_defined() && local_dk.is_defined() && local_dv.is_defined())
    << "SaveGradient inputs should be allocated in advance";
  HT_ASSERT(local_dq->storage() == _acc_dq->storage())
    << "currently _acc_dq is local_dq (avoid another memory copy and consumption)";
  // 需要保证next的storage上的通讯已经结束
  // 因为它上面捎带了前一个（即cur）的acc的grad
  _final_next_kv_block->wait_until_comm_done(Stream(_local_device, _stream_idx));
  NDArray::copy(_final_next_kv_block->get_4d_acc_dk(), _stream_idx, local_dk);
  NDArray::copy(_final_next_kv_block->get_4d_acc_dv(), _stream_idx, local_dv);
}

void AttnCommRing::ExecCorr(const NDArray& out, const NDArray& softmax_lse,
                            NDArray& acc_out, NDArray& acc_softmax_lse_transposed,
                            bool is_first_time) {
  // old version:
  // new_lse = lse + torch.log(1 + torch.exp(block_lse - lse))
  // torch.exp(lse - new_lse) * out + torch.exp(block_lse - new_lse) * block_out
  // new version:
  // out = out - F.sigmoid(block_lse - lse) * (out - block_out)
  // lse = lse - F.logsigmoid(lse - block_lse)
  // For additional context and discussion, please refer to:
  // https://github.com/zhuzilin/ring-flash-attention/pull/34#issuecomment-2076126795
  // TODO: kernel fusion
  // HT_LOG_DEBUG << "[ParallelAttn]: ExecCorr begin";
  HT_ASSERT(out.is_defined() && softmax_lse.is_defined() 
            && acc_out.is_defined() && acc_softmax_lse_transposed.is_defined())
    << "ExecCorr inputs should be allocated in advance";
  HT_ASSERT(out->shape().size() == 4 && softmax_lse->shape().size() == 3
            && acc_out->shape().size() == 4 && acc_softmax_lse_transposed->shape().size() == 4)
    << "out should be [batch_size, seq_len, num_heads, head_dim] format" 
    << ", and softmax_lse should be [batch_size, num_heads, seq_len] format"
    << ", and acc_out should be [batch_size, seq_len, num_heads, head_dim] format"
    << ", and acc_softmax_lse_transposed should be [batch_size, seq_len, num_heads, 1] format"
    << ", but found their shapes " << out->shape() << ", " << softmax_lse->shape() 
    << ", " << acc_out->shape() << ", " << acc_softmax_lse_transposed->shape();
  HTShape transpose_shape = {softmax_lse->shape(0), softmax_lse->shape(2), softmax_lse->shape(1), 1};
  if (is_first_time) {
    NDArray::copy(out, _stream_idx, acc_out);
    NDArray::reshape(NDArray::permute(softmax_lse, HTAxes{0, 2, 1}, _stream_idx), transpose_shape, _stream_idx, acc_softmax_lse_transposed);
    return;
  } 
  // softmax_lse -> softmax_lse_transposed
  // [batch_size, seq_len, num_heads, 1]
  auto intermediate = NDArray::reshape(NDArray::permute(softmax_lse, HTAxes{0, 2, 1}, _stream_idx), transpose_shape, _stream_idx); 
  // intermediate = softmax_lse_transposed - acc_softmax_lse_transposed
  // [batch_size, seq_len, num_heads, 1]
  NDArray::sub(intermediate, acc_softmax_lse_transposed, _stream_idx, intermediate);
  // HT_LOG_DEBUG << "[ParallelAttn]: intermediate = " << intermediate;
  // acc_out = acc_out - sigmoid(intermediate) * (acc_out - out)
  // [batch_size, seq_len, num_heads, head_dim]
  // here NDArray::mul is a broadcast mul
  // TODO: lse is fp 32 and out is bf16, need to support mul across different data types
  NDArray::sub(acc_out, NDArray::mul(NDArray::to(NDArray::sigmoid(intermediate, _stream_idx), _local_device, out->dtype()), NDArray::sub(acc_out, out, _stream_idx), _stream_idx), _stream_idx, acc_out);
  // acc_softmax_lse_transposed = acc_softmax_lse_transposed - logsigmoid(-intermediate)
  // [batch_size, seq_len, num_heads, 1]
  NDArray::sub(acc_softmax_lse_transposed, NDArray::logsigmoid(intermediate, true, _stream_idx), _stream_idx, acc_softmax_lse_transposed);                
  // HT_LOG_DEBUG << "[ParallelAttn]: ExecCorr end";
}

void AttnCommRing::ExecFlashAttn(int64_t q_idx, int64_t kv_idx, 
                                 const NDArray& q, const NDArray& k, const NDArray& v,
                                 NDArray& out, NDArray& softmax_lse, NDArray& rng_state,
                                 bool is_bwd, NDArray grad_output, NDArray dq, NDArray dk, NDArray dv) {
  // HT_LOG_DEBUG << "[ParallelAttn]: ExecFlashAttn begin";
  HT_ASSERT(q.is_defined() && k.is_defined() && v.is_defined()
            && out.is_defined() && softmax_lse.is_defined() && rng_state.is_defined())
    << "ExecFlashAttn inputs should be allocated in advance";
  if (is_bwd) {
    HT_ASSERT(grad_output.is_defined() && dq.is_defined() && dk.is_defined() && dv.is_defined())
      << "ExecFlashAttn inputs should be allocated in advance";
  }
  HT_ASSERT(q->shape().size() == 4
            && k->shape().size() == 4
            && k->shape() == v->shape())
    << "q, k and v should be [batch_size, seq_len, num_heads, head_dim] format"
    << ", and shapes of k and v should be equal";
  auto& attn_info = _attn_info_list.at(q_idx * _ring_size + kv_idx);
  bool is_causal = attn_info->is_causal();
  NDArray q_slice = q, k_slice = k, v_slice = v;
  HT_ASSERT(attn_info->get_mask() != AttnMask::EMPTY)
    << "currently empty attn is handled outside this func";
  if (attn_info->get_mask() == AttnMask::ROW) {
    HTShape begin_pos = {0, q->shape(1) - attn_info->get_valid_len(), 0, 0};
    HTShape slice_shape = q->shape();
    slice_shape[1] = attn_info->get_valid_len();
    q_slice = NDArray::slice(q, begin_pos, slice_shape, _stream_idx);
  } 
  if (attn_info->get_mask() == AttnMask::COL) {
    HTShape begin_pos = {0, 0, 0, 0};
    HTShape slice_shape = k->shape();
    slice_shape[1] = attn_info->get_valid_len();
    k_slice = NDArray::slice(k, begin_pos, slice_shape, _stream_idx);
    v_slice = NDArray::slice(v, begin_pos, slice_shape, _stream_idx);
  }
  NDArray empty_ndarray = NDArray();
  if (!is_bwd) {
    // HT_LOG_DEBUG << "[ParallelAttn]: FlashAttnCuda begin, q idx = " << q_idx << " and kv idx = " << kv_idx << ", q_slice is " << q_slice << " and k_slice (similar to v_slice) is " << k_slice;
    // 这里的softmax_lse与out都是一个block的局部的输出
    HT_DISPATCH_KERNEL_CUDA_ONLY(DeviceType::CUDA, "FlashAttn", hetu::impl::FlashAttn,
                                 q_slice, k_slice, v_slice, out, empty_ndarray,
                                 empty_ndarray, empty_ndarray, empty_ndarray, softmax_lse,
                                 empty_ndarray, rng_state, _p_dropout, _softmax_scale,
                                 is_causal, false, Stream(_local_device, _stream_idx));
    // HT_LOG_DEBUG << "[ParallelAttn]: FlashAttnCuda end, out is " << out;
  } else {
    // HT_LOG_DEBUG << "[ParallelAttn]: FlashAttnGradientCuda begin";
    // 这里的softmax_lse与out则是全部block累积的
    HT_DISPATCH_KERNEL_CUDA_ONLY(DeviceType::CUDA, "FlashAttnGradient", hetu::impl::FlashAttnGradient,
                                 grad_output, q_slice, k_slice, v_slice, out, softmax_lse, rng_state, 
                                 dq, dk, dv, _p_dropout, _softmax_scale,
                                 is_causal, Stream(_local_device, _stream_idx));
    // HT_LOG_DEBUG << "[ParallelAttn]: FlashAttnGradientCuda end";
  }
  // HT_LOG_DEBUG << "[ParallelAttn]: ExecFlashAttn end";
}

void AttnCommRing::ExecComm(const NDArray& send_data, const NDArray& recv_data,
                            const std::vector<Device>& dst_devices, const std::vector<Device>& src_devices,
                            const Stream& comm_stream, bool is_3d) {
  // HT_LOG_DEBUG << "[ParallelAttn]: ExecComm begin";
  HT_ASSERT(send_data.is_defined() && recv_data.is_defined())
    << "ExecComm inputs should be allocated in advance";
  // auto comp_stream = Stream(_local_device, _stream_idx);
  // auto start_event = std::make_shared<hetu::impl::CUDAEvent>(_local_device);
  // start_event->Record(comp_stream);
  // ring-send-recv
  auto dst_split_num = dst_devices.size();
  auto src_split_num = src_devices.size();
  size_t num_heads_dim;
  // 在batch维度堆叠
  // k和v或者dk和dv
  // 使用4d的
  if (!is_3d) {
    num_heads_dim = 2;
    HT_ASSERT(send_data->shape().size() == 4 
              && recv_data->shape().size() == 4
              && send_data->shape(2) % dst_split_num == 0
              && recv_data->shape(2) % src_split_num == 0)
      << "send & recv data should be kv or dkv, whose shape is in [2 * batch_size, seq_len, num_heads, head_dim] format"
      << ", and the split happens on num_heads dim (should be integer after splitted)";
  }
  // 在batch*seq维度堆叠
  // kv和dkv的seq_len可能不一致
  // 需要使用3d的
  else {
    num_heads_dim = 1;
    HT_ASSERT(send_data->shape().size() == 3 
             && recv_data->shape().size() == 3
             && send_data->shape(1) % dst_split_num == 0
             && recv_data->shape(1) % src_split_num == 0)
    << "piggyback send & recv data should be kv with dkv, whose shape is in [2 * batch_size * (kv_seq_len + dkv_seq_len), num_heads, head_dim] format"
    << ", and the split happens on num_heads dim (should be integer after splitted)";
  }
  std::vector<hetu::impl::comm::CommTask> ring_send_recv_tasks;
  HTShape send_data_begin_pos(send_data->shape().size(), 0), recv_data_begin_pos(recv_data->shape().size(), 0);
  HTShape send_data_slice_shape = send_data->shape(), recv_data_slice_shape = recv_data->shape();
  send_data_slice_shape[num_heads_dim] /= dst_split_num;
  recv_data_slice_shape[num_heads_dim] /= src_split_num;
  // 当split num是1的时候
  // slice和contiguous操作都在原地做
  // 因此实际上没有新的显存的分配与拷贝
  for (size_t i = 0; i < dst_split_num; i++) {
    auto send_data_slice = NDArray::contiguous(NDArray::slice(send_data, send_data_begin_pos, send_data_slice_shape, comm_stream.stream_index()), comm_stream.stream_index());
    if (dst_split_num == 1) {
      HT_ASSERT(send_data_slice->storage() == send_data->storage())
        << "contiguous + slice should be inplace";
    }
    ring_send_recv_tasks.push_back(_nccl_comm_group->ISend(send_data_slice, hetu::impl::comm::DeviceToWorldRank(dst_devices[i])));
    send_data_begin_pos[num_heads_dim] += send_data_slice_shape[num_heads_dim];
  }
  NDArrayList recv_data_slices;
  for (size_t i = 0; i < src_split_num; i++) {
    auto recv_data_slice = NDArray::contiguous(NDArray::slice(recv_data, recv_data_begin_pos, recv_data_slice_shape, comm_stream.stream_index()), comm_stream.stream_index());
    if (src_split_num == 1) {
      HT_ASSERT(recv_data_slice->storage() == recv_data->storage())
        << "contiguous + slice should be inplace";
    }
    ring_send_recv_tasks.push_back(_nccl_comm_group->IRecv(recv_data_slice, hetu::impl::comm::DeviceToWorldRank(src_devices[i])));
    recv_data_begin_pos[num_heads_dim] += recv_data_slice_shape[num_heads_dim];
    recv_data_slices.emplace_back(std::move(recv_data_slice));
  }
  // 实际的ring comm
  _nccl_comm_group->BatchedISendIRecv(ring_send_recv_tasks);
  // auto end_event = std::make_shared<hetu::impl::CUDAEvent>(_local_device);
  // end_event->Record(comp_stream);
  // comp_stream.Sync();
  // HT_LOG_INFO << "src_split_num = " << src_split_num << ", dst_split_num = " << dst_split_num  << ", BatchedISendIRecv Time Elapse: " << end_event->TimeSince(*start_event) * 1.0 / 1e6 << "ms";
  if (src_split_num == 1) {
    HT_ASSERT(recv_data_slices[0]->storage() == recv_data->storage())
      << "no data alloc & copy should happen";
    // HT_LOG_DEBUG << "[ParallelAttn]: ExecComm end";
    return;
  }
  // 拼接起来
  HT_DISPATCH_KERNEL_CUDA_ONLY(DeviceType::CUDA, "AttnCommConcatRecvSlices",
                               hetu::impl::Concat, recv_data_slices,
                               const_cast<NDArray&>(recv_data), num_heads_dim, comm_stream);
  // HT_LOG_DEBUG << "[ParallelAttn]: ExecComm end";
}

void AttnCommRing::Run(bool is_bwd) {
  int64_t dst_ring_idx = (_ring_idx + 1) % _ring_size;
  int64_t src_ring_idx = (_ring_idx + _ring_size - 1) % _ring_size;
  auto& dst_tp_group = _tp_group_list[dst_ring_idx];
  auto& src_tp_group = _tp_group_list[src_ring_idx];
  auto& cur_tp_group = _tp_group_list[_ring_idx];
  auto dst_tp_degree = dst_tp_group.num_devices();
  auto src_tp_degree = src_tp_group.num_devices();
  auto cur_tp_degree = cur_tp_group.num_devices();
  int64_t cur_tp_inner_idx = cur_tp_group.get_index(_local_device);
  std::vector<Device> dst_devices, src_devices;
  int64_t dst_split_ratio = 1, src_split_ratio = 1;
  // 考虑local device作为发送端
  if (dst_tp_degree >= cur_tp_degree) {
    HT_ASSERT(dst_tp_degree % cur_tp_degree == 0)
      << "tp degree should be 2^n";
    // 向dst发送时需要split
    dst_split_ratio = dst_tp_degree / cur_tp_degree;
    for (size_t idx = cur_tp_inner_idx * dst_split_ratio; idx < (cur_tp_inner_idx + 1) * dst_split_ratio; idx++) {
      dst_devices.emplace_back(dst_tp_group.get(idx));
    }
  } else {
    HT_ASSERT(cur_tp_degree % dst_tp_degree == 0)
      << "tp degree should be 2^n";
    // 只用发给一个dst
    dst_devices.emplace_back(dst_tp_group.get(cur_tp_inner_idx / (cur_tp_degree / dst_tp_degree)));
  }
  // 考虑local device作为接收端
  if (src_tp_degree >= cur_tp_degree) {
    HT_ASSERT(src_tp_degree % cur_tp_degree == 0)
      << "tp degree should be 2^n";
    // 从src接收时需要merge
    src_split_ratio = src_tp_degree / cur_tp_degree;
    for (size_t idx = cur_tp_inner_idx * src_split_ratio; idx < (cur_tp_inner_idx + 1) * src_split_ratio; idx++) {
      src_devices.emplace_back(src_tp_group.get(idx));
    }
  } else {
    HT_ASSERT(cur_tp_degree % src_tp_degree == 0)
      << "tp degree should be 2^n";
    // 只用接收一个src
    src_devices.emplace_back(src_tp_group.get(cur_tp_inner_idx / (cur_tp_degree / src_tp_degree)));
  }
  // ring-attn核心部分
  // 通信与计算overlap
  // 首先要让通信和计算的stream同步一下
  // 因为之前prepare storage时在计算stream上用到了一些contiguous操作
  HT_ASSERT(_nccl_comm_group->stream().stream_index() != _stream_idx)
    << "comm stream and comp stream shouldn't be the same stream";
  auto comm_stream = _nccl_comm_group->stream();
  auto comp_stream = Stream(_local_device, _stream_idx);
  auto tmp_event = std::make_unique<hetu::impl::CUDAEvent>(_local_device);
  tmp_event->Record(comp_stream);
  tmp_event->Block(comm_stream);
  int64_t q_idx = _ring_idx; 
  for (size_t round = 0; round < _ring_size; round++) {
    int64_t cur_kv_idx = (_ring_idx + _ring_size - round) % _ring_size;
    int64_t next_kv_idx = (_ring_idx + _ring_size - 1 - round) % _ring_size;
    auto cur_kv_block = _kv_block_list.at(cur_kv_idx);
    auto next_kv_block = _kv_block_list.at(next_kv_idx);
    if (round == _ring_size - 1) {
      // 最后一次通信的next kv block需要特殊处理
      // 不能使用和第一次一样的kv block（即_kv_block_list中的第_ring_idx个）
      // 否则当cp无法整除storage数时
      // 可能出现storage上的冲突
      next_kv_block = _final_kv_block;
    }
    bool empty_round = (_attn_info_list.at(_ring_idx * _ring_size + cur_kv_idx)->get_mask() == AttnMask::EMPTY);
    HT_ASSERT(cur_kv_block->get_attn_storage() != next_kv_block->get_attn_storage())
      << "the adjacent kv block shouldn't share the same attn storage"
      << ", ensure _kv_storage_size >= 2";
    // forward
    // Question: 最后一次是否需要通信
    // 目前因为只有两个buffer
    // local的k和v被换走了因此最后一次还要换回来
    if (!is_bwd) {
      // HT_LOG_DEBUG << "[ParallelAttn]: run fwd round " << round << " begin";
      // 通信时的next_kv_block所在的storage必须已经完成了计算
      next_kv_block->wait_until_attn_done(comm_stream);
      {
        if (_need_profile) _comm_profile_start_event_list.at(round)->Record(comm_stream);
        ExecComm(cur_kv_block->get_4d_kv(), next_kv_block->get_4d_kv(), dst_devices, src_devices, comm_stream);
        if (_need_profile) _comm_profile_end_event_list.at(round)->Record(comm_stream);
      }
      next_kv_block->record_comm(comm_stream);
      // 计算时当前的cur_kv_block所在的storage必须已经完成了通信
      if (!empty_round) cur_kv_block->wait_until_comm_done(comp_stream);
      {
        if (_need_profile) _attn_profile_start_event_list.at(round)->Record(comp_stream);
        if (!empty_round) ExecFlashAttn(q_idx, cur_kv_idx, _local_q, cur_kv_block->get_4d_k(), cur_kv_block->get_4d_v(), _out, _softmax_lse, _rng_state_list.at(round));
        if (_need_profile) _attn_profile_end_event_list.at(round)->Record(comp_stream);
      }
      cur_kv_block->record_attn(comp_stream);
      {
        if (_need_profile) _corr_profile_start_event_list.at(round)->Record(comp_stream);
        // 如果当前block attn是空的
        // 那么不需要进行block的累积修正
        if (!empty_round) ExecCorr(_out, _softmax_lse, _acc_out, _acc_softmax_lse_transposed, round == 0 ? true : false);
        if (_need_profile) _corr_profile_end_event_list.at(round)->Record(comp_stream);
      }
      // HT_LOG_DEBUG << "[ParallelAttn]: run fwd round " << round << " end";
    }
    // backward
    // 由于使用了grad的捎带
    // 累积梯度时next_kv_block所在的storage必须也已经完成了通信
    // 因此实际上storage大于2之后没有任何效果
    // 因为第2个storage的通信一定会等第1个storage的attn的计算
    else {
      // HT_LOG_DEBUG << "[ParallelAttn]: run bwd round " << round << " begin";
      // 通信时的next_kv_block所在的storage必须已经完成了计算
      // 且当前的cur_kv_block必须已经捎带上了梯度
      // 注意如果只有2个storage时第一个同步要求实际上被包含在第二个同步要求里
      // 但为了与fwd部分的代码保持一致因此这里这么写
      next_kv_block->wait_until_attn_done(comm_stream);
      cur_kv_block->wait_until_grad_done(comm_stream);
      {
        if (_need_profile) _comm_profile_start_event_list.at(round)->Record(comm_stream);
        // 第一次无grad可以捎带
        if (round == 0) {
          ExecComm(cur_kv_block->get_4d_kv(), next_kv_block->get_4d_kv(), dst_devices, src_devices, comm_stream);
        } 
        // 最后一次只用通信grad
        else if (round == _ring_size - 1) {
          ExecComm(cur_kv_block->get_4d_acc_dkv(), next_kv_block->get_4d_acc_dkv(), dst_devices, src_devices, comm_stream);
        }
        // 剩下都需要进行kv通信以及grad的捎带
        else {
          ExecComm(cur_kv_block->get_3d_all(), next_kv_block->get_3d_all(), dst_devices, src_devices, comm_stream, true);
        }
        if (_need_profile) _comm_profile_end_event_list.at(round)->Record(comm_stream);
      }
      next_kv_block->record_comm(comm_stream);
      // workaround: 分配临时的dq、dk以及dv
      // 理论上按照现在mempool的实现
      // dq恰好可以直接复用
      // 但由于seq_len不定长因此dk和dv可能出现split和merge
      auto dq = NDArray();
      auto dk = NDArray();
      auto dv = NDArray();
      if (!empty_round) {
        dq = NDArray::empty_like(_local_q, _stream_idx);
        dk = NDArray::empty_like(cur_kv_block->get_4d_k(), _stream_idx);
        dv = NDArray::empty_like(cur_kv_block->get_4d_v(), _stream_idx);
      }
      // 计算时当前的cur_kv_block所在的storage必须已经完成了通信
      if (!empty_round) cur_kv_block->wait_until_comm_done(comp_stream);
      {
        if (_need_profile) _attn_profile_start_event_list.at(round)->Record(comp_stream);
        if (!empty_round) ExecFlashAttn(q_idx, cur_kv_idx, _local_q, cur_kv_block->get_4d_k(), cur_kv_block->get_4d_v(), _acc_out, _acc_softmax_lse, _rng_state_list.at(round), true, _grad_output, dq, dk, dv);
        if (_need_profile) _attn_profile_end_event_list.at(round)->Record(comp_stream);
      }
      cur_kv_block->record_attn(comp_stream);
      // 累积梯度时next_kv_block所在的storage必须也已经完成了通信
      if (!empty_round) next_kv_block->wait_until_comm_done(Stream(_local_device, _stream_idx));
      auto acc_dk = next_kv_block->get_4d_acc_dk();
      auto acc_dv = next_kv_block->get_4d_acc_dv();
      {
        if (_need_profile) _grad_profile_start_event_list.at(round)->Record(comp_stream);
        // 如果当前block attn是空的
        // 那么不需要进行grad的累积
        if (!empty_round) {
          NDArray::add(_acc_dq, dq, _stream_idx, _acc_dq);
          NDArray::add(acc_dk, dk, _stream_idx, acc_dk);
          NDArray::add(acc_dv, dv, _stream_idx, acc_dv);
        }
        if (_need_profile) _grad_profile_end_event_list.at(round)->Record(comp_stream);
      }
      next_kv_block->record_grad(comp_stream);
      // HT_LOG_DEBUG << "[ParallelAttn]: run bwd round " << round << " end";
    }
  }
  // bwd需要再通信一次grad
  // Question: Megatron-CP并没有这一步
  // 应该是Megatron的实现有问题
  if (is_bwd) {
    _final_next_kv_block->wait_until_attn_done(comm_stream);
    _final_kv_block->wait_until_grad_done(comm_stream);
    ExecComm(_final_kv_block->get_4d_acc_dkv(), _final_next_kv_block->get_4d_acc_dkv(), dst_devices, src_devices, comm_stream);
  }
}

void AttnCommRing::Profile(const Operator& op, size_t micro_batch_id, bool is_bwd) {
  if (!_need_profile) {
    return;
  }
  Stream(_local_device, _stream_idx).Sync();
  auto flag = dynamic_cast<ExecutableGraph&>(op->graph())._parallel_attn_flag;
  auto& log_file_path = dynamic_cast<ExecutableGraph&>(op->graph())._parallel_attn_log_file_path;
  std::string s = "";
  s += "Micro Batch Id: " + std::to_string(micro_batch_id) + ", " + (is_bwd ? "FWD, " : "BWD, ") + op->name() + "\n";
  s += "Comm Timeline: ";
  for (size_t i = 0; i < _ring_size; i++) {
    auto time_cost = _comm_profile_end_event_list.at(i)->TimeSince(*_comm_profile_start_event_list.at(i))* 1.0 / 1e6;
    s += std::to_string(time_cost) + "ms";
    if (i != _ring_size - 1) {
      auto interval = _comm_profile_start_event_list.at(i + 1)->TimeSince(*_comm_profile_end_event_list.at(i)) * 1.0 / 1e6;
      s += " -> " + std::to_string(interval) + "ms(interval)" + " -> ";
    }
  }
  s += "\nAttn Timeline: ";
  for (size_t i = 0; i < _ring_size; i++) {
    auto time_cost = _attn_profile_end_event_list.at(i)->TimeSince(*_attn_profile_start_event_list.at(i)) * 1.0 / 1e6;
    s += std::to_string(time_cost) + "ms";
    if (i != _ring_size - 1) {
      auto interval = _attn_profile_start_event_list.at(i + 1)->TimeSince(*_attn_profile_end_event_list.at(i)) * 1.0 / 1e6;
      s += " -> " + std::to_string(interval) + "ms(interval)" + " -> ";
    }
  }
  s += "\nAttn Start Time - Comm Start Time: ";
  for (size_t i = 0; i < _ring_size; i++) {
    auto time_cost = _attn_profile_start_event_list.at(i)->TimeSince(*_comm_profile_start_event_list.at(i)) * 1.0 / 1e6;
    s += std::to_string(time_cost) + "ms";
    if (i != _ring_size - 1) {
      s += ", ";
    }
  }
  if (!is_bwd) {
    s += "\nCorr Timeline: ";
    for (size_t i = 0; i < _ring_size; i++) {
      auto time_cost = _corr_profile_end_event_list.at(i)->TimeSince(*_corr_profile_start_event_list.at(i)) * 1.0 / 1e6;
      s += std::to_string(time_cost) + "ms";
      if (i != _ring_size - 1) {
        auto interval = _corr_profile_start_event_list.at(i + 1)->TimeSince(*_corr_profile_end_event_list.at(i)) * 1.0 / 1e6;
        s += " -> " + std::to_string(interval) + "ms(interval)" + " -> ";
      }
    }
  } else {
    s += "\nGrad Timeline: ";
    for (size_t i = 0; i < _ring_size; i++) {
      auto time_cost = _grad_profile_end_event_list.at(i)->TimeSince(*_grad_profile_start_event_list.at(i)) * 1.0 / 1e6;
      s += std::to_string(time_cost) + "ms";
      if (i != _ring_size - 1) {
        auto interval = _grad_profile_start_event_list.at(i + 1)->TimeSince(*_grad_profile_end_event_list.at(i)) * 1.0 / 1e6;
        s += " -> " + std::to_string(interval) + "ms(interval)" + " -> ";
      }
    }
  }
  if (log_file_path != "") {
    if (flag == 1) {
      ofstream_sync file(log_file_path, std::ios_base::app);
      if (file.is_open()) {
        file << s << std::endl;
      } else {
        HT_RUNTIME_ERROR << "Error opening the file";
      }
    } else {
      HT_RUNTIME_ERROR << "NotImplementedError";
    }
  } else {
    if (flag == 1) {
      HT_LOG_INFO << s;
    } else {
      HT_RUNTIME_ERROR << "NotImplementedError";
    }
  }
}

/****************************************************************
 ------------------------ Normal Op Impl ------------------------
*****************************************************************/

std::vector<NDArrayMeta> ParallelAttentionOpImpl::DoInferMeta(const TensorList& inputs,
                                                              const InstantiationContext& inst_ctx) const {
  std::vector<NDArrayMeta> out_metas = {};
  auto& input = inputs.at(0); // packed qkv
  NDArrayMeta base = input->meta();
  HT_ASSERT(input->shape().size() == 2)
    << "ParallelAttentionOp only support input shape [batch_size * seq_len, num_head * head_dim]"
    << ", but found " << input->shape();
  int64_t batch_size_mul_seq_len = input->shape(0);
  int64_t num_heads_mul_head_dim = input->shape(1);
  // workaround 
  // 这里不得不使用CUR_HETERO_ID（就算外部没有进行USE_HETERO_ID）
  // 在define graph中，该值一定是0
  // 在exec graph中，该值会在Instantiate中MakeOp前被合理地设置
  input->producer()->graph().USE_HETERO_ID = true;
  int64_t seq_len = get_local_seq_len(input, _multi_seq_lens_symbol);
  input->producer()->graph().USE_HETERO_ID = false;
  HT_ASSERT(batch_size_mul_seq_len % seq_len == 0)
    << "packed qkv dim 0 should be divided by seq len"
    << ", but found batch_size_mul_seq_len = " << batch_size_mul_seq_len << " and seq_len = " << seq_len;
  int64_t batch_size = batch_size_mul_seq_len / seq_len;
  HT_ASSERT(num_heads_mul_head_dim % _head_dim == 0)
    << "packed qkv dim 1 should be divided by head dim";
  int64_t total_num_heads = num_heads_mul_head_dim / _head_dim;
  HT_ASSERT(total_num_heads % (_group_query_ratio + 2) == 0)
    << "total_num_heads should be divided by (group_query_ratio + 2)";
  int64_t q_num_heads = total_num_heads / (_group_query_ratio + 2) * _group_query_ratio;
  int64_t kv_num_heads = total_num_heads / (_group_query_ratio + 2);
  HT_ASSERT(_head_dim <= 256 && _head_dim % 8 == 0)
    << "ParallelFlashAttention forward only supports head dimension at most 256 and must be divided by 8";
  // TODO: support padding
  out_metas.emplace_back(base.set_shape({batch_size_mul_seq_len, q_num_heads * _head_dim})); // output
  // out_metas.emplace_back(base.set_shape({batch_size, q_num_heads, seq_len}).set_dtype(kFloat)); // softmax_lse
  // out_metas.emplace_back(base.set_shape({2}).set_device(kCPU).set_dtype(kInt64)); // rng_state
  return out_metas;
}

void ParallelAttentionOpImpl::DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                                             const OpMeta& op_meta,
                                             const InstantiationContext& inst_ctx) const {
  const DistributedStates& ds_input = inputs.at(0)->get_distributed_states();
  HT_ASSERT(ds_input.is_valid()) 
    << "ParallelAttentionOpImpl: distributed states for input must be valid!";
  HT_ASSERT(ds_input.get_dim(-2) == 1)
    << "Input tensor shouldn't be partial!";
  outputs.at(0)->set_distributed_states(ds_input);    
}

void ParallelAttentionOpImpl::DoCompute(Operator& op, 
                                        const NDArrayList& inputs, NDArrayList& outputs,
                                        RuntimeContext& ctx) const {
  /*
  // deprecated version: output splitted q, k, v as well
  HT_ASSERT(op->input(0)->num_consumers() == 1)
    << "qkv should only consumed by the ParallelAttentionOp"
    << ", so that it will be released but the splitted output (q, k, v) remains";
  */
  // 拿到qkv先做reshape和split
  const auto& qkv = inputs.at(0);
  auto& output = outputs.at(0);
  // auto& softmax_lse = outputs.at(1);
  // auto& rng_state = outputs.at(2);
  HTShape input_shape = qkv->shape();
  HT_ASSERT(input_shape.size() == 2)
    << "ParallelAttentionOp only support input shape [batch_size * seq_len, num_heads * head_dim * 3]";
  int64_t batch_size_mul_seq_len = input_shape.at(0);
  int64_t num_heads_mul_head_dim = input_shape.at(1);
  int64_t seq_len = get_local_seq_len(op->input(0), _multi_seq_lens_symbol);
  HT_ASSERT(batch_size_mul_seq_len % seq_len == 0)
    << "packed qkv dim 0 should be divided by seq len"
    << ", but found batch_size_mul_seq_len = " << batch_size_mul_seq_len << " and seq_len = " << seq_len;
  int64_t batch_size = batch_size_mul_seq_len / seq_len;
    HT_ASSERT(num_heads_mul_head_dim % _head_dim == 0)
    << "packed qkv dim 1 should be divided by head dim";
  int64_t total_num_heads = num_heads_mul_head_dim / _head_dim;
  HT_ASSERT(total_num_heads % (_group_query_ratio + 2) == 0)
    << "total_num_heads should be divided by (group_query_ratio + 2)";
  int64_t q_num_heads = total_num_heads / (_group_query_ratio + 2) * _group_query_ratio;
  int64_t kv_num_heads = total_num_heads / (_group_query_ratio + 2);
  HT_ASSERT(_head_dim <= 256 && _head_dim % 8 == 0)
    << "ParallelFlashAttention forward only supports head dimension at most 256 and must be divided by 8";
  auto stream_idx = op->instantiation_ctx().stream_index;
  auto reshaped_qkv = NDArray::view(qkv, {batch_size, seq_len, total_num_heads, _head_dim});
  auto reshaped_output = NDArray::view(output, {batch_size, seq_len, q_num_heads, _head_dim});
  // self-attn
  HTShape q_shape = {batch_size, seq_len, q_num_heads, _head_dim};
  HTShape k_shape = {batch_size, seq_len, kv_num_heads, _head_dim};
  HTShape v_shape = {batch_size, seq_len, kv_num_heads, _head_dim};
  HTShape q_begin_pos = {0, 0, 0, 0};
  HTShape k_begin_pos = {0, 0, q_num_heads, 0};
  HTShape v_begin_pos = {0, 0, q_num_heads + kv_num_heads, 0};
  auto q = NDArray::contiguous(NDArray::slice(reshaped_qkv, q_begin_pos, q_shape, stream_idx));
  auto k = NDArray::contiguous(NDArray::slice(reshaped_qkv, k_begin_pos, k_shape, stream_idx));
  auto v = NDArray::contiguous(NDArray::slice(reshaped_qkv, v_begin_pos, v_shape, stream_idx));
  // HT_LOG_DEBUG << "[ParallelAttn]: q shape is " << q->shape() << " and k (same as v) shape is " << k->shape();
  double softmax_scale_ = softmax_scale() >= 0 ? softmax_scale() : std::pow(_head_dim, -0.5);
  int64_t ring_idx;
  DeviceGroupList tp_group_list;
  std::vector<int64_t> seq_len_list;
  std::tie(ring_idx, tp_group_list, seq_len_list) = get_local_ring(op->input(0), _multi_seq_lens_symbol, _multi_cp_group_symbol);
  // HT_LOG_DEBUG << "[ParallelAttn]: the tp group list is " << tp_group_list << " and seq len list is " << seq_len_list;
  // 开cp
  if (tp_group_list.size() >= 2) {
    HT_ASSERT(!_packing)
      << "currently not support Ring-Attn w/ Packing";
    auto used_ranks = dynamic_cast<ExecutableGraph&>(op->graph()).GetUsedRanks();
    auto local_device = hetu::impl::comm::GetLocalDevice();
    auto& nccl_comm_group = hetu::impl::comm::NCCLCommunicationGroup::GetOrCreate(used_ranks, local_device);
    // HT_LOG_DEBUG << "[ParallelAttn]: construct fwd comm ring for " << local_device;
    auto attn_comm_ring = AttnCommRing(op,
                                       nccl_comm_group, 
                                       stream_idx,
                                       ring_idx,
                                       tp_group_list, 
                                       seq_len_list, 
                                       batch_size, 
                                       q_num_heads,
                                       kv_num_heads, 
                                       _head_dim,
                                       softmax_scale_,
                                       p_dropout());
    attn_comm_ring.PrepareStorageFwd(q, k, v, reshaped_output);
    attn_comm_ring.GenerateAttnInfo();
    attn_comm_ring.Run();
    attn_comm_ring.SaveCtx(attn_ctx());
    attn_comm_ring.Profile(op, _attn_ctx_num);
  }
  // 不开cp
  else {
    // no ring-attn
    // HT_LOG_DEBUG << "[ParallelAttn]: no fwd comm ring needed for " << local_device;
    attn_ctx()->rng_state_list = {NDArray::empty(HTShape{2},
                                                 Device(kCPU),
                                                 kInt64,
                                                 stream_idx)};
    NDArray empty_ndarray = NDArray();
    // HT_LOG_DEBUG << "[ParallelAttn]: fwd (no cp), acc_softmax_lse shape is " << attn_ctx()->acc_softmax_lse->shape() << ", acc_out shape is " << attn_ctx()->acc_out->shape();
    if (!_packing) {
      attn_ctx()->q = q;
      attn_ctx()->k = k;
      attn_ctx()->v = v;
      // *注意softmax_lse是fp32的
      attn_ctx()->acc_softmax_lse = NDArray::empty(HTShape{batch_size, q_num_heads, seq_len_list[ring_idx]},
                                                   q->device(),
                                                   kFloat,
                                                   stream_idx);
      attn_ctx()->acc_out = reshaped_output;
      HT_DISPATCH_KERNEL_CUDA_ONLY(op->instantiation_ctx().placement.type(), type(), hetu::impl::FlashAttn,
                                   q, k, v, attn_ctx()->acc_out, q,
                                   k, v, attn_ctx()->acc_out, attn_ctx()->acc_softmax_lse,
                                   empty_ndarray, attn_ctx()->rng_state_list.at(0), p_dropout(), softmax_scale_,
                                   true, return_softmax(), op->instantiation_ctx().stream());
    } else {
      HT_ASSERT(inputs.size() == 3)
        << "packing should have 3 inputs: qkv, cu_seqlens_q and cu_seqlens_k";
      auto cu_seqlens_q = inputs.at(1);
      auto cu_seqlens_k = inputs.at(2);
      int64_t packing_num = cu_seqlens_q->numel() - 1;
      HT_ASSERT(packing_num == cu_seqlens_k->numel() - 1 && packing_num >= 1)
        << "packing num (>= 1) mismatches";
      attn_ctx()->q = NDArray::view(q, {batch_size_mul_seq_len, q_num_heads, _head_dim});
      attn_ctx()->k = NDArray::view(k, {batch_size_mul_seq_len, kv_num_heads, _head_dim});
      attn_ctx()->v = NDArray::view(v, {batch_size_mul_seq_len, kv_num_heads, _head_dim});
      // *注意softmax_lse是fp32的
      attn_ctx()->acc_softmax_lse = NDArray::empty(HTShape{packing_num, q_num_heads, max_seqlen_q()},
                                                   q->device(),
                                                   kFloat,
                                                   stream_idx);
      attn_ctx()->acc_out = NDArray::view(reshaped_output, {batch_size_mul_seq_len, q_num_heads, _head_dim});
      HT_DISPATCH_KERNEL_CUDA_ONLY(op->instantiation_ctx().placement.type(), type(), hetu::impl::FlashAttnVarlen, 
                                   attn_ctx()->q, attn_ctx()->k, attn_ctx()->v, cu_seqlens_q, cu_seqlens_k, attn_ctx()->acc_out, attn_ctx()->q,
                                   attn_ctx()->k, attn_ctx()->v, attn_ctx()->acc_out, attn_ctx()->acc_softmax_lse,
                                   empty_ndarray, attn_ctx()->rng_state_list.at(0), 
                                   max_seqlen_q(), max_seqlen_k(), 
                                   p_dropout(), softmax_scale_, false,
                                   true, return_softmax(), op->instantiation_ctx().stream());
      // HT_LOG_INFO << "Varlen attn cu_seqlens_q is " << cu_seqlens_q;
    }
  }
}

TensorList ParallelAttentionOpImpl::DoGradient(Operator& op, const TensorList& grad_outputs) const {
  Tensor cu_seqlens_q = Tensor(), cu_seqlens_k = Tensor();
  if (_packing) {
    HT_ASSERT(op->num_inputs() == 3)
      << "packing should have 3 inputs: qkv, cu_seqlens_q and cu_seqlens_k";
    cu_seqlens_q = op->input(1);
    cu_seqlens_k = op->input(2);
  }
  if (op->requires_grad(0)) {
    auto grads = MakeParallelAttentionGradientOp(_attn_ctx_list, grad_outputs.at(0),  
                                                 _head_dim, _group_query_ratio, 
                                                 _multi_seq_lens_symbol, _multi_cp_group_symbol, 
                                                 _packing, cu_seqlens_q, cu_seqlens_k, _max_seqlen_q, _max_seqlen_k,
                                                 p_dropout(), softmax_scale(), is_causal(), op->grad_op_meta().set_name(op->grad_name()));
    HT_ASSERT(grads.size() == 1)
      << "ParallelAttentionGradientOp should only have one output, that is the grad for the fused qkv";
    if (_packing) {
      grads.emplace_back(Tensor());
      grads.emplace_back(Tensor());
    }
    return grads;
  } else {
    if (_packing) {
      return {Tensor(), Tensor(), Tensor()};
    }
    return {Tensor()};
  }
}

HTShapeList ParallelAttentionOpImpl::DoInferShape(Operator& op, 
                                                  const HTShapeList& input_shapes, 
                                                  RuntimeContext& ctx) const {
  HTShapeList out_shapes;
  HT_ASSERT(input_shapes.at(0).size() == 2)
    << "ParallelAttentionOp only support input shape [batch_size * seq_len, q_num_heads * head_dim]";
  int64_t batch_size_mul_seq_len = input_shapes.at(0).at(0);
  int64_t num_heads_mul_head_dim = input_shapes.at(0).at(1);
  int64_t seq_len = get_local_seq_len(op->input(0), _multi_seq_lens_symbol);
  HT_ASSERT(batch_size_mul_seq_len % seq_len == 0)
    << "packed qkv dim 0 should be divided by seq len"
    << ", but found batch_size_mul_seq_len = " << batch_size_mul_seq_len << " and seq_len = " << seq_len;
  int64_t batch_size = batch_size_mul_seq_len / seq_len;
  HT_ASSERT(num_heads_mul_head_dim % _head_dim == 0)
    << "packed qkv dim 1 should be divided by head dim";
  int64_t total_num_heads = num_heads_mul_head_dim / _head_dim;
  HT_ASSERT(total_num_heads % (_group_query_ratio + 2) == 0)
    << "total_num_heads should be divided by (group_query_ratio + 2)";
  int64_t q_num_heads = total_num_heads / (_group_query_ratio + 2) * _group_query_ratio;
  int64_t kv_num_heads = total_num_heads / (_group_query_ratio + 2);
  HT_ASSERT(_head_dim <= 256 && _head_dim % 8 == 0)
    << "ParallelFlashAttention forward only supports head dimension at most 256 and must be divided by 8";
  out_shapes.emplace_back(HTShape{batch_size_mul_seq_len, q_num_heads * _head_dim}); // output
  // out_shapes.emplace_back(HTShape{batch_size, q_num_heads, seq_len}); // softmax_lse
  // out_shapes.emplace_back(HTShape{2}); // rng_state
  return out_shapes;
}

std::vector<NDArrayMeta> ParallelAttentionGradientOpImpl::DoInferMeta(const TensorList& inputs,
                                                                      const InstantiationContext& inst_ctx) const {
  HT_ASSERT(inputs.at(0)->shape().size() == 2)
    << "ParallelAttentionGradientOp input shape should be [batch_size * seq_len, q_num_heads * head_dim]";
  NDArrayMeta output_meta = inputs.at(0)->meta();
  // [batch_size * seq_len, num_heads * head_dim]
  output_meta.set_shape(HTShape{inputs.at(0)->shape(0), inputs.at(0)->shape(1) + inputs.at(0)->shape(1) / _group_query_ratio * 2});
  return {output_meta};
}

void ParallelAttentionGradientOpImpl::DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                                                     const OpMeta& op_meta,
                                                     const InstantiationContext& inst_ctx) const {
  const DistributedStates& ds_input = inputs.at(0)->get_distributed_states();
  HT_ASSERT(ds_input.is_valid()) 
    << "ParallelAttentionGradientOpImpl: distributed states for input must be valid!";
  HT_ASSERT(ds_input.get_dim(-2) == 1)
    << "Input tensor shouldn't be partial!";
  outputs.at(0)->set_distributed_states(ds_input);    
}

void ParallelAttentionGradientOpImpl::DoCompute(Operator& op, const NDArrayList& inputs,
                                                NDArrayList& outputs, RuntimeContext& ctx) const {
  const auto& grad_output = inputs.at(0);
  // 改为从context中取
  // 主要原因是rng_state不定长
  /*
  const auto& qkv = inputs.at(1);
  const auto& out = inputs.at(2);
  const auto& softmax_lse = inputs.at(3);
  const auto& rng_state = inputs.at(4);
  */
  auto& grad_input = outputs.at(0);
  HTShape grad_output_shape = grad_output->shape();
  HT_ASSERT(grad_output_shape.size() == 2)
    << "ParallelAttentionGradientOp only support input shape [batch_size * seq_len, q_num_heads * head_dim]";
  int64_t batch_size_mul_seq_len = grad_output_shape.at(0);
  int64_t q_num_heads_mul_head_dim = grad_output_shape.at(1);
  int64_t seq_len = get_local_seq_len(op->input(0), _multi_seq_lens_symbol);
  HT_ASSERT(batch_size_mul_seq_len % seq_len == 0)
    << "grad output dim 0 should be divided by seq len"
    << ", but found batch_size_mul_seq_len = " << batch_size_mul_seq_len << " and seq_len = " << seq_len;
  int64_t batch_size = batch_size_mul_seq_len / seq_len;
  HT_ASSERT(q_num_heads_mul_head_dim % _head_dim == 0)
    << "grad output dim 1 should be divided by head dim";
  int64_t q_num_heads = q_num_heads_mul_head_dim / _head_dim;
  HT_ASSERT(q_num_heads % _group_query_ratio == 0)
    << "q_num_heads should be divided by group_query_ratio";
  int64_t kv_num_heads = q_num_heads / _group_query_ratio;
  int64_t total_num_heads = q_num_heads + 2 * kv_num_heads;
  HT_ASSERT(_head_dim <= 256 && _head_dim % 8 == 0)
    << "ParallelFlashAttention backward only supports head dimension at most 256 and must be divided by 8";
  auto stream_idx = op->instantiation_ctx().stream_index;
  auto reshaped_grad_output = NDArray::view(grad_output, {batch_size, seq_len, q_num_heads, _head_dim});
  auto reshaped_grad_input = NDArray::view(grad_input, {batch_size, seq_len, total_num_heads, _head_dim});
  HTShape dq_shape = {batch_size, seq_len, q_num_heads, _head_dim};
  HTShape dk_shape = {batch_size, seq_len, kv_num_heads, _head_dim};
  HTShape dv_shape = {batch_size, seq_len, kv_num_heads, _head_dim};
  HTShape dq_begin_pos = {0, 0, 0, 0};
  HTShape dk_begin_pos = {0, 0, q_num_heads, 0};
  HTShape dv_begin_pos = {0, 0, q_num_heads + kv_num_heads, 0};
  auto dq = NDArray::slice(reshaped_grad_input, dq_begin_pos, dq_shape, stream_idx);
  auto dk = NDArray::slice(reshaped_grad_input, dk_begin_pos, dk_shape, stream_idx);
  auto dv = NDArray::slice(reshaped_grad_input, dv_begin_pos, dv_shape, stream_idx);
  double softmax_scale_ = softmax_scale() >= 0 ? softmax_scale() : std::pow(_head_dim, -0.5);
  int64_t ring_idx;
  DeviceGroupList tp_group_list;
  std::vector<int64_t> seq_len_list;
  std::tie(ring_idx, tp_group_list, seq_len_list) = get_local_ring(op->input(0), _multi_seq_lens_symbol, _multi_cp_group_symbol);
  // 开cp
  if (tp_group_list.size() >= 2) {
    HT_ASSERT(!_packing)
      << "currently not support Ring-Attn w/ Packing";
    auto used_ranks = dynamic_cast<ExecutableGraph&>(op->graph()).GetUsedRanks();
    auto& nccl_comm_group = hetu::impl::comm::NCCLCommunicationGroup::GetOrCreate(used_ranks, hetu::impl::comm::GetLocalDevice());
    auto attn_comm_ring = AttnCommRing(op,
                                       nccl_comm_group, 
                                       stream_idx,
                                       ring_idx,
                                       tp_group_list, 
                                       seq_len_list, 
                                       batch_size, 
                                       q_num_heads,
                                       kv_num_heads, 
                                       _head_dim,
                                       softmax_scale_,
                                       p_dropout());
    attn_comm_ring.PrepareStorageBwd(attn_ctx(), reshaped_grad_output, dq);
    attn_comm_ring.GenerateAttnInfo();
    attn_comm_ring.Run(true);
    attn_comm_ring.SaveGradient(dq, dk, dv);
    attn_comm_ring.Profile(op, _attn_ctx_num, true);
  }
  // 不开cp
  else {
    // no ring-attn
    // HT_LOG_DEBUG << "[ParallelAttn]: bwd (no cp), dq shape is " << dq->shape();
    HT_ASSERT(attn_ctx()->rng_state_list.size() == 1)
      << "there should only be one single rng_state when cp is off"
      << ", but for attn ctx of mirco batch " << _attn_ctx_num << ", the rng_state num is " << attn_ctx()->rng_state_list.size();
    if (!_packing) {
      HT_DISPATCH_KERNEL_CUDA_ONLY(op->instantiation_ctx().placement.type(), type(),
                                   hetu::impl::FlashAttnGradient, reshaped_grad_output,
                                   attn_ctx()->q, attn_ctx()->k, attn_ctx()->v, attn_ctx()->acc_out,
                                   attn_ctx()->acc_softmax_lse, attn_ctx()->rng_state_list.at(0), 
                                   dq, dk, dv, p_dropout(), softmax_scale_,
                                   true, op->instantiation_ctx().stream());
      // flash-attn already supports uncontiguous outputs
      /*
      // concat dq, dk, dv to reshaped_grad_input
      HT_DISPATCH_KERNEL_CUDA_ONLY(op->instantiation_ctx().placement.type(), type(),
                                  hetu::impl::Concatenate, dq,
                                  reshaped_grad_input, 2, 0, op->instantiation_ctx().stream());
      HT_DISPATCH_KERNEL_CUDA_ONLY(op->instantiation_ctx().placement.type(), type(),
                                  hetu::impl::Concatenate, dk,
                                  reshaped_grad_input, 2, q_num_heads, op->instantiation_ctx().stream());
      HT_DISPATCH_KERNEL_CUDA_ONLY(op->instantiation_ctx().placement.type(), type(),
                                  hetu::impl::Concatenate, dv,
                                  reshaped_grad_input, 2, q_num_heads + kv_num_heads, op->instantiation_ctx().stream());
      */
    } else {
      HT_ASSERT(inputs.size() == 3)
        << "packing should have 3 inputs: grad_output, cu_seqlens_q and cu_seqlens_k";
      auto cu_seqlens_q = inputs.at(1);
      auto cu_seqlens_k = inputs.at(2);
      auto dq_new = NDArray::view(NDArray::contiguous(dq), {batch_size_mul_seq_len, q_num_heads, _head_dim});
      auto dk_new = NDArray::view(NDArray::contiguous(dk), {batch_size_mul_seq_len, kv_num_heads, _head_dim});
      auto dv_new = NDArray::view(NDArray::contiguous(dv), {batch_size_mul_seq_len, kv_num_heads, _head_dim});
      reshaped_grad_output = NDArray::view(reshaped_grad_output, {batch_size_mul_seq_len, q_num_heads, _head_dim});
      HT_DISPATCH_KERNEL_CUDA_ONLY(op->instantiation_ctx().placement.type(), type(),
                                   hetu::impl::FlashAttnVarlenGradient, reshaped_grad_output,
                                   attn_ctx()->q, attn_ctx()->k, attn_ctx()->v, cu_seqlens_q, cu_seqlens_k, 
                                   attn_ctx()->acc_out, attn_ctx()->acc_softmax_lse, attn_ctx()->rng_state_list.at(0), 
                                   dq_new, dk_new, dv_new, max_seqlen_q(), max_seqlen_k(), p_dropout(), softmax_scale_, false,
                                   true, op->instantiation_ctx().stream());
      dq_new = NDArray::view(dq_new, dq_shape);
      dk_new = NDArray::view(dk_new, dk_shape);
      dv_new = NDArray::view(dv_new, dv_shape);
      NDArray::copy(dq_new, op->instantiation_ctx().stream().stream_index(), dq);
      NDArray::copy(dk_new, op->instantiation_ctx().stream().stream_index(), dk);
      NDArray::copy(dv_new, op->instantiation_ctx().stream().stream_index(), dv);
    }

  }
  // 清空该micro batch的ctx
  attn_ctx()->release();
}

HTShapeList ParallelAttentionGradientOpImpl::DoInferShape(Operator& op, 
                                                          const HTShapeList& input_shapes, 
                                                          RuntimeContext& ctx) const {
  HT_ASSERT(input_shapes.at(0).size() == 2)
    << "ParallelAttentionGradientOp input shape should be [batch_size * seq_len, q_num_heads * head_dim]";
  // [batch_size * seq_len, num_heads * head_dim]
  return {HTShape{input_shapes.at(0).at(0), input_shapes.at(0).at(1) + input_shapes.at(0).at(1) / _group_query_ratio * 2}};
}

TensorList MakeParallelAttentionOp(Tensor qkv, int64_t head_dim, int64_t group_query_ratio,
                                   SyShapeList multi_seq_lens_symbol, SyShapeList multi_cp_group_symbol, 
                                   bool packing, Tensor cu_seqlens_q, Tensor cu_seqlens_k, IntSymbol max_seqlen_q, IntSymbol max_seqlen_k,
                                   double p_dropout, double softmax_scale, 
                                   bool is_causal, bool return_softmax, OpMeta op_meta) {
  HT_ASSERT(is_causal)
    << "Currently only support causal attn";
  TensorList inputs = {std::move(qkv)};
  if (packing) {
    inputs.emplace_back(std::move(cu_seqlens_q));
    inputs.emplace_back(std::move(cu_seqlens_k));
  }
  return Graph::MakeOp(
    std::make_shared<ParallelAttentionOpImpl>(head_dim, group_query_ratio, std::move(multi_seq_lens_symbol), std::move(multi_cp_group_symbol), 
      packing, max_seqlen_q, max_seqlen_k,
      p_dropout, softmax_scale, is_causal, return_softmax),
    std::move(inputs),
    std::move(op_meta))->outputs();
}

TensorList MakeParallelAttentionGradientOp(const std::vector<std::shared_ptr<AttnCtx>>& attn_ctx_list, 
                                           Tensor grad_out, int64_t head_dim, int64_t group_query_ratio,
                                           SyShapeList multi_seq_lens_symbol, SyShapeList multi_cp_group_symbol,
                                           bool packing, Tensor cu_seqlens_q, Tensor cu_seqlens_k, IntSymbol max_seqlen_q, IntSymbol max_seqlen_k,
                                           double p_dropout, double softmax_scale,
                                           bool is_causal, OpMeta op_meta) {
  HT_ASSERT(is_causal)
    << "Currently only support causal attn";
  TensorList inputs = {std::move(grad_out)};
  if (packing) {
    inputs.emplace_back(std::move(cu_seqlens_q));
    inputs.emplace_back(std::move(cu_seqlens_k));
  }
  return Graph::MakeOp(
    std::make_shared<ParallelAttentionGradientOpImpl>(attn_ctx_list, head_dim, group_query_ratio, std::move(multi_seq_lens_symbol), std::move(multi_cp_group_symbol),
      packing, max_seqlen_q, max_seqlen_k,
      p_dropout, softmax_scale, is_causal),
    std::move(inputs),
    std::move(op_meta))->outputs();
}

} // namespace graph
} // namespace hetu
