#pragma once

#include "hetu/impl/communication/nccl_comm_group.h"
#include "hetu/impl/stream/CUDAStream.h"
#include "hetu/core/symbol.h"
#include "hetu/core/device.h"
#include "hetu/core/ndarray.h"
#include "hetu/graph/operator.h"
#include "hetu/graph/distributed_states.h"
#include "hetu/graph/utils/tensor_utils.h"

namespace hetu {
namespace graph {

/****************************************************************
 ---------------------- Parallel Attn Impl ----------------------
*****************************************************************/

std::ostream& operator<<(std::ostream& os, const std::vector<std::pair<int32_t, int32_t>>& vec);

enum class AttnSplitPattern {
  NORMAL = 0,
  STRIPE,
  SYM,
};

enum class AttnMask {
  CAUSAL = 0,
  ROW,
  COL,
  EMPTY,
  FULL,
};

class AttnInfo {
 public:
  AttnInfo(const Device& local_device, AttnMask attn_mask, int32_t valid_len)
  : _local_device(local_device), _attn_mask_list({attn_mask}), _valid_len_list({valid_len}) {
  }

  AttnInfo(const Device& local_device, const std::vector<AttnMask>& attn_mask_list, const std::vector<int32_t>& valid_len_list, 
           const std::vector<int32_t>& cu_seqlens_q_vec, const std::vector<int32_t>& cu_seqlens_k_vec)
  : _local_device(local_device), _attn_mask_list(attn_mask_list), _valid_len_list(valid_len_list),
    _cu_seqlens_q_vec(cu_seqlens_q_vec), _cu_seqlens_k_vec(cu_seqlens_k_vec) {
    HT_ASSERT(_cu_seqlens_q_vec.size() == _cu_seqlens_k_vec.size()
              && _cu_seqlens_q_vec.size() == _valid_len_list.size() + 1)
      << "sizes mismatch";
    /*
    if (valid_cu_seqlens_q == nullptr || valid_cu_seqlens_k == nullptr) {
      // TODO: support build valid_cu_seqlens from attn_mask_list + valid_len_list + cu_seqlens_vec,
      HT_NOT_IMPLEMENTED << "valid_cu_seqlens can be simply obtained by /2 from cu_seqlens"
        << ", no need to support more complicated settings for now";
    }
    */
    generate_valid_cu_seqlens_and_indices();
  }

  bool is_causal(size_t i = 0) const {
    return _attn_mask_list.at(i) == AttnMask::CAUSAL;
  }

  const AttnMask get_mask(size_t i = 0) const {
    return _attn_mask_list.at(i);
  }

  int32_t get_valid_len(size_t i = 0) const {
    return _valid_len_list.at(i);
  }

  const std::vector<int32_t>& get_cu_seqlens_q_vec() const {
    return _cu_seqlens_q_vec;
  }

  const std::vector<int32_t>& get_cu_seqlens_k_vec() const {
    return _cu_seqlens_k_vec;
  }

  const NDArray& get_valid_cu_seqlens_q() const {
    return _valid_cu_seqlens_q;
  }

  const NDArray& get_valid_cu_seqlens_k() const {
    return _valid_cu_seqlens_k;
  }

  const std::vector<std::pair<int32_t, int32_t>>& get_valid_indices_q() const {
    return _valid_indices_q;
  }

  const std::vector<std::pair<int32_t, int32_t>>& get_valid_indices_k() const {
    return _valid_indices_k;
  }

 protected:

  // valid_cu_seqlens is on GPU, valid_indices is on CPU
  // valid_cu_seqlens用于flash attn中的运算
  // valid_indices用于flash attn运算前对qkv去slice
  void generate_valid_cu_seqlens_and_indices();

  Device _local_device;
  std::vector<AttnMask> _attn_mask_list;
  std::vector<int32_t> _valid_len_list;
  std::vector<int32_t> _cu_seqlens_q_vec, _cu_seqlens_k_vec;
  NDArray _valid_cu_seqlens_q, _valid_cu_seqlens_k;
  std::vector<std::pair<int32_t, int32_t>> _valid_indices_q, _valid_indices_k;
};

// NDarrayStorage + comm & comp event
class AttnStorage {
 public: 
  AttnStorage(const std::shared_ptr<NDArrayStorage>& storage)
  : _storage(storage) {
    _comm_event = std::make_shared<hetu::impl::CUDAEvent>(storage->device());
    _attn_event = std::make_shared<hetu::impl::CUDAEvent>(storage->device());
    _grad_event = std::make_shared<hetu::impl::CUDAEvent>(storage->device());
  }

  const std::shared_ptr<NDArrayStorage>& storage() const {
    return _storage;
  }

  void record_comm(const Stream& stream) {
    _comm_event->Record(stream);
    _is_comm = true;
  }

  void record_attn(const Stream& stream) {
    _attn_event->Record(stream);
    _is_attn = true;
  }

  void record_grad(const Stream& stream) {
    _grad_event->Record(stream);
    _is_grad = true;
  }

  void wait_until_comm_done(const Stream& stream) {
    if (!_is_comm) {
      return;
    }
    _comm_event->Block(stream);
  }

  void wait_until_attn_done(const Stream& stream) {
    if (!_is_attn) {
      return;
    }
    _attn_event->Block(stream);
  }

  void wait_until_grad_done(const Stream& stream) {
    if (!_is_grad) {
      return;
    }
    _grad_event->Block(stream);
  }
  
 protected:
  std::shared_ptr<NDArrayStorage> _storage;
  bool _is_comm{false};
  std::shared_ptr<hetu::impl::CUDAEvent> _comm_event;
  bool _is_attn{false};
  std::shared_ptr<hetu::impl::CUDAEvent> _attn_event;
  bool _is_grad{false};
  std::shared_ptr<hetu::impl::CUDAEvent> _grad_event;
};

class AttnBlock {
 public:
  AttnBlock(const Device& local_device, const HTShape& kv_shape, DataType dtype, 
            const std::string name = {}, bool piggyback_grad = false, const HTShape& dkv_shape = {})
  : _local_device(local_device), _kv_shape(kv_shape), _dtype(dtype), 
    _name(name), _piggyback_grad(piggyback_grad), _dkv_shape(dkv_shape) {
    HT_ASSERT(_kv_shape.size() == 4 && _kv_shape.at(0) % 2 == 0)
      << "AttnBlock kv_shape should be [2 * batch_size, kv_seq_len, num_heads, head_dim]"
      << ", which consists of k and v";
    if (_piggyback_grad) {
      HT_ASSERT(_dkv_shape.size() == 4 && _dkv_shape[0] == _kv_shape[0] && _dkv_shape[2] == _kv_shape[2] && _dkv_shape[3] == _kv_shape[3])
        << "AttnBlock dkv_shape should be [2 * batch_size, dkv_seq_len, num_heads, head_dim]"
        << ", which consists of dk and dv";
    }
  }

  void record_comm(const Stream& stream) {
    HT_ASSERT(_has_storage)
      << "AttnBlock " << _name << " should have a binded storage";
    _attn_storage->record_comm(stream);
  }

  void record_attn(const Stream& stream) {
    HT_ASSERT(_has_storage)
      << "AttnBlock " << _name << " should have a binded storage";
    _attn_storage->record_attn(stream);
  }

  void record_grad(const Stream& stream) {
    HT_ASSERT(_has_storage)
      << "AttnBlock " << _name << " should have a binded storage";
    _attn_storage->record_grad(stream);
  }

  void wait_until_comm_done(const Stream& stream) {
    HT_ASSERT(_has_storage)
      << "AttnBlock " << _name << " should have a binded storage";
    _attn_storage->wait_until_comm_done(stream);
  }

  void wait_until_attn_done(const Stream& stream) {
    HT_ASSERT(_has_storage)
      << "AttnBlock " << _name << " should have a binded storage";
    _attn_storage->wait_until_attn_done(stream);
  }

  void wait_until_grad_done(const Stream& stream) {
    HT_ASSERT(_has_storage)
      << "AttnBlock " << _name << " should have a binded storage";
    _attn_storage->wait_until_grad_done(stream);
  }

  void bind_attn_storage(const std::shared_ptr<AttnStorage>& attn_storage) {
    HT_ASSERT(!_has_storage)
      << "AttnBlock " << _name << " already had a storage";
    int64_t kv_numel = 1;
    for (const auto x : _kv_shape) {
      kv_numel *= x;
    }
    int64_t dkv_numel = _piggyback_grad ? 1 : 0;
    for (const auto x : _dkv_shape) {
      dkv_numel *= x;
    }
    auto needed_byte = (kv_numel + dkv_numel) * DataType2Size(_dtype);
    HT_ASSERT(needed_byte <= attn_storage->storage()->size())
      << "storage must be larger than the wanted attn block";
    _has_storage = true;
    _attn_storage = attn_storage;
  }

  std::shared_ptr<AttnStorage> get_attn_storage() {
    return _attn_storage;
  }

  // 只会在comm时使用
  // [2 * batch_size * (kv_seq_len + dkv_seq_len or 0), num_heads, head_dim]
  NDArray get_3d_all() {
    HT_ASSERT(_has_storage == true)
      << "please ensure you've binded " << _name << " to a storage in advance";
    int64_t kv_numel = 1;
    for (const auto x : _kv_shape) {
      kv_numel *= x;
    }
    int64_t dkv_numel = _piggyback_grad ? 1 : 0;
    for (const auto x : _dkv_shape) {
      dkv_numel *= x;
    }
    int64_t all_numel = kv_numel + dkv_numel;
    int64_t num_heads = _kv_shape.at(2);
    int64_t head_dim = _kv_shape.at(3);
    auto meta = NDArrayMeta().set_dtype(_dtype)
                             .set_device(_local_device)
                             .set_shape(HTShape{all_numel / num_heads / head_dim, num_heads, head_dim});
    return NDArray(meta, _attn_storage->storage());
  }

  // [2 * batch_size, kv_seq_len, num_heads, head_dim]
  NDArray get_4d_kv() {
    HT_ASSERT(_has_storage == true)
      << "please ensure you've binded " << _name << " to a storage in advance";
    auto meta = NDArrayMeta().set_dtype(_dtype)
                             .set_device(_local_device)
                             .set_shape(_kv_shape);
    return NDArray(meta, _attn_storage->storage());
  }

  // [2 * batch_size, dkv_seq_len, num_heads, head_dim]
  NDArray get_4d_acc_dkv() {
    HT_ASSERT(_piggyback_grad == true && _has_storage == true)
      << "please ensure you've binded " << _name << " to a storage in advance"
      << ", and dk & dv only exist when piggybacking gradients of the last round";
    auto meta = NDArrayMeta().set_dtype(_dtype)
                             .set_device(_local_device)
                             .set_shape(_dkv_shape);
    int64_t kv_numel = 1;
    for (auto x : _kv_shape) {
      kv_numel *= x;
    }
    return NDArray(meta, _attn_storage->storage(), kv_numel);
  }

  // [batch_size, kv_seq_len, num_heads, head_dim]
  NDArray get_4d_k() {
    HT_ASSERT(_has_storage == true)
      << "please ensure you've binded " << _name << " to a storage in advance";
    HTShape k_shape = _kv_shape;
    k_shape.at(0) /= 2;
    auto meta = NDArrayMeta().set_dtype(_dtype)
                             .set_device(_local_device)
                             .set_shape(k_shape);
    return NDArray(meta, _attn_storage->storage());
  }

  // [batch_size, kv_seq_len, num_heads, head_dim]
  NDArray get_4d_v() {
    HT_ASSERT(_has_storage == true)
      << "please ensure you've binded " << _name << " to a storage in advance";
    HTShape v_shape = _kv_shape;
    v_shape.at(0) /= 2;
    auto meta = NDArrayMeta().set_dtype(_dtype)
                             .set_device(_local_device)
                             .set_shape(v_shape);
    int64_t kv_numel = 1;
    for (auto x : _kv_shape) {
      kv_numel *= x;
    }
    return NDArray(meta, _attn_storage->storage(), kv_numel / 2);
  }

  // [batch_size, dkv_seq_len, num_heads, head_dim]
  NDArray get_4d_acc_dk() {
    HT_ASSERT(_piggyback_grad == true && _has_storage == true)
      << "please ensure you've binded " << _name << " to a storage in advance"
      << ", and dk & dv only exist when piggybacking gradients of the last round";
    HTShape dk_shape = _dkv_shape;
    dk_shape.at(0) /= 2;
    auto meta = NDArrayMeta().set_dtype(_dtype)
                             .set_device(_local_device)
                             .set_shape(dk_shape);
    int64_t kv_numel = 1;
    for (auto x : _kv_shape) {
      kv_numel *= x;
    }
    return NDArray(meta, _attn_storage->storage(), kv_numel);
  }

  // [batch_size, dkv_seq_len, num_heads, head_dim]
  NDArray get_4d_acc_dv() {
    HT_ASSERT(_piggyback_grad == true && _has_storage == true)
      << "please ensure you've binded " << _name << " to a storage in advance"
      << ", and dk & dv only exist when piggybacking gradients of the last round";
    HTShape dv_shape = _dkv_shape;
    dv_shape.at(0) /= 2;
    auto meta = NDArrayMeta().set_dtype(_dtype)
                             .set_device(_local_device)
                             .set_shape(dv_shape);
    int64_t kv_numel = 1;
    for (auto x : _kv_shape) {
      kv_numel *= x;
    }
    int64_t dkv_numel = 1;
    for (auto x : _dkv_shape) {
      dkv_numel *= x;
    }
    return NDArray(meta, _attn_storage->storage(), kv_numel + dkv_numel / 2);
  }

 protected:
  bool _piggyback_grad;
  Device _local_device;
  HTShape _kv_shape;
  HTShape _dkv_shape;
  DataType _dtype;
  std::string _name;
  bool _has_storage{false};
  std::shared_ptr<AttnStorage> _attn_storage;
};

class AttnCtx {
 public:
  AttnCtx() = default;

  void release() {
    q = NDArray();
    k = NDArray();
    v = NDArray();
    acc_out = NDArray();
    acc_softmax_lse = NDArray();
    rng_state_list = std::vector<NDArray>();
  }

 public:
  NDArray q;
  NDArray k; 
  NDArray v;
  NDArray acc_out; 
  NDArray acc_softmax_lse;
  std::vector<NDArray> rng_state_list;
};

// Runtime时才能建立
// 因为每个micro batch的seq len与batch size都可能不一样
class AttnCommRing {
 public:
  AttnCommRing(const Operator& op, const hetu::impl::comm::NCCLCommunicationGroup& nccl_comm_group, StreamIndex stream_idx, 
               int64_t ring_idx, const DeviceGroupList& tp_group_list, const std::vector<int64_t>& seq_len_list,
               int64_t batch_size, int64_t q_num_heads, int64_t kv_num_heads, int64_t head_dim, double softmax_scale, double p_dropout, 
               bool packing, int64_t packing_num, const NDArray& cu_seqlens_q, const NDArray& cu_seqlens_k, int64_t max_seqlen_q, int64_t max_seqlen_k,
               size_t kv_storage_size = 2);

  void GenerateAttnInfo();

  void PrepareKVBlocks(const NDArray& local_k, const NDArray& local_v, bool reuse_local_kv_storage = false, bool piggyback_grad = false);

  void PrepareStorageFwd(const NDArray& local_q, const NDArray& local_k, const NDArray& local_v, const NDArray& local_out);

  void PrepareStorageBwd(const std::shared_ptr<AttnCtx>& attn_ctx, const NDArray& grad_output, const NDArray& local_dq);

  void SaveCtx(const std::shared_ptr<AttnCtx>& attn_ctx);

  void SaveGradient(NDArray& local_dq, NDArray& local_dk, NDArray& local_dv);

  void ExecCorr(const NDArray& out, const NDArray& softmax_lse,
                NDArray& acc_out, NDArray& acc_softmax_lse, bool is_first_time = false);
  
  void ExecFlashAttn(int64_t q_idx, int64_t kv_idx, 
                     const NDArray& q, const NDArray& k, const NDArray& v,
                     NDArray& out, NDArray& softmax_lse, NDArray& rng_state,
                     bool is_bwd = false, NDArray grad_output = NDArray(), 
                     NDArray dq = NDArray(), NDArray dk = NDArray(), NDArray dv = NDArray());

  void ExecComm(const NDArray& send_data, const NDArray& recv_data,
                const std::vector<Device>& dst_devices, const std::vector<Device>& src_devices,
                const Stream& comm_stream, bool is_3d = false);

  void Run(bool is_bwd = false);

  void Profile(const Operator& op, size_t micro_batch_id, bool is_bwd = false);
 
 protected:
  // basic settings
  Device _local_device;
  StreamIndex _stream_idx;
  hetu::impl::comm::NCCLCommunicationGroup _nccl_comm_group;
  int64_t _ring_idx;
  int64_t _ring_size;
  DeviceGroupList _tp_group_list;
  std::vector<int64_t> _seq_len_list;
  int64_t _batch_size;
  int64_t _q_num_heads; // local device
  int64_t _kv_num_heads; // local device
  int64_t _head_dim;
  double _softmax_scale;
  double _p_dropout;

  // packing settings
  bool _packing;
  int64_t _packing_num;
  NDArray _cu_seqlens_q, _cu_seqlens_k; 
  int64_t _max_seqlen_q, _max_seqlen_k;

  // alg-related settings
  AttnSplitPattern _attn_split_pattern;
  std::vector<std::shared_ptr<AttnInfo>> _attn_info_list; // ring的两两rank间有一个info

  // memory settings
  NDArray _local_q, _grad_output, _acc_dq; 
  NDArray _out, _acc_out;
  NDArray _softmax_lse, _acc_softmax_lse, _acc_softmax_lse_transposed; // _transposed is for fwd
  // std::shared_ptr<NDArrayStorage> _dq_storage, _dk_storage, _dv_storage; // 不定长因此这里按最大seq_len去分配显存
  std::vector<NDArray> _rng_state_list;
  std::vector<std::shared_ptr<AttnBlock>> _kv_block_list; // ring的每个rank有一个kv堆叠的block
  std::shared_ptr<AttnBlock> _final_kv_block, _final_next_kv_block; // 转一轮后回到初始状态的kv block和一开始的需要进行区分（如果cp数不能整除storage数）
  std::vector<std::shared_ptr<AttnStorage>> _kv_storage_list; // less or equal than _kv_block_list, controlled by _kv_storage_size
  std::unordered_map<int64_t, int64_t> _kv_block_to_storage_map;
  size_t _kv_storage_size{2};
  bool _is_storage_prepared{false};

  // profile settings
  bool _need_profile{false};
  std::vector<std::shared_ptr<hetu::impl::CUDAEvent>> _comm_profile_start_event_list, _comm_profile_end_event_list, _attn_profile_start_event_list, _attn_profile_end_event_list, _corr_profile_start_event_list, _corr_profile_end_event_list, _grad_profile_start_event_list, _grad_profile_end_event_list;
};

/****************************************************************
 ------------------------ Normal Op Impl ------------------------
*****************************************************************/

class ParallelAttentionOpImpl;
class ParallelAttentionOp;
class ParallelAttentionGradientOpImpl;
class ParallelAttentionGradientOp;
class FlashAttentionOpImpl;
class FlashAttentionOp;
class FlashAttentionGradientOpImpl;
class FlashAttentionGradientOp;

class ParallelAttentionOpImpl : public OpInterface {
 private:
  friend class ParallelAttentionOp;
  struct constructor_access_key {};

 public:
  ParallelAttentionOpImpl(int64_t head_dim, int64_t group_query_ratio,
                          SyShapeList multi_seq_lens_symbol, SyShapeList multi_cp_group_symbol,
                          bool packing, IntSymbol max_seqlen_q, IntSymbol max_seqlen_k,
                          double p_dropout, double softmax_scale, 
                          bool is_causal, bool return_softmax)
  : OpInterface(quote(ParallelAttentionOp)), 
    _head_dim(head_dim), _group_query_ratio(group_query_ratio), _multi_seq_lens_symbol(std::move(multi_seq_lens_symbol)), _multi_cp_group_symbol(std::move(multi_cp_group_symbol)),
    _packing(packing), _max_seqlen_q(std::move(max_seqlen_q)), _max_seqlen_k(std::move(max_seqlen_k)),
    _p_dropout(p_dropout), _softmax_scale(softmax_scale), _is_causal(is_causal), _return_softmax(return_softmax) {
    _attn_ctx_list.reserve(HT_MAX_NUM_MICRO_BATCHES);
    for (size_t i = 0; i < HT_MAX_NUM_MICRO_BATCHES; i++) {
      _attn_ctx_list.emplace_back(std::make_shared<AttnCtx>());
    }
  }

  uint64_t op_indicator() const noexcept override {
    return PARALLEL_ATTN_OP;
  } 

  inline const std::shared_ptr<AttnCtx>& attn_ctx() const {
    return _attn_ctx_list.at(_attn_ctx_num);
  }
  
  inline int64_t get_max_seqlen_q() const {
    HT_ASSERT(_max_seqlen_q)
      << "get_max_seqlen_q() can only be called when packing";
    return _max_seqlen_q->get_val();
  }

  inline int64_t get_max_seqlen_k() const {
    HT_ASSERT(_max_seqlen_k)
      << "get_max_seqlen_k() can only be called when packing";
    return _max_seqlen_k->get_val();
  }

  inline int64_t head_dim() const {
    return _head_dim;
  }

  inline int64_t group_query_ratio() const {
    return _group_query_ratio;
  }

  inline double p_dropout() const {
    return _p_dropout;
  }

  inline double softmax_scale() const {
    return _softmax_scale;
  }

  inline bool is_causal() const {
    return _is_causal;
  }

  inline bool return_softmax() const {
    return _return_softmax;
  }

 protected:
  std::vector<NDArrayMeta> DoInferMeta(const TensorList& inputs,
                                       const InstantiationContext& inst_ctx) const override;

  void DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                      const OpMeta& op_meta,
                      const InstantiationContext& inst_ctx) const override;

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) const override;

  TensorList DoGradient(Operator& op, const TensorList& grad_outputs) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes, RuntimeContext& ctx) const override;

  mutable size_t _attn_ctx_num{0};
  std::vector<std::shared_ptr<AttnCtx>> _attn_ctx_list;
  int64_t _head_dim;
  int64_t _group_query_ratio;
  SyShapeList _multi_seq_lens_symbol;
  SyShapeList _multi_cp_group_symbol;
  bool _packing;
  IntSymbol _max_seqlen_q;
  IntSymbol _max_seqlen_k;
  double _p_dropout;
  double _softmax_scale;
  bool _is_causal;
  bool _return_softmax;

 public:
  inline bool require_contig_inputs() const override {
    return false;
  }

  bool operator==(const OpInterface& rhs) const override {
    if (OpInterface::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const ParallelAttentionOpImpl&>(rhs);
      return _head_dim == rhs_.head_dim() 
             && _group_query_ratio == rhs_.group_query_ratio() 
             && p_dropout() == rhs_.p_dropout() 
             && softmax_scale() == rhs_.softmax_scale() 
             && is_causal() == rhs_.is_causal() 
             && return_softmax() == rhs_.return_softmax();
    } 
    else
      return false;
  }
};

TensorList MakeParallelAttentionOp(Tensor qkv, int64_t head_dim, int64_t group_query_ratio,
                                   SyShapeList multi_seq_lens_symbol, SyShapeList multi_cp_group_symbol, 
                                   bool packing, Tensor cu_seqlens_q, Tensor cu_seqlens_k, IntSymbol max_seqlen_q, IntSymbol max_seqlen_k,
                                   double p_dropout = 0.0, double softmax_scale = -1.0, 
                                   bool is_causal = false, bool return_softmax = false, OpMeta op_meta = OpMeta());

class ParallelAttentionGradientOpImpl: public OpInterface {

 public:
  ParallelAttentionGradientOpImpl(const std::vector<std::shared_ptr<AttnCtx>>& attn_ctx_list, int64_t head_dim, int64_t group_query_ratio,
                                  SyShapeList multi_seq_lens_symbol, SyShapeList multi_cp_group_symbol,
                                  bool packing, IntSymbol max_seqlen_q, IntSymbol max_seqlen_k,
                                  double p_dropout, double softmax_scale, bool is_causal)
  : OpInterface(quote(ParallelAttentionGradientOp)), 
    _attn_ctx_list(attn_ctx_list), _head_dim(head_dim), _group_query_ratio(group_query_ratio), 
    _multi_seq_lens_symbol(std::move(multi_seq_lens_symbol)), _multi_cp_group_symbol(std::move(multi_cp_group_symbol)),
    _packing(packing), _max_seqlen_q(std::move(max_seqlen_q)), _max_seqlen_k(std::move(max_seqlen_k)),
    _p_dropout(p_dropout), _softmax_scale(softmax_scale), _is_causal(is_causal) {
  }

  uint64_t op_indicator() const noexcept override {
    return PARALLEL_ATTN_GRAD_OP;
  } 

  inline const std::shared_ptr<AttnCtx>& attn_ctx() const {
    return _attn_ctx_list.at(_attn_ctx_num);
  }

  inline int64_t get_max_seqlen_q() const {
    HT_ASSERT(_max_seqlen_q)
      << "get_max_seqlen_q() can only be called when packing";
    return _max_seqlen_q->get_val();
  }

  inline int64_t get_max_seqlen_k() const {
    HT_ASSERT(_max_seqlen_k)
      << "get_max_seqlen_k() can only be called when packing";
    return _max_seqlen_k->get_val();
  }

  inline int64_t head_dim() const {
    return _head_dim;
  }

  inline int64_t group_query_ratio() const {
    return _group_query_ratio;
  }

  inline double p_dropout() const {
    return _p_dropout;
  }

  inline double softmax_scale() const {
    return _softmax_scale;
  }

  inline bool is_causal() const {
    return _is_causal;
  }

 protected:
  std::vector<NDArrayMeta> DoInferMeta(const TensorList& inputs,
                                       const InstantiationContext& inst_ctx) const override;

  void DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                      const OpMeta& op_meta,
                      const InstantiationContext& inst_ctx) const override;

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes, RuntimeContext& ctx) const override;

  mutable size_t _attn_ctx_num{0};
  std::vector<std::shared_ptr<AttnCtx>> _attn_ctx_list; // for different micro batches
  int64_t _head_dim;
  int64_t _group_query_ratio;
  SyShapeList _multi_seq_lens_symbol;
  SyShapeList _multi_cp_group_symbol;
  bool _packing;
  IntSymbol _max_seqlen_q;
  IntSymbol _max_seqlen_k;
  double _p_dropout;
  double _softmax_scale;
  bool _is_causal;

 public:
  inline bool require_contig_inputs() const override {
    return false;
  }

  bool operator==(const OpInterface& rhs) const override {
    if (OpInterface::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const ParallelAttentionGradientOpImpl&>(rhs);
      return _head_dim == rhs_.head_dim()
             && _group_query_ratio == rhs_.group_query_ratio()
             && p_dropout() == rhs_.p_dropout()
             && softmax_scale() == rhs_.softmax_scale()
             && is_causal() == rhs_.is_causal();
    } 
    else
      return false;
  }
};

TensorList MakeParallelAttentionGradientOp(const std::vector<std::shared_ptr<AttnCtx>>& attn_ctx_list, 
                                           Tensor grad_out, int64_t head_dim, int64_t group_query_ratio,
                                           SyShapeList multi_seq_lens_symbol, SyShapeList multi_cp_group_symbol, 
                                           bool packing, Tensor cu_seqlens_q, Tensor cu_seqlens_k, IntSymbol max_seqlen_q, IntSymbol max_seqlen_k,
                                           double p_dropout = 0.0, double softmax_scale = -1.0,
                                           bool is_causal = false, OpMeta op_meta = OpMeta());

class FlashAttentionOpImpl final : public ParallelAttentionOpImpl {
 private:
  friend class FlashAttentionOp;
  struct constructor_access_key {};

 public:
  FlashAttentionOpImpl(int64_t head_dim, int64_t group_query_ratio,
                       SyShapeList multi_seq_lens_symbol, SyShapeList multi_cp_group_symbol,
                       bool packing, IntSymbol max_seqlen_q, IntSymbol max_seqlen_k,
                       double p_dropout, double softmax_scale, 
                       bool is_causal, bool return_softmax)
  : ParallelAttentionOpImpl(head_dim, group_query_ratio, multi_seq_lens_symbol, 
                            multi_cp_group_symbol, packing, max_seqlen_q, max_seqlen_k,
                            p_dropout, softmax_scale, is_causal, return_softmax){
  }

  uint64_t op_indicator() const noexcept override {
    return PARALLEL_ATTN_OP;
  } 

  inline void set_attn_ctx_num(size_t attn_ctx_num) {
    _attn_ctx_num = attn_ctx_num;
  }

  inline const std::shared_ptr<AttnCtx>& attn_ctx() const {
    return _attn_ctx_list.at(_attn_ctx_num);
  }
  
  inline int64_t max_seqlen_q() const {
    return _max_seqlen_q->get_val();
  }

  inline int64_t max_seqlen_k() const {
    return _max_seqlen_k->get_val();
  }

  inline int64_t head_dim() const {
    return _head_dim;
  }

  inline int64_t group_query_ratio() const {
    return _group_query_ratio;
  }

  inline double p_dropout() const {
    return _p_dropout;
  }

  inline double softmax_scale() const {
    return _softmax_scale;
  }

  inline bool is_causal() const {
    return _is_causal;
  }

  inline bool return_softmax() const {
    return _return_softmax;
  }

 protected:
  std::vector<NDArrayMeta> DoInferMeta(const TensorList& inputs,
                                       const InstantiationContext& inst_ctx) const override;

  void DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                      const OpMeta& op_meta,
                      const InstantiationContext& inst_ctx) const override;

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) const override;

  TensorList DoGradient(Operator& op, const TensorList& grad_outputs) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes, RuntimeContext& ctx) const override;

  // size_t _attn_ctx_num{0};
  // std::vector<std::shared_ptr<AttnCtx>> _attn_ctx_list;
  // int64_t _head_dim;
  // int64_t _group_query_ratio;
  // SyShapeList _multi_seq_lens_symbol;
  // SyShapeList _multi_cp_group_symbol;
  // bool _packing;
  // IntSymbol _max_seqlen_q;
  // IntSymbol _max_seqlen_k;
  // double _p_dropout;
  // double _softmax_scale;
  // bool _is_causal;
  // bool _return_softmax;

 public:
  inline bool require_contig_inputs() const override {
    return false;
  }

  bool operator==(const OpInterface& rhs) const override {
    if (OpInterface::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const FlashAttentionOpImpl&>(rhs);
      return _head_dim == rhs_.head_dim() 
             && _group_query_ratio == rhs_.group_query_ratio() 
             && p_dropout() == rhs_.p_dropout() 
             && softmax_scale() == rhs_.softmax_scale() 
             && is_causal() == rhs_.is_causal() 
             && return_softmax() == rhs_.return_softmax();
    } 
    else
      return false;
  }
};

TensorList MakeFlashAttentionOp(Tensor q, Tensor k, Tensor v,
                                int64_t head_dim, int64_t group_query_ratio,
                                SyShapeList multi_seq_lens_symbol, SyShapeList multi_cp_group_symbol, 
                                bool packing, Tensor cu_seqlens_q, Tensor cu_seqlens_k, IntSymbol max_seqlen_q, IntSymbol max_seqlen_k,
                                double p_dropout = 0.0, double softmax_scale = -1.0, 
                                bool is_causal = false, bool return_softmax = false, OpMeta op_meta = OpMeta());

class FlashAttentionGradientOpImpl final : public ParallelAttentionGradientOpImpl {

 public:
  FlashAttentionGradientOpImpl(const std::vector<std::shared_ptr<AttnCtx>>& attn_ctx_list, int64_t head_dim, int64_t group_query_ratio,
                               SyShapeList multi_seq_lens_symbol, SyShapeList multi_cp_group_symbol,
                               bool packing, IntSymbol max_seqlen_q, IntSymbol max_seqlen_k,
                               double p_dropout, double softmax_scale, bool is_causal)
  : ParallelAttentionGradientOpImpl(attn_ctx_list, head_dim, group_query_ratio, 
                                    multi_seq_lens_symbol, multi_cp_group_symbol, 
                                    packing, max_seqlen_q, max_seqlen_k, p_dropout, 
                                    softmax_scale, is_causal) {
  }

  uint64_t op_indicator() const noexcept override {
    return PARALLEL_ATTN_GRAD_OP;
  } 

  inline void set_attn_ctx_num(size_t attn_ctx_num) {
    _attn_ctx_num = attn_ctx_num;
  }

  inline const std::shared_ptr<AttnCtx>& attn_ctx() const {
    return _attn_ctx_list.at(_attn_ctx_num);
  }

  inline int64_t max_seqlen_q() const {
    return _max_seqlen_q->get_val();
  }

  inline int64_t max_seqlen_k() const {
    return _max_seqlen_k->get_val();
  }

  inline int64_t head_dim() const {
    return _head_dim;
  }

  inline int64_t group_query_ratio() const {
    return _group_query_ratio;
  }

  inline double p_dropout() const {
    return _p_dropout;
  }

  inline double softmax_scale() const {
    return _softmax_scale;
  }

  inline bool is_causal() const {
    return _is_causal;
  }

 protected:
  std::vector<NDArrayMeta> DoInferMeta(const TensorList& inputs,
                                       const InstantiationContext& inst_ctx) const override;

  void DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                      const OpMeta& op_meta,
                      const InstantiationContext& inst_ctx) const override;

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes, RuntimeContext& ctx) const override;

  // size_t _attn_ctx_num{0};
  // std::vector<std::shared_ptr<AttnCtx>> _attn_ctx_list; // for different micro batches
  // int64_t _head_dim;
  // int64_t _group_query_ratio;
  // SyShapeList _multi_seq_lens_symbol;
  // SyShapeList _multi_cp_group_symbol;
  // bool _packing;
  // IntSymbol _max_seqlen_q;
  // IntSymbol _max_seqlen_k;
  // double _p_dropout;
  // double _softmax_scale;
  // bool _is_causal;

 public:
  inline bool require_contig_inputs() const override {
    return false;
  }

  bool operator==(const OpInterface& rhs) const override {
    if (OpInterface::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const FlashAttentionGradientOpImpl&>(rhs);
      return _head_dim == rhs_.head_dim()
             && _group_query_ratio == rhs_.group_query_ratio()
             && p_dropout() == rhs_.p_dropout()
             && softmax_scale() == rhs_.softmax_scale()
             && is_causal() == rhs_.is_causal();
    } 
    else
      return false;
  }
};

TensorList MakeFlashAttentionGradientOp(const std::vector<std::shared_ptr<AttnCtx>>& attn_ctx_list, 
                                        Tensor grad_out, int64_t head_dim, int64_t group_query_ratio,
                                        SyShapeList multi_seq_lens_symbol, SyShapeList multi_cp_group_symbol, 
                                        bool packing, Tensor cu_seqlens_q, Tensor cu_seqlens_k, IntSymbol max_seqlen_q, IntSymbol max_seqlen_k,
                                        double p_dropout = 0.0, double softmax_scale = -1.0,
                                        bool is_causal = false, OpMeta op_meta = OpMeta());

} // namespace graph
} // namespace hetu
