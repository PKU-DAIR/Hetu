#include "hetu/graph/ops/Rotary.h"
#include "hetu/graph/headers.h"
#include "hetu/graph/ops/kernel_links.h"
#include "hetu/impl/utils/dispatch.h"
#include <numeric>
#include <vector>

namespace hetu {
namespace graph {

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

static int get_split_pattern_idx() {
  auto split_pattern = std::getenv("HETU_PARALLEL_ATTN_SPLIT_PATTERN");
  if(split_pattern == nullptr) return 0;
  int split_pattern_idx = 0;  // 默认为0，即顺序切分
  if(std::string(split_pattern) == "NORMAL") split_pattern_idx = 0;
  else if(std::string(split_pattern) == "SYM") split_pattern_idx = 1;
  else if(std::string(split_pattern) == "STRIPE") split_pattern_idx = 2;
  return split_pattern_idx;
}

void calculate_start_and_end_seq_id(int& start_seq_id, int& end_seq_id, const std::vector<int64_t>& seq_len_list, 
                                    int cp_rank, int cp_size, int split_pattern_idx) {
  start_seq_id = 0;
  end_seq_id = 0;
  for(int i = 0; i < cp_size; ++i) {
    end_seq_id += seq_len_list[i];
  }
  for(int i = 0; i < cp_rank; ++i) {
    if(split_pattern_idx == 0) {
      start_seq_id += seq_len_list[i];
    }
    else if(split_pattern_idx == 1) {
      start_seq_id += seq_len_list[i] / 2;
      end_seq_id -= (seq_len_list[i] - seq_len_list[i] / 2);
    }
  }
}

void calculate_start_and_end_seq_ids(std::vector<int>& start_seq_ids_q, std::vector<int>& end_seq_ids_q,
                                      std::vector<int>& original_seqlens_q, const NDArray& cu_seqlens_q,
                                      std::vector<int>& start_seq_ids_k, std::vector<int>& end_seq_ids_k,
                                      std::vector<int>& original_seqlens_k, const NDArray& cu_seqlens_k,
                                      int cp_rank, int cp_size, int split_pattern_idx) {
  int packing_num = cu_seqlens_q->shape(-1) - 1;
  HT_DISPATCH_INTEGER_TYPES(
    cu_seqlens_q->dtype(), spec_t, __FUNCTION__, [&]() {
      size_t size_q = cu_seqlens_q->numel();
      size_t size_k = cu_seqlens_k->numel();
      spec_t cu_seqlens_q_data_ptr[size_q];
      spec_t cu_seqlens_k_data_ptr[size_k];
      if(cu_seqlens_q->is_cpu()) {
        memcpy(cu_seqlens_q_data_ptr, cu_seqlens_q->data_ptr<spec_t>(), size_q * sizeof(spec_t));
        memcpy(cu_seqlens_k_data_ptr, cu_seqlens_k->data_ptr<spec_t>(), size_k * sizeof(spec_t));
      }
      else {
        hetu::cuda::CUDADeviceGuard guard(cu_seqlens_q->device().index());
        const spec_t* dev_ptr_q = cu_seqlens_q->data_ptr<spec_t>();
        const spec_t* dev_ptr_k = cu_seqlens_k->data_ptr<spec_t>();
        CudaMemcpy(cu_seqlens_q_data_ptr, dev_ptr_q, size_q * sizeof(spec_t),
                  cudaMemcpyDeviceToHost);
        CudaMemcpy(cu_seqlens_k_data_ptr, dev_ptr_k, size_k * sizeof(spec_t),
                  cudaMemcpyDeviceToHost);
      }
      
      for(int i = 0; i < cp_size; ++i) {
        for(int j = 0; j < packing_num; ++j) {
          original_seqlens_q[j] += cu_seqlens_q_data_ptr[i*(packing_num+1) + j+1] 
                                      - cu_seqlens_q_data_ptr[i*(packing_num+1) + j];
          original_seqlens_k[j] += cu_seqlens_k_data_ptr[i*(packing_num+1) + j+1] 
                                      - cu_seqlens_k_data_ptr[i*(packing_num+1) + j];
        }
      }
      for(int j = 0; j < packing_num; ++j) {
        end_seq_ids_q[j] = original_seqlens_q[j];
        end_seq_ids_k[j] = original_seqlens_k[j];
      }
      for(int i = 0; i < cp_rank; ++i) {
        for(int j = 0; j < packing_num; ++j) {
          int cp_seqlen_q = cu_seqlens_q_data_ptr[i*(packing_num+1) + j+1] 
                                - cu_seqlens_q_data_ptr[i*(packing_num+1) + j];
          int cp_seqlen_k = cu_seqlens_k_data_ptr[i*(packing_num+1) + j+1] 
                                - cu_seqlens_k_data_ptr[i*(packing_num+1) + j];
          if(split_pattern_idx == 0) {
            // 此时并不需要end_seq_id
            start_seq_ids_q[j] += cp_seqlen_q;
            start_seq_ids_k[j] += cp_seqlen_k;
          }
          else if(split_pattern_idx == 1) {
            start_seq_ids_q[j] += cp_seqlen_q / 2;
            start_seq_ids_k[j] += cp_seqlen_k / 2;
            end_seq_ids_q[j] -= (cp_seqlen_q - cp_seqlen_q / 2);
            end_seq_ids_k[j] -= (cp_seqlen_k - cp_seqlen_k / 2);
          }
          else {
            HT_ASSERT(false) << "Rotary op currently doesn't support other split_pattern";
          }
        }
      }
    }
  );
}

NDArrayList RotaryOpImpl::DoCompute(Operator& op,
                                    const NDArrayList& inputs,
                                    RuntimeContext& ctx) const {
  NDArrayList outputs = inplace() ? inputs : DoAllocOutputs(op, inputs, ctx);
  DoCompute(op, inputs, outputs, ctx);
  return outputs;
}

void RotaryOpImpl::DoCompute(Operator& op, 
                             const NDArrayList& inputs, NDArrayList& outputs,
                             RuntimeContext& ctx) const {
  auto qkv = inputs.at(0);  // {batch_size_mul_seq_len, num_heads_mul_head_dim}
  auto cos = inputs.at(1);
  auto sin = inputs.at(2);
  auto output = outputs.at(0);  // same shape as qkv

  HTShape input_shape = qkv->shape();
  HT_ASSERT(input_shape.size() == 2)
    << "Rotary only support input shape [batch_size * seq_len, num_heads * head_dim * 3]";
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
  // HT_LOG_INFO << "total_num_heads is " << total_num_heads;
  // HT_LOG_INFO << "_head_dim is " << _head_dim;
  HT_ASSERT(total_num_heads % (_group_query_ratio + 2) == 0)
    << "total_num_heads should be divided by (group_query_ratio + 2)";
  int64_t q_num_heads = total_num_heads / (_group_query_ratio + 2) * _group_query_ratio;
  int64_t kv_num_heads = total_num_heads / (_group_query_ratio + 2);
  auto stream_idx = op->instantiation_ctx().stream_index;
  auto reshaped_qkv = NDArray::view(qkv, {batch_size, seq_len, total_num_heads, _head_dim});
  auto reshaped_output = NDArray::view(output, {batch_size, seq_len, total_num_heads, _head_dim});
  // self-attn
  HTShape q_shape = {batch_size, seq_len, q_num_heads, _head_dim};
  HTShape k_shape = {batch_size, seq_len, kv_num_heads, _head_dim};
  HTShape v_shape = {batch_size, seq_len, kv_num_heads, _head_dim};
  HTShape q_begin_pos = {0, 0, 0, 0};
  HTShape k_begin_pos = {0, 0, q_num_heads, 0};
  HTShape v_begin_pos = {0, 0, q_num_heads + kv_num_heads, 0};
  // 这里应该不需要这个NDArray::contiguous，因为Rotary.cu中会使用stride
  auto q = NDArray::slice(reshaped_qkv, q_begin_pos, q_shape, stream_idx);
  auto k = NDArray::slice(reshaped_qkv, k_begin_pos, k_shape, stream_idx);
  auto v = NDArray::slice(reshaped_qkv, v_begin_pos, v_shape, stream_idx);

  auto out_q = NDArray::slice(reshaped_output, q_begin_pos, q_shape, stream_idx);
  auto out_k = NDArray::slice(reshaped_output, k_begin_pos, k_shape, stream_idx);
  auto out_v = NDArray::slice(reshaped_output, v_begin_pos, v_shape, stream_idx);

  int64_t ring_idx;
  DeviceGroupList tp_group_list;
  std::vector<int64_t> seq_len_list;
  std::tie(ring_idx, tp_group_list, seq_len_list) = get_local_ring(op->input(0), _multi_seq_lens_symbol, _multi_cp_group_symbol);

  if (tp_group_list.size() >= 2) {
    // 开cp
    int cp_size = tp_group_list.size();
    int cp_rank = ring_idx;

    int split_pattern_idx = get_split_pattern_idx();

    if(!_packing) {
      int start_seq_id = 0;
      int end_seq_id = 0;
      calculate_start_and_end_seq_id(start_seq_id, end_seq_id, seq_len_list, 
                                     cp_rank, cp_size, split_pattern_idx);
      HT_DISPATCH_KERNEL_CPU_AND_CUDA(op->instantiation_ctx().placement.type(), type(), hetu::impl::Rotary,
                                    q, cos, sin, out_q, start_seq_id, end_seq_id, split_pattern_idx,
                                    op->instantiation_ctx().stream());
      HT_DISPATCH_KERNEL_CPU_AND_CUDA(op->instantiation_ctx().placement.type(), type(), hetu::impl::Rotary,
                                    k, cos, sin, out_k, start_seq_id, end_seq_id, split_pattern_idx,
                                    op->instantiation_ctx().stream());
    }
    else {
      HT_ASSERT(inputs.size() == 5)
        << "packing should have 5 inputs: qkv, cu_seqlens_q and cu_seqlens_k";
      auto cu_seqlens_q = inputs.at(3);  // (cp_size, packed_batch_num+1)
      auto cu_seqlens_k = inputs.at(4);
      HT_LOG_INFO << "here3";
      HT_ASSERT(cu_seqlens_q->shape(0) == cp_size) << "cu_seqlens_q.shape(0) should be equal to cp_size";

      int packing_num = cu_seqlens_q->shape(-1) - 1;
      HT_ASSERT(packing_num == cu_seqlens_k->shape(-1) - 1 && packing_num >= 1)
        << "packing num (>= 1) mismatches";

      // 记录当前cp的每一个packing_seq的开始序号和结束序号
      auto start_seq_ids_q = std::vector(packing_num, 0);
      auto end_seq_ids_q = std::vector(packing_num, 0);
      auto original_seqlens_q = std::vector(packing_num, 0);
      auto start_seq_ids_k = std::vector(packing_num, 0);
      auto end_seq_ids_k = std::vector(packing_num, 0);
      auto original_seqlens_k = std::vector(packing_num, 0);

      calculate_start_and_end_seq_ids(start_seq_ids_q, end_seq_ids_q, original_seqlens_q,
                                      cu_seqlens_q, start_seq_ids_k, end_seq_ids_k,
                                      original_seqlens_k, cu_seqlens_k, cp_rank, cp_size,
                                      split_pattern_idx);

      auto reshaped_qkv = NDArray::view(qkv, {batch_size_mul_seq_len, total_num_heads, _head_dim});
      auto reshaped_output = NDArray::view(output, {batch_size_mul_seq_len, total_num_heads, _head_dim});
      HTShape q_shape = {batch_size_mul_seq_len, q_num_heads, _head_dim};
      HTShape k_shape = {batch_size_mul_seq_len, kv_num_heads, _head_dim};
      HTShape q_begin_pos = {0, 0, 0};
      HTShape k_begin_pos = {0, q_num_heads, 0};
      auto q = NDArray::slice(reshaped_qkv, q_begin_pos, q_shape, stream_idx);
      auto k = NDArray::slice(reshaped_qkv, k_begin_pos, k_shape, stream_idx);
      auto out_q = NDArray::slice(reshaped_output, q_begin_pos, q_shape, stream_idx);
      auto out_k = NDArray::slice(reshaped_output, k_begin_pos, k_shape, stream_idx);

      auto cu_seqlens_q_slice = NDArray::view(NDArray::slice(cu_seqlens_q, {cp_rank, 0}, {1, packing_num+1}, stream_idx), {packing_num+1});
      auto cu_seqlens_k_slice = NDArray::view(NDArray::slice(cu_seqlens_k, {cp_rank, 0}, {1, packing_num+1}, stream_idx), {packing_num+1});

      HT_DISPATCH_KERNEL_CPU_AND_CUDA(op->instantiation_ctx().placement.type(), type(), hetu::impl::RotaryVarlen,
                                   q, cos, sin, out_q, cu_seqlens_q_slice, max_seqlen_q(), start_seq_ids_q.data(),
                                   end_seq_ids_q.data(), split_pattern_idx, op->instantiation_ctx().stream());
      HT_DISPATCH_KERNEL_CPU_AND_CUDA(op->instantiation_ctx().placement.type(), type(), hetu::impl::RotaryVarlen,
                                   k, cos, sin, out_k, cu_seqlens_k_slice, max_seqlen_k(), start_seq_ids_k.data(),
                                   end_seq_ids_k.data(), split_pattern_idx, op->instantiation_ctx().stream());
    }
  }
  else {
    // 不开cp
    if(!_packing) {
      HT_DISPATCH_KERNEL_CPU_AND_CUDA(op->instantiation_ctx().placement.type(), type(), hetu::impl::Rotary,
                                   q, cos, sin, out_q, 0, 0, 0, op->instantiation_ctx().stream());
      HT_DISPATCH_KERNEL_CPU_AND_CUDA(op->instantiation_ctx().placement.type(), type(), hetu::impl::Rotary,
                                   k, cos, sin, out_k, 0, 0, 0, op->instantiation_ctx().stream());
    }
    else {
      HT_ASSERT(inputs.size() == 5)
        << "packing should have 5 inputs: qkv, cu_seqlens_q and cu_seqlens_k";
      auto cu_seqlens_q = inputs.at(3);
      auto cu_seqlens_k = inputs.at(4);
      // 注意，与ParallenAttn不同，此处不能使用view，因为该函数会假设输入Tensor是连续的，
      // 而实际上q,k在此处并不连续
      auto reshaped_qkv = NDArray::view(qkv, {batch_size_mul_seq_len, total_num_heads, _head_dim});
      auto reshaped_output = NDArray::view(output, {batch_size_mul_seq_len, total_num_heads, _head_dim});
      HTShape q_shape = {batch_size_mul_seq_len, q_num_heads, _head_dim};
      HTShape k_shape = {batch_size_mul_seq_len, kv_num_heads, _head_dim};
      HTShape q_begin_pos = {0, 0, 0};
      HTShape k_begin_pos = {0, q_num_heads, 0};
      auto q = NDArray::slice(reshaped_qkv, q_begin_pos, q_shape, stream_idx);
      auto k = NDArray::slice(reshaped_qkv, k_begin_pos, k_shape, stream_idx);
      auto out_q = NDArray::slice(reshaped_output, q_begin_pos, q_shape, stream_idx);
      auto out_k = NDArray::slice(reshaped_output, k_begin_pos, k_shape, stream_idx);

      int packing_num = cu_seqlens_q->shape(-1) - 1;
      auto start_seq_ids = std::vector(packing_num, 0);
      auto end_seq_ids = std::vector(packing_num, 0);

      HT_DISPATCH_KERNEL_CPU_AND_CUDA(op->instantiation_ctx().placement.type(), type(), hetu::impl::RotaryVarlen,
                                   q, cos, sin, out_q, cu_seqlens_q, max_seqlen_q(), start_seq_ids.data(),
                                   end_seq_ids.data(), 0, op->instantiation_ctx().stream());
      HT_DISPATCH_KERNEL_CPU_AND_CUDA(op->instantiation_ctx().placement.type(), type(), hetu::impl::RotaryVarlen,
                                   k, cos, sin, out_k, cu_seqlens_k, max_seqlen_k(), start_seq_ids.data(),
                                   end_seq_ids.data(), 0, op->instantiation_ctx().stream());
    }
  }

  if(!inplace()) {
    // copy v
    NDArray::copy(v, op->instantiation_ctx().stream_index, out_v);
  }
}

TensorList RotaryOpImpl::DoGradient(Operator& op, const TensorList& grad_outputs) const {
  auto grad_input = op->requires_grad(0) ? MakeRotaryGradientOp(
                                           grad_outputs.at(0), op->input(1), op->input(2),
                                           _head_dim, _group_query_ratio, 
                                           _multi_seq_lens_symbol, _multi_cp_group_symbol, _packing, 
                                           _packing? op->input(3): Tensor(),
                                           _packing? op->input(4): Tensor(),
                                           _max_seqlen_q, _max_seqlen_k,
                                           interleaved(), inplace(),
                                           op->grad_op_meta().set_name(op->grad_name()))
                                         : Tensor();
  if(_packing) {
    return {grad_input, Tensor(), Tensor(), Tensor(), Tensor()};
  }
  return {grad_input, Tensor(), Tensor()};
}

HTShapeList RotaryOpImpl::DoInferShape(Operator& op, const HTShapeList& input_shapes, RuntimeContext& ctx) const {
  return {input_shapes.at(0)};
}

void RotaryOpImpl::DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                                  const OpMeta& op_meta, const InstantiationContext& inst_ctx) const {
  return outputs.at(0)->set_distributed_states(inputs.at(0)->get_distributed_states());
}

void RotaryOpImpl::DoDeduceHeterProp(const std::vector<int32_t>& inputs_hetero_dim,
                                     TensorList& outputs, const OpMeta& op_meta,
                                     const InstantiationContext& inst_ctx) const {
  outputs.at(0)->cur_ds_union().set_hetero_dim(inputs_hetero_dim.at(0));
}

NDArrayList RotaryGradientOpImpl::DoCompute(Operator& op,
                                            const NDArrayList& inputs,
                                            RuntimeContext& ctx) const {
  NDArrayList outputs = inplace() ? inputs : DoAllocOutputs(op, inputs, ctx);
  DoCompute(op, inputs, outputs, ctx);
  return outputs;
}

void RotaryGradientOpImpl::DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                             RuntimeContext& ctx) const {
  auto dout = inputs.at(0); // {batch_size_mul_seq_len, num_heads_mul_head_dim}
  auto cos = inputs.at(1);
  auto sin = inputs.at(2);
  auto dqkv = outputs.at(0);

  HTShape input_shape = dout->shape();
  HT_ASSERT(input_shape.size() == 2)
    << "Rotary only support input shape [batch_size * seq_len, num_heads * head_dim * 3]";
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
  // HT_LOG_INFO << "total_num_heads is " << total_num_heads;
  // HT_LOG_INFO << "_head_dim is " << _head_dim;
  HT_ASSERT(total_num_heads % (_group_query_ratio + 2) == 0)
    << "total_num_heads should be divided by (group_query_ratio + 2)";
  int64_t q_num_heads = total_num_heads / (_group_query_ratio + 2) * _group_query_ratio;
  int64_t kv_num_heads = total_num_heads / (_group_query_ratio + 2);
  auto stream_idx = op->instantiation_ctx().stream_index;
  auto reshaped_dout = NDArray::view(dout, {batch_size, seq_len, total_num_heads, _head_dim});
  auto reshaped_dqkv = NDArray::view(dqkv, {batch_size, seq_len, total_num_heads, _head_dim});
  // self-attn
  HTShape q_shape = {batch_size, seq_len, q_num_heads, _head_dim};
  HTShape k_shape = {batch_size, seq_len, kv_num_heads, _head_dim};
  HTShape v_shape = {batch_size, seq_len, kv_num_heads, _head_dim};
  HTShape q_begin_pos = {0, 0, 0, 0};
  HTShape k_begin_pos = {0, 0, q_num_heads, 0};
  HTShape v_begin_pos = {0, 0, q_num_heads + kv_num_heads, 0};
  // 这里应该不需要这个NDArray::contiguous，因为Rotary.cu中会使用offset_calculator
  auto dout_q = NDArray::slice(reshaped_dout, q_begin_pos, q_shape, stream_idx);
  auto dout_k = NDArray::slice(reshaped_dout, k_begin_pos, k_shape, stream_idx);
  auto dout_v = NDArray::slice(reshaped_dout, v_begin_pos, v_shape, stream_idx);

  auto dq = NDArray::slice(reshaped_dqkv, q_begin_pos, q_shape, stream_idx);
  auto dk = NDArray::slice(reshaped_dqkv, k_begin_pos, k_shape, stream_idx);
  auto dv = NDArray::slice(reshaped_dqkv, v_begin_pos, v_shape, stream_idx);

  int64_t ring_idx;
  DeviceGroupList tp_group_list;
  std::vector<int64_t> seq_len_list;
  std::tie(ring_idx, tp_group_list, seq_len_list) = get_local_ring(op->input(0), _multi_seq_lens_symbol, _multi_cp_group_symbol);

  if (tp_group_list.size() >= 2) {
    // 开cp
    int cp_size = tp_group_list.size();
    int cp_rank = ring_idx;

    int split_pattern_idx = get_split_pattern_idx();

    if(!_packing) {
      int start_seq_id = 0;
      int end_seq_id = 0;
      calculate_start_and_end_seq_id(start_seq_id, end_seq_id, seq_len_list, 
                                     cp_rank, cp_size, split_pattern_idx);
      HT_DISPATCH_KERNEL_CPU_AND_CUDA(op->instantiation_ctx().placement.type(), type(), hetu::impl::RotaryGradient,
                                    dout_q, cos, sin, dq, start_seq_id, end_seq_id, split_pattern_idx,
                                    op->instantiation_ctx().stream());
      HT_DISPATCH_KERNEL_CPU_AND_CUDA(op->instantiation_ctx().placement.type(), type(), hetu::impl::RotaryGradient,
                                    dout_k, cos, sin, dk, start_seq_id, end_seq_id, split_pattern_idx,
                                    op->instantiation_ctx().stream());
    }
    else {
      HT_ASSERT(inputs.size() == 5)
        << "packing should have 5 inputs: qkv, cu_seqlens_q and cu_seqlens_k";
      auto cu_seqlens_q = inputs.at(3);  // (cp_size, packed_batch_num+1)
      auto cu_seqlens_k = inputs.at(4);
      HT_ASSERT(cu_seqlens_q->shape(0) == cp_size) << "cu_seqlens_q.shape(0) should be equal to cp_size";

      int packing_num = cu_seqlens_q->shape(-1) - 1;
      HT_ASSERT(packing_num == cu_seqlens_k->shape(-1) - 1 && packing_num >= 1)
        << "packing num (>= 1) mismatches";

      // 记录当前cp的每一个packing_seq的开始序号和结束序号
      auto start_seq_ids_q = std::vector(packing_num, 0);
      auto end_seq_ids_q = std::vector(packing_num, 0);
      auto original_seqlens_q = std::vector(packing_num, 0);
      auto start_seq_ids_k = std::vector(packing_num, 0);
      auto end_seq_ids_k = std::vector(packing_num, 0);
      auto original_seqlens_k = std::vector(packing_num, 0);

      calculate_start_and_end_seq_ids(start_seq_ids_q, end_seq_ids_q, original_seqlens_q,
                                      cu_seqlens_q, start_seq_ids_k, end_seq_ids_k,
                                      original_seqlens_k, cu_seqlens_k, cp_rank, cp_size,
                                      split_pattern_idx);

      auto reshaped_dout = NDArray::view(dout, {batch_size_mul_seq_len, total_num_heads, _head_dim});
      auto reshaped_dqkv = NDArray::view(dqkv, {batch_size_mul_seq_len, total_num_heads, _head_dim});
      HTShape q_shape = {batch_size_mul_seq_len, q_num_heads, _head_dim};
      HTShape k_shape = {batch_size_mul_seq_len, kv_num_heads, _head_dim};
      HTShape q_begin_pos = {0, 0, 0};
      HTShape k_begin_pos = {0, q_num_heads, 0};
      auto dout_q = NDArray::slice(reshaped_dout, q_begin_pos, q_shape, stream_idx);
      auto dout_k = NDArray::slice(reshaped_dout, k_begin_pos, k_shape, stream_idx);
      auto dq = NDArray::slice(reshaped_dqkv, q_begin_pos, q_shape, stream_idx);
      auto dk = NDArray::slice(reshaped_dqkv, k_begin_pos, k_shape, stream_idx);

      auto cu_seqlens_q_slice = NDArray::slice(cu_seqlens_q, {cp_rank, 0}, {1, packing_num+1}, stream_idx);
      auto cu_seqlens_k_slice = NDArray::slice(cu_seqlens_k, {cp_rank, 0}, {1, packing_num+1}, stream_idx);

      HT_DISPATCH_KERNEL_CPU_AND_CUDA(op->instantiation_ctx().placement.type(), type(), hetu::impl::RotaryVarlenGradient,
                                   dout_q, cos, sin, dq, cu_seqlens_q_slice, max_seqlen_q(), start_seq_ids_q.data(),
                                   end_seq_ids_q.data(), split_pattern_idx, op->instantiation_ctx().stream());
      HT_DISPATCH_KERNEL_CPU_AND_CUDA(op->instantiation_ctx().placement.type(), type(), hetu::impl::RotaryVarlenGradient,
                                   dout_k, cos, sin, dk, cu_seqlens_k_slice, max_seqlen_k(), start_seq_ids_k.data(),
                                   end_seq_ids_k.data(), split_pattern_idx, op->instantiation_ctx().stream());
    }
  }
  else {
    // 不开cp
    if(!_packing) {
      HT_DISPATCH_KERNEL_CPU_AND_CUDA(op->instantiation_ctx().placement.type(), type(), hetu::impl::RotaryGradient,
                                   dout_q, cos, sin, dq, 0, 0, 0, op->instantiation_ctx().stream());
      HT_DISPATCH_KERNEL_CPU_AND_CUDA(op->instantiation_ctx().placement.type(), type(), hetu::impl::RotaryGradient,
                                   dout_k, cos, sin, dk, 0, 0, 0, op->instantiation_ctx().stream());
    }
    else {
      HT_ASSERT(inputs.size() == 5)
        << "packing should have 3 inputs: qkv, cu_seqlens_q and cu_seqlens_k";
      auto cu_seqlens_q = inputs.at(3);
      auto cu_seqlens_k = inputs.at(4);

      auto reshaped_dout = NDArray::view(dout, {batch_size_mul_seq_len, total_num_heads, _head_dim});
      auto reshaped_dqkv = NDArray::view(dqkv, {batch_size_mul_seq_len, total_num_heads, _head_dim});
      HTShape q_shape = {batch_size_mul_seq_len, q_num_heads, _head_dim};
      HTShape k_shape = {batch_size_mul_seq_len, kv_num_heads, _head_dim};
      HTShape q_begin_pos = {0, 0, 0};
      HTShape k_begin_pos = {0, q_num_heads, 0};
      auto dout_q = NDArray::slice(reshaped_dout, q_begin_pos, q_shape, stream_idx);
      auto dout_k = NDArray::slice(reshaped_dout, k_begin_pos, k_shape, stream_idx);
      auto dq = NDArray::slice(reshaped_dqkv, q_begin_pos, q_shape, stream_idx);
      auto dk = NDArray::slice(reshaped_dqkv, k_begin_pos, k_shape, stream_idx);

      int packing_num = cu_seqlens_q->shape(-1) - 1;
      auto start_seq_ids = std::vector(packing_num, 0);

      HT_DISPATCH_KERNEL_CPU_AND_CUDA(op->instantiation_ctx().placement.type(), type(), hetu::impl::RotaryVarlenGradient,
                                   dout_q, cos, sin, dq, cu_seqlens_q, max_seqlen_q(), start_seq_ids.data(),
                                   nullptr, 0, op->instantiation_ctx().stream());
      HT_DISPATCH_KERNEL_CPU_AND_CUDA(op->instantiation_ctx().placement.type(), type(), hetu::impl::RotaryVarlenGradient,
                                   dout_k, cos, sin, dk, cu_seqlens_k, max_seqlen_k(), start_seq_ids.data(),
                                   nullptr, 0, op->instantiation_ctx().stream());
    }
  }

  if(!inplace()) {
    // copy v
    NDArray::copy(dout_v, op->instantiation_ctx().stream_index, dv);
  }
}

HTShapeList RotaryGradientOpImpl::DoInferShape(Operator& op, 
                                           const HTShapeList& input_shapes, 
                                           RuntimeContext& ctx) const {
  return {input_shapes.at(0)};
}

void RotaryGradientOpImpl::DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                                          const OpMeta& op_meta, const InstantiationContext& inst_ctx) const {
  return outputs.at(0)->set_distributed_states(inputs.at(0)->get_distributed_states());
}

void RotaryGradientOpImpl::DoDeduceHeterProp(const std::vector<int32_t>& inputs_hetero_dim,
                                             TensorList& outputs, const OpMeta& op_meta, 
                                             const InstantiationContext& inst_ctx) const {
  outputs.at(0)->cur_ds_union().set_hetero_dim(inputs_hetero_dim.at(0));
}

Tensor MakeRotaryOp(Tensor qkv, Tensor cos, Tensor sin, int64_t head_dim, int64_t group_query_ratio,
        SyShapeList multi_seq_lens_symbol, SyShapeList multi_cp_group_symbol, 
        bool packing, Tensor cu_seqlens_q, Tensor cu_seqlens_k, IntSymbol max_seqlen_q, IntSymbol max_seqlen_k,
        bool interleaved, bool inplace, OpMeta op_meta) {
  TensorList inputs = {std::move(qkv), std::move(cos), std::move(sin)};
  if(packing) {
    inputs.emplace_back(std::move(cu_seqlens_q));
    inputs.emplace_back(std::move(cu_seqlens_k));
  }
  return Graph::MakeOp(
    std::make_shared<RotaryOpImpl>(head_dim, group_query_ratio, 
        std::move(multi_seq_lens_symbol), std::move(multi_cp_group_symbol),
        packing, max_seqlen_q, max_seqlen_k, interleaved, inplace),
    std::move(inputs),
    std::move(op_meta)
  )->output(0);
}

Tensor MakeRotaryGradientOp(Tensor dout, Tensor cos, Tensor sin, int64_t head_dim, int64_t group_query_ratio,
        SyShapeList multi_seq_lens_symbol, SyShapeList multi_cp_group_symbol, 
        bool packing, Tensor cu_seqlens_q, Tensor cu_seqlens_k, IntSymbol max_seqlen_q, IntSymbol max_seqlen_k,
        bool interleaved, bool inplace, OpMeta op_meta) {
  TensorList inputs = {std::move(dout), std::move(cos), std::move(sin)};
  if (packing) {
    inputs.emplace_back(std::move(cu_seqlens_q));
    inputs.emplace_back(std::move(cu_seqlens_k));
  }
  return Graph::MakeOp(
    std::make_shared<RotaryGradientOpImpl>(head_dim, group_query_ratio, 
        std::move(multi_seq_lens_symbol), std::move(multi_cp_group_symbol),
        packing, max_seqlen_q, max_seqlen_k, interleaved, inplace),
    std::move(inputs),
    std::move(op_meta)
  )->output(0);
}

} // namespace graph
} // namespace hetu
