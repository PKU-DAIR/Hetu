#include "cutlass/numeric_types.h"
#include "flash_attn/flash.h"
#include "flash_attn/static_switch.h"
#include "hetu/core/ndarray.h"
#include "hetu/core/memory_pool.h"
#include "hetu/impl/stream/CUDAStream.h"
#include "hetu/impl/random/CPURandomState.h"
#include "hetu/impl/random/CUDARandomState.h"
#include "hetu/impl/cuda/CUDADnn.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/cuda_utils.h"
#include "hetu/impl/utils/cuda_math.h"

namespace hetu {
namespace impl {

void set_params_fprop(Flash_fwd_params& params,
                      // sizes
                      const size_t b, const size_t seqlen_q,
                      const size_t seqlen_k, const size_t seqlen_q_rounded,
                      const size_t seqlen_k_rounded, const size_t h,
                      const size_t h_k, const size_t d, const size_t d_rounded,
                      // device pointers
                      const NDArray& q, const NDArray& k, const NDArray& v,
                      NDArray& out, void* cu_seqlens_q_d, void* cu_seqlens_k_d,
                      void* p_d, void* softmax_lse_d, float p_dropout,
                      float softmax_scale, bool is_causal,
                      bool seqlenq_ngroups_swapped=false, const bool unpadded_lse=false) {
  // Reset the parameters
  memset(&params, 0, sizeof(params));

  params.is_bf16 = q->dtype() == kBFloat16;

  // Set the pointers and strides.
  params.q_ptr = q->raw_data_ptr();
  params.k_ptr = k->raw_data_ptr();
  params.v_ptr = v->raw_data_ptr();
  // All stride are in elements, not bytes.
  params.q_row_stride = q->stride(-3);
  params.k_row_stride = k->stride(-3);
  params.v_row_stride = v->stride(-3);
  params.q_head_stride = q->stride(-2);
  params.k_head_stride = k->stride(-2);
  params.v_head_stride = v->stride(-2);
  params.o_ptr = out->raw_data_ptr();
  params.o_row_stride = out->stride(-3);
  params.o_head_stride = out->stride(-2);

  if (cu_seqlens_q_d == nullptr) {
    params.q_batch_stride = q->stride(0);
    params.k_batch_stride = k->stride(0);
    params.v_batch_stride = v->stride(0);
    params.o_batch_stride = out->stride(0);
  }

  params.cu_seqlens_q = static_cast<int*>(cu_seqlens_q_d);
  params.cu_seqlens_k = static_cast<int*>(cu_seqlens_k_d);

  // P = softmax(QK^T)
  params.p_ptr = p_d;

  // Softmax sum
  params.softmax_lse_ptr = softmax_lse_d;

  // Set the dimensions.
  params.b = b;
  params.h = h;
  params.h_k = h_k;
  params.h_h_k_ratio = h / h_k;
  params.seqlen_q = seqlen_q;
  params.seqlen_k = seqlen_k;
  params.seqlen_q_rounded = seqlen_q_rounded;
  params.seqlen_k_rounded = seqlen_k_rounded;
  params.d = d;
  params.d_rounded = d_rounded;

  // Set the different scale values.
  params.scale_softmax = softmax_scale;
  params.scale_softmax_log2 = softmax_scale * M_LOG2E;

  // Set this to probability of keeping an element to simplify things.
  params.p_dropout = 1.f - p_dropout;
  // Convert p from float to int so we don't have to convert the random uint to
  // float to compare. [Minor] We want to round down since when we do the
  // comparison we use <= instead of < params.p_dropout_in_uint =
  // uint32_t(std::floor(params.p_dropout * 4294967295.0));
  // params.p_dropout_in_uint16_t = uint16_t(std::floor(params.p_dropout *
  // 65535.0));
  params.p_dropout_in_uint8_t = uint8_t(std::floor(params.p_dropout * 255.0));
  params.rp_dropout = 1.f / params.p_dropout;
  params.scale_softmax_rp_dropout = params.rp_dropout * params.scale_softmax;
  HT_ASSERT(p_dropout < 1.f) << "p_dropout > 1.";

  params.is_causal = is_causal;
  params.is_seqlens_k_cumulative = true;

  params.unpadded_lse = unpadded_lse;
  params.seqlenq_ngroups_swapped = seqlenq_ngroups_swapped;
}

void set_params_dgrad(Flash_bwd_params& params,
                      // sizes
                      const size_t b, const size_t seqlen_q,
                      const size_t seqlen_k, const size_t seqlen_q_rounded,
                      const size_t seqlen_k_rounded, const size_t h,
                      const size_t h_k, const size_t d, const size_t d_rounded,
                      // device pointers
                      const NDArray& q, const NDArray& k, const NDArray& v,
                      NDArray& out, NDArray& dout, NDArray& dq, NDArray& dk,
                      NDArray& dv, void* cu_seqlens_q_d, void* cu_seqlens_k_d,
                      void* dq_accum_d, void* dk_accum_d, void* dv_accum_d,
                      void* softmax_lse_d, void* dsoftmax_sum_d,
                      float p_dropout, float softmax_scale, bool is_causal,
                      const bool unpadded_lse=false) {
  set_params_fprop(params, b, seqlen_q, seqlen_k, seqlen_q_rounded,
                   seqlen_k_rounded, h, h_k, d, d_rounded, q, k, v, out,
                   cu_seqlens_q_d, cu_seqlens_k_d, nullptr, softmax_lse_d,
                   p_dropout, softmax_scale, is_causal, 
                   false, unpadded_lse);

  // Set the pointers and strides.
  params.do_ptr = dout->raw_data_ptr();
  params.do_row_stride = dout->stride(-3);
  params.do_head_stride = dout->stride(-2);
  params.dq_ptr = dq->raw_data_ptr();
  params.dk_ptr = dk->raw_data_ptr();
  params.dv_ptr = dv->raw_data_ptr();
  params.dq_row_stride = dq->stride(-3);
  params.dk_row_stride = dk->stride(-3);
  params.dv_row_stride = dv->stride(-3);
  params.dq_head_stride = dq->stride(-2);
  params.dk_head_stride = dk->stride(-2);
  params.dv_head_stride = dv->stride(-2);

  if (cu_seqlens_q_d == nullptr) {
    params.do_batch_stride = dout->stride(0);
    params.dq_batch_stride = dq->stride(0);
    params.dk_batch_stride = dk->stride(0);
    params.dv_batch_stride = dv->stride(0);
  }

  params.dq_accum_ptr = dq_accum_d;
  params.dk_accum_ptr = dk_accum_d;
  params.dv_accum_ptr = dv_accum_d;

  // Softmax sum
  params.dsoftmax_sum = dsoftmax_sum_d;
}

void run_mha_fwd(Flash_fwd_params& params, cudaStream_t stream,
                 bool force_split_kernel = false) {
  FP16_SWITCH(!params.is_bf16, [&] {
    FWD_HEADDIM_SWITCH(params.d, [&] {
      if (params.num_splits <= 1 &&
          !force_split_kernel) { // If we don't set it num_splits == 0
        run_mha_fwd_<elem_type, kHeadDim>(params, stream);
      } else {
        run_mha_fwd_splitkv_dispatch<elem_type, kHeadDim>(params, stream);
      }
    });
  });
}

// Find the number of splits that maximizes the occupancy. For example, if we
// have batch * n_heads = 48 and we have 108 SMs, having 2 splits (efficiency =
// 0.89) is better than having 3 splits (efficiency = 0.67). However, we also
// don't want too many splits as that would incur more HBM reads/writes. So we
// find the best efficiency, then find the smallest number of splits that gets
// 85% of the best efficiency.
inline int num_splits_heuristic(int batch_nheads_mblocks, int num_SMs,
                                int num_n_blocks, int max_splits) {
  // If we have enough to almost fill the SMs, then just use 1 split
  if (batch_nheads_mblocks >= 0.8f * num_SMs) {
    return 1;
  }
  max_splits = std::min({max_splits, num_SMs, num_n_blocks});
  float max_efficiency = 0.f;
  std::vector<float> efficiency;
  efficiency.reserve(max_splits);
  auto ceildiv = [](int a, int b) { return (a + b - 1) / b; };
  // Some splits are not eligible. For example, if we have 64 blocks and choose
  // 11 splits, we'll have 6 * 10 + 4 blocks. If we choose 12 splits, we'll have
  // 6 * 11 + (-2) blocks (i.e. it's 11 splits anyway). So we check if the
  // number of blocks per split is the same as the previous num_splits.
  auto is_split_eligible = [&ceildiv, &num_n_blocks](int num_splits) {
    return num_splits == 1 ||
      ceildiv(num_n_blocks, num_splits) !=
      ceildiv(num_n_blocks, num_splits - 1);
  };
  for (int num_splits = 1; num_splits <= max_splits; num_splits++) {
    if (!is_split_eligible(num_splits)) {
      efficiency.push_back(0.f);
    } else {
      float n_waves = float(batch_nheads_mblocks * num_splits) / num_SMs;
      float eff = n_waves / ceil(n_waves);
      // printf("num_splits = %d, eff = %f\n", num_splits, eff);
      if (eff > max_efficiency) {
        max_efficiency = eff;
      }
      efficiency.push_back(eff);
    }
  }
  for (int num_splits = 1; num_splits <= max_splits; num_splits++) {
    if (!is_split_eligible(num_splits)) {
      continue;
    }
    if (efficiency[num_splits - 1] >= 0.85 * max_efficiency) {
      // printf("num_splits chosen = %d\n", num_splits);
      return num_splits;
    }
  }
  return 1;
}

void FlashAttnCuda(
  const NDArray& q, // batch_size x seqlen_q x num_heads x head_size
  const NDArray& k, // batch_size x seqlen_k x num_heads_k x head_size
  const NDArray& v, // batch_size x seqlen_k x num_heads_k x head_size
  NDArray& out_, // batch_size x seqlen_q x num_heads x head_size
  NDArray& q_padded, // batch_size x seqlen_q x num_heads x head_size_rounded
  NDArray& k_padded, // batch_size x seqlen_k x num_heads_k x head_size_rounded
  NDArray& v_padded, // batch_size x seqlen_k x num_heads_k x head_size_rounded
  NDArray& out_padded, // batch_size x seqlen_q x num_heads x head_size_rounded
  NDArray& softmax_lse, // batch_size × num_heads × seqlen_q
  NDArray& p, // batch_size × num_heads × seqlen_q_rounded × seqlen_k_rounded
  NDArray& rng_state, // 2  kCUDA  kInt64
  const float p_dropout, const float softmax_scale, const bool is_causal,
  const bool return_softmax, const Stream& stream) {
  CUDAStream cuda_stream(stream);
  hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
  auto dprops = Device::dprop(q->device().index());
  // cudaGetDeviceProperties(&dprops, q->device().index());
  // bool is_sm75 = dprops.major == 7 && dprops.minor == 5;
  bool is_sm8x = dprops.major == 8 && dprops.minor >= 0;
  bool is_sm90 = dprops.major == 9 && dprops.minor == 0;
  HT_ASSERT(is_sm90 || is_sm8x)
    << "FlashAttention only supports Ampere GPUs or newer.";
  // We will support Turing in the near future
  // HT_ASSERT(is_sm90 || is_sm8x || is_sm75, "FlashAttention only supports
  // Turing GPUs or newer.");

  auto q_dtype = q->dtype();
  HT_ASSERT(q_dtype == kFloat16 || q_dtype == kBFloat16)
    << "FlashAttention only support fp16 and bf16 data type";
  if (q_dtype == kBFloat16) {
    HT_ASSERT(is_sm90 || is_sm8x)
      << "bfloat16 is only supported on Ampere GPUs or newer";
  }
  HT_ASSERT(k->dtype() == q_dtype) << "query and key must have the same dtype";
  HT_ASSERT(v->dtype() == q_dtype)
    << "query and value must have the same dtype";

  HT_ASSERT(q->device().is_cuda()) << "Input tensor must be on CUDA device";
  HT_ASSERT(k->device().is_cuda()) << "Input tensor must be on CUDA device";
  HT_ASSERT(v->device().is_cuda()) << "Input tensor must be on CUDA device";

  HT_ASSERT(q->stride(-1) == 1)
    << "Input tensor must have contiguous last dimension";
  HT_ASSERT(k->stride(-1) == 1)
    << "Input tensor must have contiguous last dimension";
  HT_ASSERT(v->stride(-1) == 1)
    << "Input tensor must have contiguous last dimension";

  const auto sizes = q->shape();

  const int batch_size = sizes[0];
  const int seqlen_q = sizes[1];
  const int num_heads = sizes[2];
  const int head_size_og = sizes[3];
  const int seqlen_k = k->shape(1);
  const int num_heads_k = k->shape(2);
  HT_ASSERT(batch_size > 0) << "batch size must be postive";
  HT_ASSERT(head_size_og <= 256)
    << "FlashAttention forward only supports head dimension at most 256";
  HT_ASSERT(num_heads % num_heads_k == 0)
    << "Number of heads in key/value must divide number of heads in query";

  if (head_size_og % 8 != 0) {
    HTShape pad_shape = {0, 8 - head_size_og % 8};
    NDArray::pad(q, pad_shape, "constant", 0, stream.stream_index(), q_padded);
    NDArray::pad(k, pad_shape, "constant", 0, stream.stream_index(), k_padded);
    NDArray::pad(v, pad_shape, "constant", 0, stream.stream_index(), v_padded);
  } else {
    q_padded = q;
    k_padded = k;
    v_padded = v;
  }

  NDArray out;
  if (out_.is_defined()) {
    out = out_;
    HT_ASSERT(out->dtype() == q_dtype)
      << "Output must have the same dtype as inputs" << out->dtype() << " "
      << q_dtype;
    HT_ASSERT(out->device().is_cuda())
      << "Output tensor must be on CUDA device";
    HT_ASSERT(out->stride(-1) == 1)
      << "Output tensor must have contiguous last dimension";
    if (head_size_og % 8 != 0) {
      out = NDArray::empty_like(q_padded, stream.stream_index());
    }
  } else {
    out = NDArray::empty_like(q_padded, stream.stream_index());
  }

  auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
  const int head_size = round_multiple(head_size_og, 8);
  const int head_size_rounded = round_multiple(head_size, 32);
  const int seqlen_q_rounded = round_multiple(seqlen_q, 128);
  const int seqlen_k_rounded = round_multiple(seqlen_k, 128);

  // Otherwise the kernel will be launched from cuda:0 device
  // Cast to char to avoid compiler warning about narrowing

  // auto opts = q.options();

  // auto softmax_lse = NDArray::empty({batch_size, num_heads, seqlen_q},
  // q->device(), kFloat); NDArray p; Only return softmax if there's dropout to
  // reduce compilation time
  if (return_softmax) {
    HT_ASSERT(p_dropout > 0.0f)
      << "return_softmax is only supported when p_dropout > 0.0";
    p = NDArray::empty(
      {batch_size, num_heads, seqlen_q_rounded, seqlen_k_rounded}, q->device(),
      q->dtype(), stream.stream_index());
  }

  Flash_fwd_params params;
  set_params_fprop(
    params, batch_size, seqlen_q, seqlen_k, seqlen_q_rounded, seqlen_k_rounded,
    num_heads, num_heads_k, head_size, head_size_rounded, q_padded, k_padded,
    v_padded, out,
    /*cu_seqlens_q_d=*/nullptr,
    /*cu_seqlens_k_d=*/nullptr, return_softmax ? p->raw_data_ptr() : nullptr,
    softmax_lse->raw_data_ptr(), p_dropout, softmax_scale, is_causal);

  // This needs to match with run_mha_fwd_splitkv_dispatch
  const int block_n = is_sm90 || is_sm8x
    ? (head_size <= 64 ? 256 : (head_size <= 160 ? 128 : 64))
    : (head_size <= 64 ? 256 : (head_size <= 128 ? 128 : 64));
  const int num_n_blocks = (seqlen_k + block_n - 1) / block_n;
  // Technically kBlockM = 64 only for the splitKV kernels, not the standard
  // kernel. In any case we don't expect seqlen_q to be larger than 64 for
  // inference.
  const int num_m_blocks = (seqlen_q + 64 - 1) / 64;
  params.num_splits = 1;
  if (p_dropout == 0.0f) { // SplitKV is not implemented for dropout
    params.num_splits =
      num_splits_heuristic(batch_size * num_heads * num_m_blocks,
                           dprops.multiProcessorCount, num_n_blocks, 128);
    if (params.num_splits > 1) {
      NDArray softmax_lse_accum =
        NDArray::empty({params.num_splits, batch_size, num_heads, seqlen_q},
                       q->device(), kFloat, stream.stream_index());
      NDArray out_accum = NDArray::empty(
        {params.num_splits, batch_size, num_heads, seqlen_q, head_size_rounded},
        q->device(), kFloat, stream.stream_index());
      params.softmax_lseaccum_ptr = softmax_lse_accum->raw_data_ptr();
      params.oaccum_ptr = out_accum->raw_data_ptr();
    }
  }

  int64_t counter_offset = params.b * params.h * 32;
  params.rng_state = reinterpret_cast<uint64_t*>(rng_state->raw_data_ptr());

  if (p_dropout > 0.0) {
    params.philox_args = std::pair<uint64_t, uint64_t>(
      hetu::impl::GenNextRandomSeed(), counter_offset);
  }

  run_mha_fwd(params, cuda_stream);

  if (head_size_og % 8 != 0) {
    HTShape pad_shape1 = {0, 8 - head_size_og % 8};
    NDArray::pad(out, pad_shape1, "constant", 0, stream.stream_index(),
                 out_padded);
  } else {
    out_padded = out;
    // NDArray::copy(out, stream.stream_index(), out_padded);
  }

  // if (head_size_og % 8 != 0) {
  //     out = out.index({"...", torch::indexing::Slice(torch::indexing::None,
  //     head_size_og)}); if (out_.is_defined()) { out_.value().copy_(out); }
  // }
  NDArray::MarkUsedBy({q, k, v, out_, q_padded, k_padded, v_padded, out_padded,
                       softmax_lse, p, rng_state},
                      stream);
}

void FlashAttnVarlenCuda(
  const NDArray& q, // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
  const NDArray& k, // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
  const NDArray& v, // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
  const NDArray& cu_seqlens_q, // b+1
  const NDArray& cu_seqlens_k, // b+1
  NDArray& out_, // total_q x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
  NDArray& q_padded, // batch_size x seqlen_q x num_heads x head_size_rounded
  NDArray& k_padded, // batch_size x seqlen_k x num_heads_k x head_size_rounded
  NDArray& v_padded, // batch_size x seqlen_k x num_heads_k x head_size_rounded
  NDArray& out_padded, // batch_size x seqlen_q x num_heads x head_size_rounded
  NDArray& softmax_lse, // num_heads × total_q
  NDArray& p, // batch_size × num_heads × seqlen_q_rounded × seqlen_k_rounded
  NDArray& rng_state, // 2  kCUDA  kInt64
  const int max_seqlen_q, const int max_seqlen_k, const float p_dropout,
  const float softmax_scale, const bool zero_tensors, const bool is_causal,
  const bool return_softmax, const Stream& stream) {
  CUDAStream cuda_stream(stream);
  hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
  auto dprops = Device::dprop(q->device().index());
  // bool is_sm75 = dprops.major == 7 && dprops.minor == 5;
  bool is_sm8x = dprops.major == 8 && dprops.minor >= 0;
  bool is_sm90 = dprops.major == 9 && dprops.minor == 0;
  HT_ASSERT(is_sm90 || is_sm8x)
    << "FlashAttention only supports Ampere GPUs or newer.";
  // We will support Turing in the near future
  // HT_ASSERT(is_sm90 || is_sm8x || is_sm75, "FlashAttention only supports
  // Turing GPUs or newer.");

  auto q_dtype = q->dtype();
  HT_ASSERT(q_dtype == kFloat16 || q_dtype == kBFloat16)
    << "FlashAttention only support fp16 and bf16 data type";
  if (q_dtype == kBFloat16) {
    HT_ASSERT(is_sm90 || is_sm8x)
      << "bfloat16 is only supported on Ampere GPUs or newer";
  }
  HT_ASSERT(k->dtype() == q_dtype) << "query and key must have the same dtype";
  HT_ASSERT(v->dtype() == q_dtype)
    << "query and value must have the same dtype";
  HT_ASSERT(cu_seqlens_q->dtype() == kInt32)
    << "cu_seqlens_q must have dtype int32";
  HT_ASSERT(cu_seqlens_k->dtype() == kInt32)
    << "cu_seqlens_k must have dtype int32";

  HT_ASSERT(q->is_cuda()) << "Input tensor must be on CUDA device";
  HT_ASSERT(k->is_cuda()) << "Input tensor must be on CUDA device";
  HT_ASSERT(v->is_cuda()) << "Input tensor must be on CUDA device";
  HT_ASSERT(cu_seqlens_q->is_cuda()) << "cu_seqlens_q must be on CUDA device";
  HT_ASSERT(cu_seqlens_k->is_cuda()) << "cu_seqlens_k must be on CUDA device";

  HT_ASSERT(q->stride(-1) == 1)
    << "Input tensor must have contiguous last dimension";
  HT_ASSERT(k->stride(-1) == 1)
    << "Input tensor must have contiguous last dimension";
  HT_ASSERT(v->stride(-1) == 1)
    << "Input tensor must have contiguous last dimension";
  HT_ASSERT(cu_seqlens_q->is_contiguous()) << "cu_seqlens_q must be contiguous";
  HT_ASSERT(cu_seqlens_k->is_contiguous()) << "cu_seqlens_k must be contiguous";

  const auto sizes = q->shape();

  const int total_q = sizes[0];
  const int batch_size = cu_seqlens_q->numel() - 1;
  const int num_heads = sizes[1];
  const int head_size_og = sizes[2];
  const int total_k = k->shape(0);
  const int num_heads_k = k->shape(1);
  HT_ASSERT(batch_size > 0) << "batch size must be positive";
  HT_ASSERT(head_size_og <= 256)
    << "FlashAttention forward only supports head dimension at most 256";
  HT_ASSERT(num_heads % num_heads_k == 0)
    << "Number of heads in key/value must divide number of heads in query";

  if (head_size_og % 8 != 0) {
    HTShape pad_shape = {0, 8 - head_size_og % 8};
    NDArray::pad(q, pad_shape, "constant", 0, stream.stream_index(), q_padded);
    NDArray::pad(k, pad_shape, "constant", 0, stream.stream_index(), k_padded);
    NDArray::pad(v, pad_shape, "constant", 0, stream.stream_index(), v_padded);
  } else {
    q_padded = q;
    k_padded = k;
    v_padded = v;
  }

  NDArray out;
  if (out_.is_defined()) {
    out = out_;
    HT_ASSERT(out->dtype() == q_dtype)
      << "Output must have the same dtype as inputs" << out->dtype() << " "
      << q_dtype;
    HT_ASSERT(out->device().is_cuda())
      << "Output tensor must be on CUDA device";
    HT_ASSERT(out->stride(-1) == 1)
      << "Output tensor must have contiguous last dimension";
    if (head_size_og % 8 != 0) {
      out = NDArray::empty_like(q_padded, stream.stream_index());
    }
  } else {
    out = NDArray::empty_like(q_padded, stream.stream_index());
  }

  auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
  const int head_size = round_multiple(head_size_og, 8);
  const int head_size_rounded = round_multiple(head_size, 32);
  const int seqlen_q_rounded = round_multiple(max_seqlen_q, 128);
  const int seqlen_k_rounded = round_multiple(max_seqlen_k, 128);

  // auto opts = q.options();

  // auto softmax_lse =
  //   torch::empty({batch_size, num_heads, max_seqlen_q},
  //   opts.dtype(at::kFloat));
  // at::Tensor p;
  // Only return softmax if there's dropout to reduce compilation time
  if (return_softmax) {
    HT_ASSERT(p_dropout > 0.0f)
      << "return_softmax is only supported when p_dropout > 0.0";
    p = NDArray::empty(
      {batch_size, num_heads, seqlen_q_rounded, seqlen_k_rounded}, q->device(),
      q->dtype(), stream.stream_index());
  }

  if (zero_tensors) {
    NDArray::zeros_(out, stream.stream_index());
    NDArray::full_(softmax_lse, -std::numeric_limits<float>::infinity(),
                   stream.stream_index());
    if (return_softmax) {
      NDArray::zeros_(p, stream.stream_index());
    }
  }

  Flash_fwd_params params;
  set_params_fprop(
    params, batch_size, max_seqlen_q, max_seqlen_k, seqlen_q_rounded,
    seqlen_k_rounded, num_heads, num_heads_k, head_size, head_size_rounded,
    q_padded, k_padded, v_padded, out, cu_seqlens_q->raw_data_ptr(),
    cu_seqlens_k->raw_data_ptr(), return_softmax ? p->raw_data_ptr() : nullptr,
    softmax_lse->raw_data_ptr(), p_dropout, softmax_scale, is_causal,
    /*seqlenq_ngroups_swapped*/false, /*unpadded_lse*/true);
  params.total_q = total_q;

  // number of times random will be generated per thread, to offset philox
  // counter in thc random state We use a custom RNG that increases the offset
  // by batch_size * nheads * 32.
  int64_t counter_offset = params.b * params.h * 32;
  params.rng_state = reinterpret_cast<uint64_t*>(rng_state->raw_data_ptr());
  // auto options =
  //   torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
  // auto rng_state = torch::empty({2}, options.dtype(torch::kInt64));
  // // Forward kernel will populate memory with the seed and offset.
  // params.rng_state = reinterpret_cast<uint64_t*>(rng_state.data_ptr());

  if (p_dropout > 0.0) {
    params.philox_args = std::pair<uint64_t, uint64_t>(
      hetu::impl::GenNextRandomSeed(), counter_offset);
  }

  run_mha_fwd(params, cuda_stream);

  if (head_size_og % 8 != 0) {
    HTShape pad_shape1 = {0, 8 - head_size_og % 8};
    NDArray::pad(out, pad_shape1, "constant", 0, stream.stream_index(),
                 out_padded);
  } else {
    out_padded = out;
    // NDArray::copy(out, stream.stream_index(), out_padded);
  }

  // if (head_size_og % 8 != 0) {
  //     out = out.index({"...", torch::indexing::Slice(torch::indexing::None,
  //     head_size_og)}); if (out_.is_defined()) { out_.value().copy_(out); }
  // }
  NDArray::MarkUsedBy({q, k, v, cu_seqlens_q, cu_seqlens_k, out_, q_padded,
                       k_padded, v_padded, out_padded, softmax_lse, p,
                       rng_state},
                      stream);
}

void run_mha_bwd(Flash_bwd_params& params, cudaStream_t stream,
                 const bool configure) {
  FP16_SWITCH(!params.is_bf16, [&] {
    if (params.d <= 32) {
      run_mha_bwd_<elem_type, 32>(params, stream, configure);
    } else if (params.d <= 64) {
      run_mha_bwd_<elem_type, 64>(params, stream, configure);
    } else if (params.d <= 96) {
      run_mha_bwd_<elem_type, 96>(params, stream, configure);
    } else if (params.d <= 128) {
      run_mha_bwd_<elem_type, 128>(params, stream, configure);
    } else if (params.d <= 160) {
      run_mha_bwd_<elem_type, 160>(params, stream, configure);
    } else if (params.d <= 192) {
      run_mha_bwd_<elem_type, 192>(params, stream, configure);
    } else if (params.d <= 224) {
      run_mha_bwd_<elem_type, 224>(params, stream, configure);
    } else if (params.d <= 256) {
      run_mha_bwd_<elem_type, 256>(params, stream, configure);
    }
  });
}

void FlashAttnGradientCuda(
  const NDArray& dout, // batch_size x seqlen_q x num_heads, x head_size_og
  const NDArray& q, // batch_size x seqlen_q x num_heads x head_size
  const NDArray& k, // batch_size x seqlen_k x num_heads_k x head_size
  const NDArray& v, // batch_size x seqlen_k x num_heads_k x head_size
  NDArray& out, // batch_size x seqlen_q x num_heads x head_size
  NDArray& softmax_lse, // b x h x seqlen_q
  NDArray& rng_state,
  NDArray& dq_, // batch_size x seqlen_q x num_heads x head_size
  NDArray& dk_, // batch_size x seqlen_k x num_heads_k x head_size
  NDArray& dv_, // batch_size x seqlen_k x num_heads_k x head_size
  const float p_dropout, // probability to drop
  const float softmax_scale, const bool is_causal, const Stream& stream) {
  auto dprops = Device::dprop(q->device().index());
  // bool is_sm75 = dprops.major == 7 && dprops.minor == 5;
  bool is_sm8x = dprops.major == 8 && dprops.minor >= 0;
  bool is_sm80 = dprops.major == 8 && dprops.minor == 0;
  bool is_sm90 = dprops.major == 9 && dprops.minor == 0;
  HT_ASSERT(is_sm90 || is_sm8x)
    << "FlashAttention only supports Ampere GPUs or newer.";
  // We will support Turing in the near future
  // HT_ASSERT(is_sm90 || is_sm8x || is_sm75, "FlashAttention only supports
  // Turing GPUs or newer.");

  bool is_dropout = p_dropout > 0.0;
  auto q_dtype = q->dtype();
  HT_ASSERT(q_dtype == kFloat16 || q_dtype == kBFloat16)
    << "FlashAttention only support fp16 and bf16 data type";
  if (q_dtype == kBFloat16) {
    HT_ASSERT(is_sm90 || is_sm8x)
      << "bfloat16 is only supported on Ampere GPUs or newer";
  }
  HT_ASSERT(k->dtype() == q_dtype) << "query and key must have the same dtype";
  HT_ASSERT(v->dtype() == q_dtype)
    << "query and value must have the same dtype";
  HT_ASSERT(out->dtype() == q_dtype)
    << "query and out must have the same dtype";
  HT_ASSERT(dout->dtype() == q_dtype)
    << "query and dout must have the same dtype";

  HT_ASSERT(q->device().is_cuda()) << "Input tensor must be on CUDA device";
  HT_ASSERT(k->device().is_cuda()) << "Input tensor must be on CUDA device";
  HT_ASSERT(v->device().is_cuda()) << "Input tensor must be on CUDA device";
  HT_ASSERT(out->device().is_cuda()) << "out tensor must be on CUDA device";
  HT_ASSERT(dout->device().is_cuda()) << "dout tensor must be on CUDA device";
  HT_ASSERT(softmax_lse->device().is_cuda())
    << "softmax_lse tensor must be on CUDA device";

  HT_ASSERT(q->stride(-1) == 1)
    << "Input tensor must have contiguous last dimension";
  HT_ASSERT(k->stride(-1) == 1)
    << "Input tensor must have contiguous last dimension";
  HT_ASSERT(v->stride(-1) == 1)
    << "Input tensor must have contiguous last dimension";
  HT_ASSERT(out->stride(-1) == 1)
    << "out tensor must have contiguous last dimension";
  HT_ASSERT(dout->stride(-1) == 1)
    << "dout tensor must have contiguous last dimension";

  const auto sizes = q->shape();

  const int batch_size = sizes[0];
  const int seqlen_q = sizes[1];
  const int num_heads = sizes[2];
  const int head_size_og = dout->shape(3);
  const int head_size = sizes[3];
  const int seqlen_k = k->shape(1);
  const int num_heads_k = k->shape(2);
  HT_ASSERT(batch_size > 0) << "batch size must be positive";
  HT_ASSERT(head_size % 8 == 0) << "head_size should be a multiple of 8";
  HT_ASSERT(head_size <= 256)
    << "FlashAttention backward only supports head dimension at most 256";
  if (head_size > 192) {
    HT_ASSERT(is_sm80 || is_sm90)
      << "FlashAttention backward for head dim > 192 requires A100/A800 or H100/H800";
  }
  HT_ASSERT(num_heads % num_heads_k == 0)
    << "Number of heads in key/value must divide number of heads in query";

  auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
  const int head_size_rounded = round_multiple(head_size, 32);
  const int seqlen_q_rounded = round_multiple(seqlen_q, 128);
  const int seqlen_k_rounded = round_multiple(seqlen_k, 128);

  HT_ASSERT(head_size == round_multiple(head_size_og, 8))
    << "head_size must be head_size_og rounded to a multiple of 8";

  CUDAStream cuda_stream(stream);
  hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());

  NDArray dq, dk, dv;
  if (dq_.is_defined()) {
    dq = dq_;
    HT_ASSERT(dq->dtype() == q_dtype) << "dq must have the same dtype as q";
    HT_ASSERT(dq->device().is_cuda()) << "dq must be on CUDA device";
    HT_ASSERT(dq->stride(-1) == 1) << "dq must have contiguous last dimension";
    // CHECK_SHAPE(dq, batch_size, seqlen_q, num_heads, head_size);
  } else {
    dq = NDArray::empty_like(q);
  }
  if (dk_.is_defined()) {
    dk = dk_;
    HT_ASSERT(dk->dtype() == q_dtype) << "dk must have the same dtype as q";
    HT_ASSERT(dk->device().is_cuda()) << "dk must be on CUDA device";
    HT_ASSERT(dk->stride(-1) == 1) << "dk must have contiguous last dimension";
    // CHECK_SHAPE(dk, batch_size, seqlen_k, num_heads_k, head_size);
  } else {
    dk = NDArray::empty_like(k, stream.stream_index());
  }
  if (dv_.is_defined()) {
    dv = dv_;
    HT_ASSERT(dv->dtype() == q_dtype) << "dv must have the same dtype as q";
    HT_ASSERT(dv->device().is_cuda()) << "dv must be on CUDA device";
    HT_ASSERT(dv->stride(-1) == 1) << "dv must have contiguous last dimension";
    // CHECK_SHAPE(dv, batch_size, seqlen_k, num_heads_k, head_size);
  } else {
    dv = NDArray::empty_like(k, stream.stream_index());
  }

  NDArray dout_padded;
  if (head_size_og % 8 != 0) {
    HTShape pad_shape = {0, 8 - head_size_og % 8};
    NDArray::pad(dout, pad_shape, "constant", 0, stream.stream_index(),
                 dout_padded);
  } else {
    dout_padded = dout;
  }

  // bool loop = seqlen_k > blocksize_c;
  // TODO: change later, for now set to true for simplicity
  bool loop = true;

  // auto opts = q.options();
  auto softmax_d = NDArray::empty({batch_size, num_heads, seqlen_q_rounded},
                                  q->device(), kFloat, stream.stream_index());
  NDArray dq_accum;
  NDArray dk_accum, dv_accum;
  if (loop) {
    dq_accum = NDArray::empty(
      {batch_size, seqlen_q_rounded, num_heads, head_size_rounded}, q->device(),
      kFloat, stream.stream_index());
    // dk_accum = NDArray::empty({batch_size, num_heads_k, seqlen_k_rounded,
    // head_size_rounded}, opts->dtype(at::kFloat)); dv_accum =
    // NDArray::empty({batch_size, num_heads_k, seqlen_k_rounded,
    // head_size_rounded}, opts->dtype(at::kFloat));
  }

  NDArray dk_expanded, dv_expanded;
  if (num_heads_k != num_heads) { // MQA / GQA
    dk_expanded =
      NDArray::empty({batch_size, seqlen_k, num_heads, head_size}, q->device(),
                     q->dtype(), stream.stream_index());
    dv_expanded =
      NDArray::empty({batch_size, seqlen_k, num_heads, head_size}, q->device(),
                     q->dtype(), stream.stream_index());
  } else {
    dk_expanded = dk;
    dv_expanded = dv;
  }

  Flash_bwd_params params;

  set_params_dgrad(
    params, batch_size, seqlen_q, seqlen_k, seqlen_q_rounded, seqlen_k_rounded,
    num_heads, num_heads_k, head_size, head_size_rounded, q, k, v, out,
    dout_padded, dq, dk_expanded, dv_expanded, nullptr, nullptr,
    loop ? dq_accum->raw_data_ptr() : nullptr,
    // loop ? dk_accum->raw_data_ptr() : nullptr,
    // loop ? dv_accum->raw_data_ptr() : nullptr,
    nullptr, nullptr, softmax_lse->raw_data_ptr(), softmax_d->raw_data_ptr(),
    p_dropout, softmax_scale, is_causal);

  auto launch = &run_mha_bwd;
  // launch(params, stream, /*configure=*/true);

  // auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
  //     gen_, at::cuda::detail::getDefaultCUDAGenerator());

  // We use a custom RNG that increases the offset by batch_size * nheads * 32.
  int64_t counter_offset = params.b * params.h * 32;

  if (rng_state.is_defined()) {
    params.rng_state = reinterpret_cast<uint64_t*>(rng_state->raw_data_ptr());
  } else if (is_dropout) {
    // See Note [Acquire lock when using random generators]
    // std::lock_guard<std::mutex> lock(gen->mutex_);
    params.philox_args = std::pair<uint64_t, uint64_t>(
      hetu::impl::GenNextRandomSeed(), counter_offset);
    params.rng_state[0] = params.philox_args.first;
    params.rng_state[1] = params.philox_args.second;
  }

  launch(params, cuda_stream, /*configure=*/false);

  // For MQA/GQA we need to sum dK and dV across the groups
  if (num_heads_k != num_heads) {
    NDArray::sum(NDArray::reshape(dk_expanded,
                                  {batch_size, seqlen_k, num_heads_k,
                                   num_heads / num_heads_k, head_size},
                                  stream.stream_index()),
                 {3}, false, stream.stream_index(), dk);
    NDArray::sum(NDArray::reshape(dv_expanded,
                                  {batch_size, seqlen_k, num_heads_k,
                                   num_heads / num_heads_k, head_size},
                                  stream.stream_index()),
                 {3}, false, stream.stream_index(), dv);
  }
  // if (head_size_og % 8 != 0) {
  //     dq = dq.index({"...", torch::indexing::Slice(torch::indexing::None,
  //     head_size_og)}); dk = dk.index({"...",
  //     torch::indexing::Slice(torch::indexing::None, head_size_og)}); dv =
  //     dv.index({"...", torch::indexing::Slice(torch::indexing::None,
  //     head_size_og)});
  // }

  NDArray::MarkUsedBy(
    {dout, q, k, v, out, softmax_lse, rng_state, dq_, dk_, dv_}, stream);
}

void FlashAttnVarlenGradientCuda(
  const NDArray& dout, // total_q x num_heads x head_size
  const NDArray& q, // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
  const NDArray& k, // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
  const NDArray& v, // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
  const NDArray& cu_seqlens_q, // b+1
  const NDArray& cu_seqlens_k, // b+1
  NDArray& out, // total_q x num_heads x head_size
  NDArray& softmax_lse, // h x total_q   softmax logsumexp
  NDArray& rng_state,
  NDArray& dq_, // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
  NDArray& dk_, // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
  NDArray& dv_, // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
  const int max_seqlen_q,
  const int max_seqlen_k, // max sequence length to choose the kernel
  const float p_dropout, // probability to drop
  const float softmax_scale, const bool zero_tensors, 
	const bool is_causal, const Stream& stream) {
   auto dprops = Device::dprop(q->device().index());
  // bool is_sm75 = dprops.major == 7 && dprops.minor == 5;
  bool is_sm8x = dprops.major == 8 && dprops.minor >= 0;
  bool is_sm80 = dprops.major == 8 && dprops.minor == 0;
  bool is_sm90 = dprops.major == 9 && dprops.minor == 0;
  HT_ASSERT(is_sm90 || is_sm8x)
    << "FlashAttention only supports Ampere GPUs or newer.";
  // We will support Turing in the near future
  // HT_ASSERT(is_sm90 || is_sm8x || is_sm75, "FlashAttention only supports
  // Turing GPUs or newer.");

  bool is_dropout = p_dropout > 0.0;

auto q_dtype = q->dtype();
  HT_ASSERT(q_dtype == kFloat16 || q_dtype == kBFloat16)
    << "FlashAttention only support fp16 and bf16 data type";
  if (q_dtype == kBFloat16) {
    HT_ASSERT(is_sm90 || is_sm8x)
      << "bfloat16 is only supported on Ampere GPUs or newer";
  }
  HT_ASSERT(k->dtype() == q_dtype) << "query and key must have the same dtype";
  HT_ASSERT(v->dtype() == q_dtype)
    << "query and value must have the same dtype";
  HT_ASSERT(out->dtype() == q_dtype)
    << "query and out must have the same dtype";
  HT_ASSERT(dout->dtype() == q_dtype)
    << "query and dout must have the same dtype";

  HT_ASSERT(q->device().is_cuda()) << "Input tensor must be on CUDA device";
  HT_ASSERT(k->device().is_cuda()) << "Input tensor must be on CUDA device";
  HT_ASSERT(v->device().is_cuda()) << "Input tensor must be on CUDA device";
  HT_ASSERT(out->device().is_cuda()) << "out tensor must be on CUDA device";
  HT_ASSERT(dout->device().is_cuda()) << "dout tensor must be on CUDA device";
  HT_ASSERT(softmax_lse->device().is_cuda())
    << "softmax_lse tensor must be on CUDA device";
  HT_ASSERT(cu_seqlens_q->is_cuda())
	<< "cu_seqlens_q must be on CUDA device";
  HT_ASSERT(cu_seqlens_k->is_cuda())
	<< "cu_seqlens_k must be on CUDA device";

  HT_ASSERT(q->stride(-1) == 1)
    << "Input tensor must have contiguous last dimension";
  HT_ASSERT(k->stride(-1) == 1)
    << "Input tensor must have contiguous last dimension";
  HT_ASSERT(v->stride(-1) == 1)
    << "Input tensor must have contiguous last dimension";
  HT_ASSERT(out->stride(-1) == 1)
    << "out tensor must have contiguous last dimension";
  HT_ASSERT(dout->stride(-1) == 1)
    << "dout tensor must have contiguous last dimension";
  HT_ASSERT(cu_seqlens_q->is_contiguous())
		<< "cu_seqlens_q must be contiguous";
  HT_ASSERT(cu_seqlens_k->is_contiguous())
	  << "cu_seqlens_k must be contiguous";

  const auto sizes = q->shape();

  const int total_q = sizes[0];
  const int batch_size = cu_seqlens_q->numel() - 1;
  const int num_heads = sizes[1];
  const int head_size_og = dout->shape(2);
  const int head_size = sizes[2];
  const int total_k = k->shape(0);
  const int num_heads_k = k->shape(1);
  HT_ASSERT(batch_size > 0)
		<< "batch size must be positive";
  HT_ASSERT(head_size % 8 == 0)
		<< "head_size should be a multiple of 8";
  HT_ASSERT(head_size <= 256)
		<< "FlashAttention backward only supports head dimension at most 256";
  if (head_size > 192) {
    HT_ASSERT(is_sm80 || is_sm90)
			<< "FlashAttention backward for head dim > 192 requires A100/A800 or H100/H800";
  }
  HT_ASSERT(num_heads % num_heads_k == 0)
    << "Number of heads in key/value must divide number of heads in query";

  auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
  const int head_size_rounded = round_multiple(head_size, 32);
  const int seqlen_q_rounded = round_multiple(max_seqlen_q, 128);
  const int seqlen_k_rounded = round_multiple(max_seqlen_k, 128);

  HT_ASSERT(head_size == round_multiple(head_size_og, 8))
    << "head_size must be head_size_og rounded to a multiple of 8"
    << ", but found head_size = " << head_size << " and head_size_og = " << head_size_og;

  CUDAStream cuda_stream(stream);
  hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());

	NDArray dq, dk, dv;
  if (dq_.is_defined()) {
    dq = dq_;
    HT_ASSERT(dq->dtype() == q_dtype) << "dq must have the same dtype as q";
    HT_ASSERT(dq->device().is_cuda()) << "dq must be on CUDA device";
    HT_ASSERT(dq->stride(-1) == 1) << "dq must have contiguous last dimension";
  } else {
    dq = NDArray::empty_like(q);
  }
  if (dk_.is_defined()) {
    dk = dk_;
    HT_ASSERT(dk->dtype() == q_dtype) << "dk must have the same dtype as q";
    HT_ASSERT(dk->device().is_cuda()) << "dk must be on CUDA device";
    HT_ASSERT(dk->stride(-1) == 1) << "dk must have contiguous last dimension";
  } else {
    dk = NDArray::empty_like(k, stream.stream_index());
  }
  if (dv_.is_defined()) {
    dv = dv_;
    HT_ASSERT(dv->dtype() == q_dtype) << "dv must have the same dtype as q";
    HT_ASSERT(dv->device().is_cuda()) << "dv must be on CUDA device";
    HT_ASSERT(dv->stride(-1) == 1) << "dv must have contiguous last dimension";
  } else {
    dv = NDArray::empty_like(k, stream.stream_index());
  }

  NDArray dout_padded;
  if (head_size_og % 8 != 0) {
    HTShape pad_shape = {0, 8 - head_size_og % 8};
    NDArray::pad(dout, pad_shape, "constant", 0, stream.stream_index(),
                 dout_padded);
  } else {
    dout_padded = dout;
  }

  // bool loop = max_seqlen_k > blocksize_c;
  // TODO: change later, for now set to true for simplicity
  bool loop = true;

  /*
	auto softmax_d = NDArray::empty({batch_size, num_heads, seqlen_q_rounded},
                                  q->device(), kFloat, stream.stream_index());
  */
  // softmax_d is also stored in unpadded format.
  auto softmax_d = NDArray::empty({num_heads, total_q + 128 * batch_size}, q->device(), kFloat, stream.stream_index());
  NDArray dq_accum;
  if (loop) {
    /*
    dq_accum = NDArray::empty(
      {batch_size, num_heads, seqlen_q_rounded, head_size_rounded}, q->device(),
      kFloat, stream.stream_index());
    */
    // We don't want to allocate dq_accum of size (batch, seqlen_q_rounded, num_heads, head_size_rounded)
    // because that would be too large if there is a very long sequence and the rest of the sequences are short.
    // Instead, we allocate dq_accum of size (total_q + 128 * batch, num_heads, head_size_rounded).
    // Note that 128 is the max block size on the seqlen_q dimension.
    // For dQ, the i-th sequence is stored in indices from cu_seqlens[i] + 128 * i to
    // cu_seqlens[i + 1] * 128 * i - 1. This ensures that the i-th sequence and (i + 1)-th sequence will
    // be at least 128 apart. It's ok for us to do atomicAdds up to 128 rows beyond what we're normally
    // allowed to do. So we won't have to do any bound checking, and performance should stay the same.
    dq_accum = NDArray::empty(
      {total_q + 128 * batch_size, num_heads, head_size_rounded}, q->device(),
      kFloat, stream.stream_index());
  }

  NDArray dk_expanded, dv_expanded;
  if (num_heads_k != num_heads) { // MQA / GQA
    dk_expanded =
      NDArray::empty({total_k, num_heads, head_size}, q->device(),
                     q->dtype(), stream.stream_index());
    dv_expanded =
      NDArray::empty({total_k, num_heads, head_size}, q->device(),
                     q->dtype(), stream.stream_index());
  } else {
    dk_expanded = dk;
    dv_expanded = dv;
  }

  if (zero_tensors) {
    NDArray::zeros_(dq, stream.stream_index());
		NDArray::zeros_(dk_expanded, stream.stream_index());
		NDArray::zeros_(dv_expanded, stream.stream_index());  
		NDArray::zeros_(softmax_d, stream.stream_index());
  }

  Flash_bwd_params params;

  set_params_dgrad(params, batch_size, max_seqlen_q, max_seqlen_k,
                   seqlen_q_rounded, seqlen_k_rounded, num_heads, num_heads_k,
                   head_size, head_size_rounded, q, k, v, out, dout_padded, dq,
                   dk_expanded, dv_expanded, cu_seqlens_q->raw_data_ptr(),
                   cu_seqlens_k->raw_data_ptr(),
                   loop ? dq_accum->raw_data_ptr() : nullptr, nullptr, nullptr,
                   softmax_lse->raw_data_ptr(), softmax_d->raw_data_ptr(), p_dropout,
                   softmax_scale, is_causal,
                   /*unpadded_lse*/true);
  params.total_q = total_q;

  auto launch = &run_mha_bwd;

  // We use a custom RNG that increases the offset by batch_size * nheads * 32.
  int64_t counter_offset = params.b * params.h * 32;

  if (rng_state.is_defined()) {
    params.rng_state = reinterpret_cast<uint64_t*>(rng_state->raw_data_ptr());
  } else if (is_dropout) {
    // See Note [Acquire lock when using random generators]
    // std::lock_guard<std::mutex> lock(gen->mutex_);
    params.philox_args = std::pair<uint64_t, uint64_t>(
      hetu::impl::GenNextRandomSeed(), counter_offset);
    params.rng_state[0] = params.philox_args.first;
    params.rng_state[1] = params.philox_args.second;
  }

  launch(params, cuda_stream, /*configure=*/false);

  // For MQA/GQA we need to sum dK and dV across the groups
  if (num_heads_k != num_heads) {
    NDArray::sum(NDArray::reshape(dk_expanded,
                                  {total_k, num_heads_k, num_heads / num_heads_k, head_size},
                                  stream.stream_index()),
                 {2}, false, stream.stream_index(), dk);
    NDArray::sum(NDArray::reshape(dv_expanded,
                                  {total_k, num_heads_k, num_heads / num_heads_k, head_size},
                                  stream.stream_index()),
                 {2}, false, stream.stream_index(), dv);
  }
  // if (head_size_og % 8 != 0) {
  //     dq = dq.index({"...", torch::indexing::Slice(torch::indexing::None,
  //     head_size_og)}); dk = dk.index({"...",
  //     torch::indexing::Slice(torch::indexing::None, head_size_og)}); dv =
  //     dv.index({"...", torch::indexing::Slice(torch::indexing::None,
  //     head_size_og)});
  // }

  NDArray::MarkUsedBy(
    {dout, q, k, v, cu_seqlens_q, cu_seqlens_k,
		 out, softmax_lse, rng_state, dq_, dk_, dv_}, stream);
}

} // namespace impl
} // namespace hetu
