#include "hetu/core/ndarray.h"
#include "hetu/impl/stream/CPUStream.h"
#include "hetu/impl/utils/common_utils.h"

namespace hetu {
namespace impl {

template <typename spec_t>
void rotary_block_forward_cpu(const spec_t* input, const spec_t* cos,
                              const spec_t* sin, spec_t* out, const int s_id,
                              const int offset_block,
                              const int offset_block_dst, const int num_heads,
                              const int head_dim, const int freqs_dim,
                              const int stride_h, const int stride_d,
                              const int o_stride_h, const int o_stride_d) {
#pragma unroll
  for (int d_id = 0; d_id < freqs_dim && d_id < head_dim; d_id += 1) {
    spec_t v_cos = cos[s_id * freqs_dim + d_id];
    spec_t v_sin = sin[s_id * freqs_dim + d_id];
#pragma unroll
    for (int h_id = 0; h_id < num_heads; h_id += 1) {
      int offset_src = offset_block + h_id * stride_h + d_id * stride_d;
      int offset_dst = offset_block_dst + h_id * o_stride_h + d_id * o_stride_d;
      spec_t v_src = input[offset_src];
      spec_t v_src_rotate = (d_id + freqs_dim / 2 < freqs_dim)
        ? -input[offset_src + (freqs_dim / 2) * stride_d]
        : input[offset_src + (freqs_dim / 2 - freqs_dim) * stride_d];
      out[offset_dst] = v_src * v_cos + v_src_rotate * v_sin;
    }
  }

  // 处理head_dim超出freqs_dim的部分——直接复制过去
  if (head_dim > freqs_dim) {
#pragma unroll
    for (int h_id = 0; h_id < num_heads; h_id += 1) {
      int offset_head = offset_block + h_id * stride_h;
      int offset_head_dst = offset_block_dst + h_id * o_stride_h;
#pragma unroll
      for (int d_id = freqs_dim; d_id < head_dim; d_id += 1) {
        out[offset_head_dst + d_id * o_stride_d] =
          input[offset_head + d_id * stride_d];
      }
    }
  }
}

template <typename spec_t>
void rotary_block_backward_cpu(const spec_t* input, const spec_t* cos,
                               const spec_t* sin, spec_t* out, const int s_id,
                               const int offset_block,
                               const int offset_block_dst, const int num_heads,
                               const int head_dim, const int freqs_dim,
                               const int stride_h, const int stride_d,
                               const int o_stride_h, const int o_stride_d) {
#pragma unroll
  for (int d_id = 0; d_id < freqs_dim && d_id < head_dim; d_id += 1) {
    spec_t v_cos = cos[s_id * freqs_dim + d_id];
    spec_t v_sin = (d_id + freqs_dim / 2 < freqs_dim)
      ? sin[s_id * freqs_dim + d_id + freqs_dim / 2]
      : sin[s_id * freqs_dim + d_id + freqs_dim / 2 - freqs_dim];
#pragma unroll
    for (int h_id = 0; h_id < num_heads; h_id += 1) {
      int offset_src = offset_block + h_id * stride_h + d_id * stride_d;
      int offset_dst = offset_block_dst + h_id * o_stride_h + d_id * o_stride_d;
      spec_t v_src = input[offset_src];
      spec_t v_src_rotate = (d_id + freqs_dim / 2 < freqs_dim)
        ? input[offset_src + (freqs_dim / 2) * stride_d]
        : -input[offset_src + (freqs_dim / 2 - freqs_dim) * stride_d];
      out[offset_dst] = v_src * v_cos + v_src_rotate * v_sin;
    }
  }

  // 处理head_dim超出freqs_dim的部分——直接复制过去
  if (head_dim > freqs_dim) {
#pragma unroll
    for (int h_id = 0; h_id < num_heads; h_id += 1) {
      int offset_head = offset_block + h_id * stride_h;
      int offset_head_dst = offset_block_dst + h_id * o_stride_h;
#pragma unroll
      for (int d_id = freqs_dim; d_id < head_dim; d_id += 1) {
        out[offset_head_dst + d_id * o_stride_d] =
          input[offset_head + d_id * stride_d];
      }
    }
  }
}

template <typename spec_t>
void rotary_forward_cpu(const spec_t* input, const spec_t* cos,
                        const spec_t* sin, spec_t* out, const int start_seq_id,
                        const int end_seq_id, const int split_pattern,
                        const int batch_size, const int seq_len,
                        const int num_heads, const int head_dim,
                        const int freqs_dim, const int stride_b,
                        const int stride_s, const int stride_h,
                        const int stride_d, const int o_stride_b,
                        const int o_stride_s, const int o_stride_h,
                        const int o_stride_d) {
#pragma unroll
  for (int b_id = 0; b_id < batch_size; ++b_id) {
#pragma unroll
    for (int s_id = 0; s_id < seq_len; ++s_id) {
      int offset_block = s_id * stride_s + b_id * stride_b;
      int offset_block_dst = s_id * o_stride_s + b_id * o_stride_b;
      int s_id_for_freqs = s_id;
      if (split_pattern == 0) {
        s_id_for_freqs = s_id + start_seq_id;
      } else if (split_pattern == 1) {
        if (s_id < seq_len / 2) {
          s_id_for_freqs = s_id + start_seq_id;
        } else {
          s_id_for_freqs = end_seq_id - (seq_len - s_id);
        }
      }
      rotary_block_forward_cpu(input, cos, sin, out, s_id_for_freqs,
                               offset_block, offset_block_dst, num_heads,
                               head_dim, freqs_dim, stride_h, stride_d,
                               o_stride_h, o_stride_d);
    }
  }
}

template <typename spec_t>
void rotary_backward_cpu(const spec_t* input, const spec_t* cos,
                         const spec_t* sin, spec_t* out, const int start_seq_id,
                         const int end_seq_id, const int split_pattern,
                         const int batch_size, const int seq_len,
                         const int num_heads, const int head_dim,
                         const int freqs_dim, const int stride_b,
                         const int stride_s, const int stride_h,
                         const int stride_d, const int o_stride_b,
                         const int o_stride_s, const int o_stride_h,
                         const int o_stride_d) {
#pragma unroll
  for (int b_id = 0; b_id < batch_size; ++b_id) {
#pragma unroll
    for (int s_id = 0; s_id < seq_len; ++s_id) {
      int offset_block = s_id * stride_s + b_id * stride_b;
      int offset_block_dst = s_id * o_stride_s + b_id * o_stride_b;
      int s_id_for_freqs = s_id;
      if (split_pattern == 0) {
        s_id_for_freqs = s_id + start_seq_id;
      } else if (split_pattern == 1) {
        if (s_id < seq_len / 2) {
          s_id_for_freqs = s_id + start_seq_id;
        } else {
          s_id_for_freqs = end_seq_id - (seq_len - s_id);
        }
      }
      rotary_block_backward_cpu(input, cos, sin, out, s_id_for_freqs,
                                offset_block, offset_block_dst, num_heads,
                                head_dim, freqs_dim, stride_h, stride_d,
                                o_stride_h, o_stride_d);
    }
  }
}

template <typename spec_t>
void rotary_varlen_forward_cpu(
  const spec_t* input, const spec_t* cos, const spec_t* sin, spec_t* out,
  const int* cu_seqlens, const int max_seqlen, const int* start_seq_ids,
  const int* end_seq_ids, const int split_pattern, const int batch_size,
  const int num_heads, const int head_dim, const int freqs_dim,
  const int stride_t, const int stride_h, const int stride_d,
  const int o_stride_t, const int o_stride_h, const int o_stride_d) {
#pragma unroll
  for (int b_id = 0; b_id < batch_size; ++b_id) {
    int start = cu_seqlens[b_id];
    int end = cu_seqlens[b_id + 1];
#pragma unroll
    for (int s_id = 0; s_id + start < end; ++s_id) {
      int t_id = start + s_id;
      if (t_id >= end)
        return;
      int offset_block = t_id * stride_t;
      int offset_block_dst = t_id * o_stride_t;

      int s_id_for_freqs = s_id;
      if (split_pattern == 0) {
        s_id_for_freqs = s_id + start_seq_ids[b_id];
      } else if (split_pattern == 1) {
        int cp_seqlen = end - start;
        if (s_id < cp_seqlen / 2) {
          s_id_for_freqs = s_id + start_seq_ids[b_id];
        } else {
          s_id_for_freqs = end_seq_ids[b_id] - (cp_seqlen - s_id);
        }
      }
      rotary_block_forward_cpu(input, cos, sin, out, s_id_for_freqs,
                               offset_block, offset_block_dst, num_heads,
                               head_dim, freqs_dim, stride_h, stride_d,
                               o_stride_h, o_stride_d);
    }
  }
}

template <typename spec_t>
void rotary_varlen_backward_cpu(
  const spec_t* input, const spec_t* cos, const spec_t* sin, spec_t* out,
  const int* cu_seqlens, const int max_seqlen, const int* start_seq_ids,
  const int* end_seq_ids, const int split_pattern, const int batch_size,
  const int num_heads, const int head_dim, const int freqs_dim,
  const int stride_t, const int stride_h, const int stride_d,
  const int o_stride_t, const int o_stride_h, const int o_stride_d) {
#pragma unroll
  for (int b_id = 0; b_id < batch_size; ++b_id) {
    int start = cu_seqlens[b_id];
    int end = cu_seqlens[b_id + 1];
#pragma unroll
    for (int s_id = 0; s_id + start < end; ++s_id) {
      int t_id = start + s_id;
      if (t_id >= end)
        return;
      int offset_block = t_id * stride_t;
      int offset_block_dst = t_id * o_stride_t;

      int s_id_for_freqs = s_id;
      if (split_pattern == 0) {
        s_id_for_freqs = s_id + start_seq_ids[b_id];
      } else if (split_pattern == 1) {
        int cp_seqlen = end - start;
        if (s_id < cp_seqlen / 2) {
          s_id_for_freqs = s_id + start_seq_ids[b_id];
        } else {
          s_id_for_freqs = end_seq_ids[b_id] - (cp_seqlen - s_id);
        }
      }
      rotary_block_backward_cpu(
        input, cos, sin, out, s_id, offset_block, offset_block_dst, num_heads,
        head_dim, freqs_dim, stride_h, stride_d, o_stride_h, o_stride_d);
    }
  }
}

void RotaryCpu(const NDArray& input, const NDArray& cos, const NDArray& sin,
               NDArray& out, const int start_seq_id, const int end_seq_id,
               const int split_pattern, const Stream& stream) {
  // input为q或k, shape: {batch_size, seq_len, num_heads, head_dim}, out相同
  // cos,sin shape: {s, d}
  // start_s_id: 对于cp的情况，代表当前cp_rank的第一个seq_id在整个序列中的seq_id
  // 对于非cp的情况，该值应为0
  HT_ASSERT_CPU_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, cos);
  HT_ASSERT_SAME_DEVICE(input, sin);
  HT_ASSERT_SAME_DEVICE(input, out);

  HT_ASSERT(input->ndim() == 4)
    << "RotaryCpu requires input->ndim() == 4, got " << input->ndim();
  HT_ASSERT(out->ndim() == 4)
    << "RotaryCpu requires out->ndim() == 4, got " << out->ndim();

  CPUStream cpu_stream(stream);

  auto b = input->shape(0);
  auto s = input->shape(1);
  auto h = input->shape(2);
  auto d = input->shape(3);

  HT_ASSERT(d % 2 == 0) << "Rotary op requires head_dim must be even";

  HT_DISPATCH_FLOATING_TYPES(input->dtype(), spec_t, "RotaryCpu", [&]() {
    auto _future = cpu_stream.EnqueueTask(
      [input, cos, sin, out, start_seq_id, end_seq_id, split_pattern, b, s, h,
       d]() {
        rotary_forward_cpu<spec_t>(
          input->data_ptr<spec_t>(), cos->data_ptr<spec_t>(),
          sin->data_ptr<spec_t>(), out->data_ptr<spec_t>(), start_seq_id,
          end_seq_id, split_pattern, b, s, h, d, cos->shape(1),
          input->stride(0), input->stride(1), input->stride(2),
          input->stride(3), out->stride(0), out->stride(1), out->stride(2),
          out->stride(3));
      },
      "Rotary");
  });

  NDArray::MarkUsedBy({input, cos, sin, out}, stream);
}

void RotaryGradientCpu(const NDArray& input, const NDArray& cos,
                       const NDArray& sin, NDArray& out, const int start_seq_id,
                       const int end_seq_id, const int split_pattern,
                       const Stream& stream) {
  HT_ASSERT_CPU_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, cos);
  HT_ASSERT_SAME_DEVICE(input, sin);
  HT_ASSERT_SAME_DEVICE(input, out);

  HT_ASSERT(input->ndim() == 4)
    << "RotaryGradientCpu requires input->ndim() == 4, got " << input->ndim();
  HT_ASSERT(out->ndim() == 4)
    << "RotaryGradientCpu requires out->ndim() == 4, got " << out->ndim();

  CPUStream cpu_stream(stream);

  auto b = input->shape(0);
  auto s = input->shape(1);
  auto h = input->shape(2);
  auto d = input->shape(3);

  HT_ASSERT(d % 2 == 0) << "Rotary op requires head_dim must be even";

  HT_DISPATCH_FLOATING_TYPES(
    input->dtype(), spec_t, "RotaryGradientCpu", [&]() {
      auto _future = cpu_stream.EnqueueTask(
        [input, cos, sin, out, start_seq_id, end_seq_id, split_pattern, b, s, h,
         d]() {
          rotary_backward_cpu<spec_t>(
            input->data_ptr<spec_t>(), cos->data_ptr<spec_t>(),
            sin->data_ptr<spec_t>(), out->data_ptr<spec_t>(), start_seq_id,
            end_seq_id, split_pattern, b, s, h, d, cos->shape(1),
            input->stride(0), input->stride(1), input->stride(2),
            input->stride(3), out->stride(0), out->stride(1), out->stride(2),
            out->stride(3));
        },
        "RotaryGradient");
    });

  NDArray::MarkUsedBy({input, cos, sin, out}, stream);
}

void RotaryVarlenCpu(const NDArray& input, const NDArray& cos,
                     const NDArray& sin, NDArray& out,
                     const NDArray& cu_seqlens, const int max_seqlen,
                     const int* start_seq_ids, const int* end_seq_ids,
                     const int split_pattern, const Stream& stream) {
  // input为q或k, shape: {batch_size_mul_seq_len, num_heads, head_dim}, out相同
  HT_ASSERT_CPU_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, cos);
  HT_ASSERT_SAME_DEVICE(input, sin);
  HT_ASSERT_SAME_DEVICE(input, out);
  HT_ASSERT_SAME_DEVICE(input, cu_seqlens);

  HT_ASSERT(input->ndim() == 3)
    << "RotaryVarlenCpu requires input->ndim() == 3, got " << input->ndim();
  HT_ASSERT(out->ndim() == 3)
    << "RotaryVarlenCpu requires out->ndim() == 3, got " << out->ndim();

  CPUStream cpu_stream(stream);

  auto b = cu_seqlens->numel() - 1;
  auto h = input->shape(1);
  auto d = input->shape(2);

  HT_ASSERT(d % 2 == 0) << "Rotary op requires head_dim must be even";

  HT_DISPATCH_FLOATING_TYPES(input->dtype(), spec_t, "RotaryVarlenCpu", [&]() {
    auto _future = cpu_stream.EnqueueTask(
      [input, cos, sin, out, cu_seqlens, max_seqlen, start_seq_ids, end_seq_ids,
       split_pattern, b, h, d]() {
        rotary_varlen_forward_cpu<spec_t>(
          input->data_ptr<spec_t>(), cos->data_ptr<spec_t>(),
          sin->data_ptr<spec_t>(), out->data_ptr<spec_t>(),
          cu_seqlens->data_ptr<int>(), max_seqlen, start_seq_ids, end_seq_ids,
          split_pattern, b, h, d, cos->shape(1), input->stride(0),
          input->stride(1), input->stride(2), out->stride(0), out->stride(1),
          out->stride(2));
      },
      "RotaryVarlen");
  });

  NDArray::MarkUsedBy({input, cos, sin, out, cu_seqlens}, stream);
}

void RotaryVarlenGradientCpu(const NDArray& input, const NDArray& cos,
                             const NDArray& sin, NDArray& out,
                             const NDArray& cu_seqlens, const int max_seqlen,
                             const int* start_seq_ids, const int* end_seq_ids,
                             const int split_pattern, const Stream& stream) {
  // input为q或k, shape: {batch_size_mul_seq_len, num_heads, head_dim}, out相同
  HT_ASSERT_CPU_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, cos);
  HT_ASSERT_SAME_DEVICE(input, sin);
  HT_ASSERT_SAME_DEVICE(input, out);
  HT_ASSERT_SAME_DEVICE(input, cu_seqlens);

  HT_ASSERT(input->ndim() == 3)
    << "RotaryVarlenGradientCpu requires input->ndim() == 3, got "
    << input->ndim();
  HT_ASSERT(out->ndim() == 3)
    << "RotaryVarlenGradientCpu requires out->ndim() == 3, got " << out->ndim();

  CPUStream cpu_stream(stream);

  auto b = cu_seqlens->numel() - 1;
  auto h = input->shape(1);
  auto d = input->shape(2);

  HT_ASSERT(d % 2 == 0) << "Rotary op requires head_dim must be even";

  HT_DISPATCH_FLOATING_TYPES(
    input->dtype(), spec_t, "RotaryVarlenGradientCpu", [&]() {
      auto _future = cpu_stream.EnqueueTask(
        [input, cos, sin, out, cu_seqlens, max_seqlen, start_seq_ids,
         end_seq_ids, split_pattern, b, h, d]() {
          rotary_varlen_backward_cpu<spec_t>(
            input->data_ptr<spec_t>(), cos->data_ptr<spec_t>(),
            sin->data_ptr<spec_t>(), out->data_ptr<spec_t>(),
            cu_seqlens->data_ptr<int>(), max_seqlen, start_seq_ids, end_seq_ids,
            split_pattern, b, h, d, cos->shape(1), input->stride(0),
            input->stride(1), input->stride(2), out->stride(0), out->stride(1),
            out->stride(2));
        },
        "RotaryVarlenGradient");
    });

  NDArray::MarkUsedBy({input, cos, sin, out, cu_seqlens}, stream);
}

} // namespace impl
} // namespace hetu
