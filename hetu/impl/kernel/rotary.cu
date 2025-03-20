#include "hetu/core/ndarray.h"
#include "hetu/impl/stream/CUDAStream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/cuda_utils.h"
#include "hetu/impl/utils/cuda_math.h"
#include "hetu/impl/utils/offset_calculator.cuh"
#include "hetu/impl/kernel/Vectorized.cuh"


namespace hetu {
namespace impl {

template <typename spec_t>
__device__ void rotary_block_forward(const spec_t* input, const spec_t* cos,
                                        const spec_t* sin, spec_t* out,
                                        const int s_id, const int offset_block,
                                        const int offset_block_dst, const int num_heads,
                                        const int head_dim, const int freqs_dim,
                                        const int stride_h, const int stride_d,
                                        const int o_stride_h, const int o_stride_d) {
#pragma unroll
    for(int d_id = threadIdx.x; d_id < freqs_dim && d_id < head_dim; d_id += blockDim.x) {
        spec_t v_cos = cos[s_id * freqs_dim + d_id];
        spec_t v_sin = sin[s_id * freqs_dim + d_id];
#pragma unroll
        for(int h_id = threadIdx.y; h_id < num_heads; h_id += blockDim.y) {
            int offset_src = offset_block + h_id * stride_h + d_id * stride_d;
            int offset_dst = offset_block_dst + h_id * o_stride_h + d_id * o_stride_d;
            spec_t v_src = input[offset_src];
            spec_t v_src_rotate = (d_id + freqs_dim / 2 < freqs_dim)
                                    ? -input[offset_src + (freqs_dim / 2) * stride_d]
                                    :  input[offset_src + (freqs_dim / 2 - freqs_dim) * stride_d];
            out[offset_dst] = v_src * v_cos + v_src_rotate * v_sin;
        }        
    }

    // 处理head_dim超出freqs_dim的部分——直接复制过去
    if(head_dim > freqs_dim) {
#pragma unroll
        for(int h_id = threadIdx.y; h_id < num_heads; h_id += blockDim.y) {
            int offset_head = offset_block + h_id * stride_h;
            int offset_head_dst = offset_block_dst + h_id * o_stride_h;
#pragma unroll
            for(int d_id = freqs_dim + threadIdx.x; d_id < head_dim; d_id += blockDim.x) {
                out[offset_head_dst + d_id * o_stride_d] = input[offset_head + d_id * stride_d];
            }
        }
    }
}

template <typename spec_t>
__device__ void rotary_block_backward(const spec_t* input, const spec_t* cos,
                                        const spec_t* sin, spec_t* out,
                                        const int s_id, const int offset_block,
                                        const int offset_block_dst, const int num_heads,
                                        const int head_dim, const int freqs_dim,
                                        const int stride_h, const int stride_d,
                                        const int o_stride_h, const int o_stride_d) {
#pragma unroll
    for(int d_id = threadIdx.x; d_id < freqs_dim && d_id < head_dim; d_id += blockDim.x) {
        spec_t v_cos = cos[s_id * freqs_dim + d_id];
        spec_t v_sin = (d_id + freqs_dim / 2 < freqs_dim)
                        ? sin[s_id * freqs_dim + d_id + freqs_dim / 2]
                        : sin[s_id * freqs_dim + d_id + freqs_dim / 2 - freqs_dim];
#pragma unroll
        for(int h_id = threadIdx.y; h_id < num_heads; h_id += blockDim.y) {
            int offset_src = offset_block + h_id * stride_h + d_id * stride_d;
            int offset_dst = offset_block_dst + h_id * o_stride_h + d_id * o_stride_d;
            spec_t v_src = input[offset_src];
            spec_t v_src_rotate = (d_id + freqs_dim / 2 < freqs_dim)
                                    ?  input[offset_src + (freqs_dim / 2) * stride_d]
                                    : -input[offset_src + (freqs_dim / 2 - freqs_dim) * stride_d];
            out[offset_dst] = v_src * v_cos + v_src_rotate * v_sin;
        }        
    }

    // 处理head_dim超出freqs_dim的部分——直接复制过去
    if(head_dim > freqs_dim) {
#pragma unroll
        for(int h_id = threadIdx.y; h_id < num_heads; h_id += blockDim.y) {
            int offset_head = offset_block + h_id * stride_h;
            int offset_head_dst = offset_block_dst + h_id * o_stride_h;
#pragma unroll
            for(int d_id = freqs_dim + threadIdx.x; d_id < head_dim; d_id += blockDim.x) {
                out[offset_head_dst + d_id * o_stride_d] = input[offset_head + d_id * stride_d];
            }
        }
    }
}

template <typename spec_t>
__global__ void rotary_forward_kernel(const spec_t* input, const spec_t* cos,
                                      const spec_t* sin, spec_t* out, 
                                      const int start_seq_id, const int end_seq_id, 
                                      const int split_pattern, const int seq_len,
                                      const int num_heads, const int head_dim, const int freqs_dim,
                                      const int stride_b, const int stride_s,
                                      const int stride_h, const int stride_d,
                                      const int o_stride_b, const int o_stride_s,
                                      const int o_stride_h, const int o_stride_d) {
    int s_id = blockIdx.x;
    int b_id = blockIdx.y;
    int offset_block = s_id * stride_s + b_id * stride_b;
    int offset_block_dst = s_id * o_stride_s + b_id * o_stride_b;
    int s_id_for_freqs = s_id;
    if(split_pattern == 0) {
        s_id_for_freqs = s_id + start_seq_id;
    }
    else if(split_pattern == 1) {
        if(s_id < seq_len / 2) {
            s_id_for_freqs = s_id + start_seq_id;
        }
        else {
            s_id_for_freqs = end_seq_id - (seq_len - s_id);
        }
    }
    rotary_block_forward(input, cos, sin, out, s_id_for_freqs, offset_block, offset_block_dst, num_heads,
                         head_dim, freqs_dim, stride_h, stride_d, o_stride_h, o_stride_d);
}

template <typename spec_t>
__global__ void rotary_backward_kernel(const spec_t* input, const spec_t* cos,
                                      const spec_t* sin, spec_t* out, 
                                      const int start_seq_id, const int end_seq_id, 
                                      const int split_pattern, const int seq_len,          
                                      const int num_heads, const int head_dim, const int freqs_dim,
                                      const int stride_b, const int stride_s,
                                      const int stride_h, const int stride_d,
                                      const int o_stride_b, const int o_stride_s,
                                      const int o_stride_h, const int o_stride_d) {
    int s_id = blockIdx.x;
    int b_id = blockIdx.y;
    int offset_block = s_id * stride_s + b_id * stride_b;
    int offset_block_dst = s_id * o_stride_s + b_id * o_stride_b;
    int s_id_for_freqs = s_id;
    if(split_pattern == 0) {
        s_id_for_freqs = s_id + start_seq_id;
    }
    else if(split_pattern == 1) {
        if(s_id < seq_len / 2) {
            s_id_for_freqs = s_id + start_seq_id;
        }
        else {
            s_id_for_freqs = end_seq_id - (seq_len - s_id);
        }
    }
    rotary_block_backward(input, cos, sin, out, s_id_for_freqs, offset_block, offset_block_dst, num_heads,
                         head_dim, freqs_dim, stride_h, stride_d, o_stride_h, o_stride_d);
}

template <typename spec_t>
__global__ void rotary_varlen_forward_kernel(const spec_t* input, const spec_t* cos,
                                             const spec_t* sin, spec_t* out, const int* cu_seqlens,
                                             const int max_seqlen, const int* start_seq_ids,
                                             const int* end_seq_ids, const int split_pattern,
                                             const int num_heads, const int head_dim, const int freqs_dim,
                                             const int stride_t, const int stride_h,
                                             const int stride_d, const int o_stride_t,
                                             const int o_stride_h, const int o_stride_d) {
    int s_id = blockIdx.x;
    int b_id = blockIdx.y;
    int start = cu_seqlens[b_id];
    int end = cu_seqlens[b_id + 1];
    int t_id = start + s_id;
    if(t_id >= end) return;
    int offset_block = t_id * stride_t;
    int offset_block_dst = t_id * o_stride_t;

    int s_id_for_freqs = s_id;
    if(split_pattern == 0) {
        s_id_for_freqs = s_id + start_seq_ids[b_id];
    }
    else if(split_pattern == 1) {
        int cp_seqlen = end - start;
        if(s_id < cp_seqlen / 2) {
            s_id_for_freqs = s_id + start_seq_ids[b_id];
        } else {
            s_id_for_freqs = end_seq_ids[b_id] - (cp_seqlen - s_id);
        }
    }
    rotary_block_forward(input, cos, sin, out, s_id_for_freqs, offset_block, offset_block_dst, num_heads,
                         head_dim, freqs_dim, stride_h, stride_d, o_stride_h, o_stride_d);
}

template <typename spec_t>
__global__ void rotary_varlen_backward_kernel(const spec_t* input, const spec_t* cos,
                                             const spec_t* sin, spec_t* out, const int* cu_seqlens,
                                             const int max_seqlen, const int* start_seq_ids,
                                             const int* end_seq_ids, const int split_pattern,
                                             const int num_heads, const int head_dim, const int freqs_dim,
                                             const int stride_t, const int stride_h,
                                             const int stride_d, const int o_stride_t,
                                             const int o_stride_h, const int o_stride_d) {
    int s_id = blockIdx.x;
    int b_id = blockIdx.y;
    int start = cu_seqlens[b_id];
    int end = cu_seqlens[b_id + 1];
    int t_id = start + s_id;
    if(t_id >= end) return;
    int offset_block = t_id * stride_t;
    int offset_block_dst = t_id * o_stride_t;

    int s_id_for_freqs = s_id;
    if(split_pattern == 0) {
        s_id_for_freqs = s_id + start_seq_ids[b_id];
    }
    else if(split_pattern == 1) {
        int cp_seqlen = end - start;
        if(s_id < cp_seqlen / 2) {
            s_id_for_freqs = s_id + start_seq_ids[b_id];
        } else {
            s_id_for_freqs = end_seq_ids[b_id] - (cp_seqlen - s_id);
        }
    }
    rotary_block_backward(input, cos, sin, out, s_id, offset_block, offset_block_dst, num_heads,
                         head_dim, freqs_dim, stride_h, stride_d, o_stride_h, o_stride_d);
}


void RotaryCuda(const NDArray& input, const NDArray& cos, const NDArray& sin, 
                NDArray& out, const int start_seq_id, const int end_seq_id,
                const int split_pattern, const Stream& stream) {
    // input为q或k, shape: {batch_size, seq_len, num_heads, head_dim}, out相同
    // cos,sin shape: {s, d}
    // start_s_id: 对于cp的情况，代表当前cp_rank的第一个seq_id在整个序列中的seq_id
    // 对于非cp的情况，该值应为0
    HT_ASSERT_CUDA_DEVICE(input);
    HT_ASSERT_SAME_DEVICE(input, cos);
    HT_ASSERT_SAME_DEVICE(input, sin);
    HT_ASSERT_SAME_DEVICE(input, out);

    HT_ASSERT(input->ndim() == 4) << "RotaryCuda requires input->ndim() == 4, got " << input->ndim();
    HT_ASSERT(out->ndim() == 4) << "RotaryCuda requires out->ndim() == 4, got " << out->ndim();

    CUDAStream cuda_stream(stream);
    hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());

    auto b = input->shape(0);
    auto s = input->shape(1);
    auto h = input->shape(2);
    auto d = input->shape(3);

    HT_ASSERT(d % 2 == 0) << "Rotary op requires head_dim must be even";

    int wraps_per_block = h < 16 ? 4: 8;
    dim3 blocks(s, b);
    dim3 threads(32, wraps_per_block);
    HT_DISPATCH_FLOATING_TYPES(
    input->dtype(), spec_t, "RotaryCuda", [&](){
        rotary_forward_kernel<spec_t><<<blocks, threads, 0, cuda_stream>>>(
            input->data_ptr<spec_t>(), cos->data_ptr<spec_t>(), sin->data_ptr<spec_t>(),
            out->data_ptr<spec_t>(), start_seq_id, end_seq_id, split_pattern, 
            s, h, d, cos->shape(1), input->stride(0), input->stride(1),
            input->stride(2), input->stride(3), out->stride(0), out->stride(1),
            out->stride(2), out->stride(3)
        );
    });

    NDArray::MarkUsedBy({input, cos, sin, out}, stream);
}

void RotaryGradientCuda(const NDArray& input, const NDArray& cos, const NDArray& sin, 
                        NDArray& out, const int start_seq_id, const int end_seq_id,
                        const int split_pattern, const Stream& stream) {
    HT_ASSERT_CUDA_DEVICE(input);
    HT_ASSERT_SAME_DEVICE(input, cos);
    HT_ASSERT_SAME_DEVICE(input, sin);
    HT_ASSERT_SAME_DEVICE(input, out);

    HT_ASSERT(input->ndim() == 4) << "RotaryGradientCuda requires input->ndim() == 4, got " << input->ndim();
    HT_ASSERT(out->ndim() == 4) << "RotaryGradientCuda requires out->ndim() == 4, got " << out->ndim();

    CUDAStream cuda_stream(stream);
    hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());

    auto b = input->shape(0);
    auto s = input->shape(1);
    auto h = input->shape(2);
    auto d = input->shape(3);

    HT_ASSERT(d % 2 == 0) << "Rotary op requires head_dim must be even";

    int wraps_per_block = h < 16 ? 4: 8;
    dim3 blocks(s, b);
    dim3 threads(32, wraps_per_block);
    HT_DISPATCH_FLOATING_TYPES(
    input->dtype(), spec_t, "RotaryGradientCuda", [&](){
        rotary_backward_kernel<spec_t><<<blocks, threads, 0, cuda_stream>>>(
            input->data_ptr<spec_t>(), cos->data_ptr<spec_t>(), sin->data_ptr<spec_t>(),
            out->data_ptr<spec_t>(), start_seq_id, end_seq_id, split_pattern,
            s, h, d, cos->shape(1), input->stride(0), input->stride(1),
            input->stride(2), input->stride(3), out->stride(0), out->stride(1),
            out->stride(2), out->stride(3)
        );
    });

    NDArray::MarkUsedBy({input, cos, sin, out}, stream);
}

void RotaryVarlenCuda(const NDArray& input, const NDArray& cos, const NDArray& sin,
                      NDArray& out, const NDArray& cu_seqlens, const int max_seqlen,
                      const int* start_seq_ids, const int* end_seq_ids, const int split_pattern,
                      const Stream& stream) {
    // input为q或k, shape: {batch_size_mul_seq_len, num_heads, head_dim}, out相同
    HT_ASSERT_CUDA_DEVICE(input);
    HT_ASSERT_SAME_DEVICE(input, cos);
    HT_ASSERT_SAME_DEVICE(input, sin);
    HT_ASSERT_SAME_DEVICE(input, out);
    HT_ASSERT_SAME_DEVICE(input, cu_seqlens);

    HT_ASSERT(input->ndim() == 3) << "RotaryVarlenCuda requires input->ndim() == 3, got " << input->ndim();
    HT_ASSERT(out->ndim() == 3) << "RotaryVarlenCuda requires out->ndim() == 3, got " << out->ndim();
    
    CUDAStream cuda_stream(stream);
    hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());

    auto b = cu_seqlens->numel() - 1;
    auto h = input->shape(1);
    auto d = input->shape(2);

    HT_ASSERT(d % 2 == 0) << "Rotary op requires head_dim must be even";

    int wraps_per_block = h < 16 ? 4: 8;
    dim3 blocks(max_seqlen, b);
    dim3 threads(32, wraps_per_block);

    // 注意：传入的start_seq_ids是在cpu上的，而传入核函数的应该是cuda上的！
    int* start_seq_ids_cuda;
    int* end_seq_ids_cuda;
    size_t data_size = sizeof(int) * b;
    cudaMalloc((void**)&start_seq_ids_cuda, data_size);
    cudaMalloc((void**)&end_seq_ids_cuda, data_size);
    cudaMemcpy(start_seq_ids_cuda, start_seq_ids, data_size, cudaMemcpyHostToDevice);
    cudaMemcpy(end_seq_ids_cuda, end_seq_ids, data_size, cudaMemcpyHostToDevice);

    HT_DISPATCH_FLOATING_TYPES(
    input->dtype(), spec_t, "RotaryVarlenCuda", [&](){
        rotary_varlen_forward_kernel<spec_t><<<blocks, threads, 0, cuda_stream>>>(
            input->data_ptr<spec_t>(), cos->data_ptr<spec_t>(), sin->data_ptr<spec_t>(),
            out->data_ptr<spec_t>(), cu_seqlens->data_ptr<int>(), max_seqlen, 
            start_seq_ids_cuda, end_seq_ids_cuda, split_pattern, h, d, cos->shape(1), 
            input->stride(0), input->stride(1), input->stride(2), 
            out->stride(0), out->stride(1), out->stride(2)
        );
    });

    cudaFree(start_seq_ids_cuda);
    cudaFree(end_seq_ids_cuda);

    NDArray::MarkUsedBy({input, cos, sin, out, cu_seqlens}, stream);
}

void RotaryVarlenGradientCuda(const NDArray& input, const NDArray& cos, const NDArray& sin,
                      NDArray& out, const NDArray& cu_seqlens, const int max_seqlen,
                      const int* start_seq_ids, const int* end_seq_ids, const int split_pattern,
                      const Stream& stream) {
    // input为q或k, shape: {batch_size_mul_seq_len, num_heads, head_dim}, out相同
    HT_ASSERT_CUDA_DEVICE(input);
    HT_ASSERT_SAME_DEVICE(input, cos);
    HT_ASSERT_SAME_DEVICE(input, sin);
    HT_ASSERT_SAME_DEVICE(input, out);
    HT_ASSERT_SAME_DEVICE(input, cu_seqlens);

    HT_ASSERT(input->ndim() == 3) << "RotaryVarlenGradientCuda requires input->ndim() == 3, got " << input->ndim();
    HT_ASSERT(out->ndim() == 3) << "RotaryVarlenGradientCuda requires out->ndim() == 3, got " << out->ndim();

    CUDAStream cuda_stream(stream);
    hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());

    auto b = cu_seqlens->numel() - 1;
    auto h = input->shape(1);
    auto d = input->shape(2);

    HT_ASSERT(d % 2 == 0) << "Rotary op requires head_dim must be even";

    int wraps_per_block = h < 16 ? 4: 8;
    dim3 blocks(max_seqlen, b);
    dim3 threads(32, wraps_per_block);

    int* start_seq_ids_cuda;
    int* end_seq_ids_cuda;
    size_t data_size = sizeof(int) * b;
    cudaMalloc((void**)&start_seq_ids_cuda, data_size);
    cudaMalloc((void**)&end_seq_ids_cuda, data_size);
    cudaMemcpy(start_seq_ids_cuda, start_seq_ids, data_size, cudaMemcpyHostToDevice);
    cudaMemcpy(end_seq_ids_cuda, end_seq_ids, data_size, cudaMemcpyHostToDevice);

    HT_DISPATCH_FLOATING_TYPES(
    input->dtype(), spec_t, "RotaryVarlenGradientCuda", [&](){
        rotary_varlen_backward_kernel<spec_t><<<blocks, threads, 0, cuda_stream>>>(
            input->data_ptr<spec_t>(), cos->data_ptr<spec_t>(), sin->data_ptr<spec_t>(),
            out->data_ptr<spec_t>(), cu_seqlens->data_ptr<int>(), max_seqlen, 
            start_seq_ids_cuda, end_seq_ids_cuda, split_pattern, h, d, cos->shape(1), 
            input->stride(0), input->stride(1), input->stride(2), 
            out->stride(0), out->stride(1), out->stride(2)
        );
    });

    cudaFree(start_seq_ids_cuda);
    cudaFree(end_seq_ids_cuda);

    NDArray::MarkUsedBy({input, cos, sin, out, cu_seqlens}, stream);
}


}
}
