#pragma once

#include "hetu/core/stream.h"

namespace hetu {
namespace impl {

/******************************************************
 * External declaration of kernels
 ******************************************************/

#define DECLARE_KERNEL(KERNEL, DEVICE, ...)                                    \
  extern void KERNEL##DEVICE(__VA_ARGS__)

#define DECLARE_KERNEL_CPU(KERNEL, ...) DECLARE_KERNEL(KERNEL, Cpu, __VA_ARGS__)
#define DECLARE_KERNEL_CUDA(KERNEL, ...)                                       \
  DECLARE_KERNEL(KERNEL, Cuda, __VA_ARGS__)

#define DECLARE_KERNEL_CPU_AND_CUDA(KERNEL, ...)                               \
  DECLARE_KERNEL_CPU(KERNEL, __VA_ARGS__);                                     \
  DECLARE_KERNEL_CUDA(KERNEL, __VA_ARGS__)

DECLARE_KERNEL_CPU_AND_CUDA(Abs, const NDArray&, NDArray&, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(AbsGradient, const NDArray&, const NDArray&, NDArray&, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(Adam, const NDArray&, NDArray&, NDArray&, NDArray&, NDArray&,
                            float, float, float, float, float, bool, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(Arange, double, double, NDArray&, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(ArraySet, NDArray&, double, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(AddConst, const NDArray&, double, NDArray&,
                            const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(AddElewise, const NDArray&, const NDArray&,
                            NDArray&, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(AsStrided, const NDArray&, NDArray&,
                            const HTStride&, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(AsStridedGradient, const NDArray&, NDArray&,
                            const HTStride&, const Stream&);
DECLARE_KERNEL_CUDA(AsStridedGradient, const NDArray&, NDArray&, const HTShape&, const HTStride&,
                    const HTShape&, const HTStride&, int64_t, int64_t, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(AvgPool, const NDArray&, const size_t, const size_t,
                            NDArray&, const size_t, const size_t,
                            const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(AvgPoolGradient, const NDArray&, const NDArray&,
                            const NDArray&, const size_t, const size_t,
                            NDArray&, const size_t, const size_t,
                            const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(BatchMatMul, const NDArray& a, bool trans_a,
                            const NDArray& b, bool trans_b, NDArray& output,
                            const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(BatchNorm, const NDArray&, const NDArray&,
                            const NDArray&, NDArray&, double, double, NDArray&,
                            NDArray&, NDArray&, NDArray&, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(BatchNormGradient, const NDArray&, const NDArray&,
                            const NDArray&, NDArray&, NDArray&, NDArray&,
                            double, NDArray&, NDArray&, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(BinaryCrossEntropy, const NDArray& pred,
                            const NDArray& label, NDArray& loss, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(BinaryCrossEntropyGradient, const NDArray& pred,
                            const NDArray& label, const NDArray& grad_loss,
                            NDArray& output, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(Bool, const NDArray&, NDArray&, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(Broadcast, const NDArray&, NDArray&, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(BroadcastGradient, const NDArray&, NDArray&,
                            const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(BroadcastShape, const NDArray&, NDArray&,
                            const HTShape&, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(BroadcastShapeMul, const NDArray&, double, NDArray&,
                            const HTShape&, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(Ceil, const NDArray&, NDArray&, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(CheckFinite, const NDArray&, NDArray&, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(CheckNumeric, const NDArray&, NDArray&, const Stream&);
DECLARE_KERNEL_CUDA(Concat, const NDArrayList&, NDArray&, size_t, const Stream&);
DECLARE_KERNEL_CUDA(ConcatGradient, const NDArray&, NDArrayList&, size_t, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(Contiguous, const NDArray&, NDArray&, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(ContiguousGradient, const NDArray&, NDArray&, const Stream&);
DECLARE_KERNEL_CUDA(DynamicConcatenate, const NDArray&, NDArray&, size_t,
                    size_t, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(Conv2d, const NDArray&, const NDArray&, NDArray&,
                            const int, const int, const int, const int,
                            const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(Conv2dNaive, const NDArray&, const NDArray&, NDArray&,
                            const int, const int, const int, const int,
                            const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(Conv2dGradientofFilter, const NDArray&,
                            const NDArray&, NDArray&, const int, const int,
                            const int, const int, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(Conv2dGradientofFilterNaive, const NDArray&,
                            const NDArray&, NDArray&, const int, const int,
                            const int, const int, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(Conv2dGradientofData, const NDArray&,
                            const NDArray&, NDArray&, const int, const int,
                            const int, const int, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(Conv2dGradientofDataNaive, const NDArray&,
                            const NDArray&, NDArray&, const int, const int,
                            const int, const int, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(Conv2dAddBias, const NDArray&, const NDArray&,
                            const NDArray&, NDArray&, const int, const int,
                            const int, const int, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(Conv2dAddBiasNaive, const NDArray&, const NDArray&,
                            const NDArray&, NDArray&, const int, const int,
                            const int, const int, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(Conv2dBroadcast, const NDArray&, NDArray&,
                            const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(Conv2dReduceSum, const NDArray&, NDArray&,
                            const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(DataTransfer, const NDArray& from, NDArray& to,
                            const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(DeQuantization, const NDArray&, NDArray&, const NDArray&, NDArray&, 
                            int64_t, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(Diagonal, const NDArray&, NDArray&, int, int, int,
                            const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(DiagonalGradient, const NDArray&, NDArray&, int,
                            int, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(DivConst, const NDArray&, double, NDArray&,
                            const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(DivElewise, const NDArray&, const NDArray&,
                            NDArray&, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(Dot, const NDArray&, const NDArray&,
                            NDArray&, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(Dropout, const NDArray&, double, uint64_t, NDArray&,
                            NDArray&, const Stream&);
DECLARE_KERNEL_CUDA(DropoutAddLnFwd, const NDArray&, const NDArray&, const NDArray&,
                    const NDArray&, const NDArray& , const NDArray&, const NDArray&,
                    const NDArray&, NDArray&, NDArray&, NDArray&, NDArray&, NDArray&,
                    const float, const float, const float, const int64_t, 
                    bool, bool, const Stream&);
DECLARE_KERNEL_CUDA(DropoutAddLnBwd, const NDArray&, const NDArray&, const NDArray&,
                    const NDArray&, const NDArray& , const NDArray&, const NDArray&,
                    const NDArray&, const NDArray& , const NDArray&, const NDArray&,
                    const NDArray&, NDArray&, NDArray&, NDArray&, NDArray&, NDArray&,
                    NDArray&, NDArray&, NDArray&, const float, const float, 
                    const int64_t, const bool, bool, const Stream&);
DECLARE_KERNEL_CUDA(DropoutAddLnParallelResidualFwd, const NDArray&, const NDArray&,
                    const NDArray&, const NDArray& , const NDArray&, const NDArray&,
                    const NDArray&, NDArray&, NDArray&, NDArray&, NDArray&, NDArray&,
                    NDArray&, NDArray&, const float, const float, bool, bool, const Stream&);
DECLARE_KERNEL_CUDA(DropoutAddLnParallelResidualBwd, const NDArray&, const NDArray&, 
                    const NDArray&, const NDArray& , const NDArray&, const NDArray&,
                    const NDArray&, const NDArray& , const NDArray&, const NDArray&,
                    NDArray&, NDArray&, NDArray&, NDArray&, NDArray&, NDArray&, NDArray&,
                    NDArray&, NDArray&, NDArray&, NDArray&, const float,
                    const bool, const bool, bool, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(DropoutGradientWithRecomputation, const NDArray&,
                            double, uint64_t, NDArray&, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(DropoutGradient, const NDArray&, const NDArray&,
                            double, NDArray&, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(Dropout2d, const NDArray&, double, uint64_t, NDArray&,
                            NDArray&, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(Dropout2dGradient, const NDArray&, const NDArray&,
                            double, NDArray&, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(EmbeddingLookup, const NDArray&, const NDArray&,
                            NDArray&, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(EmbeddingLookupGradient, const NDArray&,
                            const NDArray&, NDArray&, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(Exp, const NDArray&, NDArray&, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(Eye, NDArray&, const Stream&);
DECLARE_KERNEL_CUDA(FlashAttn, const NDArray&, const NDArray&, const NDArray&,        
                    NDArray&, NDArray&, NDArray&, NDArray&, NDArray&, NDArray&,     
                    NDArray&, NDArray&, const float, const float,
                    const bool, const bool, const Stream&);
DECLARE_KERNEL_CUDA(FlashAttnGradient, const NDArray&, const NDArray&, const NDArray&,        
                    const NDArray&, NDArray&, NDArray&, NDArray&, NDArray&, NDArray&,     
                    NDArray&, const float, const float, const bool, const Stream&);
DECLARE_KERNEL_CUDA(FlashAttnVarlen, const NDArray&, const NDArray&,const NDArray&,
                    const NDArray&, const NDArray&, NDArray&, NDArray&, NDArray&, 
                    NDArray&, NDArray&, NDArray&, NDArray&, NDArray& rng_state,
                    const int, const int, const float, const float, const bool, 
                    const bool, const bool, const Stream&);
DECLARE_KERNEL_CUDA(FlashAttnVarlenGradient, const NDArray&, const NDArray&, const NDArray&,        
                    const NDArray&, const NDArray&, const NDArray&, 
                    NDArray&, NDArray&, NDArray&, NDArray&, NDArray&,     
                    NDArray&, const int, const int, const float, const float, 
                    const bool, const bool, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(Floor, const NDArray&, NDArray&, const Stream&);
DECLARE_KERNEL_CUDA(FusedLayerNorm, const NDArray&, const NDArray&,
                    const NDArray&, NDArray&, NDArray&, NDArray&,
                    int64_t, float,
                    const Stream&);
DECLARE_KERNEL_CUDA(FusedLayerNormGradient, const NDArray&, const NDArray&,
                    const NDArray&, const NDArray&, NDArray&, NDArray&, NDArray&,
                    const NDArray&, const NDArray&, int64_t, float, bool,
                    const Stream&);
DECLARE_KERNEL_CUDA(FusedRMSNorm, const NDArray&, const NDArray&,
                    NDArray&, NDArray&, int64_t, float,
                    const Stream&);
DECLARE_KERNEL_CUDA(FusedRMSNormGradient, const NDArray&, const NDArray&,
                    const NDArray&, NDArray&, NDArray&, const NDArray&, 
                    int64_t, float, bool, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(Gather, const NDArray&, const NDArray&, NDArray&,
                            size_t, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(GatherGradient, const NDArray&, const NDArray&, 
                            NDArray&, size_t, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(Gelu, const NDArray&, NDArray&, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(GeluGradient, const NDArray&, const NDArray&,
                            NDArray&, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(IndexAdd, const NDArray&, const NDArray&, NDArray&,
                            size_t, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(InstanceNorm, const NDArray&, NDArray&, NDArray&,
                            NDArray&, float, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(InstanceNormGradient, const NDArray&,
                            const NDArray&, NDArray&, const NDArray&,
                            const NDArray&, float, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(Interpolate, const NDArray&, NDArray&,
                            bool, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(InterpolateGradient, const NDArray&,
                            NDArray&, bool, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(KLDivLoss, const NDArray& pred,
                            const NDArray& label, NDArray& loss, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(KLDivLossGradient, const NDArray& pred,
                            const NDArray& label, const NDArray& grad_loss,
                            NDArray& output, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(LayerNorm, const NDArray&, const NDArray&,
                            const NDArray&, NDArray&, NDArray&, NDArray&,
                            int64_t, float,
                            const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(LayerNormGradient, const NDArray&, const NDArray&,
                            const NDArray&, NDArray&, NDArray&, NDArray&,
                            const NDArray&, const NDArray&, int64_t, float,
                            const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(LeakyRelu, const NDArray&, double, NDArray&,
                            const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(LeakyReluGradient, const NDArray&, const NDArray&,
                            double, NDArray&, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(Linear, const NDArray& a, bool trans_a, const NDArray& b,
                            bool trans_b, const NDArray& bias, NDArray& output,
                            const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(Log, const NDArray&, NDArray&, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(Maskedfill, const NDArray&, const NDArray&,
                            double, NDArray&, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(MatDot, const NDArray&, const NDArray&, NDArray&,
                            const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(MatMul, const NDArray& a, bool trans_a, const NDArray& b,
                            bool trans_b, NDArray& output, const Stream&);
DECLARE_KERNEL_CUDA(MatMul4Bit, const NDArray&, bool, const NDArray&, bool,
                    const NDArray&, const NDArray&, NDArray&,
                    int blocksize, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(MatVecMul, const NDArray&, bool, const NDArray&,
                            NDArray&, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(MaxPool, const NDArray&, const size_t, const size_t,
                            NDArray&, const size_t, const size_t,
                            const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(MaxPoolGradient, const NDArray&, const NDArray&,
                            const NDArray&, const size_t, const size_t,
                            NDArray&, const size_t, const size_t,
                            const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(MSELoss, const NDArray& pred,
                            const NDArray& label, NDArray& loss, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(MSELossGradient, const NDArray& pred,
                            const NDArray& label, const NDArray& grad_loss,
                            NDArray& output, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(MulConst, const NDArray&, double, NDArray&,
                            const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(MulElewise, const NDArray&, const NDArray&,
                            NDArray&, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(NLLLoss, const NDArray& pred,
                            const NDArray& label, NDArray& loss, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(NLLLossGradient, const NDArray& pred,
                            const NDArray& label, const NDArray& grad_loss,
                            NDArray& output, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(Norm, const NDArray&, NDArray&,
                            int64_t, int64_t, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(NormGradient, const NDArray&, const NDArray&, const NDArray&,
                            NDArray&, int64_t, int64_t, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(Onehot, const NDArray&, size_t, NDArray&,
                            const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(Opposite, const NDArray&, NDArray&, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(Outer, const NDArray&, const NDArray&, NDArray&, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(Pad, const NDArray&, NDArray&, const HTShape&,
                            const Stream&, std::string, double);
DECLARE_KERNEL_CPU_AND_CUDA(PadGradient, const NDArray&, NDArray&,
                            const HTShape&, const Stream&, std::string);
DECLARE_KERNEL_CPU_AND_CUDA(Pow, const NDArray&, double, NDArray&,
                            const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(Quantization, const NDArray&, NDArray&, const NDArray&, NDArray&, 
                            int64_t, bool, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(RangeMask, const NDArray&, int64_t, int64_t,
                            NDArray&, const Stream&);                            
DECLARE_KERNEL_CPU_AND_CUDA(Reduce, const NDArray&, NDArray&, const HTAxes&,
                            ReductionType, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(ReduceMax, const NDArray&, NDArray&, const int64_t*,
                            int64_t, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(ReduceMean, const NDArray&, NDArray&,
                            const int64_t*, int64_t, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(ReduceMin, const NDArray&, NDArray&, const int64_t*,
                            int64_t, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(ReduceSum, const NDArray&, NDArray&, const int64_t*,
                            int64_t, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(Relu, const NDArray&, NDArray&, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(ReluGradient, const NDArray&, const NDArray&,
                            NDArray&, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(Repeat, const NDArray&, NDArray&, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(RepeatGradient, const NDArray&, NDArray&, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(Reshape, const NDArray&, NDArray&, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(Roll, const NDArray&, const HTShape&, const HTAxes&,
                            NDArray&, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(RollGradient, const NDArray&, NDArray&, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(Rotary, const NDArray&, const NDArray&, const NDArray&, 
                            NDArray&, const int, const int, const int, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(RotaryGradient, const NDArray&, const NDArray&, const NDArray&, 
                            NDArray&, const int, const int, const int, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(RotaryVarlen, const NDArray&, const NDArray&, const NDArray&, 
                            NDArray&, const NDArray&, const int, const int*, const int*,
                            const int, const Stream&);  
DECLARE_KERNEL_CPU_AND_CUDA(RotaryVarlenGradient, const NDArray&, const NDArray&, const NDArray&, 
                            NDArray&, const NDArray&, const int, const int*, const int*,
                            const int, const Stream&);  
DECLARE_KERNEL_CPU_AND_CUDA(Round, const NDArray&, NDArray&, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(SGDUpdate, const NDArray&, NDArray&, NDArray&,
                            float, float, bool, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(SGDUpdateWithGradScaler, const NDArray&, const NDArray&, NDArray&, NDArray&,
                            float, float, bool, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(Sigmoid, const NDArray&, NDArray&, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(SigmoidGradient, const NDArray&, const NDArray&, NDArray&, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(Sin, const NDArray&, NDArray&, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(SinGradient, const NDArray&, const NDArray&, NDArray&, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(Cos, const NDArray&, NDArray&, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(CosGradient, const NDArray&, const NDArray&, NDArray&, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(Slice, const NDArray&, NDArray&, const HTShape&,
                            const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(SliceGradient, const NDArray&, NDArray&, const HTShape&,
                            const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(Softmax, const NDArray&, NDArray&, int64_t, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(SoftmaxGradient, const NDArray&, const NDArray&,
                            NDArray&, int64_t, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(SoftmaxCrossEntropy, const NDArray&, const NDArray&,
                            NDArray&, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(SoftmaxCrossEntropyGradient, const NDArray&,
                            const NDArray&, const NDArray&, NDArray&,
                            const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(SoftmaxCrossEntropySparse, const NDArray&, const NDArray&,
                            NDArray&, const int64_t, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(SoftmaxCrossEntropySparseGradient, const NDArray&,
                            const NDArray&, const NDArray&, NDArray&, const int64_t,
                            const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(Sqrt, const NDArray&, NDArray&, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(SubConst, const NDArray&, double, NDArray&,
                            const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(SubElewise, const NDArray&, const NDArray&,
                            NDArray&, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(Swiglu, const NDArray&, NDArray&, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(SwigluGradient, const NDArray&, const NDArray&, NDArray&, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(ReciprocalSqrt, const NDArray&, NDArray&,
                            const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(Reciprocal, const NDArray&, NDArray&,
                            const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(Tanh, const NDArray&, NDArray&, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(TanhGradient, const NDArray&, const NDArray&,
                            NDArray&, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(Transpose, const NDArray&, NDArray&, const HTAxes&,
                            const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(TriuTril, const NDArray&, NDArray&, bool, 
                            int64_t, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(UpdateScale, NDArray&, NDArray&, const NDArray&, double, 
                            double, int, const Stream&);
DECLARE_KERNEL_CUDA(VocabParallelCrossEntropy, const NDArray&, const NDArray&, 
                    const int64_t, const int64_t, const int64_t, NDArray&, 
                    NDArray&, const Stream&);
DECLARE_KERNEL_CUDA(VocabParallelCrossEntropyGradient, const NDArray&, const NDArray&, 
                    const int64_t, const int64_t, const int64_t, const NDArray&, 
                    NDArray&, const Stream&);                    
DECLARE_KERNEL_CPU_AND_CUDA(Where, const NDArray&, const NDArray&,
                            const NDArray&, NDArray&, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(NormalInits, NDArray&, double, double, uint64_t,
                            const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(UniformInits, NDArray&, double, double, uint64_t,
                            const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(TruncatedNormalInits, NDArray&, double, double,
                            double, double, uint64_t, const Stream&);

// Communication kernels
DECLARE_KERNEL_CPU_AND_CUDA(AllReduce, const NDArray&, NDArray&, ReductionType,
                            const DeviceGroup&, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(P2PSend, const NDArray&, const Device&, 
                            const std::vector<int>&, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(P2PRecv, NDArray&, const Device&, 
                            const std::vector<int>&, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(BatchedISendIRecv, const NDArrayList&, 
                            const std::vector<Device>&, NDArrayList&,
                            const std::vector<Device>&, const std::vector<Device>&, 
                            const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(AllGather, const NDArray&, NDArray&,
                            const DeviceGroup&, int32_t gather_dim, const Stream&);
DECLARE_KERNEL_CPU_AND_CUDA(ReduceScatter, const NDArray&, NDArray&, ReductionType,
                            const DeviceGroup&, int32_t scatter_dim, const Stream&);

//other activations
DECLARE_KERNEL_CUDA(Elu, const NDArray&, double, double, NDArray&,
                    const Stream&);
DECLARE_KERNEL_CUDA(EluGradient, const NDArray&, const NDArray&,
                    double, double, NDArray&, const Stream&);
DECLARE_KERNEL_CUDA(Hardshrink, const NDArray&, double, NDArray&,
                    const Stream&);
DECLARE_KERNEL_CUDA(HardshrinkGradient, const NDArray&, const NDArray&,
                    double, NDArray&, const Stream&);
DECLARE_KERNEL_CUDA(Hardsigmoid, const NDArray&, NDArray&,
                    const Stream&);
DECLARE_KERNEL_CUDA(HardsigmoidGradient, const NDArray&, const NDArray&,
                    NDArray&, const Stream&);
DECLARE_KERNEL_CUDA(Hardtanh, const NDArray&, double, double, NDArray&,
                    const Stream&);
DECLARE_KERNEL_CUDA(HardtanhGradient, const NDArray&, const NDArray&,
                    double, double, NDArray&, const Stream&);
DECLARE_KERNEL_CUDA(Hardswish, const NDArray&, NDArray&,
                    const Stream&);
DECLARE_KERNEL_CUDA(HardswishGradient, const NDArray&, const NDArray&,
                    NDArray&, const Stream&);
DECLARE_KERNEL_CUDA(Logsigmoid, const NDArray&, NDArray&, bool,
                    const Stream&);
DECLARE_KERNEL_CUDA(LogsigmoidGradient, const NDArray&, const NDArray&,
                    NDArray&, const Stream&);
DECLARE_KERNEL_CUDA(Silu, const NDArray&, NDArray&,
                    const Stream&);
DECLARE_KERNEL_CUDA(SiluGradient, const NDArray&, const NDArray&,
                    NDArray&, const Stream&);
DECLARE_KERNEL_CUDA(Mish, const NDArray&, NDArray&,
                    const Stream&);
DECLARE_KERNEL_CUDA(MishGradient, const NDArray&, const NDArray&,
                    NDArray&, const Stream&);
DECLARE_KERNEL_CUDA(Softplus, const NDArray&, double, double, NDArray&,
                    const Stream&);
DECLARE_KERNEL_CUDA(SoftplusGradient, const NDArray&, const NDArray&,
                    double, double, NDArray&, const Stream&);   
DECLARE_KERNEL_CUDA(Softshrink, const NDArray&, double, NDArray&,
                    const Stream&);
DECLARE_KERNEL_CUDA(SoftshrinkGradient, const NDArray&, const NDArray&,
                    double, NDArray&, const Stream&);
               
/******************************************************
 * Dispatching kernels for operations
 ******************************************************/

#define HT_DISPATCH_KERNEL_CASE_CPU(KERNEL, ...)                               \
  case kCPU: {                                                                 \
    KERNEL##Cpu(__VA_ARGS__);                                                  \
    break;                                                                     \
  }

#define HT_DISPATCH_KERNEL_CASE_CUDA(KERNEL, ...)                              \
  case kCUDA: {                                                                \
    KERNEL##Cuda(__VA_ARGS__);                                                 \
    break;                                                                     \
  }

#define HT_DISPATCH_KERNEL_SWITCH(DEVICE_TYPE, OP_TYPE, ...)                   \
  do {                                                                         \
    const auto& _device_type = DEVICE_TYPE;                                    \
    switch (_device_type) {                                                    \
      __VA_ARGS__                                                              \
      default:                                                                 \
        HT_NOT_IMPLEMENTED << "\"" << OP_TYPE << "\" is not implemented for "  \
                           << "\"" << DeviceType2Str(_device_type) << "\"";    \
    }                                                                          \
  } while (0)

#define HT_DISPATCH_KERNEL_CPU_AND_CUDA(DEVICE_TYPE, OP_TYPE, KERNEL, ...)     \
  HT_DISPATCH_KERNEL_SWITCH(                                                   \
    DEVICE_TYPE, OP_TYPE,                                                      \
    HT_DISPATCH_KERNEL_CASE_CPU(KERNEL, __VA_ARGS__)                           \
      HT_DISPATCH_KERNEL_CASE_CUDA(KERNEL, __VA_ARGS__))

#define HT_DISPATCH_KERNEL_CPU_ONLY(DEVICE_TYPE, OP_TYPE, KERNEL, ...)         \
  HT_DISPATCH_KERNEL_SWITCH(DEVICE_TYPE, OP_TYPE,                              \
                            HT_DISPATCH_KERNEL_CASE_CPU(KERNEL, __VA_ARGS__))

#define HT_DISPATCH_KERNEL_CUDA_ONLY(DEVICE_TYPE, OP_TYPE, KERNEL, ...)        \
  HT_DISPATCH_KERNEL_SWITCH(DEVICE_TYPE, OP_TYPE,                              \
                            HT_DISPATCH_KERNEL_CASE_CUDA(KERNEL, __VA_ARGS__))

} // namespace impl
} // namespace hetu
