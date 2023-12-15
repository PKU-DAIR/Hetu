#include "hetu/core/ndarray.h"
#include "hetu/impl/stream/CUDAStream.h"
#include "hetu/impl/kernel/Reduce.cuh"

namespace hetu {
namespace impl {

template <typename arg_t>
struct ProdFunctor {
  __device__ __forceinline__ arg_t operator()(arg_t a, arg_t b) const {
    return a * b;
  }
};

void ReduceProdCuda(const NDArray& in_arr, NDArray& out_arr, const int64_t* axes,
                   int64_t num_ax, const Stream& stream) {
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    in_arr->dtype(), spec_t, "ReduceProdCuda", [&]() {
        using arg_t = opmath_type<spec_t>;
        launch_reduce_kernel<spec_t, arg_t>(in_arr, out_arr, axes, num_ax,
                             functor_wrapper<spec_t, arg_t>(ProdFunctor<arg_t>()), 1., stream);
    });
}

} // namespace impl
} // namespace hetu