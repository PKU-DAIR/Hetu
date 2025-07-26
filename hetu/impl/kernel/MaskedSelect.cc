#include "hetu/core/ndarray.h"
#include "hetu/core/stream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/omp_utils.h"
#include "hetu/impl/stream/CPUStream.h"
#include <cmath>


namespace hetu{
namespace impl{


void MaskedSelectCpu(const NDArray& input, const NDArray& mask,
                NDArray& output, const Stream& stream) {
    int x = 1;
    x ++;
    return;

}

}
}