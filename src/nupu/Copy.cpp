#include <ATen/core/Tensor.h>
#include <ATen/native/Copy.h>
#include <ATen/native/TensorIterator.h>
#include <c10/core/ScalarType.h>
#include <c10/util/Logging.h>

#include <gen_functions/PrivateUse1NativeFunctions.h>

using namespace at;
using namespace at::xpu;

namespace at_nupu {

at::Tensor& NupuNativeFunctions::copy_(
    at::Tensor& self,
    const at::Tensor& src,
    bool non_blocking) {
  LOG(INFO) << "NupuNativeFunctions::copy_";
  // TODO: support copy_
  return self;
}

} // namespace at_nupu
