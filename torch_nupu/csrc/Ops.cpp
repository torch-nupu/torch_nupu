#include <ATen/core/TensorBody.h>
#include <ATen/ops/empty.h>
#include <torch/library.h>

namespace nupu {

namespace {

at::Tensor nupu_add_Tensor(
    const at::Tensor& self,
    const at::Tensor& other,
    const at::Scalar& alpha) {
  // TODO: add add_Tensor sycl kernel
  return self;
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("add.Tensor", &nupu_add_Tensor);
}

} // namespace

} // namespace nupu
