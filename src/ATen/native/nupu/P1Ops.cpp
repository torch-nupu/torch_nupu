#include <ATen/native/DispatchStub.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/UnaryOps.h>
#include "ATen/ops/empty.h"
#include "ATen/ops/empty_like.h"

#include <ATen/native/xpu/sycl/AbsKernel.h>
#include <ATen/native/xpu/sycl/ActivationLogSigmoidKernels.h>
#include <ATen/native/xpu/sycl/BinaryKernels.h>

#include <gen_functions/PrivateUse1NativeFunctions.h>

namespace at::native {
// src/ATen/native/xpu/UnaryOps.cpp
REGISTER_PRIVATEUSE1_DISPATCH(abs_stub, &xpu::abs_kernel);
} // namespace at::native

namespace at::native {

// src/ATen/native/xpu/Activation.cpp
std::tuple<Tensor&, Tensor&> log_sigmoid_forward_out_xpu(
    const Tensor& input,
    Tensor& result,
    Tensor& buffer) {
  auto iter =
      TensorIteratorConfig().add_output(result).add_const_input(input).build();
  native::xpu::log_sigmoid_forward_kernel(iter);
  return std::forward_as_tuple(result, buffer);
}

// src/ATen/native/xpu/Activation.cpp
std::tuple<Tensor, Tensor> log_sigmoid_forward_xpu(const Tensor& input) {
  auto result = at::empty_like(input);
  auto buffer = at::empty({0}, input.options());
  log_sigmoid_forward_out_xpu(input, result, buffer);
  return std::forward_as_tuple(result, buffer);
}
} // namespace at::native

namespace at_nupu {

at::Tensor& NupuNativeFunctions::add_(
    at::Tensor& self,
    const at::Tensor& other,
    const at::Scalar& alpha) {
  auto iter = at::TensorIterator::binary_op(self, self, other);
  at::native::xpu::add_kernel(iter, alpha);
  return self;
}

at::Tensor& NupuNativeFunctions::add_out(
    const at::Tensor& self,
    const at::Tensor& other,
    const at::Scalar& alpha,
    at::Tensor& out) {
  auto iter = at::TensorIterator::binary_op(out, self, other);
  at::native::xpu::add_kernel(iter, alpha);
  return out;
}

at::Tensor& NupuNativeFunctions::abs_out(
    const at::Tensor& self,
    at::Tensor& out) {
  auto iter = at::TensorIterator::unary_op(out, self);
  at::native::xpu::abs_kernel(iter);
  return out;
}

} // namespace at_nupu
