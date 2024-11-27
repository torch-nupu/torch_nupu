#include <ATen/native/DispatchStub.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/UnaryOps.h>
#include "ATen/ops/empty.h"
#include "ATen/ops/empty_like.h"

#include <ATen/native/xpu/sycl/AbsKernel.h>
#include <ATen/native/xpu/sycl/ActivationLogSigmoidKernels.h>

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
