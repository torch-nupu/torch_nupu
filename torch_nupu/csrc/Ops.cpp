#include <ATen/core/TensorBody.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/resize_native.h>
#include <torch/library.h>

#include <ATen/native/CPUFallback.h>

// TODO: mv ops to `src`
// TODO: rm printf usage

namespace nupu {

namespace {

at::Tensor nupu_add_Tensor(
    const at::Tensor& self,
    const at::Tensor& other,
    const at::Scalar& alpha) {
  // TODO: add add_Tensor sycl kernel
  printf(" -- add.Tensor is not supported, just return self");
  return self;
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("add.Tensor", &nupu_add_Tensor);
}

} // namespace

namespace {

void nupu_cpu_fallback(
    const c10::OperatorHandle& op,
    torch::jit::Stack* stack) {
  printf(" -- nupu_cpu_fallback, op: %s \n", op.operator_name().name.c_str());
  at::native::cpu_fallback(op, stack, /*error_on_views=*/true);
}

TORCH_LIBRARY_IMPL(_, PrivateUse1, m) {
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&nupu_cpu_fallback>());
}

} // namespace

} // namespace nupu
