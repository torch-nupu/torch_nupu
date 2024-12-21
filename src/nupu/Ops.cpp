#include <ATen/core/TensorBody.h>
#include <torch/library.h>

#include <ATen/native/CPUFallback.h>

namespace nupu {

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
