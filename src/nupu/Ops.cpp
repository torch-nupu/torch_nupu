#include <ATen/core/TensorBody.h>
#include <torch/library.h>

#include <ATen/native/CPUFallback.h>

namespace nupu {

namespace {

void nupu_cpu_fallback(
    const c10::OperatorHandle& op,
    torch::jit::Stack* stack) {
  LOG(INFO) << "nupu_cpu_fallback, op: " << op.operator_name().name;
  at::native::cpu_fallback(op, stack, true);
}

TORCH_LIBRARY_IMPL(_, PrivateUse1, m) {
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&nupu_cpu_fallback>());
}

} // namespace

} // namespace nupu
