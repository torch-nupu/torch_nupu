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

namespace {

// TODO: support _copy_from_and_resize
at::Tensor _copy_from_and_resize(
    const at::Tensor& self,
    const at::Tensor& dst) {
  printf(" -- _copy_from_and_resize \n");
  // TODO: support `resize`
  dst.copy_(self);
  return dst;
}

// TODO: support _copy_from
at::Tensor _copy_from(
    const at::Tensor& self,
    const at::Tensor& dst,
    bool non_blocking) {
  // dst.resize_as_(self);
  return const_cast<at::Tensor&>(dst.copy_(self, non_blocking));
}

// TODO: support `resize_`
const at::Tensor& resize_(
    const at::Tensor& self,
    at::IntArrayRef size,
    std::optional<at::MemoryFormat> optional_memory_format) {
  return self;
}

// TODO: support copy_
at::Tensor& copy_(at::Tensor& self, const at::Tensor& src, bool non_blocking) {
  return self;
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("_copy_from_and_resize", _copy_from_and_resize);
  m.impl("_copy_from", _copy_from);
  m.impl("copy_", copy_);
  m.impl("resize_", resize_);
}

} // namespace
} // namespace nupu
