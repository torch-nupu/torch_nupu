#include <ATen/EmptyTensor.h>
#include <ATen/ops/as_strided_native.h>
#include <ATen/ops/set_cpu_dispatch.h>
#include <c10/core/Allocator.h>
#include <c10/core/TensorOptions.h>
#include <c10/util/ArrayRef.h>
#include <torch/library.h>

#include <CL/opencl.hpp>

namespace nupu {

namespace {

struct NupuAllocator final : at::Allocator {
  NupuAllocator() = default;
  at::DataPtr allocate(size_t nbytes) override {
    LOG(INFO) << "NupuAllocator::allocate - nbytes: " << nbytes;
    auto buffer =
        new cl::Buffer(cl::Context::getDefault(), CL_MEM_READ_WRITE, nbytes);
    LOG(INFO) << "buffer: " << buffer;
    return {
        buffer,
        buffer,
        raw_deleter(),
        at::Device(at::DeviceType::PrivateUse1, 0)};
  }

  static void nupu_raw_deleter(void* ptr) {
    if (!ptr) {
      return;
    }
    LOG(INFO) << "nupu_raw_deleter";
    auto buffer = static_cast<cl::Buffer*>(ptr);
    LOG(INFO) << "buffer: " << buffer;
    delete buffer;
    LOG(INFO) << "fin nupu_raw_deleter";
  }

  at::DeleterFnPtr raw_deleter() const override {
    return &nupu_raw_deleter;
  }

  void copy_data(void* dest, const void* src, std::size_t count) const final {
    auto buffer_dest = static_cast<cl::Buffer*>(dest);
    auto buffer_src = static_cast<const cl::Buffer*>(src);
    cl::enqueueCopyBuffer(*buffer_src, *buffer_dest, 0, 0, count);
    // TODO: sync ?
  }
};

static NupuAllocator global_nupu_alloc;
REGISTER_ALLOCATOR(c10::DeviceType::PrivateUse1, &global_nupu_alloc);

} // namespace

namespace {

at::Tensor empty_nupu(
    c10::IntArrayRef size,
    std::optional<c10::ScalarType> dtype_opt,
    std::optional<c10::Layout> layout_opt,
    std::optional<c10::Device> device_opt,
    std::optional<bool> pin_memory_opt,
    std::optional<c10::MemoryFormat> memory_format_opt) {
  const auto device = c10::device_or_default(device_opt);
  const auto dtype = c10::dtype_or_default(dtype_opt);
  TORCH_CHECK(device.is_privateuseone());
  TORCH_CHECK(
      c10::layout_or_default(layout_opt) == c10::Layout::Strided,
      "Non strided layout not supported");
  TORCH_CHECK(
      !c10::pinned_memory_or_default(pin_memory_opt),
      "Pin memory can only be on CPU");
  const c10::DeviceGuard device_guard(device);
  constexpr c10::DispatchKeySet pu1_dks(c10::DispatchKey::PrivateUse1);
  return at::detail::empty_generic(
      size, &global_nupu_alloc, pu1_dks, dtype, memory_format_opt);
}

at::Tensor empty_strided_nupu(
    c10::IntArrayRef size,
    c10::IntArrayRef stride,
    std::optional<c10::ScalarType> dtype_opt,
    std::optional<c10::Layout> layout_opt,
    std::optional<c10::Device> device_opt,
    std::optional<bool> pin_memory_opt) {
  const auto device = c10::device_or_default(device_opt);
  const auto dtype = c10::dtype_or_default(dtype_opt);
  TORCH_CHECK(device.is_privateuseone());
  TORCH_CHECK(
      c10::layout_or_default(layout_opt) == c10::Layout::Strided,
      "Non strided layout not supported");
  TORCH_CHECK(
      !c10::pinned_memory_or_default(pin_memory_opt),
      "Pin memory can only be on CPU");
  const c10::DeviceGuard device_guard(device);
  constexpr c10::DispatchKeySet pu1_dks(c10::DispatchKey::PrivateUse1);
  return at::detail::empty_strided_generic(
      size, stride, &global_nupu_alloc, pu1_dks, dtype);
}

at::Tensor as_strided_nupu(
    const at::Tensor& self,
    c10::IntArrayRef size,
    c10::IntArrayRef stride,
    std::optional<int64_t> storage_offset_) {
  return at::native::as_strided_tensorimpl(self, size, stride, storage_offset_);
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("empty.memory_format", empty_nupu);
  m.impl("empty_strided", empty_strided_nupu);
  m.impl("as_strided", as_strided_nupu);
}

} // namespace

} // namespace nupu
