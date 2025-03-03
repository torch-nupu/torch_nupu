#include <ATen/core/Tensor.h>
#include <ATen/native/Copy.h>
#include <ATen/native/TensorIterator.h>
#include <c10/core/DeviceType.h>
#include <c10/core/ScalarType.h>
#include <c10/util/Logging.h>

#include <gen_functions/PrivateUse1NativeFunctions.h>

#include <CL/opencl.hpp>

namespace at_nupu {

at::Tensor& NupuNativeFunctions::copy_(
    at::Tensor& self,
    const at::Tensor& src,
    bool non_blocking) {
  // TODO: support copy_
  LOG(INFO) << "NupuNativeFunctions::copy_";
  LOG(INFO) << "self.data_ptr: " << self.data_ptr();
  LOG(INFO) << "src.src: " << src.data_ptr();
  LOG(INFO) << "self.device: " << self.device().str();
  LOG(INFO) << "src.device: " << src.device().str();

  if (self.device().type() == at::kCPU &&
      src.device().type() == at::kPrivateUse1) {
    LOG(INFO) << "nupu -> cpu";
    auto buffer_src = static_cast<cl::Buffer*>(src.data_ptr());
    cl::enqueueReadBuffer(*buffer_src, true, 0, self.nbytes(), self.data_ptr());
  } else if (
      self.device().type() == at::kPrivateUse1 &&
      src.device().type() == at::kCPU) {
    LOG(INFO) << "cpu -> nupu";
    auto buffer_dest = static_cast<cl::Buffer*>(self.data_ptr());
    cl::enqueueWriteBuffer(
        *buffer_dest, true, 0, self.nbytes(), src.data_ptr());
  } else if (
      self.device().type() == at::kCPU && src.device().type() == at::kCPU) {
    LOG(INFO) << "cpu -> cpu";
  } else if (
      self.device().type() == at::kPrivateUse1 &&
      src.device().type() == at::kPrivateUse1) {
    LOG(INFO) << "nupu -> nupu";
  } else {
    LOG(INFO) << "unknown copy_";
  }

  return self;
}

} // namespace at_nupu
