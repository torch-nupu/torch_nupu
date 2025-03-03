#include <ATen/core/Tensor.h>
#include <ATen/native/Copy.h>
#include <ATen/native/TensorIterator.h>
#include <c10/core/ScalarType.h>
#include <c10/util/Logging.h>

#include <gen_functions/PrivateUse1NativeFunctions.h>
#include "c10/core/DeviceType.h"

namespace at_nupu {

at::Tensor& NupuNativeFunctions::copy_(
    at::Tensor& self,
    const at::Tensor& src,
    bool non_blocking) {
  LOG(INFO) << "NupuNativeFunctions::copy_";
  // TODO: support copy_

  LOG(INFO) << "self.data_ptr: " << self.data_ptr();
  LOG(INFO) << "src.src: " << src.data_ptr();
  LOG(INFO) << "self.device: " << self.device().str();
  LOG(INFO) << "src.device: " << src.device().str();

  if (self.device().type() == at::kCPU &&
      src.device().type() == at::kPrivateUse1) {
    LOG(INFO) << "nupu -> cpu";
  } else if (
      self.device().type() == at::kPrivateUse1 &&
      src.device().type() == at::kCPU) {
    LOG(INFO) << "cpu -> nupu";
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
