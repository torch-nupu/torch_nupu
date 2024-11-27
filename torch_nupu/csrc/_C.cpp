#include <ATen/Context.h>
#include <torch/csrc/utils/pybind.h>

#include "Device.h"

namespace py = pybind11;

namespace {
void nupu_init() {
  at::globalContext().lazyInitPrivateUse1();
}
} // namespace

PYBIND11_MODULE(_C, m) {
  m.def("nupu_init", &nupu_init, "torch_nupu init");
  m.def("nupu_device", &nupu::get_nupu_device, "get nupu device object");
}
