#include <torch/csrc/utils/pybind.h>

#include "Device.h"

namespace py = pybind11;

PYBIND11_MODULE(_C, m) {
  m.def("nupu_device", &nupu::get_nupu_device, "get nupu device object");
}
