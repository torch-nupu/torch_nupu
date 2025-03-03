#include <ATen/Context.h>
#include <torch/csrc/utils.h>
#include <torch/csrc/utils/pybind.h>

#include "nupu/Module.h"

namespace py = pybind11;

static PyObject* module;
static std::vector<PyMethodDef> methods;

namespace {
void nupu_init() {
  C10_LOG_API_USAGE_ONCE("torch_nupu._C.nupu_init");
  at::globalContext().lazyInitDevice(at::kPrivateUse1);
}
} // namespace

extern "C" C10_EXPORT PyObject* initNupuModule();
PyObject* initNupuModule() {
  THPUtils_addPyMethodDefs(methods, THNPModule_methods());
  static struct PyModuleDef nupumodule = {
      PyModuleDef_HEAD_INIT, "torch_nupu._C", nullptr, -1, methods.data()};
  module = PyModule_Create(&nupumodule);

  auto m = py::reinterpret_borrow<py::module>(module);
  m.def("nupu_init", &nupu_init, "torch_nupu init");

  return module;
}
