#include <torch/csrc/Module.h>
#include <torch/csrc/THP.h>

#include <CL/opencl.hpp>

PyObject* _get_current_stream(PyObject* self, PyObject* device_index) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(
      THPUtils_checkLong(device_index), "invalid argument to current_stream");
  auto c10_device_index = THPUtils_unpackDeviceIndex(device_index);
  // TODO: support non-default and multi queues
  auto queue = cl::CommandQueue::getDefault();
  PyObject* output_tuple = PyTuple_New(3);
  return PyLong_FromVoidPtr(queue.get());
  END_HANDLE_TH_ERRORS
}

// NOLINTNEXTLINE(*-c-arrays*, *-global-variables)
static struct PyMethodDef _THNPModule_methods[] = {
    {"_get_current_stream", _get_current_stream, METH_O, nullptr},
    {nullptr}};

PyMethodDef* THNPModule_methods() {
  return _THNPModule_methods;
}
