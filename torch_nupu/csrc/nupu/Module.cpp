#include <torch/csrc/Module.h>
#include <torch/csrc/THP.h>

#include <CL/opencl.hpp>

PyObject* _get_current_stream(PyObject* self, PyObject* device_index) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(
      THPUtils_checkLong(device_index), "invalid argument to current_stream");
  auto c10_device_index = THPUtils_unpackDeviceIndex(device_index);
  // TODO: support non-default and multi queues
  auto current_stream = new cl::CommandQueue(cl::CommandQueue::getDefault());
  LOG(INFO) << "current_stream: " << current_stream;

  PyObject* capsule =
      PyCapsule_New(current_stream, "clCommandQueue", [](PyObject* capsule) {
        auto ptr = static_cast<cl::CommandQueue*>(
            PyCapsule_GetPointer(capsule, "clCommandQueue"));
        delete ptr;
      });
  if (!capsule) {
    delete current_stream;
    return nullptr;
  }
  return capsule;
  END_HANDLE_TH_ERRORS
}

// NOLINTNEXTLINE(*-c-arrays*, *-global-variables)
static struct PyMethodDef _THNPModule_methods[] = {
    {"_get_current_stream", _get_current_stream, METH_O, nullptr},
    {nullptr}};

PyMethodDef* THNPModule_methods() {
  return _THNPModule_methods;
}
