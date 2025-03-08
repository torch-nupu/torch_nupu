#include <ATen/detail/PrivateUse1HooksInterface.h>
#include <c10/util/Logging.h>

#include <CL/opencl.hpp>

namespace at::detail {
namespace {

struct NupuHooksArgs : public at::PrivateUse1HooksArgs {};

struct NupuHooksInterface : public at::PrivateUse1HooksInterface {
  NupuHooksInterface(NupuHooksArgs) {}
  ~NupuHooksInterface() override = default;

  void init() const final {
#if __APPLE__
    auto platform = cl::Platform::getDefault();
    cl_context_properties properties[3] = {
        CL_CONTEXT_PLATFORM, (cl_context_properties)platform(), 0};
    cl::Context::setDefault(cl::Context(CL_DEVICE_TYPE_GPU, properties));
#else
    // TODO: support config default device from args/env
    cl::Context::setDefault(cl::Context(CL_DEVICE_TYPE_GPU));
#endif
    auto d = cl::Device::getDefault();
    LOG(INFO) << "default opencl device: " << d.getInfo<CL_DEVICE_NAME>();
  }
};

C10_DECLARE_REGISTRY(
    PrivateUse1HooksRegistry,
    NupuHooksInterface,
    NupuHooksArgs);
C10_DEFINE_REGISTRY(PrivateUse1HooksRegistry, NupuHooksInterface, NupuHooksArgs)

static at::PrivateUse1HooksInterface* get_nupu_hooks() {
  static at::PrivateUse1HooksInterface* nupu_hooks;
  static std::once_flag once;
  std::call_once(
      once, [] { nupu_hooks = new NupuHooksInterface(NupuHooksArgs{}); });
  return nupu_hooks;
}

int register_hook() {
  at::RegisterPrivateUse1HooksInterface(get_nupu_hooks());
  return 0;
}

static const int _torch_nupu_hooks_flag = register_hook();

} // namespace
} // namespace at::detail
