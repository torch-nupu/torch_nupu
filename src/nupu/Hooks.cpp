#include <ATen/detail/PrivateUse1HooksInterface.h>

#include <CL/opencl.hpp>

namespace at::detail {
namespace {

struct NupuHooksArgs : public at::PrivateUse1HooksArgs {};

struct NupuHooksInterface : public at::PrivateUse1HooksInterface {
  NupuHooksInterface(NupuHooksArgs) {}
  ~NupuHooksInterface() override = default;

  void init() const final {
    // TODO: support config default device from args/env
    cl::Context::setDefault(cl::Context(CL_DEVICE_TYPE_GPU));
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
