#include <c10/core/Allocator.h>
#include <c10/core/Device.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>

#include "Device.h"

namespace at {
namespace detail {
// TODO: support device guard
C10_REGISTER_GUARD_IMPL(
    PrivateUse1,
    c10::impl::NoOpDeviceGuardImpl<DeviceType::PrivateUse1>);
} // namespace detail
} // namespace at

namespace nupu {
c10::Device get_nupu_device() {
  return c10::Device(c10::DeviceType::PrivateUse1, 0);
}
} // namespace nupu
