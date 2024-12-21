#include <ATen/Context.h>
#include <c10/core/DeviceType.h>
#include <torch/torch.h>

int main() {
  auto context = at::Context();
  context.lazyInitDevice(torch::kPrivateUse1);

  auto arg0 = at::tensor({1.0f, 2.0f, 3.0f}).to(torch::kCPU);
  auto arg1 = arg0.to(torch::kPrivateUse1);

  auto res0 = arg1 + arg1;
  auto ref0 = arg0 + arg0;

  assert(torch::allclose(res0.to(torch::kCPU), ref0));

  return 0;
}
