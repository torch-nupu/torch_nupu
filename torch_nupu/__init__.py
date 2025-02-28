from __future__ import annotations


# https://pytorch.org/tutorials/prototype/python_extension_autoload.html
def _autoload():
    pass


import torch
from torch._dynamo.device_interface import register_interface_for_device
from torch._inductor.codegen.common import (
    register_backend_for_device,
    register_device_op_overrides,
)

import torch_nupu._nupu as _NupuMod
from torch_nupu._nupu_codegen import (
    NupuDeviceOpOverrides,
    NupuScheduling,
    NupuWrapperCodegen,
)
from torch_nupu._nupu_device_interface import NupuDeviceInterface

# TODO: mv to cpp side
torch.utils.rename_privateuse1_backend("nupu")
# https://pytorch.org/tutorials/advanced/privateuseone.html#register-new-backend-module-to-pytorch
torch._register_device_module("nupu", _NupuMod)

# support torch.inductor
register_backend_for_device(
    "nupu",
    NupuScheduling,
    NupuWrapperCodegen,
)
register_device_op_overrides("nupu", NupuDeviceOpOverrides())
register_interface_for_device("nupu", NupuDeviceInterface)
