import torch

import torch_nupu._C  # noqa


# TODO: support `_register_device_module`
# https://pytorch.org/tutorials/advanced/privateuseone.html#register-new-backend-module-to-pytorch
class _NupuMod:
    pass


# TODO: mv to cpp side
torch.utils.rename_privateuse1_backend("nupu")
torch._register_device_module("nupu", _NupuMod())


# TODO: support `_autoload`
# https://pytorch.org/tutorials/prototype/python_extension_autoload.html
def _autoload():
    pass
