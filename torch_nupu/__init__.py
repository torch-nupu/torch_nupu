import torch

import torch_nupu._C  # noqa

import torch_nupu._nupu as _NupuMod

# TODO: mv to cpp side
torch.utils.rename_privateuse1_backend("nupu")
# https://pytorch.org/tutorials/advanced/privateuseone.html#register-new-backend-module-to-pytorch
torch._register_device_module("nupu", _NupuMod)


# TODO: make it lazy?
def _init():
    torch_nupu._C.nupu_init()


_init()


# https://pytorch.org/tutorials/prototype/python_extension_autoload.html
def _autoload():
    pass
