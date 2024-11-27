import torch

import torch_nupu._C  # noqa


# TODO: make it lazy?
def _init():
    torch_nupu._C.nupu_init()


_init()


# TODO: support `_register_device_module`
# https://pytorch.org/tutorials/advanced/privateuseone.html#register-new-backend-module-to-pytorch
class _NupuMod:
    @staticmethod
    def is_available():
        return True

    @staticmethod
    def device_count():
        return 1

    @staticmethod
    def current_device():
        return 0


# TODO: mv to cpp side
torch.utils.rename_privateuse1_backend("nupu")
torch._register_device_module("nupu", _NupuMod())


# https://pytorch.org/tutorials/prototype/python_extension_autoload.html
def _autoload():
    pass
