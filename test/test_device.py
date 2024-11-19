import torch  # noqa: F401
import torch_nupu  # noqa: F401


def test_device():
    nupu_device = torch_nupu._C.nupu_device()
    assert nupu_device.type == "nupu"
    assert nupu_device.index == 0
