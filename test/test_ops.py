import torch
import torch_nupu  # noqa: F401


def test_copy():
    a = torch.rand(10, device="cpu")
    nupu_device = torch_nupu._C.nupu_device()
    # TODO: `aten::_copy_from`
    # a.to(device=nupu_device)
