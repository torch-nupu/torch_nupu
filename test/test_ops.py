import torch
import torch_nupu  # noqa: F401


def test_copy():
    a = torch.rand(10, device="cpu")
    nupu_device = torch_nupu._C.nupu_device()
    b = a.to(device=nupu_device).to("cpu")
    assert torch.equal(a, b)


def test_add_tensor():
    nupu_device = torch_nupu._C.nupu_device()
    a = torch.rand(10, device="cpu").to(device=nupu_device)
    b = torch.rand(10, device="cpu").to(device=nupu_device)
    # `add.Tensor` just return `a` currently
    c = a + b
    assert torch.equal(a.to("cpu"), c.to("cpu"))
