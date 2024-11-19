import torch


def test_copy():
    a = torch.rand(10, device="cpu")
    b = a.to("nupu").to("cpu")
    # b = a.to("privateuseone").to("cpu")
    assert torch.equal(a, b)


def test_add_tensor():
    a = torch.rand(10, device="cpu").to("nupu")
    b = torch.rand(10, device="cpu").to("nupu")
    # `add.Tensor` just return `a` currently
    c = a + b
    assert torch.equal(a.to("cpu"), c.to("cpu"))
