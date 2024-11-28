import torch


def test_copy():
    a = torch.rand(10, device="cpu")
    b = a.to("nupu").to("cpu")
    # b = a.to("privateuseone").to("cpu")
    assert torch.equal(a, b)


def test_add():
    a = torch.rand(10, device="cpu").to("nupu")
    b = torch.rand(10, device="cpu").to("nupu")
    # `add.Tensor` just return `a` currently
    c = a + b
    assert torch.equal(a.to("cpu"), c.to("cpu"))


def test_abs():
    a = torch.empty(10, device="nupu")
    torch.ops.aten.abs.default(a)


def test_log_sigmoid():
    a = torch.empty(10, device="nupu")
    torch.ops.aten.log_sigmoid.default(a)
