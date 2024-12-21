from __future__ import annotations

import importlib.util

import pytest
import torch

triton_nupu_not_found = True
spec = importlib.util.find_spec("triton")
if spec is not None:
    triton_nupu_not_found = False


@pytest.mark.skipif(triton_nupu_not_found, reason="")
def test_op_add():
    arg0 = torch.ones((3, 3))
    arg1 = arg0.to("nupu")
    op = torch.compile(torch.add)
    res0 = op(arg1, arg1)
    ref0 = torch.add(arg0, arg0)
    assert torch.equal(res0, ref0)


@pytest.mark.xfail(
    reason="it requires OCK supporting `OpCapability GroupNonUniformClustered`"
)
@pytest.mark.skipif(triton_nupu_not_found, reason="")
def test_op_mean():
    arg0 = torch.ones((3, 3))
    arg1 = arg0.to("nupu")
    op = torch.compile(torch.mean)
    res0 = op(arg1)
    ref0 = torch.mean(arg0)
    assert torch.equal(res0, ref0)
