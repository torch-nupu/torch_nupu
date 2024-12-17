import importlib

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
