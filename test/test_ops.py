from __future__ import annotations

import torch


def test_copy():
    a = torch.rand(10, device="cpu")
    b = a.to("nupu").to("cpu")
    # b = a.to("privateuseone").to("cpu")
    assert torch.equal(a, b)
