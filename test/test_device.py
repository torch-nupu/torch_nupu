from __future__ import annotations

import torch


def test_device():
    import torch_nupu

    nupu_device = torch_nupu._C.nupu_device()
    assert nupu_device.type == "nupu"
    assert nupu_device.index == 0


def test_device_module():
    assert torch.nupu.is_available()
    assert torch.nupu.device_count() == 1
    assert torch.nupu.current_device() == 0
