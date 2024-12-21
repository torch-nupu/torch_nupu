from __future__ import annotations

import torch  # noqa


def test_device_module():
    import torch.nupu  # noqa # type:ignore

    assert torch.nupu.is_available()
    assert torch.nupu.device_count() == 1
    assert torch.nupu.current_device() == 0
