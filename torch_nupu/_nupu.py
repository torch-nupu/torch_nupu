from functools import lru_cache
from typing import Any
import torch_nupu._C  # noqa


# TODO: support torch.xpu.* for triton-xpu usage


def is_available():
    return True


def device_count():
    return 1


def current_device():
    return 0


def current_stream(device=None):
    streamdata = torch_nupu._C._xpu_getCurrentStream(device)
    return torch_nupu._C._XpuStreamBase(
        stream_id=streamdata[0], device_index=streamdata[1], device_type=streamdata[2]
    )


@lru_cache(None)
def get_device_capability(device=None) -> dict[str, Any]:
    props = get_device_properties(device)
    return {
        prop: getattr(props, prop)
        for prop in dir(props)
        if not prop.startswith(("__", "_pybind11_"))
    }


def get_device_properties(device=None):
    return torch_nupu._C._get_device_properties(device)
