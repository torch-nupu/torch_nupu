from __future__ import annotations

import threading
from functools import lru_cache
from typing import Any

import torch_nupu._C  # noqa

_initialized = False
_initialization_lock = threading.Lock()

# TODO: support more APIs
# TODO: support `_is_in_bad_fork`


def init():
    _lazy_init()


def _lazy_init():
    global _initialized
    with _initialization_lock:
        if not _initialized:
            torch_nupu._C.nupu_init()
            _initialized = True


def is_available():
    return True


def device_count():
    return 1


def current_device():
    return 0


def current_stream(device=None):
    if device is None:
        device = current_device()
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
    if device is None:
        return torch_nupu._C._get_device_properties(current_device())
    else:
        if not isinstance(device, int | str):
            device = device.index
        return torch_nupu._C._get_device_properties(device)
