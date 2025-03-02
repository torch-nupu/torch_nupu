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
    return torch_nupu._C._get_current_stream(device)


def _get_device_properties(device=None):
    # TODO: support API
    pass


def get_device_properties(device=None):
    return _get_device_properties(current_device())
