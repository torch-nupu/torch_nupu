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
    return torch_nupu._C._get_current_stream(device or current_device())


class NupuDeviceProperties:
    name: str = "refsi"
    platform_name: str = "ComputeAortaOpenCL"
    type: str = "gpu"
    driver_version: str = "4.0"
    total_memory = 2048 * 1024 * 1024  # 2048MB
    max_compute_units: int = 1
    gpu_eu_count: int = 512
    gpu_subslice_count: int = 64
    max_work_group_size: int = 1024
    max_num_sub_groups: int = 1024
    sub_group_sizes: list[int] = [32]
    has_fp16: int = 1
    has_fp64: int = 1
    has_atomic64: int = 1


def _get_device_properties(device=None):
    # TODO: support parse from opencl api
    return NupuDeviceProperties()


def get_device_properties(device=None):
    return _get_device_properties(device or current_device())
