from __future__ import annotations

import torch
from torch._dynamo.device_interface import DeviceInterface

# TODO: support more API


class NupuDeviceInterface(DeviceInterface):
    class Worker(DeviceInterface.Worker):
        @staticmethod
        def set_device(device: int) -> None:
            pass

        @staticmethod
        def current_device() -> int:
            return 0

        @staticmethod
        def get_device_properties(device=None):
            return torch.nupu.get_device_properties(device)

    @staticmethod
    def current_device() -> int:
        return 0

    @staticmethod
    def set_device(device) -> None:
        pass

    @staticmethod
    def device_count() -> int:
        return 1

    @staticmethod
    def maybe_exchange_device(device: int) -> int:
        assert (
            device == 0
        ), f"Only device index 0 is supported, tried to set index to {device}"
        return 0  # previous device is always 0

    @staticmethod
    def exchange_device(device: int) -> int:
        assert (
            device == 0
        ), f"Only device index 0 is supported, tried to set index to {device}"
        return 0  # previous device is always 0

    @staticmethod
    def synchronize(device) -> None:
        pass

    @staticmethod
    def is_available() -> bool:
        return True

    @staticmethod
    def get_device_properties(device=None):
        return torch.nupu.get_device_properties(device)

    @staticmethod
    def get_compute_capability(device) -> int:
        return 0
