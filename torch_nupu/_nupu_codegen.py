from torch._inductor.codegen.wrapper import PythonWrapperCodegen
from torch._inductor.codegen.common import DeviceOpOverrides
from torch._inductor.codegen.triton import TritonScheduling


# TODO: support more API


class NupuWrapperCodegen(PythonWrapperCodegen):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class NupuScheduling(TritonScheduling):
    pass


class NupuDeviceOpOverrides(DeviceOpOverrides):
    def import_get_raw_stream_as(self, name: str) -> str:
        return f"""
def {name}(device_index):
    return torch.nupu.current_stream(device_index).sycl_queue
"""
