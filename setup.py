from __future__ import annotations

import os
from pathlib import Path

from setuptools import find_packages, setup

ROOT_DIR = Path(__file__).absolute().parent


if __name__ == "__main__":
    if not os.path.exists(f"{ROOT_DIR}/torch_nupu/_C.so"):
        # TODO: support building using pip
        raise Exception("must run cmake building before `pip install`")

    setup(
        packages=find_packages(exclude=("test",)),
        package_data={"torch_nupu": ["*.dll", "*.dylib", "*.so", "lib/*", "py.typed"]},
        entry_points={
            "torch.backends": [
                "torch_nupu = torch_nupu:_autoload",
            ],
        },
    )
