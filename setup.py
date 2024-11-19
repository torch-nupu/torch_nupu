from pathlib import Path

from setuptools import find_packages, setup

from torch.utils.cpp_extension import BuildExtension, CppExtension


# TODO: mv to project.toml

ROOT_DIR = Path(__file__).absolute().parent
CSRS_DIR = ROOT_DIR / "torch_nupu/csrc"


if __name__ == "__main__":
    sources = list(CSRS_DIR.glob("*.cpp"))

    # Note that we always compile with debug info
    ext_modules = [
        CppExtension(
            name="torch_nupu._C",
            sources=sorted(str(s) for s in sources),
            include_dirs=[CSRS_DIR],
            extra_compile_args={"cxx": ["-g", "-Wall", "-Werror"]},
        )
    ]

    setup(
        name="torch_nupu",
        version="0.1",
        author="nuway.ai",
        description="torch-nupu from nuway.ai",
        packages=find_packages(exclude=("test",)),
        package_data={"torch_nupu": ["*.dll", "*.dylib", "*.so"]},
        install_requires=[
            "torch",
        ],
        ext_modules=ext_modules,
        # target os: ubuntu-22.04+
        python_requires=">=3.10",
        cmdclass={
            "build_ext": BuildExtension.with_options(no_python_abi_suffix=True),
        },
        # entry_points={
        #     # TODO: support `_autoload`
        #     "torch.backends": [
        #         "torch_nupu = torch_nupu:_autoload",
        #     ],
        # },
    )
