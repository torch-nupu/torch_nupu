# TODO: deal rpath to `torch_cpu`
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:$(python -c "import torch, pathlib; print(pathlib.Path(torch.__file__).parent)")/lib"

# run cpp tests
torch_nupu/bin/test_ops
