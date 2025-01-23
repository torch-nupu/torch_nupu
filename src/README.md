# notes

torch-xpu-ops
https://github.com/torch-nupu/torch-xpu-ops/tree/dev

pytorch
https://github.com/torch-nupu/pytorch/tree/dev

NOTE: Always update sub modules when updating from upstream

```bash
git submodule update --remote torch-xpu-ops
git submodule update --remote pytorch
```

## dir mapping

`torch_nupu/csrc/xpu` - `pytorch/torch/csrc/xpu`

`src/c10/nupu` - `pytorch/c10/xpu`
`src/ATen/nupu` - `pytorch/aten/src/ATen/xpu`

`src/ATen/native/nupu` - `torch-xpu-ops/src/ATen/native/xpu`
`src/ATen/native/nupu/sycl` - `torch-xpu-ops/src/ATen/native/xpu/sycl`
`src/comm` - `torch-xpu-ops/src/comm`

## dir diff

when using VSCode, install ext
https://marketplace.visualstudio.com/items?itemName=moshfeu.compare-folders

```bash
COMPARE_FOLDERS=DIFF code torch_nupu/csrc/xpu pytorch/torch/csrc/xpu

COMPARE_FOLDERS=DIFF code src/c10/nupu pytorch/c10/xpu
COMPARE_FOLDERS=DIFF code src/ATen/nupu pytorch/aten/src/ATen/xpu

COMPARE_FOLDERS=DIFF code src/ATen/native/nupu torch-xpu-ops/src/ATen/native/xpu
COMPARE_FOLDERS=DIFF code src/ATen/native/nupu/sycl torch-xpu-ops/src/ATen/native/xpu/sycl
COMPARE_FOLDERS=DIFF code src/comm torch-xpu-ops/src/comm
```

## dir sync

NOTE: always need deal conflict by hand

```bash
yes | cp -rf pytorch/torch/csrc/xpu/* torch_nupu/csrc/xpu/

# yes | cp -rf pytorch/c10/xpu/* src/c10/nupu/
yes | cp -rf pytorch/aten/src/ATen/xpu/* src/ATen/nupu/

# src/ATen/native/nupu torch-xpu-ops/src/ATen/native/xpu
# src/ATen/native/nupu/sycl torch-xpu-ops/src/ATen/native/xpu/sycl
# src/comm torch-xpu-ops/src/comm
```
