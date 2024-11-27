# torch_nupu

## build

```bash
# run torchgen
bash yaml/torchgen.sh

# check `CMakePresets.json` and use a right preset
cmake --preset demo
cmake --build $PWD/build -t install
```

```bash
pip install --no-build-isolation -e . -vvv
```
