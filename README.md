# torch_nupu

## build

install pytorch cpu from https://pytorch.org/

```bash
# currently needs nightly version
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu
# stable
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

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
