#!/bin/bash

################################################################################

TORCHGEN_PATH=$(python -c "import torchgen; from pathlib import Path; print(Path(torchgen.__file__).parent, end='')")
echo "TORCHGEN_PATH: $TORCHGEN_PATH"

ln -s $TORCHGEN_PATH/packaged/ATen/templates $PWD/yaml
ln -s $TORCHGEN_PATH/packaged/ATen/native $PWD/yaml

################################################################################

pushd src
args=(
  $PWD/../yaml/gen_nupu_functions.py
  --source_yaml ../yaml/nupu_functions.yaml
  --output_dir gen_functions
)
python "${args[@]}"
popd

################################################################################
