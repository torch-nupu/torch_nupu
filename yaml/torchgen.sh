#!/bin/bash

################################################################################

TORCH_PATH=$(python -c "import torch; from pathlib import Path; print(Path(torch.__file__).parent, end='')")
echo "TORCH_PATH: $TORCH_PATH"

# TODO: use `site-packages/torchgen/packaged/ATen/templates` for torch release
ln -s $TORCH_PATH/../aten/src/ATen/templates $PWD/yaml

################################################################################

args=(
  -m torchgen.gen
  --source-path yaml
  --install_dir src/gen_native
  # TODO: support aoti
  --aoti_install_dir yaml
  --per-operator-headers
  --backend_whitelist=PrivateUse1
)
python "${args[@]}"

# add extra include
rm -f src/gen_native/ops_privateuse1.h
find src/gen_native/ops/*_native.h -printf "%f\n" | xargs -d "\n" -I{} echo "#include \"ops/{}\"" >>src/gen_native/ops_privateuse1.h
sed -i -e \
  '0,/<ATen\/ops/s//"ops_privateuse1.h"\n#include <ATen\/ops/' \
  src/gen_native/RegisterPrivateUse1.cpp

################################################################################

pushd src
args=(
  -m torchgen.gen_backend_stubs
  --source_yaml ../yaml/nupu_functions.yaml
  --output_dir gen_functions
)
python "${args[@]}"
popd

################################################################################
