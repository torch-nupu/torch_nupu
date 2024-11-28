set(CMAKE_CXX_FLAGS "-fsycl -fsycl-targets=spir64")

set(ATEN_NUPU_SRCS)
list(APPEND ATEN_NUPU_SRCS ATen/native/nupu/sycl/AbsKernel.cpp)
list(APPEND ATEN_NUPU_SRCS ATen/native/nupu/sycl/ActivationLogSigmoidKernels.cpp)
list(APPEND ATEN_NUPU_SRCS ATen/native/nupu/sycl/CopyKernel.cpp)
list(APPEND ATEN_NUPU_SRCS ATen/native/nupu/sycl/ResizeKernel.cpp)
list(APPEND ATEN_NUPU_SRCS ATen/native/nupu/sycl/UnaryComplexKernels.cpp)

list(APPEND ATEN_NUPU_SRCS ATen/native/nupu/Copy.cpp)
list(APPEND ATEN_NUPU_SRCS ATen/native/nupu/PoneOps.cpp)
list(APPEND ATEN_NUPU_SRCS ATen/native/nupu/Resize.cpp)

# `torchgen.gen`
list(APPEND ATEN_NUPU_SRCS gen_native/RegisterPrivateUse1.cpp)
# `torchgen.gen_backend_stubs`
list(APPEND ATEN_NUPU_SRCS gen_functions/RegisterPrivateUse1.cpp)

add_library(torch_nupu_ops SHARED ${ATEN_NUPU_SRCS})

# TODO(nupu): can we rm it?
target_compile_definitions(torch_nupu_ops PRIVATE "-DC10_USING_CUSTOM_GENERATED_MACROS")
# TODO(nupu): rm xpu
target_compile_definitions(torch_nupu_ops PRIVATE "-DUSE_XPU")

# include dpcpp headers for clangd
cmake_path(GET CMAKE_CXX_COMPILER PARENT_PATH DIR_DPCPP_BIN)
target_include_directories(torch_nupu_ops PRIVATE ${DIR_DPCPP_BIN}/../include)

target_include_directories(torch_nupu_ops PRIVATE ${TORCH_INCLUDE_DIRS})
target_include_directories(torch_nupu_ops PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(torch_nupu_ops ${TORCH_LIBRARY})

install(TARGETS torch_nupu_ops)
