# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -fsycl-targets=spir64")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --offload-new-driver")

# add_subdirectory(c10/nupu)

set(ATEN_NUPU_SRCS)

list(APPEND ATEN_NUPU_SRCS nupu/Allocator.cpp)
list(APPEND ATEN_NUPU_SRCS nupu/Device.cpp)
list(APPEND ATEN_NUPU_SRCS nupu/Hooks.cpp)
list(APPEND ATEN_NUPU_SRCS nupu/Ops.cpp)
list(APPEND ATEN_NUPU_SRCS nupu/Copy.cpp)

# # TODO: split into multi libs?
# list(APPEND ATEN_NUPU_SRCS ATen/nupu/detail/XPUHooks.cpp)
# list(APPEND ATEN_NUPU_SRCS ATen/nupu/CachingHostAllocator.cpp)
# list(APPEND ATEN_NUPU_SRCS ATen/nupu/XPUContext.cpp)
# list(APPEND ATEN_NUPU_SRCS ATen/nupu/XPUGeneratorImpl.cpp)

# # list(APPEND ATEN_NUPU_SRCS ATen/native/nupu/sycl/AbsKernel.cpp)
# # list(APPEND ATEN_NUPU_SRCS ATen/native/nupu/sycl/ActivationLogSigmoidKernels.cpp)
# # list(APPEND ATEN_NUPU_SRCS ATen/native/nupu/sycl/BinaryKernels.cpp)
# list(APPEND ATEN_NUPU_SRCS ATen/native/nupu/sycl/CopyKernel.cpp)
# # list(APPEND ATEN_NUPU_SRCS ATen/native/nupu/sycl/ResizeKernel.cpp)
# list(APPEND ATEN_NUPU_SRCS ATen/native/nupu/sycl/UnaryComplexKernels.cpp)

# list(APPEND ATEN_NUPU_SRCS ATen/native/nupu/Copy.cpp)
# # list(APPEND ATEN_NUPU_SRCS ATen/native/nupu/Resize.cpp)

# # list(APPEND ATEN_NUPU_SRCS ATen/native/nupu/P1Ops.cpp)

# `torchgen.gen_backend_stubs`
list(APPEND ATEN_NUPU_SRCS gen_functions/RegisterPrivateUse1.cpp)

add_library(torch_nupu SHARED ${ATEN_NUPU_SRCS})

# TODO(nupu): can we rm it?
# target_compile_definitions(torch_nupu PRIVATE "-DC10_USING_CUSTOM_GENERATED_MACROS")

# target_include_directories(torch_nupu PRIVATE ${SYCL_INCLUDE_DIR})
target_include_directories(torch_nupu PRIVATE ${TORCH_INCLUDE_DIRS})
target_include_directories(torch_nupu PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(torch_nupu ${TORCH_LIBRARY})
# target_link_libraries(torch_nupu ${SYCL_LIBRARY})
# target_link_libraries(torch_nupu c10_nupu)
target_link_libraries(torch_nupu ${OPENCL_LIBS})

install(TARGETS torch_nupu)
