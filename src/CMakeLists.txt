set(ATEN_NUPU_SRCS)

list(APPEND ATEN_NUPU_SRCS nupu/Allocator.cpp)
list(APPEND ATEN_NUPU_SRCS nupu/Device.cpp)
list(APPEND ATEN_NUPU_SRCS nupu/Hooks.cpp)
list(APPEND ATEN_NUPU_SRCS nupu/Ops.cpp)
list(APPEND ATEN_NUPU_SRCS nupu/Copy.cpp)

# `torchgen.gen_backend_stubs`
list(APPEND ATEN_NUPU_SRCS gen_functions/RegisterPrivateUse1.cpp)

add_library(torch_nupu SHARED ${ATEN_NUPU_SRCS})

target_include_directories(torch_nupu PRIVATE ${TORCH_INCLUDE_DIRS})
target_include_directories(torch_nupu PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(torch_nupu ${TORCH_LIBRARY})
target_link_libraries(torch_nupu ${OPENCL_LIBS})

install(TARGETS torch_nupu)
