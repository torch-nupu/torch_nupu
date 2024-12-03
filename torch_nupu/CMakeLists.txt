set(TORCH_NUPU_SRCS)
list(APPEND TORCH_NUPU_SRCS csrc/stub.c)
list(APPEND TORCH_NUPU_SRCS csrc/_C.cpp)

list(APPEND TORCH_NUPU_SRCS csrc/Device.cpp)
list(APPEND TORCH_NUPU_SRCS csrc/Hooks.cpp)
list(APPEND TORCH_NUPU_SRCS csrc/Allocator.cpp)
list(APPEND TORCH_NUPU_SRCS csrc/Ops.cpp)

list(APPEND TORCH_NUPU_SRCS csrc/xpu/Event.cpp)
list(APPEND TORCH_NUPU_SRCS csrc/xpu/Module.cpp)
list(APPEND TORCH_NUPU_SRCS csrc/xpu/Stream.cpp)

pybind11_add_module(_C ${TORCH_NUPU_SRCS} WITHOUT_SOABI)
target_include_directories(_C PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(_C PUBLIC torch_nupu_ops)

# -lc10 -ltorch -ltorch_cpu -ltorch_python
target_link_directories(_C PUBLIC "${TORCH_PATH}/lib")
target_link_libraries(_C PUBLIC c10 torch torch_cpu torch_python)

install(TARGETS _C LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX})
