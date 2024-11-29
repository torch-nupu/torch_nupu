set(C10_NUPU_SRCS
    XPUCachingAllocator.cpp
    XPUFunctions.cpp
    XPUStream.cpp
    impl/XPUGuardImpl.cpp
)
set(C10_NUPU_HEADERS
    XPUCachingAllocator.h
    XPUDeviceProp.h
    XPUException.h
    XPUFunctions.h
    XPUMacros.h
    XPUStream.h
    impl/XPUGuardImpl.h
)

add_library(c10_nupu ${C10_NUPU_SRCS} ${C10_NUPU_HEADERS})
target_compile_options(c10_nupu PRIVATE "-DC10_XPU_BUILD_MAIN_LIB")
target_include_directories(c10_nupu PUBLIC ${SYCL_INCLUDE_DIR})
# NOTE: must not use headers from torch
target_include_directories(c10_nupu PUBLIC SYSTEM BEFORE ${CMAKE_CURRENT_SOURCE_DIR}/../../)
target_link_libraries(c10_nupu PUBLIC c10 ${SYCL_LIBRARY})
# TODO: do we need it?
# target_compile_options(c10_nupu PRIVATE "-fvisibility=hidden")
