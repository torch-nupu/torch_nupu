cmake_minimum_required(VERSION 3.30.0 FATAL_ERROR)

project(torch_nupu CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# TODO: can we pack libsycl.so into wheel ?
set(RPATH_ORIGIN $ORIGIN)
set(CMAKE_INSTALL_RPATH "${RPATH_ORIGIN}")

option(WITH_TEST "building with cpp test" TRUE)

# FetchContent
include(fetch.cmake)
find_package(OpenCLHeaders REQUIRED)
find_package(OpenCLICDLoader REQUIRED)
find_package(OpenCLHeadersCpp REQUIRED)
set(OPENCL_LIBS OpenCL::Headers OpenCL::OpenCL OpenCL::HeadersCpp)
add_definitions("-DCL_TARGET_OPENCL_VERSION=300")
add_definitions("-DCL_HPP_ENABLE_EXCEPTIONS")

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG)

if(NOT TORCH_PATH)
  execute_process(
    COMMAND
    ${Python3_EXECUTABLE} -c "import torch; from pathlib import Path; print(Path(torch.__file__).parent, end='')"
    OUTPUT_VARIABLE
    TORCH_PATH)
  endif()
if(NOT TORCH_PATH)
  message(FATAL_ERROR "set TORCH_PATH failed.")
endif()
message("TORCH_PATH: ${TORCH_PATH}")
find_package(Torch REQUIRED PATHS "${TORCH_PATH}/share/cmake/Torch")
message("TORCH_LIBRARY: ${TORCH_LIBRARY}")
message("TORCH_INCLUDE_DIRS: ${TORCH_INCLUDE_DIRS}")

add_subdirectory(src)
add_subdirectory(torch_nupu)

if(WITH_TEST)
  add_subdirectory(test/cpp)
endif()
