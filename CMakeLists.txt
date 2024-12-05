cmake_minimum_required(VERSION 3.30.0 FATAL_ERROR)

project(torch_nupu CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# TODO: can we pack libsycl.so into wheel ?
set(RPATH_ORIGIN $ORIGIN)
set(CMAKE_INSTALL_RPATH "${RPATH_ORIGIN}")

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules)
find_package(SYCLToolkit REQUIRED)
if(NOT SYCL_FOUND)
  message(FATAL_ERROR "SYCLToolkit is not found.")
endif()
message("SYCL_LIBRARY: ${SYCL_LIBRARY}")
# SYCL_COMPILER_VERSION > `20250000`
set(SYCL_COMPILER_VERSION 20990000)
add_definitions("-DSYCL_COMPILER_VERSION=20990000")

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
