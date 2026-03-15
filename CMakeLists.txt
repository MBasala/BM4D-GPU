cmake_minimum_required(VERSION 3.16)
project(bm4d-gpu)

if(POLICY CMP0146)
  cmake_policy(SET CMP0146 OLD)
endif()

# Add support for C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2")
set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
option(BUILD_CLI "Build cli (requires OpenCV)" ON)
option(BUILD_CLI_STRICT "Fail configure when BUILD_CLI is ON and OpenCV is unavailable" OFF)
option(BUILD_TESTS "Build unit-tests" OFF)
option(BUILD_PYTHON "Build Python bindings (requires pybind11 and Python)" OFF)

# BM4D library
find_package(CUDA REQUIRED)

# Enable it only if debugging
if(CMAKE_BUILD_TYPE STREQUAL "Debug" OR CMAKE_BUILD_TYPE STREQUAL
                                        "RelWithDebInfo")
  set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} "-g -G")
  set(CUDA_NVCC_FLAGS "-g -G")
endif()

list(APPEND CUDA_NVCC_FLAGS --std=c++17)

if(MSVC)
  add_compile_options(/Zc:preprocessor)
  list(APPEND CUDA_NVCC_FLAGS -Xcompiler=/Zc:preprocessor)
  # Match the dynamic CRT (/MD) used by OpenCV and the C++ targets.
  # FindCUDA defaults to /MT which causes LNK2038 RuntimeLibrary mismatches.
  list(APPEND CUDA_NVCC_FLAGS -Xcompiler=/MD)
endif()

# Autodetect cuda architecture on current platform and build for them
cuda_select_nvcc_arch_flags(ARCH_FLAGS Auto)
list(APPEND CUDA_NVCC_FLAGS ${ARCH_FLAGS})

include_directories(include)

set(BM4D-H
    include/bm4d-gpu/bm4d.h
    include/bm4d-gpu/helper_cuda.h
    include/bm4d-gpu/helper_string.h
    include/bm4d-gpu/kernels.cuh
    include/bm4d-gpu/parameters.h
    include/bm4d-gpu/stopwatch.hpp
    include/bm4d-gpu/utils.h)

set(BM4D-CXX src/bm4d.cpp)
set(BM4d-CU src/kernels.cu)

cuda_add_library(bm4d-gpu-lib ${BM4D-CXX} ${BM4d-CU} ${BM4D-H})
target_include_directories(bm4d-gpu-lib PUBLIC include)

# BM4D CLI
if(BUILD_CLI)
  if(WIN32 AND DEFINED OpenCV_DIR AND EXISTS "${OpenCV_DIR}/OpenCVConfig.cmake")
    if(NOT DEFINED OpenCV_ARCH AND CMAKE_SIZEOF_VOID_P EQUAL 8 AND EXISTS "${OpenCV_DIR}/x64")
      set(OpenCV_ARCH x64 CACHE STRING "OpenCV Windows pack architecture" FORCE)
    endif()

    if(MSVC AND (NOT OpenCV_RUNTIME OR OpenCV_RUNTIME STREQUAL ""))
      if(EXISTS "${OpenCV_DIR}/x64/vc17/OpenCVConfig.cmake")
        set(OpenCV_RUNTIME vc17 CACHE STRING "OpenCV Windows pack runtime" FORCE)
      elseif(EXISTS "${OpenCV_DIR}/x64/vc16/OpenCVConfig.cmake")
        set(OpenCV_RUNTIME vc16 CACHE STRING "OpenCV Windows pack runtime" FORCE)
      endif()
    endif()
  endif()

  set(BM4D_GPU_OPENCV_COMPONENTS core imgcodecs videoio highgui)
  find_package(OpenCV QUIET CONFIG COMPONENTS ${BM4D_GPU_OPENCV_COMPONENTS})

  if(OpenCV_FOUND)
    set(CLI-CXX src/allreader.cpp src/main.cpp)
    set(CLI-H include/bm4d-gpu/allreader.h)
    cuda_add_executable(bm4d-gpu ${CLI-CXX} ${CLI-H})
    target_include_directories(bm4d-gpu PUBLIC include PRIVATE ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(bm4d-gpu bm4d-gpu-lib ${OpenCV_LIBRARIES})
  elseif(BUILD_CLI_STRICT)
    message(FATAL_ERROR
            "BUILD_CLI is ON, but a compatible OpenCV package was not found.\n"
            "Set OpenCV_DIR to a compatible OpenCVConfig.cmake directory or disable the CLI with -DBUILD_CLI=OFF.")
  else()
    message(WARNING
            "OpenCV was not found or is incompatible with the active toolchain.\n"
            "Skipping the optional bm4d-gpu CLI target and continuing with bm4d-gpu-lib only.\n"
            "If you need the CLI on Windows, point OpenCV_DIR to a compatible OpenCV build; the official Windows pack often mismatches MinGW or newer MSVC toolsets.")
  endif()
endif()

if(BUILD_TESTS)
  add_subdirectory(tests)
endif()

# Python bindings
if(BUILD_PYTHON)
  find_package(Python REQUIRED COMPONENTS Interpreter Development)
  find_package(pybind11 REQUIRED)

  pybind11_add_module(bm4d_gpu src/pybind_bm4d.cpp)
  target_link_libraries(bm4d_gpu PRIVATE bm4d-gpu-lib)
  target_include_directories(bm4d_gpu PRIVATE include)
endif()
