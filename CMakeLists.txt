# Copyright © 2023 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
# LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
# is subject to the terms and conditions of the software license agreements for the Software Package,
# which may also include notices, disclaimers, or license terms for third party or open source software
# included in or with the Software Package, and your use indicates your acceptance of all such terms.
# Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
# Software Package for additional details.

cmake_minimum_required(VERSION 3.12)

set(CMAKE_CXX_STANDARD 14 CACHE STRING "C++ standard to conform to")

# set the project name
project(VPUNN)

option(VPUNN_BUILD_SHARED_LIB "build shared library" ON)
option(VPUNN_BUILD_EXAMPLES "build examples" ON)
option(VPUNN_BUILD_TESTS "build tests" ON)
option(VPUNN_ENABLE_LOGGING "enable logging" OFF)
option(ENABLE_PYTHON_BINDING "Build the python bindings" OFF)
option(GENERATE_PYTHON_BINDING "Generate the python bindings code" OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(EXTERNAL_INSTALL_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/external)
set(CMAKE_VPUNN_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
find_package(Threads REQUIRED)

include_directories(${EXTERNAL_INSTALL_LOCATION}/include ${CMAKE_VPUNN_BINARY_DIR}/include)

if(NOT CMAKE_BUILD_TYPE)
    message(AUTHOR_WARNING "-- Defaulting build type to Release because CMAKE_BUILD_TYPE is not set")
    set(CMAKE_BUILD_TYPE Release)
endif()

if(DEFINED ENV{EMSDK} AND DEFINED ENV{EMSCRIPTEN})
    set(JS_BUILD ON)
else()
    set(JS_BUILD OFF)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Coverage")
    add_compile_definitions(NO_PROFILING_ALLOWED)
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        message(STATUS "-- Enable Code Coverage")
        add_compile_options(--coverage -g -O0)
        add_link_options(--coverage -g -O0)
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
        message(STATUS "-- Enable Code Coverage")
        add_compile_options(--coverage -g -O0)
        add_link_options(--coverage -g -O0)
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
        message(AUTHOR_WARNING "-- Building with MSVC, cannot enable code coverage")
    else()
        message(AUTHOR_WARNING "-- Building with unrecognised compiler, cannot enable code coverage")
    endif()
endif()

if(NOT ${JS_BUILD})
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        # add_compile_options(--gcc-toolchain=/usr/intel/pkgs/gcc/9.2.0)
    endif()

    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "GNU")
        # lots of warnings and all warnings as errors
        message(STATUS "Setting GCC/Clang specific flags for the entire build")
        add_compile_options(-Wall -Wextra -Werror -pedantic -Wdouble-promotion -Wfloat-conversion)
        set(CMAKE_CXX_FLAGS_DEBUG "-g")
        set(CMAKE_CXX_FLAGS_RELEASE "-O3")
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
        message(STATUS "Setting Visual Studio specific flags for the entire build")
        add_compile_options(/W4 /WX)
        add_link_options(/WX)

        # Disable dll build
        set(VPUNN_BUILD_SHARED_LIB OFF)
    else()
        message(AUTHOR_WARNING "-- Building with unrecognised compiler, not setting any specific flags")
    endif()
endif()

include_directories(include)

add_subdirectory(src)
add_subdirectory(js)

if(VPUNN_BUILD_EXAMPLES)
    add_subdirectory(example)
endif()

if(VPUNN_BUILD_TESTS)
    add_subdirectory(tests/cpp)
endif()

# Check that the models are LFS expanded
file(GLOB MODEL_FILES ${PROJECT_SOURCE_DIR}/models/*.vpunn)

function(get_file_size filename var)
    file(READ "${filename}" content HEX)
    string(LENGTH "${content}" content_length)
    math(EXPR content_length "${content_length} / 2")
    set(${var} ${content_length} PARENT_SCOPE)
endfunction()

foreach(FILENAME ${MODEL_FILES})
    get_file_size(${FILENAME} MODEL_SIZE)
    message(STATUS "VPUNN model ${FILENAME} size ${MODEL_SIZE}")

    if(${MODEL_SIZE} LESS 200)
        message(FATAL_ERROR "model ${FILENAME} hasn't been expanded with Git LFS (size = ${MODEL_SIZE}). Please fetch the file using `git lfs pull`")
    endif()
endforeach()
