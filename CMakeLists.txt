# Copyright © 2024 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
# LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
# is subject to the terms and conditions of the software license agreements for the Software Package,
# which may also include notices, disclaimers, or license terms for third party or open source software
# included in or with the Software Package, and your use indicates your acceptance of all such terms.
# Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
# Software Package for additional details.

# root

# Use modern CMake version
cmake_minimum_required(VERSION 3.20)

# Set the project name and language
project(VPUNN LANGUAGES CXX)

# Set default build type 
if(NOT CMAKE_BUILD_TYPE)
    message(STATUS "-- Defaulting build type to 'Release' because CMAKE_BUILD_TYPE is not set")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo" "Coverage")
endif()

message(STATUS "-- Building for ${CMAKE_BUILD_TYPE}")

# Project options

# impacts the blas library , if ON and linux will enable vector instructions by -msse[N]
# on win(MSVC) will define USE_SIMD
# if target architecture does not support it, deactivate it and enable manually the required optimizations 
# (for vector instructions if runtime is important) at least for blas
option(VPUNN_ENABLE_VECTOR_INSTRUCTIONS "use vector instructions" ON)
option(VPUNN_BUILD_SHARED_LIB "build shared library" OFF)
option(VPUNN_BUILD_EXAMPLES "build examples" ON)
option(VPUNN_BUILD_APPS "build apps" OFF)
option(VPUNN_BUILD_TESTS "build tests" ON)
option(VPUNN_ENABLE_LOGGING "enable logging" OFF)
option(ENABLE_PYTHON_BINDING "Build the python bindings" OFF)
option(GENERATE_PYTHON_BINDING "Generate the python bindings code" OFF)
option(VPUNN_BUILD_HTTP_CLIENT "Build support for cost provider http service" OFF)
option(VPUNN_OPT_LEGACY_ZTILING "Use legacy ZTiling mechanism" ON)
option(VPUNN_OPT_LEGACY_DMA_TH_4 "Use legacy Theoretical DMA for Gen4" OFF)
option(ENABLE_FLATBUFFERS_INSTALLATION "Install flatbuffers and its dependency ProjectConfig" ON)

# Detect JavaScript build env
if(DEFINED ENV{EMSDK} AND DEFINED ENV{EMSCRIPTEN})
    set(JS_BUILD ON)
else()
    set(JS_BUILD OFF)
endif()

# Global settings
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(COST_MODEL_ROOT_DIR         ${CMAKE_CURRENT_SOURCE_DIR})
set(COST_MODEL_BINARY_DIR       ${CMAKE_CURRENT_BINARY_DIR})
set(EXTERNAL_INSTALL_LOCATION   ${CMAKE_CURRENT_BINARY_DIR}/external)

# Output directory setup
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)

# Required packages
# Flatbuffers is included in the root because subdirectories
# and directories of subdirectories needs this target and 
# certain flatbuffers variables.
include(cmake/FlatbuffersSetup.cmake)

# Common settings library
# Consumer libs should always link this as PRIVATE to avoid transitive propagation.
add_library(vpunn_common_settings STATIC)

target_sources(vpunn_common_settings PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/common_settings_dummy.cpp)

# Apply C++ standard requirement to this target and (optionally) to dependents
target_compile_features(vpunn_common_settings PUBLIC cxx_std_17)

# Compile definitions (PUBLIC so direct link consumers inherit them; they can choose PRIVATE linkage)
target_compile_definitions(vpunn_common_settings PUBLIC
    INTEL_EMBARGO_NPU5    # INTEL_EMBARGO_NPU5_Content
    INTEL_EMBARGO_NPU5020 # INTEL_EMBARGO_NPU5020_Content
    INTEL_EMBARGO_NPU6    # INTEL_EMBARGO_NPU6_Content
    INTEL_EMBARGO_COMMON  # INTEL_EMBARGO_COMMON_Content
)

if(VPUNN_OPT_LEGACY_DMA_TH_4)
    target_compile_definitions(vpunn_common_settings PUBLIC VPUNN_OPT_LEGACY_DMA_TH_4)
    message(STATUS "-- Enable Legacy Gen4 DMA Theoretical")
endif()

# Conditional compile definitions
if (VPUNN_BUILD_HTTP_CLIENT)
    target_compile_definitions(vpunn_common_settings PUBLIC 
        VPUNN_BUILD_HTTP_CLIENT
    )
    message(STATUS "-- Enable HTTP service cost provider ")
endif()

if(VPUNN_OPT_LEGACY_ZTILING)
    target_compile_definitions(vpunn_common_settings PUBLIC
        VPUNN_OPT_LEGACY_ZTILING
    )
    message(STATUS "-- Enable Legacy ZTiling (intratile) ")
endif()

# Coverage build configuration
if(CMAKE_BUILD_TYPE STREQUAL "Coverage")
    target_compile_definitions(vpunn_common_settings PUBLIC
        NO_PROFILING_ALLOWED
    )
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
        message(STATUS "-- Enable Code Coverage")
        target_compile_options(vpunn_common_settings PUBLIC
                --coverage -g -O0
            )
        target_link_options(vpunn_common_settings PUBLIC
                --coverage -g -O0
            )
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
        message(WARNING "-- Building with MSVC, cannot enable code coverage")
    else()
        message(WARNING "-- Building with unrecognised compiler, cannot enable code coverage")
    endif()
endif()

# Non JS builds
if(NOT JS_BUILD)
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
        # lots of warnings and all warnings as errors
        message(STATUS "Setting GCC/Clang specific flags for common settings")
        target_compile_options(vpunn_common_settings PUBLIC
                -Wall -Wextra -Werror -pedantic -Wdouble-promotion -Wfloat-conversion
                $<$<CONFIG:Debug>:-g>
                $<$<CONFIG:Release>:-O3>
            )
        elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
            message(STATUS "Setting Visual Studio specific flags for common settings")
        target_compile_options(vpunn_common_settings PUBLIC
                # Enable /bigobj for debug build type with MSVC - remove after vpunn headers are cleaned up
                $<$<CONFIG:Debug>:/bigobj>
                /W4 /WX
            )
        # TODO figure out why it propagates to VPUx even though it is linked as PRIVATE in other libs
        # and re-enable
        # target_link_options(vpunn_common_settings PUBLIC
        #         /WX
        #     )

        # Disable dll build
        set(VPUNN_BUILD_SHARED_LIB OFF)
    else()
        message(WARNING "-- Building with unrecognised compiler, not setting any specific flags")
    endif()
endif()

# Add subdirectories
add_subdirectory(src)
add_subdirectory(js)

# Optional subdirectories based on build options
if(VPUNN_BUILD_EXAMPLES)
    add_subdirectory(example)
endif()

if(VPUNN_BUILD_APPS)
    add_subdirectory(apps/cache_app)
endif()

if(VPUNN_BUILD_TESTS)
    add_subdirectory(tests/cpp)
endif()

# Model file validation function
# Ensures that all .vpunn model files in the models/ directory 
# have been properly downloaded via Git LFS
function(validate_model_files)
    file(GLOB MODEL_FILES ${PROJECT_SOURCE_DIR}/models/*.vpunn)

    foreach(FILENAME ${MODEL_FILES})
        FILE(READ "${FILENAME}" FILE_CONTENT HEX)
        STRING(LENGTH "${FILE_CONTENT}" CONTENT_LENGTH)
        MATH(EXPR MODEL_SIZE "${CONTENT_LENGTH} / 2")

        message(STATUS "VPUNN model ${FILENAME} size ${MODEL_SIZE} bytes")

        if(${MODEL_SIZE} LESS 200)
            message(FATAL_ERROR 
                "model ${FILENAME} hasn't been expanded with Git LFS (size = ${MODEL_SIZE}). "
                "Please fetch the file using `git lfs pull`")
        endif()
    endforeach()
endfunction()

# Validate model files
validate_model_files()
