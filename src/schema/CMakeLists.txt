# Copyright © 2024 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
# LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
# is subject to the terms and conditions of the software license agreements for the Software Package,
# which may also include notices, disclaimers, or license terms for third party or open source software
# included in or with the Software Package, and your use indicates your acceptance of all such terms.
# Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
# Software Package for additional details.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(DEFINED flatc_COMMAND AND DEFINED flatc_TARGET)
    # Use flatc target from vpux-plugin
    message(STATUS "FLATBUFFERS:Use flatc target from vpux-plugin")
else()
    message(STATUS "FLATBUFFERS: have to solve them locally, nothing present")
    if(CMAKE_GENERATOR MATCHES "Visual Studio" OR NOT(DEFINED ENV{EMSDK} AND DEFINED ENV{EMSCRIPTEN}))
        message(STATUS "FLATBUFFERS: Have to fetch the flatbuffers!")
        include(FetchContent)

        set(FLATBUFFERS_BUILD_TESTS OFF CACHE BOOL "" FORCE)
        set(FLATBUFFERS_INSTALL OFF CACHE BOOL "" FORCE)
        set(FLATBUFFERS_BUILD_FLATC ON CACHE BOOL "" FORCE)

        FetchContent_Declare(
            flatbuffers
            GIT_REPOSITORY https://github.com/google/flatbuffers.git
            GIT_TAG v1.12.0
        )

        set_property(DIRECTORY ${flatbuffers_SOURCE_DIR} PROPERTY EXCLUDE_FROM_ALL YES)

        get_property(FLATBUFFERS_COMPILE_OPTIONS DIRECTORY ${flatbuffers_SOURCE_DIR} PROPERTY COMPILE_OPTIONS)

        # The flatbuffers library does not build with these options on
        foreach(flag IN ITEMS "-Werror" "-Wfloat-conversion" "-Wdouble-promotion")
            string(REPLACE "${flag}" "" FLATBUFFERS_COMPILE_OPTIONS "${FLATBUFFERS_COMPILE_OPTIONS}")
        endforeach()

        set_property(DIRECTORY ${flatbuffers_SOURCE_DIR} PROPERTY COMPILE_OPTIONS "${FLATBUFFERS_COMPILE_OPTIONS}")

        FetchContent_MakeAvailable(flatbuffers)

        if(NOT MSVC)
            if ((CMAKE_CXX_COMPILER_ID STREQUAL "Clang") AND (CMAKE_CXX_COMPILER_VERSION VERSION_LESS  13))
                # no options , clang 10 not working 
            else()
                 # clang larger or any GCC
                target_compile_options(flatbuffers PRIVATE -Wno-suggest-override)
                target_compile_options(flatc PRIVATE -Wno-suggest-override)
            endif()


            if ((CMAKE_CXX_COMPILER_ID STREQUAL "Clang") AND (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 13))
                target_compile_options(flatbuffers PRIVATE -Wno-unused-but-set-variable)
                target_compile_options(flatc PRIVATE -Wno-unused-but-set-variable)
            endif()
        endif()

        set(FLATBUFFERS_SRC_DIR ${flatbuffers_SOURCE_DIR} PARENT_SCOPE)

        set(flatc_COMMAND "flatc")
        set(flatc_TARGET "flatc")

        set(flatbuffers_INCLUDE_DIRECTORIES ${flatbuffers_SOURCE_DIR}/include PARENT_SCOPE)
        message(STATUS "FLATBUFFERS: command and target were set!")
    else()
        message(STATUS "FLATBUFFERS:Expected FLATC is present!")
        find_program(FLATC flatc)
        message(STATUS "flatc binary found ${FLATC}")

        if(NOT FLATC)
            message(FATAL_ERROR "Impossible to find flatc binary... needed for JS binding")
        endif()

        set(flatc_COMMAND ${FLATC})
    endif()
endif()

add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/../../include/vpunn_generated.h" "${CMAKE_CURRENT_BINARY_DIR}/../../include/cycles_cache_generated.h"
    COMMAND ${flatc_COMMAND} --cpp -o "${CMAKE_CURRENT_BINARY_DIR}/../../include" "vpunn.fbs" "cycles_cache.fbs"
    DEPENDS "vpunn.fbs" "cycles_cache.fbs"
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/src/schema"
    COMMENT "Generating c++ schema files")

add_custom_target(cpp_schema
    DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/../../include/vpunn_generated.h" "${CMAKE_CURRENT_BINARY_DIR}/../../include/cycles_cache_generated.h")

add_custom_target(python_schema ALL
    COMMAND ${flatc_COMMAND} --python -o "${CMAKE_CURRENT_BINARY_DIR}/../../include" "../src/schema/vpunn.fbs" "../src/schema/cycles_cache.fbs"
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/python"
    COMMENT "Generating python schema files")

if(flatc_TARGET)
    message(STATUS "FLATBUFFERS: The flatc_TARGET exists, specifying dependencies")
    add_dependencies(cpp_schema ${flatc_TARGET})
    add_dependencies(python_schema ${flatc_TARGET})
endif()
