# Copyright © 2022 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
# LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
# is subject to the terms and conditions of the software license agreements for the Software Package,
# which may also include notices, disclaimers, or license terms for third party or open source software
# included in or with the Software Package, and your use indicates your acceptance of all such terms.
# Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
# Software Package for additional details.

if(DEFINED flatc_COMMAND AND DEFINED flatc_TARGET)
    # Use flatc target from vpux-plugin
else()
    if(CMAKE_GENERATOR MATCHES "Visual Studio" OR NOT (DEFINED ENV{EMSDK} AND DEFINED ENV{EMSCRIPTEN}))
        include(FetchContent)

        FetchContent_Declare(
            flatbuffers
            GIT_REPOSITORY https://github.com/google/flatbuffers.git
            GIT_TAG        v1.12.0
        )

        set_property(DIRECTORY ${flatbuffers_SOURCE_DIR} PROPERTY EXCLUDE_FROM_ALL YES)

        get_property(FLATBUFFERS_COMPILE_OPTIONS DIRECTORY ${flatbuffers_SOURCE_DIR} PROPERTY COMPILE_OPTIONS)

        # The flatbuffers library does not build with these options on
        foreach(flag IN ITEMS "-Werror" "-Wfloat-conversion" "-Wdouble-promotion")
            string(REPLACE "${flag}" "" FLATBUFFERS_COMPILE_OPTIONS "${FLATBUFFERS_COMPILE_OPTIONS}")
        endforeach()

        set_property(DIRECTORY ${flatbuffers_SOURCE_DIR} PROPERTY COMPILE_OPTIONS "${FLATBUFFERS_COMPILE_OPTIONS}")

        FetchContent_MakeAvailable(flatbuffers)

        set(flatc_COMMAND "flatc")
        set(flatc_TARGET "flatc")
    else()
        find_program(FLATC flatc)
        message(STATUS "flatc binary found ${FLATC}")
        if(NOT FLATC)
            message(FATAL_ERROR "Impossible to find flatc binary... needed for JS binding")
        endif()

        set(flatc_COMMAND ${FLATC})
    endif()
endif()

add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/../../include/vpunn_generated.h"
                  COMMAND ${flatc_COMMAND} --cpp -o "${CMAKE_CURRENT_BINARY_DIR}/../../include" "vpunn.fbs"
                  DEPENDS "vpunn.fbs"
                  WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/src/schema"
                  COMMENT "Generating c++ schema files")

add_custom_target(cpp_schema
                  DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/../../include/vpunn_generated.h")

add_custom_target(python_schema ALL
                  COMMAND ${flatc_COMMAND} --python -o "${CMAKE_CURRENT_BINARY_DIR}/../../include" "../src/schema/vpunn.fbs"
                  WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/python"
                  COMMENT "Generating python schema files")

if(flatc_TARGET)
    add_dependencies(cpp_schema ${flatc_TARGET})
    add_dependencies(python_schema ${flatc_TARGET})
endif()
