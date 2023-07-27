# Copyright © 2023 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
# LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
# is subject to the terms and conditions of the software license agreements for the Software Package,
# which may also include notices, disclaimers, or license terms for third party or open source software
# included in or with the Software Package, and your use indicates your acceptance of all such terms.
# Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
# Software Package for additional details.

include_directories(${CMAKE_BINARY_DIR}/include)
include_directories(${FLATBUFFERS_SRC_DIR}/include)

# Set JS binding application properties: set bind, output type, etc. Runs only if Emscripten is enabled.
if(DEFINED ENV{EMSDK} AND DEFINED ENV{EMSCRIPTEN})
    find_package(Git)

    if(GIT_FOUND)
        execute_process(
            COMMAND ${GIT_EXECUTABLE} describe --tags --abbrev=0
            OUTPUT_VARIABLE VPUNN_RELEASE_TAG
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
    else()
        set(VPUNN_RELEASE_TAG "v1.0.0")
    endif(GIT_FOUND)

    message(STATUS "VPUNN_RELEASE_TAG: ${VPUNN_RELEASE_TAG}")

    # add flatbuffer libraries
    include(FetchContent)
    FetchContent_Declare(
        flatbuffers
        GIT_REPOSITORY https://github.com/google/flatbuffers.git
        GIT_TAG v1.12.0
    )
    FetchContent_MakeAvailable(flatbuffers)

    set_property(DIRECTORY ${flatbuffers_SOURCE_DIR} PROPERTY EXCLUDE_FROM_ALL YES)

    function(GENERATE_SHV_KERNEL_BINDINGS KERNEL_TYPE KERNEL_NAME)
        string(TOUPPER ${KERNEL_NAME} KERNEL_NAME_UPPER)
        set(SW_LAYER_ENUMERATIONS "${SW_LAYER_ENUMERATIONS} ${KERNEL_NAME_UPPER}, " PARENT_SCOPE)
        set(SW_LAYER_BINDINGS "${SW_LAYER_BINDINGS} .value(\"${KERNEL_NAME_UPPER}\", SWActivationFunction::${KERNEL_NAME_UPPER}) \\\n" PARENT_SCOPE)

        if(${KERNEL_TYPE} STREQUAL "ACTIVATION")
            set(SW_LAYER_SWITCH_CASE "${SW_LAYER_SWITCH_CASE}  case SWActivationFunction::${KERNEL_NAME_UPPER}:  return std::make_unique<SHV${KERNEL_NAME}>(device, inT, outT); \\\n" PARENT_SCOPE)
        endif()
    endfunction()

    file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/../src/shave/layers_perf.csv" config)
    STRING(REPLACE " " "" config "${config}")
    STRING(REPLACE "\n" ";" config "${config}")
    set(SW_LAYER_ENUMERATIONS "{")

    set(SKIP_HEADER TRUE)

    foreach(LINE ${config})
        if(SKIP_HEADER)
            set(SKIP_HEADER FALSE)
        else()
            STRING(REPLACE "," ";" PARAMETERS ${LINE})
            GENERATE_SHV_KERNEL_BINDINGS(${PARAMETERS})
        endif()
    endforeach()

    set(SW_LAYER_ENUMERATIONS "${SW_LAYER_ENUMERATIONS} __size }")

    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/vpunn_bind.cpp.in"
        "${CMAKE_CURRENT_BINARY_DIR}/vpunn_bind.cpp"
    )

    add_executable(vpunn_bind "${CMAKE_CURRENT_BINARY_DIR}/vpunn_bind.cpp")
    target_link_libraries(vpunn_bind inferenceStatic)

    set_target_properties(
        vpunn_bind
        PROPERTIES OUTPUT_NAME vpunn_bind
        SUFFIX ".js"
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/dist)

    set_target_properties(vpunn_bind PROPERTIES LINK_FLAGS "--bind")

    target_link_options(
        vpunn_bind
        PUBLIC
        -v
        --bind
        -O3
        -sEXCEPTION_DEBUG=1
        -sDEMANGLE_SUPPORT=1
        -sNO_DISABLE_EXCEPTION_CATCHING=1
        -sALLOW_MEMORY_GROWTH=1

        # -sFORCE_FILESYSTEM=1
        --embed-file ${CMAKE_SOURCE_DIR}/models@models
        -msimd128

        # -sWASM_BIGINT=1
        # -sMODULARIZE=1
        # -sEXPORT_ES6=1
        -sEXPORT_NAME=VPUNN)

    add_custom_command(
        TARGET vpunn_bind POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E remove -f
        ${CMAKE_BINARY_DIR}/dist/vpunn-*.tgz)

    add_custom_command(
        TARGET vpunn_bind POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_SOURCE_DIR}/js/package.json
        ${CMAKE_BINARY_DIR}/dist/package.json)

    add_custom_command(
        TARGET vpunn_bind POST_BUILD
        COMMAND sed -i "s/{{VPUNN_RELEASE_TAG}}/${VPUNN_RELEASE_TAG}/"
        ${CMAKE_BINARY_DIR}/dist/package.json)

    add_custom_target(vpunn_js ALL
        COMMAND npm pack
        DEPENDS vpunn_bind
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/dist
        COMMENT "Generating javascript module")
endif(DEFINED ENV{EMSDK} AND DEFINED ENV{EMSCRIPTEN})