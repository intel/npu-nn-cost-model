# Copyright © 2022 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
# LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
# is subject to the terms and conditions of the software license agreements for the Software Package,
# which may also include notices, disclaimers, or license terms for third party or open source software
# included in or with the Software Package, and your use indicates your acceptance of all such terms.
# Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
# Software Package for additional details.

function(GENERATE_SHV_KERNEL KERNEL_TYPE KERNEL_NAME)
    set(KERNEL_NAME "SHV${KERNEL_NAME}")
    set(KERNEL_TYPE "SHV_${KERNEL_TYPE}_KERNEL")

    if(${ARGC} EQUAL 2)
        set(KERNEL_EFFICIENCY 1.0)
        set(KERNEL_LATENCY 0)
    elseif(${ARGC} EQUAL 4)
        set(KERNEL_EFFICIENCY ${ARGV2})
        set(KERNEL_LATENCY ${ARGV3})
    else()
        message(FATAL_ERROR "Wrong number of parameters in GENERATE_SHV_KERNEL function")
    endif()

    message(STATUS "Generate SHV Kernel ${KERNEL_NAME} with efficiency ${KERNEL_EFFICIENCY} bytes/cycle and latency ${KERNEL_LATENCY} cycles")
    set(GEN_HEADER_FILE "vpu/shave/${KERNEL_NAME}.h")
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/layer.h.in"
        "${CMAKE_VPUNN_BINARY_DIR}/include/${GEN_HEADER_FILE}"
    )

    set(GLOBAL_INCLUDE_FILE "${GLOBAL_INCLUDE_FILE}#include \"${GEN_HEADER_FILE}\"\n" PARENT_SCOPE)
endfunction()

file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/layers_perf.csv" config)
STRING(REPLACE "null" "" config "${config}")
STRING(REPLACE " " "" config "${config}")
STRING(REPLACE "\n" ";" config "${config}")

set(SKIP_HEADER TRUE)
foreach(LINE ${config})
    if (SKIP_HEADER)
        set(SKIP_HEADER FALSE)
    else()
        STRING(REPLACE "," ";" PARAMETERS ${LINE})
        GENERATE_SHV_KERNEL(${PARAMETERS})
    endif()
endforeach()

file(WRITE "${CMAKE_VPUNN_BINARY_DIR}/include/vpu/shave/layers.h" ${GLOBAL_INCLUDE_FILE})
