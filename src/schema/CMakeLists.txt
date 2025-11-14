# Copyright © 2024 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
# LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
# is subject to the terms and conditions of the software license agreements for the Software Package,
# which may also include notices, disclaimers, or license terms for third party or open source software
# included in or with the Software Package, and your use indicates your acceptance of all such terms.
# Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
# Software Package for additional details.

# src/schema/

# Define generated header paths
set(GENERATED_INCLUDE_DIR "${COST_MODEL_BINARY_DIR}/include")
set(VPUNN_GENERATED_HEADER "${GENERATED_INCLUDE_DIR}/vpunn_generated.h")
set(CYCLES_CACHE_GENERATED_HEADER "${GENERATED_INCLUDE_DIR}/cycles_cache_generated.h")

# Ensure output directory exists
file(MAKE_DIRECTORY ${GENERATED_INCLUDE_DIR})

add_custom_command(
    OUTPUT 
        ${VPUNN_GENERATED_HEADER}
        ${CYCLES_CACHE_GENERATED_HEADER}
    COMMAND 
        ${FLATC_COMMAND} --cpp -o ${GENERATED_INCLUDE_DIR} "vpunn.fbs" "cycles_cache.fbs"
    DEPENDS 
        vpunn.fbs 
        cycles_cache.fbs
        ${flatc_TARGET}
    WORKING_DIRECTORY 
        "${PROJECT_SOURCE_DIR}/src/schema"
    COMMENT 
        "Generating c++ schema files"
)

# C++ schema target
add_custom_target(vpunn_cpp_schema
    DEPENDS 
        ${VPUNN_GENERATED_HEADER}
        ${CYCLES_CACHE_GENERATED_HEADER}
)

# Python schema target
add_custom_target(vpunn_python_schema ALL
    COMMAND 
        ${FLATC_COMMAND} --python -o ${GENERATED_INCLUDE_DIR} 
        ${PROJECT_SOURCE_DIR}/src/schema/vpunn.fbs
        ${PROJECT_SOURCE_DIR}/src/schema/cycles_cache.fbs
    DEPENDS
        ${flatc_TARGET}
    WORKING_DIRECTORY 
        "${PROJECT_SOURCE_DIR}/python"
    COMMENT 
        "Generating python schema files"
)
