# Copyright © 2024 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
# LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
# is subject to the terms and conditions of the software license agreements for the Software Package,
# which may also include notices, disclaimers, or license terms for third party or open source software
# included in or with the Software Package, and your use indicates your acceptance of all such terms.
# Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
# Software Package for additional details.

# src/inference/

# Create library

# Ensure that the shared library will be created
# only if VPUNN_BUILD_SHARED_LIB will be set ON
# and JS_BUILD OFF
if(VPUNN_BUILD_SHARED_LIB AND NOT JS_BUILD)
    add_library(vpunn_inference SHARED)
else()
    add_library(vpunn_inference STATIC)
endif()

# Add source files
target_sources(vpunn_inference 
    PRIVATE 
        model.cpp 
        preprop_factory.cpp 
        inference_execution_data.cpp
)

target_include_directories(vpunn_inference
    PRIVATE 
        $<BUILD_INTERFACE:${COST_MODEL_ROOT_DIR}/include>
        $<BUILD_INTERFACE:${COST_MODEL_BINARY_DIR}/include>
        $<BUILD_INTERFACE:${FLATBUFFERS_SRC_DIR}/include>
) 

# Link to common project settings
target_link_libraries(vpunn_inference 
    PRIVATE
        vpunn_common_settings
        vpunn_kernels
        vpunn_core
        flatbuffers
)

add_dependencies(vpunn_inference vpunn_cpp_schema)

if(VPUNN_BUILD_SHARED_LIB)
    add_custom_target(vpunn_python ALL 
        DEPENDS vpunn_python_schema vpunn_inference
        COMMENT "Generating python files"
    )
endif()