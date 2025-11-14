# Copyright © 2024 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
# LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
# is subject to the terms and conditions of the software license agreements for the Software Package,
# which may also include notices, disclaimers, or license terms for third party or open source software
# included in or with the Software Package, and your use indicates your acceptance of all such terms.
# Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
# Software Package for additional details.

# src/vpu/

if(VPUNN_BUILD_SHARED_LIB AND NOT JS_BUILD)
    add_library(vpunn_vpu SHARED)
else()
    add_library(vpunn_vpu STATIC)
endif()

target_include_directories(vpunn_vpu
    PRIVATE 
        $<BUILD_INTERFACE:${COST_MODEL_ROOT_DIR}/include>
        # Accessing generated files that energy_impl.cpp depends on
        $<BUILD_INTERFACE:${COST_MODEL_BINARY_DIR}/include>
)

target_sources(vpunn_vpu
    PRIVATE 
        validation/checker_utils.cpp 
        validation/data_dpu_operation.cpp
        energy_impl.cpp
        layer.cpp
)

target_link_libraries(vpunn_vpu
    PRIVATE 
        vpunn_common_settings
        flatbuffers
        vpunn_inference
        $<$<BOOL:${VPUNN_BUILD_HTTP_CLIENT}>:vpunn_http_client>
)

add_dependencies(vpunn_vpu vpunn_cpp_schema)