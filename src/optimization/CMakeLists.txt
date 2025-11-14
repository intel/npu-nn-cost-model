# Copyright © 2024 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
# LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
# is subject to the terms and conditions of the software license agreements for the Software Package,
# which may also include notices, disclaimers, or license terms for third party or open source software
# included in or with the Software Package, and your use indicates your acceptance of all such terms.
# Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
# Software Package for additional details.

# src/optimization/

if(VPUNN_BUILD_SHARED_LIB AND NOT JS_BUILD)
    add_library(vpunn_optimization SHARED)
else()
    add_library(vpunn_optimization STATIC)
endif()

# Add source files
target_sources(vpunn_optimization 
    PRIVATE 
        workload_optimization.cpp 
        tiler.cpp 
        splits.cpp
)

target_include_directories(vpunn_optimization
    PRIVATE 
        $<BUILD_INTERFACE:${COST_MODEL_ROOT_DIR}/include>
        $<BUILD_INTERFACE:${COST_MODEL_BINARY_DIR}/include>
)

target_link_libraries(vpunn_optimization
    PRIVATE
        vpunn_common_settings
        flatbuffers
        $<$<BOOL:${VPUNN_BUILD_HTTP_CLIENT}>:vpunn_http_client>
)

add_dependencies(vpunn_optimization vpunn_cpp_schema)

if(VPUNN_BUILD_SHARED_LIB)
    add_dependencies(vpunn_python vpunn_optimization)
endif()