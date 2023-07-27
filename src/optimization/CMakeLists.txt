# Copyright © 2023 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
# LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
# is subject to the terms and conditions of the software license agreements for the Software Package,
# which may also include notices, disclaimers, or license terms for third party or open source software
# included in or with the Software Package, and your use indicates your acceptance of all such terms.
# Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
# Software Package for additional details.

include_directories(${CMAKE_CURRENT_BINARY_DIR}/../../include)
include_directories(${FLATBUFFERS_SRC_DIR}/include)

add_library(vpunn_optimization workload_optimization.cpp tiler.cpp splits.cpp)
add_dependencies(vpunn_optimization cpp_schema)
target_link_libraries(vpunn_optimization flatbuffers)

if(VPUNN_BUILD_SHARED_LIB)
    add_library(vpunn_optimization_shared SHARED workload_optimization.cpp tiler.cpp splits.cpp)
    add_dependencies(vpunn_optimization_shared cpp_schema)
    target_link_libraries(vpunn_optimization_shared flatbuffers)
    set_target_properties(vpunn_optimization_shared PROPERTIES OUTPUT_NAME vpunn_optimization)

    add_dependencies(python vpunn_optimization_shared)
endif()