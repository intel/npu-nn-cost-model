# Copyright © 2024 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
# LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
# is subject to the terms and conditions of the software license agreements for the Software Package,
# which may also include notices, disclaimers, or license terms for third party or open source software
# included in or with the Software Package, and your use indicates your acceptance of all such terms.
# Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
# Software Package for additional details.

include_directories(${CMAKE_CURRENT_BINARY_DIR}/../../include)
include_directories(${FLATBUFFERS_SRC_DIR}/include)

add_library(vpunn_vpu ./validation/checker_utils.cpp ./validation/data_dpu_operation.cpp)
# add_dependencies(vpunn_vpu cpp_schema)
# target_link_libraries(vpunn_vpu flatbuffers)

if(VPUNN_BUILD_SHARED_LIB)
    add_library(vpunn_vpu_shared SHARED ./validation/checker_utils.cpp ./validation/data_dpu_operation.cpp)
    # add_dependencies(vpunn_vpu_shared cpp_schema)
    # target_link_libraries(vpunn_vpu_shared flatbuffers)
    set_target_properties(vpunn_vpu_shared PROPERTIES OUTPUT_NAME vpunn_vpu)

    # add_dependencies(python vpunn_vpu_shared)
endif()