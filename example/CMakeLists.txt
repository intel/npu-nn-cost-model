# Copyright © 2024 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
# LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
# is subject to the terms and conditions of the software license agreements for the Software Package,
# which may also include notices, disclaimers, or license terms for third party or open source software
# included in or with the Software Package, and your use indicates your acceptance of all such terms.
# Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
# Software Package for additional details.

include_directories(${CMAKE_BINARY_DIR}/include)
include_directories(${FLATBUFFERS_SRC_DIR}/include)

add_executable(workload_mode_selection workload_mode_selection.cpp)
target_link_libraries(workload_mode_selection inferenceStatic)

add_executable(vpunn_profile cost_model.profiler.cpp)
target_link_libraries(vpunn_profile inferenceStatic)
