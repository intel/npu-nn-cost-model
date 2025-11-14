# Copyright © 2024 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
# LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
# is subject to the terms and conditions of the software license agreements for the Software Package,
# which may also include notices, disclaimers, or license terms for third party or open source software
# included in or with the Software Package, and your use indicates your acceptance of all such terms.
# Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
# Software Package for additional details.

# src/core/

add_library(vpunn_core STATIC 
	logger.cpp
)

target_include_directories(vpunn_core
    PRIVATE 
        $<BUILD_INTERFACE:${COST_MODEL_ROOT_DIR}/include>
        $<BUILD_INTERFACE:${COST_MODEL_BINARY_DIR}/include>
)

target_link_libraries(vpunn_core 
    PRIVATE
		vpunn_common_settings
)
