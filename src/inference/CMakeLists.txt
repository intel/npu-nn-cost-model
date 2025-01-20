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

if(VPUNN_BUILD_SHARED_LIB)
    add_library(inference SHARED model.cpp ../core/logger.cpp)
    add_dependencies(inference cpp_schema)

    if(CBLAS_LIB STREQUAL "internal")
        target_link_libraries(inference flatbuffers kernels vpunn_optimization blas vpunn_vpu vpunn_shave)
    else()
        target_link_libraries(inference flatbuffers kernels vpunn_optimization vpunn_vpu vpunn_shave ${BLAS_LIBRARIES} ${BLAS_LINKER_FLAGS})
    endif()

    add_custom_target(python ALL
        DEPENDS python_schema inference
        COMMENT "Generating python files")
endif()

add_library(inferenceStatic STATIC model.cpp preprop_factory.cpp)
add_dependencies(inferenceStatic cpp_schema)

if(CBLAS_LIB STREQUAL "internal")
    target_link_libraries(inferenceStatic flatbuffers kernels vpunn_core vpunn_optimization blasStatic vpunn_vpu vpunn_shave)
else()
    set(BLAS_STATIC ON)
    target_link_libraries(inferenceStatic flatbuffers kernels vpunn_core vpunn_optimization vpunn_vpu vpunn_shave ${BLAS_LIBRARIES} ${BLAS_LINKER_FLAGS})
endif()

set_target_properties(inferenceStatic PROPERTIES OUTPUT_NAME inference)
