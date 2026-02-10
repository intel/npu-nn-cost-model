# Copyright © 2024 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
# LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
# is subject to the terms and conditions of the software license agreements for the Software Package,
# which may also include notices, disclaimers, or license terms for third party or open source software
# included in or with the Software Package, and your use indicates your acceptance of all such terms.
# Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
# Software Package for additional details.

# src/python/

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 QUIET CONFIG)

if(NOT pybind11_FOUND)
    message(WARNING "Could not find pybind11, handling dependency with FetchContent")

    include(FetchContent)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11
        GIT_TAG v2.10.0
    )

    FetchContent_GetProperties(pybind11)

    if(NOT pybind11_POPULATED)
        FetchContent_Populate(pybind11)
        add_subdirectory(${pybind11_SOURCE_DIR} ${pybind11_BINARY_DIR})
    endif()
endif()

set(PYTHON_BINDING_CPP ${CMAKE_CURRENT_SOURCE_DIR}/VPUNN.cpp)

pybind11_add_module(vpunn_python_bindings MODULE
    ${PYTHON_BINDING_CPP}
    binding.h
)

target_include_directories(vpunn_python_bindings
    PRIVATE 
        $<BUILD_INTERFACE:${COST_MODEL_ROOT_DIR}/include>
        $<BUILD_INTERFACE:${COST_MODEL_BINARY_DIR}/include>
        $<BUILD_INTERFACE:${FLATBUFFERS_SRC_DIR}/include>
)

# The python bindings depend on some internal libraries
target_link_libraries(vpunn_python_bindings 
    PRIVATE 
        vpunn_common_settings
        vpunn_inference 
        vpunn_optimization
        vpunn_vpu
        vpunn_shave
        flatbuffers
)

add_dependencies(vpunn_python_bindings vpunn_python_schema)

# The name of the shared library must match the module name to keep python happy
set_target_properties(vpunn_python_bindings 
    PROPERTIES 
    OUTPUT_NAME "VPUNN"
)

add_custom_target(vpunn_python_create_init_file ALL
    COMMAND echo "from .VPUNN import DPUWorkload, VPUCostModel, DMAWorkload, DMACostModel_NPU27, DMACostModel_NPU40_50, DMANNWorkload_NPU27, DMANNWorkload_NPU40_50, VPUTensor; VPUNN_lib  = VPUNN" > ${COST_MODEL_BINARY_DIR}/lib/__init__.py
    VERBATIM
)

if(SKBUILD)
    find_package(PythonExtensions REQUIRED)
    set(VPUNN_PYTHON_INSTALLATION_FOLDER ${PYTHON_RELATIVE_SITE_PACKAGES_DIR}/VPUNN)
    set(VPUNN_SCHEMA_PYTHON_INSTALLATION_FOLDER ${PYTHON_RELATIVE_SITE_PACKAGES_DIR}/VPUNN_SCHEMA)

    install(
        TARGETS vpunn_python_bindings
        COMPONENT bindings
        DESTINATION ${VPUNN_PYTHON_INSTALLATION_FOLDER}
    )

    install(
        FILES ${COST_MODEL_BINARY_DIR}/lib/__init__.py
        COMPONENT bindings
        DESTINATION ${VPUNN_PYTHON_INSTALLATION_FOLDER}
    )

    install(
        DIRECTORY ${COST_MODEL_BINARY_DIR}/include/VPUNN_SCHEMA/
        COMPONENT bindings
        DESTINATION ${VPUNN_SCHEMA_PYTHON_INSTALLATION_FOLDER}
    )

    install(
        DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../../models
        COMPONENT bindings
        DESTINATION ${PYTHON_RELATIVE_SITE_PACKAGES_DIR}/vpunn
    )

    install(
        DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../schema
        COMPONENT bindings
        DESTINATION ${PYTHON_RELATIVE_SITE_PACKAGES_DIR}/vpunn
    )

    add_custom_target(vpunn-install-bindings
        ${CMAKE_COMMAND}
        -DCMAKE_INSTALL_COMPONENT=bindings
        -P "${PROJECT_BINARY_DIR}/cmake_install.cmake"
        DEPENDS vpunn_python_bindings vpunn_python_create_init_file
    )
endif()
