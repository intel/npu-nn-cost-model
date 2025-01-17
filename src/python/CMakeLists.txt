# Copyright © 2024 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
# LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
# is subject to the terms and conditions of the software license agreements for the Software Package,
# which may also include notices, disclaimers, or license terms for third party or open source software
# included in or with the Software Package, and your use indicates your acceptance of all such terms.
# Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
# Software Package for additional details.

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

if (${GENERATE_PYTHON_BINDING})
    include(ExternalProject)
    message(STATUS "Generating Python bindings source ${PYTHON_BINDING_CPP}")

    # We use binder to generate the python bindings for VPUNN
    ExternalProject_Add(binder
        GIT_REPOSITORY https://github.com/RosettaCommons/binder.git
        GIT_TAG v1.3.0
        SOURCE_DIR ${EXTERNAL_INSTALL_LOCATION}/binder/src
        BINARY_DIR ${EXTERNAL_INSTALL_LOCATION}/binder/build
        INSTALL_DIR ${EXTERNAL_INSTALL_LOCATION}/binder/install
        CMAKE_ARGS
        -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
        -DBINDER_ENABLE_TEST=OFF)

    add_custom_command(
	OUTPUT ${PYTHON_BINDING_CPP}
        DEPENDS
        binder inferenceStatic vpunn_optimization
        ${CMAKE_CURRENT_SOURCE_DIR}/binding.h
        ${CMAKE_CURRENT_SOURCE_DIR}/binding.cfg # We also want to rebuild the bindings if we change the config file
        COMMAND ${EXTERNAL_INSTALL_LOCATION}/binder/install/bin/binder
        --annotate-includes
        --bind=VPUNN
        -p ${EXTERNAL_INSTALL_LOCATION}
        --prefix=${CMAKE_CURRENT_SOURCE_DIR}
        --flat
        --single-file
        --root-module=VPUNN
        --include-pybind11-stl
        --config ${CMAKE_CURRENT_SOURCE_DIR}/binding.cfg
        ${CMAKE_CURRENT_SOURCE_DIR}/binding.h
        --
        -x c++
        -std=c++17
        -I${CMAKE_CURRENT_SOURCE_DIR}
        -I${FLATBUFFERS_SRC_DIR}/include
        -I${CMAKE_VPUNN_BINARY_DIR}/include
        -I${pybind11_SOURCE_DIR}/include
        -I${CMAKE_CURRENT_SOURCE_DIR}/../../include)

    add_custom_target(generated-bindings-code EXCLUDE_FROM_ALL
        DEPENDS ${PYTHON_BINDING_CPP})
endif()

pybind11_add_module(vpunn_python_bindings MODULE
    ${PYTHON_BINDING_CPP}
    binding.h)

add_dependencies(vpunn_python_bindings python_schema)

# The python bindings depend on some internal libraries
target_link_libraries(vpunn_python_bindings PRIVATE inferenceStatic vpunn_optimization)

# The name of the shared library must match the module name to keep python happy
set_target_properties(vpunn_python_bindings PROPERTIES OUTPUT_NAME "VPUNN")

add_custom_target(vpunn_python_create_init_file ALL
    COMMAND echo "from .VPUNN import VPUNN as bindings; del VPUNN" > ${CMAKE_VPUNN_BINARY_DIR}/lib/__init__.py
    VERBATIM
)

include(GNUInstallDirs)

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
        FILES ${CMAKE_VPUNN_BINARY_DIR}/lib/__init__.py
        COMPONENT bindings
        DESTINATION ${VPUNN_PYTHON_INSTALLATION_FOLDER}
    )

    install(
        DIRECTORY ${CMAKE_VPUNN_BINARY_DIR}/include/VPUNN_SCHEMA/
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
