# Copyright © 2024 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
# LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
# is subject to the terms and conditions of the software license agreements for the Software Package,
# which may also include notices, disclaimers, or license terms for third party or open source software
# included in or with the Software Package, and your use indicates your acceptance of all such terms.
# Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
# Software Package for additional details.

# src/

# All component subdirectories
add_subdirectory(core)
add_subdirectory(kernels)
add_subdirectory(inference)
add_subdirectory(optimization)
add_subdirectory(shave)
add_subdirectory(schema)
add_subdirectory(vpu)

if (VPUNN_BUILD_HTTP_CLIENT)
    add_subdirectory(http_client)
endif()

# Main unified cost model library
if(VPUNN_BUILD_SHARED_LIB)
    add_library(npu_costmodel SHARED)
else()
    add_library(npu_costmodel STATIC)
endif()

# Public API headers (exposed to users)
target_include_directories(npu_costmodel
    PUBLIC
        $<BUILD_INTERFACE:${COST_MODEL_ROOT_DIR}/include>
        $<BUILD_INTERFACE:${COST_MODEL_BINARY_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
        $<BUILD_INTERFACE:${FLATBUFFERS_SRC_DIR}/include>
    PRIVATE
        ${COST_MODEL_ROOT_DIR}/include/core
        ${COST_MODEL_ROOT_DIR}/include/inference
        ${COST_MODEL_ROOT_DIR}/include/kernels
        ${COST_MODEL_ROOT_DIR}/include/vpu
)

target_sources(npu_costmodel
    PRIVATE 
        vpu_cost_model.cpp
        vpu_layer_cost_model.cpp
        vpu_dma_cost_model.cpp
        vpu_network_cost_model.cpp
        vpu_shave_cost_model.cpp
)

target_link_libraries(npu_costmodel    
    PRIVATE 
        vpunn_common_settings
        flatbuffers
        vpunn_core
        vpunn_kernels
        vpunn_inference
        vpunn_optimization
        vpunn_shave
        vpunn_vpu
        $<$<BOOL:${VPUNN_BUILD_HTTP_CLIENT}>:vpunn_http_client>
)

add_dependencies(npu_costmodel vpunn_cpp_schema)

# Alias for consistent naming
add_library(VPUNN::npu_costmodel ALIAS npu_costmodel)

# Python bindings (must be after npu_costmodel is defined)
if(ENABLE_PYTHON_BINDING OR SKBUILD)
    add_subdirectory(python)
endif()

if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR) # Only configure installation for VPUNN standalone build
    include(GNUInstallDirs)
    include(CMakePackageConfigHelpers)

    # Install the main interface library
    install(TARGETS npu_costmodel
        EXPORT VPUNNTargets 
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} # where header files are expected to be found
    )

    set(VPUNN_INSTALL_TARGETS
        vpunn_common_settings
        vpunn_core
        vpunn_kernels
        vpunn_inference
        vpunn_optimization
        vpunn_shave
        vpunn_vpu
        flatbuffers # flatbuffers is required by vpunn libs in the installation
        ProjectConfig # flatbuffers dependency
    )

    if(TARGET blas)
        list(APPEND VPUNN_INSTALL_TARGETS blas)
    endif()

    if(VPUNN_BUILD_HTTP_CLIENT)
        list(APPEND VPUNN_INSTALL_TARGETS vpunn_http_client httplib nlohmann_json)
    endif()

    if(TARGET cache_app)
        install(TARGETS cache_app
            EXPORT VPUNNTargets
            RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}/$<$<BOOL:${CMAKE_CONFIGURATION_TYPES}>:$<CONFIG>>" # specifies where executable files (runtime artifacts) should be installed
            OPTIONAL # the install will not fail if target doesn't exist at install time
        )
    endif()

    # For multi-config generators, install to config-specific subdirectories
    # For single-config generators, install directly to lib/bin directories
    install(TARGETS
        ${VPUNN_INSTALL_TARGETS}
        EXPORT VPUNNTargets
        LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}/$<$<BOOL:${CMAKE_CONFIGURATION_TYPES}>:$<CONFIG>>"
        ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}/$<$<BOOL:${CMAKE_CONFIGURATION_TYPES}>:$<CONFIG>>"
        RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}/$<$<BOOL:${CMAKE_CONFIGURATION_TYPES}>:$<CONFIG>>"
    )

    # Install public headers
    # source headers
    install(DIRECTORY ${COST_MODEL_ROOT_DIR}/include/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        FILES_MATCHING 
            PATTERN "*.hpp" 
            PATTERN "*.h"
            PATTERN "*.hxx"
            PATTERN "*.hh"
    )

    # generated headers
    if(EXISTS ${COST_MODEL_BINARY_DIR}/include)
        install(DIRECTORY ${COST_MODEL_BINARY_DIR}/include/
            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} 
            FILES_MATCHING 
                PATTERN "*.hpp" 
                PATTERN "*.h"
                PATTERN "*.hxx"
                PATTERN "*.hh"
        )
    endif()

    # install FlatBuffers headers (if needed by public API)
    #if(EXISTS ${FLATBUFFERS_SRC_DIR}/include)
    #    install(DIRECTORY ${FLATBUFFERS_SRC_DIR}/include/
    #        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    #        FILES_MATCHING PATTERN "*.h"
    #    )
    #endif()

    # Generate version file (for find_package)
    write_basic_package_version_file(
        "${CMAKE_CURRENT_BINARY_DIR}/VPUNNConfigVersion.cmake"
        VERSION 1.0.0
        COMPATIBILITY SameMajorVersion
    )

    # Create package config template
    set(VPUNN_CONFIG_TEMPLATE "${CMAKE_CURRENT_BINARY_DIR}/VPUNNConfig.cmake.in")
    file(WRITE ${VPUNN_CONFIG_TEMPLATE}
        "@PACKAGE_INIT@

        include(CMakeFindDependencyMacro)
        # Find required dependencies
        find_dependency(Threads)

        # Include the targets file
        include(\"\${CMAKE_CURRENT_LIST_DIR}/VPUNNTargets.cmake\")

        # Only expose the main target
        if(NOT TARGET VPUNN::npu_costmodel)
            message(FATAL_ERROR \"VPUNN::npu_costmodel target not found\")
        endif()

        check_required_components(VPUNN)
        "
    )

    # Generate config file
    # Standard path for CMake config files are in lib/cmake/<PackageName>, i.e. CMAKE_INSTALL_LIBDIR
    configure_package_config_file(
        "${VPUNN_CONFIG_TEMPLATE}"
        "${CMAKE_CURRENT_BINARY_DIR}/VPUNNConfig.cmake" 
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/VPUNN
    )

    # Install config files
    install(FILES 
        "${CMAKE_CURRENT_BINARY_DIR}/VPUNNConfig.cmake" 
        "${CMAKE_CURRENT_BINARY_DIR}/VPUNNConfigVersion.cmake"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/VPUNN 
    )

    # Install the targets file (with VPUNN namespace)
    install(EXPORT VPUNNTargets 
        FILE VPUNNTargets.cmake
        NAMESPACE VPUNN:: 
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/VPUNN 
    )

    # Export targets for build tree usage
    export(EXPORT VPUNNTargets 
        FILE "${CMAKE_CURRENT_BINARY_DIR}/VPUNNTargets.cmake" 
        NAMESPACE VPUNN::
    )

    # Register package in user's package registry 
    export(PACKAGE VPUNN)

    # Print instalation summary 
    message(STATUS "VPUNN installation configured:")
    message(STATUS "  - Main target: VPUNN::npu_costmodel")
    message(STATUS "  - Install prefix: ${CMAKE_INSTALL_PREFIX}")
    message(STATUS "  - Library dir:    ${CMAKE_INSTALL_LIBDIR}") 
    message(STATUS "  - Include dir:    ${CMAKE_INSTALL_INCLUDEDIR}")
    message(STATUS "  - Config dir:     ${CMAKE_INSTALL_LIBDIR}/cmake/VPUNN")

endif()
