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

# Python bindings
if(ENABLE_PYTHON_BINDING OR SKBUILD)
    add_subdirectory(python)
endif()

# Combine all
add_library(npu_costmodel INTERFACE)

target_include_directories(npu_costmodel
    INTERFACE 
        $<BUILD_INTERFACE:${COST_MODEL_ROOT_DIR}/include>
        $<BUILD_INTERFACE:${COST_MODEL_BINARY_DIR}/include>
        $<BUILD_INTERFACE:${FLATBUFFERS_SRC_DIR}/include>
)

target_link_libraries(npu_costmodel
    INTERFACE
        vpunn_core
        vpunn_kernels
        vpunn_inference
        vpunn_optimization
        vpunn_shave
        vpunn_vpu
        flatbuffers
        $<$<BOOL:${ENABLE_PYTHON_BINDING}>:vpunn_python_bindings>
        $<$<BOOL:${VPUNN_BUILD_HTTP_CLIENT}>:vpunn_http_client>
)

# Alias for consistent naming
add_library(VPUNN::npu_costmodel ALIAS npu_costmodel)

if(NOT JS_BUILD)
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
        flatbuffers
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
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
            OPTIONAL
        )
    endif()

    install(TARGETS 
        ${VPUNN_INSTALL_TARGETS}
        EXPORT VPUNNTargets
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        OPTIONAL
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
