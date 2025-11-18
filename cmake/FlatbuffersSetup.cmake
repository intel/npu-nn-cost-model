# Copyright © 2024 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
# LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
# is subject to the terms and conditions of the software license agreements for the Software Package,
# which may also include notices, disclaimers, or license terms for third party or open source software
# included in or with the Software Package, and your use indicates your acceptance of all such terms.
# Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
# Software Package for additional details.

# It will automatically download and integrate the FlatBuffers library
# set flags and target related to Flatbuffers that will be used 
# across different CMakeLists 

if(TARGET flatbuffers AND TARGET flatc)
    message(STATUS "FLATBUFFERS: flatbuffers and flatc targets already available")
    set(FLATC_COMMAND $<TARGET_FILE:flatc>)
    set(FLATC_TARGET flatc)
endif()

if(DEFINED FLATC_COMMAND AND DEFINED FLATC_TARGET)
    message(STATUS "FLATBUFFERS: Using flatc target from parent project")
else()
    message(STATUS "FLATBUFFERS: Fetching FlatBuffers from source...")
    set(VPUNN_USING_INTERNAL_FLATBUFFERS TRUE)

    include(FetchContent)
    FetchContent_Declare(
        flatbuffers
        GIT_REPOSITORY https://github.com/google/flatbuffers.git
        GIT_TAG v24.3.25
    )

    set(FLATBUFFERS_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(FLATBUFFERS_INSTALL OFF CACHE BOOL "" FORCE)
    set(FLATBUFFERS_BUILD_FLATC ON CACHE BOOL "" FORCE)
    
    if(JS_BUILD)
        # Populate sources (no build yet)
        FetchContent_GetProperties(flatbuffers)
        
        if(NOT flatbuffers_POPULATED)
            FetchContent_Populate(flatbuffers)
        endif()

        # Build host flatc (native) with ExternalProject
        include(ExternalProject)
        set(_FLATC_HOST_BINARY_DIR ${CMAKE_BINARY_DIR}/flatbuffers-host)
        set(_FLATC_HOST_BIN ${_FLATC_HOST_BINARY_DIR}/flatc)

        ExternalProject_Add(flatc_host_build
            SOURCE_DIR ${flatbuffers_SOURCE_DIR}
            BINARY_DIR ${_FLATC_HOST_BINARY_DIR}
            CMAKE_ARGS
                -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
                -DCMAKE_CXX_COMPILER=g++
            BUILD_COMMAND ${CMAKE_COMMAND} --build . --target flatc --parallel 8
            INSTALL_COMMAND ""
        )
        add_executable(flatc_host IMPORTED GLOBAL)
        add_dependencies(flatc_host flatc_host_build)
        set_target_properties(flatc_host PROPERTIES IMPORTED_LOCATION ${_FLATC_HOST_BIN})

        # Provide a flatbuffers interface target for headers
        if(NOT TARGET flatbuffers)
            add_library(flatbuffers INTERFACE)
            target_include_directories(flatbuffers INTERFACE ${flatbuffers_SOURCE_DIR}/include)
            # Avoid expecting generated sources from a built library
            target_compile_definitions(flatbuffers INTERFACE FLATBUFFERS_HEADER_ONLY)
        endif()

        set(FLATC_COMMAND $<TARGET_FILE:flatc_host>)
        set(FLATC_TARGET flatc_host)
        message(STATUS "FLATBUFFERS: Host flatc will be built at ${_FLATC_HOST_BIN}")
    else()
        FetchContent_MakeAvailable(flatbuffers)

        if(NOT MSVC)
            # only for clang 13+
            if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
                # if clang version < 13, do nothing
                if ((CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 13))
                    target_compile_options(flatbuffers PRIVATE -Wno-unused-but-set-variable)
                    target_compile_options(flatc PRIVATE -Wno-unused-but-set-variable)
                    target_compile_options(flatbuffers PRIVATE -Wno-suggest-override)
                    target_compile_options(flatc PRIVATE -Wno-suggest-override)
                endif()
            # For ther compilers that are not clang (GNU)
            else()
                target_compile_options(flatbuffers PRIVATE -Wno-suggest-override)
                target_compile_options(flatc PRIVATE -Wno-suggest-override)  
            endif()
        endif()

        set(FLATC_COMMAND $<TARGET_FILE:flatc>)
        set(FLATC_TARGET flatc)

        message(STATUS "FLATBUFFERS: Fetched and configured")
    
    endif()
endif()

# Export variables to the parent scope
set(FLATBUFFERS_SRC_DIR ${flatbuffers_SOURCE_DIR})
set(FLATBUFFERS_INCLUDE_DIRECTORIES ${flatbuffers_SOURCE_DIR}/include)
