# Copyright © 2024 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
# LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
# is subject to the terms and conditions of the software license agreements for the Software Package,
# which may also include notices, disclaimers, or license terms for third party or open source software
# included in or with the Software Package, and your use indicates your acceptance of all such terms.
# Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
# Software Package for additional details.
# Software Package for additional details.

# src/http_client/

include(FetchContent)

message(STATUS "Preparing dependencies for VPUNN http client.")

# Workaround for libzstd error (uncomment if error involving linking zstdlib found)
## Setup zstd dependency
#if(NOT TARGET zstd::libzstd)
#    message(STATUS "Setting up zstd dependency")
#    
#    # Disable building extra components from zstd
#    set(ZSTD_BUILD_PROGRAMS  OFF CACHE BOOL "Disable zstd CLI" FORCE)
#    set(ZSTD_BUILD_SHARED    OFF CACHE BOOL "Disable zstd shared libs" FORCE)
#    set(ZSTD_BUILD_STATIC    ON  CACHE BOOL "Enable zstd static lib" FORCE)
#    
#    FetchContent_Declare(
#        zstd
#        GIT_REPOSITORY https://github.com/facebook/zstd.git
#        GIT_TAG        v1.5.5
#        SOURCE_SUBDIR  build/cmake
#    )
#    set(NEED_FETCH_ZSTD TRUE)
#
#
#    FetchContent_MakeAvailable(zstd)
#endif()
#
#if(NEED_FETCH_ZSTD)
#	# Check existance
#	if(TARGET libzstd_static)
#        message(STATUS "libzstd_static: EXISTS")
#    else()
#        message(STATUS "libzstd_static: NOT FOUND")
#    endif()
#    
#    # Try different possible target names and create alias
#    if(TARGET libzstd_static)
#        add_library(zstd::libzstd ALIAS libzstd_static)
#        message(STATUS "Created alias zstd::libzstd -> libzstd_static")
#    else()
#        message(FATAL_ERROR "Could not find any zstd target to create alias. Check the debug output above.")
#    endif()
#endif()

# Fetch httplib (header-only) if not already available
if(NOT TARGET httplib::httplib)
	FetchContent_Declare(
		httplib
		GIT_REPOSITORY https://github.com/yhirose/cpp-httplib.git
		GIT_TAG        v0.20.0
	)
	set(NEED_FETCH_HTTPLIB TRUE)
endif()

# Fetch nlohmann/json (header-only, optional) if not already available
if(NOT TARGET nlohmann_json::nlohmann_json)
	FetchContent_Declare(
		json
		GIT_REPOSITORY https://github.com/nlohmann/json.git
		GIT_TAG        v3.11.3
	)
	set(NEED_FETCH_JSON TRUE)
endif()

if(NEED_FETCH_HTTPLIB)
	FetchContent_MakeAvailable(
		httplib
	)
endif()

if(NEED_FETCH_JSON)
	FetchContent_MakeAvailable(
		json
	)
endif()

# Configure httplib options before making available
set(HTTPLIB_BUILD_TESTS OFF CACHE BOOL "Disable httplib tests" FORCE)
set(HTTPLIB_BUILD_EXAMPLES OFF CACHE BOOL "Disable httplib examples" FORCE)
set(HTTPLIB_BUILD_DOCS OFF CACHE BOOL "Disable httplib documentation" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build httplib as static library" FORCE)

# Create library
add_library(vpunn_http_client STATIC)

target_include_directories(vpunn_http_client
	PRIVATE 
		$<BUILD_INTERFACE:${COST_MODEL_ROOT_DIR}/include/>
	INTERFACE
		$<TARGET_PROPERTY:httplib::httplib,INTERFACE_INCLUDE_DIRECTORIES>
)

# Add source files
target_sources(vpunn_http_client
	PRIVATE
		http_cost_provider.cpp
)

# Link to common settings
target_link_libraries(vpunn_http_client
	PRIVATE
		vpunn_common_settings
		httplib::httplib
		nlohmann_json::nlohmann_json
)

# Platform-specific linking
if(WIN32)
    target_link_libraries(vpunn_http_client PRIVATE ws2_32)
endif()

# Legacy header copy (if needed for backward compatibility)
# add_custom_command(
#     TARGET vpunn_http_client PRE_BUILD
#     COMMAND ${CMAKE_COMMAND} -E copy 
#         ${httplib_SOURCE_DIR}/httplib.h
#         ${COST_MODEL_BINARY_DIR}/include
#     COMMENT "Copying httplib.h for legacy compatibility"
# )


