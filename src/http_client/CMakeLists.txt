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



# This error sometimes occurs on Linux when building the HTTP client due to
# the zstd dependency of httplib. The error looks like: CMake error,
# zstd::libzstd target was not found. Set up the zstd dependency.
set(NEED_FETCH_ZSTD FALSE)

if(NOT TARGET zstd::libzstd)
    # 1. Try a system-installed (or already find_package'd) zstd.
    find_package(zstd QUIET)

    if(TARGET zstd::libzstd)
        message(STATUS "Using system zstd ${zstd_VERSION}")
    elseif(TARGET libzstd_static)
        # 2. libzstd_static was provided by a parent CMake configuration.
        add_library(zstd::libzstd ALIAS libzstd_static)
        message(STATUS "Created alias zstd::libzstd -> libzstd_static")
    else()
        # 3. Last resort: fetch from source.
        message(STATUS "zstd not found locally, fetching v1.5.7 from source")
        set(ZSTD_BUILD_PROGRAMS OFF CACHE BOOL "Disable zstd CLI"         FORCE)
        set(ZSTD_BUILD_SHARED   OFF CACHE BOOL "Disable zstd shared libs" FORCE)
        set(ZSTD_BUILD_STATIC   ON  CACHE BOOL "Enable zstd static lib"   FORCE)

        # Scope policy changes to avoid affecting the rest of the project  
        cmake_policy(PUSH)
        #set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
        FetchContent_Declare(
            zstd
            GIT_REPOSITORY https://github.com/facebook/zstd.git
            GIT_TAG        v1.5.7
            SOURCE_SUBDIR  build/cmake
        )
        set(NEED_FETCH_ZSTD TRUE)
        FetchContent_MakeAvailable(zstd)
        cmake_policy(POP)
    endif()
endif()

if(NEED_FETCH_ZSTD)
    if(TARGET libzstd_static)
        add_library(zstd::libzstd ALIAS libzstd_static)
        message(STATUS "Created alias zstd::libzstd -> libzstd_static")
    else()
        message(FATAL_ERROR "zstd was fetched but libzstd_static target not found.")
    endif()
endif()
# END Setup zstd dependency

# Fetch httplib (header-only) if not already available
if(NOT TARGET httplib::httplib)
	# Save the current BUILD_SHARED_LIBS value
    if(DEFINED CACHE{BUILD_SHARED_LIBS})
        set(_SAVED_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS})
        set(_BUILD_SHARED_LIBS_WAS_SET TRUE)
    else()
        set(_BUILD_SHARED_LIBS_WAS_SET FALSE)
    endif()

    FetchContent_Declare(
		httplib
		GIT_REPOSITORY https://github.com/yhirose/cpp-httplib.git
		GIT_TAG        v0.20.0
	)

	# Configure httplib options here before FetchContent_MakeAvailable(httplib)
	# so they are applied during dependency configuration.
    set(HTTPLIB_BUILD_TESTS OFF CACHE BOOL "Disable httplib tests" FORCE)
    set(HTTPLIB_BUILD_EXAMPLES OFF CACHE BOOL "Disable httplib examples" FORCE)
    set(HTTPLIB_BUILD_DOCS OFF CACHE BOOL "Disable httplib documentation" FORCE)
    # Temporarily force static build for httplib
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build httplib as static library" FORCE)

	FetchContent_MakeAvailable(httplib)
     
    # Restore the original BUILD_SHARED_LIBS value
  if(_BUILD_SHARED_LIBS_WAS_SET)
        set(BUILD_SHARED_LIBS ${_SAVED_BUILD_SHARED_LIBS} CACHE BOOL "Build shared libraries" FORCE)
    else()
        # BUILD_SHARED_LIBS wasn't in cache originally, remove it
        unset(BUILD_SHARED_LIBS CACHE)
    endif()

        # Clean up temporary variables
    unset(_SAVED_BUILD_SHARED_LIBS)
    unset(_BUILD_SHARED_LIBS_WAS_SET)

    # Check if httplib found zstd support
    get_target_property(HTTPLIB_DEFS httplib::httplib INTERFACE_COMPILE_DEFINITIONS)
    message(STATUS "httplib compile definitions: ${HTTPLIB_DEFS}")

    # Look for the actual BOOL value, not just the presence of the string
    if(HTTPLIB_DEFS MATCHES "\\$<\\$<BOOL:([^>]+)>:CPPHTTPLIB_ZSTD_SUPPORT>")
        set(ZSTD_BOOL_VALUE "${CMAKE_MATCH_1}")
        if(ZSTD_BOOL_VALUE STREQUAL "0" OR ZSTD_BOOL_VALUE STREQUAL "FALSE" OR ZSTD_BOOL_VALUE STREQUAL "")
            message(STATUS "httplib: zstd compression DISABLED (value: '${ZSTD_BOOL_VALUE}')")
        else()
            message(STATUS "httplib: zstd compression ENABLED (value: '${ZSTD_BOOL_VALUE}')")
        endif()
    else()
        message(STATUS "httplib: zstd compression status unknown")
    endif()
	
	message(STATUS "httplib fetched and configured with or without zstd dependencies")
endif()

# Fetch nlohmann/json (header-only, optional) if not already available
if(NOT TARGET nlohmann_json::nlohmann_json)
	FetchContent_Declare(
		json
		GIT_REPOSITORY https://github.com/nlohmann/json.git
		GIT_TAG        v3.11.3
	)

	FetchContent_MakeAvailable(json)
endif()


# Create library
add_library(vpunn_http_client STATIC)

target_include_directories(vpunn_http_client
	PRIVATE 
		$<BUILD_INTERFACE:${COST_MODEL_ROOT_DIR}/include/>
	INTERFACE
		$<TARGET_PROPERTY:httplib::httplib,INTERFACE_INCLUDE_DIRECTORIES>
		$<TARGET_PROPERTY:nlohmann_json::nlohmann_json,INTERFACE_INCLUDE_DIRECTORIES>
)

# Add source files
target_sources(vpunn_http_client
	PRIVATE
		http_cost_provider.cpp
		workload_json.cpp
)

# Link to common settings
target_link_libraries(vpunn_http_client
	PRIVATE
		vpunn_common_settings
		vpunn_vpu
	PUBLIC
	    # Expose these to consumers that will end up linking vpunn_http_client as private
		nlohmann_json::nlohmann_json
		httplib::httplib
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


