# Copyright © 2026 Intel Corporation
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



# httplib and its zstd dependency were removed in favor of the in-tree
# HttpClient (see include/http_client/http_client.h).
# Only nlohmann/json remains as a header-only dependency.

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
		$<TARGET_PROPERTY:nlohmann_json::nlohmann_json,INTERFACE_INCLUDE_DIRECTORIES>
)

# Add source files
target_sources(vpunn_http_client
	PRIVATE
		http_cost_provider.cpp
		http_client.cpp
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
)

# Platform-specific linking (Winsock for HttpClient)
if(WIN32)
    target_link_libraries(vpunn_http_client PRIVATE ws2_32)
endif()

