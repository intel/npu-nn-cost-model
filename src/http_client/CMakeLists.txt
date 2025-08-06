include(FetchContent)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

message("Preparing dependencies for VPUNN http client.")

# Fetch httplib (header-only)
FetchContent_Declare(
  httplib
  GIT_REPOSITORY https://github.com/yhirose/cpp-httplib.git
  GIT_TAG        v0.20.0
)

# Fetch nlohmann/json (header-only, optional)
FetchContent_Declare(
  json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG        v3.11.3
)

set(HTTPLIB_BUILD_TESTS OFF CACHE BOOL "Disable httplib tests" FORCE)
set(HTTPLIB_BUILD_EXAMPLES OFF CACHE BOOL "Disable httplib examples" FORCE)
set(HTTPLIB_BUILD_DOCS OFF CACHE BOOL "Disable httplib documentation" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build httplib as static library" FORCE)

FetchContent_MakeAvailable(httplib json)

add_library(vpunn_http_client STATIC http_cost_provider.cpp)

target_include_directories(vpunn_http_client 
	PUBLIC 
		${httplib_SOURCE_DIR}
)

add_custom_command(
  TARGET vpunn_http_client PRE_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy 
    ${httplib_SOURCE_DIR}/httplib.h
    ${CMAKE_VPUNN_BINARY_DIR}/include
)

target_link_libraries(vpunn_http_client
	PUBLIC
		httplib::httplib
		nlohmann_json::nlohmann_json
)

if(WIN32)
    target_link_libraries(vpunn_http_client PRIVATE ws2_32)
endif()
