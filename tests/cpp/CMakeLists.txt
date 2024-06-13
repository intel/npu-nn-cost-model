# Copyright © 2023 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
# LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
# is subject to the terms and conditions of the software license agreements for the Software Package,
# which may also include notices, disclaimers, or license terms for third party or open source software
# included in or with the Software Package, and your use indicates your acceptance of all such terms.
# Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
# Software Package for additional details.

include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG release-1.12.1
)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

include_directories(${CMAKE_BINARY_DIR}/include)
include_directories(${FLATBUFFERS_SRC_DIR}/include)

set (LIST_ACT_TEST_CASES "")

function(GENERATE_SHV_KERNEL_TEST KERNEL_TYPE KERNEL_NAME)
  set(K_NAME_csv "${KERNEL_NAME}")# original from csv

  set(KERNEL_NAME "SHV${KERNEL_NAME}")
  set(TEST_NAME "SHV${KERNEL_NAME}Cost")

  if(${KERNEL_TYPE} STREQUAL "ACTIVATION")
 #   configure_file(
 #     "${CMAKE_CURRENT_SOURCE_DIR}/shave_act.cpp.in"
 #     "${CMAKE_CURRENT_BINARY_DIR}/shave/${KERNEL_NAME}.cpp"
 #   )
 #  message(STATUS "  > Generated SHV ACT Kernel TEST: ${TEST_NAME} for:${KERNEL_NAME}") 

   set(TEST_NAME_SMOKE "${K_NAME_csv}_SmokeTest")
   set (TEST_CASE_LINE "LEGACY_SHV_TEST_F( ${TEST_NAME_SMOKE}, ${KERNEL_NAME}) //${K_NAME_csv}")
   #message(STATUS "  >                  ${TEST_CASE_LINE}")

   set (LIST_ACT_TEST_CASES "${LIST_ACT_TEST_CASES} \n ${TEST_CASE_LINE} \n" PARENT_SCOPE)

  endif()
endfunction()

file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/../../src/shave/layers_perf.csv" config)
STRING(REPLACE " " "" config "${config}")
STRING(REPLACE "\n" ";" config "${config}")

set(SKIP_HEADER TRUE)
message(STATUS " >> Generating Legacy Shave Unit TEST ")

foreach(LINE ${config})
  if(SKIP_HEADER)
    set(SKIP_HEADER FALSE)
  else()
    STRING(REPLACE "," ";" PARAMETERS ${LINE})
    GENERATE_SHV_KERNEL_TEST(${PARAMETERS})
  endif()
endforeach()

#message(STATUS " >> Generated TEST code is : \n ${LIST_ACT_TEST_CASES} \n  >> Generated test code END")
 configure_file(
      "${CMAKE_CURRENT_SOURCE_DIR}/shave_legacy_all.cpp.in"
      "${CMAKE_CURRENT_BINARY_DIR}/shave/shave_legacy_all.cpp"
    )

file(GLOB test_src "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
file(GLOB shv_test_src "${CMAKE_CURRENT_BINARY_DIR}/shave/*.cpp")

add_executable(
  test_cost_model
  ${test_src}
  ${shv_test_src}
)
target_compile_definitions(test_cost_model PRIVATE VPU_2_7_MODEL_PATH="${CMAKE_SOURCE_DIR}/models/vpu_2_7.vpunn")
target_compile_definitions(test_cost_model PRIVATE VPU_2_0_MODEL_PATH="${CMAKE_SOURCE_DIR}/models/vpu_2_0.vpunn")
target_compile_definitions(test_cost_model PRIVATE VPU_4_0_MODEL_PATH="${CMAKE_SOURCE_DIR}/models/vpu_2_7.vpunn")

add_dependencies(test_cost_model cpp_schema)
target_link_libraries(
  test_cost_model
  gtest_main
  inferenceStatic
)

include(GoogleTest)
gtest_discover_tests(test_cost_model DISCOVERY_TIMEOUT 30)

if(CMAKE_BUILD_TYPE STREQUAL "Coverage")
  # Run all tests
  add_custom_target(run_tests
    COMMAND ctest --test-dir tests/cpp/
    DEPENDS test_cost_model
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Run all tests"
  )

  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(GCOV_BIN "llvm-cov-14 gcov")
  elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    set(GCOV_BIN "gcov-9")
  else()
    message(FATAL "-- Building with unrecognised compiler, cannot enable code coverage")
  endif()

  set(COVERAGE_DIR ${CMAKE_BINARY_DIR}/coverage)
  set(COVERAGE_TITLE "VPUNN Code Coverage")
  set(COVERAGE_HTML_OPTIONS --html --html-title=${COVERAGE_TITLE})
  set(GCOV_COMMAND gcovr --gcov-executable ${GCOV_BIN} -s -r .. . ${COVERAGE_HTML_OPTIONS} -e=_deps)

  # Generate coverage report for CI (no details)
  add_custom_target(coverage_ci
    COMMAND ${GCOV_COMMAND} -o coverage.html
    DEPENDS run_tests
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Create code coverage"
  )

  # Generate coverage report with extra details
  add_custom_target(coverage
    COMMAND mkdir -p ${COVERAGE_DIR} && ${GCOV_COMMAND} --html-details -o ${COVERAGE_DIR}/coverage.html
    DEPENDS run_tests
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Create code coverage"
  )
endif()
