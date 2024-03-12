# Copyright © 2023 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
# LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
# is subject to the terms and conditions of the software license agreements for the Software Package,
# which may also include notices, disclaimers, or license terms for third party or open source software
# included in or with the Software Package, and your use indicates your acceptance of all such terms.
# Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
# Software Package for additional details.

set (LIST_LEGACY_CSV "")

function(GENERATE_SHV_KERNEL KERNEL_TYPE KERNEL_NAME)
    set(K_NAME_csv "${KERNEL_NAME}")# original from csv
    set(K_TYPE_csv "${KERNEL_TYPE}")# original from csv

    set(KERNEL_NAME "SHV${KERNEL_NAME}")
    set(KERNEL_TYPE "SHV_${KERNEL_TYPE}_CLASS")

    if(${ARGC} EQUAL 2)
        set(KERNEL_EFFICIENCY 1.0)
        set(KERNEL_LATENCY 0)
    elseif(${ARGC} EQUAL 4)
        set(KERNEL_EFFICIENCY ${ARGV2})
        set(KERNEL_LATENCY ${ARGV3})
    else()
        message(FATAL_ERROR "Wrong number of parameters in GENERATE_SHV_KERNEL function")
    endif()

    message(STATUS "Generate SHV Kernel ${KERNEL_NAME} with efficiency ${KERNEL_EFFICIENCY} bytes/cycle and latency ${KERNEL_LATENCY} cycles")
    set(GEN_HEADER_FILE "vpu/shave/${KERNEL_NAME}.h")
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/layer.h.in"
        "${CMAKE_VPUNN_BINARY_DIR}/include/${GEN_HEADER_FILE}"
    )
    set (ADD_LINE "Add<${KERNEL_NAME}, int(${KERNEL_EFFICIENCY}F * 1000), ${KERNEL_LATENCY}>(\"${K_NAME_csv}\");  //${K_TYPE_csv}")
    #message(STATUS " >                  ${ADD_LINE}")

    set (LIST_LEGACY_CSV "${LIST_LEGACY_CSV}${ADD_LINE} \n " PARENT_SCOPE)
  
    set(GLOBAL_INCLUDE_FILE "${GLOBAL_INCLUDE_FILE}#include \"${GEN_HEADER_FILE}\"\n" PARENT_SCOPE)
endfunction()

file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/layers_perf.csv" config)
STRING(REPLACE "null" "" config "${config}")
STRING(REPLACE " " "" config "${config}")
STRING(REPLACE "\n" ";" config "${config}")

set(SKIP_HEADER TRUE)

foreach(LINE ${config})
    if(SKIP_HEADER)
        set(SKIP_HEADER FALSE)
    else()
        STRING(REPLACE "," ";" PARAMETERS ${LINE})
        GENERATE_SHV_KERNEL(${PARAMETERS})
    endif()
endforeach()

file(WRITE "${CMAKE_VPUNN_BINARY_DIR}/include/vpu/shave/layers.h" ${GLOBAL_INCLUDE_FILE})

#place inl file for legacy
#message(STATUS " >> Generated code for Shave Legacy is : \n ${LIST_LEGACY_CSV} \n  >> Generated code END")
set (SHAVE_ADD_LIST "${LIST_LEGACY_CSV}")
message(STATUS " >> Generated COllection initializer for Shave Legacy is : \n ${SHAVE_ADD_LIST} \n  >> Generated code END")
   configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/SHAVE_collection.inl.in"
        "${CMAKE_VPUNN_BINARY_DIR}/include/vpu/shave/SHAVE_V27_Linear.inl"
   )


#NEW v27 CSV import


function(GENERATE_SHV_INITIALIZER kernel_type kernel_name slope intercept scalar_offset unroll_offset unroll_size vector_size DPU_frequency SHV_frequency)

    if(${ARGC} EQUAL 10)
        #set(KERNEL_EFFICIENCY ${ARGV2})
        #set(KERNEL_LATENCY ${ARGV3})
    else()
        message(FATAL_ERROR "Wrong number of parameters in GENERATE_SHV_INITIALIZER function: ${ARGC} instead of  10 , \n ${ARGV}")
    endif()

    set(data_type "DataType::FLOAT16")
    message(STATUS "Parse SHV Kernel ${kernel_name} with: ${kernel_type} => ${slope},${intercept},${scalar_offset},${unroll_offset},${unroll_size},${vector_size},${DPU_frequency},${SHV_frequency}")

    if( (${kernel_type} STREQUAL "ShaveModel1on1") OR (${kernel_type} STREQUAL "ShaveModel2on1"))

    
        set (ADD_LINE "Add<${data_type},${vector_size},${unroll_size},${DPU_frequency},${SHV_frequency} >(\"${kernel_name}\", (float)${slope} , (float)${intercept}, (float)${scalar_offset}, (float)${unroll_offset});  //${kernel_type}")
        message(STATUS " >                  ${ADD_LINE}")
    endif()

    set (LIST_INITIALIZER_FROM_CSV "${LIST_INITIALIZER_FROM_CSV}${ADD_LINE} \n " PARENT_SCOPE)
  
endfunction()

message(STATUS "\n >> Starting to generate initializer for Shave 27 ")

file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/shave_layers_vpu_2_7.csv" config)
STRING(REPLACE "null" "" config "${config}")
STRING(REPLACE " " "" config "${config}")
STRING(REPLACE "\n" ";" config "${config}")

set(SKIP_HEADER TRUE)
set (LIST_INITIALIZER_FROM_CSV "")

foreach(LINE ${config})
    if(SKIP_HEADER)
        set(SKIP_HEADER FALSE)
    else()
        STRING(REPLACE "," ";" PARAMETERS ${LINE})
        GENERATE_SHV_INITIALIZER(${PARAMETERS})
    endif()
endforeach()

#place inl file for legacy
set (SHAVE_ADD_LIST "${LIST_INITIALIZER_FROM_CSV}")
message(STATUS " >> Generated COllection initializer for Shave 27 is : \n ${SHAVE_ADD_LIST} \n  >> Generated code END")
   configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/SHAVE_collection.inl.in"
        "${CMAKE_VPUNN_BINARY_DIR}/include/vpu/shave/SHAVE_V27.inl"
   )



# shave library for new cpps
include_directories(${CMAKE_CURRENT_BINARY_DIR}/../../include)
include_directories(${FLATBUFFERS_SRC_DIR}/include)

add_library(vpunn_shave shave_collection.cpp)
# add_dependencies(vpunn_shave_shared cpp_schema)
# target_link_libraries(vpunn_shave_shared flatbuffers)

if(VPUNN_BUILD_SHARED_LIB)
    add_library(vpunn_shave_shared SHARED shave_collection.cpp)
    # add_dependencies(vpunn_shave_shared cpp_schema)
    # target_link_libraries(vpunn_shave_shared flatbuffers)

    set_target_properties(vpunn_shave_shared PROPERTIES OUTPUT_NAME vpunn_shave)

endif()
