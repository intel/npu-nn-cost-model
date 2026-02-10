# Copyright © 2024 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
# LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
# is subject to the terms and conditions of the software license agreements for the Software Package,
# which may also include notices, disclaimers, or license terms for third party or open source software
# included in or with the Software Package, and your use indicates your acceptance of all such terms.
# Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
# Software Package for additional details.

# src/shave/

include(cmake/FileProcessing)

set (LIST_LEGACY_CSV "")

# clasic shave  (old/Deprecated) based on latency and efficiency (linear)
function(GENERATE_SHV_KERNEL KERNEL_TYPE KERNEL_NAME)
    set(K_NAME_csv "${KERNEL_NAME}")# original from csv
    set(K_TYPE_csv "${KERNEL_TYPE}")# original from csv

    set(KERNEL_NAME "SHV${KERNEL_NAME}")
    set(KERNEL_TYPE "SHV_${KERNEL_TYPE}_CLASS")

    if(ARGC EQUAL 2)
        set(KERNEL_EFFICIENCY 1.0)
        set(KERNEL_LATENCY 0)
    elseif(ARGC EQUAL 4)
        set(KERNEL_EFFICIENCY ${ARGV2})
        set(KERNEL_LATENCY ${ARGV3})
    else()
        message(FATAL_ERROR "Wrong number of parameters in GENERATE_SHV_KERNEL function")
    endif()

    # message(STATUS "Generate SHV Kernel ${KERNEL_NAME} with efficiency ${KERNEL_EFFICIENCY} bytes/cycle and latency ${KERNEL_LATENCY} cycles")
    set(GEN_HEADER_FILE "vpu/shave/${KERNEL_NAME}.h")
    set(GEN_HEADER_FULL_FILE "${COST_MODEL_BINARY_DIR}/include/${GEN_HEADER_FILE}")

    set(content "
#pragma once

#include \"vpu/shave/activation.h\"
#include \"vpu/shave/data_movement.h\"
#include \"vpu/shave/elementwise.h\"

namespace VPUNN \{

struct ${KERNEL_NAME} : public ${KERNEL_TYPE}<int(${KERNEL_EFFICIENCY} * 1000), ${KERNEL_LATENCY}> \{
    using ${KERNEL_TYPE}::${KERNEL_TYPE}\;
\}\;

\}

")
    vpu_cost_model_safe_file_write("${GEN_HEADER_FULL_FILE}" ${content} WRITE)

    set(ADD_LINE "Add<${KERNEL_NAME}, int(${KERNEL_EFFICIENCY}F * 1000), ${KERNEL_LATENCY}>(\"${K_NAME_csv}\")\;  //${K_TYPE_csv}")
    #message(STATUS " >                  ${ADD_LINE}")

    set(LIST_LEGACY_CSV "${LIST_LEGACY_CSV}${ADD_LINE} \n " PARENT_SCOPE)

    set(GLOBAL_INCLUDE_FILE "${GLOBAL_INCLUDE_FILE}#include \"${GEN_HEADER_FILE}\"\n" PARENT_SCOPE)
endfunction()

file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/layers_perf.csv" config)
STRING(REPLACE "null" "" config "${config}")
STRING(REPLACE " " "" config "${config}")
STRING(REPLACE "\n" ";" config "${config}")

set(SKIP_HEADER TRUE)

foreach(LINE IN LISTS config)
    if(SKIP_HEADER)
        set(SKIP_HEADER FALSE)
    else()
        STRING(REPLACE "," ";" PARAMETERS ${LINE})
        GENERATE_SHV_KERNEL(${PARAMETERS})
    endif()
endforeach()

vpu_cost_model_safe_file_write("${COST_MODEL_BINARY_DIR}/include/vpu/shave/layers.h" ${GLOBAL_INCLUDE_FILE} WRITE)

#place inl file for legacy
#message(STATUS " >> Generated code for Shave Legacy is : \n ${LIST_LEGACY_CSV} \n  >> Generated code END")
set (SHAVE_ADD_LIST "${LIST_LEGACY_CSV}")

#message(STATUS " >> Generated Collection initializer for Shave Legacy is : \n ${SHAVE_ADD_LIST} \n  >> Generated code END")

vpu_cost_model_safe_file_write("${COST_MODEL_BINARY_DIR}/include/vpu/shave/SHAVE_V27_Linear.inl" ${SHAVE_ADD_LIST} WRITE)

#NEW SHAVE modeling.  v27 CSV import

function(GENERATE_SHV_INITIALIZER kernel_type kernel_name slope intercept scalar_offset unroll_offset unroll_size vector_size DPU_frequency SHV_frequency)
    if(ARGC EQUAL 10)
        # use the passed params
    else()
        message(FATAL_ERROR "Wrong number of parameters in GENERATE_SHV_INITIALIZER function: ${ARGC} instead of  10 , \n ${ARGV}")
    endif()

    set(data_type "DataType::FLOAT16")

    if(kernel_type STREQUAL "ShaveModel1on1" OR kernel_type STREQUAL "ShaveModel2on1")
        set (ADD_LINE "Add<${data_type},${vector_size},${unroll_size},${DPU_frequency},${SHV_frequency} >(\"${kernel_name}\", (float)${slope} , (float)${intercept}, (float)${scalar_offset}, (float)${unroll_offset})\;  //${kernel_type}")
        #message(STATUS "Parse SHV Kernel ${kernel_name} with: ${kernel_type} => ${slope},${intercept},${scalar_offset},${unroll_offset},${unroll_size},${vector_size},${DPU_frequency},${SHV_frequency}")
        # message(STATUS " >                  ${ADD_LINE}")
    endif()

    set (LIST_INITIALIZER_FROM_CSV "${LIST_INITIALIZER_FROM_CSV}${ADD_LINE} \n " PARENT_SCOPE)
endfunction()

function(GENERATE_SHV_INITIALIZER_NPU40 kernel_type kernel_name slope intercept unroll_offset intra_block_offset vector_offset displacement_size unroll_size vector_size DPU_frequency SHV_frequency)
    if(ARGC EQUAL 12)
        # use the passed params
    else()
        message(FATAL_ERROR "Wrong number of parameters in GENERATE_SHV_INITIALIZER function: ${ARGC} instead of  11 , \n ${ARGV}")
    endif()

    set(data_type "DataType::FLOAT16")
    # message(STATUS "Parse SHV Kernel ${kernel_name} with: ${kernel_type} => ${slope},${intercept},${scalar_offset},${unroll_offset},${unroll_size},${vector_size},${DPU_frequency},${SHV_frequency}")


    if(kernel_type STREQUAL "ShaveModel1on1NPU40" OR kernel_type STREQUAL "ShaveModel2on1NPU40")
        #message(STATUS "Parse SHV Kernel ${kernel_name} with: ${kernel_type} => ${slope},${intercept},${unroll_offset},${intra_block_offset},${vector_offset},${unroll_size},${vector_size},${DPU_frequency},${SHV_frequency}")
        set (ADD_LINE "Add<${data_type},${vector_size},${unroll_size},${DPU_frequency},${SHV_frequency} >(\"${kernel_name}\", (float)${slope} , (float)${intercept}, (float)${unroll_offset}, (float)${intra_block_offset}, (float)${vector_offset}, ${displacement_size})\;  //${kernel_type}")
    endif()

    set (LIST_INITIALIZER_FROM_CSV "${LIST_INITIALIZER_FROM_CSV}${ADD_LINE} \n " PARENT_SCOPE)
endfunction()

function(GET_MODEL_TYPE kernel_type)
    # message(STATUS "Identify SHV Kernel ${kernel_type}")
    if(kernel_type STREQUAL "ShaveModel1on1" OR kernel_type STREQUAL "ShaveModel2on1")
        set(kmode "KType1") # this is the type for 1-1 or 2-1 simple
        # message(STATUS "Identified as  mode1: ${kmode}")
    elseif(kernel_type STREQUAL "ShaveModel1on1NPU40" OR kernel_type STREQUAL "ShaveModel2on1NPU40")
        set(kmode "KType1_NPU40") # this is the type for 1-1 or 2-1 simple for NPU 40
    else()
        set(kmode "KTypeUnknown")
        # message(STATUS "Identified as  not recognized: ${kmode}")
    endif()

    set(KERNEL_INFO ${kmode} PARENT_SCOPE)
    # message(STATUS "Identified as: ${KERNEL_INFO}, ${kmode}")
endfunction()

function(PARSE_CSV csv_file  inl_out_file  info)
    message(STATUS "\n >> Starting to generate initializer for ${info}")
    message(STATUS "\n >> PARAMS:\n csv_file: " ${csv_file} "\n inl_out_file: " ${inl_out_file} "\n info: " ${info} )

    file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/${csv_file}" config)
  # message(STATUS "\n >> config: " ${config})  # content of file
    STRING(REPLACE "null" "" config "${config}")
    STRING(REPLACE " " "" config "${config}")
    STRING(REPLACE "\n" ";" config "${config}")

    set(SKIP_HEADER TRUE)
    set(LIST_INITIALIZER_FROM_CSV "")

    foreach(LINE IN LISTS config)
        if(SKIP_HEADER)
            set(SKIP_HEADER FALSE)
        else()
            STRING(REPLACE "," ";" PARAMETERS ${LINE})

            set (KERNEL_INFO "XXX")
            GET_MODEL_TYPE(${PARAMETERS})

            if(KERNEL_INFO STREQUAL "KType1")
             GENERATE_SHV_INITIALIZER(${PARAMETERS})
            elseif(KERNEL_INFO STREQUAL "KType1_NPU40")
             GENERATE_SHV_INITIALIZER_NPU40(${PARAMETERS})
            elseif(KERNEL_INFO STREQUAL "KTypeUnknown"  )
             message(STATUS " >    Line to parse not known:              ${PARAMETERS}")
            else()
             message(STATUS " >   ERROR in  Line to parse (unexpected kernel info):     ${KERNEL_INFO}  ::params=        ${PARAMETERS}")
            endif()
        endif()
    endforeach()

    #place inl file for legacy
    set(SHAVE_ADD_LIST "${LIST_INITIALIZER_FROM_CSV}")
    #message(STATUS " >> Generated Collection initializer for ${info} is : \n ${SHAVE_ADD_LIST} \n  >> Generated code END (${info}) \n")
    vpu_cost_model_safe_file_write("${COST_MODEL_BINARY_DIR}/include/vpu/shave/${inl_out_file}" ${SHAVE_ADD_LIST} WRITE)  #file is implicitly without prev content
endfunction()


set(P1 "shave_layers_vpu_2_7.csv")
set(P2 "SHAVE_V27.inl")
#PARSE_CSV( shave_layers_vpu_2_7.csv SHAVE_V27.inl)

PARSE_CSV( ${P1} ${P2} "Shave NPU27 WIP")

# Deactivating till figuring out the output format

set(P1 "shave_layers_vpu_40.csv") #must contain at least one input
set(P2 "SHAVE_V40.inl")
PARSE_CSV( ${P1} ${P2} "Shave NPU40 WIP")

# ####################################################################################################

# shave library for new cpps
if(VPUNN_BUILD_SHARED_LIB)
    add_library(vpunn_shave SHARED)
else()
    add_library(vpunn_shave STATIC)
endif()

target_include_directories(vpunn_shave
    PRIVATE 
        $<BUILD_INTERFACE:${COST_MODEL_ROOT_DIR}/include/>
        $<BUILD_INTERFACE:${COST_MODEL_BINARY_DIR}/include>
        $<BUILD_INTERFACE:${FLATBUFFERS_SRC_DIR}/include>
)

target_sources(vpunn_shave 
    PRIVATE 
        shave_cache.cpp
        shave_collection.cpp # for shared builds
)

target_link_libraries(vpunn_shave
    PRIVATE 
        vpunn_common_settings
        flatbuffers
)

# === Generate Shave Factors Mapping from CSV === #

# Function to generate header file with populate class
# @Params:
#   CSV_FILE - path to the csv file containing factors
#   CLASS_NAME - name of the class to generate content for, which will have prefix: `PopulatedFactorsLUT<CLASS_NAME>`
#   output_file - path to the output header file to generate
# @Returns:
#   Generates header file with populate class
function(GEN_HEADER_FILE_FOR_FACTORS_POPULATE CSV_FILE CLASS_NAME output_file)
    # set POPULATE_INFO for the file header
    set(POPULATE_INFO "// This file is auto-generated. Do not edit directly.\n")
    string(TIMESTAMP current_time "%Y-%m-%d %H:%M:%S")
    string(APPEND POPULATE_INFO "// Generated at: ${current_time}\n")

    GEN_CONTENT_FROM_CSV(
        "${CSV_FILE}"
        POPULATE_CONTENT
        "            add(\"{0}\", {1}f);\n" # {0} = shave op name, {1} = float factor
        DESCRIPTION "Shave Speedup Factor Functions content generation"
    )

    configure_file(ShaveFactorsPopulation.h.in ${output_file})
    message(STATUS " >> Generated Shave Factors Population Header File: ${output_file} for class PopulatedFactorsLUT${CLASS_NAME} \n  >> Generated code END")
endfunction()

# --- This section should be changed in dependence of the files containing factors and the output format --- #

GEN_HEADER_FILE_FOR_FACTORS_POPULATE(
    "${CMAKE_CURRENT_SOURCE_DIR}/shave_factors_npu5.csv"
    "_NPU5"         # the class name will be PopulatedFactorsLUT_NPU5
    "${COST_MODEL_BINARY_DIR}/include/vpu/shave/generated_shave_factors_population_npu5.h"
)
