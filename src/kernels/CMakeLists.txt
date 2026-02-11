# Copyright © 2024 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
# LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
# is subject to the terms and conditions of the software license agreements for the Software Package,
# which may also include notices, disclaimers, or license terms for third party or open source software
# included in or with the Software Package, and your use indicates your acceptance of all such terms.
# Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
# Software Package for additional details.

# src/kernels/

# Blas setup is needed for this target
include(cmake/BlasSetup)

add_library(vpunn_kernels STATIC)

# Add include dirs
target_include_directories(vpunn_kernels
    PRIVATE  
        $<BUILD_INTERFACE:${COST_MODEL_ROOT_DIR}/include>
)

# Add source files
target_sources(vpunn_kernels
    PRIVATE
        bias.cpp
        fully_conneted.cpp
        l2_normalization.cpp 
        kNN.cpp 
        sigmoid.cpp 
)

# Link to common settings
target_link_libraries(vpunn_kernels 
    PRIVATE 
        vpunn_common_settings 
)

# Set MKL threading default if not defined
if(NOT DEFINED MKL_THREADING)
    set(MKL_THREADING "tbb")
endif()

# Depending on the found BLAS lib, link it to the kernels target
if(CBLAS_LIB STREQUAL "mkl")
    message(STATUS "Using Intel MKL BLAS library")
    target_compile_definitions(vpunn_kernels PRIVATE USE_MKL)
    target_link_libraries(vpunn_kernels PRIVATE ${BLAS_LIBRARIES})
    # Apply BLAS linker flags if they exist
    if(BLAS_LINKER_FLAGS)
        target_link_options(vpunn_kernels PRIVATE ${BLAS_LINKER_FLAGS})
    endif()

elseif(CBLAS_LIB STREQUAL "openblas")
    message(STATUS "Using OpenBLAS BLAS library")
    target_compile_definitions(vpunn_kernels PRIVATE USE_OPENBLAS)
    target_link_libraries(vpunn_kernels PRIVATE ${BLAS_LIBRARIES})

    if(BLAS_LINKER_FLAGS)
        target_link_options(vpunn_kernels PRIVATE ${BLAS_LINKER_FLAGS})
    endif()

elseif(CBLAS_LIB STREQUAL "internal")
    message(STATUS "Using internal BLAS library for kernels target")

    # Create internal BLAS libraries
    add_library(blas STATIC)

    target_sources(blas PRIVATE blas.cpp)

    target_include_directories(blas
        PRIVATE 
            $<BUILD_INTERFACE:${COST_MODEL_ROOT_DIR}/include>
    )

    target_link_libraries(blas 
        PRIVATE 
            vpunn_common_settings 
    )

    # vector instructions are required for fast code
    if(VPUNN_ENABLE_VECTOR_INSTRUCTIONS)
        if(NOT MSVC) 
            message(STATUS "BLAS: -msse -msse2 -msse3 -msse4 -msse4.2 set")
            set(BLAS_VECTOR_FLAGS -msse -msse2 -msse3 -msse4 -msse4.2)
            target_compile_options(blas PRIVATE ${BLAS_VECTOR_FLAGS})
        else()
            #for Win/MSVC also SIMD specific code can be enabled
            message(STATUS "BLAS: USE_SIMD ON manually: msvc")
            target_compile_definitions(blas PRIVATE USE_SIMD)

        endif()
    else()
        message(STATUS "BLAS: no vector optimization set (linux)")
    endif()

    target_link_libraries(vpunn_kernels PRIVATE blas)

else()
    message(FATAL_ERROR "Unknown CBLAS_LIB value: ${CBLAS_LIB}")
endif()
