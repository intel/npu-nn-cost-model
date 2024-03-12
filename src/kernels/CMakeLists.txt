# Copyright © 2023 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
# LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
# is subject to the terms and conditions of the software license agreements for the Software Package,
# which may also include notices, disclaimers, or license terms for third party or open source software
# included in or with the Software Package, and your use indicates your acceptance of all such terms.
# Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
# Software Package for additional details.

# add the executable
add_library(kernels bias.cpp fully_conneted.cpp l2_normalization.cpp kNN.cpp sigmoid.cpp)

if(NOT DEFINED MKL_THREADING)
    set(MKL_THREADING "tbb")
endif()

if(CBLAS_LIB STREQUAL "mkl")
    message(STATUS "Using Intel MKL BLAS library")
    target_compile_definitions(kernels PRIVATE USE_MKL)
elseif(CBLAS_LIB STREQUAL "openblas")
    message(STATUS "Using OpenBLAS BLAS library")
    target_compile_definitions(kernels PRIVATE USE_OPENBLAS)
elseif(CBLAS_LIB STREQUAL "internal")
    message(STATUS "Using internal BLAS library")
    add_library(blas blas.cpp)
    add_library(blasStatic STATIC blas.cpp)

    # vector instructions are required for fast code
    if(NOT MSVC)
        if(NOT ${JS_BUILD}) #NA for WASM compiler 
            if(VPUNN_ENABLE_VECTOR_INSTRUCTIONS)
                message(STATUS "BLAS: -msse -msse2 -msse3 -msse4 -msse4.2 set")
                target_compile_options(blas PRIVATE -msse -msse2 -msse3 -msse4 -msse4.2)
                target_compile_options(blasStatic PRIVATE -msse -msse2 -msse3 -msse4 -msse4.2) 
            else()
                 message(STATUS "BLAS: no vector optimization set (linux)")
            endif()
        endif()
    else()
        if(VPUNN_ENABLE_VECTOR_INSTRUCTIONS)
            #for Win/MSVC also SIMD specific code can be enabled
            message(STATUS "BLAS: USE_SIMD ON manually: msvc")
            target_compile_definitions(blas PRIVATE USE_SIMD)
            target_compile_definitions(blasStatic PRIVATE USE_SIMD)
        else()
             message(STATUS "BLAS: no USE_SIMD change (MSVC) ")
        endif()
    endif()

    add_dependencies(kernels blas)
else()
    # Do nothing, we already printed an error above in this case.
endif()

if(CBLAS_LIB STREQUAL "internal")
    target_link_libraries(kernels blas ${CMAKE_THREAD_LIBS_INIT})
else()
    target_link_libraries(kernels ${BLAS_LIBRARIES} ${BLAS_LINKER_FLAGS} ${CMAKE_THREAD_LIBS_INIT})
endif()
