# Copyright © 2022 Intel Corporation
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

    if(NOT MSVC)
        # Add vector instructions support for linux
        target_compile_options(blas PRIVATE -march=native)
        target_compile_options(blasStatic PRIVATE -march=native)
    else()
        target_compile_definitions(blas PRIVATE USE_SIMD)
        target_compile_definitions(blasStatic PRIVATE USE_SIMD)
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
