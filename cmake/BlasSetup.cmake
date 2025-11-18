# Copyright © 2024 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
# LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
# is subject to the terms and conditions of the software license agreements for the Software Package,
# which may also include notices, disclaimers, or license terms for third party or open source software
# included in or with the Software Package, and your use indicates your acceptance of all such terms.
# Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
# Software Package for additional details.

# This script will detect the blas library and set corresponding flags
# to be used by other targets. It will search for mkl and openblas, and 
# if not found will fallback to internal implementation

# Needed variablea are CBLAS_LIB and BLAS_LIBRARIES

if(NOT BLAS_SETUP_DONE)
    find_package(BLAS)
    # make it default if not set
    if(NOT DEFINED CBLAS_LIB)
        set(CBLAS_LIB "internal")
    endif()

    # CBLAS library selection with cache variable
    set(CBLAS_LIB ${CBLAS_LIB} CACHE STRING "CBLAS library selection")
    set_property(CACHE CBLAS_LIB PROPERTY STRINGS "auto" "mkl" "openblas" "internal")

    # added auto for automatic detection
    # which means that it will search first for mkl, openblas
    # and if not found then will fallback to internal implementation
    # instead of auto, there can be specified any lib from (mkl, openblas, internal) that
    # will be used instead, without the fallback mechanism
    if(CBLAS_LIB STREQUAL "auto")
        message(STATUS "Auto-detecting CBLAS library")
        if(DEFINED ENV{EMSCRIPTEN})
            set(CBLAS_LIB "internal")
            message(STATUS "Emscripted detected - using internal CBLAS")
        elseif(EXISTS $ENV{MKLROOT})
            set(CBLAS_LIB "mkl")
            message(STATUS "MKL detected - using MKL CBLAS")
        else()
            if(BLAS_FOUND)  
                string(JOIN " " BLAS_LIBRARIES_STR ${BLAS_LIBRARIES})
                if(BLAS_LIBRARIES_STR MATCHES ".*openblas.*")
                    set(CBLAS_LIB "openblas")
                    message(STATUS "Generic BLAS found - using OpenBLAS")
                else()
                    set(CBLAS_LIB "internal")
                    message(STATUS "BLAS found but not OpenBLAS - using internal implementation")
                endif()
            else()
                set(CBLAS_LIB "internal")
                message(STATUS "No optimized BLAS found - using internal implementation")
            endif()
        endif()
    else()
        message(STATUS "Using CBLAS library: ${CBLAS_LIB}")
    endif()

    # Configure BLAS vendor based on selection
    if(CBLAS_LIB STREQUAL "mkl")
        set(BLA_VENDOR "Intel")
        message(STATUS "Using Intel MKL BLAS")
    elseif(CBLAS_LIB STREQUAL "openblas")
        set(BLA_VENDOR "OpenBLAS")
        message(STATUS "Using OpenBLAS")
    elseif(CBLAS_LIB STREQUAL "internal")
        message(STATUS "Using internal BLAS implementation")
    else()
        message(FATAL_ERROR "Unrecognised CBLAS_LIB: ${CBLAS_LIB} (valid: auto, mkl, openblas, internal)")
    endif()

    set(CBLAS_LIB ${CBLAS_LIB} PARENT_SCOPE)
    set(BLAS_LIBRARIES ${BLAS_LIBRARIES} PARENT_SCOPE)  
    set(BLAS_SETUP_DONE TRUE)

endif()