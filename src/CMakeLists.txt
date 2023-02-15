# Copyright © 2022 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
# LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
# is subject to the terms and conditions of the software license agreements for the Software Package,
# which may also include notices, disclaimers, or license terms for third party or open source software
# included in or with the Software Package, and your use indicates your acceptance of all such terms.
# Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
# Software Package for additional details.

find_package(BLAS)

if (NOT DEFINED CBLAS_LIB)
    # If we haven't been asked to use a particular implementation, make a guess at what the user wants
    message(STATUS "No CBLAS_LIB specified")
    if(DEFINED ENV{EMSCRIPTEN})
        set(CBLAS_LIB "internal")
        message(STATUS "Defaulting CBLAS_LIB to internal")
    elseif (EXISTS $ENV{MKLROOT})
        set(CBLAS_LIB "mkl")
        message(STATUS "Defaulting CBLAS_LIB to mkl")
    else()
        set(CBLAS_LIB "internal")
        message(STATUS "Defaulting CBLAS_LIB to internal")
    endif()
elseif (CBLAS_LIB STREQUAL "mkl")
    set(BLA_VENDOR "Intel")
elseif (CBLAS_LIB STREQUAL "openblas")
    set(BLA_VENDOR "OpenBLAS")
elseif (CBLAS_LIB STREQUAL "internal")
    # Do nothing, manually requested to use internal implementation
else()
    message("Unrecognised option for CBLAS_LIB: ${CBLAS_LIB} (valid: mkl, openblas, internal)")
endif()

# Kernels
add_subdirectory(core)
add_subdirectory(kernels)
add_subdirectory(inference)
add_subdirectory(optimization)
add_subdirectory(shave)
add_subdirectory(schema)
