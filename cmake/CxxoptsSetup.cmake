# Copyright © 2026 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
# LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
# is subject to the terms and conditions of the software license agreements for the Software Package,
# which may also include notices, disclaimers, or license terms for third party or open source software
# included in or with the Software Package, and your use indicates your acceptance of all such terms.
# Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
# Software Package for additional details.

# cmake/CxxoptsSetup.cmake
# Fetches cxxopts if a target isn't already provided by a parent project.

if(TARGET cxxopts::cxxopts)
    message(STATUS "CXXOPTS: target already available")
else()
    include(FetchContent)
    FetchContent_Declare(
        cxxopts
        GIT_REPOSITORY https://github.com/jarro2783/cxxopts.git
        GIT_TAG v3.2.0
    )
    FetchContent_MakeAvailable(cxxopts)
endif()