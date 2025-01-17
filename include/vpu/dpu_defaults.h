// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_DPU_DEFAULTS_H
#define VPUNN_DPU_DEFAULTS_H

#include <string> 

#include "dpu_types.h"

namespace VPUNN {

/// @brief default Layout that is equivalent with legacy ZMAJOR
constexpr Layout getDefaultLayout() {
    return Layout::ZXY;
}

inline constexpr Swizzling default_init_swizzling() {
    return Swizzling::KEY_5;
}

inline std::string out_terminator() {
    return "END-";
}

}  // namespace VPUNN

#endif
