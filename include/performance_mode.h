// Copyright © 2025 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_PERFORMANCE_MODE_H
#define VPUNN_PERFORMANCE_MODE_H

namespace VPUNN {

class PerformanceMode {
public:
    static constexpr bool forceLegacy_G4{true};
    static constexpr bool forceLegacy_G5{false};

    /// if true allows legacy swizzling for G5, otherwise applies sanitization
    static constexpr bool allowLegacySwizzling_G5{false};
};

}  // namespace VPUNN

#endif  // VPUNN_PERFORMANCE_H
