// Copyright © 2026 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_DIMENSION_TILER_H
#define VPUNN_DIMENSION_TILER_H

#include <vector>
#include "vpu/ranges.h"

namespace VPUNN {

// to split a dimension based on some rules
class SplitDimension {
public:
    using SplitContainer = std::vector<int>;
    const SmartRanges range;  // the range of allowed values

    // ctor with externally received smartranges
    SplitDimension(const SmartRanges& range_): range(range_) {
    }

    // recursive
    //  0 / 0 OK by definition
    bool divideBalanced(int toSplit, int bins_desired, SplitContainer& binsAccumulated) const;
};

}  // namespace VPUNN

#endif  // VPUNN_DIMENSION_TILER_H
