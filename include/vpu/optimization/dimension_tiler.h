// Copyright © 2024 Intel Corporation
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
    bool divideBalanced(int toSplit, int bins_desired, SplitContainer& binsAccumulated) const {
        //// success if bins is zero, and nothing else to split (all consumed)
        // if (bins <= 0 && dim <= 0) {
        //     return true;
        // }

        //// failure
        //// if bins is zero, but still something to split
        //// if bins still available but nothing to split
        // if ((bins <= 0 && dim > 0) || (bins > 0 && dim <= 0)) {
        //     return false;  // may be we should empty the resultAccumulator?
        // }

        if (toSplit <= 0)  // all consumed
        {
            if (bins_desired > 0) {
                return false;  // failure   if bins still available
            } else {
                return true;  // success if bins is zero
            }
        } else {
            if (bins_desired <= 0) {
                return false;  // failure , no more bins
            } else {
                // iterate recursive
                const int tentative_slice_raw{ceil_division(toSplit, bins_desired)};  // largest part
                // must check that this slice is in allowed list (round up to nearest in range allowed) (nut not larger
                // than dim)
                //  for now we assume it is.All numbers are allowed

                const auto tentative_slice{range.roundToNextLarger(tentative_slice_raw)};

                if (!tentative_slice.has_value()) {
                    return false;  // not existing in allowed set
                }
                if (*tentative_slice > toSplit) {  // too large to fit  (value too small to split)
                    return false;
                }

                binsAccumulated.push_back(*tentative_slice);
                return divideBalanced(toSplit - (*tentative_slice), (bins_desired - 1), binsAccumulated);
            }
        }

        //
    }
};

}  // namespace VPUNN

#endif  // VPUNN_TILER_H