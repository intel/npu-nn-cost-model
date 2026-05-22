// Copyright © 2026 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_WTILING_H
#define VPUNN_WTILING_H

#include <string>
#include <list>

#include "vpu/layer_split_info.h"
#include "vpu/dpu_types.h"
#include "hwtiling.h"

namespace VPUNN {

/**
 * @brief A Tiler child class that implement the WTiling algorithm
 *
 */
class WTiling : public HWTiling {
public:
    /**
     * @brief Using the HWTiling class constructor
     *
     */
    using HWTiling::HWTiling;

    /**
     * @brief Tile multiple workloads using the WTiling algorithm
     *
     * @param splitPool the pool of valid splits returned by this function
     * @param mode the MPE mode
     * @param nWorkloads the number of workloads to generate
     */
    void tileMultipleWl(std::list<DPUWorkloadsWithCyclesSplit>& splitPool, const ExecutionMode mode,
                        const unsigned int nWorkloads) override;

    std::string name() const override {
        return "WTiling";
    }
};

} // namespace VPUNN

#endif // WTILING_H
