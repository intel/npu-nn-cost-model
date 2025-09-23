// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef SHAVE_THEORETICAL_COST_PROVIDER_H
#define SHAVE_THEORETICAL_COST_PROVIDER_H

#include "vpu/types.h"
#include "vpu/performance.h"
#include "vpu/shave_old.h"

namespace VPUNN {

/**
 * @class ShaveTheoreticalCostProvider
 * @brief Provides theoretical performance modeling for SHAVE kernel workloads.
 *
 * This class estimates the number of execution cycles required for SHAVE kernel operations.
 * It provides methods to compute the theoretical execution cycles for a given software operation,
 * based on the operation's parameters and output configuration.
 *
 * An instance of this class is intended to be use as a provider for theoretical cost for SWOperations
 * An example of usage can be seen in class VPUCostModel where we either need just theoretical cost or 
 * we use this as a fallback when NN cost not available
 */
class ShaveTheoreticalCostProvider {
public:
    /**
     * @brief Compute the Shave Kernel theoretical cycles
     *
     * @param swl a Shave Kernel
     * @return unsigned int theoretical execution cycles
     */
    unsigned int SHAVETheoreticalCycles(const SWOperation& swl) const {
        if (swl.outputs.size() == 0) {  // If it computes no output, its duration is 0 cycles
            return 0;
        }
        return swl.cycles();
    }

};

}  // namespace VPUNN

#endif
