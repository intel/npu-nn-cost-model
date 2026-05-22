// Copyright © 2026 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_DPU_TILER_FACTORY_H
#define VPUNN_DPU_TILER_FACTORY_H

#include <memory>
#include "vpu_cost_model.h"
#include "dpu_tiler_intf.h"

namespace VPUNN {

class DPUTilerFactory {
public:
    /**
     * @brief Factory function that generates a IDPUTiler instance
     *
     * @param _model a reference to a VPUCostModel object
     * @return std::unique_ptr<IDPUTiler>
     */
    static std::unique_ptr<IDPUTiler> getDPUTiler(const VPUCostModel& _model);
};
} // namespace VPUNN

#endif // DPU_TILER_FACTORY_H
