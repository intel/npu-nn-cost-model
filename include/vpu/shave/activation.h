// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_SHV_ACTIVATION_H
#define VPUNN_SHV_ACTIVATION_H

#include <cmath>
#include "vpu/types.h"
#include "vpu/shave_old.h"

namespace VPUNN {

/**
 * @brief A class to compute generic SHV activation cost. Because C++ do not allow
 * floating point template arguments, we need to specify a macro to expand it
 *
 * @tparam kernel efficiency in bytes/cycle
 * @tparam kernel latency in cycles
 */
template <unsigned int efficiency, unsigned int latency>
struct SHVActivation : public SWOperation {
    /**
     * @brief Construct a new SHVActivation object
     *
     * @param device a VPUDevice element
     * @param input a single size array of inputs
     * @param output a single size array of outputs
     */
    SHVActivation(const VPUDevice& device, const VPUTensor& input, const VPUTensor& output)
            : SWOperation(device, {input}, {output}) {
    }

    /**
     * @brief Get the Kernel efficiency
     *
     * @return float
     */
    float getKernelEfficiency() const {
        return float(efficiency) / 1000.0f;
    }

    /**
     * @brief Get the kernel latency
     *
     * @return unsigned int
     */
    unsigned int getLatency() const {
        return latency;
    }

    /**
     * @brief Get the cycles of the shave operation
     *
     * @return unsigned int number of cycles
     */
    unsigned int cycles() const override {
        float size = static_cast<float>(outputs[0].size());
        return static_cast<unsigned int>(std::round(size / getKernelEfficiency())) + getLatency();
    }
};

// Defining a Shave activation kernel
#define SHV_ACTIVATION_CLASS SHVActivation
#define SHV_ACTIVATION_KERNEL(name, efficiency, latency) typedef SHVActivation<int(efficiency * 1000), latency> name
#define SHV_ACTIVATION_KERNEL_DEFAULT(name) SHV_ACTIVATION_KERNEL(name, 1.0, 0)

}  // namespace VPUNN

#endif  // VPUNN_SHV_ACTIVATION_H
