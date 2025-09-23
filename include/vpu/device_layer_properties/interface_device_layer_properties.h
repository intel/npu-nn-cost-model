// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef INTERFACE_LAYER_PROPERTIES_H
#define INTERFACE_LAYER_PROPERTIES_H

#include <vector>
#include "vpu/types.h"
#include "vpu/vpu_tensor.h"
#include "vpu/vpu_tiling_strategy.h"

namespace VPUNN {
struct DPULayer;  // forward declaration

/**
 * @brief interface for device-specific layer properties
 *
 * This class defines the interface for querying layer property information such as
 * valid tiling strategies and execution modes for a given VPU device. Derived classes
 * implement device-specific logic and data.
 */
/* coverity[rule_of_five_violation:FALSE] */
class ILayerProperties {
public:
    ILayerProperties(const ILayerProperties&) = default;
    ILayerProperties& operator=(const ILayerProperties&) = delete;

    ILayerProperties(ILayerProperties&&) = default;
    ILayerProperties& operator=(ILayerProperties&&) = delete;


protected:
    ILayerProperties() = default;
    virtual ~ILayerProperties() = default;
  

public:
    /**
     * @brief Returns the valid execution modes for tiling for a given DPULayer
     * @param wl The DPULayer
     * @return A vector of supported ExecutionMode values
     */
    virtual const std::vector<ExecutionMode> getValidTilingExecutionMode(const DPULayer& wl) const=0;

    /**
     * @brief Returns the default execution mode for a given tensor
     * @param tensor The VPUTensor
     * @return The default ExecutionMode
     */
    virtual ExecutionMode getValidDefaultExecutionMode(const VPUTensor&) const = 0;

    /**
     * @brief Returns the valid tiling strategies for a certain device
     * @return A vector of supported VPUTilingStrategy values
     */
    const virtual std::vector<VPUTilingStrategy> getValidTilingStrategies() const = 0;
};

}  // namespace VPUNN

#endif
