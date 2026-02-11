// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef SHAVE_INSTANCE_HOLDER_FACTORS_H
#define SHAVE_INSTANCE_HOLDER_FACTORS_H

#include "vpu/performance.h"
#include "vpu/shave/shave_collection.h"
#include "vpu/shave/shave_factors_mapping.h"

namespace VPUNN {

/**
 * @brief Template class to create a SHAVE instance holder with speed-up factors
 * This class combines a populated lookup table of speed-up factors with a SHAVE instance holder
 * to create a DeviceShaveContainer that applies the speed-up factors to the SHAVE operations.
 * @tparam PopulatedFactorsLUT generated class type that populates the speed-up factors lookup table, it inherits `FactorsLookUpTable`
 * @tparam InstanceHolder the base SHAVE instance holder class to wrap
 * @tparam deviceGen the VPUDevice enum value representing the target device generation
 */
template <class PopulatedLUT, typename InstanceHolder, VPUDevice deviceGen>
class ShaveInstanceHolderWithFactors : public DeviceShaveContainer {
public:
    using DeviceShaveContainer::getDevice;
    const DeviceShaveContainer& getContainer() const {
        return *this;
    }

    ///@brief Constructor that initializes the SHAVE instance holder with speed-up factors
    ShaveInstanceHolderWithFactors(): DeviceShaveContainer(deviceGen) {
        populate();
    }

protected:
    void populate() {
        const DeviceShaveContainer& shave_container{shave_holder.getContainer()};
        // for all names in the shave container, there will be a speed up factor
        // if it is missing from the `factors_mapping` then it will have default value
        for (const auto& shave_op : shave_container.getShaveList()) {
            const auto& executor{shave_container.getShaveExecutor(shave_op)};
            auto speed_up_factor = factors_map.getOperatorFactor(shave_op);
            this->template AddMock<GlobalHarwdwareCharacteristics::get_dpu_fclk(deviceGen),
                                              GlobalHarwdwareCharacteristics::get_cmx_fclk(deviceGen)>(
                    executor.getName(), executor, speed_up_factor);
        }
    }

private:
    const PopulatedLUT factors_map; // initialize the factors_map that will be already populated
    InstanceHolder shave_holder;
};
}  // namespace VPUNN

#endif
