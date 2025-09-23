// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.
#ifndef VPUNN_HW_CHARACTERISTICS_SUPERSETS_H
#define VPUNN_HW_CHARACTERISTICS_SUPERSETS_H

#include "vpu/hw_characteristics/HW_characteristics_set_base.h"
#include "vpu/hw_characteristics/device_HW_characteristics_const_repo.h"
#include "vpu/hw_characteristics/itf_HW_characteristics_set.h"

namespace VPUNN {

/// \brief Holds multiple HW characteristics configurations (main, legacy, etc.)
class HWCharacteristicsSuperSets {
public:
    using MainEvo0SetType = Base_HWCharacteristicsSet<DeviceHWCHaracteristicsConstRepo::MainEvo0SetTuple,
                                                      DeviceHWCHaracteristicsConstRepo::IndexMap>;

    using MainEvo1SetType = Base_HWCharacteristicsSet<DeviceHWCHaracteristicsConstRepo::MainEvo1SetTuple,
                                                      DeviceHWCHaracteristicsConstRepo::IndexMap>;

    // Legacy configuration tuple type
    using LegacySetType = Base_HWCharacteristicsSet<DeviceHWCHaracteristicsConstRepo::LegacySetTuple,
                                                    DeviceHWCHaracteristicsConstRepo::IndexMap>;

    // the default one
    static const auto& mainConfiguration() {
        return mainEvo0Configuration();
    }

    static const IHWCharacteristicsSet& get_mainConfigurationRef() {
        return mainConfiguration();
    }

    static const MainEvo0SetType& mainEvo0Configuration() {
        return mainEvo0Config_;
    }
    // add access also to the pure interface
    static const IHWCharacteristicsSet& get_mainEvo0ConfigurationRef() {
        return mainEvo0Configuration();
    }

    static const MainEvo1SetType& mainEvo1Configuration() {
        return mainEvo1Config_;
    }
    // add access also to the pure interface
    static const IHWCharacteristicsSet& get_mainEvo1ConfigurationRef() {
        return mainEvo1Configuration();
    }

    // legacy modes
    static const LegacySetType& legacyConfiguration() {
        return legacyConfig_;
    }
    // access to legacy
    static const IHWCharacteristicsSet& get_legacyConfigurationRef() {
        return legacyConfiguration();
    }

protected:
    // instances of HW characteristics for main and legacy configurations
    inline static const MainEvo0SetType mainEvo0Config_{};
    inline static const MainEvo1SetType mainEvo1Config_{};  // with dma tx imperfection

    inline static const LegacySetType legacyConfig_{};
};

}  // namespace VPUNN

#endif  //
