// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#include "vpu/hw_characteristics/HW_characteristics_supersets.h"
#include "vpu/performance.h"

#include <gtest/gtest.h>
#include "common/common_helpers.h"
#include "vpu/types.h"

#include "vpu/dma_theoretical_cost_provider.h"  // added to test get_bandwidth_cycles_per_bytesLegacy

/// @brief namespace for Unit tests of the C++ library
namespace VPUNN_unit_tests {
using namespace VPUNN;

/// @brief tests for performance.h
class TestHWPerformanceModel_BASICS : public ::testing::Test {
public:
protected:
    static constexpr int evoX{0};  // factor to adjust expectation between evo 0 and 1
    void SetUp() override {
    }
    // instantiate(or ref) the legacy set of configuration
    const IHWCharacteristicsSet& hw_info_legacy = HWCharacteristicsSuperSets::get_legacyConfigurationRef();

    const IHWCharacteristicsSet& hw_info{HWCharacteristicsSuperSets::mainConfiguration()};  // the default one

    const IHWCharacteristicsSet& hw_info_evo0{HWCharacteristicsSuperSets::mainEvo0Configuration()};
    const IHWCharacteristicsSet& hw_info_evo1{HWCharacteristicsSuperSets::mainEvo1Configuration()};

    DMATheoreticalCostProvider_LNL_Legacy dma;

private:
};
}  // namespace VPUNN_unit_tests