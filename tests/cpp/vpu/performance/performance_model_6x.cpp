// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#include "performance_model.h"  // added to test get_bandwidth_cycles_per_bytesLegacy

namespace VPUNN_unit_tests {
using namespace VPUNN;

/// @brief tests for performance.h
class TestHWPerformanceModelNPU6x : public TestHWPerformanceModel_BASICS {
public:
};

TEST_F(TestHWPerformanceModelNPU6x, ArchTest6_0_BasicAssertions) {
    const auto device = VPUNN::VPUDevice::NPU_6_0;
    const auto& hw{hw_info.device(device)};

    EXPECT_EQ(hw.input_channels_mac(), 8u);
    EXPECT_EQ(hw.get_nr_ppe(), 32u);
    EXPECT_EQ(hw.get_nr_macs(), 4096u);
    EXPECT_EQ(hw.get_dpu_fclk(), 1950u);
}

}  // namespace VPUNN_unit_tests