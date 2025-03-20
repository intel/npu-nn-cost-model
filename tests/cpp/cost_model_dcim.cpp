// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.
#include "vpu_cost_model.h"

#include <gtest/gtest.h>
#include "common_helpers.h"
#include "vpu/compatibility/types01.h"
#include "vpu/cycles_interface_types.h"
#include "vpu/sample_generator/random_task_generator.h"
#include "vpu/validation/interface_valid_values.h"
#include "vpu_dma_cost_model.h"

#include "vpu/validation/dpu_operations_validator.h"
#include "vpu/validation/memory_calculator.h"

#include <algorithm>
#include <unordered_map>

#include <optional>
#include <variant>

// #include "cost_model_test.h"

/// @brief namespace for Unit tests of the C++ library
namespace VPUNN_unit_tests {
using namespace VPUNN;

class TestDCIMCostModel : public ::testing::Test {
public:
protected:
    VPUCostModel model{};  // empty model

    void SetUp() override {
        Logger::clear2ndlog();
        // Logger::activate2ndlog();
    }
    void TearDown() override {
        Logger::clear2ndlog();
        Logger::deactivate2ndlog();
    }

protected:
    // list of known models
    VPUCostModel g_model_invalid{""};
    VPUCostModel g_model_2_0{VPU_2_0_MODEL_PATH};
    VPUCostModel g_model_2_7{VPU_2_7_MODEL_PATH};
    VPUCostModel g_model_4_0{VPU_4_0_MODEL_PATH};

    VPUCostModel& getModel(const VPUDevice device) {
        switch (device) {
        case VPUDevice::VPU_2_0:
        case VPUDevice::VPU_2_1:
            return g_model_2_0;
        case VPUDevice::VPU_2_7:
            return g_model_2_7;
        case VPUDevice::VPU_4_0:
            return g_model_4_0;
        default:
            return g_model_invalid;
            break;
        };
    };
};

TEST_F(TestDCIMCostModel, Interface_acces_smoke) {
    ASSERT_NO_THROW(model.getDCiM_interface());
    auto& dcimCM = model.getDCiM_interface();

    DCIMWorkload workload{};
    const DCIMWorkload& workload_ref{workload};
    std::string info{};

    const CyclesInterfaceType cycles{dcimCM.dCiM(workload_ref, info)};

    ASSERT_EQ(cycles, Cycles::ERROR_INVALID_INPUT_CONFIGURATION);
    ASSERT_EQ(info, "");
}

}  // namespace VPUNN_unit_tests
