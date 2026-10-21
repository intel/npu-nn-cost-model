// SPDX-License-Identifier: Apache 2.0

#include "vpu/validation/checker_utils.h"

#include "vpu_cost_model.h"

#include <gtest/gtest.h>

#include <iostream>

namespace VPUNN_unit_tests {
using namespace VPUNN;

namespace {
class ScopedCheckerPrintTags {
public:
    explicit ScopedCheckerPrintTags(const bool print_tags): previous_print_tags_{Checker::set_print_tags(print_tags)} {
    }

    ScopedCheckerPrintTags(const ScopedCheckerPrintTags&) = delete;
    ScopedCheckerPrintTags& operator=(const ScopedCheckerPrintTags&) = delete;

    ~ScopedCheckerPrintTags() {
        Checker::set_print_tags(previous_print_tags_);
    }

private:
    bool previous_print_tags_;
};

DPUWorkload make_recreate_ticket_dwconv_workload(const VPUDevice device) {
    DPUWorkload wl{
            device,
            Operation::DW_CONVOLUTION,
            {VPUTensor(65, 2, 21, 15, DataType::INT8)},
            {VPUTensor(65, 1, 21, 15, DataType::INT8)},
            {2, 1},
            {2, 1},
            {0, 0, 0, 0},
            ExecutionMode::CUBOID_8x16,
    };
    return wl;
}
}  // namespace

TEST(CheckerUtilsTest, RecreateTicketErrors_Tags) {
    VPUCostModel model{VPUDevice::NPU_5_0, NPU_5_0_MODEL_PATH};
    const ScopedCheckerPrintTags print_tags_guard{true};

    auto wl = make_recreate_ticket_dwconv_workload(VPUDevice::NPU_5_0);
    std::string info;
    const auto cycles = model.DPU(wl, info);

    //std::cout << info << std::endl;

    EXPECT_EQ(cycles, Cycles::ERROR_INVALID_INPUT_CONFIGURATION);
    EXPECT_EQ(info,
              "[CHECK FAILED]: input_0.channels is not ok because:  Range 0 check failed:  Value :21 is not divisible by 16! [END CHECK]; "
              "[CHECK FAILED]: input_0.batch is not ok because:  Range 0 check failed:  Value :15 is not in interval [1, 1] [END CHECK]; "
              "[CHECK FAILED]: output_0.width with value: 65 is not found in interval: [32, 32] [END CHECK]; "
              "[CHECK FAILED]: output_0.height with value: 1 is not found in interval: [2, 2] [END CHECK]; "
              "[CHECK FAILED]: output_0.channels is not ok because:  Range 0 check failed:  Value :21 is not divisible by 16! [END CHECK]; "
              "[CHECK FAILED]: output_0.batch is not ok because:  Range 0 check failed:  Value :15 is not in interval [1, 1] [END CHECK]; "
              "[CHECK FAILED]: Execution_Order with value: 4 {CUBOID_8x16} is not found in allowed list: [ 3 {CUBOID_16x16} , ] [END CHECK]");
}

TEST(CheckerUtilsTest, RecreateTicketErrors_NoTags) {
    VPUCostModel model{VPUDevice::NPU_5_0, NPU_5_0_MODEL_PATH};
    const ScopedCheckerPrintTags print_tags_guard{false};

    auto wl = make_recreate_ticket_dwconv_workload(VPUDevice::NPU_5_0);
    std::string info;
    const auto cycles = model.DPU(wl, info);

    //std::cout << info << std::endl;

    EXPECT_EQ(cycles, Cycles::ERROR_INVALID_INPUT_CONFIGURATION);
    EXPECT_EQ(info,
              "input_0.channels is not ok because:  Range 0 check failed:  Value :21 is not divisible by 16!; "
              "input_0.batch is not ok because:  Range 0 check failed:  Value :15 is not in interval [1, 1]; "
              "output_0.width with value: 65 is not found in interval: [32, 32]; "
              "output_0.height with value: 1 is not found in interval: [2, 2]; "
              "output_0.channels is not ok because:  Range 0 check failed:  Value :21 is not divisible by 16!; "
              "output_0.batch is not ok because:  Range 0 check failed:  Value :15 is not in interval [1, 1]; "
              "Execution_Order with value: 4 {CUBOID_8x16} is not found in allowed list: [ 3 {CUBOID_16x16} , ]");
}

TEST(CheckerUtilsTest, RecreateTicketErrors_VPU2_7_Tags) {
    VPUCostModel model{VPUDevice::VPU_2_7, VPU_2_7_MODEL_PATH};
    const ScopedCheckerPrintTags print_tags_guard{true};

    auto wl = make_recreate_ticket_dwconv_workload(VPUDevice::VPU_2_7);
    std::string info;
    const auto cycles = model.DPU(wl, info);

    //std::cout << info << std::endl;

    EXPECT_EQ(cycles, Cycles::ERROR_INVALID_INPUT_CONFIGURATION);
    EXPECT_EQ(info,
              "[CHECK FAILED]: input_0.channels is not ok because:  Range 0 check failed:  Value :21 is not divisible by 16! [END CHECK]; "
              "[CHECK FAILED]: output_0.width with value: 65 is not found in interval: [32, 32] [END CHECK]; "
              "[CHECK FAILED]: output_0.height with value: 1 is not found in interval: [2, 2] [END CHECK]; "
              "[CHECK FAILED]: output_0.channels is not ok because:  Range 0 check failed:  Value :21 is not divisible by 16! [END CHECK]");
}

TEST(CheckerUtilsTest, RecreateTicketErrors_VPU2_7_NestedTags) {
    VPUCostModel model{VPUDevice::VPU_2_7, VPU_2_7_MODEL_PATH};
    const ScopedCheckerPrintTags print_tags_guard{true};

    auto wl = make_recreate_ticket_dwconv_workload(VPUDevice::VPU_2_7);
    wl.weight_sparsity = 1.0F;
    wl.weight_sparsity_enabled = true;
    std::string info;
    const auto cycles = model.DPU(wl, info);

    //std::cout << info << std::endl;

    EXPECT_EQ(cycles, Cycles::ERROR_INVALID_INPUT_CONFIGURATION);
    EXPECT_EQ(info.find("[CHECK FAILED]: [CHECK FAILED]"), std::string::npos);
    EXPECT_EQ(info.find("[END CHECK] [END CHECK]"), std::string::npos);
    EXPECT_EQ(info,
              "[CHECK FAILED]: input_0.channels is not ok because:  Range 0 check failed:  Value :21 is not divisible by 16! [END CHECK]; "
              "[CHECK FAILED]: output_0.width with value: 65 is not found in interval: [32, 32] [END CHECK]; "
              "[CHECK FAILED]: output_0.height with value: 1 is not found in interval: [2, 2] [END CHECK]; "
              "[CHECK FAILED]: output_0.channels is not ok because:  Range 0 check failed:  Value :21 is not divisible by 16! [END CHECK]; "
              "[CHECK FAILED]: input_1 sparsity_enabled   with value: 1 is not equal to : 0 [END CHECK]; "
              "[CHECK FAILED]: input_1 sparsity   with value: 1 is not equal to : 0 [END CHECK]");
}

TEST(CheckerUtilsTest, RecreateTicketErrors_VPU2_7_NoTags) {
    VPUCostModel model{VPUDevice::VPU_2_7, VPU_2_7_MODEL_PATH};
    const ScopedCheckerPrintTags print_tags_guard{false};

    auto wl = make_recreate_ticket_dwconv_workload(VPUDevice::VPU_2_7);
    std::string info;
    const auto cycles = model.DPU(wl, info);

    //std::cout << info << std::endl;

    EXPECT_EQ(cycles, Cycles::ERROR_INVALID_INPUT_CONFIGURATION);
    EXPECT_EQ(info,
              "input_0.channels is not ok because:  Range 0 check failed:  Value :21 is not divisible by 16!; "
              "output_0.width with value: 65 is not found in interval: [32, 32]; "
              "output_0.height with value: 1 is not found in interval: [2, 2]; "
              "output_0.channels is not ok because:  Range 0 check failed:  Value :21 is not divisible by 16!");
}



}  // namespace VPUNN_unit_tests