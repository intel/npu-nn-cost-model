// Copyright © 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#include "vpu/shave/ShaveModel1to1.h"
#include "vpu/shave/shave_collection.h"
#include "vpu/shave/shave_devices.h"

#include <gtest/gtest.h>
#include "vpu/types.h"

namespace VPUNN_unit_tests {
using namespace VPUNN;

class ShaveModel1to1Test : public ::testing::Test {
protected:
    void SetUp() override {
    }
    //  ShaveModel1to1 modelUnroll16{8,         16,        1300,   /*1300,*/ VPUNN::DataType::FLOAT16,
    //                               0.000161f /*slope*/, 2.489317f /*intercept*/, 0.234f /*ofScalar*/, 0.632571f
    //                               /*ofUnroll*/};
    ShaveModel1to1 modelUnroll16{VPUNN::DataType::FLOAT16,
                                 0.000161f /*slope*/,
                                 2.489317f /*intercept*/,
                                 0.234f /*ofScalar*/,
                                 0.632571f /*ofUnroll*/,
                                 8,   // vectsize
                                 16,  // unroll size
                                 1300,
                                 975};
    ShaveModel1to1 modelUnroll8{VPUNN::DataType::FLOAT16, 0.0001398f, 2.740062f, 0.208f, 0.590509f, 8, 8, 1300, 975};
    // NOUNROLL = 1
    ShaveModel1to1 modelNoUnroll{VPUNN::DataType::FLOAT16, 0.000579f, 2.395082f, 0.078f, 0.0f, 8, 1, 1300, 975};
};
// Test case for is_in_first_block method
TEST_F(ShaveModel1to1Test, IsInFirstBlock) {
    // Test is in first block with size = 32
    EXPECT_TRUE(modelUnroll16.is_in_first_block_of_operations(32));
    // Test is in first block with size = 129
    EXPECT_FALSE(modelUnroll16.is_in_first_block_of_operations(129));

    // Test is in first block with size = 16
    EXPECT_TRUE(modelUnroll8.is_in_first_block_of_operations(16));
    // Test is in first block with size = 65
    EXPECT_FALSE(modelUnroll8.is_in_first_block_of_operations(65));

    // Test is in first block with size = 64
    EXPECT_FALSE(modelNoUnroll.is_in_first_block_of_operations(64));
}

// Test case for is_scalar_value method
TEST_F(ShaveModel1to1Test, IsScalarValue) {
    // Test is scalar value with size = 129
    EXPECT_TRUE(modelUnroll16.is_scalar_value(129));
    // Test is scalar value with size = 136
    EXPECT_FALSE(modelUnroll16.is_scalar_value(136));

    // Test is scalar value with size = 129
    EXPECT_TRUE(modelUnroll8.is_scalar_value(129));
    // Test is scalar value with size = 136
    EXPECT_FALSE(modelUnroll8.is_scalar_value(136));

    // Test is scalar value with size = 129
    EXPECT_TRUE(modelNoUnroll.is_scalar_value(129));
    // Test is scalar value with size = 136
    EXPECT_FALSE(modelNoUnroll.is_scalar_value(136));
}

// Test case for is_first_value_in_block method
TEST_F(ShaveModel1to1Test, IsFirstValueInBlock) {
    // Test is first value in block with size = 128
    EXPECT_TRUE(modelUnroll16.is_first_value_in_block(128));
    // Test is first value in block with size = 129
    EXPECT_FALSE(modelUnroll16.is_first_value_in_block(129));

    // Test is first value in block with size = 64
    EXPECT_TRUE(modelUnroll8.is_first_value_in_block(64));
    // Test is first value in block with size = 65
    EXPECT_FALSE(modelUnroll8.is_first_value_in_block(65));

    // Test is first value in block with size = 64
    EXPECT_FALSE(modelNoUnroll.is_first_value_in_block(64));
}

TEST_F(ShaveModel1to1Test, GetMicroSeconds) {
    // Test for generating second coordinate with x = 2112
    EXPECT_NEAR(modelUnroll16.getMicroSeconds(2112), 2.829f, 0.001);
    // Test for generating second coordinate with x = 180288
    EXPECT_NEAR(modelUnroll16.getMicroSeconds(180288), 31.516f, 0.001);

    // Test for generating second coordinate with x = 5344
    EXPECT_NEAR(modelUnroll8.getMicroSeconds(5344), 3.487f, 0.001);
    // Test for generating second coordinate with x = 237984
    EXPECT_NEAR(modelUnroll8.getMicroSeconds(237984), 36.010f, 0.001);

    // Test for generating second coordinate with x = 806
    EXPECT_NEAR(modelNoUnroll.getMicroSeconds(806), 2.939f, 0.001);
    // Test for generating second coordinate with x = 261140
    EXPECT_NEAR(modelNoUnroll.getMicroSeconds(261140), 153.673f, 0.001);
}

TEST_F(ShaveModel1to1Test, GetDpuCycles) {
    // Test for transforming duration in DPU cycles with x = 2112
    EXPECT_NEAR(modelUnroll16.getDPUCycles(2112), 3679, 1);
    // Test for transforming duration in DPU cycles with x = 180288
    EXPECT_NEAR(modelUnroll16.getDPUCycles(180288), 40971, 1);

    // Test for transforming duration in DPU cycles with x = 5344
    EXPECT_NEAR(modelUnroll8.getDPUCycles(5344), 4534, 1);
    // Test for transforming duration in DPU cycles with x = 237984
    EXPECT_NEAR(modelUnroll8.getDPUCycles(237984), 46814, 1);

    // Test for transforming duration in DPU cycles with x = 806
    EXPECT_NEAR(modelNoUnroll.getDPUCycles(806), 3822, 1);
    // Test for transforming duration in DPU cycles with x = 261140
    EXPECT_NEAR(modelNoUnroll.getDPUCycles(261140), 199776, 1);
}

class ShaveCollectionTest : public ::testing::Test {
protected:
    void SetUp() override {
    }
    const SHAVEWorkload ref_shv_wrkld{"sigmoid",
                                      VPUDevice::VPU_2_7,
                                      {VPUTensor(10, 100, 5, 1, DataType::FLOAT16)},
                                      {VPUTensor(10, 100, 5, 1, DataType::FLOAT16)}};

    struct Test {
        SHAVEWorkload input;

        bool expected_exist = true;
        CyclesInterfaceType exp_cycles = 0;
    };

    void ExecuteTest(const DeviceShaveContainer& list, const Test& test) {
        const std::string fn{test.input.get_name()};
        ASSERT_EQ(list.existsShave(fn), test.expected_exist) << fn;
        if (test.expected_exist) {
            const auto& shaveOp = list.getShaveExecutor(fn);
            ASSERT_EQ(shaveOp.getName(), fn) << fn;

            const SHAVEWorkload& swl{test.input};
            CyclesInterfaceType cycles = shaveOp.dpuCycles(swl);

            if (test.exp_cycles != 0) {
                EXPECT_EQ(cycles, test.exp_cycles) << swl;
            } else {
                EXPECT_GT(cycles, 0) << swl;
            }
        }
    }
};
TEST_F(ShaveCollectionTest, instanceHolder_Smoke) {
    ShaveInstanceHolder_VPU27 ih;
    EXPECT_EQ(ih.getDevice(), VPUDevice::VPU_2_7);
    const DeviceShaveContainer& list = ih.getContainer();

    // list.find
    EXPECT_EQ(list.getDevice(), VPUDevice::VPU_2_7);
    // EXPECT_EQ(list.map.size(), 2);

    const std::string fn{"sigmoid"};
    // const auto it = list.map.find(fn);

    // EXPECT_NE(it, list.map.end());
    EXPECT_EQ(list.existsShave(fn), true);

    // auto& shaveOp = it->second;  // list[fn];
    const auto& shaveOp = list.getShaveExecutor(fn);
    const auto ss = shaveOp.getName();
    EXPECT_EQ(ss, fn);
    EXPECT_EQ(shaveOp.getName(), fn);

    SHAVEWorkload swl("sigmoid", VPUDevice::VPU_2_7, {VPUTensor(10, 100, 5, 1, DataType::FLOAT16)},
                      {VPUTensor(10, 100, 5, 1, DataType::FLOAT16)});
    CyclesInterfaceType cycles = shaveOp.dpuCycles(swl);
    EXPECT_GT(cycles, 0);
    // delete &shaveOp;  // not good
}

TEST_F(ShaveCollectionTest, instanceHolder27_Smoke) {
    ShaveInstanceHolder_VPU27 ih;
    EXPECT_EQ(ih.getDevice(), VPUDevice::VPU_2_7);
    const DeviceShaveContainer& list = ih.getContainer();
    const VPUDevice testDev{VPUDevice::VPU_2_7};

    std::vector<Test> tests{
            {{"sigmoid", testDev, ref_shv_wrkld.get_inputs(), ref_shv_wrkld.get_outputs()}, true, 0},
            {{"hardswish", testDev, ref_shv_wrkld.get_inputs(), ref_shv_wrkld.get_outputs()}, true, 0},
            {{"logicaland", testDev, ref_shv_wrkld.get_inputs(), ref_shv_wrkld.get_outputs()}, true, 0},
            {{"PermuteQuantize", testDev, ref_shv_wrkld.get_inputs(), ref_shv_wrkld.get_outputs()}, false, 0},
    };
    // Test x{{"PermuteQuantize", testDev, ref_shv_wrkld.get_inputs(), ref_shv_wrkld.get_outputs()}, false, 0}

    // list.find
    EXPECT_EQ(list.getDevice(), VPUDevice::VPU_2_7);
    // EXPECT_EQ(list.map.size(), 2);

    for (const auto& t : tests) {
        ExecuteTest(list, t);
    }
}

TEST_F(ShaveCollectionTest, instanceHolder27_Classic_Smoke) {
    ShaveInstanceHolder_VPU27CLassic ih;
    EXPECT_EQ(ih.getDevice(), VPUDevice::VPU_2_7);
    const DeviceShaveContainer& list = ih.getContainer();

    // list.find
    EXPECT_EQ(list.getDevice(), VPUDevice::VPU_2_7);
    // EXPECT_EQ(list.map.size(), 2);

    const VPUDevice testDev{VPUDevice::VPU_2_7};

    std::vector<Test> tests{
            {{"HardSigmoid", testDev, ref_shv_wrkld.get_inputs(), ref_shv_wrkld.get_outputs()}, true, 23238},
            {{"Transpose", testDev, ref_shv_wrkld.get_inputs(), ref_shv_wrkld.get_outputs()}, true, 10000},
            {{"Minimum", testDev, ref_shv_wrkld.get_inputs(), ref_shv_wrkld.get_outputs()}, true, 677714},

            {{"TransposeXX", testDev, ref_shv_wrkld.get_inputs(), ref_shv_wrkld.get_outputs()}, false, 0},
    };

    for (const auto& t : tests) {
        ExecuteTest(list, t);
    }
}

class ShaveDevicesTest : public ::testing::Test {
protected:
    void SetUp() override {
    }

private:
    const ShaveConfiguration
            shaves2{};  // must be {} value initized here if const otherwise no default ctor for test class.
public:
};

TEST_F(ShaveDevicesTest, config_Smoke) {
    const ShaveConfiguration shaves;

    {
        VPUDevice d = VPUDevice::VPU_2_0;
        const auto names = shaves.getShaveSupportedOperations(d);
        const auto count = names.size();
        EXPECT_GT(count, 2);
    }
    {
        VPUDevice d = VPUDevice::VPU_2_7;
        const auto names = shaves.getShaveSupportedOperations(d);
        const auto count = names.size();
        EXPECT_GT(count, 4);
    }
    // ShaveConfiguration shavesCopy(shaves);// non copiable?

    DeviceShaveContainer empty_shavesA{VPUDevice::__size};
    DeviceShaveContainer empty_shavesB{VPUDevice::VPU_2_0};
    DeviceShaveContainer empty_shavesC{std::move(empty_shavesA)};
    // empty_shavesB = empty_shavesC;

    // DeviceShaveContainer empty_shavesD{empty_shavesB};
}

}  // namespace VPUNN_unit_tests