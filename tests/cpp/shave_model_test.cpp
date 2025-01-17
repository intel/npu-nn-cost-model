// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.
#include <gtest/gtest.h>
#include "vpu/shave/shave_op_executors.h"

#include "common_helpers.h"
#include "vpu/shave/MVNModel.h"
#include "vpu/shave/NormalizeL2Model.h"
#include "vpu/shave/ShaveModel1to1.h"
#include "vpu/shave/SoftmaxModel.h"
#include "vpu/shave/GatherModel.h"
#include "vpu/shave/shave_collection.h"
#include "vpu/shave/shave_devices.h"
#include "vpu/types.h"
#include "vpu_cost_model.h"

#include <fstream>
#include <iostream>

namespace VPUNN_unit_tests {
using namespace VPUNN;

class ShaveModel1to1Test : public ::testing::Test {
protected:
    void SetUp() override {
    }

    // mock for exposing protected methods to be tested
    class Shave_gateway : public ShaveModel1to1 {
    public:
        using ShaveModel1to1::is_first_value_in_block;
        using ShaveModel1to1::is_in_first_block_of_operations;
        using ShaveModel1to1::is_scalar_value;
        using ShaveModel1to1::ShaveModel1to1;
    };
    //  ShaveModel1to1 modelUnroll16{8,         16,        1300,   /*1300,*/ VPUNN::DataType::FLOAT16,
    //                               0.000161f /*slope*/, 2.489317f /*intercept*/, 0.234f /*ofScalar*/, 0.632571f
    //                               /*ofUnroll*/};
    Shave_gateway modelUnroll16{VPUNN::DataType::FLOAT16,
                                0.000161f /*slope*/,
                                2.489317f /*intercept*/,
                                0.234f /*ofScalar*/,
                                0.632571f /*ofUnroll*/,
                                8,   // vectsize
                                16,  // unroll size
                                1300,
                                975};
    Shave_gateway modelUnroll8{VPUNN::DataType::FLOAT16, 0.0001398f, 2.740062f, 0.208f, 0.590509f, 8, 8, 1300, 975};
    // NOUNROLL = 1
    Shave_gateway modelNoUnroll{VPUNN::DataType::FLOAT16, 0.000579f, 2.395082f, 0.078f, 0.0f, 8, 1, 1300, 975};

    Shave_gateway modelDefault{VPUNN::DataType::FLOAT16, 0.000384615f, 0.0f, 0.0f, 0.0f, 1, 1, 1300, 975};
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

TEST_F(ShaveModel1to1Test, DefaultCaseCyclesTest) {
    EXPECT_NEAR(modelDefault.getDPUCycles(2112), 2112 / 2, 1);
    EXPECT_NEAR(modelDefault.getDPUCycles(237984), 237984 / 2, 1);
    EXPECT_NEAR(modelDefault.getDPUCycles(806), 806 / 2, 1);
}

class ShaveModel1to1NPU40Test : public ::testing::Test {
protected:
    void SetUp() override {
    }

    class Shave_gateway_NPU40 : public ShaveModel1to1NPU40 {
    public:
        using ShaveModel1to1NPU40::calculate_intra_block_offset;
        using ShaveModel1to1NPU40::calculate_vector_offset;
        using ShaveModel1to1NPU40::is_in_first_block_of_operations;
        using ShaveModel1to1NPU40::ShaveModel1to1NPU40;
    };

    Shave_gateway_NPU40 modelUnroll16{VPUNN::DataType::FLOAT16,
                                      4.019940428991265e-05f,
                                      3.3992584112239674f,
                                      0.573f,
                                      0.052000000000000046f,
                                      0.10499999999999998f,
                                      1,
                                      32,
                                      16,
                                      1700,
                                      971};
    Shave_gateway_NPU40 modelUnroll8{VPUNN::DataType::FLOAT16,
                                     4.019940428991265e-05f,
                                     3.3992584112239674f,
                                     0.573f,
                                     0.052000000000000046f,
                                     0.10499999999999998f,
                                     1,
                                     32,
                                     8,
                                     1700,
                                     971};
    Shave_gateway_NPU40 modelNoUnroll{VPUNN::DataType::FLOAT16,
                                      4.019940428991265e-05f,
                                      3.3992584112239674f,
                                      0.573f,
                                      0.052000000000000046f,
                                      0.10499999999999998f,
                                      1,
                                      32,
                                      0,
                                      1700,
                                      971};
    Shave_gateway_NPU40 modelUnroll16NoDisplacement{VPUNN::DataType::FLOAT16,
                                                    4.019940428991265e-05f,
                                                    3.3992584112239674f,
                                                    0.573f,
                                                    0.052000000000000046f,
                                                    0.10499999999999998f,
                                                    0,
                                                    32,
                                                    16,
                                                    1700,
                                                    971};
    Shave_gateway_NPU40 modelDefault{
            VPUNN::DataType::FLOAT16, 0.000294118f, 0.0f, 0.0f, 0.0f, 0.0f, 0, 1, 1, 1700, 971};
};

// Test case for is_in_first_block method
TEST_F(ShaveModel1to1NPU40Test, IsInFirstBlock) {
    // Test is in first block with size = 32
    EXPECT_TRUE(modelUnroll16.is_in_first_block_of_operations(32));
    // Test is in first block with size = 129
    EXPECT_FALSE(modelUnroll16.is_in_first_block_of_operations(513));

    // Test is in first block with size = 16
    EXPECT_TRUE(modelUnroll8.is_in_first_block_of_operations(16));
    // Test is in first block with size = 65
    EXPECT_FALSE(modelUnroll8.is_in_first_block_of_operations(257));

    // Test is in first block with size = 64
    EXPECT_FALSE(modelNoUnroll.is_in_first_block_of_operations(64));
}

TEST_F(ShaveModel1to1NPU40Test, IntrablockOffsetCalculator) {
    EXPECT_NEAR(modelUnroll16.calculate_intra_block_offset(577), 0.0069f, 0.001)
            << "returned:" << modelUnroll16.calculate_intra_block_offset(513);

    EXPECT_NEAR(modelUnroll8.calculate_intra_block_offset(577), 0.01485f, 0.001)
            << "returned:" << modelUnroll8.calculate_intra_block_offset(513);

    // Test to get same time with or without displacement
    EXPECT_EQ(modelUnroll16.calculate_intra_block_offset(577),
              modelUnroll16NoDisplacement.calculate_intra_block_offset(576));
}

TEST_F(ShaveModel1to1NPU40Test, VectorOffsetCalculator) {
    EXPECT_NEAR(modelUnroll16.calculate_vector_offset(515), 0.0067f, 0.0001)
            << "returned:" << modelUnroll16.calculate_intra_block_offset(515);

    EXPECT_NEAR(modelUnroll8.calculate_vector_offset(515), 0.0067f, 0.0001)
            << "returned:" << modelUnroll8.calculate_intra_block_offset(515);

    // Test to get same time with or without displacement
    EXPECT_EQ(modelUnroll16.calculate_vector_offset(516), modelUnroll16NoDisplacement.calculate_vector_offset(515));
}

TEST_F(ShaveModel1to1NPU40Test, DefaultCaseCyclesTest) {
    EXPECT_NEAR(modelDefault.getDPUCycles(2112), 2112 / 2, 1);
    EXPECT_NEAR(modelDefault.getDPUCycles(237984), 237984 / 2, 1);
    EXPECT_NEAR(modelDefault.getDPUCycles(806), 806 / 2, 1);
}

class ShaveGatherTest : public ::testing::Test {
protected:
    using Dimensions = std::array<int, 4>;  ///< size of selected dimensions;  Unused axis have to be ONE! OR ZERO!?

    void SetUp() override {
    }
    class Gather_gateway : public GatherModel {
    public: 
        using GatherModel::compute_vector_offset;
		using GatherModel::GatherModel;
	};
    const Gather_gateway model{
            VPUNN::DataType::FLOAT16,
            0.001883862f,
            4.790410425f,
            0.219886254182408f,
            0.0945429470829462f,
            0.013678893f,
            8,1700,971
    };
};
TEST_F(ShaveGatherTest, ComputeVectorOffTest) {
    // Testing under vector size
    EXPECT_NEAR(model.compute_vector_offset(40960, 2), 280.144f, 0.5f);
    // Testing over vector size
    EXPECT_NEAR(model.compute_vector_offset(40960, 20), 112.0574933f, 0.5f);
    // Test for 0
    EXPECT_EQ(model.compute_vector_offset(40960, 8), 0.0f);
}

TEST_F(ShaveGatherTest, GetMicroSecondsTest) {
    // All in innermost
    {
        Dimensions dims{40960, 1, 1, 1};
        EXPECT_NEAR(model.getMicroSeconds(40960, dims), 81.95340371f, 0.5f);
    }
    // All in inter
    {
        Dimensions dims = {1, 40960, 1, 1};
        EXPECT_NEAR(model.getMicroSeconds(40960, dims), 3954.432516f, 0.5f);
    }
    // Mix
    {
        Dimensions dims = {70, 7, 1, 64};
        EXPECT_NEAR(model.getMicroSeconds(31360, dims), 115.0577437, 0.5f);
    }
}
class ShaveSoftmaxTest : public ::testing::Test {
protected:
    void SetUp() override {
    }

    class Softmax_gateway : public SoftmaxModel {
    public:
        using SoftmaxModel::getEquationType;
        using SoftmaxModel::normalizeUnselectedValue;
        using SoftmaxModel::SoftmaxModel;
    };

    const SoftmaxEquationParams e1{
            {0.002682965f, 0.009700778f},  // Slope eq
            {0.09399437f, 12.44775737f}    // Intercept eq
    };
    const SoftmaxEquationParams e2{
            {0.002430508f, 0.010895337f},  // Slope eq
            {0.084965949f, 12.6691893f}    // Intercept eq
    };
    const SoftmaxEquationParams e4{
            {0.002367099f, 0.009812789f},  // Slope eq
            {0.083526381f, 12.45620083f}   // Intercept eq
    };
    const SoftmaxEquationParams e8{
            {0.001992061f, 0.010255307f},  // Slope eq
            {0.067058858f, 12.47844217f}   // Intercept eq
    };
    const SoftmaxEquationParams e16{
            {0.001258255f, 0.010924353f},  // Slope eq
            {0.046222578f, 12.42546444f}   // Intercept eq
    };
    const SoftmaxEquationParams e32{
            {0.001256785f, 0.011003726f},  // Slope eq
            {0.000409401f, 8.728951463f}   // Intercept eq
    };

    const FirstDegreeEquation baseEq{0.000783649f, 7.660420549f};

    const Softmax_gateway model{
            VPUNN::DataType::FLOAT16, baseEq.slope_, baseEq.intercept_, e1, e2, e4, e8, e16, e32, 1700, 971};
};
TEST_F(ShaveSoftmaxTest, SoftmaxNormalizeValueTest) {
    // Check type 1
    EXPECT_EQ(model.normalizeUnselectedValue(81, model.getEquationType(81)), 65);

    // Check type 2
    EXPECT_EQ(model.normalizeUnselectedValue(82, model.getEquationType(82)), 66);

    // Check type 4
    EXPECT_EQ(model.normalizeUnselectedValue(84, model.getEquationType(84)), 68);

    // Check type 8
    EXPECT_EQ(model.normalizeUnselectedValue(88, model.getEquationType(88)), 72);

    // Check type 16
    EXPECT_EQ(model.normalizeUnselectedValue(80, model.getEquationType(80)), 80);

    // Check type 32
    EXPECT_EQ(model.normalizeUnselectedValue(64, model.getEquationType(64)), 64);
}

TEST_F(ShaveSoftmaxTest, SoftmaxEqTypeGetterTest) {
    // Test Type 32
    EXPECT_EQ(model.getEquationType(32), VPUNN::SoftmaxEquationType::Type32);
    EXPECT_EQ(model.getEquationType(64), VPUNN::SoftmaxEquationType::Type32);

    // Test Type 1
    EXPECT_EQ(model.getEquationType(65), VPUNN::SoftmaxEquationType::Type1);

    // Test Type 2
    EXPECT_EQ(model.getEquationType(66), VPUNN::SoftmaxEquationType::Type2);

    // Test Type 4
    EXPECT_EQ(model.getEquationType(68), VPUNN::SoftmaxEquationType::Type4);

    // Test Type 8
    EXPECT_EQ(model.getEquationType(72), VPUNN::SoftmaxEquationType::Type8);

    // Test Type 16
    EXPECT_EQ(model.getEquationType(80), VPUNN::SoftmaxEquationType::Type16);
}

TEST_F(ShaveSoftmaxTest, GetMicroSecondsTest) {
    // Unselected tpye 32
    EXPECT_NEAR(model.getMicroSeconds(100, 512), 78.75120055f, 0.001);

    // Unselected tpye 16
    EXPECT_NEAR(model.getMicroSeconds(190, 432), 129.2046723f, 0.001);

    // Unselected tpye 8
    EXPECT_NEAR(model.getMicroSeconds(250, 344), 195.9558724f, 0.001);

    // Unselected tpye 4
    EXPECT_NEAR(model.getMicroSeconds(25, 140), 23.65107421f, 0.001);

    // Unselected tpye 2
    EXPECT_NEAR(model.getMicroSeconds(10, 38), 14.71566294f, 0.001);

    // Unselected tpye 1
    EXPECT_NEAR(model.getMicroSeconds(250, 131), 123.7233617f, 0.001);

    // Unselected Space == 1
    EXPECT_NEAR(model.getMicroSeconds(37250, 1), 36.85133622f, 0.001);
}

class ShaveNormalizeL2Test : public ::testing::Test {
protected:
    void SetUp() override {
    }

    class NormalizeL2OnlyC_gateway : public NormalizeL2OnlyC {
    public:
        using NormalizeL2OnlyC::baseTimeCalculator;
        using NormalizeL2OnlyC::NormalizeL2OnlyC;
        using NormalizeL2OnlyC::wTimeIncrease;
    };

    const FirstDegreeEquation baseEq{0.0010080518479f, 3.94000845803864f};
    const FirstDegreeEquation baseEqW{0.010424209f, 0.311680777926308f};

    const float baseVecOff = 0.039f;
    const float wVecOff = 0.004891f;
    const float slopeMod1 = 0.006735642f;
    const float slopeMod8 = 0.00323271989278241f;
    const float slopeMod9 = 0.013133901f;

    const NormalizeL2OnlyC_gateway model{VPUNN::DataType::FLOAT16,
                                         baseEq.slope_,
                                         baseEq.intercept_,
                                         baseVecOff,
                                         baseEqW.slope_,
                                         baseEqW.intercept_,
                                         slopeMod1,
                                         slopeMod8,
                                         slopeMod9,
                                         wVecOff,
                                         1700,
                                         971};
};
TEST_F(ShaveNormalizeL2Test, NormalizeBaseTimeCalcTest) {
    // No Vec time
    EXPECT_NEAR(model.baseTimeCalculator(32), 3.972266117f, 0.05f);
    EXPECT_NEAR(model.baseTimeCalculator(2560), 6.520621189f, 0.05f);

    // VecTime
    EXPECT_NEAR(model.baseTimeCalculator(14), 3.954121184f + 0.234f, 0.05f);
    EXPECT_NEAR(model.baseTimeCalculator(1140), 5.089187565f + 0.156f, 0.05f);
}

TEST_F(ShaveNormalizeL2Test, NormalizeWidthTimeIncreaseTest) {
    // Slope 16
    EXPECT_NEAR(model.wTimeIncrease(20, 112, 1), 3.641154732f, 0.05f);

    // Slope 8
    EXPECT_NEAR(model.wTimeIncrease(48, 40, 1), 2.418033435f, 0.05f);

    // Slope 1
    EXPECT_NEAR(model.wTimeIncrease(11, 2449, 1), 76.59846615f, 0.05f);

    // Slope 1 + vector
    EXPECT_NEAR(model.wTimeIncrease(160, 2, 128), 491.4965735f, 0.05f);

    // Slope 9
    EXPECT_NEAR(model.wTimeIncrease(20, 169, 10), 82.68779008f, 0.05f);

    // Slope 9 + vector
    EXPECT_NEAR(model.wTimeIncrease(20, 333, 10), 166.8419958f, 0.05f);
}

TEST_F(ShaveNormalizeL2Test, NormalizeTotalTimeCalcTest) {
    // Slope 16
    EXPECT_NEAR(model.getMicroSeconds(20, 112, 1), 7.757324227f, 0.05f);

    // Slope 8
    EXPECT_NEAR(model.getMicroSeconds(48, 40, 1), 6.406428381f, 0.05f);

    // Slope 1
    EXPECT_NEAR(model.getMicroSeconds(11, 2449, 1), 80.66656318f, 0.05f);

    // Slope 1 + vector
    EXPECT_NEAR(model.getMicroSeconds(160, 2, 128), 495.5978702f, 0.05f);

    // Slope 9
    EXPECT_NEAR(model.getMicroSeconds(20, 169, 10), 86.80395957f, 0.05f);

    // Slope 9 + vector
    EXPECT_NEAR(model.getMicroSeconds(20, 333, 10), 170.9581653f, 0.05f);
}

class ShaveMVN6Test : public ::testing::Test {
protected:
    void SetUp() override {
    }
    const VariableSlopeFirstDegreeEquation model_eq{
            0.199457115f,  ///< base slope
            11.22497677f,  ///< intercept
            0.068f,        ///< alpha
            0.369384878f,  /// <slope difference C=1 - C=S
    };
    const MVN6OneAxisModel model{VPUNN::DataType::FLOAT16,
                                 model_eq.slope_,              ///< base slope
                                 model_eq.intercept_,          ///< intercept
                                 model_eq.alpha_,              ///< alpha
                                 model_eq.maximum_diff_slope,  /// <slope difference C=1 - C=S
                                                               /* 0.0,
                                                                0.0,
                                                                1,
                                                                0,*/
                                 1300, 975};
};

TEST_F(ShaveMVN6Test, ValuesTest) {
    EXPECT_NEAR(model.getMicroSeconds(40960, 1), 23311.848, 1000);
    EXPECT_NEAR(model.getMicroSeconds(40960, 40960), 8180.546, 100);
    EXPECT_NEAR(model.getMicroSeconds(32768, 8192), 6547.473, 100);
}

TEST_F(ShaveMVN6Test, CoeffTest) {
    EXPECT_NEAR(model_eq.getCoeff(40960, 1), 1, 0.05);
}

TEST_F(ShaveMVN6Test, SlopeTest) {
    EXPECT_NEAR(model_eq.getSlope(model_eq.getCoeff(40960, 1)), 0.56, 0.05);
}

/// test that the compilation and execution is on separate paths for getDPUCycles
/// this is rather generic for the ShaveCyclesProvider base class mechanism
TEST_F(ShaveMVN6Test, Template_getDPUCycles_specialisation) {
    {  // same types
        const /*unsigned*/ int output_samples{32768};
        const /*unsigned*/ int innermost_dimension_size{8192};

        const auto cycles = model.getDPUCycles(output_samples, innermost_dimension_size);

        const int pdpu{1500};
        // const auto cycles2 = model.getDPUCyclesAnotherFreqDPU(pdpu, output_samples, innermost_dimension_size);

        const int pvpu{700};
        const auto cycles3 = model.getDPUCyclesAnotherFreqDPU_SHV(pdpu, pvpu, output_samples, innermost_dimension_size);

        // EXPECT_NE(cycles, cycles2);
        EXPECT_NE(cycles, cycles3);
        // EXPECT_NE(cycles2, cycles3);
    }

    {  // mixed
        const unsigned int output_samples{32768};
        const unsigned int innermost_dimension_size{8192};

        const auto cycles = model.getDPUCycles(output_samples, innermost_dimension_size);

        const int pdpu{1500};
        // const auto cycles2 = model.getDPUCyclesAnotherFreqDPU(pdpu, output_samples, innermost_dimension_size);

        const int pvpu{700};
        const auto cycles3 = model.getDPUCyclesAnotherFreqDPU_SHV(pdpu, pvpu, output_samples, innermost_dimension_size);

        // EXPECT_NE(cycles, cycles2);
        EXPECT_NE(cycles, cycles3);
        // EXPECT_NE(cycles2, cycles3);
    }
}
}  // namespace VPUNN_unit_tests