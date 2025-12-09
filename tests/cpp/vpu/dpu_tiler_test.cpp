// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.
#include "vpu/optimization/dimension_tiler.h"
#include "vpu/optimization/tiler.h"

#include <gtest/gtest.h>
#include <sstream>  // for error formating
#include "common/common_helpers.h"
#include "core/logger.h"

namespace VPUNN_unit_tests {
using namespace VPUNN;
// using namespace std::placeholders;

class DPUTilerTest : public ::testing::Test {
public:
protected:
    void SetUp() override {
        Logger::clear2ndlog();
        Logger::activate2ndlog();
    }
    void TearDown() override {
        Logger::clear2ndlog();
        Logger::deactivate2ndlog();
    }

public:
    struct TestInput {
        int dimension{0};  ///< Dimension to split
        int nTiles{0};     ///< Number of tiles
    };

    struct TestExpectations {
        std::vector<int> expected{};
    };

protected:
    struct TestCase {
        TestInput t_in;
        TestExpectations t_exp;
        const std::string test_case = "";
    };
    using TestsVector = std::vector<TestCase>;
    using container = SplitDimension::SplitContainer;

    auto execute(const TestCase& t, const SplitDimension& dut) const {
        container result{};
        if (t.t_exp.expected.size() > 0) {
            EXPECT_TRUE(dut.divideBalanced(t.t_in.dimension, t.t_in.nTiles, result))
                    << t.test_case << " Dimension: " << t.t_in.dimension << " , split in: " << t.t_in.nTiles << ".";
            EXPECT_EQ(result, t.t_exp.expected)
                    << t.test_case << " Dimension: " << t.t_in.dimension << " , split in: " << t.t_in.nTiles << ".\n"
                    << " Size result: " << result.size() << "  Expected Size: " << t.t_exp.expected.size();
        } else {
            EXPECT_FALSE(dut.divideBalanced(t.t_in.dimension, t.t_in.nTiles, result))
                    << t.test_case << " Dimension: " << t.t_in.dimension << " , split in: " << t.t_in.nTiles << ".";
        }
    };

public:
protected:
    // void executeT(std::function<void(const TestInput&, const TestExpectations&, const std::string&)>& f,
    //               const TestsVector& tests, std::string h = "") {
    //     int test_index = 0;
    //     for (const auto& t : tests) {
    //         std::stringstream buffer;
    //         buffer << test_index << " : " << t.test_case;
    //         const std::string test_case_info = buffer.str();

    //        f(t.t_in, t.t_exp, h + test_case_info);

    //        ++test_index;
    //    }
    //}
};

TEST_F(DPUTilerTest, SmokeTest) {
    SmartRanges rangeN{1, 8192};
    SplitDimension dut{rangeN};

    {
        container result{};
        EXPECT_FALSE(dut.divideBalanced(10, 0, result));  // no splits is  invalid for naturals
        EXPECT_EQ(result.size(), 0);
    }
    {
        container result{};
        EXPECT_TRUE(dut.divideBalanced(0, 0, result));  // 0/0 OK  by definition
        EXPECT_EQ(result.size(), 0);
    }

    {
        container result{};
        EXPECT_FALSE(dut.divideBalanced(1, 0, result));  // /0
        EXPECT_EQ(result.size(), 0);
    }
    {
        container result{};
        EXPECT_FALSE(dut.divideBalanced(1, 2, result));  // no solution
        EXPECT_LE(result.size(), 1);
    }

    {
        container result{};
        EXPECT_TRUE(dut.divideBalanced(1, 1, result));
        EXPECT_EQ(result.size(), 1);
        container result_e{1};
        EXPECT_EQ(result, result_e);
    }

    {
        container result{};
        EXPECT_TRUE(dut.divideBalanced(16, 1, result));
        EXPECT_EQ(result.size(), 1);
        container result_e{16};
        EXPECT_EQ(result, result_e);
    }

    {
        container result{};
        EXPECT_TRUE(dut.divideBalanced(10, 10, result));
        EXPECT_EQ(result.size(), 10);
        container result_e{1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
        EXPECT_EQ(result, result_e);
    }
    {
        container result{};
        EXPECT_TRUE(dut.divideBalanced(10, 1, result));
        EXPECT_EQ(result.size(), 1);
        container result_e{10};
        EXPECT_EQ(result, result_e);
    }

    {
        container result{};
        EXPECT_TRUE(dut.divideBalanced(10, 2, result));
        EXPECT_EQ(result.size(), 2);
        container result_e{5, 5};
        EXPECT_EQ(result, result_e);
    }
    {
        container result{};
        EXPECT_TRUE(dut.divideBalanced(10, 3, result));
        EXPECT_EQ(result.size(), 3);
        container result_e{4, 3, 3};
        EXPECT_EQ(result, result_e);
    }
    {
        container result{};
        EXPECT_TRUE(dut.divideBalanced(10, 1, result));
        EXPECT_EQ(result.size(), 1);
        container result_e{10};
        EXPECT_EQ(result, result_e);
    }

    {
        container result{};
        EXPECT_TRUE(dut.divideBalanced(3, 2, result));
        EXPECT_EQ(result.size(), 2);
        container result_e{2, 1};
        EXPECT_EQ(result, result_e);
    }
    {
        container result{};
        EXPECT_TRUE(dut.divideBalanced(7, 3, result));
        EXPECT_EQ(result.size(), 3);
        container result_e{3, 2, 2};
        EXPECT_EQ(result, result_e);
    }
}

TEST_F(DPUTilerTest, Div16Range_Neg_Test) {
    SmartRanges rangeN{1, 8192, 16};
    SplitDimension dut{rangeN};

    {
        container result{};
        EXPECT_FALSE(dut.divideBalanced(10, 0, result));  // no splits is  invalid for naturals
        EXPECT_EQ(result.size(), 0);
    }
    {
        container result{};
        EXPECT_TRUE(dut.divideBalanced(0, 0, result));  // 0/0 OK  by definition
        EXPECT_EQ(result.size(), 0);
    }

    {
        container result{};
        EXPECT_FALSE(dut.divideBalanced(1, 0, result));  // /0
        EXPECT_EQ(result.size(), 0);
    }
    {
        container result{};
        EXPECT_FALSE(dut.divideBalanced(1, 2, result));  // no solution
        EXPECT_LE(result.size(), 1);
    }
    {
        container result{};
        EXPECT_TRUE(dut.divideBalanced(16, 1, result));
        EXPECT_EQ(result.size(), 1);
        container result_e{16};
        EXPECT_EQ(result, result_e);
    }

    {
        container result{};
        EXPECT_FALSE(dut.divideBalanced(15, 1, result));  // no solution
        EXPECT_LE(result.size(), 1);
    }
    {
        container result{};
        EXPECT_FALSE(dut.divideBalanced(17, 1, result));  // no solution
        EXPECT_LE(result.size(), 1);
    }
    {
        container result{};
        EXPECT_FALSE(dut.divideBalanced(49, 1, result));  // no solution
        EXPECT_LE(result.size(), 1);
    }

    {
        container result{};
        EXPECT_FALSE(dut.divideBalanced(1, 1, result));
        // EXPECT_EQ(result, result_e);
    }

    {
        container result{};
        EXPECT_FALSE(dut.divideBalanced(10, 10, result));
    }
    {
        container result{};
        EXPECT_FALSE(dut.divideBalanced(10, 1, result));
    }

    {
        container result{};
        EXPECT_FALSE(dut.divideBalanced(10, 2, result));
    }
    {
        container result{};
        EXPECT_FALSE(dut.divideBalanced(10, 3, result));
    }
    {
        container result{};
        EXPECT_FALSE(dut.divideBalanced(10, 1, result));
    }

    {
        container result{};
        EXPECT_FALSE(dut.divideBalanced(3, 2, result));
    }
    {
        container result{};
        EXPECT_FALSE(dut.divideBalanced(7, 3, result));
    }
}

TEST_F(DPUTilerTest, Div16Range_Norm_Test) {
    SmartRanges rangeN{1, 8192, 16};
    SplitDimension dut_local{rangeN};

    std::vector<TestCase> tests{
            {{10, 2}, {{}}, "10/2"},             //
            {{16, 1}, {{16}}, "16/1"},           //
            {{48, 1}, {{48}}, "Div16"},          //
            {{48, 2}, {{32, 16}}, "Div16"},      //
            {{48, 3}, {{16, 16, 16}}, "Div16"},  //
            {{48, 4}, {{}}, "Div16"},            //

            {{80, 2}, {{48, 32}}, "Div16"},                                   //
            {{80, 3}, {{32, 32, 16}}, "Div16"},                               //
            {{8192, 2}, {{8192 / 2, 8192 / 2}}, "Div16 8192."},               //
            {{8192 + 16, 2}, {{8192 / 2 + 16, 8192 / 2}}, "Div16 8192+16."},  //
            {{8192 * 2, 2}, {{8192, 8192}}, "Div16 8192*2."},                 //
            {{8192 * 2 + 16, 2}, {{}}, "Div16 8192*2+16."},                   //
    };

    {
        container result{};
        EXPECT_FALSE(dut_local.divideBalanced(1, 2, result));  // no solution
        EXPECT_LE(result.size(), 1);
    }
    {
        container result{};
        EXPECT_TRUE(dut_local.divideBalanced(16, 1, result));
        EXPECT_EQ(result.size(), 1);
        container result_e{16};
        EXPECT_EQ(result, result_e);
    }

    for (const auto& t : tests) {
        execute(t, dut_local);
    }
}

TEST_F(DPUTilerTest, Range16_32_64_BasicTest) {
    SmartRanges rangeN{16, 64, 16, 32};
    SplitDimension dut_local{rangeN};

    std::vector<TestCase> tests{
            {{10, 2}, {{}}, "10/2"},  //

            {{16, 1}, {{16}}, "16/1"},  //
            {{16, 2}, {{}}, "16/2"},    //

            {{32, 1}, {{32}}, ""},      //
            {{32, 2}, {{16, 16}}, ""},  //
            {{32, 3}, {{}}, ""},        //

            {{48, 1}, {{}}, "D32"},           //
            {{48, 2}, {{32, 16}}, "D32/16"},  //
            {{48, 3}, {{16, 16, 16}}, ""},    //
            {{48, 4}, {{}}, ""},              //

            {{64, 1}, {{64}}, "64"},            //
            {{64, 2}, {{32, 32}}, ""},          //
            {{64, 3}, {{32, 16, 16}}, ""},      //
            {{64, 4}, {{16, 16, 16, 16}}, ""},  //

            {{80, 1}, {{}}, ""},        //
            {{80, 2}, {{64, 16}}, ""},  //

            {{96, 2}, {{64, 32}}, ""},                  //
            {{96, 3}, {{32, 32, 32}}, ""},              //
            {{96, 4}, {{32, 32, 16, 16}}, ""},          //
            {{96, 5}, {{32, 16, 16, 16, 16}}, ""},      //
            {{96, 6}, {{16, 16, 16, 16, 16, 16}}, ""},  //

            {{112, 3}, {{64, 32, 16}}, ""},      //
            {{176, 4}, {{64, 64, 32, 16}}, ""},  //
            {{144, 4}, {{64, 32, 32, 16}}, ""},  //
            {{128, 4}, {{32, 32, 32, 32}}, ""},  //

            {{176, 6}, {{32, 32, 32, 32, 32, 16}}, ""},  //
            {{160, 6}, {{32, 32, 32, 32, 16, 16}}, ""},  //
            {{144, 6}, {{32, 32, 32, 16, 16, 16}}, ""},  //

            {{208, 6}, {{64, 32, 32, 32, 32, 16}}, ""},  //
            {{224, 6}, {{64, 32, 32, 32, 32, 32}}, ""},  //

            {{1200, 20},
             {{64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 32, 16}},
             "1200/20TEST"},  //

    };

    {
        container result{};
        EXPECT_FALSE(dut_local.divideBalanced(1, 2, result));  // no solution
        EXPECT_LE(result.size(), 1);
    }
    {
        container result{};
        EXPECT_TRUE(dut_local.divideBalanced(16, 1, result));
        EXPECT_EQ(result.size(), 1);
        container result_e{16};
        EXPECT_EQ(result, result_e);
    }

    for (const auto& t : tests) {
        execute(t, dut_local);
    }
}

TEST_F(DPUTilerTest, Range16_32_64_BIGTest) {
    // std::vector<TestCase> tests{
    //         {{1200, 20},
    //          {{64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 32, 16}},
    //          "1200/20TEST"},  //

    //};

    for (int i = 1; i <= 255; ++i) {
        container result{};
        const SmartRanges rangeN{16, 64, 16, 32};
        const SplitDimension dut_local{rangeN};
        EXPECT_NO_THROW(dut_local.divideBalanced(1200, i, result));
    }
}

class ZTilingTest : public ::testing::Test {
public:

protected:
    static constexpr bool legacy_ztiling_flag{
        #ifdef VPUNN_OPT_LEGACY_ZTILING
                true
        #else
                false
        #endif
    };
    
    // values for channels that should be splitted, correlated the names mentioned above
    const std::vector<unsigned int> k_values {
        128U,
        512U,
        333U
    };

    // Reusing same workload, and changing only the output channels on which the algorithm is tested
    const std::vector < VPUNN::DPUWorkload> test_wl = [this]() {
        std::vector<VPUNN::DPUWorkload> tmp;
        for (size_t i = 0; i < k_values.size(); i++) {
            tmp.push_back(VPUNN::DPUWorkload{VPUNN::VPUDevice::NPU_5_0,
                                        VPUNN::Operation::CONVOLUTION,
                                        {VPUNN::VPUTensor(64, 108, 256, 1, VPUNN::DataType::UINT8,
                                                            Layout::ZXY)},   // input dimensions
                                        {VPUNN::VPUTensor(21, 36, k_values[i], 1, VPUNN::DataType::UINT8,
                                                            Layout::ZXY)},   // output dimensions
                                        {3, 3},                              // kernels
                                        {3, 3},                              // strides
                                        {0, 0, 0, 0},                        // padding
                                        VPUNN::ExecutionMode::CUBOID_16x16,  // execution mode
                                        VPUNN::ActivationFunction::NONE,     // activation
                                        0.0F,                                // act_sparsity
                                        0.0F,                                // weight_sparsity
                                        {swz_def, swz_def},                  // input_swizzling
                                        {swz_def},                           // output_swizzling
                                        1,                                   // output_write_tiles
                                        {0, 0, 0, 0},                        // offsets
                                        VPUNN::ISIStrategy::CLUSTERING,      // isi_strategy
                                        false});
        }
        return tmp;
    }(); // declaring lambda with immediate invokation for const array expressions
};

TEST_F(ZTilingTest, splitInNOverZOffset_Test) {

    struct TestInput {
        DPUWorkload wl;
        unsigned int nWorkloads;
    };

    struct TestExpectations {
        std::vector<unsigned int> channels_offset_gt;
    };

    struct TestCase {
    public:
        TestInput t_in;
        TestExpectations t_exp;

        std::string toString() const {
            std::stringstream ss;
            ss << "\nTest case for: \n"
               << "\tInput:\n"
               << "\t\tChannels: " << t_in.wl.outputs[0].z() << "\n"
               << "\t\tWorkloads number: " << t_in.nWorkloads << "\n"
               << "\tExpectations: \n"
               << "\t\tOffset ground truth of length " << t_exp.channels_offset_gt.size() << ": {";
            for (size_t i = 0; i < t_exp.channels_offset_gt.size(); i++) {
                ss << t_exp.channels_offset_gt[i];
                if (i != t_exp.channels_offset_gt.size() - 1)
                    ss << ", ";
            }
            ss << "}\nLegacy_ZTiling_Flag is " << (legacy_ztiling_flag ? "True" : "False") << "\n";
            return ss.str();
        };
    };

    using TestVector = std::vector<TestCase>;

    const ExecutionMode mode{ExecutionMode::CUBOID_8x16};
    const SplitOptions options{128U,  ///< Maximum number of workloads available. Default is 128 because of FIFO size
                               0,     ///< Number of DPU to optimize for. maxLatencyMs = 0 means full search
                               0,  ///< Number of DPU to optimize for. Setting nDPU = 0 VPUNN auto-detects the number of
                                   ///< DPUs based on the device
                               0,  ///< Per workload runtime overhead in cycles
                               VPUOptimizationTarget::LATENCY,
                               {VPUSplitStrategy::Z_TILING}};

    auto exec_tests = [&options, modeInternal = mode](TestVector& tests) {

        for (auto& t : tests) {

            DPULayer layer{t.t_in.wl};
            TilingAlgorithmsContainer tiler_z = getTilingAlgorithms(layer, options);
            ASSERT_TRUE(tiler_z.front() != nullptr) << "Could not retrieve the ZTiling algorithm from ITilerAlgorithm interface"
                                            << t.toString();

            // Use the method for splitting and extracting the splitPool with it's offset setted
            std::list<DPUWorkloadsWithCyclesSplit> splitPool =
                    tiler_z.front()->split_tile_in_workloads(modeInternal, t.t_in.nWorkloads);

            // If it is a valid split case it means that the ground truth offset vector shouldn't be emtpy
            if (splitPool.empty()) {
                ASSERT_EQ(t.t_exp.channels_offset_gt.size(), 0) << "For a single ZTiling algorithm test case, there "
                                                                   "should be only 1 split pool if success, and 0 "
                                                                   "if the split is impossible"
                                                                << t.toString();
            } else {
                const auto& workloads = splitPool.front().workloads;
                std::stringstream ss;
                ss << "{ ";
                for (const auto& w : workloads) {
                    ss << w.offsets[Dim::Act::Z] << " ";
                }
                ss << "}";

                ASSERT_EQ(workloads.size(), t.t_exp.channels_offset_gt.size())
                        << t.toString() << "\n\tWorkloads " << "of length " << workloads.size()
                        << " values : " << ss.str();
            }

            if (t.t_exp.channels_offset_gt.size() != 0) {
                // Extract the only available split
                DPUWorkloadsWithCyclesSplit split = splitPool.front();

                EXPECT_EQ(split.workloads.size(), t.t_in.nWorkloads);
                for (unsigned int i = 0; i < t.t_in.nWorkloads; i++) {
                    EXPECT_EQ(split.cycles[i], Cycles::NO_ERROR) << Cycles::toErrorText(split.cycles[i]);
                    EXPECT_EQ(split.workloads[i].offsets[Dim::Act::Z], t.t_exp.channels_offset_gt[i])
                            << t.toString()
                            << "Failed for offset number: " << i << "\n";
                }
            }
        }
    };

    // This test-set is used only when force_LegacyZTiling flag from tiler.cpp is ON
    // For force_LegacyZTiling flag ON, for test to pass it should:
    //  have Z dimension of input Tensor bigger than 16, and divisible by 16
    //  then it will find nearest number divisible by 16 of channels/nWorkload expression
    //  but if the result of division is < 16, then it will round up to 16
    TestVector tests = {
        { {test_wl[0], 4U}, {{0U, 32U, 64U, 96U}} }, // same expectation
        { {test_wl[1], 3U}, {{0U, 176U, 352U}} },    // same expectation
        {
            {test_wl[1], 64U},
            legacy_ztiling_flag
                    ? TestExpectations{{0U,   16U,  32U,  48U,  64U,  80U,  96U,  112U, 128U, 144U, 160U,
                                        176U, 192U, 208U, 224U, 240U, 256U, 272U, 288U, 304U, 320U, 336U,
                                        352U, 368U, 384U, 400U, 416U, 432U, 448U, 464U, 480U, 496U, 512U,
                                        528U, 544U, 560U, 576U, 592U, 608U, 624U, 640U, 656U, 672U, 688U,
                                        704U, 720U, 736U, 752U, 768U, 784U, 800U, 816U, 832U, 848U, 864U,
                                        880U, 896U, 912U, 928U, 944U, 960U, 976U, 992U, 1008U}}
                    : TestExpectations{{}}
        },
        {
            {test_wl[0], 64U},
            legacy_ztiling_flag
                     ? TestExpectations{{0U,   16U,  32U,  48U,  64U,  80U,  96U,  112U, 128U, 144U, 160U, 176U, 192U,
                                         208U, 224U, 240U, 256U, 272U, 288U, 304U, 320U, 336U, 352U, 368U, 384U, 400U,
                                         416U, 432U, 448U, 464U, 480U, 496U, 512U, 528U, 544U, 560U, 576U, 592U, 608U,
                                         624U, 640U, 656U, 672U, 688U, 704U, 720U, 736U, 752U, 768U, 784U, 800U, 816U,
                                         832U, 848U, 864U, 880U, 896U, 912U, 928U, 944U, 960U, 976U, 992U, 1008U}}
                     : TestExpectations{{}}
        },
        {
            {test_wl[0], 13U},
            legacy_ztiling_flag
                     ? TestExpectations{{0U,   16U,  32U,  48U,  64U,  80U,  96U,  112U, 128U, 144U, 160U, 176U, 192U}}
                     : TestExpectations{{}}
        },
        {
            {test_wl[1], 5U},       
            legacy_ztiling_flag 
                     // divides it in equal slices with 112 offset each, even though it takes more space, because the division is not even
                     ? TestExpectations{{0U, 112U, 224U, 336U, 448U}} 
                     // divides in slices with 112 and 96 offset, without additional offset space, and sums up to 512 channels
                     : TestExpectations{{0U, 112U, 224U, 320U, 416U}}
        },
        { {test_wl[2], 4U}, {} },   // same expectations
        { {test_wl[2], 55U}, {} }   // same expectations 
    };

    exec_tests(tests);
    }

}  // namespace VPUNN_unit_tests
