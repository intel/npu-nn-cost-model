// Copyright © 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#include <gtest/gtest.h>
#include <sstream>  // for error formating
#include "common_helpers.h"
#include "vpu_layer_cost_model.h"

namespace VPUNN_unit_tests {
using namespace VPUNN;
using namespace std::placeholders;

class DPULayerTest : public ::testing::Test {
public:
protected:
    void SetUp() override {
        VPUNN::Logger::clear2ndlog();
        VPUNN::Logger::activate2ndlog();
    }
    void TearDown() override {
        VPUNN::Logger::clear2ndlog();
        VPUNN::Logger::deactivate2ndlog();
    }

    VPUNN::DPULayer generate_helper_layer(const unsigned int dim, const unsigned int channels) {
        return VPUNN::DPULayer(
                VPUNN::VPUDevice::VPU_2_0,                                            // VPU device
                VPUNN::Operation::CONVOLUTION,                                        // Operation
                {VPUNN::VPUTensor(dim, dim, channels, 1, VPUNN::DataType::FLOAT16)},  // input dimensions
                {VPUNN::VPUTensor(dim, dim, channels, 1, VPUNN::DataType::FLOAT16)},  // output dimensions
                {3, 3},                                                               // kernels
                {1, 1},                                                               // strides
                {1, 1, 1, 1}                                                          // padding
        );
    }
    DPULayer layer_H(unsigned int h_input, unsigned int h_output, unsigned int kernel, unsigned int padtop,
                     unsigned int padbot, unsigned int stride) {
        return DPULayer(VPUDevice::VPU_2_7, Operation::CONVOLUTION,
                        {VPUTensor(1, h_input, 128, 1, DataType::FLOAT16)},   // input dimensions WH
                        {VPUTensor(1, h_output, 128, 1, DataType::FLOAT16)},  // output dimensions
                        {1, kernel},                                          // kernels WH
                        {1, stride},                                          // stridesWH
                        {padtop, padbot, 0, 0}                                // padding TBLR
        );
    }

public:
    struct TestInput {
        DPULayer l1;

        unsigned int nTiles{1};                                      ///< Number of tiles
        VPUTilingStrategy tiling_strategy{VPUTilingStrategy::NONE};  ///< tiling strategy
    };

    struct TestExpectations {
        std::vector<int> in_size;
        std::vector<int> out_size;
        std::vector<int> k;
        std::vector<int> pad1;
        std::vector<int> pad2;
        std::vector<ISIStrategy> isi{};

        bool has_consistent_size() const {
            const auto s{in_size.size()};
            if ((s == out_size.size() && (s == k.size()) && (s == pad1.size()) && (s == pad2.size()))) {
                return true;
            }
            return false;
        }
    };

    struct TestCase {
        TestInput t_in;
        TestExpectations t_exp;
        const std::string test_case = "";
    };
    using TestsVector = std::vector<TestCase>;

    std::function<std::vector<DPULayer>(const DPULayer*, unsigned int)> split_SOHO_heuristic =
            std::bind(&DPULayer::SOH_overlapped, _1, _2);

    std::function<void(const TestInput&, const TestExpectations&, const std::string&)> SOHO_test =
            std::bind(&DPULayerTest::TestSoh, this, split_SOHO_heuristic, _1, _2, _3);

    void executeT_SOHOVR(const TestsVector& tests) {
        executeT(SOHO_test, tests, "SOHO:: ");
    }

    void TestSoh(std::function<std::vector<DPULayer>(const DPULayer*, unsigned int)>& hSplit, const TestInput& t_in,
                 const TestExpectations& t_exp, const std::string& test_case = "") const {
        const DPULayer& l1{t_in.l1};
        const auto tiles{t_in.nTiles};

        std::string details = test_case + ":";
        {  // "5-5 k3 p11"},
            std::stringstream buffer;
            buffer << " [" << l1.inputs[0].height() << "-" << l1.outputs[0].height() << " k" << l1.kernels[Dim::H]
                   << " p" << l1.padding[Dim::TOP] << "-" << l1.padding[Dim::BOTTOM] << " s" << l1.strides[Dim::H]
                   << "]";

            details += buffer.str();
        }

        std::string t_header{"** Test Case: " + details + "\n"};
        std::cout << ">> " << t_header << std::endl;

        EXPECT_TRUE(t_exp.has_consistent_size()) << "problem with expected values consistency";

        auto split = hSplit(&l1, tiles);

        // Basic expectations
        ASSERT_EQ(split.size(), t_exp.in_size.size()) << "Expected size of split is different than real split";

        for (size_t i = 0; i < split.size(); ++i) {
            const auto& s = split[i];
            ASSERT_EQ(s.outputs[0].height(), t_exp.out_size[i]) << " i= " << i << s << t_exp;

            ASSERT_EQ(s.inputs[0].height(), t_exp.in_size[i]) << " i= " << i << s << t_exp;

            ASSERT_EQ(s.kernels[Dim::H], t_exp.k[i]) << " i= " << i << s << t_exp;
            ASSERT_EQ(s.padding[Dim::TOP], t_exp.pad1[i]) << " i= " << i << s << t_exp;
            ASSERT_EQ(s.padding[Dim::BOTTOM], t_exp.pad2[i]) << " i= " << i << s << t_exp;

            ISIStrategy expected_isi{(t_exp.isi.size() > i) ? t_exp.isi[i]
                                                            : ISIStrategy::CLUSTERING};  // clustering by default

            ASSERT_EQ(s.isi_strategy, expected_isi) << "ISI is marked BAD"
                                                    << " i= " << i << s << t_exp;
            ASSERT_EQ(s.output_write_tiles, l1.output_write_tiles) << "output_write_tiles not kept"
                                                                   << " i= " << i << s << t_exp << l1;

            // check also stride??

            // can do checks that other fields are not changed!
        }

        std::cout << t_header << "------------------------------------------------------------------------"
                  << std::endl;
    }

    void executeT(std::function<void(const TestInput&, const TestExpectations&, const std::string&)>& f,
                  const TestsVector& tests, std::string h = "") {
        int test_index = 0;
        for (const auto& t : tests) {
            std::stringstream buffer;
            buffer << test_index << " : " << t.test_case;
            const std::string test_case_info = buffer.str();

            f(t.t_in, t.t_exp, h + test_case_info);

            ++test_index;
        }
    }
};

std::ostream& operator<<(std::ostream& stream, const DPULayerTest::TestExpectations& t) {
    stream << "TestExpectations:";  //
    {
        stream << "\nin size: ";
        std::for_each(t.in_size.begin(), t.in_size.end(), [&stream](auto& t) {
            stream << " " << t << " ,";
        });
    }
    {
        stream << "\nout_size: ";
        std::for_each(t.out_size.begin(), t.out_size.end(), [&stream](auto& t) {
            stream << " " << t << " ,";
        });
    }
    {
        stream << "\nk: ";
        std::for_each(t.k.begin(), t.k.end(), [&stream](auto& t) {
            stream << " " << t << " ,";
        });
    }
    {
        stream << "\npad1: ";
        std::for_each(t.pad1.begin(), t.pad1.end(), [&stream](auto& t) {
            stream << " " << t << " ,";
        });
    }
    {
        stream << "\npad2: ";
        std::for_each(t.pad2.begin(), t.pad2.end(), [&stream](auto& t) {
            stream << " " << t << " ,";
        });
    }
    stream << "\n";
    return stream;
}

TEST_F(DPULayerTest, SplitAcrossTileSOH) {
    auto wl = generate_helper_layer(16, 64);

    auto SOH_single_tile = wl.splitAcrossTiles(VPUNN::VPUTilingStrategy::SOH, 1);
    auto SOH_two_tile = wl.splitAcrossTiles(VPUNN::VPUTilingStrategy::SOH, 2);
    auto SOH_four_tile = wl.splitAcrossTiles(VPUNN::VPUTilingStrategy::SOH, 4);

    // Basic expectations
    EXPECT_EQ(SOH_single_tile.size(), 1);
    EXPECT_EQ(SOH_two_tile.size(), 2);
    EXPECT_EQ(SOH_four_tile.size(), 4);

    // The shape of the single split must be equal to the origial layer
    EXPECT_EQ(SOH_single_tile[0].outputs[0].get_shape(), wl.outputs[0].get_shape());

    EXPECT_EQ(SOH_two_tile[0].outputs[0].size(), wl.outputs[0].size() / 2);
    EXPECT_EQ(SOH_two_tile[0].outputs[0].get_shape()[1] * 2, wl.outputs[0].get_shape()[1]);

    EXPECT_EQ(SOH_four_tile[0].outputs[0].size(), wl.outputs[0].size() / 4);
    EXPECT_EQ(SOH_four_tile[0].outputs[0].get_shape()[1] * 4, wl.outputs[0].get_shape()[1]);
}

TEST_F(DPULayerTest, SplitAcrossTileSOK) {
    auto wl = generate_helper_layer(16, 64);

    auto SOK_single_tile = wl.splitAcrossTiles(VPUNN::VPUTilingStrategy::SOK, 1);
    auto SOK_two_tile = wl.splitAcrossTiles(VPUNN::VPUTilingStrategy::SOK, 2);
    auto SOK_four_tile = wl.splitAcrossTiles(VPUNN::VPUTilingStrategy::SOK, 4);

    // Basic expectations
    EXPECT_EQ(SOK_single_tile.size(), 1);
    EXPECT_EQ(SOK_two_tile.size(), 2);
    EXPECT_EQ(SOK_four_tile.size(), 4);

    // The shape of the single split must be equal to the origial layer
    EXPECT_EQ(SOK_single_tile[0].outputs[0].get_shape(), wl.outputs[0].get_shape());

    EXPECT_EQ(SOK_two_tile[0].outputs[0].size(), wl.outputs[0].size() / 2);
    EXPECT_EQ(SOK_two_tile[0].outputs[0].get_shape()[2] * 2, wl.outputs[0].get_shape()[2]);

    EXPECT_EQ(SOK_four_tile[0].outputs[0].size(), wl.outputs[0].size() / 4);
    EXPECT_EQ(SOK_four_tile[0].outputs[0].get_shape()[2] * 4, wl.outputs[0].get_shape()[2]);
}

TEST_F(DPULayerTest, SplitAcrossTileSOKAsymmetric) {
    auto wl = generate_helper_layer(16, 64);

    auto SOK_asymmetric = wl.splitAcrossTiles(VPUNN::VPUTilingStrategy::SOK, 5);

    // Basic expectations
    EXPECT_EQ(SOK_asymmetric.size(), 4);

    for (unsigned int idx = 0; idx < SOK_asymmetric.size(); idx++) {
        EXPECT_EQ(SOK_asymmetric[idx].outputs[0].get_shape()[2], 16u);
    }

    auto wl_2 = generate_helper_layer(16, 48);

    auto SOK_asymmetric_2 = wl_2.splitAcrossTiles(VPUNN::VPUTilingStrategy::SOK, 5);

    // Basic expectations
    EXPECT_EQ(SOK_asymmetric_2.size(), 3);

    for (unsigned int idx = 0; idx < SOK_asymmetric_2.size(); idx++) {
        EXPECT_EQ(SOK_asymmetric_2[idx].outputs[0].get_shape()[2], 16u);
    }
}

TEST_F(DPULayerTest, SplitSOHOVERLAPPED_basic_K3_2T) {
    // IN ,O, K , PT, PB, S
    const int k = 3;
    const std::vector<TestCase> tests{
            {{layer_H(5, 5, k, 1, 1, 1), 2},
             {
                     {4, 3},  // in
                     {3, 2},  // out
                     {k, k},  // k
                     {1, 0},  // pad top
                     {0, 1},  // pad bottom
             },
             "5-5 k3 p11"},
            {{layer_H(5, 3, 3, 0, 0, 1), 2},
             {
                     {4, 3},  // in
                     {2, 1},  // out
                     {k, k},  // k
                     {0, 0},  // pad top
                     {0, 0},  // pad bottom
             },
             "5-3 k3 p00"},
            {{layer_H(5, 4, 3, 1, 0, 1), 2},
             {
                     {3, 4},  // in
                     {2, 2},  // out
                     {k, k},  // k
                     {1, 0},  // pad top
                     {0, 0},  // pad bottom
             },
             "5-4 k3 p10"},
            {{layer_H(5, 4, 3, 0, 1, 1), 2},
             {
                     {4, 3},  // in
                     {2, 2},  // out
                     {k, k},  // k
                     {0, 0},  // pad top
                     {0, 1},  // pad bottom
             },
             "5-4 k3 p01"},
            /// 6 inputs (even)
            {{layer_H(6, 6, 3, 1, 1, 1), 2},
             {
                     {4, 4},  // in
                     {3, 3},  // out
                     {k, k},  // k
                     {1, 0},  // pad top
                     {0, 1},  // pad bottom
             },
             "6-6 k3 p11"},
            {{layer_H(6, 4, 3, 0, 0, 1), 2},
             {
                     {4, 4},  // in
                     {2, 2},  // out
                     {k, k},  // k
                     {0, 0},  // pad top
                     {0, 0},  // pad bottom
             },
             "6-4 k3 p00"},
            {{layer_H(6, 5, 3, 1, 0, 1), 2},
             {
                     {4, 4},  // in
                     {3, 2},  // out
                     {k, k},  // k
                     {1, 0},  // pad top
                     {0, 0},  // pad bottom
             },
             "6-5 k3 p10"},
            {{layer_H(6, 5, 3, 0, 1, 1), 2},
             {
                     {5, 3},  // in
                     {3, 2},  // out
                     {k, k},  // k
                     {0, 0},  // pad top
                     {0, 1},  // pad bottom
             },
             "6-5 k3 p01"},
    };

    // auto split = tst_layer_ref.splitAcrossTiles(VPUNN::VPUTilingStrategy::SOH, 2);
    // auto split = tst_layer_ref.SOH_overlapped(2);
    // auto split = tst_layer_ref.SOH(2);

    executeT_SOHOVR(tests);
}

TEST_F(DPULayerTest, SplitSOHOVERLAPPED_basic_K2_2T) {
    // IN ,O, K , PT, PB, S
    const int k = 2;
    const std::vector<TestCase> tests{
            {{layer_H(5, 6, k, 1, 1, 1), 2},
             {
                     {3, 3},  // in
                     {3, 3},  // out
                     {k, k},  // k
                     {1, 0},  // pad top
                     {0, 1},  // pad bottom
             },
             ""},
            {{layer_H(5, 4, k, 0, 0, 1), 2},
             {
                     {3, 3},  // in
                     {2, 2},  // out
                     {k, k},  // k
                     {0, 0},  // pad top
                     {0, 0},  // pad bottom
             },
             ""},
            {{layer_H(5, 5, k, 1, 0, 1), 2},
             {
                     {3, 3},  // in
                     {3, 2},  // out
                     {k, k},  // k
                     {1, 0},  // pad top
                     {0, 0},  // pad bottom
             },
             ""},
            {{layer_H(5, 5, k, 0, 1, 1), 2},
             {
                     {4, 2},  // in
                     {3, 2},  // out
                     {k, k},  // k
                     {0, 0},  // pad top
                     {0, 1},  // pad bottom
             },
             ""},
            /// 6 inputs (even)
            {{layer_H(6, 7, k, 1, 1, 1), 2},
             {
                     {4, 3},  // in
                     {4, 3},  // out
                     {k, k},  // k
                     {1, 0},  // pad top
                     {0, 1},  // pad bottom
             },
             ""},
            {{layer_H(6, 5, k, 0, 0, 1), 2},
             {
                     {4, 3},  // in
                     {3, 2},  // out
                     {k, k},  // k
                     {0, 0},  // pad top
                     {0, 0},  // pad bottom
             },
             ""},
            {{layer_H(6, 6, k, 1, 0, 1), 2},
             {
                     {3, 4},  // in
                     {3, 3},  // out
                     {k, k},  // k
                     {1, 0},  // pad top
                     {0, 0},  // pad bottom
             },
             ""},
            {{layer_H(6, 6, k, 0, 1, 1), 2},
             {
                     {4, 3},  // in
                     {3, 3},  // out
                     {k, k},  // k
                     {0, 0},  // pad top
                     {0, 1},  // pad bottom
             },
             ""},
    };

    executeT_SOHOVR(tests);
}

TEST_F(DPULayerTest, SplitSOHOVERLAPPED_basic_K1_2T) {
    // IN ,O, K , PT, PB, S
    const int k = 1;
    const std::vector<TestCase> tests{
            {{layer_H(5, 7, k, 1, 1, 1), 2},
             {
                     {3, 2},  // in
                     {4, 3},  // out
                     {k, k},  // k
                     {1, 0},  // pad top
                     {0, 1},  // pad bottom
             },
             ""},
            {{layer_H(5, 5, k, 0, 0, 1), 2},
             {
                     {3, 2},  // in
                     {3, 2},  // out
                     {k, k},  // k
                     {0, 0},  // pad top
                     {0, 0},  // pad bottom
             },
             ""},
            {{layer_H(5, 6, k, 1, 0, 1), 2},
             {
                     {2, 3},  // in
                     {3, 3},  // out
                     {k, k},  // k
                     {1, 0},  // pad top
                     {0, 0},  // pad bottom
             },
             ""},
            {{layer_H(5, 6, k, 0, 1, 1), 2},
             {
                     {3, 2},  // in
                     {3, 3},  // out
                     {k, k},  // k
                     {0, 0},  // pad top
                     {0, 1},  // pad bottom
             },
             ""},
            /// 6 inputs (even)
            {{layer_H(6, 8, k, 1, 1, 1), 2},
             {
                     {3, 3},  // in
                     {4, 4},  // out
                     {k, k},  // k
                     {1, 0},  // pad top
                     {0, 1},  // pad bottom
             },
             ""},
            {{layer_H(6, 6, k, 0, 0, 1), 2},
             {
                     {3, 3},  // in
                     {3, 3},  // out
                     {k, k},  // k
                     {0, 0},  // pad top
                     {0, 0},  // pad bottom
             },
             ""},
            {{layer_H(6, 7, k, 1, 0, 1), 2},
             {
                     {3, 3},  // in
                     {4, 3},  // out
                     {k, k},  // k
                     {1, 0},  // pad top
                     {0, 0},  // pad bottom
             },
             ""},
            {{layer_H(6, 7, k, 0, 1, 1), 2},
             {
                     {4, 2},  // in
                     {4, 3},  // out
                     {k, k},  // k
                     {0, 0},  // pad top
                     {0, 1},  // pad bottom
             },
             ""},
    };

    executeT_SOHOVR(tests);
}

TEST_F(DPULayerTest, SplitSOHOVERLAPPED_basic_K4_2T) {
    // IN ,O, K , PT, PB, S
    int k = 4;
    const std::vector<TestCase> tests{
            {{layer_H(5, 4, k, 1, 1, 1), 2},
             {
                     {4, 4},  // in
                     {2, 2},  // out
                     {k, k},  // k
                     {1, 0},  // pad top
                     {0, 1},  // pad bottom
             },
             ""},
            {{layer_H(5, 2, k, 0, 0, 1), 2},
             {
                     {4, 4},  // in
                     {1, 1},  // out
                     {k, k},  // k
                     {0, 0},  // pad top
                     {0, 0},  // pad bottom
             },
             ""},
            {{layer_H(5, 3, k, 1, 0, 1), 2},
             {
                     {4, 4},  // in
                     {2, 1},  // out
                     {k, k},  // k
                     {1, 0},  // pad top
                     {0, 0},  // pad bottom
             },
             ""},
            {{layer_H(5, 3, k, 0, 1, 1), 2},
             {
                     {5, 3},  // in
                     {2, 1},  // out
                     {k, k},  // k
                     {0, 0},  // pad top
                     {0, 1},  // pad bottom
             },
             ""},
            /// 5 inputs  , p2 combinations
            {{layer_H(5, 6, k, 2, 2, 1), 2},
             {
                     {4, 4},  // in
                     {3, 3},  // out
                     {k, k},  // k
                     {2, 0},  // pad top
                     {0, 2},  // pad bottom
             },
             "P+"},
            {{layer_H(5, 4, k, 2, 0, 1), 2},
             {
                     {3, 5},  // in
                     {2, 2},  // out
                     {k, k},  // k
                     {2, 0},  // pad top
                     {0, 0},  // pad bottom
             },
             ""},
            {{layer_H(5, 4, k, 0, 2, 1), 2},
             {
                     {5, 3},  // in
                     {2, 2},  // out
                     {k, k},  // k
                     {0, 0},  // pad top
                     {0, 2},  // pad bottom
             },
             ""},
            {{layer_H(5, 5, k, 2, 1, 1), 2},
             {
                     {4, 4},  // in
                     {3, 2},  // out
                     {k, k},  // k
                     {2, 0},  // pad top
                     {0, 1},  // pad bottom
             },
             ""},
            {{layer_H(5, 5, k, 1, 2, 1), 2},
             {
                     {5, 3},  // in
                     {3, 2},  // out
                     {k, k},  // k
                     {1, 0},  // pad top
                     {0, 2},  // pad bottom
             },
             ""},
    };

    executeT_SOHOVR(tests);
}

TEST_F(DPULayerTest, SplitSOHOVERLAPPED_basic_K5_2T) {
    // IN ,O, K , PT, PB, S
    int k = 5;
    const std::vector<TestCase> tests{
            {{layer_H(5, 3, k, 1, 1, 1), 2},
             {
                     {5, 4},  // in
                     {2, 1},  // out
                     {k, k},  // k
                     {1, 0},  // pad top
                     {0, 1},  // pad bottom
             },
             ""},
            //{{layer_H(5, 1, k, 0, 0, 1), 2},
            // {
            //         {4, 4},  // in
            //         {1, 0},  // out
            //         {k, k},  // k
            //         {0, 0},  // pad top
            //         {0, 0},  // pad bottom
            // },
            // ""},
            {{layer_H(5, 2, k, 1, 0, 1), 2},
             {
                     {4, 5},  // in
                     {1, 1},  // out
                     {k, k},  // k
                     {1, 0},  // pad top
                     {0, 0},  // pad bottom
             },
             ""},
            {{layer_H(5, 2, k, 0, 1, 1), 2},
             {
                     {5, 4},  // in
                     {1, 1},  // out
                     {k, k},  // k
                     {0, 0},  // pad top
                     {0, 1},  // pad bottom
             },
             ""},
            /// 5 inputs  , p2 combinations
            {{layer_H(5, 5, k, 2, 2, 1), 2},
             {
                     {5, 4},  // in
                     {3, 2},  // out
                     {k, k},  // k
                     {2, 0},  // pad top
                     {0, 2},  // pad bottom
             },
             "P+"},
            {{layer_H(5, 3, k, 2, 0, 1), 2},
             {
                     {4, 5},  // in
                     {2, 1},  // out
                     {k, k},  // k
                     {2, 0},  // pad top
                     {0, 0},  // pad bottom
             },
             ""},
            {{layer_H(5, 3, k, 0, 2, 1), 2},
             {
                     {5, 3},  // in
                     {2, 1},  // out
                     {k, k},  // k
                     {0, 0},  // pad top
                     {1, 2},  // pad bottom
             },
             " GROWS MORE THAN INput!!"},
            {{layer_H(5, 4, k, 2, 1, 1), 2},
             {
                     {4, 5},  // in
                     {2, 2},  // out
                     {k, k},  // k
                     {2, 0},  // pad top
                     {0, 1},  // pad bottom
             },
             ""},
            {{layer_H(5, 4, k, 1, 2, 1), 2},
             {
                     {5, 4},  // in
                     {2, 2},  // out
                     {k, k},  // k
                     {1, 0},  // pad top
                     {0, 2},  // pad bottom
             },
             ""},
    };

    executeT_SOHOVR(tests);
}

TEST_F(DPULayerTest, SplitSOHOVERLAPPED_basic_K7_2T_CornerC) {
    // IN ,O, K , PT, PB, S
    int k = 7;
    const std::vector<TestCase> tests{
            {{layer_H(2, 2, k, 3, 3, 1), 2},
             {
                     {2, 2},  // in
                     {1, 1},  // out
                     {k, k},  // k
                     {3, 2},  // pad top
                     {2, 3},  // pad bottom
             },
             "Both Tile takes all, except pad "},

    };

    executeT_SOHOVR(tests);
}

TEST_F(DPULayerTest, SplitSOHOVERLAPPED_basic_K3_3T) {
    // IN ,O, K , PT, PB, S
    const int T = 3;
    const int k = 3;
    const std::vector<TestCase> tests{
            // div by T
            {{layer_H(15, 15, k, 1, 1, 1), T},
             {
                     {6, 7, 6},  // in
                     {5, 5, 5},  // out
                     {k, k, k},  // k
                     {1, 0, 0},  // pad top
                     {0, 0, 1},  // pad bottom
             },
             ""},
            {{layer_H(15, 13, k, 0, 0, 1), T},
             {
                     {7, 7, 5},  // in
                     {5, 5, 3},  // out
                     {k, k, k},  // k
                     {0, 0, 0},  // pad top
                     {0, 0, 0},  // pad bottom
             },
             ""},
            {{layer_H(15, 14, k, 1, 0, 1), T},
             {
                     {6, 7, 6},  // in
                     {5, 5, 4},  // out
                     {k, k, k},  // k
                     {1, 0, 0},  // pad top
                     {0, 0, 0},  // pad bottom
             },
             ""},
            {{layer_H(15, 14, k, 0, 1, 1), T},
             {
                     {7, 7, 5},  // in
                     {5, 5, 4},  // out
                     {k, k, k},  // k
                     {0, 0, 0},  // pad top
                     {0, 0, 1},  // pad bottom
             },
             ""},
            /// 16 inputs (mod+1)
            {{layer_H(16, 16, k, 1, 1, 1), T},
             {
                     {7, 8, 5},  // in
                     {6, 6, 4},  // out
                     {k, k, k},  // k
                     {1, 0, 0},  // pad top
                     {0, 0, 1},  // pad bottom
             },
             "Goes beyond"},
            {{layer_H(16, 14, k, 0, 0, 1), T},
             {
                     {7, 7, 6},  // in
                     {5, 5, 4},  // out
                     {k, k, k},  // k
                     {0, 0, 0},  // pad top
                     {0, 0, 0},  // pad bottom
             },
             ""},
            {{layer_H(16, 15, k, 1, 0, 1), T},
             {
                     {6, 7, 7},  // in
                     {5, 5, 5},  // out
                     {k, k, k},  // k
                     {1, 0, 0},  // pad top
                     {0, 0, 0},  // pad bottom
             },
             ""},
            {{layer_H(16, 15, k, 0, 1, 1), T},
             {
                     {7, 7, 6},  // in
                     {5, 5, 5},  // out
                     {k, k, k},  // k
                     {0, 0, 0},  // pad top
                     {0, 0, 1},  // pad bottom
             },
             ""},

            /// 17 inputs (mod+2)
            {{layer_H(17, 17, k, 1, 1, 1), T},
             {
                     {7, 8, 6},  // in
                     {6, 6, 5},  // out
                     {k, k, k},  // k
                     {1, 0, 0},  // pad top
                     {0, 0, 1},  // pad bottom
             },
             "Goes beyond"},
            {{layer_H(17, 15, k, 0, 0, 1), T},
             {
                     {7, 7, 7},  // in
                     {5, 5, 5},  // out
                     {k, k, k},  // k
                     {0, 0, 0},  // pad top
                     {0, 0, 0},  // pad bottom
             },
             ""},
            {{layer_H(17, 16, k, 1, 0, 1), T},
             {
                     {7, 8, 6},  // in
                     {6, 6, 4},  // out
                     {k, k, k},  // k
                     {1, 0, 0},  // pad top
                     {0, 0, 0},  // pad bottom
             },
             ""},
            {{layer_H(17, 16, k, 0, 1, 1), T},
             {
                     {8, 8, 5},  // in
                     {6, 6, 4},  // out
                     {k, k, k},  // k
                     {0, 0, 0},  // pad top
                     {0, 0, 1},  // pad bottom
             },
             ""},
    };

    executeT_SOHOVR(tests);
}

TEST_F(DPULayerTest, SplitSOHOVERLAPPED_basic_K5_3T) {
    // IN ,O, K , PT, PB, S
    int k = 5;
    const std::vector<TestCase> tests{
            {{layer_H(5, 3, k, 1, 1, 1), 3},
             {
                     {4, 5, 4},  // in
                     {1, 1, 1},  // out
                     {k, k, k},  // k
                     {1, 0, 0},  // pad top
                     {0, 0, 1},  // pad bottom
             },
             ""},
            //{{layer_H(5, 1, k, 0, 0, 1), 2},
            // {
            //         {4, 4},  // in
            //         {1, 0},  // out
            //         {k, k},  // k
            //         {0, 0},  // pad top
            //         {0, 0},  // pad bottom
            // },
            // ""},
            //{{layer_H(5, 2, k, 1, 0, 1), 3},
            // {
            //         {4, 5},  // in
            //         {1, 1},  // out
            //         {k, k},  // k
            //         {1, 0},  // pad top
            //         {0, 0},  // pad bottom
            // },
            // ""},
            //{{layer_H(5, 2, k, 0, 1, 1), 3},
            // {
            //         {5, 4},  // in
            //         {1, 1},  // out
            //         {k, k},  // k
            //         {0, 0},  // pad top
            //         {0, 1},  // pad bottom
            // },
            // ""},
            /// 5 inputs  , p2 combinations
            {{layer_H(5, 5, k, 2, 2, 1), 3},
             {
                     {4, 5, 3},  // in
                     {2, 2, 1},  // out
                     {k, k, k},  // k
                     {2, 0, 0},  // pad top
                     {0, 1, 2},  // pad bottom
             },
             "Second is at end already"},
            {{layer_H(5, 3, k, 2, 0, 1), 3},
             {
                     {3, 4, 5},  // in
                     {1, 1, 1},  // out
                     {k, k, k},  // k
                     {2, 1, 0},  // pad top
                     {0, 0, 0},  // pad bottom
             },
             "second still uses pad begin layer"},
            {{layer_H(5, 3, k, 0, 2, 1), 3},
             {
                     {5, 4, 3},  // in
                     {1, 1, 1},  // out
                     {k, k, k},  // k
                     {0, 0, 0},  // pad top
                     {0, 1, 2},  // pad bottom
             },
             " "},
            {{layer_H(5, 4, k, 2, 1, 1), 3},
             {
                     {4, 5},  // in
                     {2, 2},  // out
                     {k, k},  // k
                     {2, 0},  // pad top
                     {0, 1},  // pad bottom
             },
             "ONly 2 splits a."},
            {{layer_H(5, 4, k, 1, 2, 1), 3},
             {
                     {5, 4},  // in
                     {2, 2},  // out
                     {k, k},  // k
                     {1, 0},  // pad top
                     {0, 2},  // pad bottom
             },
             "only 2 splits b."},
    };

    executeT_SOHOVR(tests);
}

TEST_F(DPULayerTest, SplitSOHOVERLAPPED_S2_K3_2T) {
    // IN ,O, K , PT, PB, S
    const int k = 3;
    const std::vector<TestCase> tests{
            {{layer_H(5, 3, k, 1, 1, 2), 2},
             {
                     {4, 2},  // in
                     {2, 1},  // out
                     {k, k},  // k
                     {1, 0},  // pad top
                     {0, 1},  // pad bottom
             },
             ""},
            {{layer_H(5, 2, 3, 0, 0, 2), 2},
             {
                     {3, 3},  // in
                     {1, 1},  // out
                     {k, k},  // k
                     {0, 0},  // pad top
                     {0, 0},  // pad bottom
             },
             ""},
            {{layer_H(5, 2, 3, 1, 0, 2), 2},
             {
                     {2, 3},  // in
                     {1, 1},  // out
                     {k, k},  // k
                     {1, 0},  // pad top
                     {0, 0},  // pad bottom
             },
             ""},
            {{layer_H(5, 2, 3, 0, 1, 2), 2},
             {
                     {3, 3},  // in
                     {1, 1},  // out
                     {k, k},  // k
                     {0, 0},  // pad top
                     {0, 1},  // pad bottom
             },
             "extra padding is bad! "},
            ///// 6 inputs (even)
            {{layer_H(6, 3, 3, 1, 1, 2), 2},
             {
                     {4, 3},  // in
                     {2, 1},  // out
                     {k, k},  // k
                     {1, 0},  // pad top
                     {0, 1},  // pad bottom
             },
             "extra pad! "},
            {{layer_H(6, 2, 3, 0, 0, 2), 2},
             {
                     {3, 3},  // in
                     {1, 1},  // out
                     {k, k},  // k
                     {0, 0},  // pad top
                     {0, 0},  // pad bottom
             },
             "MOre needs"},
            {{layer_H(6, 3, 3, 1, 0, 2), 2},
             {
                     {4, 3},  // in
                     {2, 1},  // out
                     {k, k},  // k
                     {1, 0},  // pad top
                     {0, 0},  // pad bottom
             },
             ""},
            {{layer_H(6, 3, 3, 0, 1, 2), 2},
             {
                     {5, 2},  // in
                     {2, 1},  // out
                     {k, k},  // k
                     {0, 0},  // pad top
                     {0, 1},  // pad bottom
             },
             ""},
    };

    // auto split = tst_layer_ref.splitAcrossTiles(VPUNN::VPUTilingStrategy::SOH, 2);
    // auto split = tst_layer_ref.SOH_overlapped(2);
    // auto split = tst_layer_ref.SOH(2);

    executeT_SOHOVR(tests);
}

TEST_F(DPULayerTest, SplitSOHOVERLAPPED_S1_K3_2T_EISW_76882) {
    // const VPUNN::DPULayer tst_layer_ref(
    //         VPUNN::VPUDevice::VPU_2_7, VPUNN::Operation::CONVOLUTION,
    //         {VPUNN::VPUTensor(60, 7, 512, 1, VPUNN::DataType::FLOAT16)},  // input dimensions
    //         {VPUNN::VPUTensor(60, 6, 512, 1, VPUNN::DataType::UINT8)},    // output dimensions
    //         {3, 3},                                                       // kernels
    //         {1, 1},                                                       // strides
    //         {1, 0, 1, 1}                                                  // padding
    //);

    // IN ,O, K , PT, PB, S
    const int k = 3;
    const std::vector<TestCase> tests{
            {{layer_H(7, 6, k, 1, 0, 1), 2},
             {
                     {4, 5},  // in
                     {3, 3},  // out
                     {k, k},  // k
                     {1, 0},  // pad top
                     {0, 0},  // pad bottom
             },
             ""},

    };

    executeT_SOHOVR(tests);
}

TEST_F(DPULayerTest, Split_SOHO_SOH_S8_K8_2T_DW_Cnv) {
    // const VPUNN::DPULayer tst_layer_ref(
    //         VPUNN::VPUDevice::VPU_2_7, VPUNN::Operation::DW_CONVOLUTION,
    //         {VPUNN::VPUTensor(8, 8, 128, 1, VPUNN::DataType::FLOAT16)},  // input dimensions
    //         {VPUNN::VPUTensor(1, 1, 128, 1, VPUNN::DataType::FLOAT16)},  // output dimensions
    //         {8, 8},                                                      // kernels
    //         {8, 8},                                                      // strides
    //         {0, 0, 0, 0}                                                 // padding
    //);

    //    const VPUNN::DPULayer tst_layer_ref(
    //        VPUNN::VPUDevice::VPU_2_7, VPUNN::Operation::DW_CONVOLUTION,
    //        {VPUNN::VPUTensor(32, 32, 64, 1, VPUNN::DataType::FLOAT16)},  // input dimensions
    //        {VPUNN::VPUTensor(4, 4, 64, 1, VPUNN::DataType::FLOAT16)},    // output dimensions
    //        {8, 8},                                                       // kernels
    //        {8, 8},                                                       // strides
    //        {0, 0, 0, 0}                                                  // padding
    //);

    // IN ,O, K , PT, PB, S
    const int k = 8;
    const std::vector<TestCase> tests_SOHO{
            {{layer_H(8, 1, k, 0, 0, 8), 2},
             {
                     {8},  // in
                     {1},  // out
                     {k},  // k
                     {0},  // pad top
                     {0},  // pad bottom
             },
             " One output H8"},
            {{layer_H(32, 4, k, 0, 0, 8), 2},
             {
                     {16, 16},  // in
                     {2, 2},    // out
                     {k, k},    // k
                     {0, 0},    // pad top
                     {0, 0},    // pad bottom
             },
             " 4 outputs H32"},
    };
    const std::vector<TestCase> tests_SOH{
            {{layer_H(8, 1, k, 0, 0, 8), 2},
             {
                     {8},                          // in
                     {1},                          // out
                     {k},                          // k
                     {0},                          // pad top
                     {0},                          // pad bottom
                     {ISIStrategy::SPLIT_OVER_H},  // ISI
             },
             " One output H8"},
            {{layer_H(32, 4, k, 0, 0, 8), 2},
             {
                     {16, 16},                                                // in
                     {2, 2},                                                  // out
                     {k, k},                                                  // k
                     {0, 0},                                                  // pad top
                     {0, 0},                                                  // pad bottom
                     {ISIStrategy::SPLIT_OVER_H, ISIStrategy::SPLIT_OVER_H},  // ISI
             },
             " 4 outputs H32"},
    };

    {
        std::function<std::vector<DPULayer>(const DPULayer*, unsigned int)> split_H =
                std::bind(&DPULayer::SOH_overlapped, _1, _2);

        std::function<void(const TestInput&, const TestExpectations&, const std::string&)> f =
                std::bind(&DPULayerTest::TestSoh, this, split_H, _1, _2, _3);

        executeT(f, tests_SOHO, " SOH OVERLAPPED : ");
    }

    {
        std::function<std::vector<DPULayer>(const DPULayer*, unsigned int)> split_H = std::bind(&DPULayer::SOH, _1, _2);

        std::function<void(const TestInput&, const TestExpectations&, const std::string&)> f =
                std::bind(&DPULayerTest::TestSoh, this, split_H, _1, _2, _3);

        executeT(f, tests_SOH, " SOH :");
    }
}

TEST_F(DPULayerTest, Split_SOHO_SOH_S1_K9_2T_redowa_deeplab_v3_dense_IRv11_FP16) {
    // IN ,O, K , PT, PB, S
    const int k = 9;
    const auto isiC = ISIStrategy::CLUSTERING;
    const auto isiH = ISIStrategy::SPLIT_OVER_H;

    const std::vector<TestCase> tests_SOHO{

            {{layer_H(16, 8, k, 0, 0, 1), 2},
             {
                     {12, 12},      // in
                     {4, 4},        // out
                     {k, k},        // k
                     {0, 0},        // pad top
                     {0, 0},        // pad bottom
                     {isiC, isiC},  // ISI
             },
             " redowa_deeplab_v3_dense_IRv11"},
    };
    const std::vector<TestCase> tests_SOH{

            {{layer_H(16, 8, k, 0, 0, 1), 2},
             {
                     {12, 12},      // in
                     {4, 4},        // out
                     {k, k},        // k
                     {0, 0},        // pad top
                     {0, 0},        // pad bottom
                     {isiH, isiH},  // ISI
             },
             " redowa_deeplab_v3_dense_IRv11"},
    };

    {
        std::function<std::vector<DPULayer>(const DPULayer*, unsigned int)> split_H =
                std::bind(&DPULayer::SOH_overlapped, _1, _2);

        std::function<void(const TestInput&, const TestExpectations&, const std::string&)> f =
                std::bind(&DPULayerTest::TestSoh, this, split_H, _1, _2, _3);

        executeT(f, tests_SOHO, " SOH OVERLAPPED : ");
    }

    {
        std::function<std::vector<DPULayer>(const DPULayer*, unsigned int)> split_H = std::bind(&DPULayer::SOH, _1, _2);

        std::function<void(const TestInput&, const TestExpectations&, const std::string&)> f =
                std::bind(&DPULayerTest::TestSoh, this, split_H, _1, _2, _3);

        executeT(f, tests_SOH, " SOH :");
    }
}

}  // namespace VPUNN_unit_tests
