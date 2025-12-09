// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.
#include "vpu_layer_cost_model.h"

#include <gtest/gtest.h>
#include <sstream>  // for error formating
#include "common/common_helpers.h"

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
    DPULayer layer_W(unsigned int w_input, unsigned int w_output, unsigned int kernel, unsigned int padleft,
                     unsigned int padright, unsigned int stride) {
        return DPULayer(VPUDevice::VPU_4_0, Operation::CONVOLUTION,
                        {VPUTensor(w_input, 1, 128, 1, DataType::FLOAT16)},   // input dimensions WH
                        {VPUTensor(w_output, 1, 128, 1, DataType::FLOAT16)},  // output dimensions
                        {kernel, 1},                                          // kernels WH
                        {stride, 1},                                          // stridesWH
                        {0, 0, padleft, padright}                             // padding TBLR
        );
    }

public:
    struct TestInput {
        DPULayer l1;

        unsigned int nTiles{1};  ///< Number of tiles
        // VPUTilingStrategy tiling_strategy{VPUTilingStrategy::NONE};  ///< tiling strategy
    };

    struct TestExpectations {
        std::vector<int> in_size;
        std::vector<int> out_size;
        std::vector<int> k;
        std::vector<int> pad1;
        std::vector<int> pad2;
        std::vector<ISIStrategy> isi{ISIStrategy::CLUSTERING};
        // halo aspects, for now only input halo dfor SOHO_in (zero) and SOH in_HALO  (must be set)
        std::vector<int> haloIn1{};  //<first in pair of halo
        std::vector<int> haloIn2{};  //<secnd in pair of halo

        TestExpectations(const std::vector<int>& in_size, const std::vector<int>& out_size, const std::vector<int>& k,
                         const std::vector<int>& pad1, const std::vector<int>& pad2)
                : in_size{in_size}, out_size{out_size}, k{k}, pad1{pad1}, pad2{pad2} {
        }

        TestExpectations(const std::vector<int>& in_size, const std::vector<int>& out_size, const std::vector<int>& k,
                         const std::vector<int>& pad1, const std::vector<int>& pad2,  //
                         const std::vector<ISIStrategy>& isi)
                : in_size{in_size}, out_size{out_size}, k{k}, pad1{pad1}, pad2{pad2}, isi{isi} {
        }
        TestExpectations(const std::vector<int>& in_size, const std::vector<int>& out_size, const std::vector<int>& k,
                         const std::vector<int>& pad1, const std::vector<int>& pad2,
                         const std::vector<ISIStrategy>& isi,  //
                         const std::vector<int>& haloIn1, const std::vector<int>& haloIn2)
                : in_size{in_size},
                  out_size{out_size},
                  k{k},
                  pad1{pad1},
                  pad2{pad2},
                  isi{isi},
                  haloIn1{haloIn1},
                  haloIn2{haloIn2} {
        }
        TestExpectations(const std::vector<int>& in_size, const std::vector<int>& out_size, const std::vector<int>& k,
                         const std::vector<int>& pad1, const std::vector<int>& pad2,
                         // const std::vector<ISIStrategy>& isi,  //
                         const std::vector<int>& haloIn1, const std::vector<int>& haloIn2)
                : in_size{in_size},
                  out_size{out_size},
                  k{k},
                  pad1{pad1},
                  pad2{pad2},
                  // isi{isi},
                  haloIn1{haloIn1},
                  haloIn2{haloIn2} {
        }

        bool has_consistent_size() const {
            const auto s{in_size.size()};
            if ((s == out_size.size()) && (s == k.size()) && (s == pad1.size()) &&
                (s == pad2.size())  //
                                    //&& (s == haloIn1.size()) && (s == haloIn2.size()) //
            ) {
                return true;
            }
            return false;
        }
    };

protected:
    struct TestCase {
        TestInput t_in;
        TestExpectations t_exp;
        const std::string test_case = "";
    };
    using TestsVector = std::vector<TestCase>;

    std::function<std::vector<DPULayer>(const DPULayer* /*thisObject*/, unsigned int /*nTiles*/)> split_SOHO_heuristic =
            std::bind(&DPULayer::SOH_overlapped_inputs, _1, _2,
                      false);  //_1 is this of DPULayer, _2 is n tiles, false is forced_broadcast

    std::function<void(const TestInput&, const TestExpectations&, const std::string&)> SOHO_test =
            std::bind(&DPULayerTest::TestSoh, this /*(DPULayerTest::TestSoh)*/,  //
                      split_SOHO_heuristic /*hSplit function*/,                  //
                      _1 /*t_in*/, _2 /*t_exp*/, _3 /*test_case*/);              //

    void executeT_SOHOVR(const TestsVector& tests) {
        executeT(SOHO_test, tests, "SOHOin:: ");
    }

    std::function<std::vector<DPULayer>(const DPULayer* /*thisObject*/, unsigned int /*nTiles*/)>
            split_SOH_inHALO_heuristic =
                    std::bind(&DPULayer::SOH_HALO_inputs, _1, _2);  //_1 is this of DPULayer, _2 is n tiles

    std::function<void(const TestInput&, const TestExpectations&, const std::string&)> SOH_inHALO_test =
            std::bind(&DPULayerTest::TestSoh, this /*(DPULayerTest::TestSoh)*/,  //
                      split_SOH_inHALO_heuristic /*hSplit function*/,            //
                      _1 /*t_in*/, _2 /*t_exp*/, _3 /*test_case*/);              //
    void executeT_SOH_inHALO(const TestsVector& tests) {
        executeT(SOH_inHALO_test, tests, "SOHinHALO:: ");
    }

    std::function<std::vector<DPULayer>(const DPULayer* /*thisObject*/, unsigned int /*nTiles*/)> split_SOWO_heuristic =
            std::bind(&DPULayer::SOW_overlapped_inputs, _1, _2,
                      false);  //_1 is this of DPULayer, _2 is n tiles, false is forced_broadcast

    std::function<void(const TestInput&, const TestExpectations&, const std::string&)> SOWO_test =
            std::bind(&DPULayerTest::TestSow, this /*(DPULayerTest::TestSow)*/,  //
                      split_SOWO_heuristic /*wSplit function*/,                  //
                      _1 /*t_in*/, _2 /*t_exp*/, _3 /*test_case*/);              //

    void executeT_SOWOVR(const TestsVector& tests) {
        executeT(SOWO_test, tests, "SOWOin:: ");
    }

public:
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

            int exp_inhalo1{(t_exp.haloIn1.size() > i) ? t_exp.haloIn1[i] : 0};  // zero halo by default
            int exp_inhalo2{(t_exp.haloIn2.size() > i) ? t_exp.haloIn2[i] : 0};  // zero halo by default
            ASSERT_EQ(s.halo.input_0_halo.top, exp_inhalo1) << "top in halo"
                                                            << " i= " << i << s << t_exp;
            ASSERT_EQ(s.halo.input_0_halo.bottom, exp_inhalo2) << "bot in halo"
                                                               << " i= " << i << s << t_exp;

            // check also stride??

            // can do checks that other fields are not changed!
        }

        std::cout << t_header << "------------------------------------------------------------------------"
                  << std::endl;
    }

    void TestSow(std::function<std::vector<DPULayer>(const DPULayer*, unsigned int)>& wSplit, const TestInput& t_in,
                 const TestExpectations& t_exp, const std::string& test_case = "") const {
        const DPULayer& l1{t_in.l1};
        const auto tiles{t_in.nTiles};

        std::string details = test_case + ":";
        {  // "5-5 k3 p11"},
            std::stringstream buffer;
            buffer << " [" << l1.inputs[0].width() << "-" << l1.outputs[0].width() << " k" << l1.kernels[Dim::W]
                   << " p" << l1.padding[Dim::LEFT] << "-" << l1.padding[Dim::RIGHT] << " s" << l1.strides[Dim::W]
                   << "]";

            details += buffer.str();
        }

        std::string t_header{"** Test Case: " + details + "\n"};
        std::cout << ">> " << t_header << std::endl;

        EXPECT_TRUE(t_exp.has_consistent_size()) << "problem with expected values consistency";

        auto split = wSplit(&l1, tiles);

        // Basic expectations
        ASSERT_EQ(split.size(), t_exp.in_size.size()) << "Expected size of split is different than real split";

        for (size_t i = 0; i < split.size(); ++i) {
            const auto& s = split[i];
            ASSERT_EQ(s.outputs[0].width(), t_exp.out_size[i]) << " i= " << i << s << t_exp;

            ASSERT_EQ(s.inputs[0].width(), t_exp.in_size[i]) << " i= " << i << s << t_exp;

            ASSERT_EQ(s.kernels[Dim::W], t_exp.k[i]) << " i= " << i << s << t_exp;
            ASSERT_EQ(s.padding[Dim::LEFT], t_exp.pad1[i]) << " i= " << i << s << t_exp;
            ASSERT_EQ(s.padding[Dim::RIGHT], t_exp.pad2[i]) << " i= " << i << s << t_exp;

            ISIStrategy expected_isi{(t_exp.isi.size() > i) ? t_exp.isi[i]
                                                            : ISIStrategy::CLUSTERING};  // clustering by default

            ASSERT_EQ(s.isi_strategy, expected_isi) << "ISI is marked BAD"
                                                    << " i= " << i << s << t_exp;
            ASSERT_EQ(s.output_write_tiles, l1.output_write_tiles) << "output_write_tiles not kept"
                                                                   << " i= " << i << s << t_exp << l1;

            int exp_inhalo1{(t_exp.haloIn1.size() > i) ? t_exp.haloIn1[i] : 0};  // zero halo by default
            int exp_inhalo2{(t_exp.haloIn2.size() > i) ? t_exp.haloIn2[i] : 0};  // zero halo by default
            ASSERT_EQ(s.halo.input_0_halo.left, exp_inhalo1) << "left in halo"
                                                            << " i= " << i << s << t_exp;
            ASSERT_EQ(s.halo.input_0_halo.right, exp_inhalo2) << "right in halo "
                                                               << " i= " << i << s << t_exp;

            // check also stride??

            // can do checks that other fields are not changed!
        }

        std::cout << t_header << "------------------------------------------------------------------------"
                  << std::endl;
    }

protected:
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

    const ISIStrategy isiC{ISIStrategy::CLUSTERING};
    const ISIStrategy isiH{ISIStrategy::SPLIT_OVER_H};
    const ISIStrategy isiK{ISIStrategy::SPLIT_OVER_K};
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
    {
        stream << "\nisi: ";
        std::for_each(t.isi.begin(), t.isi.end(), [&stream](auto& t) {
            stream << " " << ISIStrategy_ToText.at(static_cast<int>(t)) << " ,";
        });
    }

    {
        stream << "\nhaloIn1: ";
        std::for_each(t.haloIn1.begin(), t.haloIn1.end(), [&stream](auto& t) {
            stream << " " << t << " ,";
        });
    }
    {
        stream << "\nhaloIn2: ";
        std::for_each(t.haloIn2.begin(), t.haloIn2.end(), [&stream](auto& t) {
            stream << " " << t << " ,";
        });
    }

    stream << "\n";
    return stream;
}

TEST_F(DPULayerTest, SplitAcrossTileSOHO) {
    auto wl = generate_helper_layer(16, 64);

    auto SOH_single_tile = wl.splitAcrossTiles(VPUNN::VPUTilingStrategy::SOH_Overlapped, 1);
    auto SOH_two_tile = wl.splitAcrossTiles(VPUNN::VPUTilingStrategy::SOH_Overlapped, 2);
    auto SOH_four_tile = wl.splitAcrossTiles(VPUNN::VPUTilingStrategy::SOH_Overlapped, 4);

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

TEST_F(DPULayerTest, SplitAcrossTileSOW) {
    auto wl = generate_helper_layer(16, 64);

    auto SOH_single_tile = wl.splitAcrossTiles(VPUNN::VPUTilingStrategy::SOW, 1);
    auto SOH_two_tile = wl.splitAcrossTiles(VPUNN::VPUTilingStrategy::SOW, 2);
    auto SOH_four_tile = wl.splitAcrossTiles(VPUNN::VPUTilingStrategy::SOW, 4);

    // Basic expectations
    EXPECT_EQ(SOH_single_tile.size(), 1);
    EXPECT_EQ(SOH_two_tile.size(), 2);
    EXPECT_EQ(SOH_four_tile.size(), 4);

    // The shape of the single split must be equal to the origial layer
    EXPECT_EQ(SOH_single_tile[0].outputs[0].get_shape(), wl.outputs[0].get_shape());

    EXPECT_EQ(SOH_two_tile[0].outputs[0].size(), wl.outputs[0].size() / 2);
    EXPECT_EQ(SOH_two_tile[0].outputs[0].get_shape()[0] * 2, wl.outputs[0].get_shape()[0]);

    EXPECT_EQ(SOH_four_tile[0].outputs[0].size(), wl.outputs[0].size() / 4);
    EXPECT_EQ(SOH_four_tile[0].outputs[0].get_shape()[0] * 4, wl.outputs[0].get_shape()[0]);
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

    // The shape of the single split must be equal to the original layer
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

#define HALO_ZERO2            \
    {0, 0} /*in halo top*/, { \
        0, 0                  \
    } /*in halo bottom*/
#define HALO_ZERO3            \
    {0, 0} /*in halo top*/, { \
        0, 0                  \
    } /*in halo bottom*/

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
                              //{isiC, isiC},  // ISI
                              //{0, 0} /*in halo top*/,
                              //{0, 0} /*in halo bottom*/
                     HALO_ZERO2,
             },
             "5-5 k3 p11"},
            {{layer_H(5, 3, 3, 0, 0, 1), 2},
             {
                     {4, 3},  // in
                     {2, 1},  // out
                     {k, k},  // k
                     {0, 0},  // pad top
                     {0, 0},  // pad bottom
                     HALO_ZERO2,
             },
             "5-3 k3 p00"},
            {{layer_H(5, 4, 3, 1, 0, 1), 2},
             {
                     {3, 4},  // in
                     {2, 2},  // out
                     {k, k},  // k
                     {1, 0},  // pad top
                     {0, 0},  // pad bottom
                     HALO_ZERO2,
             },
             "5-4 k3 p10"},
            {{layer_H(5, 4, 3, 0, 1, 1), 2},
             {
                     {4, 3},  // in
                     {2, 2},  // out
                     {k, k},  // k
                     {0, 0},  // pad top
                     {0, 1},  // pad bottom
                     HALO_ZERO2,
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
                     HALO_ZERO2,
             },
             "6-6 k3 p11"},
            {{layer_H(6, 4, 3, 0, 0, 1), 2},
             {
                     {4, 4},  // in
                     {2, 2},  // out
                     {k, k},  // k
                     {0, 0},  // pad top
                     {0, 0},  // pad bottom
                     HALO_ZERO2,
             },
             "6-4 k3 p00"},
            {{layer_H(6, 5, 3, 1, 0, 1), 2},
             {
                     {4, 4},  // in
                     {3, 2},  // out
                     {k, k},  // k
                     {1, 0},  // pad top
                     {0, 0},  // pad bottom
                     HALO_ZERO2,
             },
             "6-5 k3 p10"},
            {{layer_H(6, 5, 3, 0, 1, 1), 2},
             {
                     {5, 3},  // in
                     {3, 2},  // out
                     {k, k},  // k
                     {0, 0},  // pad top
                     {0, 1},  // pad bottom
                     HALO_ZERO2,
             },
             "6-5 k3 p01"},
    };

    // auto split = tst_layer_ref.splitAcrossTiles(VPUNN::VPUTilingStrategy::SOH, 2);
    // auto split = tst_layer_ref.SOH_overlapped_inputs(2);
    // auto split = tst_layer_ref.SOH(2);

    executeT_SOHOVR(tests);
}

TEST_F(DPULayerTest, SplitSOWOVERLAPPED_basic_K3_2T) {
    // IN ,O, K , PT, PB, S
    const int k = 3;
    const std::vector<TestCase> tests{
            {{layer_W(5, 5, k, 1, 1, 1), 2},
             {
                     {4, 3},  // in
                     {3, 2},  // out
                     {k, k},  // k
                     {1, 0},  // pad top
                     {0, 1},  // pad bottom
                              //{isiC, isiC},  // ISI
                              //{0, 0} /*in halo top*/,
                              //{0, 0} /*in halo bottom*/
                     HALO_ZERO2,
             },
             "5-5 k3 p11"},
            {{layer_W(5, 3, 3, 0, 0, 1), 2},
             {
                     {4, 3},  // in
                     {2, 1},  // out
                     {k, k},  // k
                     {0, 0},  // pad top
                     {0, 0},  // pad bottom
                     HALO_ZERO2,
             },
             "5-3 k3 p00"},
            {{layer_W(5, 4, 3, 1, 0, 1), 2},
             {
                     {3, 4},  // in
                     {2, 2},  // out
                     {k, k},  // k
                     {1, 0},  // pad top
                     {0, 0},  // pad bottom
                     HALO_ZERO2,
             },
             "5-4 k3 p10"},
            {{layer_W(5, 4, 3, 0, 1, 1), 2},
             {
                     {4, 3},  // in
                     {2, 2},  // out
                     {k, k},  // k
                     {0, 0},  // pad top
                     {0, 1},  // pad bottom
                     HALO_ZERO2,
             },
             "5-4 k3 p01"},
            /// 6 inputs (even)
            {{layer_W(6, 6, 3, 1, 1, 1), 2},
             {
                     {4, 4},  // in
                     {3, 3},  // out
                     {k, k},  // k
                     {1, 0},  // pad top
                     {0, 1},  // pad bottom
                     HALO_ZERO2,
             },
             "6-6 k3 p11"},
            {{layer_W(6, 4, 3, 0, 0, 1), 2},
             {
                     {4, 4},  // in
                     {2, 2},  // out
                     {k, k},  // k
                     {0, 0},  // pad top
                     {0, 0},  // pad bottom
                     HALO_ZERO2,
             },
             "6-4 k3 p00"},
            {{layer_W(6, 5, 3, 1, 0, 1), 2},
             {
                     {4, 4},  // in
                     {3, 2},  // out
                     {k, k},  // k
                     {1, 0},  // pad top
                     {0, 0},  // pad bottom
                     HALO_ZERO2,
             },
             "6-5 k3 p10"},
            {{layer_W(6, 5, 3, 0, 1, 1), 2},
             {
                     {5, 3},  // in
                     {3, 2},  // out
                     {k, k},  // k
                     {0, 0},  // pad top
                     {0, 1},  // pad bottom
                     HALO_ZERO2,
             },
             "6-5 k3 p01"},
    };

    executeT_SOWOVR(tests);
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
                     HALO_ZERO2,
             },
             ""},
            {{layer_H(5, 4, k, 0, 0, 1), 2},
             {
                     {3, 3},  // in
                     {2, 2},  // out
                     {k, k},  // k
                     {0, 0},  // pad top
                     {0, 0},  // pad bottom
                     HALO_ZERO2,
             },
             ""},
            {{layer_H(5, 5, k, 1, 0, 1), 2},
             {
                     {3, 3},  // in
                     {3, 2},  // out
                     {k, k},  // k
                     {1, 0},  // pad top
                     {0, 0},  // pad bottom
                     HALO_ZERO2,
             },
             ""},
            {{layer_H(5, 5, k, 0, 1, 1), 2},
             {
                     {4, 2},  // in
                     {3, 2},  // out
                     {k, k},  // k
                     {0, 0},  // pad top
                     {0, 1},  // pad bottom
                     HALO_ZERO2,
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
                     HALO_ZERO2,
             },
             ""},
            {{layer_H(6, 5, k, 0, 0, 1), 2},
             {
                     {4, 3},  // in
                     {3, 2},  // out
                     {k, k},  // k
                     {0, 0},  // pad top
                     {0, 0},  // pad bottom
                     HALO_ZERO2,
             },
             ""},
            {{layer_H(6, 6, k, 1, 0, 1), 2},
             {
                     {3, 4},  // in
                     {3, 3},  // out
                     {k, k},  // k
                     {1, 0},  // pad top
                     {0, 0},  // pad bottom
                     HALO_ZERO2,
             },
             ""},
            {{layer_H(6, 6, k, 0, 1, 1), 2},
             {
                     {4, 3},  // in
                     {3, 3},  // out
                     {k, k},  // k
                     {0, 0},  // pad top
                     {0, 1},  // pad bottom
                     HALO_ZERO2,
             },
             ""},
    };

    executeT_SOHOVR(tests);
}

TEST_F(DPULayerTest, SplitSOWOVERLAPPED_basic_K2_2T) {
    // IN ,O, K , PT, PB, S
    const int k = 2;
    const std::vector<TestCase> tests{
            {{layer_W(5, 6, k, 1, 1, 1), 2},
             {
                     {3, 3},  // in
                     {3, 3},  // out
                     {k, k},  // k
                     {1, 0},  // pad top
                     {0, 1},  // pad bottom
                     HALO_ZERO2,
             },
             ""},
            {{layer_W(5, 4, k, 0, 0, 1), 2},
             {
                     {3, 3},  // in
                     {2, 2},  // out
                     {k, k},  // k
                     {0, 0},  // pad top
                     {0, 0},  // pad bottom
                     HALO_ZERO2,
             },
             ""},
            {{layer_W(5, 5, k, 1, 0, 1), 2},
             {
                     {3, 3},  // in
                     {3, 2},  // out
                     {k, k},  // k
                     {1, 0},  // pad top
                     {0, 0},  // pad bottom
                     HALO_ZERO2,
             },
             ""},
            {{layer_W(5, 5, k, 0, 1, 1), 2},
             {
                     {4, 2},  // in
                     {3, 2},  // out
                     {k, k},  // k
                     {0, 0},  // pad top
                     {0, 1},  // pad bottom
                     HALO_ZERO2,
             },
             ""},
            /// 6 inputs (even)
            {{layer_W(6, 7, k, 1, 1, 1), 2},
             {
                     {4, 3},  // in
                     {4, 3},  // out
                     {k, k},  // k
                     {1, 0},  // pad top
                     {0, 1},  // pad bottom
                     HALO_ZERO2,
             },
             ""},
            {{layer_W(6, 5, k, 0, 0, 1), 2},
             {
                     {4, 3},  // in
                     {3, 2},  // out
                     {k, k},  // k
                     {0, 0},  // pad top
                     {0, 0},  // pad bottom
                     HALO_ZERO2,
             },
             ""},
            {{layer_W(6, 6, k, 1, 0, 1), 2},
             {
                     {3, 4},  // in
                     {3, 3},  // out
                     {k, k},  // k
                     {1, 0},  // pad top
                     {0, 0},  // pad bottom
                     HALO_ZERO2,
             },
             ""},
            {{layer_W(6, 6, k, 0, 1, 1), 2},
             {
                     {4, 3},  // in
                     {3, 3},  // out
                     {k, k},  // k
                     {0, 0},  // pad top
                     {0, 1},  // pad bottom
                     HALO_ZERO2,
             },
             ""},
    };

    executeT_SOWOVR(tests);
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
                     HALO_ZERO2,
             },
             ""},
            {{layer_H(5, 5, k, 0, 0, 1), 2},
             {
                     {3, 2},  // in
                     {3, 2},  // out
                     {k, k},  // k
                     {0, 0},  // pad top
                     {0, 0},  // pad bottom
                     HALO_ZERO2,
             },
             ""},
            {{layer_H(5, 6, k, 1, 0, 1), 2},
             {
                     {2, 3},  // in
                     {3, 3},  // out
                     {k, k},  // k
                     {1, 0},  // pad top
                     {0, 0},  // pad bottom
                     HALO_ZERO2,
             },
             ""},
            {{layer_H(5, 6, k, 0, 1, 1), 2},
             {
                     {3, 2},  // in
                     {3, 3},  // out
                     {k, k},  // k
                     {0, 0},  // pad top
                     {0, 1},  // pad bottom
                     HALO_ZERO2,
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
                     HALO_ZERO2,
             },
             ""},
            {{layer_H(6, 6, k, 0, 0, 1), 2},
             {
                     {3, 3},  // in
                     {3, 3},  // out
                     {k, k},  // k
                     {0, 0},  // pad top
                     {0, 0},  // pad bottom
                     HALO_ZERO2,
             },
             ""},
            {{layer_H(6, 7, k, 1, 0, 1), 2},
             {
                     {3, 3},  // in
                     {4, 3},  // out
                     {k, k},  // k
                     {1, 0},  // pad top
                     {0, 0},  // pad bottom
                     HALO_ZERO2,
             },
             ""},
            {{layer_H(6, 7, k, 0, 1, 1), 2},
             {
                     {4, 2},  // in
                     {4, 3},  // out
                     {k, k},  // k
                     {0, 0},  // pad top
                     {0, 1},  // pad bottom
                     HALO_ZERO2,
             },
             ""},
    };

    executeT_SOHOVR(tests);
}

TEST_F(DPULayerTest, SplitSOWOVERLAPPED_basic_K1_2T) {
    // IN ,O, K , PT, PB, S
    const int k = 1;
    const std::vector<TestCase> tests{
            {{layer_W(5, 7, k, 1, 1, 1), 2},
             {
                     {3, 2},  // in
                     {4, 3},  // out
                     {k, k},  // k
                     {1, 0},  // pad top
                     {0, 1},  // pad bottom
                     HALO_ZERO2,
             },
             ""},
            {{layer_W(5, 5, k, 0, 0, 1), 2},
             {
                     {3, 2},  // in
                     {3, 2},  // out
                     {k, k},  // k
                     {0, 0},  // pad top
                     {0, 0},  // pad bottom
                     HALO_ZERO2,
             },
             ""},
            {{layer_W(5, 6, k, 1, 0, 1), 2},
             {
                     {2, 3},  // in
                     {3, 3},  // out
                     {k, k},  // k
                     {1, 0},  // pad top
                     {0, 0},  // pad bottom
                     HALO_ZERO2,
             },
             ""},
            {{layer_W(5, 6, k, 0, 1, 1), 2},
             {
                     {3, 2},  // in
                     {3, 3},  // out
                     {k, k},  // k
                     {0, 0},  // pad top
                     {0, 1},  // pad bottom
                     HALO_ZERO2,
             },
             ""},
            /// 6 inputs (even)
            {{layer_W(6, 8, k, 1, 1, 1), 2},
             {
                     {3, 3},  // in
                     {4, 4},  // out
                     {k, k},  // k
                     {1, 0},  // pad top
                     {0, 1},  // pad bottom
                     HALO_ZERO2,
             },
             ""},
            {{layer_W(6, 6, k, 0, 0, 1), 2},
             {
                     {3, 3},  // in
                     {3, 3},  // out
                     {k, k},  // k
                     {0, 0},  // pad top
                     {0, 0},  // pad bottom
                     HALO_ZERO2,
             },
             ""},
            {{layer_W(6, 7, k, 1, 0, 1), 2},
             {
                     {3, 3},  // in
                     {4, 3},  // out
                     {k, k},  // k
                     {1, 0},  // pad top
                     {0, 0},  // pad bottom
                     HALO_ZERO2,
             },
             ""},
            {{layer_W(6, 7, k, 0, 1, 1), 2},
             {
                     {4, 2},  // in
                     {4, 3},  // out
                     {k, k},  // k
                     {0, 0},  // pad top
                     {0, 1},  // pad bottom
                     HALO_ZERO2,
             },
             ""},
    };

    executeT_SOWOVR(tests);
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
                     HALO_ZERO2,
             },
             ""},
            {{layer_H(5, 2, k, 0, 0, 1), 2},
             {
                     {4, 4},  // in
                     {1, 1},  // out
                     {k, k},  // k
                     {0, 0},  // pad top
                     {0, 0},  // pad bottom
                     HALO_ZERO2,
             },
             ""},
            {{layer_H(5, 3, k, 1, 0, 1), 2},
             {
                     {4, 4},  // in
                     {2, 1},  // out
                     {k, k},  // k
                     {1, 0},  // pad top
                     {0, 0},  // pad bottom
                     HALO_ZERO2,
             },
             ""},
            {{layer_H(5, 3, k, 0, 1, 1), 2},
             {
                     {5, 3},  // in
                     {2, 1},  // out
                     {k, k},  // k
                     {0, 0},  // pad top
                     {0, 1},  // pad bottom
                     HALO_ZERO2,
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
                     HALO_ZERO2,
             },
             "P+"},
            {{layer_H(5, 4, k, 2, 0, 1), 2},
             {
                     {3, 5},  // in
                     {2, 2},  // out
                     {k, k},  // k
                     {2, 0},  // pad top
                     {0, 0},  // pad bottom
                     HALO_ZERO2,
             },
             ""},
            {{layer_H(5, 4, k, 0, 2, 1), 2},
             {
                     {5, 3},  // in
                     {2, 2},  // out
                     {k, k},  // k
                     {0, 0},  // pad top
                     {0, 2},  // pad bottom
                     HALO_ZERO2,
             },
             ""},
            {{layer_H(5, 5, k, 2, 1, 1), 2},
             {
                     {4, 4},  // in
                     {3, 2},  // out
                     {k, k},  // k
                     {2, 0},  // pad top
                     {0, 1},  // pad bottom
                     HALO_ZERO2,
             },
             ""},
            {{layer_H(5, 5, k, 1, 2, 1), 2},
             {
                     {5, 3},  // in
                     {3, 2},  // out
                     {k, k},  // k
                     {1, 0},  // pad top
                     {0, 2},  // pad bottom
                     HALO_ZERO2,
             },
             ""},
    };

    executeT_SOHOVR(tests);
}

TEST_F(DPULayerTest, SplitSOWOVERLAPPED_basic_K4_2T) {
    // IN ,O, K , PT, PB, S
    int k = 4;
    const std::vector<TestCase> tests{
            {{layer_W(5, 4, k, 1, 1, 1), 2},
             {
                     {4, 4},  // in
                     {2, 2},  // out
                     {k, k},  // k
                     {1, 0},  // pad top
                     {0, 1},  // pad bottom
                     HALO_ZERO2,
             },
             ""},
            {{layer_W(5, 2, k, 0, 0, 1), 2},
             {
                     {4, 4},  // in
                     {1, 1},  // out
                     {k, k},  // k
                     {0, 0},  // pad top
                     {0, 0},  // pad bottom
                     HALO_ZERO2,
             },
             ""},
            {{layer_W(5, 3, k, 1, 0, 1), 2},
             {
                     {4, 4},  // in
                     {2, 1},  // out
                     {k, k},  // k
                     {1, 0},  // pad top
                     {0, 0},  // pad bottom
                     HALO_ZERO2,
             },
             ""},
            {{layer_W(5, 3, k, 0, 1, 1), 2},
             {
                     {5, 3},  // in
                     {2, 1},  // out
                     {k, k},  // k
                     {0, 0},  // pad top
                     {0, 1},  // pad bottom
                     HALO_ZERO2,
             },
             ""},
            /// 5 inputs  , p2 combinations
            {{layer_W(5, 6, k, 2, 2, 1), 2},
             {
                     {4, 4},  // in
                     {3, 3},  // out
                     {k, k},  // k
                     {2, 0},  // pad top
                     {0, 2},  // pad bottom
                     HALO_ZERO2,
             },
             "P+"},
            {{layer_W(5, 4, k, 2, 0, 1), 2},
             {
                     {3, 5},  // in
                     {2, 2},  // out
                     {k, k},  // k
                     {2, 0},  // pad top
                     {0, 0},  // pad bottom
                     HALO_ZERO2,
             },
             ""},
            {{layer_W(5, 4, k, 0, 2, 1), 2},
             {
                     {5, 3},  // in
                     {2, 2},  // out
                     {k, k},  // k
                     {0, 0},  // pad top
                     {0, 2},  // pad bottom
                     HALO_ZERO2,
             },
             ""},
            {{layer_W(5, 5, k, 2, 1, 1), 2},
             {
                     {4, 4},  // in
                     {3, 2},  // out
                     {k, k},  // k
                     {2, 0},  // pad top
                     {0, 1},  // pad bottom
                     HALO_ZERO2,
             },
             ""},
            {{layer_W(5, 5, k, 1, 2, 1), 2},
             {
                     {5, 3},  // in
                     {3, 2},  // out
                     {k, k},  // k
                     {1, 0},  // pad top
                     {0, 2},  // pad bottom
                     HALO_ZERO2,
             },
             ""},
    };

    executeT_SOWOVR(tests);
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
                     HALO_ZERO2,
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
                     HALO_ZERO2,
             },
             ""},
            {{layer_H(5, 2, k, 0, 1, 1), 2},
             {
                     {5, 4},  // in
                     {1, 1},  // out
                     {k, k},  // k
                     {0, 0},  // pad top
                     {0, 1},  // pad bottom
                     HALO_ZERO2,
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
                     HALO_ZERO2,
             },
             "P+"},
            {{layer_H(5, 3, k, 2, 0, 1), 2},
             {
                     {4, 5},  // in
                     {2, 1},  // out
                     {k, k},  // k
                     {2, 0},  // pad top
                     {0, 0},  // pad bottom
                     HALO_ZERO2,
             },
             ""},
            {{layer_H(5, 3, k, 0, 2, 1), 2},
             {
                     {5, 3},  // in
                     {2, 1},  // out
                     {k, k},  // k
                     {0, 0},  // pad top
                     {1, 2},  // pad bottom
                     HALO_ZERO2,
             },
             " GROWS MORE THAN INput!!"},
            {{layer_H(5, 4, k, 2, 1, 1), 2},
             {
                     {4, 5},  // in
                     {2, 2},  // out
                     {k, k},  // k
                     {2, 0},  // pad top
                     {0, 1},  // pad bottom
                     HALO_ZERO2,
             },
             ""},
            {{layer_H(5, 4, k, 1, 2, 1), 2},
             {
                     {5, 4},  // in
                     {2, 2},  // out
                     {k, k},  // k
                     {1, 0},  // pad top
                     {0, 2},  // pad bottom
                     HALO_ZERO2,
             },
             ""},
    };

    executeT_SOHOVR(tests);
}

TEST_F(DPULayerTest, SplitSOWOVERLAPPED_basic_K5_2T) {
    // IN ,O, K , PT, PB, S
    int k = 5;
    const std::vector<TestCase> tests{
            {{layer_W(5, 3, k, 1, 1, 1), 2},
             {
                     {5, 4},  // in
                     {2, 1},  // out
                     {k, k},  // k
                     {1, 0},  // pad top
                     {0, 1},  // pad bottom
                     HALO_ZERO2,
             },
             ""},
            {{layer_W(5, 2, k, 1, 0, 1), 2},
             {
                     {4, 5},  // in
                     {1, 1},  // out
                     {k, k},  // k
                     {1, 0},  // pad top
                     {0, 0},  // pad bottom
                     HALO_ZERO2,
             },
             ""},
            {{layer_W(5, 2, k, 0, 1, 1), 2},
             {
                     {5, 4},  // in
                     {1, 1},  // out
                     {k, k},  // k
                     {0, 0},  // pad top
                     {0, 1},  // pad bottom
                     HALO_ZERO2,
             },
             ""},
            /// 5 inputs  , p2 combinations
            {{layer_W(5, 5, k, 2, 2, 1), 2},
             {
                     {5, 4},  // in
                     {3, 2},  // out
                     {k, k},  // k
                     {2, 0},  // pad top
                     {0, 2},  // pad bottom
                     HALO_ZERO2,
             },
             "P+"},
            {{layer_W(5, 3, k, 2, 0, 1), 2},
             {
                     {4, 5},  // in
                     {2, 1},  // out
                     {k, k},  // k
                     {2, 0},  // pad top
                     {0, 0},  // pad bottom
                     HALO_ZERO2,
             },
             ""},
            {{layer_W(5, 3, k, 0, 2, 1), 2},
             {
                     {5, 3},  // in
                     {2, 1},  // out
                     {k, k},  // k
                     {0, 0},  // pad top
                     {1, 2},  // pad bottom
                     HALO_ZERO2,
             },
             " GROWS MORE THAN INput!!"},
            {{layer_W(5, 4, k, 2, 1, 1), 2},
             {
                     {4, 5},  // in
                     {2, 2},  // out
                     {k, k},  // k
                     {2, 0},  // pad top
                     {0, 1},  // pad bottom
                     HALO_ZERO2,
             },
             ""},
            {{layer_W(5, 4, k, 1, 2, 1), 2},
             {
                     {5, 4},  // in
                     {2, 2},  // out
                     {k, k},  // k
                     {1, 0},  // pad top
                     {0, 2},  // pad bottom
                     HALO_ZERO2,
             },
             ""},
    };

    executeT_SOWOVR(tests);
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
                     HALO_ZERO2,
             },
             "Both Tile takes all, except pad "},

    };

    executeT_SOHOVR(tests);
}

TEST_F(DPULayerTest, SplitSOWOVERLAPPED_basic_K7_2T_CornerC) {
    // IN ,O, K , PT, PB, S
    int k = 7;
    const std::vector<TestCase> tests{
            {{layer_W(2, 2, k, 3, 3, 1), 2},
             {
                     {2, 2},  // in
                     {1, 1},  // out
                     {k, k},  // k
                     {3, 2},  // pad top
                     {2, 3},  // pad bottom
                     HALO_ZERO2,
             },
             "Both Tile takes all, except pad "},

    };

    executeT_SOWOVR(tests);
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
                     HALO_ZERO3,
             },
             ""},
            {{layer_H(15, 13, k, 0, 0, 1), T},
             {
                     {7, 7, 5},  // in
                     {5, 5, 3},  // out
                     {k, k, k},  // k
                     {0, 0, 0},  // pad top
                     {0, 0, 0},  // pad bottom
                     HALO_ZERO3,
             },
             ""},
            {{layer_H(15, 14, k, 1, 0, 1), T},
             {
                     {6, 7, 6},  // in
                     {5, 5, 4},  // out
                     {k, k, k},  // k
                     {1, 0, 0},  // pad top
                     {0, 0, 0},  // pad bottom
                     HALO_ZERO3,
             },
             ""},
            {{layer_H(15, 14, k, 0, 1, 1), T},
             {
                     {7, 7, 5},  // in
                     {5, 5, 4},  // out
                     {k, k, k},  // k
                     {0, 0, 0},  // pad top
                     {0, 0, 1},  // pad bottom
                     HALO_ZERO3,
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
                     HALO_ZERO3,
             },
             "Goes beyond"},
            {{layer_H(16, 14, k, 0, 0, 1), T},
             {
                     {7, 7, 6},  // in
                     {5, 5, 4},  // out
                     {k, k, k},  // k
                     {0, 0, 0},  // pad top
                     {0, 0, 0},  // pad bottom
                     HALO_ZERO3,
             },
             ""},
            {{layer_H(16, 15, k, 1, 0, 1), T},
             {
                     {6, 7, 7},  // in
                     {5, 5, 5},  // out
                     {k, k, k},  // k
                     {1, 0, 0},  // pad top
                     {0, 0, 0},  // pad bottom
                     HALO_ZERO3,
             },
             ""},
            {{layer_H(16, 15, k, 0, 1, 1), T},
             {
                     {7, 7, 6},  // in
                     {5, 5, 5},  // out
                     {k, k, k},  // k
                     {0, 0, 0},  // pad top
                     {0, 0, 1},  // pad bottom
                     HALO_ZERO3,
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
                     HALO_ZERO3,
             },
             "Goes beyond"},
            {{layer_H(17, 15, k, 0, 0, 1), T},
             {
                     {7, 7, 7},  // in
                     {5, 5, 5},  // out
                     {k, k, k},  // k
                     {0, 0, 0},  // pad top
                     {0, 0, 0},  // pad bottom
                     HALO_ZERO3,
             },
             ""},
            {{layer_H(17, 16, k, 1, 0, 1), T},
             {
                     {7, 8, 6},  // in
                     {6, 6, 4},  // out
                     {k, k, k},  // k
                     {1, 0, 0},  // pad top
                     {0, 0, 0},  // pad bottom
                     HALO_ZERO3,
             },
             ""},
            {{layer_H(17, 16, k, 0, 1, 1), T},
             {
                     {8, 8, 5},  // in
                     {6, 6, 4},  // out
                     {k, k, k},  // k
                     {0, 0, 0},  // pad top
                     {0, 0, 1},  // pad bottom
                     HALO_ZERO3,
             },
             ""},
    };

    executeT_SOHOVR(tests);
}

TEST_F(DPULayerTest, SplitSOWOVERLAPPED_basic_K3_3T) {
    // IN ,O, K , PT, PB, S
    const int T = 3;
    const int k = 3;
    const std::vector<TestCase> tests{
            // div by T
            {{layer_W(15, 15, k, 1, 1, 1), T},
             {
                     {6, 7, 6},  // in
                     {5, 5, 5},  // out
                     {k, k, k},  // k
                     {1, 0, 0},  // pad top
                     {0, 0, 1},  // pad bottom
                     HALO_ZERO3,
             },
             ""},
            {{layer_W(15, 13, k, 0, 0, 1), T},
             {
                     {7, 7, 5},  // in
                     {5, 5, 3},  // out
                     {k, k, k},  // k
                     {0, 0, 0},  // pad top
                     {0, 0, 0},  // pad bottom
                     HALO_ZERO3,
             },
             ""},
            {{layer_W(15, 14, k, 1, 0, 1), T},
             {
                     {6, 7, 6},  // in
                     {5, 5, 4},  // out
                     {k, k, k},  // k
                     {1, 0, 0},  // pad top
                     {0, 0, 0},  // pad bottom
                     HALO_ZERO3,
             },
             ""},
            {{layer_W(15, 14, k, 0, 1, 1), T},
             {
                     {7, 7, 5},  // in
                     {5, 5, 4},  // out
                     {k, k, k},  // k
                     {0, 0, 0},  // pad top
                     {0, 0, 1},  // pad bottom
                     HALO_ZERO3,
             },
             ""},
            /// 16 inputs (mod+1)
            {{layer_W(16, 16, k, 1, 1, 1), T},
             {
                     {7, 8, 5},  // in
                     {6, 6, 4},  // out
                     {k, k, k},  // k
                     {1, 0, 0},  // pad top
                     {0, 0, 1},  // pad bottom
                     HALO_ZERO3,
             },
             "Goes beyond"},
            {{layer_W(16, 14, k, 0, 0, 1), T},
             {
                     {7, 7, 6},  // in
                     {5, 5, 4},  // out
                     {k, k, k},  // k
                     {0, 0, 0},  // pad top
                     {0, 0, 0},  // pad bottom
                     HALO_ZERO3,
             },
             ""},
            {{layer_W(16, 15, k, 1, 0, 1), T},
             {
                     {6, 7, 7},  // in
                     {5, 5, 5},  // out
                     {k, k, k},  // k
                     {1, 0, 0},  // pad top
                     {0, 0, 0},  // pad bottom
                     HALO_ZERO3,
             },
             ""},
            {{layer_W(16, 15, k, 0, 1, 1), T},
             {
                     {7, 7, 6},  // in
                     {5, 5, 5},  // out
                     {k, k, k},  // k
                     {0, 0, 0},  // pad top
                     {0, 0, 1},  // pad bottom
                     HALO_ZERO3,
             },
             ""},

            /// 17 inputs (mod+2)
            {{layer_W(17, 17, k, 1, 1, 1), T},
             {
                     {7, 8, 6},  // in
                     {6, 6, 5},  // out
                     {k, k, k},  // k
                     {1, 0, 0},  // pad top
                     {0, 0, 1},  // pad bottom
                     HALO_ZERO3,
             },
             "Goes beyond"},
            {{layer_W(17, 15, k, 0, 0, 1), T},
             {
                     {7, 7, 7},  // in
                     {5, 5, 5},  // out
                     {k, k, k},  // k
                     {0, 0, 0},  // pad top
                     {0, 0, 0},  // pad bottom
                     HALO_ZERO3,
             },
             ""},
            {{layer_W(17, 16, k, 1, 0, 1), T},
             {
                     {7, 8, 6},  // in
                     {6, 6, 4},  // out
                     {k, k, k},  // k
                     {1, 0, 0},  // pad top
                     {0, 0, 0},  // pad bottom
                     HALO_ZERO3,
             },
             ""},
            {{layer_W(17, 16, k, 0, 1, 1), T},
             {
                     {8, 8, 5},  // in
                     {6, 6, 4},  // out
                     {k, k, k},  // k
                     {0, 0, 0},  // pad top
                     {0, 0, 1},  // pad bottom
                     HALO_ZERO3,
             },
             ""},
    };

    executeT_SOWOVR(tests);
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
                     HALO_ZERO3,
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
                     HALO_ZERO3,
             },
             "Second is at end already"},
            {{layer_H(5, 3, k, 2, 0, 1), 3},
             {
                     {3, 4, 5},  // in
                     {1, 1, 1},  // out
                     {k, k, k},  // k
                     {2, 1, 0},  // pad top
                     {0, 0, 0},  // pad bottom
                     HALO_ZERO3,
             },
             "second still uses pad begin layer"},
            {{layer_H(5, 3, k, 0, 2, 1), 3},
             {
                     {5, 4, 3},  // in
                     {1, 1, 1},  // out
                     {k, k, k},  // k
                     {0, 0, 0},  // pad top
                     {0, 1, 2},  // pad bottom
                     HALO_ZERO3,
             },
             " "},
            {{layer_H(5, 4, k, 2, 1, 1), 3},
             {
                     {4, 5},  // in
                     {2, 2},  // out
                     {k, k},  // k
                     {2, 0},  // pad top
                     {0, 1},  // pad bottom
                     HALO_ZERO3,
             },
             "ONly 2 splits a."},
            {{layer_H(5, 4, k, 1, 2, 1), 3},
             {
                     {5, 4},  // in
                     {2, 2},  // out
                     {k, k},  // k
                     {1, 0},  // pad top
                     {0, 2},  // pad bottom
                     HALO_ZERO3,
             },
             "only 2 splits b."},
    };

    executeT_SOHOVR(tests);
}

TEST_F(DPULayerTest, SplitSOWOVERLAPPED_basic_K5_3T) {
    // IN ,O, K , PT, PB, S
    int k = 5;
    const std::vector<TestCase> tests{
            {{layer_W(5, 3, k, 1, 1, 1), 3},
             {
                     {4, 5, 4},  // in
                     {1, 1, 1},  // out
                     {k, k, k},  // k
                     {1, 0, 0},  // pad top
                     {0, 0, 1},  // pad bottom
                     HALO_ZERO3,
             },
             ""},
            /// 5 inputs  , p2 combinations
            {{layer_W(5, 5, k, 2, 2, 1), 3},
             {
                     {4, 5, 3},  // in
                     {2, 2, 1},  // out
                     {k, k, k},  // k
                     {2, 0, 0},  // pad top
                     {0, 1, 2},  // pad bottom
                     HALO_ZERO3,
             },
             "Second is at end already"},
            {{layer_W(5, 3, k, 2, 0, 1), 3},
             {
                     {3, 4, 5},  // in
                     {1, 1, 1},  // out
                     {k, k, k},  // k
                     {2, 1, 0},  // pad top
                     {0, 0, 0},  // pad bottom
                     HALO_ZERO3,
             },
             "second still uses pad begin layer"},
            {{layer_W(5, 3, k, 0, 2, 1), 3},
             {
                     {5, 4, 3},  // in
                     {1, 1, 1},  // out
                     {k, k, k},  // k
                     {0, 0, 0},  // pad top
                     {0, 1, 2},  // pad bottom
                     HALO_ZERO3,
             },
             " "},
            {{layer_W(5, 4, k, 2, 1, 1), 3},
             {
                     {4, 5},  // in
                     {2, 2},  // out
                     {k, k},  // k
                     {2, 0},  // pad top
                     {0, 1},  // pad bottom
                     HALO_ZERO3,
             },
             "ONly 2 splits a."},
            {{layer_W(5, 4, k, 1, 2, 1), 3},
             {
                     {5, 4},  // in
                     {2, 2},  // out
                     {k, k},  // k
                     {1, 0},  // pad top
                     {0, 2},  // pad bottom
                     HALO_ZERO3,
             },
             "only 2 splits b."},
    };

    executeT_SOWOVR(tests);
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
                     HALO_ZERO2,
             },
             ""},
            {{layer_H(5, 2, 3, 0, 0, 2), 2},
             {
                     {3, 3},  // in
                     {1, 1},  // out
                     {k, k},  // k
                     {0, 0},  // pad top
                     {0, 0},  // pad bottom
                     HALO_ZERO2,
             },
             ""},
            {{layer_H(5, 2, 3, 1, 0, 2), 2},
             {
                     {2, 3},  // in
                     {1, 1},  // out
                     {k, k},  // k
                     {1, 0},  // pad top
                     {0, 0},  // pad bottom
                     HALO_ZERO2,
             },
             ""},
            {{layer_H(5, 2, 3, 0, 1, 2), 2},
             {
                     {3, 3},  // in
                     {1, 1},  // out
                     {k, k},  // k
                     {0, 0},  // pad top
                     {0, 1},  // pad bottom
                     HALO_ZERO2,
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
                     HALO_ZERO2,
             },
             "extra pad! "},
            {{layer_H(6, 2, 3, 0, 0, 2), 2},
             {
                     {3, 3},  // in
                     {1, 1},  // out
                     {k, k},  // k
                     {0, 0},  // pad top
                     {0, 0},  // pad bottom
                     HALO_ZERO2,
             },
             "MOre needs"},
            {{layer_H(6, 3, 3, 1, 0, 2), 2},
             {
                     {4, 3},  // in
                     {2, 1},  // out
                     {k, k},  // k
                     {1, 0},  // pad top
                     {0, 0},  // pad bottom
                     HALO_ZERO2,
             },
             ""},
            {{layer_H(6, 3, 3, 0, 1, 2), 2},
             {
                     {5, 2},  // in
                     {2, 1},  // out
                     {k, k},  // k
                     {0, 0},  // pad top
                     {0, 1},  // pad bottom
                     HALO_ZERO2,
             },
             ""},
    };

    // auto split = tst_layer_ref.splitAcrossTiles(VPUNN::VPUTilingStrategy::SOH, 2);
    // auto split = tst_layer_ref.SOH_overlapped_inputs(2);
    // auto split = tst_layer_ref.SOH(2);

    executeT_SOHOVR(tests);
}

TEST_F(DPULayerTest, SplitSOWOVERLAPPED_S2_K3_2T) {
    // IN ,O, K , PT, PB, S
    const int k = 3;
    const std::vector<TestCase> tests{
            {{layer_W(5, 3, k, 1, 1, 2), 2},
             {
                     {4, 2},  // in
                     {2, 1},  // out
                     {k, k},  // k
                     {1, 0},  // pad top
                     {0, 1},  // pad bottom
                     HALO_ZERO2,
             },
             ""},
            {{layer_W(5, 2, 3, 0, 0, 2), 2},
             {
                     {3, 3},  // in
                     {1, 1},  // out
                     {k, k},  // k
                     {0, 0},  // pad top
                     {0, 0},  // pad bottom
                     HALO_ZERO2,
             },
             ""},
            {{layer_W(5, 2, 3, 1, 0, 2), 2},
             {
                     {2, 3},  // in
                     {1, 1},  // out
                     {k, k},  // k
                     {1, 0},  // pad top
                     {0, 0},  // pad bottom
                     HALO_ZERO2,
             },
             ""},
            {{layer_W(5, 2, 3, 0, 1, 2), 2},
             {
                     {3, 3},  // in
                     {1, 1},  // out
                     {k, k},  // k
                     {0, 0},  // pad top
                     {0, 1},  // pad bottom
                     HALO_ZERO2,
             },
             "extra padding is bad! "},
            ///// 6 inputs (even)
            {{layer_W(6, 3, 3, 1, 1, 2), 2},
             {
                     {4, 3},  // in
                     {2, 1},  // out
                     {k, k},  // k
                     {1, 0},  // pad top
                     {0, 1},  // pad bottom
                     HALO_ZERO2,
             },
             "extra pad! "},
            {{layer_W(6, 2, 3, 0, 0, 2), 2},
             {
                     {3, 3},  // in
                     {1, 1},  // out
                     {k, k},  // k
                     {0, 0},  // pad top
                     {0, 0},  // pad bottom
                     HALO_ZERO2,
             },
             "MOre needs"},
            {{layer_W(6, 3, 3, 1, 0, 2), 2},
             {
                     {4, 3},  // in
                     {2, 1},  // out
                     {k, k},  // k
                     {1, 0},  // pad top
                     {0, 0},  // pad bottom
                     HALO_ZERO2,
             },
             ""},
            {{layer_W(6, 3, 3, 0, 1, 2), 2},
             {
                     {5, 2},  // in
                     {2, 1},  // out
                     {k, k},  // k
                     {0, 0},  // pad top
                     {0, 1},  // pad bottom
                     HALO_ZERO2,
             },
             ""},
    };
    executeT_SOWOVR(tests);
}

TEST_F(DPULayerTest, SplitSOHOVERLAPPED_S1_K3_2T_EISXW_76882) {
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
                     HALO_ZERO2,
             },
             ""},

    };

    executeT_SOHOVR(tests);
}

//////// multi splits tests

TEST_F(DPULayerTest, Split_SOH_3X_S1_K9_2T_redowa_deeplab_v3_dense_IRv11_FP16) {
    // IN ,O, K , PT, PB, S
    const int k = 9;

    const std::vector<TestCase> tests_SOHO{

            {{layer_H(16, 8, k, 0, 0, 1), 2},
             {
                     {12, 12},      // in
                     {4, 4},        // out
                     {k, k},        // k
                     {0, 0},        // pad top
                     {0, 0},        // pad bottom
                     {isiC, isiC},  // ISI
                     HALO_ZERO2,
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
                     HALO_ZERO2,    // old obsolete
             },
             " redowa_deeplab_v3_dense_IRv11"},
    };

    { executeT(SOHO_test, tests_SOHO, " SOH OVERLAPPED : "); }

    {
        std::function<std::vector<DPULayer>(const DPULayer*, unsigned int)> split_H_OLD =
                std::bind(&DPULayer::SOH_deprecated, _1, _2);
        std::function<void(const TestInput&, const TestExpectations&, const std::string&)> f =
                std::bind(&DPULayerTest::TestSoh, this, split_H_OLD, _1, _2, _3);

        executeT(f, tests_SOH, " SOH old :");
    }

    const std::vector<TestCase> tests_SOH_inHALO{

            {{layer_H(16, 8, k, 0, 0, 1), 2},
             {
                     {12, 12},      // in
                     {4, 4},        // out
                     {k, k},        // k
                     {0, 0},        // pad top
                     {0, 0},        // pad bottom
                     {isiH, isiH},  // ISI
                     {0, 4} /*in halo top*/,
                     {4, 0} /*in halo bottom*/
             },
             " redowa_deeplab_v3_dense_IRv11"},
    };
    { executeT(SOH_inHALO_test, tests_SOH_inHALO, " SOH input HALO : "); }
}

TEST_F(DPULayerTest, Split_SOH_3X_S8_K8_2T_DW_Cnv) {
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
                     HALO_ZERO2,
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
                     {16, 16},      // in
                     {2, 2},        // out
                     {k, k},        // k
                     {0, 0},        // pad top
                     {0, 0},        // pad bottom
                     {isiH, isiH},  // ISI
                     HALO_ZERO2,
             },
             " 4 outputs H32"},
    };

    { executeT(SOHO_test, tests_SOHO, " SOH OVERLAPPED : "); }

    {
        std::function<std::vector<DPULayer>(const DPULayer*, unsigned int)> split_H_OLD =
                std::bind(&DPULayer::SOH_deprecated, _1, _2);
        std::function<void(const TestInput&, const TestExpectations&, const std::string&)> f =
                std::bind(&DPULayerTest::TestSoh, this, split_H_OLD, _1, _2, _3);

        executeT(f, tests_SOH, " SOH old :");
    }

    const std::vector<TestCase> tests_SOH_inHALO{
            {{layer_H(8, 1, k, 0, 0, 8), 2},
             {
                     {8},     // in
                     {1},     // out
                     {k},     // k
                     {0},     // pad top
                     {0},     // pad bottom
                     {isiH},  // ISI
                     {0} /*in halo top*/,
                     {0} /*in halo bottom*/
             },
             " One output H8"},
            {{layer_H(32, 4, k, 0, 0, 8), 2},
             {
                     {16, 16},      // in
                     {2, 2},        // out
                     {k, k},        // k
                     {0, 0},        // pad top
                     {0, 0},        // pad bottom
                     {isiH, isiH},  // ISI
                     {0, 0} /*in halo top*/,
                     {0, 0} /*in halo bottom*/
             },
             " 4 outputs H32"},
    };
    { executeT(SOH_inHALO_test, tests_SOH_inHALO, " SOH input HALO : "); }
}

///////// SOH input HALO tests

TEST_F(DPULayerTest, SplitSOHinHALO_basic_K3_2T) {
    // IN ,O, K , PT, PB, S
    const int k = 3;
    const std::vector<TestCase> tests{
            {{layer_H(5, 5, k, 1, 1, 1), 2},
             {
                     {4, 3},        // in
                     {3, 2},        // out
                     {k, k},        // k
                     {1, 0},        // pad top
                     {0, 1},        // pad bottom
                     {isiH, isiH},  // ISI
                     {0, 1} /*in halo top*/,
                     {1, 0} /*in halo bottom*/

             },
             "5-5 k3 p11"},
            {{layer_H(5, 3, 3, 0, 0, 1), 2},
             {
                     {4, 3},        // in
                     {2, 1},        // out
                     {k, k},        // k
                     {0, 0},        // pad top
                     {0, 0},        // pad bottom
                     {isiH, isiH},  // ISI
                     {0, 1} /*in halo top*/,
                     {1, 0} /*in halo bottom*/
             },
             "5-3 k3 p00"},
            {{layer_H(5, 4, 3, 1, 0, 1), 2},
             {
                     {3, 4},        // in
                     {2, 2},        // out
                     {k, k},        // k
                     {1, 0},        // pad top
                     {0, 0},        // pad bottom
                     {isiH, isiH},  // ISI
                     {0, 1} /*in halo top*/,
                     {1, 0} /*in halo bottom*/
             },
             "5-4 k3 p10"},
            {{layer_H(5, 4, 3, 0, 1, 1), 2},
             {
                     {4, 3},        // in
                     {2, 2},        // out
                     {k, k},        // k
                     {0, 0},        // pad top
                     {0, 1},        // pad bottom
                     {isiH, isiH},  // ISI
                     {0, 1} /*in halo top*/,
                     {1, 0} /*in halo bottom*/
             },
             "5-4 k3 p01"},
            /// 6 inputs (even)
            {{layer_H(6, 6, 3, 1, 1, 1), 2},
             {
                     {4, 4},        // in
                     {3, 3},        // out
                     {k, k},        // k
                     {1, 0},        // pad top
                     {0, 1},        // pad bottom
                     {isiH, isiH},  // ISI
                     {0, 1} /*in halo top*/,
                     {1, 0} /*in halo bottom*/
             },
             "6-6 k3 p11"},
            {{layer_H(6, 4, 3, 0, 0, 1), 2},
             {
                     {4, 4},        // in
                     {2, 2},        // out
                     {k, k},        // k
                     {0, 0},        // pad top
                     {0, 0},        // pad bottom
                     {isiH, isiH},  // ISI
                     {0, 1} /*in halo top*/,
                     {1, 0} /*in halo bottom*/
             },
             "6-4 k3 p00"},
            {{layer_H(6, 5, 3, 1, 0, 1), 2},
             {
                     {4, 4},        // in
                     {3, 2},        // out
                     {k, k},        // k
                     {1, 0},        // pad top
                     {0, 0},        // pad bottom
                     {isiH, isiH},  // ISI
                     {0, 1} /*in halo top*/,
                     {1, 0} /*in halo bottom*/
             },
             "6-5 k3 p10"},
            {{layer_H(6, 5, 3, 0, 1, 1), 2},
             {
                     {5, 3},        // in
                     {3, 2},        // out
                     {k, k},        // k
                     {0, 0},        // pad top
                     {0, 1},        // pad bottom
                     {isiH, isiH},  // ISI
                     {0, 1} /*in halo top*/,
                     {1, 0} /*in halo bottom*/
             },
             "6-5 k3 p01"},
    };

    executeT_SOH_inHALO(tests);
}

TEST_F(DPULayerTest, SplitSOHinHALO_basic_K2_2T) {
    // IN ,O, K , PT, PB, S
    const int k = 2;
    const std::vector<TestCase> tests{
            {{layer_H(5, 6, k, 1, 1, 1), 2},
             {
                     {3, 3},        // in
                     {3, 3},        // out
                     {k, k},        // k
                     {1, 0},        // pad top
                     {0, 1},        // pad bottom
                     {isiH, isiH},  // ISI
                     {0, 1} /*in halo top*/,
                     {0, 0} /*in halo bottom*/
             },
             ""},
            {{layer_H(5, 4, k, 0, 0, 1), 2},
             {
                     {3, 3},        // in
                     {2, 2},        // out
                     {k, k},        // k
                     {0, 0},        // pad top
                     {0, 0},        // pad bottom
                     {isiH, isiH},  // ISI
                     {0, 1} /*in halo top*/,
                     {0, 0} /*in halo bottom*/
             },
             ""},
            {{layer_H(5, 5, k, 1, 0, 1), 2},
             {
                     {3, 3},        // in
                     {3, 2},        // out
                     {k, k},        // k
                     {1, 0},        // pad top
                     {0, 0},        // pad bottom
                     {isiH, isiH},  // ISI
                     {0, 1} /*in halo top*/,
                     {0, 0} /*in halo bottom*/
             },
             ""},
            {{layer_H(5, 5, k, 0, 1, 1), 2},
             {
                     {4, 2},        // in
                     {3, 2},        // out
                     {k, k},        // k
                     {0, 0},        // pad top
                     {0, 1},        // pad bottom
                     {isiH, isiH},  // ISI
                     {0, 1} /*in halo top*/,
                     {0, 0} /*in halo bottom*/
             },
             ""},
            /// 6 inputs (even)
            {{layer_H(6, 7, k, 1, 1, 1), 2},
             {
                     {4, 3},        // in
                     {4, 3},        // out
                     {k, k},        // k
                     {1, 0},        // pad top
                     {0, 1},        // pad bottom
                     {isiH, isiH},  // ISI
                     {0, 1} /*in halo top*/,
                     {0, 0} /*in halo bottom*/
             },
             ""},
            {{layer_H(6, 5, k, 0, 0, 1), 2},
             {
                     {4, 3},        // in
                     {3, 2},        // out
                     {k, k},        // k
                     {0, 0},        // pad top
                     {0, 0},        // pad bottom
                     {isiH, isiH},  // ISI
                     {0, 1} /*in halo top*/,
                     {0, 0} /*in halo bottom*/
             },
             ""},
            {{layer_H(6, 6, k, 1, 0, 1), 2},
             {
                     {3, 4},        // in
                     {3, 3},        // out
                     {k, k},        // k
                     {1, 0},        // pad top
                     {0, 0},        // pad bottom
                     {isiH, isiH},  // ISI
                     {0, 1} /*in halo top*/,
                     {0, 0} /*in halo bottom*/
             },
             ""},
            {{layer_H(6, 6, k, 0, 1, 1), 2},
             {
                     {4, 3},        // in
                     {3, 3},        // out
                     {k, k},        // k
                     {0, 0},        // pad top
                     {0, 1},        // pad bottom
                     {isiH, isiH},  // ISI
                     {0, 1} /*in halo top*/,
                     {0, 0} /*in halo bottom*/
             },
             ""},
    };

    executeT_SOH_inHALO(tests);
}

TEST_F(DPULayerTest, SplitSOHinHALO_basic_K1_2T) {
    // IN ,O, K , PT, PB, S
    const int k = 1;
    const std::vector<TestCase> tests{
            {{layer_H(5, 7, k, 1, 1, 1), 2},
             {
                     {3, 2},        // in
                     {4, 3},        // out
                     {k, k},        // k
                     {1, 0},        // pad top
                     {0, 1},        // pad bottom
                     {isiH, isiH},  // ISI
                     {0, 0} /*in halo top*/,
                     {0, 0} /*in halo bottom*/
             },
             ""},
            {{layer_H(5, 5, k, 0, 0, 1), 2},
             {
                     {3, 2},        // in
                     {3, 2},        // out
                     {k, k},        // k
                     {0, 0},        // pad top
                     {0, 0},        // pad bottom
                     {isiH, isiH},  // ISI
                     HALO_ZERO2,
             },
             ""},
            {{layer_H(5, 6, k, 1, 0, 1), 2},
             {
                     {2, 3},        // in
                     {3, 3},        // out
                     {k, k},        // k
                     {1, 0},        // pad top
                     {0, 0},        // pad bottom
                     {isiH, isiH},  // ISI
                     HALO_ZERO2,
             },
             ""},
            {{layer_H(5, 6, k, 0, 1, 1), 2},
             {
                     {3, 2},        // in
                     {3, 3},        // out
                     {k, k},        // k
                     {0, 0},        // pad top
                     {0, 1},        // pad bottom
                     {isiH, isiH},  // ISI
                     HALO_ZERO2,
             },
             ""},
            /// 6 inputs (even)
            {{layer_H(6, 8, k, 1, 1, 1), 2},
             {
                     {3, 3},        // in
                     {4, 4},        // out
                     {k, k},        // k
                     {1, 0},        // pad top
                     {0, 1},        // pad bottom
                     {isiH, isiH},  // ISI
                     HALO_ZERO2,
             },
             ""},
            {{layer_H(6, 6, k, 0, 0, 1), 2},
             {
                     {3, 3},        // in
                     {3, 3},        // out
                     {k, k},        // k
                     {0, 0},        // pad top
                     {0, 0},        // pad bottom
                     {isiH, isiH},  // ISI
                     HALO_ZERO2,
             },
             ""},
            {{layer_H(6, 7, k, 1, 0, 1), 2},
             {
                     {3, 3},        // in
                     {4, 3},        // out
                     {k, k},        // k
                     {1, 0},        // pad top
                     {0, 0},        // pad bottom
                     {isiH, isiH},  // ISI
                     HALO_ZERO2,
             },
             ""},
            {{layer_H(6, 7, k, 0, 1, 1), 2},
             {
                     {4, 2},        // in
                     {4, 3},        // out
                     {k, k},        // k
                     {0, 0},        // pad top
                     {0, 1},        // pad bottom
                     {isiH, isiH},  // ISI
                     HALO_ZERO2,
             },
             ""},
    };

    executeT_SOH_inHALO(tests);
}

TEST_F(DPULayerTest, SplitSOHinHALO_basic_K4_2T) {
    // IN ,O, K , PT, PB, S
    int k = 4;
    const std::vector<TestCase> tests{
            {{layer_H(5, 4, k, 1, 1, 1), 2},
             {
                     {4, 4},        // in
                     {2, 2},        // out
                     {k, k},        // k
                     {1, 0},        // pad top
                     {0, 1},        // pad bottom
                     {isiH, isiH},  // ISI
                     {0, 2} /*in halo top*/,
                     {1, 0} /*in halo bottom*/
             },
             ""},
            {{layer_H(5, 2, k, 0, 0, 1), 2},
             {
                     {4, 4},        // in
                     {1, 1},        // out
                     {k, k},        // k
                     {0, 0},        // pad top
                     {0, 0},        // pad bottom
                     {isiH, isiH},  // ISI
                     {0, 2} /*in halo top*/,
                     {1, 0} /*in halo bottom*/
             },
             ""},
            {{layer_H(5, 3, k, 1, 0, 1), 2},
             {
                     {4, 4},        // in
                     {2, 1},        // out
                     {k, k},        // k
                     {1, 0},        // pad top
                     {0, 0},        // pad bottom
                     {isiH, isiH},  // ISI
                     {0, 2} /*in halo top*/,
                     {1, 0} /*in halo bottom*/
             },
             ""},
            {{layer_H(5, 3, k, 0, 1, 1), 2},
             {
                     {5, 3},        // in
                     {2, 1},        // out
                     {k, k},        // k
                     {0, 0},        // pad top
                     {0, 1},        // pad bottom
                     {isiH, isiH},  // ISI
                     {0, 2} /*in halo top*/,
                     {1, 0} /*in halo bottom*/
             },
             ""},
            /// 5 inputs  , p2 combinations
            {{layer_H(5, 6, k, 2, 2, 1), 2},
             {
                     {4, 4},        // in
                     {3, 3},        // out
                     {k, k},        // k
                     {2, 0},        // pad top
                     {0, 2},        // pad bottom
                     {isiH, isiH},  // ISI
                     {0, 2} /*in halo top*/,
                     {1, 0} /*in halo bottom*/
             },
             "P+"},
            {{layer_H(5, 4, k, 2, 0, 1), 2},
             {
                     {3, 5},        // in
                     {2, 2},        // out
                     {k, k},        // k
                     {2, 0},        // pad top
                     {0, 0},        // pad bottom
                     {isiH, isiH},  // ISI
                     {0, 2} /*in halo top*/,
                     {1, 0} /*in halo bottom*/
             },
             ""},
            {{layer_H(5, 4, k, 0, 2, 1), 2},
             {
                     {5, 3},        // in
                     {2, 2},        // out
                     {k, k},        // k
                     {0, 0},        // pad top
                     {0, 2},        // pad bottom
                     {isiH, isiH},  // ISI
                     {0, 2} /*in halo top*/,
                     {1, 0} /*in halo bottom*/
             },
             ""},
            {{layer_H(5, 5, k, 2, 1, 1), 2},
             {
                     {4, 4},        // in
                     {3, 2},        // out
                     {k, k},        // k
                     {2, 0},        // pad top
                     {0, 1},        // pad bottom
                     {isiH, isiH},  // ISI
                     {0, 2} /*in halo top*/,
                     {1, 0} /*in halo bottom*/
             },
             ""},
            {{layer_H(5, 5, k, 1, 2, 1), 2},
             {
                     {5, 3},        // in
                     {3, 2},        // out
                     {k, k},        // k
                     {1, 0},        // pad top
                     {0, 2},        // pad bottom
                     {isiH, isiH},  // ISI
                     {0, 2} /*in halo top*/,
                     {1, 0} /*in halo bottom*/
             },
             ""},
    };

    executeT_SOH_inHALO(tests);
}

TEST_F(DPULayerTest, SplitSOHinHALO_basic_K5_2T) {
    // IN ,O, K , PT, PB, S
    int k = 5;
    const std::vector<TestCase> tests{
            {{layer_H(5, 3, k, 1, 1, 1), 2},
             {
                     {5, 4},        // in
                     {2, 1},        // out
                     {k, k},        // k
                     {1, 0},        // pad top
                     {0, 1},        // pad bottom
                     {isiH, isiH},  // ISI
                     {0, 2} /*in halo top*/,
                     {2, 0} /*in halo bottom*/
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
                     {4, 5},        // in
                     {1, 1},        // out
                     {k, k},        // k
                     {1, 0},        // pad top
                     {0, 0},        // pad bottom
                     {isiH, isiH},  // ISI
                     {0, 2} /*in halo top*/,
                     {2, 0} /*in halo bottom*/
             },
             ""},
            {{layer_H(5, 2, k, 0, 1, 1), 2},
             {
                     {5, 4},        // in
                     {1, 1},        // out
                     {k, k},        // k
                     {0, 0},        // pad top
                     {0, 1},        // pad bottom
                     {isiH, isiH},  // ISI
                     {0, 2} /*in halo top*/,
                     {2, 0} /*in halo bottom*/
             },
             ""},
            /// 5 inputs  , p2 combinations
            {{layer_H(5, 5, k, 2, 2, 1), 2},
             {
                     {5, 4},        // in
                     {3, 2},        // out
                     {k, k},        // k
                     {2, 0},        // pad top
                     {0, 2},        // pad bottom
                     {isiH, isiH},  // ISI
                     {0, 2} /*in halo top*/,
                     {2, 0} /*in halo bottom*/
             },
             "P+"},
            {{layer_H(5, 3, k, 2, 0, 1), 2},
             {
                     {4, 5},        // in
                     {2, 1},        // out
                     {k, k},        // k
                     {2, 0},        // pad top
                     {0, 0},        // pad bottom
                     {isiH, isiH},  // ISI
                     {0, 2} /*in halo top*/,
                     {2, 0} /*in halo bottom*/
             },
             ""},
            {{layer_H(5, 3, k, 0, 2, 1), 2},
             {
                     {5, 3},                    // in
                     {2, 1},                    // out
                     {k, k},                    // k
                     {0, 0},                    // pad top
                     {1, 2},                    // pad bottom
                     {isiH, isiH},              // ISI
                     {0, 3} /*in halo top*/,    // 3 because L1 takes all memory
                     {0, 0} /*in halo bottom*/  // zero T1 bot because of pad!=0
             },
             " GROWS MORE THAN INput!! L1 takes all memory"},
            {{layer_H(5, 4, k, 2, 1, 1), 2},
             {
                     {4, 5},        // in
                     {2, 2},        // out
                     {k, k},        // k
                     {2, 0},        // pad top
                     {0, 1},        // pad bottom
                     {isiH, isiH},  // ISI
                     {0, 2} /*in halo top*/,
                     {2, 0} /*in halo bottom*/
             },
             ""},
            {{layer_H(5, 4, k, 1, 2, 1), 2},
             {
                     {5, 4},        // in
                     {2, 2},        // out
                     {k, k},        // k
                     {1, 0},        // pad top
                     {0, 2},        // pad bottom
                     {isiH, isiH},  // ISI
                     {0, 2} /*in halo top*/,
                     {2, 0} /*in halo bottom*/
             },
             ""},
    };

    executeT_SOH_inHALO(tests);
}

TEST_F(DPULayerTest, SplitSOHinHALO_basic_K7_2T_CornerC_defect) {
    // IN ,O, K , PT, PB, S
    int k = 7;
    const std::vector<TestCase> tests{
            {{layer_H(2, 2, k, 3, 3, 1), 2},
             {
                     {2, 2},        // in
                     {1, 1},        // out
                     {k, k},        // k
                     {3, 2},        // pad top
                     {2, 3},        // pad bottom
                     {isiH, isiH},  // ISI
                     {0, 0} /*in halo top*/,
                     {0, 0} /*in halo bottom*/
             },
             "Both Tile takes all, except pad: DEFECT USE CASE?!UNSUPPORTED "},

    };

    executeT_SOH_inHALO(tests);
}

TEST_F(DPULayerTest, SplitSOHinHALO_S2_K3_2T) {
    // IN ,O, K , PT, PB, S
    const int k = 3;
    const std::vector<TestCase> tests{
            {{layer_H(5, 3, k, 1, 1, 2), 2},
             {
                     {4, 2},        // in
                     {2, 1},        // out
                     {k, k},        // k
                     {1, 0},        // pad top
                     {0, 1},        // pad bottom
                     {isiH, isiH},  // ISI
                     {0, 1} /*in halo top*/,
                     {0, 0} /*in halo bottom*/,  // 1 overlap, taken as memory
             },
             ""},
            {{layer_H(5, 2, 3, 0, 0, 2), 2},
             {
                     {3, 3},        // in
                     {1, 1},        // out
                     {k, k},        // k
                     {0, 0},        // pad top
                     {0, 0},        // pad bottom
                     {isiH, isiH},  // ISI
                     {0, 1} /*in halo top*/,
                     {0, 0} /*in halo bottom*/,  // 1 overlap, taken as memory
             },
             ""},
            {{layer_H(5, 2, 3, 1, 0, 2), 2},
             {
                     {2, 3},        // in
                     {1, 1},        // out
                     {k, k},        // k
                     {1, 0},        // pad top
                     {0, 0},        // pad bottom
                     {isiH, isiH},  // ISI
                     {0, 1} /*in halo top*/,
                     {0, 0} /*in halo bottom*/,  // 1 overlap, taken as memory
             },
             ""},
            {{layer_H(5, 2, 3, 0, 1, 2), 2},
             {
                     {3, 3},        // in
                     {1, 1},        // out
                     {k, k},        // k
                     {0, 0},        // pad top
                     {0, 1},        // pad bottom
                     {isiH, isiH},  // ISI
                     {0, 1} /*in halo top*/,
                     {0, 0} /*in halo bottom*/,  // 1 overlap, taken as memory
             },
             "extra padding is bad! "},
            ///// 6 inputs (even)
            {{layer_H(6, 3, 3, 1, 1, 2), 2},
             {
                     {4, 3},        // in
                     {2, 1},        // out
                     {k, k},        // k
                     {1, 0},        // pad top
                     {0, 1},        // pad bottom
                     {isiH, isiH},  // ISI
                     {0, 1} /*in halo top*/,
                     {0, 0} /*in halo bottom*/,  // 1 overlap, taken as memory
             },
             "extra pad! "},
            {{layer_H(6, 2, 3, 0, 0, 2), 2},
             {
                     {3, 3},        // in
                     {1, 1},        // out
                     {k, k},        // k
                     {0, 0},        // pad top
                     {0, 0},        // pad bottom
                     {isiH, isiH},  // ISI
                     {0, 1} /*in halo top*/,
                     {0, 0} /*in halo bottom*/,  // 1 overlap, taken as memory
             },
             "MOre needs"},
            {{layer_H(6, 3, 3, 1, 0, 2), 2},
             {
                     {4, 3},        // in
                     {2, 1},        // out
                     {k, k},        // k
                     {1, 0},        // pad top
                     {0, 0},        // pad bottom
                     {isiH, isiH},  // ISI
                     {0, 1} /*in halo top*/,
                     {0, 0} /*in halo bottom*/,  // 1 overlap, taken as memory
             },
             ""},
            {{layer_H(6, 3, 3, 0, 1, 2), 2},
             {
                     {5, 2},        // in
                     {2, 1},        // out
                     {k, k},        // k
                     {0, 0},        // pad top
                     {0, 1},        // pad bottom
                     {isiH, isiH},  // ISI
                     {0, 1} /*in halo top*/,
                     {0, 0} /*in halo bottom*/,  // 1 overlap, taken as memory
             },
             ""},
    };

    executeT_SOH_inHALO(tests);
}

TEST_F(DPULayerTest, SplitSOHinHALO_S1_K3_2T_EISXW_76882) {
    // IN ,O, K , PT, PB, S
    const int k = 3;
    const std::vector<TestCase> tests{
            {{layer_H(7, 6, k, 1, 0, 1), 2},
             {
                     {4, 5},        // in
                     {3, 3},        // out
                     {k, k},        // k
                     {1, 0},        // pad top
                     {0, 0},        // pad bottom
                     {isiH, isiH},  // ISI
                     {0, 1} /*in halo top*/,
                     {1, 0} /*in halo bottom*/
             },
             ""},
    };

    executeT_SOH_inHALO(tests);
}

TEST_F(DPULayerTest, SplitSOHinHALO_basic_K3_3T) {
    // IN ,O, K , PT, PB, S
    const int T = 3;
    const int k = 3;
    const std::vector<TestCase> tests{
            // div by T
            {{layer_H(15, 15, k, 1, 1, 1), T},
             {
                     {6, 7, 6},           // in
                     {5, 5, 5},           // out
                     {k, k, k},           // k
                     {1, 0, 0},           // pad top
                     {0, 0, 1},           // pad bottom
                     {isiH, isiH, isiH},  // ISI
                     {0, 1, 1} /*in halo top*/,
                     {1, 1, 0} /*in halo bottom*/

             },
             ""},
            {{layer_H(15, 13, k, 0, 0, 1), T},
             {
                     {7, 7, 5},           // in
                     {5, 5, 3},           // out
                     {k, k, k},           // k
                     {0, 0, 0},           // pad top
                     {0, 0, 0},           // pad bottom
                     {isiH, isiH, isiH},  // ISI
                     {0, 1, 1} /*in halo top*/,
                     {1, 1, 0} /*in halo bottom*/
             },
             ""},
            {{layer_H(15, 14, k, 1, 0, 1), T},
             {
                     {6, 7, 6},           // in
                     {5, 5, 4},           // out
                     {k, k, k},           // k
                     {1, 0, 0},           // pad top
                     {0, 0, 0},           // pad bottom
                     {isiH, isiH, isiH},  // ISI
                     {0, 1, 1} /*in halo top*/,
                     {1, 1, 0} /*in halo bottom*/
             },
             ""},
            {{layer_H(15, 14, k, 0, 1, 1), T},
             {
                     {7, 7, 5},           // in
                     {5, 5, 4},           // out
                     {k, k, k},           // k
                     {0, 0, 0},           // pad top
                     {0, 0, 1},           // pad bottom
                     {isiH, isiH, isiH},  // ISI
                     {0, 1, 1} /*in halo top*/,
                     {1, 1, 0} /*in halo bottom*/
             },
             ""},
            /// 16 inputs (mod+1)
            {{layer_H(16, 16, k, 1, 1, 1), T},
             {
                     {7, 8, 5},           // in
                     {6, 6, 4},           // out
                     {k, k, k},           // k
                     {1, 0, 0},           // pad top
                     {0, 0, 1},           // pad bottom
                     {isiH, isiH, isiH},  // ISI
                     {0, 1, 1} /*in halo top*/,
                     {1, 1, 0} /*in halo bottom*/
             },
             "Goes beyond"},
            {{layer_H(16, 14, k, 0, 0, 1), T},
             {
                     {7, 7, 6},           // in
                     {5, 5, 4},           // out
                     {k, k, k},           // k
                     {0, 0, 0},           // pad top
                     {0, 0, 0},           // pad bottom
                     {isiH, isiH, isiH},  // ISI
                     {0, 1, 1} /*in halo top*/,
                     {1, 1, 0} /*in halo bottom*/
             },
             ""},
            {{layer_H(16, 15, k, 1, 0, 1), T},
             {
                     {6, 7, 7},           // in
                     {5, 5, 5},           // out
                     {k, k, k},           // k
                     {1, 0, 0},           // pad top
                     {0, 0, 0},           // pad bottom
                     {isiH, isiH, isiH},  // ISI
                     {0, 1, 1} /*in halo top*/,
                     {1, 1, 0} /*in halo bottom*/
             },
             ""},
            {{layer_H(16, 15, k, 0, 1, 1), T},
             {
                     {7, 7, 6},           // in
                     {5, 5, 5},           // out
                     {k, k, k},           // k
                     {0, 0, 0},           // pad top
                     {0, 0, 1},           // pad bottom
                     {isiH, isiH, isiH},  // ISI
                     {0, 1, 1} /*in halo top*/,
                     {1, 1, 0} /*in halo bottom*/
             },
             ""},

            /// 17 inputs (mod+2)
            {{layer_H(17, 17, k, 1, 1, 1), T},
             {
                     {7, 8, 6},           // in
                     {6, 6, 5},           // out
                     {k, k, k},           // k
                     {1, 0, 0},           // pad top
                     {0, 0, 1},           // pad bottom
                     {isiH, isiH, isiH},  // ISI
                     {0, 1, 1} /*in halo top*/,
                     {1, 1, 0} /*in halo bottom*/
             },
             "Goes beyond"},
            {{layer_H(17, 15, k, 0, 0, 1), T},
             {
                     {7, 7, 7},           // in
                     {5, 5, 5},           // out
                     {k, k, k},           // k
                     {0, 0, 0},           // pad top
                     {0, 0, 0},           // pad bottom
                     {isiH, isiH, isiH},  // ISI
                     {0, 1, 1} /*in halo top*/,
                     {1, 1, 0} /*in halo bottom*/
             },
             ""},
            {{layer_H(17, 16, k, 1, 0, 1), T},
             {
                     {7, 8, 6},           // in
                     {6, 6, 4},           // out
                     {k, k, k},           // k
                     {1, 0, 0},           // pad top
                     {0, 0, 0},           // pad bottom
                     {isiH, isiH, isiH},  // ISI
                     {0, 1, 1} /*in halo top*/,
                     {1, 1, 0} /*in halo bottom*/
             },
             ""},
            {{layer_H(17, 16, k, 0, 1, 1), T},
             {
                     {8, 8, 5},           // in
                     {6, 6, 4},           // out
                     {k, k, k},           // k
                     {0, 0, 0},           // pad top
                     {0, 0, 1},           // pad bottom
                     {isiH, isiH, isiH},  // ISI
                     {0, 1, 1} /*in halo top*/,
                     {1, 1, 0} /*in halo bottom*/
             },
             ""},
    };

    executeT_SOH_inHALO(tests);
}

// split on 3 tiles does not work, and is against representation rules. Here we can have second tile using padding from
// first tile , nut also having halo from first tile.
TEST_F(DPULayerTest, DISABLED_zSplitSOHinHALO_basic_K5_3T_DISABLED) {
    // IN ,O, K , PT, PB, S
    int k = 5;
    const std::vector<TestCase> tests{
            {{layer_H(5, 3, k, 1, 1, 1), 3},
             {
                     {4, 5, 4},           // in
                     {1, 1, 1},           // out
                     {k, k, k},           // k
                     {1, 0, 0},           // pad top
                     {0, 0, 1},           // pad bottom
                     {isiH, isiH, isiH},  // ISI
                     {0, 2, 2} /*in halo top*/,
                     {2, 2, 0} /*in halo bottom*/
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
                     {4, 5, 3},           // in
                     {2, 2, 1},           // out
                     {k, k, k},           // k
                     {2, 0, 0},           // pad top
                     {0, 1, 2},           // pad bottom
                     {isiH, isiH, isiH},  // ISI
                     {0, 2, 3} /*in halo top*/,
                     {2, 0, 0} /*in halo bottom*/
             },
             "Second is at end already: no middle bot halo"},
            {{layer_H(5, 3, k, 2, 0, 1), 3},
             {
                     {3, 4, 5},           // in
                     {1, 1, 1},           // out
                     {k, k, k},           // k
                     {2, 1, 0},           // pad top
                     {0, 0, 0},           // pad bottom
                     {isiH, isiH, isiH},  // ISI
                     {0, 2, 2} /*in halo top*/,
                     {1, 2, 0} /*in halo bottom*/
             },
             "second still uses pad begin layer DEFECT"},
            {{layer_H(5, 3, k, 0, 2, 1), 3},
             {
                     {5, 4, 3},  // in
                     {1, 1, 1},  // out
                     {k, k, k},  // k
                     {0, 0, 0},  // pad top
                     {0, 1, 2},  // pad bottom
                     HALO_ZERO3,
             },
             " "},
            {{layer_H(5, 4, k, 2, 1, 1), 3},
             {
                     {4, 5},  // in
                     {2, 2},  // out
                     {k, k},  // k
                     {2, 0},  // pad top
                     {0, 1},  // pad bottom
                     HALO_ZERO3,
             },
             "ONly 2 splits a."},
            {{layer_H(5, 4, k, 1, 2, 1), 3},
             {
                     {5, 4},  // in
                     {2, 2},  // out
                     {k, k},  // k
                     {1, 0},  // pad top
                     {0, 2},  // pad bottom
                     HALO_ZERO3,
             },
             "only 2 splits b."},
    };

    executeT_SOH_inHALO(tests);
}

}  // namespace VPUNN_unit_tests
