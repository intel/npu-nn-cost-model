// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_UT_COST_MODEL_H
#define VPUNN_UT_COST_MODEL_H

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "common_helpers.h"

#include "vpu/cycles_interface_types.h"

#include "core/logger.h"
#include "vpu/dpu_types.h"
#include "vpu/dpu_workload.h"
#include "vpu/vpu_tensor.h"
#include "vpu_cost_model.h"

/// @brief namespace for Unit tests of the C++ library
namespace VPUNN_unit_tests {
using namespace VPUNN;

class TestCostModel : public ::testing::Test {
public:
protected:
    DPUWorkload wl_glob_27{VPUDevice::VPU_2_7,
                           Operation::CONVOLUTION,
                           {VPUTensor(56, 56, 16, 1, DataType::UINT8)},  // input dimensions
                           {VPUTensor(56, 56, 16, 1, DataType::UINT8)},  // output dimensions
                           {3, 3},                                       // kernels
                           {1, 1},                                       // strides
                           {1, 1, 1, 1},                                 // padding
                           ExecutionMode::CUBOID_16x16};
    DPUWorkload wl_glob_20;
    DPUWorkload wl_glob_40;


    VPUCostModel model{};

    void SetUp() override {
        wl_glob_20 = wl_glob_27;
        wl_glob_20.device = VPUDevice::VPU_2_0;
        wl_glob_20.execution_order = ExecutionMode::MATRIX;

        wl_glob_40 = wl_glob_27;
        wl_glob_40.device = VPUDevice::VPU_4_0;


        Logger::clear2ndlog();
        // Logger::activate2ndlog();
    }
    void TearDown() override {
        Logger::clear2ndlog();
        Logger::deactivate2ndlog();
    }

    auto read_a_file(const std::string& filename) const {
        std::vector<char> buf(0);
        std::ifstream myFile;
        myFile.open(filename, std::ios::binary | std::ios::in);
        if (myFile.fail()) {
            // File does not exist code here
            return buf;
        }
        myFile.seekg(0, std::ios::end);
        const auto length = myFile.tellg();
        myFile.seekg(0, std::ios::beg);

        buf.resize(static_cast<size_t>(length));

        myFile.read(buf.data(), length);
        myFile.close();
        return buf;
    }

    using ModelDescriptor =
            VPUNNModelsFiles::ModelDescriptor;  ///< make this type directly visible inside of this class
    const VPUNNModelsFiles& the_NN_models{VPUNNModelsFiles::getModels()};  ///< the paths to available NN models

    bool is_error_code(unsigned int cycles) {
        if (cycles > std::numeric_limits<uint32_t>::max() - 1000)
            return true;
        return false;
    }

    struct DataOut {
        int errors_cnt{-1};
        double correlation{-10.0};
    };

    VPUNN::CyclesInterfaceType delta_cycles(const VPUNN::CyclesInterfaceType& v1,
                                            const VPUNN::CyclesInterfaceType& v2) {
        return (v1 >= v2) ? (v1 - v2) : (v2 - v1);  // aways positive
    }

    /// @brief max allowable delta between 2 cycles , so that we consider them still equal
    ///
    /// @param v1 a value
    /// @param v2 another value
    /// @param tolerance_level how permissive to be in delta.
    /// @returns max value that can be between v1 and v2 so that they are practically equal.
    VPUNN::CyclesInterfaceType max_tolerance_cycles(const VPUNN::CyclesInterfaceType& v1,
                                                    const VPUNN::CyclesInterfaceType& v2,
                                                    const int tolerance_level = 1) {
        const VPUNN::CyclesInterfaceType v{std::max(v1, v2)};

        VPUNN::CyclesInterfaceType tolerance{1U};  // rounding errors

        if (tolerance_level <= 1) {
            if (v >= 10000000U) {  // 10 millions
                tolerance = 1000U;
            } else if (v >= 1000000U) {  // 1 million
                tolerance = 200U;
            } else if (v >= 100000U) {  // 100k
                tolerance = 50U;
            } else if (v >= 1000U) {
                tolerance = 10U;
            }

        } else if (tolerance_level > 1) {
            if (v >= 10000000U) {  // 10 millions
                tolerance = 2000U;
            } else if (v >= 1000000U) {  // 1 million
                tolerance = 500U;
            } else if (v >= 100000U) {  // 100k
                tolerance = 100U;
            } else if (v >= 1000U) {
                tolerance = 10U;
            }
        }

        return tolerance;
    }

    std::string current_test_name() const {
        const ::testing::TestInfo* test_info = ::testing::UnitTest::GetInstance()->current_test_info();
        return test_info->name();
    }

    std::string current_test_fixture_name() const {
        const ::testing::TestInfo* test_info = ::testing::UnitTest::GetInstance()->current_test_info();
        return test_info->test_suite_name();
    }

protected:
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
        }
    }

    const unsigned int MAX_COST{10000000};  // ten million
    static constexpr unsigned int NO_ERROR_EXPECTED{VPUNN::Cycles::NO_ERROR};

    struct GTestInput {
        DPUWorkload w;
        int GT_val{-1};
    };

    struct GTestExpectations {
        CyclesInterfaceType err_expected{NO_ERROR_EXPECTED};
        bool strict_err_check{false};
        CyclesInterfaceType min_cyc{0};
        CyclesInterfaceType max_cyc{0};

        // if we accept exception test cases
        CyclesInterfaceType e_min_cyc{
                0};  /// lower limit for exception cases, cycles value should be in interval [e_min_cyc, min_cyc)
        CyclesInterfaceType e_max_cyc{
                0};  /// upper limit for exception cases, cycles value should be in interval (max_cyc, e_max_cyc]
    };

    struct GTestCase {
        GTestInput t_in;
        GTestExpectations t_exp;
        const std::string test_case = "";
    };
    using GTestsVector = std::vector<GTestCase>;

    bool show_info{true};  ///> controls  show in DoRegularTest
    bool show_wl_info{true};
    const bool is_excep_allowed{true};  /// if true => we accept exceptions for our tests, ex: delta is very slightly
                                        /// higher than the value taken by us and we want to accept this case as well

    void DoRegularTest(const GTestInput& t_in, const GTestExpectations& t_exp, const std::string& test_case = "") {
        DPUWorkload w1{t_in.w};

        std::string t_header{"** Test Case: " + test_case};
        // std::cout << ">> " << t_header << std::endl;

        Logger::clear2ndlog();

        std::string test_info_str = test_case;
        std::replace(test_info_str.begin(), test_info_str.end(), ',', '_');
        test_info_str.erase(std::remove(test_info_str.begin(), test_info_str.end(), '\n'), test_info_str.end());
        // TO BE REFACTORED
        const std::string info_layer{current_test_fixture_name() + "::" + current_test_name() + "::" + test_info_str +
                                     "::" + std::to_string(t_in.GT_val)};
        getModel(w1.device).get_serializer().serialize(SerializableField<std::string>("info", info_layer));

        w1.set_layer_info(info_layer);
        unsigned cost_cyc{};
        std::string info;
        ASSERT_NO_THROW(cost_cyc = getModel(w1.device).DPU(w1, info)) << t_header << w1;

        auto show_wl = [this](const DPUWorkload& wl) -> std::string {
            if (show_wl_info) {
                std::ostringstream oss;
                oss << wl;
                return oss.str();
            }
            return "";
        };

        auto computePercentageDiff = [](int cyc, int gt) -> float {
            int diff = cyc - gt;
            float percentage = (static_cast<float>(diff) / gt) * 100.0f;

            return percentage;
        };

        const auto err_expected{t_exp.err_expected};
        if (Cycles::isErrorCode(err_expected)) {  // error code is expected
            EXPECT_TRUE(Cycles::isErrorCode(cost_cyc))
                    << t_header << "Expected ERROR, but value received: " << cost_cyc << " : "
                    << Cycles::toErrorText(cost_cyc) << std::endl
                    << "Expected ERROR code: " << err_expected << " : " << Cycles::toErrorText(err_expected)
                    << std::endl
                    << show_wl(w1) << Logger::get2ndlog() << (show_info ? info : "\n");

            if (Cycles::isErrorCode(cost_cyc) &&
                t_exp.strict_err_check) {  // in case is error code AND we want to have exact value
                EXPECT_EQ(cost_cyc, err_expected) << t_header << "ERROR code received: " << cost_cyc << " : "
                                                  << Cycles::toErrorText(cost_cyc) << std::endl
                                                  << "Expected ERROR code: " << err_expected << " : "
                                                  << Cycles::toErrorText(err_expected) << std::endl
                                                  << show_wl(w1) << Logger::get2ndlog() << (show_info ? info : "\n");
            }

        } else {  // regular cycle value expected
            EXPECT_FALSE(Cycles::isErrorCode(cost_cyc))
                    << t_header << "Unexpected ERROR code: " << cost_cyc << " : " << Cycles::toErrorText(cost_cyc)
                    << "\n  expected in [ " << t_exp.min_cyc << " , " << t_exp.max_cyc << " ] \n"
                    << show_wl(w1) << Logger::get2ndlog() << (show_info ? info : "\n");

            if (!Cycles::isErrorCode(cost_cyc)) {  // if cycle value check against other ranges
                EXPECT_GT(cost_cyc, 0u) << t_header << show_wl(w1);
                EXPECT_LT(cost_cyc, MAX_COST) << t_header << show_wl(w1);  // 1 million

                if (!is_excep_allowed || (t_exp.e_min_cyc == 0 && t_exp.e_max_cyc == 0)) {
                    EXPECT_TRUE(cost_cyc >= t_exp.min_cyc && cost_cyc <= t_exp.max_cyc)
                            << t_header << " FAILED: Cost " << cost_cyc << " not in interval,  expected in [ "
                            << t_exp.min_cyc << ", "
                            << ((t_in.GT_val != -1) ? ("GT:" + std::to_string(t_in.GT_val)) : "GT") << " , "
                            << t_exp.max_cyc << " ] \n"
                            << show_wl(w1) << Logger::get2ndlog() << (show_info ? info : "\n");
                } else {
                    EXPECT_TRUE(cost_cyc >= t_exp.e_min_cyc && cost_cyc <= t_exp.e_max_cyc)
                            << t_header << " FAILED: Cost " << cost_cyc << " not in interval,  expected in  ["
                            << t_exp.e_min_cyc << " [" << t_exp.min_cyc << ", "
                            << ((t_in.GT_val != -1) ? ("GT:" + std::to_string(t_in.GT_val)) : "GT") << ", "
                            << t_exp.max_cyc << "] " << t_exp.e_max_cyc << " ] EXCEPTION ALLOWED\n"
                            << show_wl(w1) << Logger::get2ndlog() << (show_info ? info : "\n");
                }
            }
        }

        // function to format a float value to a string with a specified number of decimal places
        auto floatToString = [](float value, int precision) -> std::string {
            std::ostringstream outs;
            outs << std::fixed << std::setprecision(precision) << value;
            return outs.str();
        };

        // info about GT
        float percentage = computePercentageDiff(cost_cyc, t_in.GT_val);
        std::string delta_info = " delta: " + floatToString(percentage, 1) + "% ";

        std::string interval =
                (((-1 != t_in.GT_val) && (is_excep_allowed) && (t_exp.e_min_cyc != 0 && t_exp.e_max_cyc != 0))
                         ? ("[" + std::to_string(t_exp.e_min_cyc) + " [" + std::to_string(t_exp.min_cyc) + ", " +
                            ((t_in.GT_val != -1) ? ("GT:" + std::to_string(t_in.GT_val)) : "GT") + ", " +
                            std::to_string(t_exp.max_cyc) + "] " + std::to_string(t_exp.e_max_cyc) + "] ")
                         : ("[" + std::to_string(t_exp.min_cyc) + ", " +
                            ((t_in.GT_val != -1) ? ("GT:" + std::to_string(t_in.GT_val)) : "GT") + ", " +
                            std::to_string(t_exp.max_cyc) + " ] "));

        std::string exception_info = "PASSED BY EXCEPTION!";

        std::cout << t_header << (Cycles::isErrorCode(cost_cyc) ? " *** ERROR code: " : " *** Cycles code: ")
                  << cost_cyc << " " << ((Cycles::isErrorCode(cost_cyc)) ? Cycles::toErrorText(cost_cyc) : "")
                  << " expected in " << interval << ((-1 != t_in.GT_val) ? std::move(delta_info) : "")
                  << (((-1 != t_in.GT_val) && (is_excep_allowed) && (t_exp.e_min_cyc != 0 && t_exp.e_max_cyc != 0) &&
                       ((t_exp.e_min_cyc <= cost_cyc && t_exp.min_cyc > cost_cyc) ||
                        (t_exp.e_max_cyc >= cost_cyc && t_exp.max_cyc < cost_cyc)))
                              ? std::move(exception_info)
                              : "")
                  << Logger::get2ndlog() << (show_info ? std::move(info) : "\n") << std::endl
                  << "------------------------------------------------------------------------" << std::endl;
    }

    void executeTests(const GTestsVector& tests) {
        int test_index = 0;
        for (const auto& t : tests) {
            std::stringstream buffer;
            buffer << test_index << " : " << t.test_case;
            const std::string test_case_info = buffer.str();

            DoRegularTest(t.t_in, t.t_exp, test_case_info);

            ++test_index;
        }
    }

private:
};

}  // namespace VPUNN_unit_tests
#endif  // VPUNN_UT_COST_MODEL_H
