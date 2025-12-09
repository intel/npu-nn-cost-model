// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.
#include "cost_model_statistical.h"

/// @brief namespace for Unit tests of the C++ library
namespace VPUNN_unit_tests {
using namespace VPUNN;

class CostModelStochasticVPU2x : public CostModelStochastic {
public:
protected:

    void SetUp() override {
    }

    /// represents the low threshold cycle value that is considered invalid. All values (cycles) smaller or equal to
    /// this are too small to be real/ good results
    constexpr static unsigned int VPU20_default_low_threshold_invalid_cycles_value{1};
    constexpr static unsigned int VPU27_default_low_threshold_invalid_cycles_value{70};

    /// represents the max value that is still OK, reasonable to be real.
    constexpr static unsigned int VPU20_default_high_threshold_invalid_cycles_value{1000 * 1000 *
                                                                                    1000};  // a billion, 1 second
    constexpr static unsigned int VPU27_default_high_threshold_invalid_cycles_value{
            10 * 1000 * 1000};  // 10 millions, 10 micro second

    constexpr static float VPU20_outOfRange_ratio_threshold_FAST{0.015F};  // 15 in 1000
    constexpr static float VPU20_outOfRange_ratio_threshold_NORM{0.015F};  // 15 in 1000

    constexpr static float VPU27_outOfRange_ratio_threshold{0.0F};  // strict

    

private:
};

/// tests that no model gives DPU cycles outside of range , inputs are random workloads
TEST_F(CostModelStochasticVPU2x, DISABLED_NoOutOfRangeDPUCycles_fast_2_0_stochastic) {
    unsigned int n_workloads = 1000;
    const ModelDescriptor& model_info{the_NN_models.fast_model_paths[0]};

    auto cnt = CheckInValidIntervalCycles(model_info, VPU20_default_low_threshold_invalid_cycles_value,
                                          VPU20_default_high_threshold_invalid_cycles_value, n_workloads);

    const float ratio_accepted_out_of_range = VPU20_outOfRange_ratio_threshold_FAST;  // 5 in a 1000

    EXPECT_LE(std::get<0>(cnt) + std::get<1>(cnt), ((int)(ratio_accepted_out_of_range * (float)n_workloads)) + 1)
            << "\n FAILED count:" << std::get<0>(cnt) << "  LOW!, " << std::get<1>(cnt) << "  HIGH!, "
            << "  problems out of all: " << n_workloads << " for model: " << model_info.first << std::endl;
}
/// tests that no model gives DPU cycles outside of range , inputs are random workloads
TEST_F(CostModelStochasticVPU2x, DISABLED_NoOutOfRangeDPUCycles_fast_2_7_stochastic) {
    unsigned int n_workloads = 1000;
    const ModelDescriptor& model_info{the_NN_models.fast_model_paths[1]};

    auto cnt = CheckInValidIntervalCycles(model_info, VPU27_default_low_threshold_invalid_cycles_value,
                                          VPU27_default_high_threshold_invalid_cycles_value, n_workloads);

    const float ratio_accepted_out_of_range = VPU27_outOfRange_ratio_threshold;  // strict

    EXPECT_LE(std::get<0>(cnt) + std::get<1>(cnt), ((int)(ratio_accepted_out_of_range * (float)n_workloads)) + 1)
            << "\n FAILED count:" << std::get<0>(cnt) << "  LOW!, " << std::get<1>(cnt) << "  HIGH!, "
            << "  problems out of all: " << n_workloads << " for model: " << model_info.first << std::endl;
}

/// tests that no model gives DPU cycles outside of range , inputs are random workloads
TEST_F(CostModelStochasticVPU2x, DISABLED_NoOutOfRangeDPUCycles_2_0_normal_stochastic) {
    unsigned int n_workloads = 1000;
    const ModelDescriptor& model_info{the_NN_models.standard_model_paths[0]};

    auto cnt = CheckInValidIntervalCycles(model_info, VPU20_default_low_threshold_invalid_cycles_value,
                                          VPU20_default_high_threshold_invalid_cycles_value, n_workloads);

    const float ratio_accepted_out_of_range = VPU20_outOfRange_ratio_threshold_NORM;  // 10

    EXPECT_LE(std::get<0>(cnt) + std::get<1>(cnt), ((int)(ratio_accepted_out_of_range * (float)n_workloads)) + 1)
            << "\n FAILED count:" << std::get<0>(cnt) << "  LOW!, " << std::get<1>(cnt) << "  HIGH!, "
            << "  problems out of all: " << n_workloads << " for model: " << model_info.first << std::endl;
}

/// tests that no model gives DPU cycles outside of range , inputs are random workloads
TEST_F(CostModelStochasticVPU2x, NoOutOfRangeDPUCycles_2_7_normal_stochastic) {
    unsigned int n_workloads = 1000;
    const ModelDescriptor& model_info{the_NN_models.standard_model_paths[1]};

    auto cnt = CheckInValidIntervalCycles(model_info, VPU27_default_low_threshold_invalid_cycles_value,
                                          VPU27_default_high_threshold_invalid_cycles_value, n_workloads);

    const float ratio_accepted_out_of_range = VPU27_outOfRange_ratio_threshold;  // strict
    auto max_deviations_allowed{(int)(ratio_accepted_out_of_range * (float)n_workloads)};
    if (max_deviations_allowed <= 0) {  // at least 1 is permitted
        max_deviations_allowed = 1;
    }

    EXPECT_LE(std::get<0>(cnt) + std::get<1>(cnt), max_deviations_allowed)
            << "\n FAILED count:" << std::get<0>(cnt) << "  LOW!, " << std::get<1>(cnt) << "  HIGH!, "
            << "  problems out of all: " << n_workloads << " for model: " << model_info.first << std::endl;
}

/// tests that inference output is inside a valid interval
///  same workloads for fast and normal
TEST_F(CostModelStochasticVPU2x, DISABLED_Inference_output_in_Interval_Test_2_0_all_stochastic) {
    const float low_threshold{0.5F};
    const float high_threshold{1E9};  // one billion

    const unsigned int n_workloads = 1000;

    const VPUNN::VPUDevice device_version = VPUNN::VPUDevice::VPU_2_0;
    // Generate a bunch of random workloads
    auto workloads = std::vector<VPUNN::DPUWorkload>(n_workloads);
    std::generate_n(workloads.begin(), n_workloads, VPUNN::randDPUWorkload(device_version));

    const int modelIndex{0};  // 2.0

    {  // slow
        const ModelDescriptor& model_info{the_NN_models.standard_model_paths[modelIndex]};
        const float ratio_accepted_out_of_range = VPU20_outOfRange_ratio_threshold_NORM;  // 10 in 1000

        const auto cnt = CheckInValidInterval_RawInference(model_info, low_threshold, high_threshold, workloads);
        const auto undervalues{std::get<0>(cnt)};
        const auto overvalues{std::get<1>(cnt)};
        EXPECT_LE(undervalues + overvalues, (int)(ratio_accepted_out_of_range * (float)n_workloads))
                << "\n FAILED count:" << undervalues << " <LOW, " << overvalues << " >HIGH, "
                << "  problems out of all: " << n_workloads << " for model: " << model_info.first << std::endl;
        std::cout << " Normal NN Step DONE !\n";
    }

    {  // fast
        const ModelDescriptor& model_info{the_NN_models.fast_model_paths[modelIndex]};
        const float ratio_accepted_out_of_range = VPU20_outOfRange_ratio_threshold_FAST;  // 5 in a 1000

        const auto cnt = CheckInValidInterval_RawInference(model_info, low_threshold, high_threshold, workloads);
        const auto undervalues{std::get<0>(cnt)};
        const auto overvalues{std::get<1>(cnt)};
        EXPECT_LE(undervalues + overvalues, (int)(ratio_accepted_out_of_range * (float)n_workloads))
                << "\n FAILED count:" << undervalues << " <LOW, " << overvalues << " >HIGH, "
                << "  problems out of all: " << n_workloads << " for model: " << model_info.first << std::endl;
        std::cout << " Fast NN Step DONE !\n";
    }
}

/// tests that inference output is inside a valid interval
///  same workloads for fast and normal
TEST_F(CostModelStochasticVPU2x, Inference_output_in_Interval_Test_2_7_all_stochastic) {
    const float low_threshold{0.5F};
    const float high_threshold{1E9};  // one billion

    const unsigned int n_workloads = 1000;

    const VPUNN::VPUDevice device_version = VPUNN::VPUDevice::VPU_2_7;
    // Generate a bunch of random workloads
    auto workloads = std::vector<VPUNN::DPUWorkload>(n_workloads);
    std::generate_n(workloads.begin(), n_workloads, VPUNN::randDPUWorkload(device_version));

    const int modelIndex{1};                                                     // 2.7
    const float ratio_accepted_out_of_range = VPU27_outOfRange_ratio_threshold;  // strict
    auto max_deviations_allowed{(int)(ratio_accepted_out_of_range * (float)n_workloads)};
    if (max_deviations_allowed <= 0) {  // at least 1 is permitted
        max_deviations_allowed = 1;
    }

    {  // slow
        const ModelDescriptor& model_info{the_NN_models.standard_model_paths[modelIndex]};

        const auto cnt = CheckInValidInterval_RawInference(model_info, low_threshold, high_threshold, workloads);
        const auto undervalues{std::get<0>(cnt)};
        const auto overvalues{std::get<1>(cnt)};
        EXPECT_LE(undervalues + overvalues, max_deviations_allowed)
                << "\n FAILED count:" << undervalues << " <LOW, " << overvalues << " >HIGH, "
                << "  problems out of all: " << n_workloads << " for model: " << model_info.first << std::endl;
        std::cout << " Normal NN Step DONE !\n";
    }

    //{  // fast
    //    const ModelDescriptor& model_info{the_NN_models.fast_model_paths[modelIndex]};

    //    const auto cnt = CheckInValidInterval_RawInference(model_info, low_threshold, high_threshold, workloads);
    //    const auto undervalues{std::get<0>(cnt)};
    //    const auto overvalues{std::get<1>(cnt)};
    //    EXPECT_LE(undervalues + overvalues, max_deviations_allowed)
    //            << "\n FAILED count:" << undervalues << " <LOW, " << overvalues << " >HIGH, "
    //            << "  problems out of all: " << n_workloads << " for model: " << model_info.first << std::endl;
    //    std::cout << " Fast NN Step DONE !\n";
    //}
}

/// tests that no model gives 1.0 as output
TEST_F(CostModelStochasticVPU2x, DISABLED_NoONE_asInference_output_all_stochastic) {
    unsigned int n_workloads = 1000;

    // nPU2.0 will give negative values

    for (auto& model_info : the_NN_models.all_model_paths) {
        auto cnt = CheckNoCase_InferenceOutput(model_info, 1.0F, n_workloads);
        EXPECT_EQ(cnt, 0) << "\n FAILED count:" << cnt << " problems out of all: " << n_workloads
                          << " for model: " << model_info.first << std::endl;
    }
}

/// Make a fast versus slow statistical comparison.   no big delta expected.
TEST_F(CostModelStochasticVPU2x, DISABLED_Comparative_fast_vs_slow_20_stochastic) {
    const float max_ratio_delta{0.5F};  // between (fast and slow) delta over min of them  . (120-80)/80
    // const float expected_deviation_ratio{0.003036F};  // Assumption: 95% (2*sigma) are in the 80-120% interval
    //  sigma = 0.1, miu=1

    const float expected_deviation_ratio{0.40F};  // big but not huge, 40%

    const int min_absolute_delta{50};  // not considered delta below this amount of cycles

    const int modelIndex{0};  // 2.0
    const std::string model_type{"2.0"};
    const VPUNN::VPUDevice device_version = VPUNN::VPUDevice::VPU_2_0;

    const unsigned int n_workloads = 1000;

    const auto errors = comparative_fast_vs_slow(max_ratio_delta, expected_deviation_ratio, min_absolute_delta,
                                                 modelIndex, std::move(model_type), device_version, n_workloads);

    EXPECT_LE(errors, (int)(expected_deviation_ratio * n_workloads) + 1) << expected_deviation_ratio * n_workloads;
}

/// Make a fast versus slow statistical comparison.   no big delta expected.
TEST_F(CostModelStochasticVPU2x, DISABLED_Comparative_fast_vs_slow_27_stochastic) {
    // 5 will produce 1.5%
    const float tolerance_factor{15.0F};  ///<  how much many samples do we allow (factor) versus strict theoretical

    const float max_ratio_delta{0.5F};  // between (fast and slow) delta over min of them  . (120-80)/80
    const float expected_deviation_ratio{(0.3036F / 100.0F) * tolerance_factor};  // Assumption: 95% (2*sigma) are in
                                                                                  // the 80-120% interval sigma = 0.1,
                                                                                  // miu=1, Latest will be ~5%

    const int min_absolute_delta{50};  // not considered delta below this amount of cycles
    const int modelIndex{1};           // 2.7

    const std::string model_type{"2.7"};
    const VPUNN::VPUDevice device_version = VPUNN::VPUDevice::VPU_2_7;

    const unsigned int n_workloads = 1000;

    const auto errors = comparative_fast_vs_slow(max_ratio_delta, expected_deviation_ratio, min_absolute_delta,
                                                 modelIndex, std::move(model_type), device_version, n_workloads);

    EXPECT_LE(errors, (int)(expected_deviation_ratio * n_workloads) + 1) << expected_deviation_ratio * n_workloads;
}

/// Make a  statistical comparison.   no big delta expected.
/// This takes a lot of time (minutes) and you need locally the NN files to compare
/// cannot be run on CI, and it is not intended for repetitive CI run.
TEST_F(CostModelStochasticVPU2x, DISABLED_Comparative_MATRIX_VPUNNs_stochastic) {
    const float max_ratio_delta{0.5F};                // between (fast and slow) delta over min of them  . (120-80)/80
    const float expected_deviation_ratio{0.003036F};  // Assumption: 95% (2*sigma) are in the 80-120% interval
                                                      // sigma = 0.1, miu=1,

    const int min_absolute_delta{50};  // not considered delta below this amount of cycles

    const unsigned int n_workloads = 1000;

    std::string model_type;
    VPUNN::VPUDevice device_version;

    using ResultMap = std::unordered_map<std::string, DataOut>;

    auto comp_lambda = [&](const std::string nn1, const std::string nn2, const int silence,
                           ResultMap& resultAccumulator) {
        std::cout << "\n ----------------------------------------------------------------------------------------";
        std::cout << "\n COMPARING VPUNNS:"
                  << "\n NN1:  " << nn1 << "\n NN2:  " << nn2 << "\n";

        const auto expected_failures{std::lround(std::ceil(expected_deviation_ratio * n_workloads))};  // how
        const std::string key = std::string(nn1 + nn2);
        const std::string key_alt = std::string(nn2 + nn1);

        auto& res = resultAccumulator[key];

        EXPECT_LE(comparative_run(max_ratio_delta, expected_deviation_ratio, min_absolute_delta, model_type,
                                  device_version, nn1, nn2, n_workloads, res, silence),
                  expected_failures)
                << "\n FAILED to match:"
                << "\n NN1:  " << nn1 << "\n NN2:  " << nn2;

        resultAccumulator[key_alt] = resultAccumulator[key];  // double it
    };

    auto results_table_show_lambda = [&](auto& nn_list, ResultMap& results) {
        std::cout << "\n xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx";
        const auto default_precision{std::cout.precision()};
        std::cout << "\n ERRORS matrix: how many samples (out of: " << n_workloads
                  << " ) have big delta (larger than: " << (int)(max_ratio_delta * 100) << "% )";
        for (const auto& nn1 : nn_list) {
            std::cout << "\n " << std::setw(100) << nn1 << "\t";
            for (const auto& nn2 : nn_list) {
                const std::string key = std::string(nn1 + nn2);
                auto& val = results[key];
                if (val.errors_cnt >= 0)
                    std::cout << "\t " << std::setw(5) << val.errors_cnt;
                else
                    std::cout << "\t " << std::setw(5) << "-";
            }
        }

        std::cout << "\n CORRELATION matrix";
        for (const auto& nn1 : nn_list) {
            std::cout << "\n " << std::setw(100) << nn1 << "\t";
            for (const auto& nn2 : nn_list) {
                const std::string key = std::string(nn1 + nn2);
                auto& val = results[key];
                if (val.correlation >= -5.0)
                    std::cout << "\t " << std::setw(5) << std::setprecision(3) << val.correlation;
                else
                    std::cout << "\t " << std::setw(5) << "-";
            }
        }
        std::cout << std::setprecision(default_precision)
                  << "\n xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx";
        /* coverity[end_of_path] */
    };

    {
        model_type = "2.0";
        device_version = VPUNN::VPUDevice::VPU_2_0;

        const std::vector<std::string> nns{
                VPU_2_0_MODEL_PATH,
                (NameHelperNN::get_model_root() + "vpu_2_0-150-1.vpunn"),
                //(NameHelperNN::get_model_root() + "vpu_2_0-orig-141.vpunn"),

                NameHelperNN::make_fast_version(VPU_2_0_MODEL_PATH),
                (NameHelperNN::get_model_root() + "vpu_2_0.fast-150-1.vpunn"),
                (NameHelperNN::get_model_root() + "vpu_2_0.fast-141.vpunn"),

        };

        ResultMap res;
        /* for (unsigned int i = 0; i < nns.size(); ++i)
             for (unsigned int j = i + 1; j < nns.size(); ++j)
                 comp_lambda(nns[i], nns[j], 2, res);*/

        results_table_show_lambda(nns, res);
    }

    {
        model_type = "2.7";
        device_version = VPUNN::VPUDevice::VPU_2_7;

        const std::vector<std::string> nns{
                VPU_2_7_MODEL_PATH,
                (NameHelperNN::get_model_root() + "vpu_2_7-160cp17L.vpunn"),
                (NameHelperNN::get_model_root() + "vpu_2_7-160cp16L.vpunn"),
                (NameHelperNN::get_model_root() + "vpu_2_7-155L.vpunn"),

                (NameHelperNN::get_model_root() + "vpu_2_7-150.vpunn"),
                (NameHelperNN::get_model_root() + "vpu_2_7-150-1.vpunn"),
                (NameHelperNN::get_model_root() + "vpu_2_7-141.vpunn"),

                NameHelperNN::make_fast_version(VPU_2_7_MODEL_PATH),
                (NameHelperNN::get_model_root() + "vpu_2_7.fast-160cp17L.vpunn"),
                (NameHelperNN::get_model_root() + "vpu_2_7.fast-160cp16L.vpunn"),
                (NameHelperNN::get_model_root() + "vpu_2_7.fast-155L.vpunn"),

                (NameHelperNN::get_model_root() + "vpu_2_7.fast-150.vpunn"),
                (NameHelperNN::get_model_root() + "vpu_2_7.fast-150-1.vpunn"),
                (NameHelperNN::get_model_root() + "vpu_2_7.fast-141.vpunn"),
        };

        ResultMap res;

        for (unsigned int i = 0; i < nns.size(); ++i)
            for (unsigned int j = i + 1; j < nns.size(); ++j)
                comp_lambda(nns[i], nns[j], 2, res);

        results_table_show_lambda(nns, res);
    }
}

}  // namespace VPUNN_unit_tests