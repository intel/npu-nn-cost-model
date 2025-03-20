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
#include "vpu/sample_generator/random_task_generator.h"

#include <unordered_map>

/// @brief namespace for Unit tests of the C++ library
namespace VPUNN_unit_tests {

class CostModelStochastic : public ::testing::Test {
public:
protected:
    VPUNN::VPUCostModel model{};

    void SetUp() override {
    }

    using ModelDescriptor =
            VPUNNModelsFiles::ModelDescriptor;  ///< make this type directly visible inside of this class
    const VPUNNModelsFiles& the_NN_models{VPUNNModelsFiles::getModels()};  ///< the paths to available NN models

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

    std::tuple<int, int> CheckInValidIntervalCycles(const ModelDescriptor& model_info, const unsigned int low_threshold,
                                                    const unsigned int high_threshold, unsigned int n_workloads) {
        const auto& model_path = model_info.first;

        VPUNN::VPUCostModel current_model{model_path, false};  // with cache, batch =1
        const VPUNN::VPUDevice device = model_info.second;

        // Generate N workloads
        auto workloads = std::vector<VPUNN::DPUWorkload>(n_workloads);
        std::generate_n(workloads.begin(), n_workloads, VPUNN::randDPUWorkload(device));

        int ilow{0};   // increments at every error
        int ihigh{0};  // increments at every error
        for (auto& wl : workloads) {
            const auto cycles = current_model.DPU(wl);

            if (cycles < low_threshold) {
                std::cout << "\n (cycles < low_threshold): " << cycles << "threshold: " << low_threshold
                          << "\nMeaning: " << VPUNN::Cycles::toErrorText(cycles) << " Model : " << model_path
                          << ".i : " << ++ilow << std::endl
                          << wl << std::endl;
            }
            if (cycles > high_threshold) {
                std::cout << "\n (cycles > high_threshold): " << cycles << "threshold: " << high_threshold
                          << "\nMeaning: " << VPUNN::Cycles::toErrorText(cycles) << " Model: " << model_path
                          << ". i:" << ++ihigh << std::endl
                          << wl << std::endl;
            }
        }
        return std::make_tuple(ilow, ihigh);
    }

    int CheckNoCase_InferenceOutput(const ModelDescriptor& model_info, const float not_valid_value,
                                    unsigned int n_workloads = 1000) {
        const auto& model_path = model_info.first;
        const float checked_result = not_valid_value;
        // const float delta_error = 0.01F;

        VPUNN::VPUCostModel current_model{model_path, false, 0U};  // with cache, batch =1
        const VPUNN::VPUDevice device = model_info.second;

        // Generate N workloads
        auto workloads = std::vector<VPUNN::DPUWorkload>(n_workloads);
        std::generate_n(workloads.begin(), n_workloads, VPUNN::randDPUWorkload(device));

        int i{0};
        for (auto& wl : workloads) {
            float infered_value = current_model.run_NN(wl);
            EXPECT_GT(infered_value, checked_result)
                    << "result: " << infered_value << " i:" << ++i << " Model: " << model_path << "\n"
                    << wl << std::endl;
        }

        return i;
    }

    std::tuple<int, int> CheckInValidInterval_RawInference(const ModelDescriptor& model_info, const float low_threshold,
                                                           const float high_threshold,
                                                           const std::vector<VPUNN::DPUWorkload>& workloads) {
        const auto& model_path = model_info.first;
        VPUNN::VPUCostModel current_model{model_path, false};  // with cache, batch =1

        int ilow{0};   // increments at every error
        int ihigh{0};  // increments at every error
        int i{0};      // index
        const auto n{workloads.size()};
        for (const auto& wl : workloads) {
            float inference = current_model.run_NN(wl);

            // EXPECT_GT(inference, low_threshold)
            if (inference < low_threshold) {
                std::cout << "\n (inference < low_threshold): " << inference << "threshold: " << low_threshold
                          << "\nModel: " << model_path << ", wl-index: " << i << " / " << n << ",occurrence :" << ++ilow
                          << std::endl
                          << wl << std::endl;
            }
            // EXPECT_LT(inference, high_threshold)
            if (inference > high_threshold) {
                std::cout << "\n (inference > high_threshold): " << inference << "threshold: " << high_threshold
                          << "\nModel: " << model_path << ", wl-index: " << i << " / " << n
                          << ",occurrence :" << ++ihigh << std::endl
                          << wl << std::endl;
            }
            i++;
        }
        return std::make_tuple(ilow, ihigh);
    }

    bool is_error_code(unsigned int cycles) {
        if (cycles > std::numeric_limits<uint32_t>::max() - 1000)
            return true;
        return false;
    }

    int comparative_fast_vs_slow(const float max_ratio_delta, const float expected_deviation_ratio,
                                 const int min_absolute_delta, const int modelIndex, const std::string model_type,
                                 const VPUNN::VPUDevice device_version, const unsigned int n_workloads) {
        // const float max_ratio_delta{0.5F};  // 50%,   One +20% , other -20%, => between them 50%
        auto workloads = std::vector<VPUNN::DPUWorkload>(n_workloads);
        std::generate_n(workloads.begin(), n_workloads, VPUNN::randDPUWorkload(device_version));

        VPUNN::VPUCostModel slow_model{the_NN_models.standard_model_paths[modelIndex].first,
                                       false};  // with cache, batch =1
        VPUNN::VPUCostModel fast_model{the_NN_models.fast_model_paths[modelIndex].first,
                                       false};  // with cache, batch =1

        std::map<VPUNN::Operation, int> ops_errors;
        std::map<VPUNN::Operation, int> all_ops;

        int i_abs_errors{0};  // increments at delta > threshold
        int i_errs{0};        // increments at every error(delta + ratio)
        int i_range_errs{0};  // increments at every error
        int i{0};             // index workload
        const auto n{workloads.size()};
        for (const auto& wl : workloads) {
            all_ops[wl.op]++;
            auto cycles_s = slow_model.DPU(wl);
            auto cycles_f = fast_model.DPU(wl);

            if (is_error_code(cycles_s) || is_error_code(cycles_f)) {
                // expect both are same error
                i_range_errs++;
                // EXPECT_EQ(cycles_s, cycles_f)
                if (cycles_s != cycles_f) {
                    std::cout << "Mismatch, error codes or valid with error combination \n"
                              << "Model: " << model_type << ", wl-index: " << i << " / " << n
                              << ",occurrence :" << ++i_errs << " BOTH SHOULD HAVE SAME ERROR :" << std::endl
                              << "slow: " << cycles_s << std::endl
                              << "fast: " << cycles_f << std::endl
                              << wl << std::endl;
                }
            } else {  // no errors
                int delta_c = std::abs(((int)cycles_f - (int)cycles_s));
                float ratio{100.0F};
                const auto min_cycles = std::min(std::abs((int)cycles_s), std::abs((int)cycles_f));
                if (0 != min_cycles) {
                    ratio = (float)delta_c / min_cycles;
                }

                if (delta_c > min_absolute_delta) {
                    i_abs_errors++;
                    // EXPECT_LE(ratio, max_ratio_delta)
                    if (ratio >= max_ratio_delta) {
                        std::cout << "\n*** Results with significant difference:  \n"
                                  << "Model: " << model_type << ", wl-index: " << i << " / " << n
                                  << ", occurrence :" << ++i_errs << " BIG DIFFERENCE :" << std::endl
                                  << "slow: " << cycles_s << std::endl
                                  << "fast: " << cycles_f << std::endl
                                  << "delta: " << delta_c << std::endl
                                  << "ratio: " << ratio << std::endl
                                  << wl << std::endl;
                        ops_errors[wl.op]++;
                    }
                }
            }
            i++;  // next workload
        }
        const auto expected_failures{std::lround(std::ceil(expected_deviation_ratio * n_workloads))};  // how

        EXPECT_LE(i_errs, expected_failures)
                << "\n ERRORS when comparing slow with fast: " << i_errs << "\t out of workloads count: " << n_workloads
                << std::endl
                << "\t range/error mismatch: " << i_range_errs << std::endl
                << "\t delta absolute: " << i_abs_errors << std::endl
                << "\t ratio threshold: " << max_ratio_delta << std::endl
                << "\t absolute threshold: " << min_absolute_delta << std::endl
                << "\t expected_deviation_ratio: " << expected_deviation_ratio << std::endl;

        std::cout << "\n Total relevant Errors: " << i_errs << "\n DIstributed on operations:";
        
        for (const auto& op : ops_errors) {
            const int err_percent{(i_errs != 0) ? (int)(((float)op.second / i_errs) * 100) : 0};
            std::cout << "\n op_id: " << (int)op.first
                      << " OP: " << VPUNN::Operation_ToText.at(static_cast<int>(op.first))
                      << "\t,  Count #:" << op.second << " , Count %: " << err_percent
                      << " %";
        }
        std::cout << "\n\n Total operations:";
        for (const auto& op : all_ops) {
            std::cout << "\n op_id: " << (int)op.first
                      << " OP: " << VPUNN::Operation_ToText.at(static_cast<int>(op.first))
                      << "\t,  Count #:" << op.second << " , Count %: " << (int)(((float)op.second / n) * 100) << " %";
        }

        return i_errs;
    }
    struct DataOut {
        int errors_cnt{-1};
        double correlation{-10.0};
    };

    int comparative_run(const float max_ratio_delta, const float expected_deviation_ratio, const int min_absolute_delta,
                        const std::string model_type, const VPUNN::VPUDevice device_version, const std::string nn1,
                        const std::string nn2, const unsigned int n_workloads, DataOut& result, const int silence = 0) {
        // Generate a bunch of random workloads
        auto workloads = std::vector<VPUNN::DPUWorkload>(n_workloads);
        std::generate_n(workloads.begin(), n_workloads, VPUNN::randDPUWorkload(device_version));

        VPUNN::VPUCostModel slow_model{nn1, false};  // with cache, batch =1
        VPUNN::VPUCostModel fast_model{nn2, false};  // with cache, batch =1

        EXPECT_TRUE(slow_model.nn_initialized()) << "\n Modelfile: " << nn1 << "  not initialized";
        EXPECT_TRUE(fast_model.nn_initialized()) << "\n Modelfile: " << nn2 << "  not initialized";

        if (!slow_model.nn_initialized() || !fast_model.nn_initialized()) {
            return 10001;  // no point to run the test
        }

        double sum_s{0};
        double sum_f{0};
        double sum_sf{0};

        double sum_ss{0};
        double sum_ff{0};
        double nan_val{1000000};

        int i_abs_errors{0};  // increments at delta > threshold
        int i_errs{0};        // increments at every error(delta + ratio)
        int i_range_errs{0};  // increments at every error
        int i{0};             // index workload
        const auto n{workloads.size()};
        for (const auto& wl : workloads) {
            auto cycles_s = slow_model.DPU(wl);
            auto cycles_f = fast_model.DPU(wl);

            if (is_error_code(cycles_s) || is_error_code(cycles_f)) {
                // expect both are same error
                i_range_errs++;
                if (silence < 1)
                    EXPECT_EQ(cycles_s, cycles_f)
                            << "Model: " << model_type << ", wl-index: " << i << " / " << n
                            << ",occurrence :" << ++i_errs << " BOTH SHOULD HAVE SAME ERROR :" << std::endl
                            << "model1: " << cycles_s << std::endl
                            << "model2: " << cycles_f << std::endl
                            << wl << std::endl;
                else if (silence < 2)
                    EXPECT_EQ(cycles_s, cycles_f)
                            << "Model: " << model_type << ", wl-index: " << i << " / " << n
                            << ",occurrence :" << ++i_errs << " BOTH SHOULD HAVE SAME ERROR :" << std::endl
                            << "model1: " << cycles_s << std::endl
                            << "model2: " << cycles_f << std::endl;
                else {
                    if (cycles_s != cycles_f)
                        ++i_errs;
                }

                if (is_error_code(cycles_s) || is_error_code(cycles_f)) {
                    // both
                    // assume zero all values
                } else {
                    // one with value one out of range
                    // one to be zero, the other to be 1 milion?
                    sum_ff += nan_val * nan_val;
                }

            } else {  // no errors
                int delta_c = std::abs(((int)cycles_f - (int)cycles_s));
                float ratio{100.0F};
                const auto min_cycles = std::min(std::abs((int)cycles_s), std::abs((int)cycles_f));
                if (0 != min_cycles) {
                    ratio = (float)delta_c / min_cycles;
                }

                if (delta_c > min_absolute_delta) {
                    i_abs_errors++;
                    if (silence < 1)
                        EXPECT_LE(ratio, max_ratio_delta)
                                << "Model: " << model_type << ", wl-index: " << i << " / " << n
                                << ",occurrence :" << ++i_errs << " BIG DIFFERENCE :" << std::endl
                                << "model1: " << cycles_s << std::endl
                                << "model2: " << cycles_f << std::endl
                                << "delta: " << delta_c << std::endl
                                << "ratio: " << ratio << std::endl
                                << wl << std::endl;
                    else if (silence < 2)
                        EXPECT_LE(ratio, max_ratio_delta)
                                << "Model: " << model_type << ", wl-index: " << i << " / " << n
                                << ",occurrence :" << ++i_errs << " BIG DIFFERENCE :" << std::endl
                                << "model1: " << cycles_s << std::endl
                                << "model2: " << cycles_f << std::endl
                                << "delta: " << delta_c << std::endl
                                << "ratio: " << ratio << std::endl;
                    else {
                        if (ratio >= max_ratio_delta)
                            ++i_errs;
                    }
                }
                // always compute correlation
                {
                    sum_s += cycles_s;
                    sum_f += cycles_f;
                    sum_sf += ((decltype(sum_sf))cycles_s * cycles_f);

                    sum_ss += ((decltype(sum_sf))cycles_s * cycles_s);
                    sum_ff += ((decltype(sum_sf))cycles_f * cycles_f);
                }
            }
            i++;  // next workload
        }
        const auto expected_failures{std::lround(std::ceil(expected_deviation_ratio * n_workloads))};  // how

        EXPECT_LE(i_errs, expected_failures) << "\n ERRORS when comparing slow with fast: " << i_errs
                                             << " out of workloads count: " << n_workloads << std::endl
                                             << "range/error mismatch: " << i_range_errs << std::endl
                                             << "delta absolute: " << i_abs_errors << std::endl
                                             << "ratio threshold: " << max_ratio_delta << std::endl
                                             << "absolute threshold: " << min_absolute_delta << std::endl
                                             << "expected_deviation_ratio: " << expected_deviation_ratio << std::endl
                                             << " Modelfile 1: " << nn1 << std::endl
                                             << " Modelfile 2: " << nn2 << std::endl;

        {  // compute the correlation
            double nom = n_workloads * sum_sf - (sum_s * sum_f);
            double denom_sq = (n_workloads * sum_ss - (sum_s * sum_s)) * (n_workloads * sum_ff - (sum_f * sum_f));

            double corr = nom / std::sqrt(denom_sq);
            std::cout << "\n CORRELATION : " << corr << "\n";

            result.correlation = corr;
        }

        result.errors_cnt = i_errs;

        return i_errs;
    }

private:
};

/// tests that no model gives DPU cycles outside of range , inputs are random workloads
TEST_F(CostModelStochastic, DISABLED_NoOutOfRangeDPUCycles_fast_2_0_stochastic) {
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
TEST_F(CostModelStochastic, NoOutOfRangeDPUCycles_fast_2_7_stochastic) {
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
TEST_F(CostModelStochastic, DISABLED_NoOutOfRangeDPUCycles_2_0_normal_stochastic) {
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
TEST_F(CostModelStochastic, NoOutOfRangeDPUCycles_2_7_normal_stochastic) {
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
TEST_F(CostModelStochastic, DISABLED_Inference_output_in_Interval_Test_2_0_all_stochastic) {
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
TEST_F(CostModelStochastic, Inference_output_in_Interval_Test_2_7_all_stochastic) {
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

    {  // fast
        const ModelDescriptor& model_info{the_NN_models.fast_model_paths[modelIndex]};

        const auto cnt = CheckInValidInterval_RawInference(model_info, low_threshold, high_threshold, workloads);
        const auto undervalues{std::get<0>(cnt)};
        const auto overvalues{std::get<1>(cnt)};
        EXPECT_LE(undervalues + overvalues, max_deviations_allowed)
                << "\n FAILED count:" << undervalues << " <LOW, " << overvalues << " >HIGH, "
                << "  problems out of all: " << n_workloads << " for model: " << model_info.first << std::endl;
        std::cout << " Fast NN Step DONE !\n";
    }
}

/// tests that no model gives 1.0 as output
TEST_F(CostModelStochastic, DISABLED_NoONE_asInference_output_all_stochastic) {
    unsigned int n_workloads = 1000;

    // nPU2.0 will give negative values

    for (auto& model_info : the_NN_models.all_model_paths) {
        auto cnt = CheckNoCase_InferenceOutput(model_info, 1.0F, n_workloads);
        EXPECT_EQ(cnt, 0) << "\n FAILED count:" << cnt << " problems out of all: " << n_workloads
                          << " for model: " << model_info.first << std::endl;
    }
}

/// Make a fast versus slow statistical comparison.   no big delta expected.
TEST_F(CostModelStochastic, DISABLED_Comparative_fast_vs_slow_20_stochastic) {
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
                                                 modelIndex, model_type, device_version, n_workloads);

    EXPECT_LE(errors, (int)(expected_deviation_ratio * n_workloads) + 1) << expected_deviation_ratio * n_workloads;
}

/// Make a fast versus slow statistical comparison.   no big delta expected.
TEST_F(CostModelStochastic, Comparative_fast_vs_slow_27_stochastic) {
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
                                                 modelIndex, model_type, device_version, n_workloads);

    EXPECT_LE(errors, (int)(expected_deviation_ratio * n_workloads) + 1) << expected_deviation_ratio * n_workloads;
}

/// Make a  statistical comparison.   no big delta expected.
/// This takes a lot of time (minutes) and you need locally the NN files to compare
/// cannot be run on CI, and it is not intended for repetitive CI run.
TEST_F(CostModelStochastic, DISABLED_Comparative_MATRIX_VPUNNs_stochastic) {
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