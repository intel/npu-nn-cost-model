// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_UT_COST_MODEL_STATISTICAL_H
#define VPUNN_UT_COST_MODEL_STATISTICAL_H

#include "vpu_cost_model.h"

#include <gtest/gtest.h>
#include "common/common_helpers.h"
#include "vpu/compatibility/types01.h"
#include "vpu/sample_generator/random_task_generator.h"

#include <unordered_map>

namespace VPUNN_unit_tests {
using namespace VPUNN;

class CostModelStochastic : public ::testing::Test {
protected:
    VPUNN::VPUCostModel model{};
    using ModelDescriptor =
            VPUNNModelsFiles::ModelDescriptor;  ///< make this type directly visible inside of this class
    const VPUNNModelsFiles& the_NN_models{VPUNNModelsFiles::getModels()};  ///< the paths to available NN models

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
        const auto checked_result{Cycles::toCycleInterfaceType(not_valid_value)};
        // const float delta_error = 0.01F;

        VPUNN::VPUCostModel current_model{model_path, false, 0U};  // with cache, batch =1
        const VPUNN::VPUDevice device = model_info.second;

        // Generate N workloads
        auto workloads = std::vector<VPUNN::DPUWorkload>(n_workloads);
        std::generate_n(workloads.begin(), n_workloads, VPUNN::randDPUWorkload(device));

        int i{0};
        for (auto& wl : workloads) {
            CyclesInterfaceType infered_value = current_model.DPU(wl);
            EXPECT_GT(infered_value, checked_result)
                    << "result: " << infered_value << " i:" << ++i << " Model: " << model_path << "\n"
                    << wl << std::endl;
        }

        return i;
    }

      std::tuple<int, int> CheckInValidInterval_RawInference(const ModelDescriptor& model_info,
                                                           const float low_threshold_, const float high_threshold_,
                                                           const std::vector<VPUNN::DPUWorkload>& workloads) {
        const auto& model_path = model_info.first;
        VPUNN::VPUCostModel current_model{model_path, false};  // with cache, batch =1

        int ilow{0};   // increments at every error
        int ihigh{0};  // increments at every error
        int i{0};      // index
        const auto n{workloads.size()};

        const auto low_threshold{Cycles::toCycleInterfaceType(low_threshold_)};
        const auto high_threshold{Cycles::toCycleInterfaceType(high_threshold_)};
        for (const auto& wl : workloads) {
            auto inference = current_model.DPU(wl);

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
                      << "\t,  Count #:" << op.second << " , Count %: " << err_percent << " %";
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


};

}

#endif
