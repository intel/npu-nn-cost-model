// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#include <gtest/gtest.h>
#include "common_helpers.h"
#include "vpu/sample_generator/random_task_generator.h"
#include "vpu_cost_model.h"

#include <algorithm>

namespace VPUNN_unit_tests {
using namespace VPUNN;

class VPUNNPerformanceTest : public ::testing::Test {
public:
protected:
    const VPUDevice ignnore_old_devices{VPUDevice::VPU_2_0};
    void SetUp() override {
    }

    const VPUNNModelsFiles& the_NN_models{VPUNNModelsFiles::getModels()};  ///< the paths to available NN models

    const float requirements_target_latency{100.F / 1000.F};  // 100 microseconds
    const float tolerance_factor_for_debug{20.0F};  ///< big enough to not cause problems when running tests in debug
#if defined(_DEBUG) || defined(NO_PROFILING_ALLOWED) || defined(DEBUG)
    const bool time_relevance{false};
    const float target_latency{requirements_target_latency * tolerance_factor_for_debug};  ///< miliseconds
    const float strict_target_latency{target_latency};                                     ///< miliseconds
#else
    const bool time_relevance{true};
    const float target_latency{requirements_target_latency};         ///< miliseconds
    const float strict_target_latency{requirements_target_latency};  ///< miliseconds
#endif

private:
};

// Demonstrate runtime compliance.NOte: disabled because the results are dependent on CPU load
// If another build is done in parallel (CI use case) the runtime will be high
TEST_F(VPUNNPerformanceTest, Standard_InferenceLatency_stochastic) {
    unsigned int n_workloads = 1000;
    // EXPECT_TRUE(false);

    for (auto& model_info : the_NN_models.standard_model_paths) {
        if (model_info.second <= ignnore_old_devices) {
            continue;  // ignore this because is not in focus for runtime
        }
        const auto& model_path = model_info.first;
        // Use no cache
        VPUNN::VPUCostModel model{model_path, false, 0};  // no cache, batch =1

        // Check device
        const VPUNN::VPUDevice device = model_info.second;

        // Generate N workloads
        auto workloads = std::vector<VPUNN::DPUWorkload>(n_workloads);
        std::generate_n(workloads.begin(), n_workloads, VPUNN::randDPUWorkload(device));

        std::cout << std::endl
                  << "** WL Latency Info for " << model_path << "   Target Latency:" << target_latency
                  << " milliseconds" << std::endl
                  << "Compiled in a time relevant mode (NDEBUG): " << time_relevance << std::endl;
        for (int r = 1; r <= 1; ++r) {  // repeat runs

            {  // separate execution
                std::vector<double> individual_latencies;
                individual_latencies.reserve(n_workloads);
                for (const auto& wl : workloads) {
                    const auto t0 = VPUNN::tick();
                    model.DPU(wl);
                    // Total latency in ms
                    const auto one_latency = VPUNN::tock(t0);
                    individual_latencies.push_back(one_latency);
                }
                // min/max,avg
                const auto min_max = std::minmax_element(begin(individual_latencies), end(individual_latencies));
                const auto min_lat{*min_max.first};
                const auto max_lat{*min_max.second};
                const auto wl_latency = std::accumulate(individual_latencies.begin(), individual_latencies.end(), 0.0) /
                                        individual_latencies.size();

                const auto first = individual_latencies[0];
                const auto last = individual_latencies[individual_latencies.size() - 1];

                std::sort(individual_latencies.begin(), individual_latencies.end());
                const auto median = individual_latencies[individual_latencies.size() / 2];
                const auto at10percentile = individual_latencies[(int)(individual_latencies.size() * 0.1F)];
                const auto at90percentile = individual_latencies[(int)(individual_latencies.size() * 0.9F)];

                std::cout << "   T: 1xN: 1 wl avg latency [ms]: " << wl_latency << " Test with: " << n_workloads
                          << " sequentially executed. "
                          << "Batch : " << 1 << ". Min: " << min_lat << ", Max: " << max_lat << ", Med: " << median
                          << "\n\t First: " << first << ", Last: " << last << ",  10th%: " << at10percentile
                          << ",  90th%: " << at90percentile << std::endl
                          << "Compiled in a time relevant mode: (NDEBUG)" << time_relevance << std::endl;
                EXPECT_LE(wl_latency, target_latency)
                        << " WL Latency Info for " << model_path << "   Target Latency[ms]:" << target_latency
                        << std::endl
                        << "   T: 1xN: 1 wl avg latency: " << wl_latency << " Test with: " << n_workloads
                        << " sequentially executed. "
                        << "Batch : " << 1 << ". Min: " << min_lat << ", Max: " << max_lat << ", Med: " << median
                        << "\n\t First: " << first << ", Last: " << last << ",  10th%: " << at10percentile
                        << ",  90th%: " << at90percentile << std::endl
                        << "Compiled in a time relevant mode: (NDEBUG)" << time_relevance << std::endl;
            }

            // do it again
            std::cout << "\n";
        }
    }
}
TEST_F(VPUNNPerformanceTest, FAST_InferenceLatencyStrict_stochastic) {
    unsigned int n_workloads = 1000;

    for (auto& model_info : the_NN_models.fast_model_paths) {
        if (model_info.second <= ignnore_old_devices) {
            continue;  // ignore this because is not in focus for runtime
        }
        const auto& model_path = model_info.first;
        // Use no cache
        VPUNN::VPUCostModel model{model_path, false, 0};  // no cache, batch =1

        // Check device
        const VPUNN::VPUDevice device = model_info.second;

        // Generate N workloads
        auto workloads = std::vector<VPUNN::DPUWorkload>(n_workloads);
        std::generate_n(workloads.begin(), n_workloads, VPUNN::randDPUWorkload(device));

        std::cout << std::endl
                  << "** WL Latency Info for " << model_path << "   Target Latency[ms]:" << strict_target_latency
                  << std::endl
                  << "Compiled in a time relevant mode: (NDEBUG)" << time_relevance << std::endl;
        for (int r = 1; r <= 1; ++r) {  // repeat runs

            {  // separate execution
                std::vector<double> individual_latencies;
                individual_latencies.reserve(n_workloads);
                for (const auto& wl : workloads) {
                    const auto t0 = VPUNN::tick();
                    model.DPU(wl);
                    // Total latency in ms
                    const auto one_latency = VPUNN::tock(t0);
                    individual_latencies.push_back(one_latency);
                }
                // min/max,avg
                const auto min_max = std::minmax_element(begin(individual_latencies), end(individual_latencies));
                const auto min_lat{*min_max.first};
                const auto max_lat{*min_max.second};
                const auto wl_latency = std::accumulate(individual_latencies.begin(), individual_latencies.end(), 0.0) /
                                        individual_latencies.size();

                const auto first = individual_latencies[0];
                const auto last = individual_latencies[individual_latencies.size() - 1];

                std::sort(individual_latencies.begin(), individual_latencies.end());
                const auto median = individual_latencies[individual_latencies.size() / 2];
                const auto at10percentile = individual_latencies[(int)(individual_latencies.size() * 0.1F)];
                const auto at90percentile = individual_latencies[(int)(individual_latencies.size() * 0.9F)];

                std::cout << "   T: 1xN: 1 wl avg latency[ms]: " << wl_latency << " Test with: " << n_workloads
                          << " sequentially executed. "
                          << "Batch : " << 1 << ". Min: " << min_lat << ", Max: " << max_lat << ", Med: " << median
                          << "\n\t First: " << first << ", Last: " << last << ",  10th%: " << at10percentile
                          << ",  90th%: " << at90percentile << std::endl
                          << "Compiled in a time relevant mode: (NDEBUG)" << time_relevance << std::endl;

                EXPECT_LE(wl_latency, strict_target_latency)
                        << " WL Latency Info for " << model_path << "   Target Latency[ms]:" << strict_target_latency
                        << std::endl
                        << "   T: 1xN: 1 wl avg latency: " << wl_latency << " Test with: " << n_workloads
                        << " sequentially executed. "
                        << "Batch : " << 1 << ". Min: " << min_lat << ", Max: " << max_lat << ", Med: " << median
                        << "\n\t First: " << first << ", Last: " << last << ",  10th%: " << at10percentile
                        << ",  90th%: " << at90percentile << std::endl
                        << "Compiled in a time relevant mode: (NDEBUG)" << time_relevance << std::endl;
            }

            // do it again
            std::cout << "\n";
        }
    }
}

}  // namespace VPUNN_unit_tests
