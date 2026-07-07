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
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <memory>
#include <numeric>

#include "nn_models.h"

#include "common_helpers.h"
#include "vpu/sample_generator/random_task_generator.h"
#include "vpu_shave_cost_model.h"

namespace VPUNN_unit_tests {
using namespace VPUNN;

class VPUNNPerformanceTest : public ::testing::Test {
public:
protected:
    const VPUDevice ignore_old_devices{VPUDevice::VPU_2_7};
    void SetUp() override {
    }

    const VPUNNModelsFiles& the_NN_models{VPUNNModelsFiles::getModels()};  ///< the paths to available NN models

    const float requirements_target_latency_ms{70.F /
                                               1000.F};  // 100 microseconds reduced to 80. typ is <70 on gcc/clang on
                                                         // linux. larger on WIndows with MSC++ compiler

#ifdef _WINDOWS
    const float os_overhead_factor{1.8F};  ///< windows has larger overheads
#else
    const float os_overhead_factor{1.0f};  ///< linux/macOS have smaller overheads
#endif

#if defined(_DEBUG) || defined(NO_PROFILING_ALLOWED) || defined(DEBUG)
    const float tolerance_factor_for_debug{20.0F};  ///< big enough to not cause problems when running tests in debug
    const bool time_relevance{false};
    const float target_latency{requirements_target_latency_ms * os_overhead_factor *
                               tolerance_factor_for_debug};  // miliseconds
    const float strict_target_latency{target_latency};       // miliseconds
    const unsigned int population_size{50};                  // keep very short in debug
#else
    const bool time_relevance{true};
    const float target_latency{requirements_target_latency_ms * os_overhead_factor};  // miliseconds
    const float strict_target_latency{requirements_target_latency_ms};                // miliseconds
    const unsigned int population_size{2000};  // put large enough so values are statistically stable
#endif

    void runModelLatency(const VPUNNModelsFiles::ModelDescriptor& model_info, unsigned int n_workloads) {
        const auto& model_path = model_info.first;
        // Use no cache
        const VPUCostModel model{model_path, false,
                                 0 /* no cache*/};  // no cache, so second iteration finds nothing, batch =1

        // Check device
        const VPUNN::VPUDevice device = model_info.second;

        // Generate N workloads
        auto workloads = std::vector<VPUNN::DPUWorkload>(n_workloads);
        std::generate_n(workloads.begin(), n_workloads, VPUNN::randDPUWorkload(device));

        std::cout << std::endl
                  << "** WL Latency Info for " << model_path << "   Target Latency:" << target_latency
                  << " milliseconds,  Samples desired: " << n_workloads  // << std::endl
                  << " Compiled in a time relevant mode (NDEBUG): " << time_relevance << std::endl;
        for (int r = 1; r <= 1; ++r) {  // repeat runs

            {  // separate execution
                std::vector<double> individual_latencies;
                individual_latencies.reserve(n_workloads);
                std::vector<double> individual_latencies_werror;  // with error
                individual_latencies_werror.reserve(n_workloads);
                for (const auto& wl : workloads) {
                    const auto t0 = VPUNN::tick();
                    const auto dpu_result = model.DPU(wl);
                    // Total latency in ms
                    const auto one_latency = VPUNN::tock(t0);
                    if (Cycles::isErrorCode(dpu_result)) {
                        individual_latencies_werror.push_back(one_latency);
                    } else {
                        individual_latencies.push_back(one_latency);
                    }
                }
                {  // min/max,avg
                    const auto latencies_count = individual_latencies.size();
                    const auto min_max = std::minmax_element(begin(individual_latencies), end(individual_latencies));
                    const auto min_lat{*min_max.first};
                    const auto max_lat{*min_max.second};
                    const auto wl_latency =
                            std::accumulate(individual_latencies.begin(), individual_latencies.end(), 0.0) /
                            latencies_count;

                    const auto first = individual_latencies[0];
                    const auto last = individual_latencies[latencies_count - 1];

                    std::sort(individual_latencies.begin(), individual_latencies.end());
                    const auto median = individual_latencies[latencies_count / 2];
                    const auto at10percentile = individual_latencies[(int)(latencies_count * 0.1F)];
                    const auto at90percentile = individual_latencies[(int)(latencies_count * 0.9F)];

                    EXPECT_LE(std::min(wl_latency, median), target_latency)
                            << " WL Latency Info for " << model_path << "   Target Latency[ms]:" << target_latency
                            << std::endl
                            << "   Tex: 1xN: 1 wl avg latency: " << wl_latency << " Test with: " << latencies_count
                            << " sequentially executed. "
                            << "Batch : " << 1 << ". Min: " << min_lat << ", Max: " << max_lat << ", Med: " << median
                            << "\n\t First: " << first << ", Last: " << last << ",  10th%: " << at10percentile
                            << ",  90th%: " << at90percentile << std::endl
                            << "Compiled in a time relevant mode: (NDEBUG)" << time_relevance << std::endl;

                    /// cout anyway
                    std::cout << " >>  Tout: 1xN: 1 wl avg latency [ms]: " << wl_latency
                              << " Test with: " << latencies_count << " sequentially executed. "
                              << "Batch : " << 1 << ". Min: " << min_lat << ", Max: " << max_lat << ", Med: " << median
                              << "\n\t First: " << first << ", Last: " << last << ",  10th%: " << at10percentile
                              << ",  90th%: " << at90percentile << std::endl
                              << "     Compiled in a time relevant mode: (NDEBUG)" << time_relevance << std::endl;
                }
                {
                    auto& latencies{individual_latencies_werror};
                    // min/max,avg
                    const auto latencies_count = latencies.size();
                    if (latencies_count > 0) {
                        const auto min_max = std::minmax_element(begin(latencies), end(latencies));
                        const auto min_lat{*min_max.first};
                        const auto max_lat{*min_max.second};
                        const auto wl_latency =
                                std::accumulate(latencies.begin(), latencies.end(), 0.0) / latencies_count;

                        const auto first = latencies[0];
                        const auto last = latencies[latencies_count - 1];

                        std::sort(latencies.begin(), latencies.end());
                        const auto median = latencies[latencies_count / 2];
                        const auto at10percentile = latencies[(int)(latencies_count * 0.1F)];
                        const auto at90percentile = latencies[(int)(latencies_count * 0.9F)];

                        /// cout anyway for Error latencies
                        std::cout << " >>X  ERROR LATENCY: 1xN: 1 wl avg latency [ms]: " << wl_latency
                                  << " Test with: " << latencies_count << " sequentially executed. "
                                  << "Batch : " << 1 << ". Min: " << min_lat << ", Max: " << max_lat
                                  << ", Med: " << median << "\n\t First: " << first << ", Last: " << last
                                  << ",  10th%: " << at10percentile << ",  90th%: " << at90percentile << std::endl;
                    }
                }
            }
        }
    }

private:
};

// Demonstrate runtime compliance.NOte: disabled because the results are dependent on CPU load
// If another build is done in parallel (CI use case) the runtime will be high
TEST_F(VPUNNPerformanceTest, DISABLED_Standard_InferenceLatency_stochastic) {
    for (const auto& model_info : the_NN_models.profilable_model_paths) {
        if (model_info.second <= ignore_old_devices) {
            continue;  // ignore this because is not in focus for runtime
        }

        runModelLatency(model_info, population_size);

        // do it again
        std::cout << "         ------------------\n";
    }
}
TEST_F(VPUNNPerformanceTest, DISABLED_FAST_InferenceLatencyStrict_stochastic) {
    const unsigned int n_workloads = population_size;

    for (const auto& model_info : the_NN_models.fast_model_paths) {
        if (model_info.second <= ignore_old_devices) {
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

TEST_F(VPUNNPerformanceTest, DISABLED_Runtime_VPUModelInstantiationVsInstanceQueryVsStaticQuery) {
#if defined(_DEBUG) || defined(DEBUG)
    const int iterations = 10;
#else
    const int iterations = 50;
#endif

    const auto& candidates = VPUNNModelsFiles::getModels().profilable_model_paths;
    ASSERT_FALSE(candidates.empty());

    auto selected = candidates.front();
    const auto it_vpu40 =
            std::find_if(candidates.begin(), candidates.end(), [](const VPUNNModelsFiles::ModelDescriptor& item) {
                return item.second == VPUDevice::VPU_4_0;
            });
    if (it_vpu40 != candidates.end()) {
        selected = *it_vpu40;
    }

    const std::string model_path = selected.first;
    const VPUDevice device = selected.second;

    std::vector<double> instantiate_only_us;
    std::vector<double> construct_plus_query_us;
    std::vector<double> instance_query_only_us;
    std::vector<double> static_query_us;
    instantiate_only_us.reserve(iterations);
    construct_plus_query_us.reserve(iterations);
    instance_query_only_us.reserve(iterations);
    static_query_us.reserve(iterations);

    using Clock = std::chrono::high_resolution_clock;

    auto median = [](std::vector<double> values) {
        std::sort(values.begin(), values.end());
        return values[values.size() / 2];
    };

    // Warm-up paths once.
    {
        auto warmup_model = std::make_unique<VPUCostModel>(model_path);
        ASSERT_TRUE(warmup_model->nn_initialized()) << "Failed to initialize NN model from: " << model_path;
        [[maybe_unused]] const auto warmup_ops = warmup_model->getShaveSupportedOperations(device);
    }
    [[maybe_unused]] const auto warmup_static_ops = VPUCostModel::queryDeviceMappedSupportedOperations(device);

    const auto measure_instantiate_only = [&]() {
        const auto t0 = Clock::now();
        auto model = std::make_unique<VPUCostModel>(model_path);
        const auto t1 = Clock::now();
        const bool initialized = model->nn_initialized();
        model.reset();
        return std::make_pair(std::chrono::duration<double, std::micro>(t1 - t0).count(), initialized);
    };

    const auto measure_construct_plus_query = [&]() {
        const auto t0 = Clock::now();
        auto model = std::make_unique<VPUCostModel>(model_path);
        const auto ops = model->getShaveSupportedOperations(device);
        const auto t1 = Clock::now();
        const bool initialized = model->nn_initialized();
        model.reset();
        return std::make_tuple(std::chrono::duration<double, std::micro>(t1 - t0).count(), initialized, !ops.empty());
    };

    // 1) Measure instantiation-only and construct+query under the same runtime conditions.
    for (int i = 0; i < iterations; ++i) {
        if ((i % 2) == 0) {
            const auto inst = measure_instantiate_only();
            const auto cqp = measure_construct_plus_query();

            EXPECT_TRUE(inst.second) << "Failed to initialize NN model from: " << model_path;
            EXPECT_TRUE(std::get<1>(cqp)) << "Failed to initialize NN model from: " << model_path;
            EXPECT_TRUE(std::get<2>(cqp));

            instantiate_only_us.push_back(inst.first);
            construct_plus_query_us.push_back(std::get<0>(cqp));
        } else {
            const auto cqp = measure_construct_plus_query();
            const auto inst = measure_instantiate_only();

            EXPECT_TRUE(inst.second) << "Failed to initialize NN model from: " << model_path;
            EXPECT_TRUE(std::get<1>(cqp)) << "Failed to initialize NN model from: " << model_path;
            EXPECT_TRUE(std::get<2>(cqp));

            instantiate_only_us.push_back(inst.first);
            construct_plus_query_us.push_back(std::get<0>(cqp));
        }
    }

    // 2) Measure getShaveSupportedOperations call only on an already-instantiated model.
    {
        auto model = std::make_unique<VPUCostModel>(model_path);
        ASSERT_TRUE(model->nn_initialized()) << "Failed to initialize NN model from: " << model_path;

        for (int i = 0; i < iterations; ++i) {
            const auto t0 = Clock::now();
            const auto ops = model->getShaveSupportedOperations(device);
            const auto t1 = Clock::now();

            EXPECT_FALSE(ops.empty());
            instance_query_only_us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
        }
    }

    // 3) Measure static lightweight query only.
    for (int i = 0; i < iterations; ++i) {
        const auto t0 = Clock::now();
        const auto ops = VPUCostModel::queryDeviceMappedSupportedOperations(device);
        const auto t1 = Clock::now();

        EXPECT_FALSE(ops.empty());
        static_query_us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
    }

    const double avg_instantiate = std::accumulate(instantiate_only_us.begin(), instantiate_only_us.end(), 0.0) /
                                   static_cast<double>(instantiate_only_us.size());
    const double avg_construct_plus_query =
            std::accumulate(construct_plus_query_us.begin(), construct_plus_query_us.end(), 0.0) /
            static_cast<double>(construct_plus_query_us.size());
    const double avg_instance_query =
            std::accumulate(instance_query_only_us.begin(), instance_query_only_us.end(), 0.0) /
            static_cast<double>(instance_query_only_us.size());
    const double avg_static_query = std::accumulate(static_query_us.begin(), static_query_us.end(), 0.0) /
                                    static_cast<double>(static_query_us.size());

    const double med_instantiate = median(instantiate_only_us);
    const double med_construct_plus_query = median(construct_plus_query_us);
    const double med_instance_query = median(instance_query_only_us);
    const double med_static_query = median(static_query_us);

    const double ratio_instantiate_to_static = (avg_static_query > 0.0) ? (avg_instantiate / avg_static_query) : 0.0;
    const double ratio_construct_plus_query_to_static =
         (avg_static_query > 0.0) ? (avg_construct_plus_query / avg_static_query) : 0.0;
    const double average_improvement_instantiate_to_static =
         (ratio_instantiate_to_static > 0.0) ? (1.0 - 1.0 / ratio_instantiate_to_static) * 100.0 : 0.0;
    const double average_improvement_construct_plus_query_to_static =
         (ratio_construct_plus_query_to_static > 0.0) ? (1.0 - 1.0 / ratio_construct_plus_query_to_static) * 100.0 : 0.0;

    std::cout << "\n[Runtime] VPUCostModel split timing: instantiate vs instance query vs static query"
              << "\n  iterations: " << iterations << "\n  device: " << VPUDevice_ToText.at(static_cast<int>(device))
              << "\n  model path: " << model_path << "\n  avg(instantiate_only)   [us]: " << std::fixed
              << std::setprecision(2) << avg_instantiate
              << "\n  avg(construct+query)    [us]: " << avg_construct_plus_query
              << "\n  avg(instance_query_only)[us]: " << avg_instance_query
              << "\n  avg(static_query_only)  [us]: " << avg_static_query
              << "\n  med(instantiate_only)   [us]: " << med_instantiate
              << "\n  med(construct+query)    [us]: " << med_construct_plus_query
              << "\n  med(instance_query_only)[us]: " << med_instance_query
              << "\n  med(static_query_only)  [us]: " << med_static_query
              << "\n  ratio avg(instantiate/static): " << ratio_instantiate_to_static << "x"
              << "\n  ratio avg(construct+query/static): " << ratio_construct_plus_query_to_static << "x"
              << "\n  average improvement instantiate->static: " << average_improvement_instantiate_to_static << "%"
              << "\n  average improvement construct+query->static: " << average_improvement_construct_plus_query_to_static << "%\n";

    EXPECT_GT(avg_instantiate, 0.0);
    EXPECT_GT(avg_construct_plus_query, 0.0);
    EXPECT_GT(avg_instance_query, 0.0);
    EXPECT_GT(avg_static_query, 0.0);
}

TEST_F(VPUNNPerformanceTest, DISABLED_Runtime_VPUModelConstructPlusQueryVsStaticQuery_MultiDevice) {
#if defined(_DEBUG) || defined(DEBUG)
    const int iterations = 10;
#else
    const int iterations = 50;
#endif

    const auto& candidates = VPUNNModelsFiles::getModels().profilable_model_paths;
    ASSERT_FALSE(candidates.empty());

    using Clock = std::chrono::high_resolution_clock;

    auto median = [](std::vector<double> values) {
        std::sort(values.begin(), values.end());
        return values[values.size() / 2];
    };

    double ratio_inst_sum = 0.0;
    double ratio_cqp_sum = 0.0;
    int ratio_count = 0;

    std::cout << "\n[Runtime] VPUCostModel construct+instance-query vs static query (multi-device)"
              << "\n  iterations/device: " << iterations << "\n";

    for (const auto& selected : candidates) {
        const std::string model_path = selected.first;
        const VPUDevice device = selected.second;

        std::vector<double> instantiate_only_us;
        std::vector<double> construct_plus_query_us;
        std::vector<double> instance_query_only_us;
        std::vector<double> static_query_us;
        instantiate_only_us.reserve(iterations);
        construct_plus_query_us.reserve(iterations);
        instance_query_only_us.reserve(iterations);
        static_query_us.reserve(iterations);

        {
            auto warmup_model = std::make_unique<VPUCostModel>(model_path);
            ASSERT_TRUE(warmup_model->nn_initialized()) << "Failed to initialize NN model from: " << model_path;
            [[maybe_unused]] const auto warmup_ops = warmup_model->getShaveSupportedOperations(device);
        }
        [[maybe_unused]] const auto warmup_static_ops = VPUCostModel::queryDeviceMappedSupportedOperations(device);

        const auto measure_instantiate_only = [&]() {
            const auto t0 = Clock::now();
            auto model = std::make_unique<VPUCostModel>(model_path);
            const auto t1 = Clock::now();
            const bool initialized = model->nn_initialized();
            model.reset();
            return std::make_pair(std::chrono::duration<double, std::micro>(t1 - t0).count(), initialized);
        };

        const auto measure_construct_plus_query = [&]() {
            const auto t0 = Clock::now();
            auto model = std::make_unique<VPUCostModel>(model_path);
            const auto ops = model->getShaveSupportedOperations(device);
            const auto t1 = Clock::now();
            const bool initialized = model->nn_initialized();
            model.reset();
            return std::make_tuple(std::chrono::duration<double, std::micro>(t1 - t0).count(), initialized,
                                   !ops.empty());
        };

        for (int i = 0; i < iterations; ++i) {
            if ((i % 2) == 0) {
                const auto inst = measure_instantiate_only();
                const auto cqp = measure_construct_plus_query();

                EXPECT_TRUE(inst.second) << "Failed to initialize NN model from: " << model_path;
                EXPECT_TRUE(std::get<1>(cqp)) << "Failed to initialize NN model from: " << model_path;
                EXPECT_TRUE(std::get<2>(cqp));

                instantiate_only_us.push_back(inst.first);
                construct_plus_query_us.push_back(std::get<0>(cqp));
            } else {
                const auto cqp = measure_construct_plus_query();
                const auto inst = measure_instantiate_only();

                EXPECT_TRUE(inst.second) << "Failed to initialize NN model from: " << model_path;
                EXPECT_TRUE(std::get<1>(cqp)) << "Failed to initialize NN model from: " << model_path;
                EXPECT_TRUE(std::get<2>(cqp));

                instantiate_only_us.push_back(inst.first);
                construct_plus_query_us.push_back(std::get<0>(cqp));
            }
        }

        {
            auto model = std::make_unique<VPUCostModel>(model_path);
            ASSERT_TRUE(model->nn_initialized()) << "Failed to initialize NN model from: " << model_path;

            for (int i = 0; i < iterations; ++i) {
                const auto t0 = Clock::now();
                const auto ops = model->getShaveSupportedOperations(device);
                const auto t1 = Clock::now();

                EXPECT_FALSE(ops.empty());
                instance_query_only_us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
            }
        }

        for (int i = 0; i < iterations; ++i) {
            const auto t0 = Clock::now();
            const auto ops = VPUCostModel::queryDeviceMappedSupportedOperations(device);
            const auto t1 = Clock::now();

            EXPECT_FALSE(ops.empty());
            static_query_us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
        }

        const double avg_instantiate = std::accumulate(instantiate_only_us.begin(), instantiate_only_us.end(), 0.0) /
                                       static_cast<double>(instantiate_only_us.size());
        const double avg_construct_plus_query =
                std::accumulate(construct_plus_query_us.begin(), construct_plus_query_us.end(), 0.0) /
                static_cast<double>(construct_plus_query_us.size());
        const double avg_instance_query =
                std::accumulate(instance_query_only_us.begin(), instance_query_only_us.end(), 0.0) /
                static_cast<double>(instance_query_only_us.size());
        const double avg_static = std::accumulate(static_query_us.begin(), static_query_us.end(), 0.0) /
                                  static_cast<double>(static_query_us.size());
        const double med_instantiate = median(instantiate_only_us);
        const double med_construct_plus_query = median(construct_plus_query_us);
        const double med_instance_query = median(instance_query_only_us);
        const double med_static = median(static_query_us);
        const double ratio_instantiate_to_static = (avg_static > 0.0) ? (avg_instantiate / avg_static) : 0.0;
        const double ratio_construct_plus_query_to_static =
                (avg_static > 0.0) ? (avg_construct_plus_query / avg_static) : 0.0;
        const double average_improvement_instantiate_to_static =
                (ratio_instantiate_to_static > 0.0) ? (1.0 - 1.0 / ratio_instantiate_to_static) * 100.0 : 0.0;
        const double average_improvement_construct_plus_query_to_static =
                (ratio_construct_plus_query_to_static > 0.0) ? (1.0 - 1.0 / ratio_construct_plus_query_to_static) * 100.0 : 0.0;

        std::cout << "\n[Runtime] VPUCostModel split timing: instantiate vs instance query vs static query"
                  << "\n  iterations: " << iterations
                  << "\n  device: " << VPUDevice_ToText.at(static_cast<int>(device))
                  << "\n  model path: " << model_path
                  << "\n  avg(instantiate_only)   [us]: " << std::fixed << std::setprecision(2) << avg_instantiate
                  << "\n  avg(construct+query)    [us]: " << avg_construct_plus_query
                  << "\n  avg(instance_query_only)[us]: " << avg_instance_query
                  << "\n  avg(static_query_only)  [us]: " << avg_static
                  << "\n  med(instantiate_only)   [us]: " << med_instantiate
                  << "\n  med(construct+query)    [us]: " << med_construct_plus_query
                  << "\n  med(instance_query_only)[us]: " << med_instance_query
                  << "\n  med(static_query_only)  [us]: " << med_static
                  << "\n  ratio avg(instantiate/static): " << ratio_instantiate_to_static << "x"
                  << "\n  ratio avg(construct+query/static): " << ratio_construct_plus_query_to_static << "x"
                  << "\n  average improvement instantiate->static: " << average_improvement_instantiate_to_static << "%"
                  << "\n  average improvement construct+query->static: " << average_improvement_construct_plus_query_to_static << "%\n";

        EXPECT_GT(avg_instantiate, 0.0);
        EXPECT_GT(avg_construct_plus_query, 0.0);
        EXPECT_GT(avg_instance_query, 0.0);
        EXPECT_GT(avg_static, 0.0);
        if ((ratio_instantiate_to_static > 0.0) && (ratio_construct_plus_query_to_static > 0.0)) {
            ratio_inst_sum += ratio_instantiate_to_static;
            ratio_cqp_sum += ratio_construct_plus_query_to_static;
            ++ratio_count;
        }
    }

    const double avg_ratio_inst = (ratio_count > 0) ? (ratio_inst_sum / static_cast<double>(ratio_count)) : 0.0;
    const double avg_ratio_cqp = (ratio_count > 0) ? (ratio_cqp_sum / static_cast<double>(ratio_count)) : 0.0;
    std::cout << "\n  Summary: average ratio instantiate/static across devices = " << std::fixed
              << std::setprecision(2) << avg_ratio_inst << "x"
              << "\n  Summary: average ratio construct+query/static across devices = " << avg_ratio_cqp << "x\n";
}

TEST_F(VPUNNPerformanceTest, DISABLED_Runtime_NPU5_ConstructPlusQueryVsStaticQuery_ThreeInitModes) {
#if defined(_DEBUG) || defined(DEBUG)
    const int iterations = 10;
#else
    const int iterations = 50;
#endif

    const std::string model_path{NPU_5_0_MODEL_PATH};
    const VPUDevice device{VPUDevice::NPU_5_0};
    const std::string cache_file_default{(std::filesystem::path{model_path}).replace_extension("cachebin").string()};

    if (!std::filesystem::exists(cache_file_default)) {
        GTEST_SKIP() << "Required cache file not found for NPU5 benchmark: " << cache_file_default;
    }

    const auto read_binary_file = [](const std::string& filename) {
        std::ifstream stream(filename, std::ios::binary);
        if (!stream) {
            return std::vector<char>{};
        }
        return std::vector<char>(std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>());
    };

    const auto cache_buffer = read_binary_file(cache_file_default);
    const auto model_buffer = read_binary_file(model_path);
    ASSERT_FALSE(cache_buffer.empty()) << "Failed to read cache file: " << cache_file_default;
    ASSERT_FALSE(model_buffer.empty()) << "Failed to read model file: " << model_path;

    using Clock = std::chrono::high_resolution_clock;

    auto median = [](std::vector<double> values) {
        std::sort(values.begin(), values.end());
        return values[values.size() / 2];
    };

    using MeasureResult = std::tuple<double, bool, bool>;
    const auto run_mode = [&](const std::string& mode_name,
                              const std::function<MeasureResult()>& measure_construct_plus_query) -> double {
        std::vector<double> construct_plus_query_us;
        std::vector<double> static_query_us;
        construct_plus_query_us.reserve(iterations);
        static_query_us.reserve(iterations);

        const auto warmup_cqp = measure_construct_plus_query();
        EXPECT_TRUE(std::get<1>(warmup_cqp)) << "Failed to initialize NPU5 model in mode: " << mode_name;
        EXPECT_TRUE(std::get<2>(warmup_cqp)) << "Empty SHAVE ops for NPU5 in mode: " << mode_name;
        if (!std::get<1>(warmup_cqp) || !std::get<2>(warmup_cqp)) {
            return 0.0;
        }
        [[maybe_unused]] const auto warmup_static_ops = VPUCostModel::queryDeviceMappedSupportedOperations(device);

        const auto measure_static_query = [&]() {
            const auto t0 = Clock::now();
            const auto ops = VPUCostModel::queryDeviceMappedSupportedOperations(device);
            const auto t1 = Clock::now();
            return std::make_pair(std::chrono::duration<double, std::micro>(t1 - t0).count(), !ops.empty());
        };

        for (int i = 0; i < iterations; ++i) {
            if ((i % 2) == 0) {
                const auto cqp = measure_construct_plus_query();
                const auto stat = measure_static_query();

                EXPECT_TRUE(std::get<1>(cqp)) << "Failed to initialize NPU5 model in mode: " << mode_name;
                EXPECT_TRUE(std::get<2>(cqp)) << "Empty SHAVE ops for NPU5 in mode: " << mode_name;
                EXPECT_TRUE(stat.second);

                construct_plus_query_us.push_back(std::get<0>(cqp));
                static_query_us.push_back(stat.first);
            } else {
                const auto stat = measure_static_query();
                const auto cqp = measure_construct_plus_query();

                EXPECT_TRUE(std::get<1>(cqp)) << "Failed to initialize NPU5 model in mode: " << mode_name;
                EXPECT_TRUE(std::get<2>(cqp)) << "Empty SHAVE ops for NPU5 in mode: " << mode_name;
                EXPECT_TRUE(stat.second);

                construct_plus_query_us.push_back(std::get<0>(cqp));
                static_query_us.push_back(stat.first);
            }
        }

        const double avg_cqp = std::accumulate(construct_plus_query_us.begin(), construct_plus_query_us.end(), 0.0) /
                               static_cast<double>(construct_plus_query_us.size());
        const double avg_static = std::accumulate(static_query_us.begin(), static_query_us.end(), 0.0) /
                                  static_cast<double>(static_query_us.size());
        const double med_cqp = median(construct_plus_query_us);
        const double med_static = median(static_query_us);
        const double ratio = (avg_static > 0.0) ? (avg_cqp / avg_static) : 0.0;
        const double average_improvement = (ratio > 0.0) ? (1.0 - 1.0 / ratio) * 100.0 : 0.0;

        std::cout << "\n[Runtime][NPU5] Construct+query vs static query"
                  << "\n  mode: " << mode_name
                  << "\n  iterations: " << iterations
                  << "\n  device: " << VPUDevice_ToText.at(static_cast<int>(device))
                  << "\n  model path: " << model_path
                  << "\n  avg(construct+query) [us]: " << std::fixed << std::setprecision(2) << avg_cqp
                  << "\n  avg(static_query)    [us]: " << avg_static
                  << "\n  med(construct+query) [us]: " << med_cqp
                  << "\n  med(static_query)    [us]: " << med_static
                  << "\n  ratio avg(construct+query/static): " << ratio << "x"
                  << "\n  average improvement construct+query->static: " << average_improvement << "%\n";

        EXPECT_GT(avg_cqp, 0.0);
        EXPECT_GT(avg_static, 0.0);
        if (!(avg_cqp > 0.0) || !(avg_static > 0.0)) {
            return 0.0;
        }

        return ratio;
    };

    const auto measure_no_cache = [&]() -> MeasureResult {
        const auto t0 = Clock::now();
        auto model = std::make_unique<VPUCostModel>(model_path, false, 0, 1, "");
        const auto ops = model->getShaveSupportedOperations(device);
        const auto t1 = Clock::now();
        const bool initialized = model->nn_initialized();
        model.reset();
        return std::make_tuple(std::chrono::duration<double, std::micro>(t1 - t0).count(), initialized, !ops.empty());
    };

    const auto measure_cache_file = [&]() -> MeasureResult {
        const auto t0 = Clock::now();
        auto model = std::make_unique<VPUCostModel>(model_path, false, 16384, 1, cache_file_default);
        const auto ops = model->getShaveSupportedOperations(device);
        const auto t1 = Clock::now();
        const bool initialized = model->nn_initialized();
        model.reset();
        return std::make_tuple(std::chrono::duration<double, std::micro>(t1 - t0).count(), initialized, !ops.empty());
    };

    const auto measure_cache_memory = [&]() -> MeasureResult {
        const auto t0 = Clock::now();
        auto model = std::make_unique<VPUCostModel>(model_buffer.data(), model_buffer.size(), false, false, 16384, 1,
                                                    cache_buffer.data(), cache_buffer.size());
        const auto ops = model->getShaveSupportedOperations(device);
        const auto t1 = Clock::now();
        const bool initialized = model->nn_initialized();
        model.reset();
        return std::make_tuple(std::chrono::duration<double, std::micro>(t1 - t0).count(), initialized, !ops.empty());
    };

    const double ratio_no_cache = run_mode("no cache", measure_no_cache);
    const double ratio_cache_file = run_mode("cache from file", measure_cache_file);
    const double ratio_cache_memory = run_mode("cache from memory", measure_cache_memory);

    std::cout << "\n[Runtime][NPU5] Summary (construct+query/static)"
              << "\n  no cache: " << std::fixed << std::setprecision(2) << ratio_no_cache << "x"
              << "\n  cache from file: " << ratio_cache_file << "x"
              << "\n  cache from memory: " << ratio_cache_memory << "x\n";
}

// Parameterized test fixture using ModelDescriptor as parameter
class InferenceLatencyTest :
        public VPUNNPerformanceTest,
        public testing::WithParamInterface<VPUNNModelsFiles::ModelDescriptor> {
protected:
    void SetUp() override {
    }
};

// Parameterized test - one test per model
TEST_P(InferenceLatencyTest, DPU_stochastic) {
    const auto& model_info = GetParam();
    if (model_info.second <= ignore_old_devices) {
        // GTEST_SKIP() << "Skipping old device";
    } else {
        runModelLatency(model_info, population_size);
    }
}

// Instantiate one test per model in profilable_model_paths
INSTANTIATE_TEST_SUITE_P(Performance, InferenceLatencyTest,
                         ::testing::ValuesIn(VPUNNModelsFiles::getModels().profilable_model_paths),
                         [](const ::testing::TestParamInfo<VPUNNModelsFiles::ModelDescriptor>& info) {
                             // Create valid test name from model path
                             const VPUDevice device = info.param.second;
                             std::string name = VPUDevice_ToText.at(static_cast<int>(device));
                             for (char& c : name) {
                                 if (!std::isalnum(c))
                                     c = '_';
                             }
                             return name;
                         });

}  // namespace VPUNN_unit_tests
