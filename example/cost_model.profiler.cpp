// Copyright © 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#include <vpu/sample_generator/random_task_generator.h>
#include <vpu/types.h>
#include <vpu_cost_model.h>
#include <iostream>

constexpr char usage_message[]{" < model.vpunn > <output file> <device_optional {VPU_2_0,VPU_2_7}>"};
constexpr int minumum_argc{3};
constexpr int optional_argc{2};

// constexpr std::array<unsigned int, 11> workloads_lst{1, 5, 20, 50, 100, 200, 300, 500, 1000, 2000, 5000};
// constexpr std::array<unsigned int, 11> batches_lst{1, 5, 20, 50, 100, 200, 300, 500, 1000, 2000, 5000};

const std::vector<unsigned int> workloads_lst{1, 5, 43, 100, 500, 5000};
const std::vector<unsigned int> batches_lst{1, 5, 50, 100, 500, 2000};

using ResultMatrix = std::map<std::pair<unsigned int, unsigned int>, double>;
template <typename W, typename B>
void print_results_batchOnRows(const W& w_list, const B& b_list, std::ostream& out, ResultMatrix& res) {
    out << "Workloads/Batch size";
    // Header row
    for (auto batch_size : b_list) {
        out << "," << batch_size;
    }
    out << std::endl;

    for (auto n_workloads : w_list) {
        out << n_workloads;
        for (auto batch_size : b_list) {
            out << "," << res[{n_workloads, batch_size}] / n_workloads;
        }
        out << std::endl;
    }
}

struct WL_Stats {
    double averarage_latency;
    double min_lat;
    double max_lat;

    double first;
    double last;

    double median;
    double at10percentile;
    double at90percentile;
};

void stepDPU(std::string name, const std::string& model_path, const VPUNN::VPUDevice device, std::ofstream& csv,
             unsigned int cache = 0) {
    // STEP 2: do one workload profiling for DPU method
    constexpr unsigned int n_workloads = 1000;
    constexpr int repetitions = 3;
    const unsigned int cache_size = cache;
    printf("\n\n STEP 2%s, DPU: Batch 1, cache: %d ,  detailed workloads profiling  ..... %d workloads\n", name.c_str(),
           cache_size, n_workloads);
    csv << std::endl
        << std::endl
        << "Step 2" << name << ", DPU, Batch 1, Workload fine details (wls = " << n_workloads
        << " ) cache size: " << cache_size << std::endl;

    std::vector<double> all_run_at_once;
    std::vector<WL_Stats> individual_run;

    {                                                              // Use no cache
        VPUNN::VPUCostModel model{model_path, false, cache_size};  // no cache, batch =1

        // Generate N workloads
        auto workloads = std::vector<VPUNN::DPUWorkload>(n_workloads);
        std::generate_n(workloads.begin(), n_workloads, VPUNN::randDPUWorkload(device));

        for (int r = 1; r <= repetitions; ++r) {  // repeat runs
            {                                     // Start inference
                const auto t0 = VPUNN::tick();
                model.DPU(workloads);  // all at once
                // Total latency in ms
                const auto total_latency = VPUNN::tock(t0);

                // Workload average latency in ms
                auto wl_latency = total_latency / n_workloads;

                std::cout << "   T: Nx1: 1 wl latency: " << wl_latency << " all workloads executed once." << std::endl;

                all_run_at_once.push_back(wl_latency);
            }
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

                WL_Stats measured_stats{wl_latency, min_lat, max_lat,        first,
                                        last,       median,  at10percentile, at90percentile};

                std::cout << "   T: 1xN: 1 wl avg latency: " << wl_latency << " sequentially executed. "
                          << ". Min: " << min_lat << ", Med: " << median << std::endl;

                individual_run.push_back(measured_stats);
            }

            // do it again
            std::cout << "\n";
        }
    }
    {  // output
        auto& out = csv;
        out << "Run/what, avg_multiple, ,avg_individual, minimum, median,maximum, 10thpercentile, "
               "90thpercentile,first,last \n";
        // Header row
        for (auto i = 0U; i < repetitions; ++i) {
            out << i << ",";
            out << all_run_at_once[i] << ",";

            out << " ,";

            const auto& stat = individual_run[i];
            out << stat.averarage_latency << ",";
            out << stat.min_lat << ",";
            out << stat.median << ",";
            out << stat.max_lat << ",";
            out << stat.at10percentile << ",";
            out << stat.at90percentile << ",";
            out << stat.first << ",";
            out << stat.last << ",";

            out << std::endl;
        }
        out << std::endl;
    }
    // step 2
}

void stepDPUINFO(std::string name, const std::string& model_path, const VPUNN::VPUDevice device, std::ofstream& csv,
                 unsigned int cache = 0) {
    // STEP 3: do one workload profiling for DPUInfo method
    constexpr unsigned int n_workloads = 1000;
    constexpr int repetitions = 4;
    const unsigned int cache_size = cache;
    printf("\n\n STEP 3%s: DPUInfo : Batch 1, cache: %d ,  detailed workloads profiling  ..... %d workloads\n",
           name.c_str(), cache_size, n_workloads);
    csv << std::endl
        << std::endl
        << "Step3" << name << ", DPUInfo,  Batch 1, Workload fine details (wls = " << n_workloads
        << " ) cache size: " << cache_size << std::endl;

    // std::vector<double> all_run_at_once;
    std::vector<WL_Stats> individual_run;

    {
        VPUNN::VPUCostModel model{model_path, false, cache_size};  // default cache, batch =1

        // Generate N workloads
        auto workloads = std::vector<VPUNN::DPUWorkload>(n_workloads);
        std::generate_n(workloads.begin(), n_workloads, VPUNN::randDPUWorkload(device));

        for (int r = 1; r <= repetitions; ++r) {  // repeat runs
            {                                     // separate execution
                std::vector<double> individual_latencies;
                individual_latencies.reserve(n_workloads);
                for (const auto& wl : workloads) {
                    const auto t0 = VPUNN::tick();
                    model.DPUInfo(wl);
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

                WL_Stats measured_stats{wl_latency, min_lat, max_lat,        first,
                                        last,       median,  at10percentile, at90percentile};

                std::cout << "   T: 1xN: 1 wl avg latency: " << wl_latency << " sequentially executed. "
                          << ". Min: " << min_lat << ", Med: " << median << std::endl;

                individual_run.push_back(measured_stats);
            }

            // do it again
            std::cout << "\n";
        }
    }
    {  // output
        auto& out = csv;
        out << "Run/what, avg_multiple, ,avg_individual, minimum, median,maximum, 10thpercentile, "
               "90thpercentile,first,last \n";
        // Header row
        for (auto i = 0U; i < repetitions; ++i) {
            out << i << ",";
            out << "NA"
                << ",";

            out << " ,";

            const auto& stat = individual_run[i];
            out << stat.averarage_latency << ",";
            out << stat.min_lat << ",";
            out << stat.median << ",";
            out << stat.max_lat << ",";
            out << stat.at10percentile << ",";
            out << stat.at90percentile << ",";
            out << stat.first << ",";
            out << stat.last << ",";

            out << std::endl;
        }
        out << std::endl;
    }
}

int main(int argc, char* argv[]) {
    try {
        printf("========================================================================\n");
        printf("=========================     VPUNN profiler   =========================\n");

        if (argc < minumum_argc) {
            printf("Not enough parameters\n");
            printf("Usage %s %s\n", argv[0], usage_message);
            return 0;
        }
        std::string model_path = std::string(argv[1]);

        // Open CSV file
        auto csv = std::ofstream(argv[2]);
        // use appropriate location if you are using MacOS or Linux
        if (csv.fail()) {
            printf("Error! Impossible to open file %s\n", argv[2]);
            exit(-1);
        }

        if (argc > minumum_argc + optional_argc) {
            printf("Too many parameters\n");
            printf("Usage %s %s\n", argv[0], usage_message);
            return 0;
        }
        const std::string desired_device{(argc >= minumum_argc + 1) ? argv[3] : ""};
        const std::string skip_option{(argc >= minumum_argc + 2) ? argv[4] : ""};
        const bool skip_step1{skip_option == "-skip1" ? true : false};

        std::cout << "Device desired is .... " << desired_device << "\n";

        // Check device
        const VPUNN::VPUDevice device =
                (desired_device == "VPU_2_7") ? VPUNN::VPUDevice::VPU_2_7 : VPUNN::VPUDevice::VPU_2_0;

        std::cout << "Device used  is .... " << VPUNN::VPUDevice_ToText.at(static_cast<int>(device)) << "\n";

        printf("Start profiling.....\n");
        std::cout << "Skip step1 .... " << (skip_step1 ? "yes" : "no") << "\n";
        const auto app_time = VPUNN::tick();
        if (!skip_step1) {  // batch workload matrix
            printf(" STEP 1 ,DPU: Batch X Workloads matrix .....\n");
            ResultMatrix result;
            ResultMatrix result2ndRun;

            for (auto batch_size : batches_lst) {
                VPUNN::Logger::debug() << "Loading model from " << model_path;
                // Use no cache
                VPUNN::VPUCostModel model{model_path, false, 0, batch_size};

                printf("\tTesting %u batchSize.....wl:", batch_size);

                for (auto n_workloads : workloads_lst) {
                    VPUNN::Logger::debug() << "Generate " << n_workloads << " random workloads";

                    printf(" %d ,", n_workloads);

                    auto workloads = std::vector<VPUNN::DPUWorkload>(n_workloads);
                    std::generate_n(workloads.begin(), n_workloads, VPUNN::randDPUWorkload(device));

                    auto t1 = VPUNN::tick();
                    model.DPU(std::move(workloads));
                    auto t2 = VPUNN::tock(t1);

                    VPUNN::Logger::debug() << "Inference took " << t2 << "ms @ batch: " << batch_size
                                           << " , workloads: " << n_workloads;
                    result[{n_workloads, batch_size}] = t2;
                }
                printf(" \t wl2:");

                for (auto n_workloads : workloads_lst) {
                    VPUNN::Logger::debug() << "Generate " << n_workloads << " random workloads";
                    printf(" %d ,", n_workloads);
                    auto workloads{std::vector<VPUNN::DPUWorkload>(n_workloads)};
                    std::generate_n(workloads.begin(), n_workloads, VPUNN::randDPUWorkload(device));

                    auto t1 = VPUNN::tick();
                    model.DPU(std::move(workloads));
                    auto t2 = VPUNN::tock(t1);

                    VPUNN::Logger::debug() << "Inference took " << t2 << "ms @ batch: " << batch_size
                                           << " , workloads: " << n_workloads;
                    result2ndRun[{n_workloads, batch_size}] = t2;
                }
                printf("....OK \n");
            }

            print_results_batchOnRows(workloads_lst, batches_lst, csv, result);

            csv << std::endl << std::endl << "Second run:" << std::endl;

            print_results_batchOnRows(workloads_lst, batches_lst, csv, result2ndRun);
        }

        // step 2
        stepDPU("a No cache", model_path, device, csv, 0);
        stepDPU("b With cache", model_path, device, csv, 16384);

        // step 3
        stepDPUINFO("a No cache", model_path, device, csv, 0);
        stepDPUINFO("b With cache", model_path, device, csv, 16384);

        csv << "\n Device used  was .... " << VPUNN::VPUDevice_ToText.at(static_cast<int>(device)) << "\n";

        csv.close();
        printf("Done! Profiling CSV written in ./%s\n", argv[2]);
        const auto full_time = VPUNN::tock(app_time);
        printf("Total time: %d seconds", (int)(full_time / 1000));
    } catch (const std::exception& e) {
        std::cout << "[ERROR]: An exception was caught in main!\n"
                  << " Original exception: " << e.what() << std::endl;

        printf("Usage %s %s\n", argv[0], usage_message);

        return 1;
    }

    return 0;
}
