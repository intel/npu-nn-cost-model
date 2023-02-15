// Copyright © 2022 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#include <vpu/types.h>
#include <vpu_cost_model.h>
#include <iostream>

int main(int argc, char* argv[]) {
    printf("========================================================================\n");
    printf("=========================     VPUNN profiler   =========================\n");

    if (argc < 3) {
        printf("Usage %s <model.vpunn> <output file>\n", argv[0]);
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

    csv << "Workloads/Batch size";
    printf("Start profiling.....\n");

    std::vector<unsigned int> workloads_lst = {1, 5, 20, 50, 100, 200, 300, 500, 1000, 2000, 5000};
    std::vector<unsigned int> batches_lst = {1, 5, 20, 50, 100, 200, 300, 500, 1000, 2000, 5000};

    // Header row
    for (auto batch_size : batches_lst) {
        csv << "," << batch_size;
    }
    csv << std::endl;

    for (auto n_workloads : workloads_lst) {
        printf("\tTesting %u workloads.....\n", n_workloads);
        csv << n_workloads;
        for (auto batch_size : batches_lst) {
            VPUNN::Logger::debug() << "Loading model from " << model_path;
            // Use no cache
            auto model = VPUNN::VPUCostModel(model_path, false, 0, batch_size);

            // printf("Generate %ld random worklodas\n", n_workloads);
            VPUNN::Logger::debug() << "Generate " << n_workloads << " random worklodas";
            auto workloads = std::vector<VPUNN::DPUWorkload>(n_workloads);
            std::generate_n(workloads.begin(), n_workloads, VPUNN::randDPUWorkload(VPUNN::VPUDevice::VPU_2_0));

            auto t1 = VPUNN::tick();
            model.DPU(workloads);
            auto t2 = VPUNN::tock(t1);

            VPUNN::Logger::debug() << "Inference took " << t2 << "ms @ batch " << batch_size;
            csv << "," << t2 / n_workloads;
        }
        csv << std::endl;
    }
    csv.close();
    printf("Done! Profiling CSV written in ./%s\n", argv[2]);

    return 0;
}
