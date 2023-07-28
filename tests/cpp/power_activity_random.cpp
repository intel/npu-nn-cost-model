// Copyright © 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the ?Software Package?)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the ?third-party-programs.txt? or other similarly-named text file included with the
// Software Package for additional details.

#include <gtest/gtest.h>
#include "core/logger.h"
#include "vpu/power.h"
#include "vpu/types.h"
#include "vpu_cost_model.h"

static auto model = VPUNN::VPUCostModel();  // a very empty model

VPUNN::DPUWorkload generate_helper_layer(const unsigned int dim, const unsigned int channels,
                                         VPUNN::Operation operation, VPUNN::VPUDevice device, VPUNN::DataType data) {
    VPUNN::DPUWorkload wl;

    switch (device) {
    case VPUNN::VPUDevice::VPU_2_0:
        wl.device = VPUNN::VPUDevice::VPU_2_0;
        break;
    case VPUNN::VPUDevice::VPU_2_7:
        wl.device = VPUNN::VPUDevice::VPU_2_7;
        break;
    default:
        exit(EXIT_FAILURE);
    }

    switch (operation) {
    case VPUNN::Operation::CONVOLUTION:
        wl.op = VPUNN::Operation::CONVOLUTION;
        break;
    case VPUNN::Operation::DW_CONVOLUTION:
        wl.op = VPUNN::Operation::DW_CONVOLUTION;
        break;
    case VPUNN::Operation::ELTWISE:
        wl.op = VPUNN::Operation::ELTWISE;
        break;
    case VPUNN::Operation::MAXPOOL:
        wl.op = VPUNN::Operation::MAXPOOL;
        break;
    case VPUNN::Operation::AVEPOOL:
        wl.op = VPUNN::Operation::AVEPOOL;
        break;
    default:
        exit(EXIT_FAILURE);
    }

    switch (data) {
    case VPUNN::DataType::FLOAT16:
        wl.inputs = {VPUNN::VPUTensor(dim, dim, channels, 1, VPUNN::DataType::FLOAT16)};  // input dimensions
        wl.execution_order = VPUNN::ExecutionMode::VECTOR_FP16;                           // execution mode
        break;
    case VPUNN::DataType::UINT8:
        wl.inputs = {VPUNN::VPUTensor(dim, dim, channels, 1, VPUNN::DataType::UINT8)};  // input dimensions
        wl.execution_order = VPUNN::ExecutionMode::VECTOR;                              // execution mode
        break;
    default:
        exit(EXIT_FAILURE);
    }

    wl.outputs = {VPUNN::VPUTensor(dim, dim, channels, 1, VPUNN::DataType::FLOAT16)};  // output dimensions
    wl.kernels = {3, 3};                                                               // kernels
    wl.strides = {1, 1};                                                               // strides
    wl.padding = {1, 1, 1, 1};                                                         // padding
    return wl;
}

TEST(ActivityFactor, TestPowerActivityFactorConv16) {
    auto wl = generate_helper_layer(16, 16, VPUNN::Operation::CONVOLUTION, VPUNN::VPUDevice::VPU_2_0,
                                    VPUNN::DataType::FLOAT16);
    auto wlu = generate_helper_layer(16, 16, VPUNN::Operation::CONVOLUTION, VPUNN::VPUDevice::VPU_2_0,
                                     VPUNN::DataType::UINT8);
    auto wl_2_7 = generate_helper_layer(16, 16, VPUNN::Operation::CONVOLUTION, VPUNN::VPUDevice::VPU_2_7,
                                        VPUNN::DataType::FLOAT16);
    auto wlu_2_7 = generate_helper_layer(16, 16, VPUNN::Operation::CONVOLUTION, VPUNN::VPUDevice::VPU_2_7,
                                         VPUNN::DataType::UINT8);

    std::list<VPUNN::DPUWorkload> workloads = {wl, wlu, wl_2_7, wlu_2_7};
    std::list<float> values = {0.7561f, 0.8691f, 2.7381f, 2.1631f};
    auto i = workloads.begin();
    auto j = values.begin();

    for (; i != workloads.end() && j != values.end(); ++i, ++j) {
        EXPECT_NEAR(model.DPUActivityFactor(*i), *j, 0.0001f);
    }
}

TEST(ActivityFactor, TestPowerActivityFactorConv32) {
    auto wl = generate_helper_layer(16, 32, VPUNN::Operation::CONVOLUTION, VPUNN::VPUDevice::VPU_2_0,
                                    VPUNN::DataType::FLOAT16);
    auto wlu = generate_helper_layer(16, 32, VPUNN::Operation::CONVOLUTION, VPUNN::VPUDevice::VPU_2_0,
                                     VPUNN::DataType::UINT8);
    auto wl_2_7 = generate_helper_layer(16, 32, VPUNN::Operation::CONVOLUTION, VPUNN::VPUDevice::VPU_2_7,
                                        VPUNN::DataType::FLOAT16);
    auto wlu_2_7 = generate_helper_layer(16, 32, VPUNN::Operation::CONVOLUTION, VPUNN::VPUDevice::VPU_2_7,
                                         VPUNN::DataType::UINT8);

    std::list<VPUNN::DPUWorkload> workloads = {wl, wlu, wl_2_7, wlu_2_7};
    std::list<float> values = {0.7996f, 0.9191f, 1.4667f, 1.1587f};
    auto i = workloads.begin();
    auto j = values.begin();

    for (; i != workloads.end() && j != values.end(); ++i, ++j) {
        EXPECT_NEAR(model.DPUActivityFactor(*i), *j, 0.0001);
    }
}

TEST(ActivityFactor, TestPowerActivityFactorConv64) {
    auto wl = generate_helper_layer(16, 64, VPUNN::Operation::CONVOLUTION, VPUNN::VPUDevice::VPU_2_0,
                                    VPUNN::DataType::FLOAT16);
    auto wlu = generate_helper_layer(16, 64, VPUNN::Operation::CONVOLUTION, VPUNN::VPUDevice::VPU_2_0,
                                     VPUNN::DataType::UINT8);
    auto wl_2_7 = generate_helper_layer(16, 64, VPUNN::Operation::CONVOLUTION, VPUNN::VPUDevice::VPU_2_7,
                                        VPUNN::DataType::FLOAT16);
    auto wlu_2_7 = generate_helper_layer(16, 64, VPUNN::Operation::CONVOLUTION, VPUNN::VPUDevice::VPU_2_7,
                                         VPUNN::DataType::UINT8);

    std::list<VPUNN::DPUWorkload> workloads = {wl, wlu, wl_2_7, wlu_2_7};
    std::list<float> values = {0.8691f, 0.9990f, 1.63417f, 1.2909f};
    auto i = workloads.begin();
    auto j = values.begin();

    for (; i != workloads.end() && j != values.end(); ++i, ++j) {
        EXPECT_NEAR(model.DPUActivityFactor(*i), *j, 0.0001);
    }
}

TEST(ActivityFactor, TestPowerActivityFactorConv128) {
    auto wl = generate_helper_layer(16, 128, VPUNN::Operation::CONVOLUTION, VPUNN::VPUDevice::VPU_2_0,
                                    VPUNN::DataType::FLOAT16);
    auto wlu = generate_helper_layer(16, 128, VPUNN::Operation::CONVOLUTION, VPUNN::VPUDevice::VPU_2_0,
                                     VPUNN::DataType::UINT8);
    auto wl_2_7 = generate_helper_layer(16, 128, VPUNN::Operation::CONVOLUTION, VPUNN::VPUDevice::VPU_2_7,
                                        VPUNN::DataType::FLOAT16);
    auto wlu_2_7 = generate_helper_layer(16, 128, VPUNN::Operation::CONVOLUTION, VPUNN::VPUDevice::VPU_2_7,
                                         VPUNN::DataType::UINT8);

    std::list<VPUNN::DPUWorkload> workloads = {wl, wlu, wl_2_7, wlu_2_7};
    std::list<float> values = {0.8256f, 0.9490f, 1.4184f, 1.1206f};
    auto i = workloads.begin();
    auto j = values.begin();

    for (; i != workloads.end() && j != values.end(); ++i, ++j) {
        EXPECT_NEAR(model.DPUActivityFactor(*i), *j, 0.0001);
    }
}

TEST(ActivityFactor, TestPowerActivityFactorConv256) {
    auto wl = generate_helper_layer(16, 256, VPUNN::Operation::CONVOLUTION, VPUNN::VPUDevice::VPU_2_0,
                                    VPUNN::DataType::FLOAT16);
    auto wlu = generate_helper_layer(16, 256, VPUNN::Operation::CONVOLUTION, VPUNN::VPUDevice::VPU_2_0,
                                     VPUNN::DataType::UINT8);
    auto wl_2_7 = generate_helper_layer(16, 256, VPUNN::Operation::CONVOLUTION, VPUNN::VPUDevice::VPU_2_7,
                                        VPUNN::DataType::FLOAT16);
    auto wlu_2_7 = generate_helper_layer(16, 256, VPUNN::Operation::CONVOLUTION, VPUNN::VPUDevice::VPU_2_7,
                                         VPUNN::DataType::UINT8);

    std::list<VPUNN::DPUWorkload> workloads = {wl, wlu, wl_2_7, wlu_2_7};
    std::list<float> values = {0.7474f, 0.8591f, 1.2610f, 0.9962f};
    auto i = workloads.begin();
    auto j = values.begin();

    for (; i != workloads.end() && j != values.end(); ++i, ++j) {
        EXPECT_NEAR(model.DPUActivityFactor(*i), *j, 0.0001);
    }
}

TEST(ActivityFactor, TestPowerActivityFactorConv512) {
    auto wl = generate_helper_layer(16, 512, VPUNN::Operation::CONVOLUTION, VPUNN::VPUDevice::VPU_2_0,
                                    VPUNN::DataType::FLOAT16);
    auto wlu = generate_helper_layer(16, 512, VPUNN::Operation::CONVOLUTION, VPUNN::VPUDevice::VPU_2_0,
                                     VPUNN::DataType::UINT8);
    auto wl_2_7 = generate_helper_layer(16, 512, VPUNN::Operation::CONVOLUTION, VPUNN::VPUDevice::VPU_2_7,
                                        VPUNN::DataType::FLOAT16);
    auto wlu_2_7 = generate_helper_layer(16, 512, VPUNN::Operation::CONVOLUTION, VPUNN::VPUDevice::VPU_2_7,
                                         VPUNN::DataType::UINT8);

    std::list<VPUNN::DPUWorkload> workloads = {wl, wlu, wl_2_7, wlu_2_7};
    std::list<float> values = {0.7561f, 0.8691f, 1.1454f, 0.9049f};
    auto i = workloads.begin();
    auto j = values.begin();

    for (; i != workloads.end() && j != values.end(); ++i, ++j) {
        EXPECT_NEAR(model.DPUActivityFactor(*i), *j, 0.0001);
    }
}

TEST(ActivityFactor, TestPowerActivityFactorConv1024) {
    auto wl_2_7 = generate_helper_layer(16, 1024, VPUNN::Operation::CONVOLUTION, VPUNN::VPUDevice::VPU_2_7,
                                        VPUNN::DataType::FLOAT16);
    auto wlu_2_7 = generate_helper_layer(16, 1024, VPUNN::Operation::CONVOLUTION, VPUNN::VPUDevice::VPU_2_7,
                                         VPUNN::DataType::UINT8);

    std::list<VPUNN::DPUWorkload> workloads = {wl_2_7, wlu_2_7};
    std::list<float> values = {1.3627f, 1.0765f};
    auto i = workloads.begin();
    auto j = values.begin();

    for (; i != workloads.end() && j != values.end(); ++i, ++j) {
        EXPECT_NEAR(model.DPUActivityFactor(*i), *j, 0.0001);
    }
}

TEST(ActivityFactor, TestPowerActivityFactorConv2048) {
    auto wl_2_7 = generate_helper_layer(16, 2048, VPUNN::Operation::CONVOLUTION, VPUNN::VPUDevice::VPU_2_7,
                                        VPUNN::DataType::FLOAT16);
    auto wlu_2_7 = generate_helper_layer(16, 2048, VPUNN::Operation::CONVOLUTION, VPUNN::VPUDevice::VPU_2_7,
                                         VPUNN::DataType::UINT8);

    std::list<VPUNN::DPUWorkload> workloads = {wl_2_7, wlu_2_7};
    std::list<float> values = {12.8893f, 10.1825f};
    auto i = workloads.begin();
    auto j = values.begin();

    for (; i != workloads.end() && j != values.end(); ++i, ++j) {
        EXPECT_NEAR(model.DPUActivityFactor(*i), *j, 0.0001);
    }
}

TEST(ActivityFactor, TestPowerActivityFactorConv) {
    auto wl_r = generate_helper_layer(16, 196, VPUNN::Operation::CONVOLUTION, VPUNN::VPUDevice::VPU_2_0,
                                      VPUNN::DataType::FLOAT16);
    auto wlu_r = generate_helper_layer(16, 196, VPUNN::Operation::CONVOLUTION, VPUNN::VPUDevice::VPU_2_0,
                                       VPUNN::DataType::UINT8);
    auto wl_r_2_7 = generate_helper_layer(16, 196, VPUNN::Operation::CONVOLUTION, VPUNN::VPUDevice::VPU_2_7,
                                          VPUNN::DataType::FLOAT16);
    auto wlu_r_2_7 = generate_helper_layer(16, 196, VPUNN::Operation::CONVOLUTION, VPUNN::VPUDevice::VPU_2_7,
                                           VPUNN::DataType::UINT8);
    auto wl_g = generate_helper_layer(16, 10, VPUNN::Operation::CONVOLUTION, VPUNN::VPUDevice::VPU_2_0,
                                      VPUNN::DataType::FLOAT16);
    auto wlu_g = generate_helper_layer(16, 10, VPUNN::Operation::CONVOLUTION, VPUNN::VPUDevice::VPU_2_0,
                                       VPUNN::DataType::UINT8);
    auto wl_g_2_7 = generate_helper_layer(16, 10, VPUNN::Operation::CONVOLUTION, VPUNN::VPUDevice::VPU_2_7,
                                          VPUNN::DataType::FLOAT16);
    auto wlu_g_2_7 = generate_helper_layer(16, 10, VPUNN::Operation::CONVOLUTION, VPUNN::VPUDevice::VPU_2_7,
                                           VPUNN::DataType::UINT8);
    auto wl_s = generate_helper_layer(16, 1024, VPUNN::Operation::CONVOLUTION, VPUNN::VPUDevice::VPU_2_0,
                                      VPUNN::DataType::FLOAT16);
    auto wlu_s = generate_helper_layer(16, 1024, VPUNN::Operation::CONVOLUTION, VPUNN::VPUDevice::VPU_2_0,
                                       VPUNN::DataType::UINT8);
    auto wl_s_2_7 = generate_helper_layer(16, 4096, VPUNN::Operation::CONVOLUTION, VPUNN::VPUDevice::VPU_2_7,
                                          VPUNN::DataType::FLOAT16);
    auto wlu_s_2_7 = generate_helper_layer(16, 4096, VPUNN::Operation::CONVOLUTION, VPUNN::VPUDevice::VPU_2_7,
                                           VPUNN::DataType::UINT8);

    std::list<VPUNN::DPUWorkload> workloads = {wl_r,     wlu_r,     wl_r_2_7, wlu_r_2_7, wl_g,     wlu_g,
                                               wl_g_2_7, wlu_g_2_7, wl_s,     wlu_s,     wl_s_2_7, wlu_s_2_7};
    std::list<float> values = {0.8691f, 0.9990f, 1.9980f, 1.5784f, 0.7561f,  0.8691f,
                               2.7381f, 2.1631f, 0.7561f, 0.8691f, 12.8893f, 10.1825f};
    auto i = workloads.begin();
    auto j = values.begin();

    for (; i != workloads.end() && j != values.end(); ++i, ++j) {
        EXPECT_NEAR(model.DPUActivityFactor(*i), *j, 0.0001);
    }
}

TEST(ActivityFactor, TestPowerActivityFactorCmConv) {
    EXPECT_EXIT(generate_helper_layer(16, 32, VPUNN::Operation::CONVOLUTION, VPUNN::VPUDevice::VPU_4_0,
                                      VPUNN::DataType::FLOAT16),
                testing::ExitedWithCode(EXIT_FAILURE), "");
    EXPECT_EXIT(generate_helper_layer(16, 32, VPUNN::Operation::CM_CONVOLUTION, VPUNN::VPUDevice::VPU_2_0,
                                      VPUNN::DataType::FLOAT16),
                testing::ExitedWithCode(EXIT_FAILURE), "");
    EXPECT_EXIT(generate_helper_layer(16, 32, VPUNN::Operation::CONVOLUTION, VPUNN::VPUDevice::VPU_2_0,
                                      VPUNN::DataType::INT8),
                testing::ExitedWithCode(EXIT_FAILURE), "");
}

TEST(ActivityFactor, TestPowerActivityFactorDwConv) {
    auto wl = generate_helper_layer(16, 32, VPUNN::Operation::DW_CONVOLUTION, VPUNN::VPUDevice::VPU_2_0,
                                    VPUNN::DataType::FLOAT16);
    auto wlu = generate_helper_layer(16, 32, VPUNN::Operation::DW_CONVOLUTION, VPUNN::VPUDevice::VPU_2_0,
                                     VPUNN::DataType::UINT8);
    auto wl_2_7 = generate_helper_layer(16, 32, VPUNN::Operation::DW_CONVOLUTION, VPUNN::VPUDevice::VPU_2_7,
                                        VPUNN::DataType::FLOAT16);
    auto wlu_2_7 = generate_helper_layer(16, 32, VPUNN::Operation::DW_CONVOLUTION, VPUNN::VPUDevice::VPU_2_7,
                                         VPUNN::DataType::UINT8);

    std::list<VPUNN::DPUWorkload> workloads = {wl, wlu, wl_2_7, wlu_2_7};
    std::list<float> values = {5.0757f, 5.8341f, 1.9583f, 1.5471f};
    auto i = workloads.begin();
    auto j = values.begin();

    for (; i != workloads.end() && j != values.end(); ++i, ++j) {
        EXPECT_NEAR(model.DPUActivityFactor(*i), *j, 0.0001);
    }
}

TEST(ActivityFactor, TestPowerActivityFactorEltWise) {
    auto wl = generate_helper_layer(16, 32, VPUNN::Operation::ELTWISE, VPUNN::VPUDevice::VPU_2_0,
                                    VPUNN::DataType::FLOAT16);
    auto wlu =
            generate_helper_layer(16, 32, VPUNN::Operation::ELTWISE, VPUNN::VPUDevice::VPU_2_0, VPUNN::DataType::UINT8);
    auto wl_2_7 = generate_helper_layer(16, 32, VPUNN::Operation::ELTWISE, VPUNN::VPUDevice::VPU_2_7,
                                        VPUNN::DataType::FLOAT16);
    auto wlu_2_7 =
            generate_helper_layer(16, 32, VPUNN::Operation::ELTWISE, VPUNN::VPUDevice::VPU_2_7, VPUNN::DataType::UINT8);

    std::list<VPUNN::DPUWorkload> workloads = {wl, wlu, wl_2_7, wlu_2_7};
    std::list<float> values = {202.2554f, 232.4775f, 112.4388f, 88.8267f};
    auto i = workloads.begin();
    auto j = values.begin();

    for (; i != workloads.end() && j != values.end(); ++i, ++j) {
        EXPECT_NEAR(model.DPUActivityFactor(*i), *j, 0.0001);
    }
}

TEST(ActivityFactor, TestPowerActivityFactorMaxpool) {
    auto wl = generate_helper_layer(16, 32, VPUNN::Operation::MAXPOOL, VPUNN::VPUDevice::VPU_2_0,
                                    VPUNN::DataType::FLOAT16);
    auto wlu =
            generate_helper_layer(16, 32, VPUNN::Operation::MAXPOOL, VPUNN::VPUDevice::VPU_2_0, VPUNN::DataType::UINT8);
    auto wl_2_7 = generate_helper_layer(16, 32, VPUNN::Operation::MAXPOOL, VPUNN::VPUDevice::VPU_2_7,
                                        VPUNN::DataType::FLOAT16);
    auto wlu_2_7 =
            generate_helper_layer(16, 32, VPUNN::Operation::MAXPOOL, VPUNN::VPUDevice::VPU_2_7, VPUNN::DataType::UINT8);

    std::list<VPUNN::DPUWorkload> workloads = {wl, wlu, wl_2_7, wlu_2_7};
    std::list<float> values = {4.5977f, 5.2847f, 1.8092f, 1.4293f};
    auto i = workloads.begin();
    auto j = values.begin();

    for (; i != workloads.end() && j != values.end(); ++i, ++j) {
        EXPECT_NEAR(model.DPUActivityFactor(*i), *j, 0.0001);
    }
}

TEST(ActivityFactor, TestPowerActivityFactorAvePool) {
    auto wl = generate_helper_layer(16, 32, VPUNN::Operation::AVEPOOL, VPUNN::VPUDevice::VPU_2_0,
                                    VPUNN::DataType::FLOAT16);
    auto wlu =
            generate_helper_layer(16, 32, VPUNN::Operation::AVEPOOL, VPUNN::VPUDevice::VPU_2_0, VPUNN::DataType::UINT8);
    auto wl_2_7 = generate_helper_layer(16, 32, VPUNN::Operation::AVEPOOL, VPUNN::VPUDevice::VPU_2_7,
                                        VPUNN::DataType::FLOAT16);
    auto wlu_2_7 =
            generate_helper_layer(16, 32, VPUNN::Operation::AVEPOOL, VPUNN::VPUDevice::VPU_2_7, VPUNN::DataType::UINT8);

    std::list<VPUNN::DPUWorkload> workloads = {wl, wlu, wl_2_7, wlu_2_7};
    std::list<float> values = {28.3337f, 32.5674f, 1.0878f, 0.8593f};
    auto i = workloads.begin();
    auto j = values.begin();

    for (; i != workloads.end() && j != values.end(); ++i, ++j) {
        EXPECT_NEAR(model.DPUActivityFactor(*i), *j, 0.0001);
    }
}
