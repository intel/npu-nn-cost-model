// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.
#include "vpu/validation/shave_workloads_sanitizer.h"
#include "vpu/shave/shave_devices.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>
#include "common_helpers.h"

namespace VPUNN_unit_tests {
using namespace VPUNN;

class SHAVEWorkloadsSanitizer_Test : public ::testing::Test {
public:
protected:
    void SetUp() override {
    }
    VPUNN::SHAVE_Workloads_Sanitizer dut;  // no overhead by default

};

TEST_F(SHAVEWorkloadsSanitizer_Test, ShaveSanitizerTestingNormalCase){
    
    {
        VPUNN::SanityReport sane;
        SHAVEWorkload swl("relu", VPUDevice::VPU_4_0, {VPUTensor(1, 50, 50, 1, DataType::FLOAT16)},
                          {VPUTensor(1, 50, 50, 1, DataType::FLOAT16)});
        
        dut.check_and_sanitize_workloads(swl, sane);

        ASSERT_EQ(sane.value(), VPUNN::Cycles::NO_ERROR)<<sane.value();      
    }
}

TEST_F(SHAVEWorkloadsSanitizer_Test, ShaveSanitizerTestingInvalidIODataType){
    
    {
        VPUNN::SanityReport sane;
        SHAVEWorkload swl("relu", VPUDevice::VPU_4_0, {VPUTensor(1, 50, 50, 1, DataType::INT8)},
                          {VPUTensor(1, 50, 50, 1, DataType::INT8)});
        
        dut.check_and_sanitize_workloads(swl, sane);

        ASSERT_EQ(sane.value(), V(VPUNN::Cycles::ERROR_SHAVE_INVALID_INPUT))
                << sane.info << "\n error is : " << VPUNN::Cycles::Cycles::toErrorText(sane.value()) << "\n"
                << swl;        
    }
    {
        VPUNN::SanityReport sane;
        SHAVEWorkload swl("relu", VPUDevice::VPU_4_0, {VPUTensor(1, 50, 50, 1, DataType::FLOAT16)},
                          {VPUTensor(1, 50, 50, 1, DataType::INT8)});
        
        dut.check_and_sanitize_workloads(swl, sane);

        ASSERT_EQ(sane.value(), V(VPUNN::Cycles::ERROR_SHAVE_INVALID_INPUT))
                << sane.info << "\n error is : " << VPUNN::Cycles::Cycles::toErrorText(sane.value()) << "\n"
                << swl;        
    }
    {
        VPUNN::SanityReport sane;
        SHAVEWorkload swl("relu", VPUDevice::VPU_4_0, {VPUTensor(1, 50, 50, 1, DataType::INT8)},
                          {VPUTensor(1, 50, 50, 1, DataType::FLOAT16)});
        
        dut.check_and_sanitize_workloads(swl, sane);

        ASSERT_EQ(sane.value(), V(VPUNN::Cycles::ERROR_SHAVE_INVALID_INPUT))
                << sane.info << "\n error is : " << VPUNN::Cycles::Cycles::toErrorText(sane.value()) << "\n"
                << swl;        
    }
}

TEST_F(SHAVEWorkloadsSanitizer_Test, ShaveSanitizerTestingInputTooBigError){
    
    {
        VPUNN::SanityReport sane;
        SHAVEWorkload swl("relu", VPUDevice::VPU_4_0, {VPUTensor(1, 400000, 1, 1, DataType::FLOAT16)},
                          {VPUTensor(1, 400000, 1, 1, DataType::FLOAT16)});
        
        dut.check_and_sanitize_workloads(swl, sane);

        ASSERT_EQ(sane.value(), V(VPUNN::Cycles::ERROR_INPUT_TOO_BIG))
                << sane.info << "\n error is : " << VPUNN::Cycles::Cycles::toErrorText(sane.value()) << "\n"
                << swl;        
    }
    {
        VPUNN::SanityReport sane;
        SHAVEWorkload swl("sigmoid", VPUDevice::VPU_2_7, {VPUTensor(1, 550000, 1, 1, DataType::FLOAT16)},
                          {VPUTensor(1, 550000, 1, 1, DataType::FLOAT16)});
        
        dut.check_and_sanitize_workloads(swl, sane);

        ASSERT_EQ(sane.value(), V(VPUNN::Cycles::ERROR_INPUT_TOO_BIG))
                << sane.info << "\n error is : " << VPUNN::Cycles::Cycles::toErrorText(sane.value()) << "\n"
                << swl;        
    }
}

TEST_F(SHAVEWorkloadsSanitizer_Test, ShaveConfigurationTestingInvalidIODataType){
    
    const ShaveConfiguration shaves;

    {
        SHAVEWorkload swl("relu", VPUDevice::VPU_4_0, {VPUTensor(1, 50, 50, 1, DataType::INT8)},
                          {VPUTensor(1, 50, 50, 1, DataType::INT8)});
        std::string info;
        CyclesInterfaceType cycles = shaves.computeCycles(swl, info);
        EXPECT_EQ(cycles, V(Cycles::ERROR_SHAVE_INVALID_INPUT));
    }
    {
        SHAVEWorkload swl("relu", VPUDevice::VPU_4_0, {VPUTensor(1, 50, 50, 1, DataType::FLOAT16)},
                          {VPUTensor(1, 50, 50, 1, DataType::INT8)});
        std::string info;
        CyclesInterfaceType cycles = shaves.computeCycles(swl, info);
        EXPECT_EQ(cycles, V(Cycles::ERROR_SHAVE_INVALID_INPUT));
    }
    {
        SHAVEWorkload swl("relu", VPUDevice::VPU_4_0, {VPUTensor(1, 50, 50, 1, DataType::INT8)},
                          {VPUTensor(1, 50, 50, 1, DataType::FLOAT16)});
        std::string info;
        CyclesInterfaceType cycles = shaves.computeCycles(swl, info);
        EXPECT_EQ(cycles, V(Cycles::ERROR_SHAVE_INVALID_INPUT));
    }
}

TEST_F(SHAVEWorkloadsSanitizer_Test, ShaveConfigurationTestingInputTooBigError){
    const ShaveConfiguration shaves;

    {
        SHAVEWorkload swl("relu", VPUDevice::VPU_4_0, {VPUTensor(1, 400000, 1, 1, DataType::FLOAT16)},
                          {VPUTensor(1, 400000, 1, 1, DataType::FLOAT16)});
        std::string info;
        CyclesInterfaceType cycles = shaves.computeCycles(swl, info);
        EXPECT_EQ(cycles, V(Cycles::ERROR_INPUT_TOO_BIG));
    }

    {
        SHAVEWorkload swl("sigmoid", VPUDevice::VPU_2_7, {VPUTensor(1, 550000, 1, 1, DataType::FLOAT16)},
                          {VPUTensor(1, 550000, 1, 1, DataType::FLOAT16)});
        std::string info;
        CyclesInterfaceType cycles = shaves.computeCycles(swl, info);
        EXPECT_EQ(cycles, V(Cycles::ERROR_INPUT_TOO_BIG));
    }
}
}  // namespace VPUNN_unit_tests