// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.
#include "vpu/validation/dpu_operations_sanitizer.h"
#include "vpu/validation/dpu_operations_validator.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>
#include "common/common_helpers.h"

namespace VPUNN_unit_tests {
using namespace VPUNN;

class DPU_OperationValidator_Test : public ::testing::Test {
public:
protected:
    void SetUp() override {
    }
    VPUNN::DPU_OperationValidator dut;  // no overhead by default

    const int cmx_overhead{0 /*80 * 1024 + 16 * 1024*/};  // cmx_memory_aligned_overhead
    // const int alignment{16384};                           // alignement_size_bytes

    const std::map<VPUDevice, int> alignement_data{
            {VPUDevice::VPU_2_0, 16 * 1024},  //
            {VPUDevice::VPU_2_1, 16 * 1024},  //
            {VPUDevice::VPU_2_7, 16 * 1024},  //
            {VPUDevice::VPU_4_0, 16 * 1024},  //
            {VPUDevice::NPU_5_0, 1},    // 32KB, but no tensor alignment requested
    };

    int get_alignment(const VPUDevice device) const {
        auto it = alignement_data.find(device);
        if (it != alignement_data.end()) {
            return it->second;
        }
        return 0;  // no alignment
    }

    bool isAligned(long long mem_size, int alignment) const {
        return ((mem_size % alignment) != 0) ? false : true;
    }
    bool isAligned(long long mem_size, const VPUDevice device) const {
        return ((mem_size % get_alignment(device)) != 0) ? false : true;
    }

    long long int align(long long mem_size, int alignment) const {
        const auto rem = mem_size % alignment;
        return (rem == 0) ? mem_size : mem_size + (alignment - rem);
    }
    long long int align(long long mem_size, const VPUDevice device) const {
        const auto rem = mem_size % get_alignment(device);
        return (rem == 0) ? mem_size : mem_size + (get_alignment(device) - rem);
    }

    DPU_OperationValidator_Test() {
    }

private:
};
}  // namespace VPUNN_unit_tests