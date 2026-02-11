// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.
// #include "vpu_cost_model.h"
#ifndef VPUNN_UT_VPU_DMA_COST_MODEL_H
#define VPUNN_UT_VPU_DMA_COST_MODEL_H

#include "vpu_dma_cost_model.h"

#include <gtest/gtest.h>
#include "common/common_helpers.h"
#include "vpu/compatibility/types01.h"
#include "vpu/cycles_interface_types.h"
#include "vpu/dma_cost_providers/dma_theoretical_cost_provider.h"
#include "vpu/sample_generator/random_task_generator.h"
#include "vpu/validation/interface_valid_values.h"
#include "vpu_cost_model.h"

#include <algorithm>
#include <unordered_map>

namespace VPUNN_unit_tests {
using namespace VPUNN;

class TestDMANNCostModel : public ::testing::Test {
protected:
    const DMANNWorkload_NPU27 wl_glob_27{
            VPUNN::VPUDevice::VPU_2_7,  // VPUDevice device;  ///< NPU device

            3,     // int num_planes;  ///< starts from 0. 1 plane = 0 as value?
            8192,  // int length;

            4096,  // int src_width;
            512,   // int dst_width;
            128,   // int src_stride;
            0,     // int dst_stride;
            128,   // int src_plane_stride;
            1024,  // int dst_plane_stride;

            MemoryDirection::DDR2DDR  // MemoryDirection transfer_direction;

            //
    };
    //   VPUNN::DPUWorkload wl_glob_20;
    DMANNWorkload_NPU27 wl_glob_40M{wl_glob_27};

    DMANNWorkload_NPU27 wl_glob_50M{wl_glob_27};  // wl_glob_27 = wl_glob_40M

    DMACostModel<DMANNWorkload_NPU27> model{};
    // DMACostModel specialEmptyDMAModel;

    const DMANNWorkload_NPU40 wl_glob_40{
            VPUNN::VPUDevice::VPU_4_0,  // VPUDevice device;  ///< NPU device

            8192,  // int src_width;
            8192,  // int dst_width;

            0,  // int num_dim;
            {{{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}}},
            Num_DMA_Engine::Num_Engine_1,
            MemoryDirection::CMX2CMX  // MemoryDirection transfer_direction;

            //
    };
    void SetUp() override {
        wl_glob_40M.device = VPUNN::VPUDevice::VPU_4_0;
        wl_glob_50M.device = VPUNN::VPUDevice::NPU_5_0;
    }

    auto read_a_file(const std::string filename) const {
        std::vector<char> buf(0);
        std::ifstream myFile;
        myFile.open(filename, std::ios::binary | std::ios::in);
        if (myFile.fail()) {
            // File does not exist code here
            return buf;
        }
        myFile.seekg(0, std::ios::end);
        const auto length = myFile.tellg();
        myFile.seekg(0, std::ios::beg);

        buf.resize(static_cast<size_t>(length));

        myFile.read(buf.data(), length);
        myFile.close();
        return buf;
    }

    using ModelDescriptor =
            VPUNNModelsFiles::ModelDescriptor;  ///< make this type directly visible inside of this class
    const VPUNNModelsFiles& the_NN_models{VPUNNModelsFiles::getModels()};  ///< the paths to available NN models

    // struct DataOut {
    //     int errors_cnt{-1};
    //     double correlation{-10.0};
    // };

    VPUNN::CyclesInterfaceType delta_cycles(const VPUNN::CyclesInterfaceType& v1,
                                            const VPUNN::CyclesInterfaceType& v2) {
        return (v1 >= v2) ? (v1 - v2) : (v2 - v1);  // aways positive
    }

    /// @brief max allowable delta between 2 cycles , so that we consider them still equal
    ///
    /// @param v1 a value
    /// @param v2 another value
    /// @param tolerance_level how permissive to be in delta.
    /// @returns max value that can be between v1 and v2 so that they are practically equal.
    VPUNN::CyclesInterfaceType max_tolerance_cycles(const VPUNN::CyclesInterfaceType& v1,
                                                    const VPUNN::CyclesInterfaceType& v2,
                                                    const int tolerance_level = 1) {
        const VPUNN::CyclesInterfaceType v{std::max(v1, v2)};

        VPUNN::CyclesInterfaceType tolerance{1U};  // rounding errors

        if (tolerance_level <= 1) {
            if (v >= 10000000U) {  // 10 millions
                tolerance = 10U;
            } else if (v >= 1000000U) {  // 1 million
                tolerance = 8U;
            } else if (v >= 100000U) {  // 100k
                tolerance = 5U;
            } else if (v >= 1000U) {
                tolerance = 2U;
            }

        } else if (tolerance_level > 1) {
            if (v >= 10000000U) {  // 10 millions
                tolerance = 20U;
            } else if (v >= 1000000U) {  // 1 million
                tolerance = 10U;
            } else if (v >= 1000U) {
                tolerance = 2U;
            }
        }

        return tolerance;
    }
};

class TestDMA_TH_CostModel : public ::testing::Test {
public:
protected:
    static constexpr int evoX{0};  // factor to adjust expectation between evo 0 and 1

    //   VPUNN::DPUWorkload wl_glob_20;
    DMACostModel<DMANNWorkload_NPU27> model{};
    // DMACostModel specialEmptyDMAModel;

    std::vector<VPUDevice> valid_dev_post_LNL{VPUDevice::VPU_4_0
                                              ,
                                              VPUDevice::NPU_5_0
    };

    using ModelDescriptor =
            VPUNNModelsFiles::ModelDescriptor;  ///< make this type directly visible inside of this class
    const VPUNNModelsFiles& the_NN_models{VPUNNModelsFiles::getModels()};  ///< the paths to available NN models

    class TestCase {
    public:
        DMAWorkload t_in;              // wl
        CyclesInterfaceType t_exp{0};  // expected time (GT)
        std::string t_name{""};        // name of the test
    };

    auto mkwl(const int bytes, MemoryLocation in_loc, MemoryLocation out_loc, VPUDevice device) const {
        return mkwl_(bytes, bytes, DataType::UINT8, DataType::UINT8, Layout::ZXY, Layout::ZXY, in_loc, out_loc, device);
    };

    DMAWorkload mkwl_compr(const int bytes_src, const int bytes_dst, MemoryLocation in_loc, MemoryLocation out_loc,
                           VPUDevice device) const {
        return mkwl_(bytes_src, bytes_dst, DataType::UINT8, DataType::UINT8, Layout::ZXY, Layout::ZXY, in_loc, out_loc,
                     device);
    };

    DMAWorkload mkwl_(const int elm_src, const int elm_dst,              // elm
                      const DataType src_type, const DataType dst_type,  // type
                      const Layout src_layout, const Layout dst_layout,  // layout
                      MemoryLocation in_loc, MemoryLocation out_loc, VPUDevice device) const {
        const DMAWorkload dmaOld_{
                device,                                               // device
                {VPUTensor(elm_src, 1, 1, 1, src_type, src_layout)},  // input dimensions WHCB
                {VPUTensor(elm_dst, 1, 1, 1, dst_type, dst_layout)},  // output dimensions
                in_loc,                                               // src
                out_loc,                                              // dst
                1,                                                    // owt
        };
        return dmaOld_;
    };

    const DataType dt{DataType::UINT8};
    const MemoryLocation DRAM{MemoryLocation::DRAM};
    const MemoryLocation CMX{MemoryLocation::CMX};

    float computeMicroseconds(CyclesInterfaceType dpuCycles, const int frequencyMHz) const {
        return dpuCycles * (1.0f / frequencyMHz);
    }
    float computeMicroseconds(CyclesInterfaceType dpuCycles, const VPUDevice device) const {
        return dpuCycles * (1.0f / GlobalHarwdwareCharacteristics::get_dpu_fclk(device));
    }

private:
};

}


#endif
