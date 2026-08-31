// Copyright © 2026 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the "Software Package")
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the "third-party-programs.txt" or other similarly-named text file included with the
// Software Package for additional details.

#include "vpu/vpu_performance_model.h"
#include "performance_model.h"

#include <algorithm>
#include <cmath>
#include <optional>
#include <vector>
namespace VPUNN_unit_tests {
using namespace VPUNN;

// ---------------------------------------------------------------------------
// Shared base fixture: device list + workload factory
// ---------------------------------------------------------------------------

/**
 * @brief Base fixture shared by all HWPerformanceModel test suites.
 *
 * Provides:
 *  - `perf`        — a default-constructed HWPerformanceModel under test.
 *  - `all_devices()` — full device list, with NPU_RESERVED/NPU_RESERVED_1 behind embargo guards.
 *  - `make_wl()`   — factory for minimal CONVOLUTION workloads with configurable
 *                    device, input dtype, output dtype and optional weight dtype.
 */
class TestHWPerformanceModel_Base : public TestHWPerformanceModel_BASICS {
protected:
    HWPerformanceModel perf{};

    /// Full list of devices always available (no embargo needed).
    const std::vector<VPUDevice> base_devices{
            VPUDevice::VPU_2_0, VPUDevice::VPU_2_1, VPUDevice::VPU_2_7,
            VPUDevice::VPU_4_0, VPUDevice::NPU_5_0, VPUDevice::NPU_5_0_W,
    };

    /// Returns the full device list extended by embargo-guarded devices.
    std::vector<VPUDevice> all_devices() const {
        std::vector<VPUDevice> devs = base_devices;
        return devs;
    }

    /// @brief Builds a minimal CONVOLUTION workload.
    /// @param dev    Target device.
    /// @param in_dt  Input tensor data type.
    /// @param out_dt Output tensor data type.
    /// @param wt_dt  Optional explicit weight data type (defaults to in_dt when absent).
    DPUWorkload make_wl(VPUDevice dev, DataType in_dt, DataType out_dt,
                        std::optional<DataType> wt_dt = std::nullopt) const {
        DPUWorkload wl{
                dev,
                Operation::CONVOLUTION,
                {VPUTensor(16, 16, 16, 1, in_dt)},   // input
                {VPUTensor(16, 16, 16, 1, out_dt)},  // output
                {1, 1},                              // kernels
                {1, 1},                              // strides
                {0, 0, 0, 0},                        // padding
                ExecutionMode::CUBOID_16x16,         // execution mode
        };
        if (wt_dt.has_value()) {
            wl.weight_type = wt_dt.value();
        }
        return wl;
    }
};

// ---------------------------------------------------------------------------
// Fixture: TestHWPerformanceModel_DPUCycles
// ---------------------------------------------------------------------------

/// @brief Tests for HWPerformanceModel MAC count and cycle computation
class TestHWPerformanceModel_DPUCycles : public TestHWPerformanceModel_Base {
public:
protected:
    // Input  : 5x5x100x1 UINT8
    // Output : 3x3x50x1  UINT8
    // Kernels: 3x3, Strides: 1x1, Padding: none
    // Expected ideal MAC count: 3*3 * (3*3*50) * 100 = 405000
    // VPU_2_7 nr_macs = 2048  => ceil(405000 / 2048) = 198 cycles
    DPUWorkload wl_ref_2_7{
            VPUDevice::VPU_2_7,
            Operation::CONVOLUTION,
            {VPUTensor(5, 5, 100, 1, DataType::UINT8)},  // input dimensions
            {VPUTensor(3, 3, 50, 1, DataType::UINT8)},   // output dimensions
            {3, 3},                                      // kernels
            {1, 1},                                      // strides
            {0, 0, 0, 0},                                // padding
            ExecutionMode::CUBOID_16x16,                 // execution mode
            ActivationFunction::NONE,                    // activation
            0.0F,                                        // act_sparsity
            0.0F,                                        // weight_sparsity
            {swz_def, swz_def},                          // input_swizzling
            {swz_def},                                   // output_swizzling
            1,                                           // output_write_tiles
            {0, 0, 0, 0},                                // offsets
            ISIStrategy::CLUSTERING,                     // isi_strategy
            false,                                       // weight_sparsity_enabled
    };
};

/// @brief Verify that DPU_Efficency_IdealCycles and DPU_Power_IdealCycles return
///        the same value when no sparsity is active, and that the value matches
///        the expected MAC-based cycle count.
TEST_F(TestHWPerformanceModel_DPUCycles, IdealCycles_Convolution_NoSparsity_VPU27) {
    constexpr unsigned long int expected_cycles{198U};

    EXPECT_EQ(perf.DPU_Efficency_IdealCycles(wl_ref_2_7), expected_cycles);
    EXPECT_EQ(perf.DPU_Power_IdealCycles(wl_ref_2_7), expected_cycles);

    // Without sparsity both methods must agree
    EXPECT_EQ(perf.DPU_Power_IdealCycles(wl_ref_2_7), perf.DPU_Efficency_IdealCycles(wl_ref_2_7));
}

// ---------------------------------------------------------------------------
// Tests: compute_Ideal_MAC_operations_cnt
// ---------------------------------------------------------------------------

/**
 * @brief For ELTWISE the MAC count equals the full volume of input[0], regardless of device.
 *
 * Formula: W * H * C * N (kernel is implicitly 1x1, no weight accumulation).
 * All devices share this formula, so the result must be identical across the full
 * device list.
 */
TEST_F(TestHWPerformanceModel_DPUCycles, IdealMAC_Eltwise_DeviceIndependent) {
    // 8x8x32x1  => 8*8*32*1 = 2048
    constexpr unsigned long int expected{8U * 8U * 32U * 1U};

    for (const auto dev : all_devices()) {
        DPUWorkload wl{
                dev,
                Operation::ELTWISE,
                {VPUTensor(8, 8, 32, 1, DataType::UINT8)},  // input
                {VPUTensor(8, 8, 32, 1, DataType::UINT8)},  // output
                {1, 1},                                     // kernels
                {1, 1},                                     // strides
                {0, 0, 0, 0},                               // padding
                ExecutionMode::CUBOID_16x16,
        };
        EXPECT_EQ(perf.compute_Ideal_MAC_operations_cnt(wl), expected)
                << "ELTWISE on device " << VPUDevice_ToText.at(static_cast<int>(dev));
    }
}

/**
 * @brief For DW_CONVOLUTION and MAXPOOL the MAC count equals kernel_W * kernel_H * out_W * out_H * out_C.
 *
 * These ops have no input-channel accumulation; the formula is device-independent.
 */
TEST_F(TestHWPerformanceModel_DPUCycles, IdealMAC_DWConvAndPool_DeviceIndependent) {
    // kernel 3x3, output 8x8x16 => 3*3 * 8*8*16 = 9216
    constexpr unsigned long int expected{3U * 3U * 8U * 8U * 16U};

    for (const auto dev : all_devices()) {
        for (const auto op : {Operation::DW_CONVOLUTION, Operation::MAXPOOL, Operation::AVEPOOL}) {
            DPUWorkload wl{
                    dev,
                    op,
                    {VPUTensor(10, 10, 16, 1, DataType::UINT8)},  // input (larger to allow 3x3 kernel)
                    {VPUTensor(8, 8, 16, 1, DataType::UINT8)},    // output
                    {3, 3},                                       // kernels
                    {1, 1},                                       // strides
                    {0, 0, 0, 0},                                 // padding
                    ExecutionMode::CUBOID_16x16,
            };
            EXPECT_EQ(perf.compute_Ideal_MAC_operations_cnt(wl), expected)
                    << "op=" << Operation_ToText.at(static_cast<int>(op))
                    << " device=" << VPUDevice_ToText.at(static_cast<int>(dev));
        }
    }
}

/**
 * @brief For CONVOLUTION on VPU < 2.7 the MAC count is kernel * output_volume * input_channels.
 *
 * Device-specific branch: VPU_2_0 and VPU_2_1 use `channels` directly
 * (no minimum-16 padding).  With 8 input channels the result is 8 * base,
 * not 16 * base.
 *
 * Formula: kernel_W*kernel_H * out_W*out_H*out_C * in_C
 *          = 1*1 * 4*4*8 * 8 = 1024
 */
TEST_F(TestHWPerformanceModel_DPUCycles, IdealMAC_Conv_VPUlt27_UsesActualChannels) {
    // kernel 1x1, output 4x4x8, input channels 8  =>  1*4*4*8*8 = 1024
    constexpr unsigned long int expected{1U * 1U * 4U * 4U * 8U * 8U};

    for (const auto dev : {VPUDevice::VPU_2_0, VPUDevice::VPU_2_1}) {
        DPUWorkload wl{
                dev,
                Operation::CONVOLUTION,
                {VPUTensor(4, 4, 8, 1, DataType::UINT8)},  // 8 input channels (< 16)
                {VPUTensor(4, 4, 8, 1, DataType::UINT8)},  // output
                {1, 1},                                    // kernels
                {1, 1},                                    // strides
                {0, 0, 0, 0},                              // padding
                ExecutionMode::CUBOID_16x16,
        };
        EXPECT_EQ(perf.compute_Ideal_MAC_operations_cnt(wl), expected)
                << "CONV device=" << VPUDevice_ToText.at(static_cast<int>(dev))
                << " — pre-2.7 must use actual channels, not padded-to-16";
    }
}

/**
 * @brief For CONVOLUTION on VPU >= 2.7 with input channels < 16 the MAC count
 *        uses 16 as the effective channel count (minimum DPU channel alignment).
 *
 * Formula: kernel_W*kernel_H * out_W*out_H*out_C * 16
 *          = 1*1 * 4*4*8 * 16 = 2048  (vs 1024 with the real 8 channels)
 */
TEST_F(TestHWPerformanceModel_DPUCycles, IdealMAC_Conv_VPUge27_ChannelLt16_PaddedTo16) {
    // kernel 1x1, output 4x4x8, input channels 8 (< 16) => padded to 16
    constexpr unsigned long int expected{1U * 1U * 4U * 4U * 8U * 16U};

    const std::vector<VPUDevice> ge27_devices{VPUDevice::VPU_2_7, VPUDevice::VPU_4_0, VPUDevice::NPU_5_0,
                                              VPUDevice::NPU_5_0_W};
    std::vector<VPUDevice> devs = ge27_devices;

    for (const auto dev : devs) {
        DPUWorkload wl{
                dev,
                Operation::CONVOLUTION,
                {VPUTensor(4, 4, 8, 1, DataType::UINT8)},  // 8 input channels (< 16)
                {VPUTensor(4, 4, 8, 1, DataType::UINT8)},  // output
                {1, 1},                                    // kernels
                {1, 1},                                    // strides
                {0, 0, 0, 0},                              // padding
                ExecutionMode::CUBOID_16x16,
        };
        EXPECT_EQ(perf.compute_Ideal_MAC_operations_cnt(wl), expected)
                << "CONV device=" << VPUDevice_ToText.at(static_cast<int>(dev))
                << " — VPU>=2.7 with channels<16 must pad to 16";
    }
}

/**
 * @brief For CONVOLUTION on VPU >= 2.7 with input channels >= 16 the MAC count
 *        uses the actual channel count (no padding).
 *
 * Formula: kernel_W*kernel_H * out_W*out_H*out_C * in_C
 *          = 3*3 * 4*4*32 * 64 = 294912
 */
TEST_F(TestHWPerformanceModel_DPUCycles, IdealMAC_Conv_VPUge27_ChannelGe16_UsesActualChannels) {
    // kernel 3x3, output 4x4x32, input channels 64 (>= 16)
    constexpr unsigned long int expected{3U * 3U * 4U * 4U * 32U * 64U};

    const std::vector<VPUDevice> ge27_devices{VPUDevice::VPU_2_7, VPUDevice::VPU_4_0, VPUDevice::NPU_5_0,
                                              VPUDevice::NPU_5_0_W};
    std::vector<VPUDevice> devs = ge27_devices;

    for (const auto dev : devs) {
        DPUWorkload wl{
                dev,
                Operation::CONVOLUTION,
                {VPUTensor(6, 6, 64, 1, DataType::UINT8)},  // 64 input channels (>= 16)
                {VPUTensor(4, 4, 32, 1, DataType::UINT8)},  // output
                {3, 3},                                     // kernels
                {1, 1},                                     // strides
                {0, 0, 0, 0},                               // padding
                ExecutionMode::CUBOID_16x16,
        };
        EXPECT_EQ(perf.compute_Ideal_MAC_operations_cnt(wl), expected)
                << "CONV device=" << VPUDevice_ToText.at(static_cast<int>(dev))
                << " — VPU>=2.7 with channels>=16 must use actual channels";
    }
}

// ---------------------------------------------------------------------------
// Tests: compute_Ideal_MAC_operations_cnt — CM_CONVOLUTION input-channel alignment
// ---------------------------------------------------------------------------

/**
 * @brief CM_CONVOLUTION with input channels <= 4 on VPU < 2.7 uses the actual channel count.
 *
 * Pre-2.7 devices apply no minimum-channel padding for any conv type.
 * Formula: kernel_W*kernel_H * out_W*out_H*out_C * in_C
 *          = 1*1 * 4*4*16 * 3 = 768
 */
TEST_F(TestHWPerformanceModel_DPUCycles, IdealMAC_CMConv_InputLe4_Pre27_UsesActualChannels) {
    constexpr unsigned int in_C{3};  // <= 4
    constexpr unsigned long int expected{1U * 1U * 4U * 4U * 16U * in_C};

    for (const auto dev : {VPUDevice::VPU_2_0, VPUDevice::VPU_2_1}) {
        DPUWorkload wl{
                dev,
                Operation::CM_CONVOLUTION,
                {VPUTensor(4, 4, in_C, 1, DataType::UINT8)},  // in_C = 3
                {VPUTensor(4, 4, 16, 1, DataType::UINT8)},    // output
                {1, 1},                                       // kernels
                {1, 1},                                       // strides
                {0, 0, 0, 0},                                 // padding
                ExecutionMode::CUBOID_16x16,
        };
        EXPECT_EQ(perf.compute_Ideal_MAC_operations_cnt(wl), expected)
                << "CM_CONV in_C=" << in_C << " device=" << VPUDevice_ToText.at(static_cast<int>(dev))
                << " - pre-2.7 must use actual channels";
    }
}

/**
 * @brief CM_CONVOLUTION with input channels <= 4 on VPU_2_7 through NPU_5_0_W
 *        is padded to 16 — the compress-conv alignment-to-4 is NOT active on these devices.
 *
 * Formula: kernel_W*kernel_H * out_W*out_H*out_C * 16
 *          = 1*1 * 4*4*16 * 16 = 4096
 */
TEST_F(TestHWPerformanceModel_DPUCycles, IdealMAC_CMConv_InputLe4_VPU27toNPU5W_PaddedTo16) {
    constexpr unsigned int in_C{3};  // <= 4
    constexpr unsigned long int expected{1U * 1U * 4U * 4U * 16U * 16U};

    for (const auto dev : {VPUDevice::VPU_2_7, VPUDevice::VPU_4_0, VPUDevice::NPU_5_0, VPUDevice::NPU_5_0_W}) {
        DPUWorkload wl{
                dev,
                Operation::CM_CONVOLUTION,
                {VPUTensor(4, 4, in_C, 1, DataType::UINT8)},  // in_C = 3
                {VPUTensor(4, 4, 16, 1, DataType::UINT8)},    // output
                {1, 1},                                       // kernels
                {1, 1},                                       // strides
                {0, 0, 0, 0},                                 // padding
                ExecutionMode::CUBOID_16x16,
        };
        EXPECT_EQ(perf.compute_Ideal_MAC_operations_cnt(wl), expected)
                << "CM_CONV in_C=" << in_C << " device=" << VPUDevice_ToText.at(static_cast<int>(dev))
                << " - must pad to 16 (CM_CONV/4 alignment not active on this device)";
    }
}

/**
 * @brief CM_CONVOLUTION with input channels in range 5..15 (> 4, < 16) is padded to 16
 *        on all VPU >= 2.7 devices, including newer ones.
 *
 * The compress-conv alignment-to-4 only applies when in_channels <= 4; inputs in 5..15
 * always fall back to the default 16-padding on every device.
 *
 * Formula: kernel_W*kernel_H * out_W*out_H*out_C * 16
 *          = 1*1 * 4*4*16 * 16 = 4096
 */
TEST_F(TestHWPerformanceModel_DPUCycles, IdealMAC_CMConv_InputGt4Lt16_AllVPUge27_PaddedTo16) {
    constexpr unsigned int in_C{8};  // > 4, < 16
    constexpr unsigned long int expected{1U * 1U * 4U * 4U * 16U * 16U};

    const std::vector<VPUDevice> base_ge27{VPUDevice::VPU_2_7, VPUDevice::VPU_4_0, VPUDevice::NPU_5_0,
                                           VPUDevice::NPU_5_0_W};
    std::vector<VPUDevice> devs = base_ge27;

    for (const auto dev : devs) {
        DPUWorkload wl{
                dev,
                Operation::CM_CONVOLUTION,
                {VPUTensor(4, 4, in_C, 1, DataType::UINT8)},  // in_C = 8
                {VPUTensor(4, 4, 16, 1, DataType::UINT8)},    // output
                {1, 1},                                       // kernels
                {1, 1},                                       // strides
                {0, 0, 0, 0},                                 // padding
                ExecutionMode::CUBOID_16x16,
        };
        EXPECT_EQ(perf.compute_Ideal_MAC_operations_cnt(wl), expected)
                << "CM_CONV in_C=" << in_C << " device=" << VPUDevice_ToText.at(static_cast<int>(dev))
                << " - in_C in 5..15 must always pad to 16 (no CM_CONV/4 special)";
    }
}

// ---------------------------------------------------------------------------
// Tests: compute_HW_MAC_operations_cnt
// ---------------------------------------------------------------------------

/**
 * @brief With no sparsity active, compute_HW_MAC_operations_cnt must equal
 *        compute_Ideal_MAC_operations_cnt for every device.
 */
TEST_F(TestHWPerformanceModel_DPUCycles, HWMAC_NoSparsity_EqualsIdeal_AllDevices) {
    for (const auto dev : all_devices()) {
        // Use >=16 channels so the channel branch is the same across all VPU>=2.7 devices
        DPUWorkload wl{
                dev,
                Operation::CONVOLUTION,
                {VPUTensor(6, 6, 32, 1, DataType::UINT8)},  // input
                {VPUTensor(4, 4, 16, 1, DataType::UINT8)},  // output
                {3, 3},                                     // kernels
                {1, 1},                                     // strides
                {0, 0, 0, 0},                               // padding
                ExecutionMode::CUBOID_16x16,
                ActivationFunction::NONE,
                0.0F,  // act_sparsity
                0.0F,  // weight_sparsity
                {swz_def, swz_def},
                {swz_def},
                1,
                {0, 0, 0, 0},
                ISIStrategy::CLUSTERING,
                false,  // weight_sparsity_enabled
        };
        const auto ideal = perf.compute_Ideal_MAC_operations_cnt(wl);
        const auto hw = perf.compute_HW_MAC_operations_cnt(wl);
        EXPECT_EQ(hw, ideal) << "HW MAC count must equal ideal when sparsity is off, device="
                             << VPUDevice_ToText.at(static_cast<int>(dev));
    }
}

/**
 * @brief Weight sparsity reduces compute_HW_MAC_operations_cnt below the ideal value.
 *
 * 60 % weight sparsity → 40 % non-zero operations → HW count = ceil(ideal * 0.4).
 * Device-independent: only the operation count changes, not which device is used.
 */
TEST_F(TestHWPerformanceModel_DPUCycles, HWMAC_WeightSparsity_ReducesCount) {
    constexpr float wt_sparsity{0.6F};  // 60 % zeros in weights

    for (const auto dev : all_devices()) {
        DPUWorkload wl{
                dev,
                Operation::CONVOLUTION,
                {VPUTensor(6, 6, 32, 1, DataType::UINT8)},
                {VPUTensor(4, 4, 16, 1, DataType::UINT8)},
                {3, 3},
                {1, 1},
                {0, 0, 0, 0},
                ExecutionMode::CUBOID_16x16,
                ActivationFunction::NONE,
                0.0F,  // act_sparsity (disabled)
                wt_sparsity,
                {swz_def, swz_def},
                {swz_def},
                1,
                {0, 0, 0, 0},
                ISIStrategy::CLUSTERING,
                true,  // weight_sparsity_enabled
        };
        const auto ideal = perf.compute_Ideal_MAC_operations_cnt(wl);
        const auto hw = perf.compute_HW_MAC_operations_cnt(wl);
        const auto expected =
                static_cast<unsigned long int>(std::ceil(static_cast<float>(ideal) * (1.0F - wt_sparsity)));

        EXPECT_EQ(hw, expected) << "Weight sparsity=" << wt_sparsity
                                << " device=" << VPUDevice_ToText.at(static_cast<int>(dev));
        EXPECT_LT(hw, ideal) << "HW count must be less than ideal when weight sparsity is active, device="
                             << VPUDevice_ToText.at(static_cast<int>(dev));
    }
}

/**
 * @brief Activation sparsity reduces compute_HW_MAC_operations_cnt below the ideal value.
 *
 * 70 % activation sparsity → 30 % non-zero operations.
 */
TEST_F(TestHWPerformanceModel_DPUCycles, HWMAC_ActSparsity_ReducesCount) {
    constexpr float act_sparsity{0.7F};  // 70 % zeros in activations

    for (const auto dev : all_devices()) {
        DPUWorkload wl{
                dev,
                Operation::CONVOLUTION,
                {VPUTensor(6, 6, 32, 1, DataType::UINT8)},
                {VPUTensor(4, 4, 16, 1, DataType::UINT8)},
                {3, 3},
                {1, 1},
                {0, 0, 0, 0},
                ExecutionMode::CUBOID_16x16,
                ActivationFunction::NONE,
                act_sparsity,  // act_sparsity
                0.0F,
                {swz_def, swz_def},
                {swz_def},
                1,
                {0, 0, 0, 0},
                ISIStrategy::CLUSTERING,
                false,
        };
        // Manually enable input sparsity on the tensor
        wl.inputs[0].set_sparsity(true);

        const auto ideal = perf.compute_Ideal_MAC_operations_cnt(wl);
        const auto hw = perf.compute_HW_MAC_operations_cnt(wl);
        const auto expected =
                static_cast<unsigned long int>(std::ceil(static_cast<float>(ideal) * (1.0F - act_sparsity)));

        EXPECT_EQ(hw, expected) << "Act sparsity=" << act_sparsity
                                << " device=" << VPUDevice_ToText.at(static_cast<int>(dev));
        EXPECT_LT(hw, ideal) << "HW count must be less than ideal when activation sparsity is active, device="
                             << VPUDevice_ToText.at(static_cast<int>(dev));
    }
}

/**
 * @brief When both weight and activation sparsity are active, the combined
 *        non-zero factor is the minimum of the two individual factors (conservative policy).
 */
TEST_F(TestHWPerformanceModel_DPUCycles, HWMAC_BothSparsities_UseMinimumFactor) {
    constexpr float wt_sparsity{0.4F};   // 60 % non-zero weights
    constexpr float act_sparsity{0.8F};  // 20 % non-zero activations  (more aggressive)

    for (const auto dev : all_devices()) {
        DPUWorkload wl{
                dev,
                Operation::CONVOLUTION,
                {VPUTensor(6, 6, 32, 1, DataType::UINT8)},
                {VPUTensor(4, 4, 16, 1, DataType::UINT8)},
                {3, 3},
                {1, 1},
                {0, 0, 0, 0},
                ExecutionMode::CUBOID_16x16,
                ActivationFunction::NONE,
                act_sparsity,
                wt_sparsity,
                {swz_def, swz_def},
                {swz_def},
                1,
                {0, 0, 0, 0},
                ISIStrategy::CLUSTERING,
                true,  // weight_sparsity_enabled
        };
        wl.inputs[0].set_sparsity(true);

        const auto ideal = perf.compute_Ideal_MAC_operations_cnt(wl);
        const auto hw = perf.compute_HW_MAC_operations_cnt(wl);

        // Conservative policy: minimum of the two non-zero factors
        const float wt_nz = 1.0F - wt_sparsity;
        const float act_nz = 1.0F - act_sparsity;
        const float combined = std::min(wt_nz, act_nz);  // min(0.6, 0.2) = 0.2
        const auto expected = static_cast<unsigned long int>(std::ceil(static_cast<float>(ideal) * combined));

        EXPECT_EQ(hw, expected) << "Both sparsities, combined factor=" << combined
                                << " device=" << VPUDevice_ToText.at(static_cast<int>(dev));
    }
}

// ---------------------------------------------------------------------------
// Tests: DPU_MAC_based_cycles
// ---------------------------------------------------------------------------

/**
 * @brief Generic: DPU_MAC_based_cycles equals ceil(MACs / nr_macs) for INT8 on every device.
 *
 * For INT8 the fp_ratio does not apply, so the formula is simply:
 *   cycles = ceil(mac_count / hw.get_nr_macs())
 *
 * The expected value is computed by reading nr_macs directly from the HW info,
 * making the test self-documenting regardless of which device is used.
 */
TEST_F(TestHWPerformanceModel_DPUCycles, MACCycles_Int8_UsesNrMacs_AllDevices) {
    // Use a fixed, easily computable MAC count: 1x1 kernel, output 4x4x32, 32 input channels
    // For VPU>=2.7: all channels >=16, so ideal = 1*1 * 4*4*32 * 32 = 16384
    // For VPU<2.7:  same formula (channels used directly) = 16384

    for (const auto dev : all_devices()) {
        DPUWorkload wl{
                dev,
                Operation::CONVOLUTION,
                {VPUTensor(4, 4, 32, 1, DataType::UINT8)},
                {VPUTensor(4, 4, 32, 1, DataType::UINT8)},
                {1, 1},
                {1, 1},
                {0, 0, 0, 0},
                ExecutionMode::CUBOID_16x16,
        };
        const unsigned long int mac_count = perf.compute_Ideal_MAC_operations_cnt(wl);
        const unsigned int nr_macs = perf.get_hw_info(dev).get_nr_macs();
        const unsigned long int expected = (mac_count + nr_macs - 1) / nr_macs;  // ceil division

        EXPECT_EQ(perf.DPU_MAC_based_cycles(wl, mac_count), expected)
                << "INT8 cycles device=" << VPUDevice_ToText.at(static_cast<int>(dev)) << " nr_macs=" << nr_macs
                << " mac_count=" << mac_count;
    }
}

/**
 * @brief Device-specific: FP16 input halves the effective MAC throughput via fp_ratio.
 *
 * For FP16 workloads the HW adjusts:
 *   effective_macs = nr_macs / fp_ratio
 *   cycles         = ceil(mac_count / effective_macs)
 *
 * VPU_2_0 / VPU_2_1: nr_macs=256,  fp_ratio=4  → effective=64
 * VPU_2_7 / VPU_4_0: nr_macs=2048, fp_ratio=2  → effective=1024
 * NPU 5/6/7:          nr_macs=4096, fp_ratio=2  → effective=2048
 *
 * The test verifies that the FP16 cycle count is strictly greater than the
 * INT8 cycle count for the same MAC count on every device.
 */
TEST_F(TestHWPerformanceModel_DPUCycles, MACCycles_FP16_MoreThanInt8_PerDevice) {
    // Fixed MAC count large enough to expose the fp_ratio difference on all devices
    constexpr unsigned long int mac_count{65536UL};

    for (const auto dev : all_devices()) {
        const unsigned int nr_macs = perf.get_hw_info(dev).get_nr_macs();
        const unsigned int fp_ratio = perf.get_hw_info(dev).get_fp_ratio();
        const unsigned int eff_macs_fp16 = (nr_macs + fp_ratio - 1) / fp_ratio;  // ceil_division

        const unsigned long int expected_fp16 = (mac_count + eff_macs_fp16 - 1) / eff_macs_fp16;
        const unsigned long int expected_i8 = (mac_count + nr_macs - 1) / nr_macs;

        // Build a FP16 workload on this device
        DPUWorkload wl_fp16 = make_wl(dev, DataType::FLOAT16, DataType::FLOAT16);
        DPUWorkload wl_i8 = make_wl(dev, DataType::UINT8, DataType::UINT8);

        EXPECT_EQ(perf.DPU_MAC_based_cycles(wl_fp16, mac_count), expected_fp16)
                << "FP16 cycles device=" << VPUDevice_ToText.at(static_cast<int>(dev)) << " nr_macs=" << nr_macs
                << " fp_ratio=" << fp_ratio;

        EXPECT_EQ(perf.DPU_MAC_based_cycles(wl_i8, mac_count), expected_i8)
                << "INT8 cycles device=" << VPUDevice_ToText.at(static_cast<int>(dev));

        EXPECT_GT(expected_fp16, expected_i8) << "FP16 must cost more cycles than INT8 due to fp_ratio, device="
                                              << VPUDevice_ToText.at(static_cast<int>(dev));
    }
}

// NOTE: DPUWorkload::outputs is std::array<VPUTensor,1> (fixed size = 1), so DPU_MAC_based_cycles never sees an empty output list.

/**
 * @brief DCIM + UINT16 input costs more cycles than INT8 via the denominator only.
 *
 * `compute_Ideal_MAC_operations_cnt` produces the same value for UINT8 and UINT16
 * (no cim_16bit_factor scaling in the numerator).  The extra cost for UINT16 comes
 * entirely from `get_dtype_mac_factor` reducing the effective MACs in the denominator
 * of `DPU_MAC_based_cycles`:
 *
 *   mac_u16 == mac_i8   (same logical operation count)
 *
 *   effective_macs_i8  = nr_macs / get_dtype_mac_factor(wl_i8,  hw) = nr_macs
 *   effective_macs_u16 = nr_macs / get_dtype_mac_factor(wl_u16, hw) = nr_macs / hw_macs_per_multibyte_op
 *
 *   cycles_u16 = ceil(mac_u16 / effective_macs_u16)
 *              > ceil(mac_i8  / effective_macs_i8 )
 *
 * VPU_2_0 and VPU_2_1 are excluded: no UINT16 / DCIM hardware support.
 */
TEST_F(TestHWPerformanceModel_DPUCycles, MACCycles_DCIM_UINT16_IncreasesCycles_ViaFactor_PerDevice) {
    // VPU_2_0 and VPU_2_1 excluded: no UINT16 / DCIM support.
    std::vector<VPUDevice> dcim_devices{
            VPUDevice::VPU_2_7,
            VPUDevice::VPU_4_0,
            VPUDevice::NPU_5_0,
            VPUDevice::NPU_5_0_W,
    };

    // A fixed MAC geometry: 1x1 kernel, output 4x4x32, 32 input channels.
    // operations_cnt_base = 1*1 * 4*4*32 = 512
    // INT8  DCIM: mac_count = 512 * 32 * 1                          = 16384
    // UINT16 DCIM: mac_count = 512 * 32 * hw_macs_per_multibyte_op  (per device)

    for (const auto dev : dcim_devices) {
        const auto& hw = perf.get_hw_info(dev);
        const unsigned int nr_macs = hw.get_nr_macs();

        // Build INT8 DCIM workload
        DPUWorkload wl_i8{
                dev,
                Operation::CONVOLUTION,
                {VPUTensor(4, 4, 32, 1, DataType::UINT8)},
                {VPUTensor(4, 4, 32, 1, DataType::UINT8)},
                {1, 1},
                {1, 1},
                {0, 0, 0, 0},
                ExecutionMode::CUBOID_16x16,
        };
        wl_i8.mpe_engine = MPEEngine::DCIM;

        // Build UINT16 DCIM workload — identical geometry, only input dtype changes
        DPUWorkload wl_u16{
                dev,
                Operation::CONVOLUTION,
                {VPUTensor(4, 4, 32, 1, DataType::UINT16)},
                {VPUTensor(4, 4, 32, 1, DataType::UINT16)},
                {1, 1},
                {1, 1},
                {0, 0, 0, 0},
                ExecutionMode::CUBOID_16x16,
        };
        wl_u16.mpe_engine = MPEEngine::DCIM;

        // Get factors from the new method — this is the single source of truth.
        // INT8:  factor = 1  (i8 family)
        // UINT16: factor = hw_macs_per_multibyte_op (i16 family → get_fp_ratio())
        const unsigned int factor_i8 = perf.get_dtype_mac_factor(wl_i8, hw);
        const unsigned int hw_macs_per_multibyte_op = perf.get_dtype_mac_factor(wl_u16, hw);

        const unsigned long int mac_i8 = perf.compute_Ideal_MAC_operations_cnt(wl_i8);
        const unsigned long int mac_u16 = perf.compute_Ideal_MAC_operations_cnt(wl_u16);

        // mac_u16 == mac_i8: compute_Ideal_MAC_operations_cnt applies no scaling for dtype;
        // the UINT16 cost is captured entirely in the denominator via get_dtype_mac_factor.
        EXPECT_EQ(mac_u16, mac_i8) << "UINT16 and INT8 DCIM must have the same logical MAC count, device="
                                   << VPUDevice_ToText.at(static_cast<int>(dev));

        // Effective MACs = ceil(nr_macs / factor) — the cost difference is in the denominator only.
        // INT8:  eff_macs = nr_macs (factor=1)
        // UINT16: eff_macs = nr_macs / hw_macs_per_multibyte_op
        const unsigned int eff_macs_i8 = (nr_macs + factor_i8 - 1) / factor_i8;
        const unsigned int eff_macs_u16 = (nr_macs + hw_macs_per_multibyte_op - 1) / hw_macs_per_multibyte_op;

        const unsigned long int expected_i8 = (mac_i8 + eff_macs_i8 - 1) / eff_macs_i8;
        const unsigned long int expected_u16 = (mac_u16 + eff_macs_u16 - 1) / eff_macs_u16;

        EXPECT_EQ(perf.DPU_MAC_based_cycles(wl_i8, mac_i8), expected_i8)
                << "INT8  DCIM cycles device=" << VPUDevice_ToText.at(static_cast<int>(dev)) << " nr_macs=" << nr_macs
                << " mac_count=" << mac_i8;

        EXPECT_EQ(perf.DPU_MAC_based_cycles(wl_u16, mac_u16), expected_u16)
                << "UINT16 DCIM cycles device=" << VPUDevice_ToText.at(static_cast<int>(dev)) << " nr_macs=" << nr_macs
                << " hw_macs_per_multibyte_op=" << hw_macs_per_multibyte_op << " mac_count=" << mac_u16;

        // UINT16 DCIM must cost more cycles than INT8 DCIM: same MAC count, smaller effective MACs
        EXPECT_GT(expected_u16, expected_i8) << "UINT16 DCIM must cost more cycles than INT8 DCIM"
                                             << " (hw_macs_per_multibyte_op=" << hw_macs_per_multibyte_op
                                             << "), device=" << VPUDevice_ToText.at(static_cast<int>(dev));
    }
}

// ---------------------------------------------------------------------------
// Tests: get_dtype_mac_factor
// ---------------------------------------------------------------------------

/**
 * @brief Tests for `HWPerformanceModel::get_dtype_mac_factor`.
 *
 * The method returns the factor by which a workload's datatype increases MAC
 * hardware resource consumption relative to the 8-bit baseline:
 *   - FP16-family (FLOAT16, BFLOAT16, FLOAT32) input or weights → fp_ratio
 *   - i16 integer (INT16, UINT16) input with non-float weights   → fp_ratio
 *   - Everything else (INT8, FP8, FP4, sub-8-bit, ...)           → 1
 *
 * Because the factor for 16-bit types is `hw.get_fp_ratio()`, the test reads
 * that value live from `get_hw_info()` so it remains correct for every device.
 *
 * Struct layout: { in_dt, out_dt, optional wt_dt, expected_factor_is_fp_ratio, info }
 * where `expected_factor_is_fp_ratio=true` means expected == hw.get_fp_ratio(),
 *       `expected_factor_is_fp_ratio=false` means expected == 1.
 */
class TestHWPerformanceModel_DtypeMacFactor : public TestHWPerformanceModel_Base {
protected:
    struct DtypeMacFactorCase {
        DataType in_dt;
        DataType out_dt;
        std::optional<DataType> wt_dt;  ///< nullopt → weight type defaults to in_dt
        bool expected_is_fp_ratio;      ///< true → factor == hw.get_fp_ratio(), false → factor == 1
        const char* info;
    };

    // clang-format off
    const std::vector<DtypeMacFactorCase> cases{
        // FP16-family input: fp16 path fires → factor = fp_ratio
        {DataType::FLOAT16,  DataType::UINT8,  {},                    true,  "FLOAT16 input"},
        {DataType::BFLOAT16, DataType::UINT8,  {},                    true,  "BFLOAT16 input"},
        {DataType::FLOAT32,  DataType::UINT8,  {},                    true,  "FLOAT32 input (fp16family)"},
        // FP16 weights with integer input: fp16 path fires via weights → factor = fp_ratio
        {DataType::UINT8,    DataType::UINT8,  DataType::FLOAT16,     true,  "INT8 input, FP16 weights"},
        {DataType::UINT8,    DataType::UINT8,  DataType::BFLOAT16,    true,  "INT8 input, BFLOAT16 weights"},
        // i16 integer family, non-float weights: i16 path fires → factor = fp_ratio
        {DataType::UINT16,   DataType::UINT8,  {},                    true,  "UINT16 input, default weights"},
        {DataType::INT16,    DataType::UINT8,  {},                    true,  "INT16 input, default weights"},
        {DataType::UINT16,   DataType::UINT8,  DataType::INT8,        true,  "UINT16 input, INT8 weights"},
        // i16 input + float weights: fp16 fires (via weights), i16 fires too → fp_ratio (union)
        {DataType::UINT16,   DataType::UINT8,  DataType::FLOAT16,     true,  "UINT16 input, FP16 weights (fp16 path fires)"},
        // INT8-family input, integer weights: neither path fires → factor = 1
        {DataType::UINT8,    DataType::UINT8,  {},                    false, "UINT8 input, default weights"},
        {DataType::INT8,     DataType::UINT8,  {},                    false, "INT8 input"},
        {DataType::UINT4,    DataType::UINT8,  {},                    false, "UINT4 input"},
        {DataType::INT4,     DataType::UINT8,  {},                    false, "INT4 input"},
        {DataType::INT32,    DataType::INT32,  {},                    false, "INT32 input"},
        // FP8-family input, non-fp16 weights: fp8 path fires, but NOT fp16 → factor = 1
        {DataType::BF8,      DataType::UINT8,  {},                    false, "BF8 input, default INT8 weights"},
        {DataType::HF8,      DataType::UINT8,  {},                    false, "HF8 input, default INT8 weights"},
        // FP8 input + FP16 weights: fp16 path fires via weights → factor = fp_ratio
        {DataType::BF8,      DataType::UINT8,  DataType::FLOAT16,     true,  "BF8 input, FP16 weights"},
        // FLOAT4: not fp16-family, not i16 → factor = 1
        {DataType::FLOAT4,   DataType::UINT8,  {},                    false, "FLOAT4 input"},
    };
    // clang-format on
};

/**
 * @brief get_dtype_mac_factor returns fp_ratio for 16-bit types and 1 for all others.
 *
 * Iterates the full case table on every device.  The expected value is computed
 * live from `get_hw_info(dev).get_fp_ratio()` so the test is self-consistent even
 * if fp_ratio changes for a device in the future.
 */
TEST_F(TestHWPerformanceModel_DtypeMacFactor, FactorMatchesExpected_AllDevices) {
    for (const auto& tc : cases) {
        for (const auto dev : all_devices()) {
            const auto& hw = perf.get_hw_info(dev);
            const unsigned int fp_ratio = hw.get_fp_ratio();
            const unsigned int expected = tc.expected_is_fp_ratio ? fp_ratio : 1u;

            const DPUWorkload wl = make_wl(dev, tc.in_dt, tc.out_dt, tc.wt_dt);

            EXPECT_EQ(perf.get_dtype_mac_factor(wl, hw), expected)
                    << "get_dtype_mac_factor mismatch"
                    << " | case: " << tc.info << " | device: " << VPUDevice_ToText.at(static_cast<int>(dev))
                    << " | fp_ratio: " << fp_ratio;
        }
    }
}

/**
 * @brief For datatypes that produce factor=1, the result is literally 1 regardless of
 *        what fp_ratio the device carries (including VPU_2_0/2_1 with fp_ratio=4).
 */
TEST_F(TestHWPerformanceModel_DtypeMacFactor, Factor1_IsAlwaysOne_NotFpRatio) {
    for (const auto& tc : cases) {
        if (tc.expected_is_fp_ratio) {
            continue;  // only checking the factor=1 cases here
        }
        for (const auto dev : all_devices()) {
            const auto& hw = perf.get_hw_info(dev);
            const DPUWorkload wl = make_wl(dev, tc.in_dt, tc.out_dt, tc.wt_dt);

            EXPECT_EQ(perf.get_dtype_mac_factor(wl, hw), 1u)
                    << "Expected factor=1 (not fp_ratio=" << hw.get_fp_ratio() << ")"
                    << " | case: " << tc.info << " | device: " << VPUDevice_ToText.at(static_cast<int>(dev));
        }
    }
}

/**
 * @brief For datatypes that produce factor=fp_ratio, the factor tracks the device's
 *        fp_ratio exactly — so VPU_2_0/2_1 (fp_ratio=4) produce 4, not 2.
 */
TEST_F(TestHWPerformanceModel_DtypeMacFactor, FactorFpRatio_TracksDeviceFpRatio) {
    for (const auto& tc : cases) {
        if (!tc.expected_is_fp_ratio) {
            continue;  // only checking the fp_ratio cases here
        }
        for (const auto dev : all_devices()) {
            const auto& hw = perf.get_hw_info(dev);
            const unsigned int fp_ratio = hw.get_fp_ratio();
            const DPUWorkload wl = make_wl(dev, tc.in_dt, tc.out_dt, tc.wt_dt);

            EXPECT_EQ(perf.get_dtype_mac_factor(wl, hw), fp_ratio)
                    << "factor must equal device fp_ratio=" << fp_ratio << " | case: " << tc.info
                    << " | device: " << VPUDevice_ToText.at(static_cast<int>(dev));
        }
    }
}

// ---------------------------------------------------------------------------
// Fixture: TestHWPerformanceModel_NativeComp
// ---------------------------------------------------------------------------

/**
 * @brief Tests that all native_comp_* classification methods in HWPerformanceModel
 *        are purely data-type-driven and completely device-independent.
 *
 * The methods under test (`native_comp_on_fp16`, `native_comp_on_fp8`, `native_comp_on_i16`, `native_comp_on_i8`) inspect only
 * the input/weight tensor data types. They must therefore return the same value regardless of the VPUDevice set on the workload.
 *

 *
 * Each TEST_F iterates over the full device list (with NPU_RESERVED/NPU_RESERVED_1 embargo guards)
 * and verifies:
 *   1. The result is consistent across all devices for the same dtype combination.
 *   2. The result matches the expected classification documented in the test cases.
 */
class TestHWPerformanceModel_NativeComp : public TestHWPerformanceModel_Base {
protected:
    /// Describes one test scenario: dtype combination + expected result for all methods.
    struct NativeCompCase {
        DataType in_dt;
        DataType out_dt;
        std::optional<DataType> wt_dt;  ///< nullopt means same as input (default)

        bool exp_on_fp16;
        bool exp_on_fp8;
        bool exp_on_i16;
        bool exp_on_i8;

        const char* info;  ///< human-readable label for assertion messages
    };

    /// @brief The canonical test-case table used by all per-method tests.
    ///        Covers every relevant dtype partition so every branch of each method
    ///        is exercised at least once.
    const std::vector<NativeCompCase> cases{
            // clang-format off
            // in_dt         out_dt          wt_dt         fp16   fp8   i16   i8    info
            {DataType::UINT8,    DataType::UINT8,    {},           false, false, false, true,  "i8 input, i8 weights (default)"},
            {DataType::INT8,     DataType::INT8,     {},           false, false, false, true,  "i8 input INT8"},
            {DataType::FLOAT16,  DataType::UINT8,    {},           true,  false, false, false, "fp16 input"},
            {DataType::BFLOAT16, DataType::UINT8,    {},           true,  false, false, false, "bf16 input"},
            {DataType::BF8,      DataType::UINT8,    {},           false, true,  false, false, "bf8 input, i8 weights (default)"},
            {DataType::HF8,      DataType::UINT8,    {},           false, true,  false, false, "hf8 input, i8 weights (default)"},
            {DataType::FLOAT4,   DataType::UINT8,    {},           false, false, false, false, "float4 input"},
            {DataType::UINT16,   DataType::UINT8,    {},           false, false, true,  false, "uint16 input (i16 family)"},
            {DataType::INT16,    DataType::UINT8,    {},           false, false, true,  false, "int16 input (i16 family)"},
            {DataType::INT4,     DataType::INT4,     {},           false, false, false, true,  "int4 input and output"},
            {DataType::UINT4,    DataType::UINT8,    {},           false, false, false, true,  "uint4 input"},
            {DataType::INT32,    DataType::INT32,    {},           false, false, false, true,  "int32 treated as i8 family"},
            // weight type overrides --
            // fp16 weights lift computation to fp16 even when input is integer
            {DataType::UINT8,    DataType::UINT8,    DataType::FLOAT16,  true,  false, false, false, "i8 input, fp16 weights -> fp16 compute"},
            // fp8 input + fp16 weights: fp16 wins for native_comp_on_fp16; fp8 is suppressed
            {DataType::HF8,      DataType::UINT8,    DataType::FLOAT16,  true,  false, false, false, "hf8 input, fp16 weights -> fp16 compute"},
            // fp8 input + i8 weights: pure fp8 path
            {DataType::BF8,      DataType::UINT8,    DataType::UINT8,    false, true,  false, false, "bf8 input, i8 weights -> fp8 compute"},
            // i8 input + fp8 weights: bf8 is_any_float_dtype()==true, so it disqualifies the i8 compute path
            {DataType::UINT8,    DataType::UINT8,    DataType::BF8,      false, false, false, false, "i8 input, bf8 weights -> bf8 is float, disqualifies i8 compute"},
            // clang-format on
    };
};

// ---------------------------------------------------------------------------
// Tests: device-independence for each native_comp_* method
// ---------------------------------------------------------------------------

/**
 * @brief native_comp_on_fp16 returns the same value for every device.
 *
 * Returns true when the input or weight tensor belongs to the fp16 family
 * (FLOAT16, BFLOAT16, FLOAT32). FP8 / FP4 inputs without FP16 weights return false.
 */
TEST_F(TestHWPerformanceModel_NativeComp, OnFP16_DeviceIndependent) {
    for (const auto& tc : cases) {
        const bool reference = tc.exp_on_fp16;

        for (const auto dev : all_devices()) {
            const DPUWorkload wl = make_wl(dev, tc.in_dt, tc.out_dt, tc.wt_dt);
            EXPECT_EQ(perf.native_comp_on_fp16(wl), reference)
                    << "native_comp_on_fp16 mismatch on device " << VPUDevice_ToText.at(static_cast<int>(dev))
                    << " | case: " << tc.info;
        }
    }
}

/**
 * @brief native_comp_on_fp8 returns the same value for every device.
 *
 * Returns true when the input tensor is an FP8 type (BF8 or HF8) AND the
 * weight tensor is NOT fp16-family. Combining FP8 input with FP16 weights
 * promotes computation to the fp16 path instead.
 */
TEST_F(TestHWPerformanceModel_NativeComp, OnFP8_DeviceIndependent) {
    for (const auto& tc : cases) {
        const bool reference = tc.exp_on_fp8;

        for (const auto dev : all_devices()) {
            const DPUWorkload wl = make_wl(dev, tc.in_dt, tc.out_dt, tc.wt_dt);
            EXPECT_EQ(perf.native_comp_on_fp8(wl), reference)
                    << "native_comp_on_fp8 mismatch on device " << VPUDevice_ToText.at(static_cast<int>(dev))
                    << " | case: " << tc.info;
        }
    }
}

/**
 * @brief native_comp_on_i16 returns the same value for every device.
 *
 * Returns true when the input tensor is 16-bit integer (INT16 / UINT16) AND
 * the weight tensor is not a floating-point type.
 */
TEST_F(TestHWPerformanceModel_NativeComp, OnI16_DeviceIndependent) {
    for (const auto& tc : cases) {
        const bool reference = tc.exp_on_i16;

        for (const auto dev : all_devices()) {
            const DPUWorkload wl = make_wl(dev, tc.in_dt, tc.out_dt, tc.wt_dt);
            EXPECT_EQ(perf.native_comp_on_i16(wl), reference)
                    << "native_comp_on_i16 mismatch on device " << VPUDevice_ToText.at(static_cast<int>(dev))
                    << " | case: " << tc.info;
        }
    }
}

/**
 * @brief native_comp_on_i8 returns the same value for every device.
 *
 * Returns true when the input tensor is integer (i8 family: UINT8, INT8, sub-8-bit,
 * INT32) AND the weight tensor is not a floating-point type. INT16/UINT16 inputs
 * are NOT part of the i8 family.
 */
TEST_F(TestHWPerformanceModel_NativeComp, OnI8_DeviceIndependent) {
    for (const auto& tc : cases) {
        const bool reference = tc.exp_on_i8;

        for (const auto dev : all_devices()) {
            const DPUWorkload wl = make_wl(dev, tc.in_dt, tc.out_dt, tc.wt_dt);
            EXPECT_EQ(perf.native_comp_on_i8(wl), reference)
                    << "native_comp_on_i8 mismatch on device " << VPUDevice_ToText.at(static_cast<int>(dev))
                    << " | case: " << tc.info;
        }
    }
}

/**
 * @brief Verify that exactly one native_comp_on_* method returns true for
 *        every case and every device (the categories are mutually exclusive).
 *
 * The four precision paths — fp16, fp8, i16, i8 — must not overlap.
 * A workload classified as fp16 must not simultaneously be fp8, i16, or i8,
 * and so on for every combination.
 */
TEST_F(TestHWPerformanceModel_NativeComp, PrecisionCategories_MutuallyExclusive) {
    for (const auto& tc : cases) {
        for (const auto dev : all_devices()) {
            const DPUWorkload wl = make_wl(dev, tc.in_dt, tc.out_dt, tc.wt_dt);

            const bool on_fp16 = perf.native_comp_on_fp16(wl);
            const bool on_fp8 = perf.native_comp_on_fp8(wl);
            const bool on_i16 = perf.native_comp_on_i16(wl);
            const bool on_i8 = perf.native_comp_on_i8(wl);

            // At most one precision category may be true at a time.
            const int active_count = static_cast<int>(on_fp16) + static_cast<int>(on_fp8) + static_cast<int>(on_i16) +
                                     static_cast<int>(on_i8);

            EXPECT_LE(active_count, 1) << "More than one precision category active on device "
                                       << VPUDevice_ToText.at(static_cast<int>(dev)) << " | case: " << tc.info
                                       << " | fp16=" << on_fp16 << " fp8=" << on_fp8 << " i16=" << on_i16
                                       << " i8=" << on_i8;
        }
    }
}

}  // namespace VPUNN_unit_tests
