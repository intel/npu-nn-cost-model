// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the "Software Package")
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the "third-party-programs.txt" or other similarly-named text file included with the
// Software Package for additional details.

// ---------------------------------------------------------------------------
// Tests for 149253: NPU6 ODU autopad relaxed to allow ANY output channel
// count — including values > 16 that are NOT multiples of 16.
//
// Background
// ----------
// Before NPU5, all output channels had to be aligned to 16. NPU5 introduced ODU
// autopad (164800), which allowed unaligned channels in the range [1, 15].
// NPU6 further relaxes ODU autopad so that ANY output channel count is valid when
// output_autopad=true — including values such as 17, 20, 24, 30, 31, 33, 40, …
//
// Current state (CAUSES THESE TESTS TO FAIL)
// -------------------------------------------
// VPU6_0_WorkloadValidValues::get_output_channels_restriction() returns, when
// output_autopad=true:
//   { SmartRanges(1, 15, 1) OR SmartRanges(16, max, 16) }
// i.e. channels in [1..15] or multiples-of-16.  Values such as 17, 20, 24, 30,
// 31, 33, 40, … are rejected with ERROR_INVALID_INPUT_CONFIGURATION.
//
// Required behaviour after fix
// ----------------------------
// With output_autopad=true the restriction should accept any value ≥ 1, e.g.
//   SmartRanges(1, max, 1)
// or equivalent.  Values > 16 that are not multiples of 16 MUST be accepted.
//
// Test structure
// --------------
// SECTION 1 – Parameterised sanitiser tests (DPU_OperationSanitizer level)
//   • Primary ops: CONVOLUTION, ELTWISE, MAXPOOL, AVEPOOL
//   • Channels under test: 17, 20, 24, 30, 31, 33, 40
//   • These tests currently FAIL (sane.value() != NO_ERROR) and will pass once
//     the output_channels_restrictions are broadened for NPU6.
//
// SECTION 2 – Contrast / control tests (must remain stable)
//   • Without output_autopad: unaligned >16 should still be rejected (no change).
//   • NPU5 with output_autopad and channels > 16 should still be rejected (NPU5
//     only allows unaligned for channels < 16).
//   • Aligned channels (16, 32, 48) with output_autopad should continue to pass.
// ---------------------------------------------------------------------------

#include "vpu/validation/dpu_operations_sanitizer.h"
#include "vpu/validation/dpu_operations_validator.h"
#include "vpu/utils.h"

#include <gtest/gtest.h>

#include <string>
#include <tuple>
#include <vector>

#include "common/common_helpers.h"

namespace VPUNN_unit_tests {
using namespace VPUNN;

// ---------------------------------------------------------------------------
// Helper: build a compact workload description for assertion messages
// ---------------------------------------------------------------------------
static std::string wl_desc(const char* op, unsigned int out_ch, bool autopad) {
    return std::string(op) + " out_ch=" + std::to_string(out_ch) +
           " output_autopad=" + (autopad ? "true" : "false");
}

// ============================================================================
// SECTION 1 – Sanitiser-level tests
//
// Each test body covers multiple unaligned-channel values > 16.
// Expected outcome: NO_ERROR (currently FAILS because these values are outside
// the current output_channels_restrictions for NPU6 with autopad).
// ============================================================================

class DPU_NPU6_OutputAutopadUnalignedGt16_SanitizerTest : public ::testing::Test {
protected:
    DPU_OperationSanitizer dut;
    SanityReport sane;

    const VPUDevice npu6{VPUDevice::NPU_6_0};
    const VPUDevice npu5{VPUDevice::NPU_5_0};

    // Unaligned output channel counts that are > 16 and must be accepted by NPU6
    // with output_autopad=true after the fix.
    const std::vector<unsigned int> unaligned_gt16_channels{17u, 20u, 24u, 30u, 31u};
};

// ---------------------------------------------------------------------------
// 1.1  CONVOLUTION — unaligned output channels > 16, output_autopad = true
//
// Shape: 32 input channels (aligned, valid without autopad), spatial 28×28.
// output channels iterate over unaligned_gt16_channels.
//
// The input channel count is deliberately kept at a valid value (32) so that
// any rejection is caused exclusively by the OUTPUT channel restriction.
// ---------------------------------------------------------------------------
TEST_F(DPU_NPU6_OutputAutopadUnalignedGt16_SanitizerTest,
       CONV_OutputAutopad_UnalignedCh_GT16_ShouldPassValidation) {
    // EXPECTED TO FAIL before 149253 is implemented.
    // After fix: all iterations must return NO_ERROR.

    for (const auto out_ch : unaligned_gt16_channels) {
        DPUWorkload wl{
                npu6,
                Operation::CONVOLUTION,
                {VPUTensor(28, 28, 32, 1, DataType::UINT8)},                              // input
                {VPUTensor(28, 28, out_ch, 1, DataType::UINT8)},                           // output (unaligned >16)
                {1, 1},                                                                    // kernels
                {1, 1},                                                                    // strides
                {0, 0, 0, 0},                                                              // padding
                ExecutionMode::CUBOID_16x16,                                               // exec mode
                ActivationFunction::NONE,                                                  // activation
                0.0F,                                                                      // act_sparsity
                0.0F,                                                                      // weight_sparsity
                {swz_def, swz_def},                                                        // input_swizzling
                {swz_def},                                                                 // output_swizzling
                1,                                                                         // output_write_tiles
                {0, 0, 0, 0},                                                              // offsets
                ISIStrategy::CLUSTERING,                                                   // isi_strategy
                false,                                                                     // weight_sparsity_enabled
        };
        wl.output_autopad = true;

        dut.check_and_sanitize(wl, sane);
        EXPECT_EQ(sane.value(), V(Cycles::NO_ERROR))
                << "FAILS: " << wl_desc("CONV", out_ch, true)
                << "\n  error: " << Cycles::toErrorText(sane.value()) << " | " << sane.info
                << "\n  Fix needed in VPU6_0_WorkloadValidValues::get_output_channels_restriction()"
                << "\n  (149253: NPU6 ODU autopad must accept any ch count, including >16 unaligned)";
    }
}

// ---------------------------------------------------------------------------
// 1.2  ELTWISE — unaligned output channels > 16, output_autopad = true
//
// ELTWISE with output < input channels (reduce-like, analogous to the
// existing autopad tests with output = 1 ch).
// input: 32 channels, output: unaligned value > 16.
// ---------------------------------------------------------------------------
TEST_F(DPU_NPU6_OutputAutopadUnalignedGt16_SanitizerTest,
       ELTWISE_OutputAutopad_UnalignedCh_GT16_ShouldPassValidation) {
    for (const auto out_ch : unaligned_gt16_channels) {
        DPUWorkload wl{
                npu6,
                Operation::ELTWISE,
                {VPUTensor(28, 9, 32, 1, DataType::UINT8)},   // input (aligned)
                {VPUTensor(28, 9, out_ch, 1, DataType::UINT8)},  // output (unaligned >16)
                {1, 1},
                {1, 1},
                {0, 0, 0, 0},
                ExecutionMode::CUBOID_8x16,
                ActivationFunction::NONE,
                0.0F,
                0.0F,
                {swz_def, swz_def},
                {swz_def},
                1,
                {0, 0, 0, 0},
                ISIStrategy::CLUSTERING,
                false,
        };
        wl.output_autopad = true;
        wl.superdense_memory = true;

        dut.check_and_sanitize(wl, sane);
        EXPECT_EQ(sane.value(), V(Cycles::NO_ERROR))
                << "FAILS: " << wl_desc("ELTWISE", out_ch, true)
                << "\n  error: " << Cycles::toErrorText(sane.value()) << " | " << sane.info
                << "\n  Fix needed in VPU6_0_WorkloadValidValues::get_output_channels_restriction()";
    }
}

// ---------------------------------------------------------------------------
// 1.3  MAXPOOL — unaligned output channels > 16, output_autopad = true
//
// input: 32 channels (aligned), output: unaligned > 16.
// ---------------------------------------------------------------------------
TEST_F(DPU_NPU6_OutputAutopadUnalignedGt16_SanitizerTest,
       MAXPOOL_OutputAutopad_UnalignedCh_GT16_ShouldPassValidation) {
    for (const auto out_ch : unaligned_gt16_channels) {
        DPUWorkload wl{
                npu6,
                Operation::MAXPOOL,
                {VPUTensor(14, 14, 32, 1, DataType::UINT8)},
                {VPUTensor(14, 14, out_ch, 1, DataType::UINT8)},
                {1, 1},
                {1, 1},
                {0, 0, 0, 0},
                ExecutionMode::CUBOID_16x16,
                ActivationFunction::NONE,
                0.0F,
                0.0F,
                {swz_def, swz_def},
                {swz_def},
                1,
                {0, 0, 0, 0},
                ISIStrategy::CLUSTERING,
                false,
        };
        wl.output_autopad = true;

        dut.check_and_sanitize(wl, sane);
        EXPECT_EQ(sane.value(), V(Cycles::NO_ERROR))
                << "FAILS: " << wl_desc("MAXPOOL", out_ch, true)
                << "\n  error: " << Cycles::toErrorText(sane.value()) << " | " << sane.info
                << "\n  Fix needed in VPU6_0_WorkloadValidValues::get_output_channels_restriction()";
    }
}

// ---------------------------------------------------------------------------
// 1.4  AVEPOOL — unaligned output channels > 16, output_autopad = true
//
// input: 32 channels (aligned), output: unaligned > 16.
// ---------------------------------------------------------------------------
TEST_F(DPU_NPU6_OutputAutopadUnalignedGt16_SanitizerTest,
       AVEPOOL_OutputAutopad_UnalignedCh_GT16_ShouldPassValidation) {
    for (const auto out_ch : unaligned_gt16_channels) {
        DPUWorkload wl{
                npu6,
                Operation::AVEPOOL,
                {VPUTensor(32, 20, 32, 1, DataType::UINT8, Layout::ZXY)},
                {VPUTensor(32, 20, out_ch, 1, DataType::UINT8, Layout::XYZ)},
                {1, 1},
                {1, 1},
                {0, 0, 0, 0},
                ExecutionMode::CUBOID_16x16,
                ActivationFunction::NONE,
                0.0F,
                0.0F,
                {swz_def, swz_def},
                {swz_def},
                1,
                {0, 0, 0, 0},
                ISIStrategy::CLUSTERING,
                false,
        };
        wl.output_autopad = true;

        dut.check_and_sanitize(wl, sane);
        EXPECT_EQ(sane.value(), V(Cycles::NO_ERROR))
                << "FAILS: " << wl_desc("AVEPOOL", out_ch, true)
                << "\n  error: " << Cycles::toErrorText(sane.value()) << " | " << sane.info
                << "\n  Fix needed in VPU6_0_WorkloadValidValues::get_output_channels_restriction()";
    }
}

// ---------------------------------------------------------------------------
// 1.5  FLOAT16 datatypes — CONV with FP16 I/O and unaligned output > 16
//
// Verifies the fix applies regardless of datatype.
// ---------------------------------------------------------------------------
TEST_F(DPU_NPU6_OutputAutopadUnalignedGt16_SanitizerTest,
       CONV_FP16_OutputAutopad_UnalignedCh_GT16_ShouldPassValidation) {
    const std::vector<unsigned int> channels_fp16{17u, 25u, 31u, 33u};

    for (const auto out_ch : channels_fp16) {
        DPUWorkload wl{
                npu6,
                Operation::CONVOLUTION,
                {VPUTensor(14, 14, 32, 1, DataType::FLOAT16, Layout::ZXY)},
                {VPUTensor(14, 14, out_ch, 1, DataType::FLOAT16, Layout::ZXY)},
                {1, 1},
                {1, 1},
                {0, 0, 0, 0},
                ExecutionMode::CUBOID_16x16,
                ActivationFunction::NONE,
                0.0F,
                0.0F,
                {swz_def, swz_def},
                {swz_def},
                1,
                {0, 0, 0, 0},
                ISIStrategy::CLUSTERING,
                false,
        };
        wl.output_autopad = true;

        dut.check_and_sanitize(wl, sane);
        EXPECT_EQ(sane.value(), V(Cycles::NO_ERROR))
                << "FAILS: " << wl_desc("CONV FP16", out_ch, true)
                << "\n  error: " << Cycles::toErrorText(sane.value()) << " | " << sane.info
                << "\n  Fix needed in VPU6_0_WorkloadValidValues::get_output_channels_restriction()";
    }
}

// ---------------------------------------------------------------------------
// 1.6  CONV with larger unaligned channels (> 32, not multiple of 16)
//
// Tests channel values well above 32 to confirm the fix is not limited to
// the 16–31 range.  E.g. 33, 40, 49, 52, 65.
// ---------------------------------------------------------------------------
TEST_F(DPU_NPU6_OutputAutopadUnalignedGt16_SanitizerTest,
       CONV_OutputAutopad_LargerUnalignedCh_ShouldPassValidation) {
    // All of these are > 32 and NOT multiples of 16
    const std::vector<unsigned int> larger_unaligned{33u, 40u, 49u, 52u, 65u};

    for (const auto out_ch : larger_unaligned) {
        DPUWorkload wl{
                npu6,
                Operation::CONVOLUTION,
                {VPUTensor(16, 16, 64, 1, DataType::UINT8)},
                {VPUTensor(16, 16, out_ch, 1, DataType::UINT8)},
                {1, 1},
                {1, 1},
                {0, 0, 0, 0},
                ExecutionMode::CUBOID_16x16,
                ActivationFunction::NONE,
                0.0F,
                0.0F,
                {swz_def, swz_def},
                {swz_def},
                1,
                {0, 0, 0, 0},
                ISIStrategy::CLUSTERING,
                false,
        };
        wl.output_autopad = true;

        dut.check_and_sanitize(wl, sane);
        EXPECT_EQ(sane.value(), V(Cycles::NO_ERROR))
                << "FAILS: " << wl_desc("CONV large", out_ch, true)
                << "\n  error: " << Cycles::toErrorText(sane.value()) << " | " << sane.info
                << "\n  Fix needed in VPU6_0_WorkloadValidValues::get_output_channels_restriction()";
    }
}

// ============================================================================
// SECTION 2 – Control / contrast tests
//
// These tests verify that:
//   (a) without output_autopad, unaligned >16 channels are still rejected (no
//       regression).
//   (b) NPU5 with output_autopad and channels >16 remains rejected (NPU5 only
//       relaxed to [1..15]).
//   (c) Channels that were already valid (1–15, multiples of 16) remain valid.
//
// These tests must NOT fail after the implementation fix.
// ============================================================================

// ---------------------------------------------------------------------------
// 2.1  NPU6, NO autopad: unaligned >16 channels must still be rejected.
// ---------------------------------------------------------------------------
TEST_F(DPU_NPU6_OutputAutopadUnalignedGt16_SanitizerTest,
       Control_NPU6_NoAutopad_UnalignedCh_GT16_MustStillFail) {
    for (const auto out_ch : unaligned_gt16_channels) {
        DPUWorkload wl{
                npu6,
                Operation::CONVOLUTION,
                {VPUTensor(28, 28, 32, 1, DataType::UINT8)},
                {VPUTensor(28, 28, out_ch, 1, DataType::UINT8)},
                {1, 1},
                {1, 1},
                {0, 0, 0, 0},
                ExecutionMode::CUBOID_16x16,
                ActivationFunction::NONE,
                0.0F,
                0.0F,
                {swz_def, swz_def},
                {swz_def},
                1,
                {0, 0, 0, 0},
                ISIStrategy::CLUSTERING,
                false,
        };
        // output_autopad is NOT set (default = false / not set)

        dut.check_and_sanitize(wl, sane);
        EXPECT_GE(sane.value(), V(Cycles::START_ERROR_RANGE))
                << "Control regression: " << wl_desc("CONV", out_ch, false)
                << " should remain invalid without output_autopad";
    }
}

// ---------------------------------------------------------------------------
// 2.2  NPU5, output_autopad = true, channels > 16: must still be rejected.
//
// NPU5 ODU autopad only allows unaligned channels in [1, 15].  Values > 16
// that are unaligned must continue to fail on NPU5 even after the NPU6 fix.
// ---------------------------------------------------------------------------
TEST_F(DPU_NPU6_OutputAutopadUnalignedGt16_SanitizerTest,
       Control_NPU5_OutputAutopad_UnalignedCh_GT16_MustStillFail) {

    const std::vector<unsigned int> channels_to_test{17u, 20u, 24u, 30u, 31u, 33u, 40u};

    for (const auto out_ch : channels_to_test) {
        DPUWorkload wl{
                npu5,
                Operation::CONVOLUTION,
                {VPUTensor(28, 28, 32, 1, DataType::UINT8)},
                {VPUTensor(28, 28, out_ch, 1, DataType::UINT8)},
                {1, 1},
                {1, 1},
                {0, 0, 0, 0},
                ExecutionMode::CUBOID_16x16,
                ActivationFunction::NONE,
                0.0F,
                0.0F,
                {swz_def, swz_def},
                {swz_def},
                1,
                {0, 0, 0, 0},
                ISIStrategy::CLUSTERING,
                false,
        };
        wl.output_autopad = true;

        dut.check_and_sanitize(wl, sane);
        EXPECT_GE(sane.value(), V(Cycles::START_ERROR_RANGE))
                << "Control regression (NPU5): " << wl_desc("CONV", out_ch, true)
                << " must still be rejected — NPU5 autopad does not cover >16 unaligned channels";
    }
}

// ---------------------------------------------------------------------------
// 2.3  NPU6, output_autopad = true, aligned channels: must still pass.
//
// Verifies that the fix does not break existing valid aligned-channel paths.
// Aligned channels that currently pass must continue to pass.
// ---------------------------------------------------------------------------
TEST_F(DPU_NPU6_OutputAutopadUnalignedGt16_SanitizerTest,
       Control_NPU6_OutputAutopad_AlignedCh_MustContinueToPass) {
    // These are either in [1..15] or multiples of 16 — already valid today.
    const std::vector<unsigned int> already_valid_channels{1u, 8u, 15u, 16u, 32u, 48u, 64u};

    for (const auto out_ch : already_valid_channels) {
        DPUWorkload wl{
                npu6,
                Operation::CONVOLUTION,
                {VPUTensor(28, 28, 32, 1, DataType::UINT8)},
                {VPUTensor(28, 28, out_ch, 1, DataType::UINT8)},
                {1, 1},
                {1, 1},
                {0, 0, 0, 0},
                ExecutionMode::CUBOID_16x16,
                ActivationFunction::NONE,
                0.0F,
                0.0F,
                {swz_def, swz_def},
                {swz_def},
                1,
                {0, 0, 0, 0},
                ISIStrategy::CLUSTERING,
                false,
        };
        wl.output_autopad = true;

        dut.check_and_sanitize(wl, sane);
        EXPECT_EQ(sane.value(), V(Cycles::NO_ERROR))
                << "Regression: already-valid ch=" << out_ch
                << " with output_autopad=true must still pass after fix";
    }
}

// ---------------------------------------------------------------------------
// 2.4  NPU6, output_autopad = true, unaligned channels < 16: must still pass.
//
// Channels in [1..15] were already allowed by autopad (NPU5 baseline).
// Verify they continue to pass after the NPU6 broadening.
// ---------------------------------------------------------------------------
TEST_F(DPU_NPU6_OutputAutopadUnalignedGt16_SanitizerTest,
       Control_NPU6_OutputAutopad_UnalignedCh_LT16_MustContinueToPass) {
    const std::vector<unsigned int> lt16_channels{1u, 2u, 5u, 8u, 10u, 14u, 15u};

    for (const auto out_ch : lt16_channels) {
        DPUWorkload wl{
                npu6,
                Operation::ELTWISE,
                {VPUTensor(28, 9, 16, 1, DataType::FLOAT16)},
                {VPUTensor(28, 9, out_ch, 1, DataType::UINT8)},
                {1, 1},
                {1, 1},
                {0, 0, 0, 0},
                ExecutionMode::CUBOID_8x16,
                ActivationFunction::NONE,
                0.0F,
                0.0F,
                {swz_def, swz_def},
                {swz_def},
                1,
                {0, 0, 0, 0},
                ISIStrategy::CLUSTERING,
                false,
        };
        wl.output_autopad = true;
        wl.superdense_memory = true;

        dut.check_and_sanitize(wl, sane);
        EXPECT_EQ(sane.value(), V(Cycles::NO_ERROR))
                << "Regression: already-valid ch=" << out_ch
                << " (< 16, with output_autopad) must still pass after fix";
    }
}

// ============================================================================
// SECTION 3 – Compiler-workload-derived tests
//
// Based on the following real workload observed in NPU6 compiler runs:
//
//   device: NPU_6_0
//   op: AVEPOOL
//   input:  {1,1,1008,1} FLOAT16 ZXY
//   output: {1,1,1000,1} FLOAT32 ZXY
//   kernels/strides: 1x1 / 1x1, no padding
//   exec: CUBOID_16x16
//   superdense_memory=true, output_autopad=true
//
// The output channel count (1000) is NOT a multiple of 16  (1000 % 16 == 8).
// This is a large-valued unaligned case that also combines:
//   - FLOAT16 input → FLOAT32 output (cross-dtype AVEPOOL)
//   - superdense_memory = true
//   - ZXY layout
//
// NOTE: For AVEPOOL at sanitizer / DPU-tile level the variant limit is 256
// output channels.  Values > 256 would be rejected by the variant size check
// rather than the autopad alignment check and must not be used here.  The
// fix covered by 149253 therefore targets the range (16, 256].
// Accordingly the "exact workload" tests use out_ch=248 (same rem-8 residue
// as the compiler-observed 1000, but within the variant limit).
//
// Channel coverage for ch > 16, ch ≤ 256 (full list for sections 3.1–3.3)
// -----------------------------------------------------------------------
//  Remainder 1 (mod 16):  17, 33, 65, 97, 129
//  Remainder 2 (mod 16):  18, 34, 50
//  Remainder 4 (mod 16):  20, 36, 52, 68
//  Remainder 8 (mod 16):  24, 40, 56, 72, 88, 104, 120, 200, 248
//  Remainder 12 (mod 16): 28, 44, 60
//  Remainder 15 (mod 16): 31, 47, 63, 79
//  Near boundaries:       17, 31, 33, 63, 65, 127, 129, 255
//  Application-scale:     100, 200
// ============================================================================

// ---------------------------------------------------------------------------
// 3.1  Compiler-workload-derived sanitizer test — AVEPOOL 256→248, FP16→FP32,
//       superdense, output_autopad=true.
//
// 248 % 16 == 8  ⟹  same remainder class as the compiler-observed 1000,
// but within the 256-channel variant limit for AVEPOOL.  Currently FAILS.
// ---------------------------------------------------------------------------
TEST_F(DPU_NPU6_OutputAutopadUnalignedGt16_SanitizerTest,
       CompilerWL_AVEPOOL_256_to_248_FP16toFP32_ShouldPassValidation) {
    // Representative of the compiler-observed workload (rem-8 remainder class),
    // capped at the AVEPOOL variant limit of 256 output channels.
    DPUWorkload wl{
            npu6,
            Operation::AVEPOOL,
            {VPUTensor(1, 1, 256, 1, DataType::FLOAT16, Layout::ZXY)},  // input  (aligned)
            {VPUTensor(1, 1, 248, 1, DataType::FLOAT32, Layout::ZXY)},  // output: 248 % 16 == 8
            {1, 1},                                                      // kernels
            {1, 1},                                                      // strides
            {0, 0, 0, 0},                                                // padding
            ExecutionMode::CUBOID_16x16,
            ActivationFunction::NONE,
            0.0F,
            0.0F,
            {Swizzling::KEY_0, Swizzling::KEY_0},
            {Swizzling::KEY_0},
            1,
            {0, 0, 0, 0},
            ISIStrategy::CLUSTERING,
            false,
    };
    wl.superdense_memory = true;
    wl.output_autopad    = true;

    dut.check_and_sanitize(wl, sane);
    EXPECT_EQ(sane.value(), V(Cycles::NO_ERROR))
            << "FAILS: AVEPOOL 256 in -> 248 out (FP16->FP32, superdense, output_autopad=true)"
            << "\n  248 % 16 == 8 (unaligned > 16, within AVEPOOL variant limit 256)"
            << "\n  error: " << Cycles::toErrorText(sane.value()) << " | " << sane.info
            << "\n  Fix needed (149253)";
}

// ---------------------------------------------------------------------------
// 3.2  AVEPOOL FP16→FP32, superdense, output_autopad=true — channel sweep.
//
// Covers representative unaligned output channel values > 16 for various
// remainder classes (mod 16).  All currently FAIL.
//
// Cases listed per remainder class (all ≤ 256, per AVEPOOL variant limit):
//   rem 1:  17, 33, 65, 97, 129
//   rem 2:  18, 34, 50
//   rem 4:  20, 36, 68
//   rem 8:  24, 40, 56, 72, 88, 104, 120, 200, 248
//   rem 12: 28, 44, 60
//   rem 15: 31, 47, 63, 79
//   near-boundary: 17, 31, 33, 63, 65, 127, 129, 255
//   application-scale: 100, 200
// ---------------------------------------------------------------------------
TEST_F(DPU_NPU6_OutputAutopadUnalignedGt16_SanitizerTest,
       AVEPOOL_FP16toFP32_Superdense_OutputAutopad_Ch_Sweep_ShouldPassValidation) {
    // Channel values grouped by remainder — all > 16, none a multiple of 16,
    // all ≤ 256 (AVEPOOL variant limit).
    const std::vector<unsigned int> channels{
            // rem 1
            17u, 33u, 65u, 97u, 129u,
            // rem 2
            18u, 34u, 50u,
            // rem 4
            20u, 36u, 68u,
            // rem 8  (248 = same residue class as compiler-observed 1000)
            24u, 40u, 56u, 72u, 88u, 104u, 120u, 200u, 248u,
            // rem 12
            28u, 44u, 60u,
            // rem 15
            31u, 47u, 63u, 79u,
            // application-scale
            100u, 200u,
    };

    for (const auto out_ch : channels) {
        // Input channels: smallest multiple of 32 that is >= out_ch (keeps the
        // input aligned so any rejection is caused only by the output restriction).
        const unsigned int in_ch = dw_channel_align(out_ch);

        DPUWorkload wl{
                npu6,
                Operation::AVEPOOL,
                {VPUTensor(1, 1, in_ch, 1, DataType::FLOAT16, Layout::ZXY)},
                {VPUTensor(1, 1, out_ch, 1, DataType::FLOAT32, Layout::ZXY)},
                {1, 1},
                {1, 1},
                {0, 0, 0, 0},
                ExecutionMode::CUBOID_16x16,
                ActivationFunction::NONE,
                0.0F,
                0.0F,
                {Swizzling::KEY_0, Swizzling::KEY_0},
                {Swizzling::KEY_0},
                1,
                {0, 0, 0, 0},
                ISIStrategy::CLUSTERING,
                false,
        };
        wl.superdense_memory = true;
        wl.output_autopad    = true;

        dut.check_and_sanitize(wl, sane);
        EXPECT_EQ(sane.value(), V(Cycles::NO_ERROR))
                << "FAILS: AVEPOOL FP16->FP32 out_ch=" << out_ch
                << " (rem32=" << (out_ch % 16) << ")"
                << "\n  error: " << Cycles::toErrorText(sane.value()) << " | " << sane.info
                << "\n  Fix needed (149253)";
    }
}

// ---------------------------------------------------------------------------
// 3.3  AVEPOOL FP16→FP16 variant — same channel sweep, same flags.
//
// Verifies the fix is not restricted to FP32 output dtype.
// ---------------------------------------------------------------------------
TEST_F(DPU_NPU6_OutputAutopadUnalignedGt16_SanitizerTest,
       AVEPOOL_FP16toFP16_Superdense_OutputAutopad_Ch_GT16_ShouldPassValidation) {
    const std::vector<unsigned int> channels{17u, 24u, 31u, 33u, 40u, 65u, 100u, 200u, 248u};

    for (const auto out_ch : channels) {
        const unsigned int in_ch = dw_channel_align(out_ch);

        DPUWorkload wl{
                npu6,
                Operation::AVEPOOL,
                {VPUTensor(1, 1, in_ch, 1, DataType::FLOAT16, Layout::ZXY)},
                {VPUTensor(1, 1, out_ch, 1, DataType::FLOAT16, Layout::ZXY)},
                {1, 1},
                {1, 1},
                {0, 0, 0, 0},
                ExecutionMode::CUBOID_16x16,
                ActivationFunction::NONE,
                0.0F,
                0.0F,
                {Swizzling::KEY_0, Swizzling::KEY_0},
                {Swizzling::KEY_0},
                1,
                {0, 0, 0, 0},
                ISIStrategy::CLUSTERING,
                false,
        };
        wl.superdense_memory = true;
        wl.output_autopad    = true;

        dut.check_and_sanitize(wl, sane);
        EXPECT_EQ(sane.value(), V(Cycles::NO_ERROR))
                << "FAILS: AVEPOOL FP16->FP16 out_ch=" << out_ch
                << "\n  error: " << Cycles::toErrorText(sane.value()) << " | " << sane.info
                << "\n  Fix needed (149253)";
    }
}

// ---------------------------------------------------------------------------
// 3.4  Control — workload with aligned output channels (256→256, within limit).
//       Must continue to pass regardless of the fix direction.
// ---------------------------------------------------------------------------
TEST_F(DPU_NPU6_OutputAutopadUnalignedGt16_SanitizerTest,
       Control_CompilerWL_AVEPOOL_AlignedCh_ShouldContinueToPass) {
    // 256 % 16 == 0  ⟹  already valid today, must remain valid after fix.
    // (Using 256 rather than 1008 to stay within the AVEPOOL variant limit.)
    DPUWorkload wl{
            npu6,
            Operation::AVEPOOL,
            {VPUTensor(1, 1, 256, 1, DataType::FLOAT16, Layout::ZXY)},
            {VPUTensor(1, 1, 256, 1, DataType::FLOAT32, Layout::ZXY)},
            {1, 1},
            {1, 1},
            {0, 0, 0, 0},
            ExecutionMode::CUBOID_16x16,
            ActivationFunction::NONE,
            0.0F,
            0.0F,
            {Swizzling::KEY_0, Swizzling::KEY_0},
            {Swizzling::KEY_0},
            1,
            {0, 0, 0, 0},
            ISIStrategy::CLUSTERING,
            false,
    };
    wl.superdense_memory = true;
    wl.output_autopad    = true;

    dut.check_and_sanitize(wl, sane);
    EXPECT_EQ(sane.value(), V(Cycles::NO_ERROR))
            << "Regression: AVEPOOL 256->256 (aligned, within variant limit) should continue to pass";
}

// ---------------------------------------------------------------------------
// 3.5  Control — workload without output_autopad: must still be rejected
//       because 248 is unaligned and autopad is not set.
// ---------------------------------------------------------------------------
TEST_F(DPU_NPU6_OutputAutopadUnalignedGt16_SanitizerTest,
       Control_CompilerWL_AVEPOOL_248ch_NoAutopad_MustStillFail) {
    DPUWorkload wl{
            npu6,
            Operation::AVEPOOL,
            {VPUTensor(1, 1, 256, 1, DataType::FLOAT16, Layout::ZXY)},
            {VPUTensor(1, 1, 248, 1, DataType::FLOAT32, Layout::ZXY)},
            {1, 1},
            {1, 1},
            {0, 0, 0, 0},
            ExecutionMode::CUBOID_16x16,
            ActivationFunction::NONE,
            0.0F,
            0.0F,
            {Swizzling::KEY_0, Swizzling::KEY_0},
            {Swizzling::KEY_0},
            1,
            {0, 0, 0, 0},
            ISIStrategy::CLUSTERING,
            false,
    };
    wl.superdense_memory = true;
    // output_autopad deliberately NOT set

    dut.check_and_sanitize(wl, sane);
    EXPECT_GE(sane.value(), V(Cycles::START_ERROR_RANGE))
            << "Control regression: AVEPOOL 248 ch (rem-8, ≤256) without output_autopad must still be rejected";
}

}  // namespace VPUNN_unit_tests
