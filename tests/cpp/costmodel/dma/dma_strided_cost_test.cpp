// Copyright © 2025 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the "Software Package")
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the "third-party-programs.txt" or other similarly-named text file included with the
// Software Package for additional details.

#include "vpu/dma_cost_providers/dma_stride_lut_fallback.h"
#include "vpu/dma_cost_providers/dma_strided_math_cost_provider.h"
#include "vpu/dma_descriptor_transformers.h"
#include "vpu/dma_types.h"
#include "vpu/hw_characteristics/HW_characteristics_supersets.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <vector>

/// @brief namespace for Unit tests of the C++ library
namespace VPUNN_unit_tests {
using namespace VPUNN;

class TestDMAStridedCost : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize hardware characteristics
    }

    VPUDevice device{VPUDevice::NPU_6_0};  ///< default device for tests

    /// Provider under the legacy NN workload interface.
    DMAStridedMathCostProvider_NNIntf provider_nn;
    /// Provider under the new descriptor interface.
    DMAStridedMathCostProvider_DescIntf provider_desc;

    /// Convert a DMANNWorkload_NPU40_50 to a VPUDMADescriptor (NN → Desc direction).
    static VPUDMADescriptor toDesc(const DMANNWorkload_NPU40_50& wl) {
        return DMADescriptorTransformer::fromDMANNWorkload_NPU40_50_to_VPUDMADescriptor(wl);
    }

    /// Convert a VPUDMADescriptor to a DMANNWorkload_NPU40_50 (Desc → NN direction).
    static DMANNWorkload_NPU40_50 toNN(const VPUDMADescriptor& desc) {
        return DMADescriptorTransformer::fromStridedTensors_toDMANNWorkload_NPU40_50(desc);
    }

    /// Assert that both providers return identical cycles starting from a DMANNWorkload_NPU40_50.
    /// Converts NN → Desc to call the descriptor provider.
    void expectSameCost_fromNN(const DMANNWorkload_NPU40_50& wl, const std::string& label = "") const {
        const auto cost_nn = provider_nn.get_cost(wl);
        const auto cost_desc = provider_desc.get_cost(toDesc(wl));
        EXPECT_EQ(cost_nn, cost_desc) << "Round-trip (NN→Desc) mismatch" << (label.empty() ? "" : " (" + label + ")");
    }

    /// Assert that both providers return identical cycles starting from a VPUDMADescriptor.
    /// Converts Desc → NN to call the NN provider.
    void expectSameCost_fromDesc(const VPUDMADescriptor& desc, const std::string& label = "") const {
        const auto cost_nn = provider_nn.get_cost(toNN(desc));
        const auto cost_desc = provider_desc.get_cost(desc);
        EXPECT_EQ(cost_nn, cost_desc) << "Round-trip (Desc→NN) mismatch" << (label.empty() ? "" : " (" + label + ")");
    }
};

/// Test computeSrcCycles and computeDstCycles together for various stride scenarios
TEST_F(TestDMAStridedCost, ComputeSrcAndDstCycles_Combined_Test) {
    DMAStridedMathModel dma_model(HWCharacteristicsSuperSets::get_mainConfigurationRef());

    // Test 1: Fully contiguous transfer (no stride) - src and dst should have same penalty
    {
        DMANNWorkload_NPU40_50 wl{device};
        wl.src_width = 256;
        wl.dst_width = 256;
        wl.num_dim = 1;
        wl.e_dim[0].src_dim_size = 15;  // 16 elements
        wl.e_dim[0].dst_dim_size = 15;
        wl.e_dim[0].src_stride = 256;  // Contiguous
        wl.e_dim[0].dst_stride = 256;  // Contiguous

        auto src_cycles = dma_model.computeSrcCycles(wl);
        auto dst_cycles = dma_model.computeDstCycles(wl);

        EXPECT_GT(src_cycles, 0) << "Source cycles should be positive";
        EXPECT_GT(dst_cycles, 0) << "Destination cycles should be positive";
        EXPECT_EQ(src_cycles, dst_cycles) << "Fully contiguous: src and dst cycles should be equal";
    }

    // Test 2: Source strided, destination contiguous - src should be slower
    {
        DMANNWorkload_NPU40_50 wl{device};
        wl.src_width = 32;
        wl.dst_width = 32;
        wl.num_dim = 1;
        wl.e_dim[0].src_dim_size = 7;  // 8 elements
        wl.e_dim[0].dst_dim_size = 7;
        wl.e_dim[0].src_stride = 64;  // Gap: stride > width (strided)
        wl.e_dim[0].dst_stride = 32;  // Contiguous

        auto src_cycles = dma_model.computeSrcCycles(wl);
        auto dst_cycles = dma_model.computeDstCycles(wl);

        EXPECT_GT(src_cycles, 0) << "Source cycles should be positive";
        EXPECT_GT(dst_cycles, 0) << "Destination cycles should be positive";
        EXPECT_GT(src_cycles, dst_cycles) << "Strided source should be slower than contiguous destination";
    }

    // Test 3: Destination strided, source contiguous - dst should be slower
    {
        DMANNWorkload_NPU40_50 wl{device};
        wl.src_width = 32;
        wl.dst_width = 32;
        wl.num_dim = 1;
        wl.e_dim[0].src_dim_size = 7;  // 8 elements
        wl.e_dim[0].dst_dim_size = 7;
        wl.e_dim[0].src_stride = 32;  // Contiguous
        wl.e_dim[0].dst_stride = 64;  // Gap: stride > width (strided)

        auto src_cycles = dma_model.computeSrcCycles(wl);
        auto dst_cycles = dma_model.computeDstCycles(wl);

        EXPECT_GT(src_cycles, 0) << "Source cycles should be positive";
        EXPECT_GT(dst_cycles, 0) << "Destination cycles should be positive";
        EXPECT_LT(src_cycles, dst_cycles) << "Contiguous source should be faster than strided destination";
    }

    // Test 4: Both strided with different chunk sizes
    {
        DMANNWorkload_NPU40_50 wl{device};
        wl.src_width = 32;
        wl.dst_width = 64;  // Different widths
        wl.num_dim = 1;
        wl.e_dim[0].src_dim_size = 7;  // 8 elements
        wl.e_dim[0].dst_dim_size = 7;
        wl.e_dim[0].src_stride = 64;  // Strided (32 bytes, gap of 32)
        wl.e_dim[0].dst_stride = 96;  // Strided (64 bytes, gap of 32)

        auto src_cycles = dma_model.computeSrcCycles(wl);
        auto dst_cycles = dma_model.computeDstCycles(wl);

        EXPECT_GT(src_cycles, 0) << "Source cycles should be positive";
        EXPECT_GT(dst_cycles, 0) << "Destination cycles should be positive";

        // Source has smaller chunks (32 bytes) vs destination (64 bytes)
        // Smaller chunks = higher penalty
        EXPECT_GT(src_cycles, dst_cycles) << "Smaller src chunks should result in higher penalty";
    }

    // Test 5: Small contiguous chunks below at least one side-specific threshold (CMX=64, DDR=256)
    {
        DMANNWorkload_NPU40_50 wl{device};
        wl.src_width = 128;  // Below DDR threshold, above CMX threshold
        wl.dst_width = 128;
        wl.num_dim = 1;
        wl.e_dim[0].src_dim_size = 1;  // 2 elements total
        wl.e_dim[0].dst_dim_size = 1;
        wl.e_dim[0].src_stride = 256;  // Gap: forces small chunks
        wl.e_dim[0].dst_stride = 256;  // Gap: forces small chunks

        auto src_cycles = dma_model.computeSrcCycles(wl);
        auto dst_cycles = dma_model.computeDstCycles(wl);

        EXPECT_GT(src_cycles, 0) << "Source cycles should be positive";
        EXPECT_GT(dst_cycles, 0) << "Destination cycles should be positive";

        // Both have same chunk size (128 bytes), so should have same penalty
        EXPECT_EQ(src_cycles, dst_cycles) << "Same chunk sizes should result in equal cycles";
    }

    // Test 6: Very small chunks (high penalty)
    {
        DMANNWorkload_NPU40_50 wl{device};
        wl.src_width = 32;  // Very small chunks
        wl.dst_width = 32;
        wl.num_dim = 1;
        wl.e_dim[0].src_dim_size = 99;  // 100 elements
        wl.e_dim[0].dst_dim_size = 99;
        wl.e_dim[0].src_stride = 64;  // Strided
        wl.e_dim[0].dst_stride = 64;  // Strided

        auto src_cycles = dma_model.computeSrcCycles(wl);
        auto dst_cycles = dma_model.computeDstCycles(wl);

        EXPECT_GT(src_cycles, 0) << "Source cycles should be positive";
        EXPECT_GT(dst_cycles, 0) << "Destination cycles should be positive";
        EXPECT_EQ(src_cycles, dst_cycles) << "Same small chunks should result in equal high penalties";

        // With 32-byte chunks, penalty is expected to be high on both sides.
        // Exact factor depends on the side-specific threshold selected by direction.
    }

    // Test 7: Asymmetric dimension sizes (different src/dst element counts)
    {
        DMANNWorkload_NPU40_50 wl{device};
        wl.src_width = 64;
        wl.dst_width = 64;
        wl.num_dim = 2;
        wl.e_dim[0].src_dim_size = 7;   // 8 elements src
        wl.e_dim[0].dst_dim_size = 15;  // 16 elements dst
        wl.e_dim[0].src_stride = 64;    // Contiguous for src
        wl.e_dim[0].dst_stride = 64;    // Contiguous for dst
        wl.e_dim[1].src_dim_size = 3;   // 4 planes
        wl.e_dim[1].dst_dim_size = 3;
        wl.e_dim[1].src_stride = 512;   // Contiguous: 64 * 8
        wl.e_dim[1].dst_stride = 1024;  // Contiguous: 64 * 16

        auto src_cycles = dma_model.computeSrcCycles(wl);
        auto dst_cycles = dma_model.computeDstCycles(wl);

        EXPECT_GT(src_cycles, 0) << "Source cycles should be positive";
        EXPECT_GT(dst_cycles, 0) << "Destination cycles should be positive";

        // Source reads 2048 bytes (64*8*4), destination writes 4096 bytes (64*16*4)
        // Destination should be slower due to more data
        EXPECT_LT(src_cycles, dst_cycles) << "Destination with more data should be slower";
    }

    // Test 8: 1D simple transfer (no extra dimensions)
    {
        DMANNWorkload_NPU40_50 wl{device, 1024, 1024, 0};  // 1D transfer

        auto src_cycles = dma_model.computeSrcCycles(wl);
        auto dst_cycles = dma_model.computeDstCycles(wl);

        EXPECT_GT(src_cycles, 0) << "Source cycles should be positive";
        EXPECT_GT(dst_cycles, 0) << "Destination cycles should be positive";
        EXPECT_EQ(src_cycles, dst_cycles) << "1D transfer: src and dst should be equal";
    }

    // Test 9: Large contiguous transfer (above threshold, minimal penalty)
    {
        DMANNWorkload_NPU40_50 wl{device};
        wl.src_width = 4096;  // Large contiguous block
        wl.dst_width = 4096;
        wl.num_dim = 1;
        wl.e_dim[0].src_dim_size = 3;  // 4 blocks
        wl.e_dim[0].dst_dim_size = 3;
        wl.e_dim[0].src_stride = 4096;  // Contiguous
        wl.e_dim[0].dst_stride = 4096;  // Contiguous

        auto src_cycles = dma_model.computeSrcCycles(wl);
        auto dst_cycles = dma_model.computeDstCycles(wl);

        EXPECT_GT(src_cycles, 0) << "Source cycles should be positive";
        EXPECT_GT(dst_cycles, 0) << "Destination cycles should be positive";
        EXPECT_EQ(src_cycles, dst_cycles) << "Large contiguous: minimal penalty, cycles should be equal";
    }

    // Test 10: Maximum penalty scenario (1-byte chunks)
    {
        DMANNWorkload_NPU40_50 wl{device};
        wl.src_width = 1;  // 1-byte chunks (maximum fragmentation)
        wl.dst_width = 1;
        wl.num_dim = 1;
        wl.e_dim[0].src_dim_size = 255;  // 256 elements
        wl.e_dim[0].dst_dim_size = 255;
        wl.e_dim[0].src_stride = 4;  // Strided
        wl.e_dim[0].dst_stride = 4;  // Strided

        auto src_cycles = dma_model.computeSrcCycles(wl);
        auto dst_cycles = dma_model.computeDstCycles(wl);

        EXPECT_GT(src_cycles, 0) << "Source cycles should be positive";
        EXPECT_GT(dst_cycles, 0) << "Destination cycles should be positive";
        EXPECT_EQ(src_cycles, dst_cycles) << "Maximum penalty: equal for both";

        // With 1-byte chunks, penalty is expected to be maximal for both CMX and DDR paths.
        // Exact factor depends on the side-specific threshold selected by direction.
    }
}

/// Explicitly validates side-specific threshold routing across all transfer directions.
/// Uses src-vs-dst comparisons within the same direction so base linear parameters (k,b)
/// are identical and only side-threshold routing (CMX vs DDR) can explain deltas.
TEST_F(TestDMAStridedCost, DirectionBasedThresholdRouting_SrcDstPaths) {
    DMAStridedMathModel dma_model(HWCharacteristicsSuperSets::get_mainConfigurationRef());

    const auto make_strided_wl = [this](MemoryDirection dir) {
        DMANNWorkload_NPU40_50 wl{device};
        wl.transfer_direction = dir;

        wl.src_width = 128;
        wl.dst_width = 128;
        wl.num_dim = 1;

        wl.e_dim[0].src_dim_size = 1;  // total src bytes: 128 * 2 = 256
        wl.e_dim[0].dst_dim_size = 1;  // total dst bytes: 128 * 2 = 256

        wl.e_dim[0].src_stride = 256;  // contiguous chunk src: 128 (strided)
        wl.e_dim[0].dst_stride = 256;  // contiguous chunk dst: 128 (strided)
        return wl;
    };

    const auto wl_cmx2ddr = make_strided_wl(MemoryDirection::CMX2DDR);
    const auto wl_ddr2cmx = make_strided_wl(MemoryDirection::DDR2CMX);
    const auto wl_cmx2cmx = make_strided_wl(MemoryDirection::CMX2CMX);
    const auto wl_ddr2ddr = make_strided_wl(MemoryDirection::DDR2DDR);

    const auto src_cmx2ddr = dma_model.computeSrcCycles(wl_cmx2ddr);  // src on CMX
    const auto src_ddr2cmx = dma_model.computeSrcCycles(wl_ddr2cmx);  // src on DDR
    const auto src_cmx2cmx = dma_model.computeSrcCycles(wl_cmx2cmx);  // src on CMX
    const auto src_ddr2ddr = dma_model.computeSrcCycles(wl_ddr2ddr);  // src on DDR

    const auto dst_cmx2ddr = dma_model.computeDstCycles(wl_cmx2ddr);  // dst on DDR
    const auto dst_ddr2cmx = dma_model.computeDstCycles(wl_ddr2cmx);  // dst on CMX
    const auto dst_cmx2cmx = dma_model.computeDstCycles(wl_cmx2cmx);  // dst on CMX
    const auto dst_ddr2ddr = dma_model.computeDstCycles(wl_ddr2ddr);  // dst on DDR

    EXPECT_GT(src_cmx2ddr, 0u);
    EXPECT_GT(src_ddr2cmx, 0u);
    EXPECT_GT(src_cmx2cmx, 0u);
    EXPECT_GT(src_ddr2ddr, 0u);
    EXPECT_GT(dst_cmx2ddr, 0u);
    EXPECT_GT(dst_ddr2cmx, 0u);
    EXPECT_GT(dst_cmx2cmx, 0u);
    EXPECT_GT(dst_ddr2ddr, 0u);

    // Within CMX2DDR: src is CMX-side, dst is DDR-side -> dst should be slower.
    EXPECT_GT(dst_cmx2ddr, src_cmx2ddr);

    // Within DDR2CMX: src is DDR-side, dst is CMX-side -> src should be slower.
    EXPECT_GT(src_ddr2cmx, dst_ddr2cmx);

    // Within CMX2CMX: both sides use CMX threshold -> equal.
    EXPECT_EQ(src_cmx2cmx, dst_cmx2cmx);

    // Within DDR2DDR: both sides use DDR threshold -> equal.
    EXPECT_EQ(src_ddr2ddr, dst_ddr2ddr);
}

/// Ensures both interfaces remain aligned when direction is explicitly controlled.
TEST_F(TestDMAStridedCost, DirectionBasedThresholdRouting_InterfaceParity) {
    const auto make_strided_wl = [this](MemoryDirection dir) {
        DMANNWorkload_NPU40_50 wl{device};
        wl.transfer_direction = dir;
        wl.src_width = 128;
        wl.dst_width = 128;
        wl.num_dim = 1;
        wl.e_dim[0].src_dim_size = 1;
        wl.e_dim[0].dst_dim_size = 1;
        wl.e_dim[0].src_stride = 256;
        wl.e_dim[0].dst_stride = 256;
        return wl;
    };

    for (const auto dir :
         {MemoryDirection::CMX2CMX, MemoryDirection::CMX2DDR, MemoryDirection::DDR2CMX, MemoryDirection::DDR2DDR}) {
        const auto wl = make_strided_wl(dir);
        expectSameCost_fromNN(wl, "direction-aware NN->Desc parity");

        VPUDMADescriptor desc;
        desc.device = device;
        desc.src.dtype = DataType::UINT8;
        desc.dst.dtype = DataType::UINT8;
        desc.src.setDimension_InnermostFirst({{128, 1}, {2, 256}});
        desc.dst.setDimension_InnermostFirst({{128, 1}, {2, 256}});

        switch (dir) {
        case MemoryDirection::CMX2CMX:
            desc.src_location = MemoryLocation::CMX;
            desc.dst_location = MemoryLocation::CMX;
            break;
        case MemoryDirection::CMX2DDR:
            desc.src_location = MemoryLocation::CMX;
            desc.dst_location = MemoryLocation::DRAM;
            break;
        case MemoryDirection::DDR2CMX:
            desc.src_location = MemoryLocation::DRAM;
            desc.dst_location = MemoryLocation::CMX;
            break;
        case MemoryDirection::DDR2DDR:
            desc.src_location = MemoryLocation::DRAM;
            desc.dst_location = MemoryLocation::DRAM;
            break;
        default:
            FAIL() << "Unsupported direction in parity test";
        }

        expectSameCost_fromDesc(desc, "direction-aware Desc->NN parity");
    }
}

// ===========================================================================
// DescIntf tests — DMAStridedMathCostProvider_DescIntf (VPUDMADescriptor input)
//
// Each sub-case constructs VPUDMADescriptor natively (dtype + shape + byte_strides),
// without ever touching DMANNWorkload_NPU40_50.  This exercises the descriptor
// interface on its own terms; round-trip equivalence is proven separately below.
//
// Descriptor tensors are built with setDimension_InnermostFirst(): the first pair
// is the innermost dimension (width block), subsequent pairs are outer dimensions.
// dtype is UINT8 throughout so that 1 byte == 1 element (byte-granularity transfer).
// ===========================================================================
TEST_F(TestDMAStridedCost, DescIntf_ComputeCycles_Combined_Test) {
    // Helpers to reduce boilerplate.
    // make1D: single contiguous flat buffer (count elements, each stride bytes apart).
    // make2D: inner block repeated outer times with outer_stride between starts.
    // make3D: adds a third (outermost) dimension.
    const auto make1D = [](VPUDevice dev, int32_t count, int32_t stride) {
        VPUDMADescriptor d;
        d.device = dev;
        d.src.dtype = DataType::UINT8;
        d.src.setDimension_InnermostFirst({{count, stride}});
        d.dst = d.src;
        return d;
    };
    const auto make2D = [](VPUDevice dev, int32_t src_inner, int32_t src_inner_stride, int32_t src_outer,
                           int32_t src_outer_stride, int32_t dst_inner, int32_t dst_inner_stride, int32_t dst_outer,
                           int32_t dst_outer_stride) {
        VPUDMADescriptor d;
        d.device = dev;
        d.src.dtype = DataType::UINT8;
        d.src.setDimension_InnermostFirst({{src_inner, src_inner_stride}, {src_outer, src_outer_stride}});
        d.dst.dtype = DataType::UINT8;
        d.dst.setDimension_InnermostFirst({{dst_inner, dst_inner_stride}, {dst_outer, dst_outer_stride}});
        return d;
    };
    const auto make2D_sym = [&make2D](VPUDevice dev, int32_t inner, int32_t inner_stride, int32_t outer,
                                      int32_t outer_stride) {
        return make2D(dev, inner, inner_stride, outer, outer_stride, inner, inner_stride, outer, outer_stride);
    };

    for (const auto dev : {
                 VPUDevice::NPU_6_0,
                 VPUDevice::NPU_5_0,
                 VPUDevice::NPU_5_0_W,
         }) {
        const std::string dev_tag = " [device=" + VPUDevice_ToText.at(static_cast<int>(dev)) + "]";

        // Test 1: Fully contiguous — 256-byte inner block, 16 repetitions, packed (stride==width).
        // src and dst identical → positive, non-error cycles.
        {
            const auto desc = make2D_sym(dev, 256, 1, 16, 256);
            const auto cost = provider_desc.get_cost(desc);
            EXPECT_GT(cost, 0u) << "Test 1: contiguous transfer should yield positive cycles" << dev_tag;
            EXPECT_FALSE(Cycles::isErrorCode(cost)) << "Test 1: must not be an error code" << dev_tag;
        }

        // Test 2: Source strided, destination contiguous.
        // src: 32-byte block × 8, stride=64 (gap 32); dst: same block × 8, stride=32 (no gap).
        {
            const auto desc_strided = make2D(dev, 32, 1, 8, 64, 32, 1, 8, 32);
            const auto desc_baseline = make2D(dev, 32, 1, 8, 32, 32, 1, 8, 32);

            const auto cost_strided = provider_desc.get_cost(desc_strided);
            const auto cost_baseline = provider_desc.get_cost(desc_baseline);

            EXPECT_GT(cost_strided, 0u) << "Test 2: strided src must yield positive cycles" << dev_tag;
            EXPECT_GT(cost_strided, cost_baseline) << "Test 2: strided src > fully contiguous" << dev_tag;
        }

        // Test 3: Destination strided, source contiguous.
        // src: 32-byte block × 8, stride=32; dst: same block × 8, stride=64 (gap 32).
        {
            const auto desc_strided = make2D(dev, 32, 1, 8, 32, 32, 1, 8, 64);
            const auto desc_baseline = make2D(dev, 32, 1, 8, 32, 32, 1, 8, 32);

            const auto cost_strided = provider_desc.get_cost(desc_strided);
            const auto cost_baseline = provider_desc.get_cost(desc_baseline);

            EXPECT_GT(cost_strided, 0u) << "Test 3: strided dst must yield positive cycles" << dev_tag;
            EXPECT_GT(cost_strided, cost_baseline) << "Test 3: strided dst > fully contiguous" << dev_tag;
        }

        // Test 4: Both strided, different chunk sizes — smaller chunk incurs higher penalty.
        // small: 32-byte chunks, stride=64;  large: 64-byte chunks, stride=96.
        {
            const auto desc_small = make2D_sym(dev, 32, 1, 8, 64);
            const auto desc_large = make2D_sym(dev, 64, 1, 8, 96);

            const auto cost_small = provider_desc.get_cost(desc_small);
            const auto cost_large = provider_desc.get_cost(desc_large);

            EXPECT_GT(cost_small, 0u) << "Test 4: small chunks must yield positive cycles" << dev_tag;
            EXPECT_GT(cost_large, 0u) << "Test 4: large chunks must yield positive cycles" << dev_tag;
            EXPECT_GE(cost_small, cost_large) << "Test 4: smaller chunks incur >= penalty vs larger chunks" << dev_tag;
        }

        // Test 5: Symmetric strided transfer with 128-byte chunks; default locations are CMX→CMX so both sides use the
        // CMX threshold. 128-byte inner block, 2 repetitions, stride=256 (gap 128).
        {
            const auto desc = make2D_sym(dev, 128, 1, 2, 256);
            const auto cost = provider_desc.get_cost(desc);
            EXPECT_GT(cost, 0u) << "Test 5: below-threshold chunks must yield positive cycles" << dev_tag;
            EXPECT_FALSE(Cycles::isErrorCode(cost)) << "Test 5: must not be an error code" << dev_tag;
        }

        // Test 6: 1-byte chunks — maximum fragmentation.
        // 1-byte inner block, 256 repetitions, stride=4.
        {
            const auto desc = make2D_sym(dev, 1, 1, 256, 4);
            const auto cost = provider_desc.get_cost(desc);
            EXPECT_GT(cost, 0u) << "Test 6: 1-byte chunks must yield positive cycles" << dev_tag;
            EXPECT_FALSE(Cycles::isErrorCode(cost)) << "Test 6: must not be an error code" << dev_tag;
        }

        // Test 7: Asymmetric src/dst shapes — src reads 512 bytes, dst writes 1024 bytes.
        // src: 64-byte inner × 8 mid × 4 outer (contiguous planes).
        // dst: 64-byte inner × 16 mid × 4 outer (contiguous planes).
        {
            VPUDMADescriptor desc;
            desc.device = dev;
            desc.src.dtype = DataType::UINT8;
            desc.src.setDimension_InnermostFirst({{64, 1}, {8, 64}, {4, 512}});
            desc.dst.dtype = DataType::UINT8;
            desc.dst.setDimension_InnermostFirst({{64, 1}, {16, 64}, {4, 1024}});

            const auto cost = provider_desc.get_cost(desc);
            EXPECT_GT(cost, 0u) << "Test 7: asymmetric dims must yield positive cycles" << dev_tag;
            EXPECT_FALSE(Cycles::isErrorCode(cost)) << "Test 7: must not be an error code" << dev_tag;
        }

        // Test 8: 1D flat transfer — 1024 contiguous bytes.
        {
            const auto desc = make1D(dev, 1024, 1);
            const auto cost = provider_desc.get_cost(desc);
            EXPECT_GT(cost, 0u) << "Test 8: 1D transfer must yield positive cycles" << dev_tag;
            EXPECT_FALSE(Cycles::isErrorCode(cost)) << "Test 8: must not be an error code" << dev_tag;
        }

        // Test 9: Large contiguous transfer — 4096-byte block, 4 repetitions, no gap.
        {
            const auto desc = make2D_sym(dev, 4096, 1, 4, 4096);
            const auto cost = provider_desc.get_cost(desc);
            EXPECT_GT(cost, 0u) << "Test 9: large contiguous transfer must yield positive cycles" << dev_tag;
            EXPECT_FALSE(Cycles::isErrorCode(cost)) << "Test 9: must not be an error code" << dev_tag;
        }

        // Test 10: Maximum penalty — 1-byte blocks, 256 repetitions, stride=4.
        {
            const auto desc = make2D_sym(dev, 1, 1, 256, 4);
            const auto cost = provider_desc.get_cost(desc);
            EXPECT_GT(cost, 0u) << "Test 10: maximum-penalty transfer must yield positive cycles" << dev_tag;
            EXPECT_FALSE(Cycles::isErrorCode(cost)) << "Test 10: must not be an error code" << dev_tag;
        }
    }
}

// ===========================================================================
// Round-trip: both interfaces must return identical cycles.
//
// Two sub-tests:
//   NN→Desc: start from DMANNWorkload_NPU40_50, convert to VPUDMADescriptor.
//   Desc→NN: start from VPUDMADescriptor, convert to DMANNWorkload_NPU40_50.
// Both directions are tested for the same 10 logical transfers.
// ===========================================================================
TEST_F(TestDMAStridedCost, RoundTrip_NNIntf_vs_DescIntf_EqualCycles) {
    // -----------------------------------------------------------------------
    // Direction 1: NN → Desc  (DMANNWorkload_NPU40_50 is the source of truth)
    // -----------------------------------------------------------------------

    // RT-NN-1: Fully contiguous
    {
        DMANNWorkload_NPU40_50 wl{device};
        wl.src_width = 256;
        wl.dst_width = 256;
        wl.num_dim = 1;
        wl.e_dim[0].src_dim_size = 15;
        wl.e_dim[0].dst_dim_size = 15;
        wl.e_dim[0].src_stride = 256;
        wl.e_dim[0].dst_stride = 256;
        expectSameCost_fromNN(wl, "NN→Desc contiguous");
    }

    // RT-NN-2: Source strided
    {
        DMANNWorkload_NPU40_50 wl{device};
        wl.src_width = 64;
        wl.dst_width = 64;
        wl.num_dim = 1;
        wl.e_dim[0].src_dim_size = 7;
        wl.e_dim[0].dst_dim_size = 7;
        wl.e_dim[0].src_stride = 128;
        wl.e_dim[0].dst_stride = 64;
        expectSameCost_fromNN(wl, "NN→Desc src strided");
    }

    // RT-NN-3: Destination strided
    {
        DMANNWorkload_NPU40_50 wl{device};
        wl.src_width = 64;
        wl.dst_width = 64;
        wl.num_dim = 1;
        wl.e_dim[0].src_dim_size = 7;
        wl.e_dim[0].dst_dim_size = 7;
        wl.e_dim[0].src_stride = 64;
        wl.e_dim[0].dst_stride = 128;
        expectSameCost_fromNN(wl, "NN→Desc dst strided");
    }

    // RT-NN-4: Both strided, different chunk sizes
    {
        DMANNWorkload_NPU40_50 wl{device};
        wl.src_width = 32;
        wl.dst_width = 64;
        wl.num_dim = 1;
        wl.e_dim[0].src_dim_size = 7;
        wl.e_dim[0].dst_dim_size = 7;
        wl.e_dim[0].src_stride = 64;
        wl.e_dim[0].dst_stride = 96;
        expectSameCost_fromNN(wl, "NN→Desc both strided diff chunks");
    }

    // RT-NN-5: Small chunks below threshold
    {
        DMANNWorkload_NPU40_50 wl{device};
        wl.src_width = 128;
        wl.dst_width = 128;
        wl.num_dim = 1;
        wl.e_dim[0].src_dim_size = 1;
        wl.e_dim[0].dst_dim_size = 1;
        wl.e_dim[0].src_stride = 256;
        wl.e_dim[0].dst_stride = 256;
        expectSameCost_fromNN(wl, "NN→Desc small chunks");
    }

    // RT-NN-6: Very small chunks, high penalty
    {
        DMANNWorkload_NPU40_50 wl{device};
        wl.src_width = 32;
        wl.dst_width = 32;
        wl.num_dim = 1;
        wl.e_dim[0].src_dim_size = 99;
        wl.e_dim[0].dst_dim_size = 99;
        wl.e_dim[0].src_stride = 64;
        wl.e_dim[0].dst_stride = 64;
        expectSameCost_fromNN(wl, "NN→Desc very small chunks");
    }

    // RT-NN-7: Asymmetric dimension sizes
    {
        DMANNWorkload_NPU40_50 wl{device};
        wl.src_width = 64;
        wl.dst_width = 64;
        wl.num_dim = 2;
        wl.e_dim[0].src_dim_size = 7;
        wl.e_dim[0].dst_dim_size = 15;
        wl.e_dim[0].src_stride = 64;
        wl.e_dim[0].dst_stride = 64;
        wl.e_dim[1].src_dim_size = 3;
        wl.e_dim[1].dst_dim_size = 3;
        wl.e_dim[1].src_stride = 512;
        wl.e_dim[1].dst_stride = 1024;
        expectSameCost_fromNN(wl, "NN→Desc asymmetric dims");
    }

    // RT-NN-8: 1D transfer
    {
        DMANNWorkload_NPU40_50 wl{device, 1024, 1024, 0};
        expectSameCost_fromNN(wl, "NN→Desc 1D");
    }

    // RT-NN-9: Large contiguous transfer
    {
        DMANNWorkload_NPU40_50 wl{device};
        wl.src_width = 4096;
        wl.dst_width = 4096;
        wl.num_dim = 1;
        wl.e_dim[0].src_dim_size = 3;
        wl.e_dim[0].dst_dim_size = 3;
        wl.e_dim[0].src_stride = 4096;
        wl.e_dim[0].dst_stride = 4096;
        expectSameCost_fromNN(wl, "NN→Desc large contiguous");
    }

    // RT-NN-10: Maximum penalty (1-byte chunks)
    {
        DMANNWorkload_NPU40_50 wl{device};
        wl.src_width = 1;
        wl.dst_width = 1;
        wl.num_dim = 1;
        wl.e_dim[0].src_dim_size = 255;
        wl.e_dim[0].dst_dim_size = 255;
        wl.e_dim[0].src_stride = 4;
        wl.e_dim[0].dst_stride = 4;
        expectSameCost_fromNN(wl, "NN→Desc max penalty");
    }

    // RT-NN-11: 3D strided workload
    {
        DMANNWorkload_NPU40_50 wl{device};
        wl.src_width = 32;
        wl.dst_width = 32;
        wl.num_dim = 2;
        wl.e_dim[0].src_dim_size = 7;
        wl.e_dim[0].dst_dim_size = 7;
        wl.e_dim[0].src_stride = 128;
        wl.e_dim[0].dst_stride = 128;
        wl.e_dim[1].src_dim_size = 3;
        wl.e_dim[1].dst_dim_size = 3;
        wl.e_dim[1].src_stride = 512;
        wl.e_dim[1].dst_stride = 512;
        expectSameCost_fromNN(wl, "NN→Desc 3D strided");
    }

    // RT-NN — different devices
    std::vector<VPUDevice> rt_devices;
    rt_devices.push_back(VPUDevice::NPU_5_0);
    rt_devices.push_back(VPUDevice::NPU_6_0);
    if (rt_devices.empty()) {
        rt_devices.push_back(device);
    }

    for (const auto dev : rt_devices) {
        DMANNWorkload_NPU40_50 wl{dev};
        wl.src_width = 256;
        wl.dst_width = 256;
        wl.num_dim = 1;
        wl.e_dim[0].src_dim_size = 15;
        wl.e_dim[0].dst_dim_size = 15;
        wl.e_dim[0].src_stride = 512;
        wl.e_dim[0].dst_stride = 512;
        expectSameCost_fromNN(wl, "NN→Desc device=" + std::to_string(static_cast<int>(dev)));
    }

    // -----------------------------------------------------------------------
    // Direction 2: Desc → NN  (VPUDMADescriptor is the source of truth)
    // The same 10 logical transfers, built natively as descriptors.
    // -----------------------------------------------------------------------

    // RT-D-1: Fully contiguous — 256-byte block × 16
    {
        VPUDMADescriptor desc;
        desc.device = device;
        desc.src.dtype = DataType::UINT8;
        desc.src.setDimension_InnermostFirst({{256, 1}, {16, 256}});
        desc.dst = desc.src;
        expectSameCost_fromDesc(desc, "Desc→NN contiguous");
    }

    // RT-D-2: Source strided, destination contiguous
    {
        VPUDMADescriptor desc;
        desc.device = device;
        desc.src.dtype = DataType::UINT8;
        desc.src.setDimension_InnermostFirst({{64, 1}, {8, 128}});  // gap 64
        desc.dst.dtype = DataType::UINT8;
        desc.dst.setDimension_InnermostFirst({{64, 1}, {8, 64}});  // contiguous
        expectSameCost_fromDesc(desc, "Desc→NN src strided");
    }

    // RT-D-3: Destination strided, source contiguous
    {
        VPUDMADescriptor desc;
        desc.device = device;
        desc.src.dtype = DataType::UINT8;
        desc.src.setDimension_InnermostFirst({{64, 1}, {8, 64}});  // contiguous
        desc.dst.dtype = DataType::UINT8;
        desc.dst.setDimension_InnermostFirst({{64, 1}, {8, 128}});  // gap 64
        expectSameCost_fromDesc(desc, "Desc→NN dst strided");
    }

    // RT-D-4: Both strided, different chunk sizes
    {
        VPUDMADescriptor desc;
        desc.device = device;
        desc.src.dtype = DataType::UINT8;
        desc.src.setDimension_InnermostFirst({{32, 1}, {8, 64}});  // 32-byte chunks
        desc.dst.dtype = DataType::UINT8;
        desc.dst.setDimension_InnermostFirst({{64, 1}, {8, 96}});  // 64-byte chunks
        expectSameCost_fromDesc(desc, "Desc→NN both strided diff chunks");
    }

    // RT-D-5: Small chunks below threshold
    {
        VPUDMADescriptor desc;
        desc.device = device;
        desc.src.dtype = DataType::UINT8;
        desc.src.setDimension_InnermostFirst({{128, 1}, {2, 256}});
        desc.dst = desc.src;
        expectSameCost_fromDesc(desc, "Desc→NN small chunks");
    }

    // RT-D-6: Very small chunks, high penalty
    {
        VPUDMADescriptor desc;
        desc.device = device;
        desc.src.dtype = DataType::UINT8;
        desc.src.setDimension_InnermostFirst({{32, 1}, {100, 64}});
        desc.dst = desc.src;
        expectSameCost_fromDesc(desc, "Desc→NN very small chunks");
    }

    // RT-D-7: Asymmetric shapes
    {
        VPUDMADescriptor desc;
        desc.device = device;
        desc.src.dtype = DataType::UINT8;
        desc.src.setDimension_InnermostFirst({{64, 1}, {8, 64}, {4, 512}});
        desc.dst.dtype = DataType::UINT8;
        desc.dst.setDimension_InnermostFirst({{64, 1}, {16, 64}, {4, 1024}});
        expectSameCost_fromDesc(desc, "Desc→NN asymmetric dims");
    }

    // RT-D-8: 1D flat transfer
    {
        VPUDMADescriptor desc;
        desc.device = device;
        desc.src.dtype = DataType::UINT8;
        desc.src.setDimension_InnermostFirst({{1024, 1}});
        desc.dst = desc.src;
        expectSameCost_fromDesc(desc, "Desc→NN 1D");
    }

    // RT-D-9: Large contiguous transfer
    {
        VPUDMADescriptor desc;
        desc.device = device;
        desc.src.dtype = DataType::UINT8;
        desc.src.setDimension_InnermostFirst({{4096, 1}, {4, 4096}});
        desc.dst = desc.src;
        expectSameCost_fromDesc(desc, "Desc→NN large contiguous");
    }

    // RT-D-10: Maximum penalty
    {
        VPUDMADescriptor desc;
        desc.device = device;
        desc.src.dtype = DataType::UINT8;
        desc.src.setDimension_InnermostFirst({{1, 1}, {256, 4}});
        desc.dst = desc.src;
        expectSameCost_fromDesc(desc, "Desc→NN max penalty");
    }
}

class TestDMAStrideLUTFallback : public ::testing::Test {
protected:
    const IHWCharacteristicsSet& hw_info{HWCharacteristicsSuperSets::get_mainConfigurationRef()};
};

// threshold_bytes() must mirror the LUT's own convention: 256B for a DDR-side endpoint, 64B for a CMX-side one.
TEST_F(TestDMAStrideLUTFallback, ThresholdBytes_MatchesLUTConvention_PerDirectionAndSide) {
    struct Case {
        MemoryDirection direction;
        bool is_source;
        int expected;
    };
    const std::vector<Case> cases{
            {MemoryDirection::DDR2CMX, true, 256},   // src = DDR
            {MemoryDirection::DDR2CMX, false, 64},   // dst = CMX
            {MemoryDirection::CMX2DDR, true, 64},    // src = CMX
            {MemoryDirection::CMX2DDR, false, 256},  // dst = DDR
            {MemoryDirection::CMX2CMX, true, 64},    // src = CMX
            {MemoryDirection::CMX2CMX, false, 64},   // dst = CMX
            {MemoryDirection::DDR2DDR, true, 256},   // src = DDR
            {MemoryDirection::DDR2DDR, false, 256},  // dst = DDR
    };

    for (const auto& c : cases) {
        EXPECT_EQ(DMAStrideLUTFallback::threshold_bytes(c.direction, c.is_source), c.expected)
                << "direction=" << MemoryDirection_ToText.at(static_cast<int>(c.direction))
                << " is_source=" << c.is_source;
    }
}

// cycles_by_bandwidth() must reproduce the legacy bandwidth-based estimate: max(1, size / bytes_per_cycle).
TEST_F(TestDMAStrideLUTFallback, CyclesByBandwidth_MatchesManualFormula) {
    for (const auto device : {VPUDevice::VPU_2_0, VPUDevice::VPU_2_7, VPUDevice::VPU_4_0}) {
        const int bw_bpc = hw_info.device(device).get_DMA_DDR_interface_bytes();
        const int safe_bw_bpc = bw_bpc > 0 ? bw_bpc : 1;

        for (const unsigned long long dma_size : {0ULL, 1ULL, 64ULL, 4096ULL, 1048576ULL}) {
            const auto expected = std::max(1ULL, static_cast<unsigned long long>(static_cast<float>(dma_size) /
                                                                                  static_cast<float>(safe_bw_bpc)));

            EXPECT_EQ(DMAStrideLUTFallback::cycles_by_bandwidth(dma_size, device, hw_info), expected)
                    << "device=" << VPUDevice_ToText.at(static_cast<int>(device)) << " dma_size=" << dma_size;
        }
    }
}

// dma_size == 0 must still yield at least 1 cycle (the std::max(1ULL, ...) floor), regardless of bandwidth.
TEST_F(TestDMAStrideLUTFallback, CyclesByBandwidth_ZeroSizeFloorsToOneCycle) {
    EXPECT_EQ(DMAStrideLUTFallback::cycles_by_bandwidth(0, VPUDevice::VPU_2_7, hw_info), 1ULL);
}

/// Integration test: for a device with no LUT entry at all (e.g. legacy devices predating the stride-math
/// model), DMAStridedMathModel must route through DMAStrideLUTFallback::cycles_by_bandwidth exactly, since
/// stride penalty is also disabled for these devices (penalty_factor == 1.0, no extra rounding).
TEST(TestDMAStridedMathModel_Fallback, DeviceWithoutLUTEntry_UsesBandwidthFallback) {
    const auto& hw_info = HWCharacteristicsSuperSets::get_mainConfigurationRef();
    DMAStridedMathModel dma_model(hw_info);

    const VPUDevice device{VPUDevice::VPU_4_0};  // not present in the stride-math LUT
    const int size_bytes{4096};

    DMANNWorkload_NPU40_50 wl{device};
    wl.src_width = size_bytes;
    wl.dst_width = size_bytes;
    wl.num_dim = 0;  // contiguous 1D transfer
    wl.transfer_direction = MemoryDirection::DDR2CMX;

    const auto expected = DMAStrideLUTFallback::cycles_by_bandwidth(size_bytes, device, hw_info);

    EXPECT_EQ(dma_model.computeSrcCycles(wl), expected);
    EXPECT_EQ(dma_model.computeDstCycles(wl), expected);
}

}  // namespace VPUNN_unit_tests
