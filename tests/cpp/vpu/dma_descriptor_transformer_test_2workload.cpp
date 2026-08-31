// Copyright © 2026 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the "Software Package")
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the "third-party-programs.txt" or other similarly-named text file included with the
// Software Package for additional details.

/// @file dma_descriptor_transformer_test_2workload.cpp
/// @brief Unit tests for DMADescriptorTransformer::fromVPUDMADescriptor_to_DMAWorkload.
///
/// Test prefix: D2W_ (Descriptor-to-Workload)
///
/// Contract under test:
///   - All source/destination dimensions are collapsed into a single 1D tensor.
///   - Byte-aligned dtypes (UINT8, INT8, FLOAT16, INT32, …) are preserved as-is.
///   - Sub-byte dtypes are promoted to their 8-bit sibling (see to_byte_aligned_dtype).
///   - input.size()  == src.getAccessedBytes()  (== desc.getSrcTransferBytes())
///   - output.size() == dst.getAccessedBytes()  (== desc.getDstTransferBytes())
///   - device, input_location, output_location are copied verbatim.

#include "vpu/dma_descriptor_transformers.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "vpu/dma_descriptors.h"
#include "vpu/dma_types.h"
#include "vpu/dma_workload.h"
#include "vpu/dpu_dtypes_dimension_info.h"

namespace VPUNN_unit_tests {
using namespace VPUNN;

// ---------------------------------------------------------------------------
// File-local helpers
// ---------------------------------------------------------------------------

/// Build a symmetric VPUDMADescriptor (identical src and dst tensors).
/// Dimensions are specified outermost-first: each pair is {element_count, byte_stride}.
static VPUDMADescriptor makeDesc_OF(VPUDevice device, MemoryLocation src_loc, MemoryLocation dst_loc, DataType dtype,
                                    std::initializer_list<std::pair<int32_t, int32_t>> dims_outermost_first) {
    VPUDMADescriptor desc;
    desc.device = device;
    desc.src_location = src_loc;
    desc.dst_location = dst_loc;
    desc.src.dtype = dtype;
    desc.src.setDimension_OutermostFirst(dims_outermost_first);
    desc.dst.dtype = dtype;
    desc.dst.setDimension_OutermostFirst(dims_outermost_first);
    return desc;
}

/// Build a VPUDMADescriptor with independent src and dst tensors.
static VPUDMADescriptor makeDescAsymmetric_OF(VPUDevice device, MemoryLocation src_loc, MemoryLocation dst_loc,
                                              DataType src_dtype,
                                              std::initializer_list<std::pair<int32_t, int32_t>> src_dims,
                                              DataType dst_dtype,
                                              std::initializer_list<std::pair<int32_t, int32_t>> dst_dims) {
    VPUDMADescriptor desc;
    desc.device = device;
    desc.src_location = src_loc;
    desc.dst_location = dst_loc;
    desc.src.dtype = src_dtype;
    desc.src.setDimension_OutermostFirst(src_dims);
    desc.dst.dtype = dst_dtype;
    desc.dst.setDimension_OutermostFirst(dst_dims);
    return desc;
}

/// Calls the function under test.
static DMAWorkload d2w(const VPUDMADescriptor& desc) {
    return DMADescriptorTransformer::fromVPUDMADescriptor_to_DMAWorkload(desc);
}

// ---------------------------------------------------------------------------
// Test fixture
// ---------------------------------------------------------------------------

class TestDMADescriptorTransformer_D2W : public ::testing::Test {
protected:
    /// Verify that the workload byte counts equal the descriptor's transfer byte counts.
    void expectBytesMatch(const DMAWorkload& wl, const VPUDMADescriptor& desc) {
        EXPECT_EQ(wl.input.size(), static_cast<unsigned int>(desc.getSrcTransferBytes()))
                << "input.size() must equal desc.getSrcTransferBytes()";
        EXPECT_EQ(wl.output.size(), static_cast<unsigned int>(desc.getDstTransferBytes()))
                << "output.size() must equal desc.getDstTransferBytes()";
    }
};

// ===========================================================================
// D2W_Device — device field is copied verbatim
// ===========================================================================

TEST_F(TestDMADescriptorTransformer_D2W, D2W_Device_NPU40) {
    auto desc = makeDesc_OF(VPUDevice::VPU_4_0, MemoryLocation::DRAM, MemoryLocation::CMX, DataType::UINT8, {{256, 1}});
    EXPECT_EQ(d2w(desc).device, VPUDevice::VPU_4_0);
}

TEST_F(TestDMADescriptorTransformer_D2W, D2W_Device_NPU50) {
    auto desc = makeDesc_OF(VPUDevice::NPU_5_0, MemoryLocation::DRAM, MemoryLocation::CMX, DataType::UINT8, {{256, 1}});
    EXPECT_EQ(d2w(desc).device, VPUDevice::NPU_5_0);
}

// ===========================================================================
// D2W_Locations — src_location→input_location, dst_location→output_location
// ===========================================================================

TEST_F(TestDMADescriptorTransformer_D2W, D2W_Locations_DDR2CMX) {
    auto desc = makeDesc_OF(VPUDevice::VPU_4_0, MemoryLocation::DRAM, MemoryLocation::CMX, DataType::UINT8, {{64, 1}});
    const auto wl = d2w(desc);
    EXPECT_EQ(wl.input_location, MemoryLocation::DRAM) << "src_location → input_location";
    EXPECT_EQ(wl.output_location, MemoryLocation::CMX) << "dst_location → output_location";
    expectBytesMatch(wl, desc);
}

TEST_F(TestDMADescriptorTransformer_D2W, D2W_Locations_CMX2DDR) {
    auto desc = makeDesc_OF(VPUDevice::VPU_4_0, MemoryLocation::CMX, MemoryLocation::DRAM, DataType::UINT8, {{64, 1}});
    const auto wl = d2w(desc);
    EXPECT_EQ(wl.input_location, MemoryLocation::CMX);
    EXPECT_EQ(wl.output_location, MemoryLocation::DRAM);
    expectBytesMatch(wl, desc);
}

TEST_F(TestDMADescriptorTransformer_D2W, D2W_Locations_CMX2CMX) {
    auto desc = makeDesc_OF(VPUDevice::VPU_4_0, MemoryLocation::CMX, MemoryLocation::CMX, DataType::UINT8, {{64, 1}});
    const auto wl = d2w(desc);
    EXPECT_EQ(wl.input_location, MemoryLocation::CMX);
    EXPECT_EQ(wl.output_location, MemoryLocation::CMX);
    expectBytesMatch(wl, desc);
}

TEST_F(TestDMADescriptorTransformer_D2W, D2W_Locations_DDR2DDR) {
    auto desc = makeDesc_OF(VPUDevice::VPU_4_0, MemoryLocation::DRAM, MemoryLocation::DRAM, DataType::UINT8, {{64, 1}});
    const auto wl = d2w(desc);
    EXPECT_EQ(wl.input_location, MemoryLocation::DRAM);
    EXPECT_EQ(wl.output_location, MemoryLocation::DRAM);
    expectBytesMatch(wl, desc);
}

// ===========================================================================
// D2W_Dtype — byte-aligned types are preserved; sub-byte types are promoted
// ===========================================================================

/// UINT8 1D: dtype preserved as UINT8; byte count == getSrcTransferBytes().
TEST_F(TestDMADescriptorTransformer_D2W, D2W_DtypeCollapse_UINT8_Passthrough) {
    auto desc = makeDesc_OF(VPUDevice::VPU_4_0, MemoryLocation::DRAM, MemoryLocation::CMX, DataType::UINT8, {{256, 1}});
    const auto wl = d2w(desc);
    EXPECT_EQ(wl.input.get_dtype(), DataType::UINT8) << "UINT8 src preserved";
    EXPECT_EQ(wl.output.get_dtype(), DataType::UINT8) << "UINT8 dst preserved";
    expectBytesMatch(wl, desc);
}

/// FP16 1D: dtype preserved as FLOAT16; byte count = 128 * 2 = 256.
TEST_F(TestDMADescriptorTransformer_D2W, D2W_DtypeCollapse_FP16_PreservedAsFloat16) {
    auto desc =
            makeDesc_OF(VPUDevice::VPU_4_0, MemoryLocation::DRAM, MemoryLocation::CMX, DataType::FLOAT16, {{128, 2}});
    const auto wl = d2w(desc);
    EXPECT_EQ(wl.input.get_dtype(), DataType::FLOAT16) << "FP16 src dtype preserved";
    EXPECT_EQ(wl.output.get_dtype(), DataType::FLOAT16) << "FP16 dst dtype preserved";
    expectBytesMatch(wl, desc);
}

/// INT32 1D: dtype preserved as INT32; byte count = 64 * 4 = 256.
TEST_F(TestDMADescriptorTransformer_D2W, D2W_DtypeCollapse_INT32_PreservedAsInt32) {
    auto desc = makeDesc_OF(VPUDevice::VPU_4_0, MemoryLocation::DRAM, MemoryLocation::CMX, DataType::INT32, {{64, 4}});
    const auto wl = d2w(desc);
    EXPECT_EQ(wl.input.get_dtype(), DataType::INT32) << "INT32 src dtype preserved";
    EXPECT_EQ(wl.output.get_dtype(), DataType::INT32) << "INT32 dst dtype preserved";
    expectBytesMatch(wl, desc);
}

// ===========================================================================
// D2W_MultiDimCollapse — all dimensions fold into one 1D tensor; dtype preserved
// ===========================================================================

/// 2D fully packed UINT8 {16×64}: collapses to 1024-byte UINT8 tensor.
TEST_F(TestDMADescriptorTransformer_D2W, D2W_MultiDim_2D_Packed_CollapsesTo1024Bytes) {
    auto desc = makeDesc_OF(VPUDevice::VPU_4_0, MemoryLocation::DRAM, MemoryLocation::CMX, DataType::UINT8,
                            {{16, 64}, {64, 1}});
    const auto wl = d2w(desc);
    EXPECT_EQ(wl.input.get_dtype(), DataType::UINT8);
    EXPECT_EQ(wl.output.get_dtype(), DataType::UINT8);
    EXPECT_EQ(wl.input.size(), 1024u) << "16×64 UINT8 = 1024 bytes";
    expectBytesMatch(wl, desc);
}

/// 2D strided UINT8 {16 rows, outer_stride=128, 64 cols}:
/// strides are discarded; only logical element count (16*64=1024) matters.
TEST_F(TestDMADescriptorTransformer_D2W, D2W_MultiDim_2D_Strided_SizeIsAccessedBytes) {
    auto desc = makeDesc_OF(VPUDevice::VPU_4_0, MemoryLocation::DRAM, MemoryLocation::CMX, DataType::UINT8,
                            {{16, 128}, {64, 1}});
    const auto wl = d2w(desc);
    EXPECT_EQ(wl.input.size(), 1024u) << "16*64 = 1024 logical bytes (outer stride=128 ignored)";
    expectBytesMatch(wl, desc);
}

/// 3D FP16 {4,8,32} all packed: accessed = 4*8*32*2 = 2048 bytes.
TEST_F(TestDMADescriptorTransformer_D2W, D2W_MultiDim_3D_FP16_Packed_CollapsesTo2048Bytes) {
    auto desc = makeDesc_OF(VPUDevice::VPU_4_0, MemoryLocation::DRAM, MemoryLocation::CMX, DataType::FLOAT16,
                            {{4, 512}, {8, 64}, {32, 2}});
    const auto wl = d2w(desc);
    EXPECT_EQ(wl.input.get_dtype(), DataType::FLOAT16) << "FP16 dtype preserved after multi-dim collapse";
    EXPECT_EQ(wl.input.size(), 2048u) << "4*8*32 FP16 = 2048 bytes";
    expectBytesMatch(wl, desc);
}

/// 3D UINT8 with middle-dim gap: accessed = 4*8*32 = 1024 bytes regardless of strides.
TEST_F(TestDMADescriptorTransformer_D2W, D2W_MultiDim_3D_GapInMiddle_SizeIsLogicalBytes) {
    auto desc = makeDesc_OF(VPUDevice::VPU_4_0, MemoryLocation::DRAM, MemoryLocation::CMX, DataType::UINT8,
                            {{4, 512}, {8, 128}, {32, 1}});
    const auto wl = d2w(desc);
    EXPECT_EQ(wl.input.size(), 1024u) << "4*8*32 = 1024 logical bytes (gaps ignored)";
    expectBytesMatch(wl, desc);
}

/// 3D FP16 with gap in innermost (stride=4 ≠ elem_bytes=2): accessed = 4*8*32*2 = 2048 bytes.
TEST_F(TestDMADescriptorTransformer_D2W, D2W_MultiDim_3D_FP16_InnerGap_SizeIsLogicalBytes) {
    auto desc = makeDesc_OF(VPUDevice::VPU_4_0, MemoryLocation::DRAM, MemoryLocation::CMX, DataType::FLOAT16,
                            {{4, 1024}, {8, 256}, {32, 4}});
    const auto wl = d2w(desc);
    EXPECT_EQ(wl.input.get_dtype(), DataType::FLOAT16) << "FP16 dtype preserved with inner stride gap";
    EXPECT_EQ(wl.output.get_dtype(), DataType::FLOAT16) << "FP16 dtype preserved with inner stride gap";
    EXPECT_EQ(wl.input.size(), 2048u) << "4*8*32 FP16 = 2048 logical bytes";
    expectBytesMatch(wl, desc);
}

// ===========================================================================
// D2W_Asymmetric — src and dst collapsed independently
// ===========================================================================

/// Src: FP16 128 elems (256 bytes).  Dst: UINT8 256 elems (256 bytes).
TEST_F(TestDMADescriptorTransformer_D2W, D2W_Asymmetric_FP16src_UINT8dst_EqualBytes) {
    auto desc = makeDescAsymmetric_OF(VPUDevice::VPU_4_0, MemoryLocation::DRAM, MemoryLocation::CMX, DataType::FLOAT16,
                                      {{128, 2}}, DataType::UINT8, {{256, 1}});
    const auto wl = d2w(desc);
    EXPECT_EQ(wl.input.get_dtype(), DataType::FLOAT16) << "FP16 src dtype preserved";
    EXPECT_EQ(wl.output.get_dtype(), DataType::UINT8) << "UINT8 dst dtype preserved";
    expectBytesMatch(wl, desc);
}

/// Src: UINT8 512 elems (512 bytes).  Dst: FP16 512 elems (1024 bytes).
TEST_F(TestDMADescriptorTransformer_D2W, D2W_Asymmetric_DifferentSrcDstBytes) {
    auto desc = makeDescAsymmetric_OF(VPUDevice::VPU_4_0, MemoryLocation::DRAM, MemoryLocation::CMX, DataType::UINT8,
                                      {{512, 1}}, DataType::FLOAT16, {{512, 2}});
    const auto wl = d2w(desc);
    EXPECT_EQ(wl.input.size(), 512u) << "src: 512 UINT8 → 512 bytes";
    EXPECT_EQ(wl.output.size(), 1024u) << "dst: 512 FP16 → 1024 bytes";
    expectBytesMatch(wl, desc);
}

/// Src: INT32 2D strided, dst: UINT8 1D packed.
TEST_F(TestDMADescriptorTransformer_D2W, D2W_Asymmetric_INT32src_Strided_UINT8dst_Packed) {
    // Src: {16, stride=128}, {16, stride=4} INT32 → accessed = 16*16*4 = 1024 bytes
    auto desc = makeDescAsymmetric_OF(VPUDevice::VPU_4_0, MemoryLocation::DRAM, MemoryLocation::CMX, DataType::INT32,
                                      {{16, 128}, {16, 4}}, DataType::UINT8, {{1024, 1}});
    const auto wl = d2w(desc);
    EXPECT_EQ(wl.input.get_dtype(), DataType::INT32) << "INT32 src dtype preserved";
    EXPECT_EQ(wl.output.get_dtype(), DataType::UINT8) << "UINT8 dst dtype preserved";
    expectBytesMatch(wl, desc);
}

// ===========================================================================
// D2W_ByteCountEqualsTransferBytes — explicit check for all dtype / shape combos
// ===========================================================================

TEST_F(TestDMADescriptorTransformer_D2W, D2W_ByteCount_1D_Packed_FP16) {
    auto desc =
            makeDesc_OF(VPUDevice::VPU_4_0, MemoryLocation::DRAM, MemoryLocation::CMX, DataType::FLOAT16, {{64, 2}});
    expectBytesMatch(d2w(desc), desc);
}

TEST_F(TestDMADescriptorTransformer_D2W, D2W_ByteCount_2D_OuterGap_UINT8) {
    auto desc = makeDesc_OF(VPUDevice::VPU_4_0, MemoryLocation::DRAM, MemoryLocation::CMX, DataType::UINT8,
                            {{16, 128}, {64, 1}});
    expectBytesMatch(d2w(desc), desc);
}

TEST_F(TestDMADescriptorTransformer_D2W, D2W_ByteCount_3D_FP16_GapMiddle) {
    auto desc = makeDesc_OF(VPUDevice::VPU_4_0, MemoryLocation::DRAM, MemoryLocation::CMX, DataType::FLOAT16,
                            {{4, 512}, {8, 200}, {32, 2}});
    expectBytesMatch(d2w(desc), desc);
}

TEST_F(TestDMADescriptorTransformer_D2W, D2W_ByteCount_3D_INT32_AllPacked) {
    auto desc = makeDesc_OF(VPUDevice::VPU_4_0, MemoryLocation::DRAM, MemoryLocation::CMX, DataType::INT32,
                            {{4, 512}, {8, 128}, {32, 4}});
    expectBytesMatch(d2w(desc), desc);
}

TEST_F(TestDMADescriptorTransformer_D2W, D2W_ByteCount_6D_AllDimsStrided_UINT8) {
    auto desc = makeDesc_OF(VPUDevice::VPU_4_0, MemoryLocation::DRAM, MemoryLocation::CMX, DataType::UINT8,
                            {{2, 256}, {2, 128}, {2, 64}, {2, 32}, {2, 8}, {4, 1}});
    expectBytesMatch(d2w(desc), desc);
}

// ===========================================================================
// D2W_ToByteAlignedDtype — unit tests for
//   DMADescriptorTransformer::to_byte_aligned_dtype(DataType)
//
// Contract:
//   1. A type whose dtype_to_bits() is a positive multiple of 8 is returned
//      unchanged (already byte-aligned).
//   2. A non-byte-aligned type is promoted to its 8-bit family sibling:
//        UINT1 / UINT2 / UINT4  → UINT8
//        INT1  / INT2  / INT4   → INT8
//        FLOAT4                 → BF8
//   3. Any future non-byte-aligned type (unknown to the switch) falls back to
//      UINT8 — captured by checking the bit-width guard is driven by
//      dtype_to_bits(), not just an explicit enum list.
// ===========================================================================

// ---------------------------------------------------------------------------
// Already byte-aligned types — must be returned unchanged.
// ---------------------------------------------------------------------------

TEST_F(TestDMADescriptorTransformer_D2W, ToByteAlignedDtype_UINT8_Unchanged) {
    EXPECT_EQ(DMADescriptorTransformer::to_byte_aligned_dtype(DataType::UINT8), DataType::UINT8);
}
TEST_F(TestDMADescriptorTransformer_D2W, ToByteAlignedDtype_INT8_Unchanged) {
    EXPECT_EQ(DMADescriptorTransformer::to_byte_aligned_dtype(DataType::INT8), DataType::INT8);
}
TEST_F(TestDMADescriptorTransformer_D2W, ToByteAlignedDtype_BF8_Unchanged) {
    EXPECT_EQ(DMADescriptorTransformer::to_byte_aligned_dtype(DataType::BF8), DataType::BF8);
}
TEST_F(TestDMADescriptorTransformer_D2W, ToByteAlignedDtype_HF8_Unchanged) {
    EXPECT_EQ(DMADescriptorTransformer::to_byte_aligned_dtype(DataType::HF8), DataType::HF8);
}
TEST_F(TestDMADescriptorTransformer_D2W, ToByteAlignedDtype_FLOAT16_Unchanged) {
    EXPECT_EQ(DMADescriptorTransformer::to_byte_aligned_dtype(DataType::FLOAT16), DataType::FLOAT16);
}
TEST_F(TestDMADescriptorTransformer_D2W, ToByteAlignedDtype_BFLOAT16_Unchanged) {
    EXPECT_EQ(DMADescriptorTransformer::to_byte_aligned_dtype(DataType::BFLOAT16), DataType::BFLOAT16);
}
TEST_F(TestDMADescriptorTransformer_D2W, ToByteAlignedDtype_UINT16_Unchanged) {
    EXPECT_EQ(DMADescriptorTransformer::to_byte_aligned_dtype(DataType::UINT16), DataType::UINT16);
}
TEST_F(TestDMADescriptorTransformer_D2W, ToByteAlignedDtype_INT16_Unchanged) {
    EXPECT_EQ(DMADescriptorTransformer::to_byte_aligned_dtype(DataType::INT16), DataType::INT16);
}
TEST_F(TestDMADescriptorTransformer_D2W, ToByteAlignedDtype_FLOAT32_Unchanged) {
    EXPECT_EQ(DMADescriptorTransformer::to_byte_aligned_dtype(DataType::FLOAT32), DataType::FLOAT32);
}
TEST_F(TestDMADescriptorTransformer_D2W, ToByteAlignedDtype_INT32_Unchanged) {
    EXPECT_EQ(DMADescriptorTransformer::to_byte_aligned_dtype(DataType::INT32), DataType::INT32);
}

// ---------------------------------------------------------------------------
// Unsigned sub-byte types — must be promoted to UINT8.
// ---------------------------------------------------------------------------

TEST_F(TestDMADescriptorTransformer_D2W, ToByteAlignedDtype_UINT4_PromotedTo_UINT8) {
    EXPECT_EQ(DMADescriptorTransformer::to_byte_aligned_dtype(DataType::UINT4), DataType::UINT8)
            << "UINT4 (4 bits) must promote to UINT8";
}
TEST_F(TestDMADescriptorTransformer_D2W, ToByteAlignedDtype_UINT2_PromotedTo_UINT8) {
    EXPECT_EQ(DMADescriptorTransformer::to_byte_aligned_dtype(DataType::UINT2), DataType::UINT8)
            << "UINT2 (2 bits) must promote to UINT8";
}
TEST_F(TestDMADescriptorTransformer_D2W, ToByteAlignedDtype_UINT1_PromotedTo_UINT8) {
    EXPECT_EQ(DMADescriptorTransformer::to_byte_aligned_dtype(DataType::UINT1), DataType::UINT8)
            << "UINT1 (1 bit) must promote to UINT8";
}

// ---------------------------------------------------------------------------
// Signed sub-byte types — must be promoted to INT8.
// ---------------------------------------------------------------------------

TEST_F(TestDMADescriptorTransformer_D2W, ToByteAlignedDtype_INT4_PromotedTo_INT8) {
    EXPECT_EQ(DMADescriptorTransformer::to_byte_aligned_dtype(DataType::INT4), DataType::INT8)
            << "INT4 (4 bits) must promote to INT8";
}
TEST_F(TestDMADescriptorTransformer_D2W, ToByteAlignedDtype_INT2_PromotedTo_INT8) {
    EXPECT_EQ(DMADescriptorTransformer::to_byte_aligned_dtype(DataType::INT2), DataType::INT8)
            << "INT2 (2 bits) must promote to INT8";
}
TEST_F(TestDMADescriptorTransformer_D2W, ToByteAlignedDtype_INT1_PromotedTo_INT8) {
    EXPECT_EQ(DMADescriptorTransformer::to_byte_aligned_dtype(DataType::INT1), DataType::INT8)
            << "INT1 (1 bit) must promote to INT8";
}

// ---------------------------------------------------------------------------
// Float sub-byte type — FLOAT4 must be promoted to BF8.
// ---------------------------------------------------------------------------

TEST_F(TestDMADescriptorTransformer_D2W, ToByteAlignedDtype_FLOAT4_PromotedTo_BF8) {
    EXPECT_EQ(DMADescriptorTransformer::to_byte_aligned_dtype(DataType::FLOAT4), DataType::BF8)
            << "FLOAT4 (4 bits) must promote to its 8-bit float sibling BF8";
}

// ---------------------------------------------------------------------------
// Promotion invariants — sweeps over all known types.
// ---------------------------------------------------------------------------

/// Every sub-byte type must promote to a result with exactly 8 bits.
/// This validates the byte-footprint contract for all currently known sub-byte dtypes.
TEST_F(TestDMADescriptorTransformer_D2W, ToByteAlignedDtype_AllSubByteTypes_ResultIs8Bits) {
    const std::vector<DataType> sub_byte_types = {
            DataType::UINT4, DataType::UINT2, DataType::UINT1,  DataType::INT4,
            DataType::INT2,  DataType::INT1,  DataType::FLOAT4,
    };
    for (const DataType dt : sub_byte_types) {
        const DataType promoted = DMADescriptorTransformer::to_byte_aligned_dtype(dt);
        const int promoted_bits = DTypeDimensionInfo::dtype_to_bits(promoted);
        EXPECT_EQ(promoted_bits, 8) << "Promoted type for " << DataType_ToText.at(static_cast<int>(dt))
                                    << " must be exactly 8 bits; got " << promoted_bits;
        EXPECT_EQ(promoted_bits % 8, 0) << "Promoted type must be byte-aligned";
    }
}

/// Every already-byte-aligned type must survive promotion unchanged (idempotency).
TEST_F(TestDMADescriptorTransformer_D2W, ToByteAlignedDtype_ByteAlignedTypes_Idempotent) {
    const std::vector<DataType> aligned_types = {
            DataType::UINT8,    DataType::INT8,   DataType::BF8,   DataType::HF8,     DataType::FLOAT16,
            DataType::BFLOAT16, DataType::UINT16, DataType::INT16, DataType::FLOAT32, DataType::INT32,
    };
    for (const DataType dt : aligned_types) {
        EXPECT_EQ(DMADescriptorTransformer::to_byte_aligned_dtype(dt), dt)
                << "Already-aligned type " << DataType_ToText.at(static_cast<int>(dt)) << " must not be changed";
    }
}

}  // namespace VPUNN_unit_tests
