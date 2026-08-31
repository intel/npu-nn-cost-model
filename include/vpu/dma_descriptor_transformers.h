// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_DMA_DESCRIPTOR_TRANSFORMERS_H
#define VPUNN_DMA_DESCRIPTOR_TRANSFORMERS_H

/// @file dma_descriptor_transformers.h
/// @brief Transformation utilities between different DMA workload flavors.
///
/// Provides conversions among the various DMA descriptor/workload types used
/// across NPU generations (e.g. DMAWorkload, DMANNWorkload_NPU27,
/// DMANNWorkload_NPU40_50, VPUDMADescriptor). Not every combination needs to
/// be supported; only the conversions that are meaningful for a given context
/// are expected to be implemented.

#include <functional>
#include "vpu/dma_descriptors.h"
#include "vpu/dma_types.h"
#include "vpu/dma_workload.h"
#include "vpu/dpu_dtypes_dimension_info.h"

namespace VPUNN {

/// @brief Generic transformation helpers between different DMA workload flavors.
///
/// Supports conversions among NPU-generation-specific workload types.
/// Not all possible conversion pairs are required to be covered.
/// All methods are static; the class is not meant to be instantiated.
class DMADescriptorTransformer {
public:
    /// @brief Construct a DMANNWorkload_NPU40_50 from a VPUDMADescriptor.
    ///
    /// Collapses contiguous inner dimensions into the innermost byte block and
    /// encodes outer dimensions as 0-based sizes + strides, matching the NPU4.0+
    /// 6D HW descriptor layout.
    ///
    /// @param desc  Source descriptor carrying device, src/dst tensors, and memory locations.
    /// @return  A fully populated DMANNWorkload_NPU40_50.
    static inline DMANNWorkload_NPU40_50 fromStridedTensors_toDMANNWorkload_NPU40_50(const VPUDMADescriptor& desc) {
        using SizeStride = DMANNWorkload_NPU40_50::SizeStride;
        constexpr int MaxExtra = DMANNWorkload_NPU40_50::MaxExtraDimensions;

        DMANNWorkload_NPU40_50 wl;
        wl.device = desc.device;
        wl.transfer_direction = desc.getDirection();
        wl.num_engine = Num_DMA_Engine::Num_Engine_1;

        // ---------------------------------------------------------------------------
        // collapse_side: converts one VPUDMATensor (outermost-first layout) into the
        // HW descriptor's width + e_dim[] encoding.
        //
        // Algorithm (mirrors VPUDMATensor::getContiguousBytes() exactly):
        //
        //   1. Seed: contiguous_bytes = dtype_to_bytes(dtype)  — one element.
        //
        //   2. Walk ALL dims from innermost (num_dims-1) down to outermost (0):
        //      • shape == 1  → degenerate (stride never stepped), skip without breaking
        //                      the chain.
        //      • stride == contiguous_bytes → packed: absorb (multiply) into the block.
        //      • otherwise   → gap found: stop. This dim (and all outer dims) become
        //                      e_dim[] entries.
        //
        //   3. out_width = contiguous_bytes after step 2.
        //
        //   4. The dimension that broke contiguity and every outer dim (non-degenerate)
        //      are encoded as e_dim[] entries in innermost-first order.
        //      Degenerate dims (shape==1) are skipped — their stride is never applied.
        //
        // NOTE: when the innermost dimension itself has a stride gap (stride != elem_bytes),
        // it is NOT absorbed and becomes e_dim[0], giving out_width = elem_bytes. This
        // correctly aligns with VPUDMATensor::getContiguousBytes() which uses the same rule.
        // ---------------------------------------------------------------------------
        auto collapse_side = [&](const DataType dtype, const int num_dims,
                                 const std::array<int32_t, VPU_DMA_MAX_DIMS>& shape,
                                 const std::array<int32_t, VPU_DMA_MAX_DIMS>& strides, int& out_width,
                                 std::array<SizeStride, MaxExtra>& out_dims, int& out_extra_count) {
            out_width = 0;
            out_extra_count = 0;
            if (num_dims <= 0) {
                return;
            }

            // Empty tensor: any dimension with shape == 0 means no elements exist.
            // Mirrors VPUDMATensor::getContiguousBytes() which returns 0 in this case.
            // Early-exit to avoid: (a) multiplying by 0 in the contiguity walk, and
            // (b) encoding shape[d]-1 == -1 as an e_dim entry (invalid HW descriptor).
            for (int d = num_dims - 1; d >= 0; --d) {
                if (shape[d] == 0) {
                    return;  // out_width=0, out_extra_count=0 already set above
                }
            }

            // Seed: one element — mirrors VPUDMATensor::getContiguousBytes().
            // Guard: dtype_to_bytes() returns -1 for unknown/unsupported types.
            // A VPUDMADescriptor that passed checkDescriptorSanity() is guaranteed to have
            // a byte-aligned dtype here, so elem_bytes <= 0 should never occur in practice.
            // The early-exit mirrors VPUDMATensor::getContiguousBytes() which also returns 0
            // for invalid dtypes, keeping both paths consistent without relying on the sanity
            // check having been called.
            int32_t contiguous_bytes = static_cast<int32_t>(DTypeDimensionInfo::dtype_to_bytes(dtype));
            if (contiguous_bytes <= 0) {
                return;  // out_width=0, out_extra_count=0 already set above
            }

            // Walk ALL dimensions from innermost to outermost.
            // first_non_packed is the index of the first dimension where contiguity breaks,
            // or -1 when every dimension is absorbed.
            int first_non_packed = -1;
            for (int d = num_dims - 1; d >= 0; --d) {
                if (shape[d] == 1) {
                    // Degenerate: stride is never used, skip without breaking the chain.
                    continue;
                }
                if (strides[d] == contiguous_bytes) {
                    // Packed: absorb this dimension into the contiguous block.
                    contiguous_bytes *= shape[d];
                } else {
                    // Gap: stop here; this dim and all outer dims become e_dim[] entries.
                    first_non_packed = d;
                    break;
                }
            }

            out_width = static_cast<int>(contiguous_bytes);

            // Encode non-contiguous dimensions as HW e_dim[] entries.
            // Walk from first_non_packed down to 0 (innermost non-packed → outermost).
            // Degenerate dims (shape==1) are skipped — they contribute nothing.
            //
            // Overflow handling (e.g. 6D tensor with a strided innermost dim):
            // When all MaxExtra slots are already filled and further outer dimensions
            // remain, each overflow dim is folded into the last slot by multiplying
            // its element count into last_slot.src_dim_size:
            //
            //   last_slot.src_dim_size = (last_slot.src_dim_size + 1) * shape[d] - 1
            //
            // This preserves the correct total element count (and therefore
            // getAccessedBytes()), at the cost of losing the stride of the overflow
            // dimension.  The stride stored in the last slot (from the previously
            // encoded dim) is retained as-is.
            int e_index = 0;
            for (int d = first_non_packed; d >= 0; --d) {
                if (shape[d] == 1) {
                    continue;  // degenerate: skip
                }
                if (e_index < MaxExtra) {
                    out_dims[e_index] = {};
                    out_dims[e_index].src_stride = static_cast<int>(strides[d]);
                    out_dims[e_index].src_dim_size = static_cast<int>(shape[d] - 1);
                    ++e_index;
                } else {
                    // No free slot: fold element count into the last representable slot.
                    // Stride of this overflow dim is lost; accessed-bytes count is preserved.
                    const int last = e_index - 1;
                    out_dims[last].src_dim_size = (out_dims[last].src_dim_size + 1) * static_cast<int>(shape[d]) - 1;
                }
            }

            out_extra_count = e_index;
        };

        std::array<SizeStride, MaxExtra> src_tmp{};
        std::array<SizeStride, MaxExtra> dst_tmp{};
        int src_extra = 0;
        int dst_extra = 0;

        collapse_side(desc.src.dtype, desc.src.num_dims, desc.src.shape, desc.src.byte_strides, wl.src_width, src_tmp,
                      src_extra);
        collapse_side(desc.dst.dtype, desc.dst.num_dims, desc.dst.shape, desc.dst.byte_strides, wl.dst_width, dst_tmp,
                      dst_extra);

        wl.num_dim = std::min(std::max(src_extra, dst_extra), MaxExtra);

        // Merge src and dst extra-dim info into the final e_dim array.
        // Slots beyond the respective side's extra count are zero-padded.
        for (int d = 0; d < wl.num_dim; ++d) {
            wl.e_dim[d].src_stride = (d < src_extra) ? src_tmp[d].src_stride : 0;
            wl.e_dim[d].src_dim_size = (d < src_extra) ? src_tmp[d].src_dim_size : 0;
            wl.e_dim[d].dst_stride = (d < dst_extra) ? dst_tmp[d].src_stride : 0;
            wl.e_dim[d].dst_dim_size = (d < dst_extra) ? dst_tmp[d].src_dim_size : 0;
        }

        return wl;
    }

    // ======================================================================
    //  Upward conversion: create a VPUDMADescriptor from a legacy DMAWorkload
    // ======================================================================

    /// @brief Promotes a legacy DMAWorkload to a VPUDMADescriptor.
    ///
    /// Each side (source and destination) is collapsed to a flat **1D contiguous**
    /// buffer.  The dtype is preserved when it is already byte-aligned, and promoted
    /// to the nearest byte-aligned sibling via `to_byte_aligned_dtype` otherwise:
    ///
    ///   | Original dtype      | Descriptor dtype | num_elems              | stride       |
    ///   |---------------------|------------------|------------------------|--------------|
    ///   | UINT8               | UINT8            | total_bytes            | 1            |
    ///   | FLOAT16             | FLOAT16          | total_bytes / 2        | 2            |
    ///   | INT32               | INT32            | total_bytes / 4        | 4            |
    ///   | UINT4 (sub-byte)    | UINT8            | total_bytes (= N/2)    | 1            |
    ///   | INT4  (sub-byte)    | INT8             | total_bytes (= N/2)    | 1            |
    ///   | FLOAT4 (sub-byte)   | BF8              | total_bytes (= N/2)    | 1            |
    ///   | UINT1 (sub-byte)    | UINT8            | total_bytes (= N/8)    | 1            |
    ///
    /// **Why preserve dtype for byte-aligned types?**
    ///
    /// Retaining the original dtype (e.g. FLOAT16, INT32) lets downstream consumers
    /// (cost models, serializers) know the element granularity of the transfer without
    /// resorting to byte-level re-interpretation.
    ///
    /// **Why promote sub-byte types?**
    ///
    /// VPUDMADescriptor strides are expressed in whole bytes; a sub-byte element cannot
    /// have a byte stride.  `to_byte_aligned_dtype` maps every sub-byte type to its
    /// 8-bit family sibling so the stride is always a valid integer number of bytes.
    /// `VPUTensor::size()` already returns the byte-aligned packed size for sub-byte
    /// tensors, so `total_bytes` is always a whole number regardless of dtype.
    ///
    /// **Fallback to UINT8:**
    ///
    /// If `total_bytes % elem_bytes != 0` (unusual edge case for sub-byte types whose
    /// packing is non-trivial) the dtype is further degraded to UINT8 and
    /// `num_elems = total_bytes`, preserving the byte footprint exactly.
    ///
    /// @note Source and destination are handled independently so that type-conversion
    ///       DMAs with different src/dst dtypes are supported.
    ///
    /// **Zero-byte tensors are allowed.**  A DMAWorkload with zero elements produces
    /// a descriptor with `shape[0]=0`, which the cost model handles as a fixed-latency
    /// empty transfer.
    ///
    /// @param wl  Legacy DMAWorkload to promote.
    /// @return    Equivalent VPUDMADescriptor with 1D contiguous src and dst tensors
    ///            whose dtype reflects the original workload type (or its byte-aligned upgrade).
    static inline VPUDMADescriptor fromDMAWorkload_to_VPUDMADescriptor(const DMAWorkload& wl) {
        // Collapse one VPUTensor side into a 1D contiguous VPUDMATensor entry.
        // Mirrors the logic of fromVPUDMADescriptor_to_DMAWorkload in reverse:
        // byte-aligned types are preserved, sub-byte types are promoted.
        const auto make_side = [](const VPUTensor& t, VPUDMATensor& side) {
            const int32_t total_bytes = static_cast<int32_t>(t.size());

            // Step 1: promote sub-byte types to the nearest byte-aligned sibling.
            const DataType effective_dtype = to_byte_aligned_dtype(t.get_dtype());
            const int32_t elem_bytes = static_cast<int32_t>(DTypeDimensionInfo::dtype_to_bytes(effective_dtype));

            // Step 2: derive element count.
            // Fall back to UINT8 if total_bytes is not evenly divisible (edge case).
            const bool exact_fit = (elem_bytes > 0) && ((total_bytes % elem_bytes) == 0);
            const DataType final_dtype = exact_fit ? effective_dtype : DataType::UINT8;
            const int32_t num_elems = exact_fit ? (total_bytes / elem_bytes) : total_bytes;
            const int32_t stride = exact_fit ? elem_bytes : 1;

            side.dtype = final_dtype;
            side.setDimension_OutermostFirst({{num_elems, stride}});

            // Step 3: propagate layout so that permutation detection (isPermute()) works
            // after round-tripping through the descriptor.  Layout::INVALID and the
            // default ZMAJOR/ZXY are both meaningful, so we always copy the value.
            // std::optional lets callers distinguish "layout was set" from "layout unknown".
            side.layout = t.get_layout();
        };

        VPUDMADescriptor desc;

        // Device and memory locations: direct copy.
        desc.device = wl.device;
        desc.src_location = wl.input_location;
        desc.dst_location = wl.output_location;

        make_side(wl.input, desc.src);   // source — 1D, dtype-preserving
        make_side(wl.output, desc.dst);  // destination — 1D, dtype-preserving

        return desc;
    }

    // ======================================================================
    //  Upward conversion: create a VPUDMADescriptor from a DMANNWorkload_NPU40_50
    // ======================================================================

    /// @brief Promotes a DMANNWorkload_NPU40_50 to a VPUDMADescriptor.
    ///
    /// The HW flat encoding (src_width / dst_width as innermost byte block + e_dim[]
    /// extra dimensions ordered innermost-first) is converted to the VPUDMATensor
    /// outermost-first shape/stride representation.
    ///
    /// dtype is always UINT8 (byte granularity matches the HW descriptor).
    /// Memory locations are derived from transfer_direction.
    ///
    /// @param wl  Source NPU40/50 workload.
    /// @return    Equivalent VPUDMADescriptor.
    static inline VPUDMADescriptor fromDMANNWorkload_NPU40_50_to_VPUDMADescriptor(const DMANNWorkload_NPU40_50& wl) {
        VPUDMADescriptor desc;
        desc.device = wl.device;

        const auto locs = DMAWorkloadTransformer::create_locations(wl.transfer_direction);
        desc.src_location = locs.first;
        desc.dst_location = locs.second;

        // Helper: fill one VPUDMATensor side from width + e_dim[] arrays.
        // The HW encoding is innermost-first (width = innermost block, e_dim[0] is next, …).
        // VPUDMATensor stores outermost-first, so we reverse the dimension list.
        auto fill_side = [](VPUDMATensor& side, int width, int num_dim, const std::function<int(int)>& get_dim_size,
                            const std::function<int(int)>& get_stride) {
            side.dtype = DataType::UINT8;  // byte granularity

            // Build innermost-first: dim 0 = width block (stride 1 byte), dims 1..num_dim from e_dim[]
            const int total_dims = 1 + num_dim;
            // store temporarily, then reverse into VPUDMATensor (outermost-first)
            std::array<int32_t, VPU_DMA_MAX_DIMS> shapes{};
            std::array<int32_t, VPU_DMA_MAX_DIMS> strides{};

            // innermost (index 0 in our temp array)
            shapes[0] = static_cast<int32_t>(width);
            strides[0] = 1;  // one byte per element

            for (int i = 0; i < num_dim; ++i) {
                shapes[i + 1] = static_cast<int32_t>(get_dim_size(i) + 1);  // 0-based -> count
                strides[i + 1] = static_cast<int32_t>(get_stride(i));
            }

            // Write to VPUDMATensor outermost-first (reverse)
            side.num_dims = total_dims;
            for (int i = 0; i < total_dims; ++i) {
                const int src_idx = total_dims - 1 - i;  // reversed
                side.shape[i] = shapes[src_idx];
                side.byte_strides[i] = strides[src_idx];
            }
            // zero-pad remaining slots
            for (int i = total_dims; i < VPU_DMA_MAX_DIMS; ++i) {
                side.shape[i] = 0;
                side.byte_strides[i] = 0;
            }
        };

        fill_side(
                desc.src, wl.src_width, wl.num_dim,
                [&](int i) {
                    return wl.e_dim[i].src_dim_size;
                },
                [&](int i) {
                    return wl.e_dim[i].src_stride;
                });

        fill_side(
                desc.dst, wl.dst_width, wl.num_dim,
                [&](int i) {
                    return wl.e_dim[i].dst_dim_size;
                },
                [&](int i) {
                    return wl.e_dim[i].dst_stride;
                });

        return desc;
    }

    // ======================================================================
    //  Downward conversion: create a DMAWorkload from a VPUDMADescriptor
    // ======================================================================

    /// @brief Returns the nearest byte-aligned dtype that preserves the type family.
    ///
    /// A type is byte-aligned when its bit width is a positive multiple of 8.
    /// The check is driven by `dtype_to_bits()` so that any future type (e.g. a
    /// hypothetical UINT3 or INT6) is caught automatically without requiring a new
    /// case in this function.
    ///
    /// **Promotion rules for non-byte-aligned types:**
    ///
    ///   | Non-aligned input       | Promoted output |
    ///   |-------------------------|-----------------|
    ///   | UINT1, UINT2, UINT4     | UINT8           |
    ///   | INT1,  INT2,  INT4      | INT8            |
    ///   | FLOAT4                  | BF8             |
    ///   | any other non-aligned † | UINT8 (fallback)|
    ///
    ///   † Covers unknown future types (e.g. UINT3, INT6, …) whose family cannot
    ///     be inferred from the enum value alone.
    ///
    /// Types that are already byte-aligned (bit width > 0 and divisible by 8) are
    /// returned unchanged.
    ///
    /// @param dtype  Source data type, possibly non-byte-aligned.
    /// @return       Byte-aligned data type with the same or wider bit width.
    static DataType to_byte_aligned_dtype(const DataType dtype) noexcept {
        const int bits = DTypeDimensionInfo::dtype_to_bits(dtype);

        // Already byte-aligned (and a known type): return unchanged.
        if (bits > 0 && (bits % 8) == 0) {
            return dtype;
        }

        // Not byte-aligned (or unknown bit width): promote to the nearest 8-bit sibling.
        // The switch below selects the family-preserving sibling for all currently known
        // sub-byte types.  Any type that falls through to `default` (e.g. a future UINT3)
        // is safely mapped to UINT8.
        switch (dtype) {
        case DataType::UINT1:
        case DataType::UINT2:
        case DataType::UINT4:
            return DataType::UINT8;
        case DataType::INT1:
        case DataType::INT2:
        case DataType::INT4:
            return DataType::INT8;
        case DataType::FLOAT4:
            return DataType::BF8;
        default:
            return DataType::UINT8;  // safe fallback for unknown/future non-aligned types
        }
    }

    /// @brief Demotes a VPUDMADescriptor to a legacy DMAWorkload.
    ///
    /// Each side (source and destination) is collapsed to a flat 1D tensor, preserving
    /// the original data type wherever possible.  The rules are:
    ///
    ///  1. **Byte-aligned types** (UINT8, INT8, BF8, HF8, FLOAT16, BFLOAT16, FLOAT32,
    ///     UINT16, INT16, INT32, …) are kept as-is.  The element count is derived as
    ///     `getAccessedBytes() / dtype_to_bytes(dtype)`.
    ///
    ///  2. **Sub-byte types** (UINT4, INT4, FLOAT4, UINT2, INT2, UINT1, INT1) are
    ///     promoted to their 8-bit sibling via `to_byte_aligned_dtype`:
    ///     - UINT1 / UINT2 / UINT4 → UINT8
    ///     - INT1  / INT2  / INT4  → INT8
    ///     - FLOAT4                → BF8
    ///     The element count then equals `getAccessedBytes()` (1 byte per element).
    ///
    ///  3. **Fallback to UINT8**: if `getAccessedBytes()` is not evenly divisible by
    ///     `dtype_to_bytes(effective_dtype)` — which can happen with strided descriptors
    ///     whose innermost block is not a whole multiple of the element size — the dtype
    ///     is further degraded to UINT8 and the element count equals `getAccessedBytes()`.
    ///
    /// **Shape layout note:**
    ///   `VPUTensor` uses shape `{X, Y, Z, B}` (indices 0–3).  The default layout is
    ///   `ZXY`, where Z (shape[2]) is the innermost dimension.  The element count is
    ///   placed in X (`shape = {num_elems, 1, 1, 1}`), which is the *outermost*
    ///   dimension under ZXY.
    ///
    ///   This is intentional and safe because:
    ///   - The resulting dtype is always byte-aligned (never sub-byte).
    ///   - For byte-aligned types `VPUTensor::size() = num_elems × elem_bytes`,
    ///     independent of which dimension holds the count.
    ///   - Sub-byte packing (packmode_0) requires elements in the innermost dimension,
    ///     but the demoted tensor never uses sub-byte types.
    ///
    /// Stride information from the descriptor is discarded; the resulting DMAWorkload
    /// models the transfer as a tightly-packed contiguous 1D buffer on each side.
    ///
    /// @param desc  VPUDMADescriptor to demote.
    /// @return      Equivalent DMAWorkload with 1D contiguous tensors whose dtype
    ///              reflects the original descriptor type (or its byte-aligned upgrade).
    static inline DMAWorkload fromVPUDMADescriptor_to_DMAWorkload(const VPUDMADescriptor& desc) {
        const auto make_tensor = [](const VPUDMATensor& t) -> VPUTensor {
            const auto num_bytes = static_cast<unsigned int>(t.getAccessedBytes());

            // Step 1: promote sub-byte types to the nearest byte-aligned sibling.
            const DataType effective_dtype = to_byte_aligned_dtype(t.dtype);
            const unsigned int elem_bytes = static_cast<unsigned int>(DTypeDimensionInfo::dtype_to_bytes(effective_dtype));

            // Step 2: derive element count.
            // Fall back to UINT8 if the byte count is not evenly divisible by elem_bytes
            // (e.g. a strided descriptor whose innermost block is not an exact multiple).
            const bool exact_fit = (elem_bytes > 0) && ((num_bytes % elem_bytes) == 0);
            const DataType final_dtype = exact_fit ? effective_dtype : DataType::UINT8;
            const unsigned int num_elems = exact_fit ? (num_bytes / elem_bytes) : num_bytes;

            // Element count placed in X (outermost under ZXY) — safe for byte-aligned types.
            // Propagate layout when present so that permutation info survives the round-trip.
            // When layout is nullopt we construct without a layout argument so that the
            // VPUTensor keeps whatever default its constructor provides.
            if (t.layout.has_value()) {
                return VPUTensor({num_elems, 1u, 1u, 1u}, final_dtype, t.layout.value());
            }
            return VPUTensor({num_elems, 1u, 1u, 1u}, final_dtype);
        };

        return DMAWorkload{
                desc.device,
                make_tensor(desc.src),  // input  — collapsed 1D, dtype-preserving
                make_tensor(desc.dst),  // output — collapsed 1D, dtype-preserving
                desc.src_location,      // input_location
                desc.dst_location,      // output_location
                1u                      // output_write_tiles
        };
    }

    // ======================================================================
    //  Downward conversion: create a DMAWorkload from a DMATransfer1D
    // ======================================================================

    /// @brief Demotes a DMATransfer1D to a legacy DMAWorkload.
    ///
    /// DMATransfer1D is a minimal 1D transfer descriptor: it carries only a device,
    /// a byte length, and a MemoryDirection.  This function lifts it to a full
    /// DMAWorkload so that the theoretical DMA cost model (which operates on
    /// DMAWorkload) can be invoked without a separately constructed workload.
    ///
    /// **Conversion rules:**
    ///   - Both input and output tensors are set to a flat 1D UINT8 tensor of
    ///     `transfer_length_bytes` elements (stride = 1 byte).  No dtype promotion
    ///     is attempted because DMATransfer1D carries no element type information.
    ///   - `MemoryDirection` is mapped to `(input_location, output_location)` via
    ///     `DMAWorkloadTransformer::create_locations()`.  If the direction is unknown
    ///     (MemoryDirection::__size or any future unmapped value) a `std::runtime_error`
    ///     is thrown.
    ///   - `output_write_tiles` defaults to 1.
    ///
    /// @param dma  Source 1D DMA transfer descriptor.
    /// @return     Equivalent DMAWorkload suitable for the theoretical cost model.
    /// @throws std::runtime_error if `dma.memory_direction` is not a mapped direction.
    static inline DMAWorkload fromDMATransfer1D_to_DMAWorkload(const DMATransfer1D& dma) {
        const auto locs = DMAWorkloadTransformer::create_locations(dma.memory_direction);
        if (locs.first == MemoryLocation::__size || locs.second == MemoryLocation::__size) {
            throw std::runtime_error("Cannot create a DMAWorkload from a DMATransfer1D: unknown memory direction");
        }

        const VPUTensor ten{{static_cast<unsigned int>(dma.transfer_length_bytes), 1u, 1u, 1u}, DataType::UINT8};
        return DMAWorkload{
                dma.device,   // device
                ten,          // input  — flat 1D UINT8
                ten,          // output — same shape (symmetric 1D transfer)
                locs.first,   // input_location
                locs.second,  // output_location
                1u            // output_write_tiles
        };
    }

    // ======================================================================
    //  Upward conversion: create a VPUDMADescriptor from a DMATransfer1D
    // ======================================================================

    /// @brief Promotes a DMATransfer1D directly to a VPUDMADescriptor.
    ///
    /// DMATransfer1D is a minimal 1D transfer descriptor carrying only a device,
    /// a byte length, and a MemoryDirection.  This function lifts it all the way
    /// to a VPUDMADescriptor so that the stride-aware cost model
    /// (DMAStridedMathCostProvider_DescIntf) can be invoked without building
    /// intermediate objects manually.
    ///
    /// **Conversion rules:**
    ///   - Both source and destination tensors are set to a flat 1D UINT8 contiguous
    ///     tensor of `transfer_length_bytes` elements (stride = 1 byte).  No dtype
    ///     promotion is attempted because DMATransfer1D carries no element type
    ///     information.
    ///   - `MemoryDirection` is mapped to `(src_location, dst_location)` via
    ///     `DMAWorkloadTransformer::create_locations()`.  If the direction is unknown
    ///     (MemoryDirection::__size or any future unmapped value) a `std::runtime_error`
    ///     is thrown.
    ///
    /// **Implementation note:**
    ///   Implemented as a two-step chain:
    ///     DMATransfer1D → DMAWorkload → VPUDMADescriptor
    ///   reusing `fromDMATransfer1D_to_DMAWorkload()` and
    ///   `fromDMAWorkload_to_VPUDMADescriptor()` so that all conversion logic
    ///   is defined in exactly one place.
    ///
    /// @param dma  Source 1D DMA transfer descriptor.
    /// @return     Equivalent VPUDMADescriptor with 1D contiguous UINT8 src and dst
    ///             tensors, suitable for the stride-aware cost model.
    /// @throws std::runtime_error if `dma.memory_direction` is not a mapped direction.
    static inline VPUDMADescriptor fromDMATransfer1D_to_VPUDMADescriptor(const DMATransfer1D& dma) {
        // Two-step chain: reuse existing single-source-of-truth conversions.
        return fromDMAWorkload_to_VPUDMADescriptor(fromDMATransfer1D_to_DMAWorkload(dma));
    }
};

// ---------------------------------------------------------------------------
// DMAWorkloadTransformer explicit specializations involving VPUDMADescriptor
// ---------------------------------------------------------------------------

/// Identity: VPUDMADescriptor -> VPUDMADescriptor
template <>
inline VPUDMADescriptor DMAWorkloadTransformer::create_workload<VPUDMADescriptor, VPUDMADescriptor>(
        const VPUDMADescriptor& from) {
    return from;
}

/// DMAWorkload -> VPUDMADescriptor
template <>
inline VPUDMADescriptor DMAWorkloadTransformer::create_workload<VPUDMADescriptor, DMAWorkload>(
        const DMAWorkload& from) {
    return DMADescriptorTransformer::fromDMAWorkload_to_VPUDMADescriptor(from);
}

}  // namespace VPUNN

#endif  // VPUNN_DMA_DESCRIPTOR_TRANSFORMERS_H
