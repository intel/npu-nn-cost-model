// Copyright © 2026 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the "Software Package")
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the "third-party-programs.txt" or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_DMA_DESCRIPTORS_H
#define VPUNN_DMA_DESCRIPTORS_H

#include <array>
#include <cstdint>  // int32_t
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>  // std::pair
#include <vector>

#include "dma_types.h"
#include "types.h"
#include "vpu/datatype_collection_size.h"
#include "vpu/dpu_dtypes_dimension_info.h"

namespace VPUNN {

/// Maximum number of logical dimensions supported in a VPUDMATensor (matches NPU4.0+ HW: 6D).
constexpr int VPU_DMA_MAX_DIMS = 6;

// ---------------------------------------------------------------------------
// VPUDMATensor — one side of a DMA transfer, compiler-friendly representation
// ---------------------------------------------------------------------------

/// Describes one side (source or destination) of a DMA transfer.
///
/// Mirrors the strided-MemRef view that VPUIP/MLIR uses: element type,
/// logical shape, and per-dimension byte strides. Dimension ordering is
/// outermost-first (NCHW convention), matching MLIR memref descriptors.
///
/// The compiler fills this from its own tensor representation and passes it
/// directly — no manual byte-level computation required.
///
/// Example — fp16 tensor 1×3×224×224 (NCHW), contiguous DDR→CMX copy:
///   dtype        = FLOAT16
///   num_dims     = 4
///   shape        = {1,      3,      224,   224}
///   byte_strides = {602112, 100352, 448,   2  }
///
struct VPUDMATensor {
    DataType dtype{DataType::UINT8};  ///< Element type. Datatype is NOT allowed to be sub-8-bit
                                      ///< (UINT4, INT4, UINT2, INT2, UINT1, INT1) or fractional bits (potential INT12).
                                      ///< The stride being represented in bytes cannot model 4 bits stride, and DMA
                                      ///< engine does not support fractional byte handling
    int num_dims{1};                  ///< Number of active dimensions (0..VPU_DMA_MAX_DIMS; 0 denotes an empty tensor).
    // make the order , what is innermost/outermost clear. Now it is outermost first.
    std::array<int32_t, VPU_DMA_MAX_DIMS> shape{1, 0, 0, 0, 0, 0};  ///< Element count per dimension (outermost first).
    std::array<int32_t, VPU_DMA_MAX_DIMS> byte_strides{
            1, 0, 0, 0, 0, 0};  ///< Byte distance between consecutive elements per dimension.

    /// Optional layout hint for this side of the DMA transfer.
    ///
    /// When set, it indicates the logical memory layout (e.g. ZXY, ZYX) of the
    /// tensor on this side.  This is used to detect CMX→CMX permutation transfers:
    /// if src.layout ≠ dst.layout and both are present, the DMA engine must
    /// scatter/gather individual elements rather than copying a contiguous byte
    /// stream.
    ///
    /// `std::nullopt` (the default) means "no layout information available".
    /// Cost models and transformation helpers treat an absent layout as if no
    /// permutation constraint exists for this side.
    std::optional<Layout> layout{std::nullopt};

    /// Sets all three dimension fields (num_dims, shape, byte_strides) atomically from a
    /// list of {shape, stride} pairs ordered outermost-first.
    ///
    /// Each element of @p dims is a pair<int32_t, int32_t> = {element_count, byte_stride}.
    /// Bundling shape and stride into a single pair per dimension makes it structurally
    /// impossible to supply mismatched counts: if you have N shape values you must supply
    /// exactly N stride values and vice versa.
    ///
    /// @param dims  Outermost-first list of (shape, byte_stride) pairs.
    ///              Must contain between 1 and VPU_DMA_MAX_DIMS entries.
    /// @throws std::invalid_argument if the number of dimensions is 0 or exceeds VPU_DMA_MAX_DIMS.
    ///
    /// Example — 1×3×224×224 FP16 tensor, contiguous NCHW layout:
    ///   t.setDimension_OutermostFirst({{1, 602112}, {3, 100352}, {224, 448}, {224, 2}});
    void setDimension_OutermostFirst(std::initializer_list<std::pair<int32_t, int32_t>> dims) {
        const int n = static_cast<int>(dims.size());
        if (n < 1 || n > VPU_DMA_MAX_DIMS) {
            throw std::invalid_argument(
                    "VPUDMATensor::setDimension_OutermostFirst: number of dimensions must be in [1, " +
                    std::to_string(VPU_DMA_MAX_DIMS) + "], got " + std::to_string(n));
        }
        num_dims = n;
        int d = 0;
        for (const auto& [s, stride] : dims) {
            shape[d] = s;
            byte_strides[d] = stride;
            ++d;
        }
        // Zero out slots beyond the active dimensions so that consumers that
        // read all VPU_DMA_MAX_DIMS entries (e.g. CSV serialization) never see
        // stale values from a previous, larger call.
        for (; d < VPU_DMA_MAX_DIMS; ++d) {
            shape[d] = 0;
            byte_strides[d] = 0;
        }
    }

    /// Sets all three dimension fields (num_dims, shape, byte_strides) atomically from a
    /// list of {shape, stride} pairs ordered innermost-first.
    ///
    /// Each element of @p dims is a pair<int32_t, int32_t> = {element_count, byte_stride}.
    /// The list is reversed internally so that the resulting shape[] and byte_strides[]
    /// arrays are always stored outermost-first, matching the rest of the API.
    ///
    /// @param dims_innermost_first  Innermost-first list of (shape, byte_stride) pairs.
    ///              Must contain between 1 and VPU_DMA_MAX_DIMS entries.
    /// @throws std::invalid_argument if the number of dimensions is 0 or exceeds VPU_DMA_MAX_DIMS.
    ///
    /// Example — 1×3×224×224 FP16 tensor, contiguous NCHW layout, supplied innermost-first:
    ///   t.setDimension_InnermostFirst({{224, 2}, {224, 448}, {3, 100352}, {1, 602112}});
    void setDimension_InnermostFirst(std::initializer_list<std::pair<int32_t, int32_t>> dims_innermost_first) {
        const int n = static_cast<int>(dims_innermost_first.size());
        if (n < 1 || n > VPU_DMA_MAX_DIMS) {
            throw std::invalid_argument(
                    "VPUDMATensor::setDimension_InnermostFirst: number of dimensions must be in [1, " +
                    std::to_string(VPU_DMA_MAX_DIMS) + "], got " + std::to_string(n));
        }
        num_dims = n;
        // Store in reverse order so that index 0 is outermost and index n-1 is innermost,
        // matching the outermost-first convention used throughout the rest of the struct.
        int d = n - 1;
        for (const auto& [s, stride] : dims_innermost_first) {
            shape[d] = s;
            byte_strides[d] = stride;
            --d;
        }
        // Zero out slots beyond the active dimensions so that consumers that
        // read all VPU_DMA_MAX_DIMS entries (e.g. CSV serialization) never see
        // stale values from a previous, larger call.
        for (int i = n; i < VPU_DMA_MAX_DIMS; ++i) {
            shape[i] = 0;
            byte_strides[i] = 0;
        }
    }

    /// Total bytes accessed (element size × product of all dimension sizes).
    /// Strides are irrelevant for this count — only logical element counts matter.
    int32_t getAccessedBytes() const {
        if (num_dims <= 0) {
            return 0;
        }
        int32_t bytes = static_cast<int32_t>(compute_size_in_bytes(shape[num_dims - 1], dtype));
        for (int d = num_dims - 2; d >= 0; --d) {
            bytes *= shape[d];
        }
        return bytes;
    }

    /// How many contiguous bytes are accessible in this tensor, considering all dimension strides.
    ///
    /// Dimensions are ordered outermost-first; the algorithm starts from the innermost
    /// dimension and walks outward, using a single uniform rule for every dimension:
    ///
    ///   seed = dtype_to_bytes(dtype)   — one element, precondition is that dtype is byte aligned, no int4 for example
    ///
    ///   For each dimension d, innermost first (d = num_dims-1 down to 0):
    ///     • shape[d] == 0 → empty tensor, return 0.
    ///     • shape[d] == 1 → degenerate (stride never used), skip without breaking chain.
    ///     • byte_strides[d] == contiguous_bytes → dimension is packed, multiply chain.
    ///     • otherwise       → gap found, stop.
    ///
    /// This treats the innermost dimension no differently from outer ones: elements are
    /// back-to-back when stride equals the dtype byte size, which is exactly the initial
    /// value of contiguous_bytes.
    ///
    /// Analogous to DMANNWorkload_NPU40_50::getContiguousBytesSrc().
    int32_t getContiguousBytes() const {
        if (num_dims <= 0) {
            return 0;
        }
        // dtype is always byte aligned
        const int32_t elem_bytes = static_cast<int32_t>(DTypeDimensionInfo::dtype_to_bytes(dtype));
        if (elem_bytes <= 0) {
            return 0;  // invalid or unsupported dtype => no well-defined contiguous byte size.
        }
        int32_t contiguous_bytes = elem_bytes;  // seed: one element

        for (int d = num_dims - 1; d >= 0; --d) {
            if (shape[d] == 0) {
                return 0;  // empty dimension — no elements at all.
            }
            if (shape[d] == 1) {
                continue;  // degenerate: stride is never used, skip without breaking chain.
            }
            if (byte_strides[d] != contiguous_bytes) {
                break;  // gap or broadcast — stop here.
            }
            contiguous_bytes *= shape[d];
        }

        return contiguous_bytes;
    }

    /// Number of contiguous memory chunks required to cover the whole tensor.
    ///
    /// Returns 1 when the tensor is fully contiguous in memory.
    /// Values > 1 indicate strided (non-contiguous) access; each chunk is
    /// `getContiguousBytes()` bytes long.
    ///
    /// Analogous to DMANNWorkload_NPU40_50::getNumContiguousChunksSrc().
    int32_t getNumContiguousChunks() const {
        const int32_t total_bytes = getAccessedBytes();
        const int32_t contiguous_bytes = getContiguousBytes();
        if (contiguous_bytes == 0) {
            return 0;  // Avoid division by zero.
        }
        return (total_bytes + contiguous_bytes - 1) / contiguous_bytes;  // Ceiling division.
    }

    /// Physical memory span of the tensor in bytes, including all stride-induced gaps.
    ///
    /// The innermost dimension contributes its full stride slot for every element
    /// (trailing gap after the last element is included), while each outer dimension
    /// contributes only the (shape-1) steps actually traversed:
    ///
    ///   footprint = shape[innermost] × |stride[innermost]|          (if shape[innermost] > 1)
    ///             + Σ_{d < innermost} ( (shape[d] - 1) × |stride[d]| )
    ///
    /// When shape[innermost] == 1 the innermost term degenerates to elem_bytes
    /// (no stride step is taken, so only one element occupies memory).
    ///
    /// Note: this differs from a pure "span-to-last-element + elem_bytes" formula
    /// because the innermost stride slot (which may include a trailing gap) is always
    /// reserved by the hardware descriptor, not just the element size.
    ///
    /// Key properties:
    ///  - For a packed (no-gap) tensor: equals getAccessedBytes().
    ///  - For a broadcast dimension (byte_strides[d] == 0):
    ///      - Innermost dim: contributes elem_bytes, because one physical element is always
    ///        present regardless of how many logical indices alias it.
    ///      - Outer dims: contributes 0, because no stride step is ever taken outward.
    ///  - For a degenerate dimension (shape[d] == 1): contributes 0 for outer dims,
    ///    or elem_bytes for the innermost dim (no stride step is ever taken).
    ///  - Negative strides are handled via absolute value, because the tensor simply
    ///    walks backward in memory with the same physical span.
    ///  - Returns 0 when any dimension has shape == 0 (empty tensor).
    int32_t getMemoryFootprintBytes() const {
        if (num_dims <= 0) {
            return 0;
        }
        const int32_t elem_bytes = static_cast<int32_t>(DTypeDimensionInfo::dtype_to_bytes(dtype));
        if (elem_bytes <= 0) {
            return 0;
        }

        // Any empty dimension means the tensor has no elements and no memory footprint.
        for (int d = num_dims - 1; d >= 0; --d) {
            if (shape[d] == 0) {
                return 0;
            }
        }

        // Walk from innermost (num_dims-1) to outermost (0), matching the style of
        // getContiguousBytes().
        //
        // Innermost dimension (d == num_dims-1):
        //   shape > 1  →  shape × |stride|   (trailing gap after the last element is included,
        //                                      because the stride slot is always reserved)
        //   shape <= 1 →  shape × elem_bytes  (0 or 1 element: no stride step is ever taken)
        //
        // Every outer dimension (d < num_dims-1):
        //   Only (shape[d] - 1) stride steps are ever taken outward; the last repetition ends
        //   at its start address + the already-accumulated inner footprint, so no extra trailing
        //   stride is added for outer dims.
        //   Contribution: (shape[d] - 1) × |stride[d]|
        //
        // Negative strides are handled via absolute value: backward traversal covers the same
        // physical span as forward traversal.

        int32_t footprint = 0;

        for (int d = num_dims - 1; d >= 0; --d) {
            const int32_t abs_stride = (byte_strides[d] >= 0) ? byte_strides[d] : -byte_strides[d];

            if (d == num_dims - 1) {
                // Innermost dimension.
                // abs_stride == 0 means broadcast: all shape[d] logical elements alias the
                // same physical location, so exactly one element is occupied in memory.
                if (abs_stride == 0) {
                    footprint += elem_bytes;
                } else {
                    footprint += (shape[d] > 1) ? (shape[d] * abs_stride) : (shape[d] * elem_bytes);
                }
            } else {
                // Outer dimension: only (shape - 1) steps are traversed.
                footprint += (shape[d] - 1) * abs_stride;
            }
        }

        return footprint;
    }
};

inline std::ostream& operator<<(std::ostream& stream, const VPUDMATensor& d) {
    stream << "VPUDMATensor: \n"                                                                      //
           << " dtype: \t" << (int)d.dtype << " : " << DataType_ToText.at(static_cast<int>(d.dtype))  //
           << " ;\n"                                                                                  //
           << " num_dims: \t" << d.num_dims << " ;\n"                                                 //
           << " shape (outermost first): \t[";
    for (int i = 0; i < d.num_dims; ++i) {
        stream << d.shape[i];
        if (i + 1 < d.num_dims)
            stream << ", ";
    }
    stream << "] ;\n"
           << " byte_strides (outermost first): \t[";
    for (int i = 0; i < d.num_dims; ++i) {
        stream << d.byte_strides[i];
        if (i + 1 < d.num_dims)
            stream << ", ";
    }
    stream << "] ;\n";
    if (d.layout.has_value()) {
        stream << " layout: \t" << (int)d.layout.value() << " : "
               << Layout_ToText.at(static_cast<int>(d.layout.value())) << " ;\n";
    } else {
        stream << " layout: \tnullopt ;\n";
    }
    stream << " accessed_bytes: \t" << d.getAccessedBytes() << " ;\n"                 //
           << " contiguous_bytes: \t" << d.getContiguousBytes() << " ;\n"             //
           << " num_contiguous_chunks: \t" << d.getNumContiguousChunks() << " ;\n"    //
           << " memory_footprint_bytes: \t" << d.getMemoryFootprintBytes() << " ;\n"  //
           << out_terminator() << "VPUDMATensor ";                                    // terminator
    return stream;
}
// ---------------------------------------------------------------------------
// VPUDMADescriptor — VPUx-facing DMA workload description
// ---------------------------------------------------------------------------

/// Describes a DMA transfer for cost model queries at the VPUx compiler level.
///
/// The compiler provides source and destination tensors as VPUDMATensor
/// (dtype + shape + byte strides, outermost-first), memory locations, and
/// device. The cost model internally derives the HW-level physical descriptor
/// (stride collapsing, 0-based dimension encoding) without exposing that
/// complexity to the caller.
/// Datatype for src and dst tensors is NOT allowed to be sub-8-bit (UINT4, INT4, UINT2, INT2, UINT1, INT1) or
/// fractional bits (potential INT12). The stride being represented in bytes cannot model 4 bits stride, and DMA engine
/// does not support fractional byte handling.
/// Source and destination byte counts are typically equal, but can differ for specialized transfers (e.g. DDR→CMX
/// decompression).
///
///
struct VPUDMADescriptor {
    VPUDevice device{VPUDevice::__size};  ///< NPU device generation.

    VPUDMATensor src;  ///< Source tensor: element type, shape, and byte strides.
    VPUDMATensor dst;  ///< Destination tensor: element type, shape, and byte strides.

    MemoryLocation src_location{MemoryLocation::CMX};  ///< Source memory space.
    MemoryLocation dst_location{MemoryLocation::CMX};  ///< Destination memory space.

    // ======================================================================
    //  Convenience queries
    // ======================================================================

    /// Transfer direction derived from memory locations.
    MemoryDirection getDirection() const {
        if (src_location == MemoryLocation::DRAM && dst_location == MemoryLocation::CMX) {
            return MemoryDirection::DDR2CMX;
        }
        if (src_location == MemoryLocation::CMX && dst_location == MemoryLocation::DRAM) {
            return MemoryDirection::CMX2DDR;
        }
        if (src_location == MemoryLocation::CMX && dst_location == MemoryLocation::CMX) {
            return MemoryDirection::CMX2CMX;
        }
        if (src_location == MemoryLocation::DRAM && dst_location == MemoryLocation::DRAM) {
            return MemoryDirection::DDR2DDR;
        }
        return MemoryDirection::__size;
    }

    /// Total bytes accessed from the source side.
    int32_t getTransferBytes() const {
        return src.getAccessedBytes();
    }

    /// Total bytes read from the source tensor (logical element count × element size; strides ignored).
    int32_t getSrcTransferBytes() const {
        return src.getAccessedBytes();
    }

    /// Alias for getSrcTransferBytes(); matches the DMANNWorkload_NPU40_50 interface.
    int32_t getReadBytes() const {
        return src.getAccessedBytes();
    }

    /// Total bytes written to the destination tensor (logical element count × element size; strides ignored).
    int32_t getDstTransferBytes() const {
        return dst.getAccessedBytes();
    }

    /// Alias for getDstTransferBytes(); matches the DMANNWorkload_NPU40_50 interface.
    int32_t getWrittenBytes() const {
        return dst.getAccessedBytes();
    }

    /// Number of contiguous memory chunks required to read the source tensor.
    /// Returns 1 for a fully packed source; values > 1 indicate strided access.
    /// Each chunk is `src.getContiguousBytes()` bytes long.
    int32_t getNumContiguousChunksSrc() const {
        return src.getNumContiguousChunks();
    }

    /// Number of contiguous memory chunks required to write the destination tensor.
    /// Returns 1 for a fully packed destination; values > 1 indicate strided access.
    /// Each chunk is `dst.getContiguousBytes()` bytes long.
    int32_t getNumContiguousChunksDst() const {
        return dst.getNumContiguousChunks();
    }

    /// Size in bytes of one contiguous memory chunk on the source side.
    /// Equals the total transfer bytes when the source is fully packed.
    int32_t getContiguousBytesSrc() const {
        return src.getContiguousBytes();
    }

    /// Size in bytes of one contiguous memory chunk on the destination side.
    /// Equals the total transfer bytes when the destination is fully packed.
    int32_t getContiguousBytesDst() const {
        return dst.getContiguousBytes();
    }

    // ======================================================================
    //  Descriptor validation
    // ======================================================================

public:
    /// @brief Checks the sanity of this descriptor and throws a meaningful exception if a precondition is violated.
    ///
    /// Preconditions verified:
    ///  1. Source and destination data types must be byte-aligned (bit width > 0 and divisible by 8).
    ///     Any type whose bit width is not a positive multiple of 8 — including sub-byte types
    ///     (INT4, UINT4, FLOAT4, INT2, UINT2, INT1, UINT1), hypothetical non-octet types (e.g. 12-bit),
    ///     and unknown types — is rejected, because DMA strides are expressed in whole bytes and
    ///     cannot model bit-level addressing.
    ///  2. (Currently disabled) The number of bytes read (src) and written (dst) must be equal.
    ///     A mismatch indicates an inconsistent descriptor that cannot be modelled.
    ///     Commented out to allow asymmetric transfers (e.g. decompression) while this is under review.
    ///  3. The device must be a known, valid VPUDevice value (not VPUDevice::__size and within range).
    ///  4. src.num_dims and dst.num_dims must each be in [0, VPU_DMA_MAX_DIMS].
    ///     Out-of-range values cause out-of-bounds array accesses in the byte-count query methods.
    ///  5. Every active dimension (d in [0, num_dims)) must have shape[d] >= 0.
    ///     Negative shapes silently produce negative byte counts that corrupt cost-model results.
    ///  6. The (src_location, dst_location) pair must map to a supported MemoryDirection
    ///     (DDR->CMX, CMX->DDR, CMX->CMX, DDR->DDR). An unsupported combination causes silent
    ///     zero-cost results in fromStridedTensors_toDMANNWorkload_NPU40_50().
    ///
    /// @throws std::invalid_argument when any precondition is violated.
    void checkDescriptorSanity() const {
        // --- 1. Byte-aligned data types ---
        // A type is byte-aligned when its bit width is a positive multiple of 8.
        // DTypeDimensionInfo::dtype_to_bits() returns -1 for unknown/unsupported types, which also fails the check.
        const int src_bits = DTypeDimensionInfo::dtype_to_bits(src.dtype);
        if (src_bits <= 0 || (src_bits % 8) != 0) {
            const std::string dtype_name = DataType_ToText.at(static_cast<int>(src.dtype));
            throw std::invalid_argument("VPUDMADescriptor: source data type '" + dtype_name + "' has " +
                                        (src_bits <= 0 ? "an unknown" : std::to_string(src_bits)) +
                                        " bit width, which is not a byte multiple."
                                        " DMA strides are byte-granular; only byte-aligned types are supported.");
        }
        const int dst_bits = DTypeDimensionInfo::dtype_to_bits(dst.dtype);
        if (dst_bits <= 0 || (dst_bits % 8) != 0) {
            const std::string dtype_name = DataType_ToText.at(static_cast<int>(dst.dtype));
            throw std::invalid_argument("VPUDMADescriptor: destination data type '" + dtype_name + "' has " +
                                        (dst_bits <= 0 ? "an unknown" : std::to_string(dst_bits)) +
                                        " bit width, which is not a byte multiple."
                                        " DMA strides are byte-granular; only byte-aligned types are supported.");
        }

        // --- 2. Bytes read must equal bytes written ---
        // Temporarily commented out: this check may be too strict for descriptors that model
        // decompression or other asymmetric transfers where src and dst byte counts legitimately differ.
        // const int32_t src_bytes = src.getAccessedBytes();
        // const int32_t dst_bytes = dst.getAccessedBytes();
        // if (src_bytes != dst_bytes) {
        //     throw std::invalid_argument("VPUDMADescriptor: bytes read (" + std::to_string(src_bytes) +
        //                                 ") and bytes written (" + std::to_string(dst_bytes) +
        //                                 ") must be equal. A DMA transfer cannot change the total amount of data.");
        // }

        // --- 3. Valid device ---
        const int device_idx = static_cast<int>(device);
        const int device_size = static_cast<int>(VPUDevice::__size);
        if (device_idx < 0 || device_idx >= device_size) {
            throw std::invalid_argument("VPUDMADescriptor: device index (" + std::to_string(device_idx) +
                                        ") is out of range [0, " + std::to_string(device_size - 1) +
                                        "]. Use a valid VPUDevice enumerator, not VPUDevice::__size.");
        }

        // --- 4. Valid num_dims for src and dst ---
        // num_dims drives array indexing into shape[] and byte_strides[], both of which have
        // VPU_DMA_MAX_DIMS entries.  Values outside [0, VPU_DMA_MAX_DIMS] cause out-of-bounds
        // array accesses in getAccessedBytes(), getContiguousBytes(), and getMemoryFootprintBytes().
        if (src.num_dims < 0 || src.num_dims > VPU_DMA_MAX_DIMS) {
            throw std::invalid_argument("VPUDMADescriptor: src.num_dims (" + std::to_string(src.num_dims) +
                                        ") is out of range [0, " + std::to_string(VPU_DMA_MAX_DIMS) +
                                        "]. Values outside this range cause out-of-bounds array accesses.");
        }
        if (dst.num_dims < 0 || dst.num_dims > VPU_DMA_MAX_DIMS) {
            throw std::invalid_argument("VPUDMADescriptor: dst.num_dims (" + std::to_string(dst.num_dims) +
                                        ") is out of range [0, " + std::to_string(VPU_DMA_MAX_DIMS) +
                                        "]. Values outside this range cause out-of-bounds array accesses.");
        }

        // --- 5. Non-negative shapes for all active dimensions ---
        // Negative shapes silently produce negative byte counts in getAccessedBytes() and
        // getMemoryFootprintBytes(), corrupting downstream cost-model results.
        for (int d = 0; d < src.num_dims; ++d) {
            if (src.shape[d] < 0) {
                throw std::invalid_argument("VPUDMADescriptor: src.shape[" + std::to_string(d) +
                                            "] = " + std::to_string(src.shape[d]) +
                                            " is negative. All active dimension shapes must be >= 0.");
            }
        }
        for (int d = 0; d < dst.num_dims; ++d) {
            if (dst.shape[d] < 0) {
                throw std::invalid_argument("VPUDMADescriptor: dst.shape[" + std::to_string(d) +
                                            "] = " + std::to_string(dst.shape[d]) +
                                            " is negative. All active dimension shapes must be >= 0.");
            }
        }

        // --- 6. Supported memory-location combination ---
        // getDirection() returns MemoryDirection::__size when the (src_location, dst_location)
        // pair is not one of the four recognised combinations (DDR->CMX, CMX->DDR, CMX->CMX,
        // DDR->DDR).  Downstream code in fromStridedTensors_toDMANNWorkload_NPU40_50() silently
        // produces a zero-cost result for any unrecognised direction.
        if (getDirection() == MemoryDirection::__size) {
            throw std::invalid_argument("VPUDMADescriptor: the memory-location combination (src_location=" +
                                        std::to_string(static_cast<int>(src_location)) +
                                        ", dst_location=" + std::to_string(static_cast<int>(dst_location)) +
                                        ") does not map to a supported MemoryDirection."
                                        " Supported combinations: DRAM->CMX, CMX->DRAM, CMX->CMX, DRAM->DRAM.");
        }
    }

};

inline std::ostream& operator<<(std::ostream& stream, const VPUDMADescriptor& d) {
    const auto dir = d.getDirection();
    const std::string dir_str = (static_cast<int>(dir) < static_cast<int>(MemoryDirection::__size))
                                        ? MemoryDirection_ToText.at(static_cast<int>(dir))
                                        : "UNKNOWN";
    stream << "VPUDMADescriptor: \n"                                                                      //
           << " device: \t" << (int)d.device << " : " << VPUDevice_ToText.at(static_cast<int>(d.device))  //
           << " ;\n"                                                                                      //
           << " src_location: \t" << (int)d.src_location << " : "
           << MemoryLocation_ToText.at(static_cast<int>(d.src_location)) << " ;\n"  //
           << " dst_location: \t" << (int)d.dst_location << " : "
           << MemoryLocation_ToText.at(static_cast<int>(d.dst_location)) << " ;\n"  //
           << " direction: \t" << (int)dir << " : " << dir_str << " ;\n"            //
           << " src: \t{\n"
           << d.src << "} ;\n"  //
           << " dst: \t{\n"
           << d.dst << "} ;\n"                                                     //
           << " transfer_bytes(src): \t" << d.getSrcTransferBytes() << " ;\n"      //
           << " transfer_bytes(dst): \t" << d.getDstTransferBytes() << " ;\n"      //
           << " contiguous_bytes(src): \t" << d.getContiguousBytesSrc() << " ;\n"  //
           << " contiguous_bytes(dst): \t" << d.getContiguousBytesDst() << " ;\n"  //
           << " num_chunks(src): \t" << d.getNumContiguousChunksSrc() << " ;\n"    //
           << " num_chunks(dst): \t" << d.getNumContiguousChunksDst() << " ;\n"    //
           << out_terminator() << "VPUDMADescriptor ";                             // terminator
    return stream;
}

}  // namespace VPUNN

#endif  // VPUNN_DMA_DESCRIPTORS_H
