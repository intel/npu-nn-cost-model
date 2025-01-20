// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_DMA_TYPES_H
#define VPUNN_DMA_TYPES_H

#include <algorithm>
#include <array>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <vector>
#include "types.h"
#include "utils.h"

#include <cmath>
#include <iostream>
#include <map>
#include <sstream>  //
#include <string>

#include "dpu_defaults.h"

namespace VPUNN {

// when making a new interface version, start copying from here

/*
/// gives the EnumMap for a E enum type
/// has to be fully implemented for each type we want to cover
template <typename E>
inline const EnumMap& mapToText();

/// creates the  EnumInverseMap for a particular E enum type
/// @pre the EnumMap<E> must exists
template <typename E>
inline const EnumInverseMap& mapFromText() {
static auto m = createInverseMap(mapToText<E>());
return m;
}
*/

/**
 * @brief Memory locations
 *
 */
enum class MemoryLocation { DRAM, CMX, CSRAM, UPA, __size };
static const EnumMap MemoryLocation_ToText{
        link(MemoryLocation::DRAM, "DRAM"),
        link(MemoryLocation::CMX, "CMX"),
        link(MemoryLocation::CSRAM, "CSRAM"),
        link(MemoryLocation::UPA, "UPA"),
};
template <>
inline const EnumMap& mapToText<MemoryLocation>() {
    return MemoryLocation_ToText;
}

/**
 * @brief Memory directions DDR <> CMX.
 * Applies to DMA transfers
 *
 */
enum class MemoryDirection { DDR2CMX, CMX2CMX, CMX2DDR, DDR2DDR, __size };
static const EnumMap MemoryDirection_ToText{
        link(MemoryDirection::DDR2CMX, "DDR2CMX"),
        link(MemoryDirection::CMX2CMX, "CMX2CMX"),
        link(MemoryDirection::CMX2DDR, "CMX2DDR"),
        link(MemoryDirection::DDR2DDR, "DDR2DDR"),
};
template <>
inline const EnumMap& mapToText<MemoryDirection>() {
    return MemoryDirection_ToText;
}

/**
 *Number of DMA engine used
 */
enum class Num_DMA_Engine { Num_Engine_1, Num_Engine_2, __size };
static const EnumMap Num_DMA_Engine_ToText{
        link(Num_DMA_Engine::Num_Engine_1, "Num_Engine_1"),
        link(Num_DMA_Engine::Num_Engine_2, "Num_Engine_2"),

};
template <>
inline const EnumMap& mapToText<Num_DMA_Engine>() {
    return Num_DMA_Engine_ToText;
}

/**
 * @brief The base structure that encodes a DMA workloads
 * @deprecated Will be removed in next releases
 */
struct DMAWorkload {
    VPUDevice device;  ///< VPU device

    VPUTensor input;   ///< input tensor
    VPUTensor output;  ///<  output tensor

    MemoryLocation input_location;   ///< Input memory location
    MemoryLocation output_location;  ///<  Output memory location

    unsigned int output_write_tiles = 1;  ///< number of CMX tiles to broadcast. NOT USED!
};

/**
 * @brief The structure that encodes a DMA workloads for DMA NN
 *
 * This is particular for NPU2.7.
 */
struct DMANNWorkload_NPU27 {
    VPUDevice device;  ///< NPU device

    int num_planes;  ///< starts from 0. In 3D mode several planes of data may be moved. This field holds the number of
                     ///< planes minus 1. For 2D or 1D mode set this field to 0
    int length;      ///< Transaction length in bytes.

    int src_width;  ///> Width in bytes of data required from line of source
    int dst_width;  ///> Width in bytes of data written to destination.

    int src_stride;  ///> Distance expressed in bytes from start of one line of source data to start of next line of
                     /// source data. Value is represented in two's complement format to support negative strides.
    int dst_stride;  ///> Distance expressed in bytes from start of one line of destination data to start of next line
                     /// of destination data. Value is represented in two's complement format to support negative
                     /// strides.

    int src_plane_stride;  ///> Displacement in bytes from the start address of one source plane to the start of the
                           /// next source plane. Value is represented in two's complement format to support negative
                           /// strides.
    int dst_plane_stride;  ///> Displacement in bytes from the start address of one destination plane to the start of
                           /// the next destination plane. Value is represented in two's complement format to support
                           /// negative strides.

    MemoryDirection transfer_direction;  ///< from where to where

    int getAccessedBytes() const {
        return (num_planes + 1) * length;
    }
};

/// placeholder/reserved name
/// DMA descriptor for NPU4.0
/// 6D addressing mode
struct DMANNWorkload_NPU40 {
    VPUDevice device{VPUDevice::VPU_4_0};  ///< NPU device

    // dimension zero, innermost. Stride is considered 1(compact data) for innermost dimension!?
    int src_width{0};  ///> represent the number of linear bytes in the fastest incrementing dimension.
    int dst_width{0};  ///> represent the number of linear bytes in the fastest incrementing dimension.

    int num_dim{0};  ///> This is a 0-based number, 0 indicating 1D. Max is 5. Copy channel supports upto 1.
                     /// 0,1,2,3,4,5   = max 5 more extra dimensions,Use num_dim -1 as last valid index in extra dim
    constexpr static int MaxExtraDimensions{6 - 1};

    struct SizeStride {
        int src_stride{0};  ///> distance in bytes between (start of) 2 consecutive elements. Byte Stride between
                            /// starting
        /// addresses of dimension[N] repetitions. Can be positive, zero or negative!?
        int dst_stride{0};

        int src_dim_size{0};  ///> Number of times (-1) the dimension is repeated. How many elements (zero = 1 element)
                              /// in this dimension.  Min value is zero!
        int dst_dim_size{0};
    };
    // 5 extra dimensions
    std::array<SizeStride, MaxExtraDimensions> e_dim;

    Num_DMA_Engine num_engine{Num_DMA_Engine::Num_Engine_1};  ///> number of engine used. normally 1 or 2

    MemoryDirection transfer_direction{MemoryDirection::CMX2CMX};  ///< from where to where

    /// How many bytes are being transferred read and written
    int getAccessedBytes() const {
        const auto unitBlock{src_width};
        int bytes_accessed{unitBlock};  // innermost dim as initialization (1D)
        for (int i = 0; i < num_dim; i++) {
            bytes_accessed = bytes_accessed * (e_dim[i].src_dim_size + 1);
        }
        return bytes_accessed;
    }
};

/**
 * @brief Encodes a simple 1D DMA transfer
 *
 */
struct DMATransfer1D {
    VPUDevice device;                  ///< VPU device.
    int transfer_length_bytes;         ///< Transaction length in bytes.
    MemoryDirection memory_direction;  ///< from where to where.
};

/// Base specialization - does nothing
/// Purpose of this class is to create different types of DMANNWorkloads from a generic DMATransfer1D
/// Each specialization returns a different type
template <typename DMADesc>
class DMANNWorkloadCreator;

// when making a new interface version, Stop copying here

}  // namespace VPUNN

#endif  // VPUNN_TYPES_H
