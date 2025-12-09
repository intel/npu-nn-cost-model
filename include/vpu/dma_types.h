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
#include <unordered_map>
#include <variant>
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

template <>
inline std::string enumName<MemoryLocation>() {
    return "MemoryLocation";
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

template <>
inline std::string enumName<MemoryDirection>() {
    return "MemoryDirection";
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

template <>
inline std::string enumName<Num_DMA_Engine>() {
    return "Num_DMA_Engine";
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
/* coverity[rule_of_five_violation:FALSE] */
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

    std::string loc_name{};  ///< The location name

    int getAccessedBytes() const {
        return (num_planes + 1) * length;
    }

    // -- Serialization --

    static const std::string get_wl_name() {
        return "dma_workload_npu27_";
    }

    using _ref_supported_type = std::variant<std::reference_wrapper<VPUDevice>, std::reference_wrapper<MemoryDirection>,
                                             std::reference_wrapper<int>, std::reference_wrapper<std::string>>;

    const std::unordered_map<std::string, _ref_supported_type> _member_map{
            {"device", std::ref(device)},
            {"num_planes", std::ref(num_planes)},
            {"length", std::ref(length)},
            {"src_width", std::ref(src_width)},
            {"dst_width", std::ref(dst_width)},
            {"src_stride", std::ref(src_stride)},
            {"dst_stride", std::ref(dst_stride)},
            {"src_plane_stride", std::ref(src_plane_stride)},
            {"dst_plane_stride", std::ref(dst_plane_stride)},
            {"transfer_direction", std::ref(transfer_direction)},
            {"loc_name", std::ref(loc_name)}};

    static const std::vector<std::string> _get_member_names() {
        return {"device",
                "num_planes",
                "length",
                "src_width",
                "dst_width"
                "src_stride",
                "dst_stride",
                "src_plane_stride",
                "dst_plane_stride",
                "transfer_direction",
                "loc_name"};
    }

    DMANNWorkload_NPU27() = default;
    DMANNWorkload_NPU27(VPUDevice _device, int _num_planes, int _length, int _src_width, int _dst_width,
                        int _src_stride, int _dst_stride, int _src_plane_stride, int _dst_plane_stride,
                        MemoryDirection _transfer_direction, std::string _loc_name = {}) /* noexcept(false)*/
            : device{_device},
              num_planes{_num_planes},
              length{_length},
              src_width{_src_width},
              dst_width{_dst_width},
              src_stride{_src_stride},
              dst_stride{_dst_stride},
              src_plane_stride{_src_plane_stride},
              dst_plane_stride{_dst_plane_stride},
              transfer_direction{_transfer_direction},
              loc_name{std::move(_loc_name)} {
    }

    DMANNWorkload_NPU27(const DMANNWorkload_NPU27& w) /*noexcept(false)*/
            : device{w.device},
              num_planes{w.num_planes},
              length{w.length},
              src_width{w.src_width},
              dst_width{w.dst_width},
              src_stride{w.src_stride},
              dst_stride{w.dst_stride},
              src_plane_stride{w.src_plane_stride},
              dst_plane_stride{w.dst_plane_stride},
              transfer_direction{w.transfer_direction},
              loc_name{w.loc_name} /*_member_map{}*/ {
    }

    // DMANNWorkload_NPU27(DMANNWorkload_NPU27&) noexcept(false) = delete;
    DMANNWorkload_NPU27& operator=(const DMANNWorkload_NPU27&) = delete;
    DMANNWorkload_NPU27(DMANNWorkload_NPU27&& w) /*noexcept(false)*/
            : device{w.device},
              num_planes{std::move(w.num_planes)},
              length{std::move(w.length)},
              src_width{std::move(w.src_width)},
              dst_width{std::move(w.dst_width)},
              src_stride{std::move(w.src_stride)},
              dst_stride{std::move(w.dst_stride)},
              src_plane_stride{std::move(w.src_plane_stride)},
              dst_plane_stride{std::move(w.dst_plane_stride)},
              transfer_direction{std::move(w.transfer_direction)},
              loc_name{std::move(w.loc_name)} /*_member_map{}*/ {
    }
    DMANNWorkload_NPU27& operator=(DMANNWorkload_NPU27&&) = delete;
    virtual ~DMANNWorkload_NPU27() = default;
};

/// placeholder/reserved name
/// DMA descriptor for NPU4.0++
/// 6D addressing mode
/* coverity[rule_of_five_violation:FALSE] */
struct DMANNWorkload_NPU40_50 {
    VPUDevice device;  ///< NPU device,  creation via create_DMANNWorkload_NPUXX functions ensures also proper init,
                       ///< otherwise please init explicitly

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
    std::array<SizeStride, MaxExtraDimensions> e_dim{};

    Num_DMA_Engine num_engine{Num_DMA_Engine::Num_Engine_1};  ///> number of engine used. normally 1 or 2

    MemoryDirection transfer_direction{MemoryDirection::CMX2CMX};  ///< from where to where

    std::string loc_name{};  ///< The location name

    /// How many bytes are being transferred read and written
    int getAccessedBytes() const {
        const auto unitBlock{src_width};
        int bytes_accessed{unitBlock};  // innermost dim as initialization (1D)
        for (int i = 0; i < num_dim; i++) {
            bytes_accessed = bytes_accessed * (e_dim[i].src_dim_size + 1);
        }
        return bytes_accessed;
    }

    // -- Serialization --

    static const std::string get_wl_name() {
        return "dma_workload_npu40_50_";
    }

    using _ref_supported_type = std::variant<std::reference_wrapper<VPUDevice>, std::reference_wrapper<MemoryDirection>,
                                             std::reference_wrapper<Num_DMA_Engine>, std::reference_wrapper<int>,
                                             std::reference_wrapper<std::string>>;

    const std::unordered_map<std::string, _ref_supported_type> _member_map{
            {"device", std::ref(device)},
            {"src_width", std::ref(src_width)},
            {"dst_width", std::ref(dst_width)},
            {"num_dim", std::ref(num_dim)},
            {"num_engine", std::ref(num_engine)},
            {"direction", std::ref(transfer_direction)},

            {"src_stride_1", std::ref(e_dim[0].src_stride)},
            {"dst_stride_1", std::ref(e_dim[0].dst_stride)},
            {"src_dim_size_1", std::ref(e_dim[0].src_dim_size)},
            {"dst_dim_size_1", std::ref(e_dim[0].dst_dim_size)},

            {"src_stride_2", std::ref(e_dim[1].src_stride)},
            {"dst_stride_2", std::ref(e_dim[1].dst_stride)},
            {"src_dim_size_2", std::ref(e_dim[1].src_dim_size)},
            {"dst_dim_size_2", std::ref(e_dim[1].dst_dim_size)},

            {"src_stride_3", std::ref(e_dim[2].src_stride)},
            {"dst_stride_3", std::ref(e_dim[2].dst_stride)},
            {"src_dim_size_3", std::ref(e_dim[2].src_dim_size)},
            {"dst_dim_size_3", std::ref(e_dim[2].dst_dim_size)},

            {"src_stride_4", std::ref(e_dim[3].src_stride)},
            {"dst_stride_4", std::ref(e_dim[3].dst_stride)},
            {"src_dim_size_4", std::ref(e_dim[3].src_dim_size)},
            {"dst_dim_size_4", std::ref(e_dim[3].dst_dim_size)},

            {"src_stride_5", std::ref(e_dim[4].src_stride)},
            {"dst_stride_5", std::ref(e_dim[4].dst_stride)},
            {"src_dim_size_5", std::ref(e_dim[4].src_dim_size)},
            {"dst_dim_size_5", std::ref(e_dim[4].dst_dim_size)},
            {"loc_name", std::ref(loc_name)}};

    static const std::vector<std::string> _get_member_names() {
        auto fields = std::vector<std::string>{"device",     "src_width", "dst_width", "num_dim",
                                               "num_engine", "direction", "loc_name"};

        for (int i = 0; i < MaxExtraDimensions; i++) {
            fields.push_back("src_stride_" + std::to_string(i + 1));
            fields.push_back("dst_stride_" + std::to_string(i + 1));
            fields.push_back("src_dim_size_" + std::to_string(i + 1));
            fields.push_back("dst_dim_size_" + std::to_string(i + 1));
        }

        return fields;
    }

    DMANNWorkload_NPU40_50(VPUDevice _device, int _src_width = 0, int _dst_width = 0, int _num_dim = 0,
                           std::array<SizeStride, MaxExtraDimensions> _e_dim =
                                   {{{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}}},
                           Num_DMA_Engine _num_engine = Num_DMA_Engine::Num_Engine_1,
                           MemoryDirection _transfer_direction = MemoryDirection::CMX2CMX, std::string _loc_name = {})
            : device{_device},
              src_width{_src_width},
              dst_width{_dst_width},
              num_dim{_num_dim},
              e_dim{_e_dim},
              num_engine{_num_engine},
              transfer_direction{_transfer_direction},
              loc_name{std::move(_loc_name)} {
    }

    DMANNWorkload_NPU40_50() = default;
    DMANNWorkload_NPU40_50(const DMANNWorkload_NPU40_50& w) noexcept(false)
            : device{w.device},
              src_width{w.src_width},
              dst_width{w.dst_width},
              num_dim{w.num_dim},
              e_dim{w.e_dim},
              num_engine{w.num_engine},
              transfer_direction{w.transfer_direction},
              loc_name{w.loc_name} /*_member_map{}*/ {
    }

    // DMANNWorkload_NPU40_50(DMANNWorkload_NPU40_50&) noexcept(false) = delete;
    DMANNWorkload_NPU40_50& operator=(const DMANNWorkload_NPU40_50&) = delete;

    DMANNWorkload_NPU40_50(DMANNWorkload_NPU40_50&& w) noexcept(false)
            : device{std::move(w.device)},
              src_width{std::move(w.src_width)},
              dst_width{std::move(w.dst_width)},
              num_dim{std::move(w.num_dim)},
              e_dim{std::move(w.e_dim)},
              num_engine{std::move(w.num_engine)},
              transfer_direction{std::move(w.transfer_direction)},
              loc_name{std::move(w.loc_name)} /*_member_map{}*/ {
    }
    // DMANNWorkload_NPU40_50(const DMANNWorkload_NPU40_50&&) /*noexcept(false)*/ = delete;
    DMANNWorkload_NPU40_50& operator=(DMANNWorkload_NPU40_50&&) = delete;
    virtual ~DMANNWorkload_NPU40_50() = default;
};

/// DMA descriptor aliases
using DMANNWorkload_NPU40 = DMANNWorkload_NPU40_50;
using DMANNWorkload_NPU50 = DMANNWorkload_NPU40_50;

inline const DMANNWorkload_NPU40 create_DMANNWorkload_NPU40() {
    return DMANNWorkload_NPU40{VPUDevice::VPU_4_0};
}

inline const DMANNWorkload_NPU50 create_DMANNWorkload_NPU50() {
    return DMANNWorkload_NPU50{VPUDevice::NPU_5_0};
}

inline const DMANNWorkload_NPU50 create_DMANNWorkload_NPU_RESERVED() {
    return DMANNWorkload_NPU50{VPUDevice::NPU_RESERVED};
}

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
