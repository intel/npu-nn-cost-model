// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef DMA_WORKLOAD_H
#define DMA_WORKLOAD_H

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
#include <sstream>
#include <string>

#include "dma_types.h"

namespace VPUNN {

/**
 * @brief Return the available memory locations per VPU IP generation
 *
 * @param device a VPUDevice representing the VPU IP generation
 * @return std::vector<MemoryLocation>
 */
inline std::vector<MemoryLocation> memoryLocation(VPUDevice device) {
    switch (device) {
    case VPUDevice::VPU_2_0:
        return {MemoryLocation::DRAM, MemoryLocation::CMX, MemoryLocation::UPA};
    case VPUDevice::VPU_2_1:
        return {MemoryLocation::DRAM, MemoryLocation::CMX, MemoryLocation::UPA, MemoryLocation::CSRAM};
    default:
        return {MemoryLocation::DRAM, MemoryLocation::CMX};
    }
}

/**
 * @brief Check if a memory location is available for a specific VPU IP generation
 *
 * @param device a VPUDevice representing the VPU IP generation
 * @param location a memory location
 * @return true
 * @return false
 */
inline bool isMemoryLocationAvailable(VPUDevice device, MemoryLocation location) {
    auto locations = memoryLocation(device);
    return std::find(locations.begin(), locations.end(), location) != locations.end();
}

inline std::ostream& operator<<(std::ostream& stream, const VPUNN::DMAWorkload& d) {
    stream << "DMAWorkload: \n"  //
           << " device: \t" << (int)d.device << " : " << VPUDevice_ToText.at(static_cast<int>(d.device))
           << " ;\n"  //
           // inputs and outputs tensors
           << " input: \t{\n"
           << d.input << " ; size(bytes): " << d.input.size() << " } ;\n"  //
           << " output: \t{\n"
           << d.output << " ; size(bytes): " << d.output.size()
           << " } ;\n"  //

           //
           << " input_location: \t" << (int)d.input_location << " : "
           << MemoryLocation_ToText.at(static_cast<int>(d.input_location)) << " ;\n"  //
           << " output_location: \t" << (int)d.output_location << " : "
           << MemoryLocation_ToText.at(static_cast<int>(d.output_location)) << " ;\n"  //

           << " output_write_tiles: \t" << d.output_write_tiles << " ;\n"  //
           << out_terminator() << "DMAWorkload "                           // terminator
            ;
    return stream;
}

/// Transforms from one workload to another
class DMAWorkloadTransformer {
public:
    /// creates a DMANNWorkload from a DMAWorkload if possible.
    /// The DMA workload has to be a simple transfer without datatype change, layout change or size change.
    /// The resulted DMANNWorkload will have all data in one plane!
    /// NN Model was trained only for :  TASK.TYPE=1 , 2D transfer where LEN = *WIDTH = Byte transferred
    ///
    /// @returns created DMANNWorkload with all data in only one plane
    /// @throws  std::runtime_error in case the preconditions are not met for a simple DMA
    static inline DMANNWorkload_NPU27 create_workload(const DMAWorkload& dma) {
        // check if same datatype , layout and  size

        const auto& in{dma.input};
        const auto& out{dma.output};
        if ((in.size() != out.size()) ||            // not same size in bytes
            (in.get_dtype() != out.get_dtype()) ||  // not same datatype
            (in.get_layout() != out.get_layout())   // not same layout
        ) {
            throw std::runtime_error("Cannot create a DMANNWorkload_NPU27 from a DMAWorkload if size or datatype or "
                                     "layout are changing!");
        }
        // check if memory direction is representable
        const MemoryDirection memory_direction{create_direction(dma.input_location, dma.output_location)};
        if (memory_direction == MemoryDirection::__size) {
            throw std::runtime_error(
                    "Cannot create a DMANNWorkload_NPU27 from a DMAWorkload : unknown memory direction");
        }

        // safe to try representation

        const int dim_in_bytes{static_cast<int>(dma.input.size())};

        const DMANNWorkload_NPU27 equivalentWorkload{
                dma.device,    // device
                0,             // num_planes,   one plane
                dim_in_bytes,  // length , all data is put in one plane

                dim_in_bytes,  //  src_width;
                dim_in_bytes,  //  dst_width;

                0,                 // src_stride;
                0,                 // dst_stride;
                0,                 // src_plane_stride;
                0,                 // dst_plane_stride;
                memory_direction,  // transfer_direction
        };
        /* coverity[copy_instead_of_move] */
        return equivalentWorkload;  // hoping for ReturnValueOptimisation
    }

    static inline DMANNWorkload_NPU27 create_NPU27_workload(const DMAWorkload& dma) {
        return create_workload(dma);
    }

    /// you can use this function both to create DMANNWorkload_NPU40 and DMANNWorkload_NPU_RESERVED
    static inline DMANNWorkload_NPU40_RESERVED create_NPU40_RESERVED_workload(const DMAWorkload& dma) {
        // check if same datatype , layout and  size

        const auto& in{dma.input};
        const auto& out{dma.output};
        if ((in.size() != out.size()) ||            // not same size in bytes
            (in.get_dtype() != out.get_dtype()) ||  // not same datatype
            (in.get_layout() != out.get_layout())   // not same layout
        ) {
            throw std::runtime_error("Cannot create a DMANNWorkload_NPU40_RESERVED from a DMAWorkload if size or datatype or "
                                     "layout are changing!");
        }
        // check if memory direction is representable
        const MemoryDirection memory_direction{create_direction(dma.input_location, dma.output_location)};
        if (memory_direction == MemoryDirection::__size) {
            throw std::runtime_error(
                    "Cannot create a DMANNWorkload_NPU40_RESERVED from a DMAWorkload : unknown memory direction");
        }

        // safe to try representation

        const int dim_in_bytes{static_cast<int>(dma.input.size())};

        const DMANNWorkload_NPU40_RESERVED equivalentWorkload{
                dma.device,    // VPUDevice device;  ///< NPU device
                dim_in_bytes,  // int src_width;
                dim_in_bytes,  // int dst_width;
                0,             // int num_dim;
                {{{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}}},
                Num_DMA_Engine::Num_Engine_1,
                memory_direction  // MemoryDirection transfer_direction;
        };
        /* coverity[copy_instead_of_move] */
        return equivalentWorkload;  // hoping for ReturnValueOptimisation
    }

    static inline DMANNWorkload_NPU40 create_NPU40_workload(const DMAWorkload& dma) {
        return create_NPU40_RESERVED_workload(dma);
    }

public:
    using LocationKey = std::pair<MemoryLocation, MemoryLocation>;  // DRAM, CMX, CSRAM, UPA
    using DirectionMap = std::map<LocationKey, MemoryDirection>;

    static inline MemoryDirection create_direction(const MemoryLocation& from, const MemoryLocation& to) {
        static const DirectionMap dirMap{
                {{MemoryLocation::DRAM, MemoryLocation::DRAM}, MemoryDirection::DDR2DDR},  //
                {{MemoryLocation::DRAM, MemoryLocation::CMX}, MemoryDirection::DDR2CMX},   //

                {{MemoryLocation::CMX, MemoryLocation::DRAM}, MemoryDirection::CMX2DDR},  //
                {{MemoryLocation::CMX, MemoryLocation::CMX}, MemoryDirection::CMX2CMX},   //

        };

        const LocationKey key{std::make_pair(from, to)};
        const auto search = dirMap.find(key);
        if (search != dirMap.cend()) {
            return search->second;
        } else {
            return MemoryDirection::__size;  // or throw?
        }
    };
};

/// Specialization for DMANNWorkload_NPU27
template <>
class DMANNWorkloadCreator<DMANNWorkload_NPU27> {
public:
    static inline DMANNWorkload_NPU27 create_workload(const DMATransfer1D& dma) {
        // check if memory direction is representable
        if (dma.memory_direction == MemoryDirection::__size) {
            throw std::runtime_error(
                    "Cannot create a DMANNWorkload_NPU27 from a DMANNWorkload : unknown memory direction");
        }

        // safe to try representation
        const DMANNWorkload_NPU27 equivalentWorkload{
                dma.device,                 // device
                0,                          // num_planes,   one plane
                dma.transfer_length_bytes,  // length , all data is put in one plane

                dma.transfer_length_bytes,  //  src_width;
                dma.transfer_length_bytes,  //  dst_width;

                0,                     // src_stride;
                0,                     // dst_stride;
                0,                     // src_plane_stride;
                0,                     // dst_plane_stride;
                dma.memory_direction,  // transfer_direction
        };
        /* coverity[copy_instead_of_move] */
        return equivalentWorkload;  // hoping for ReturnValueOptimisation
    }
};

/// Specialization for DMANNWorkload_NPU40_RESERVED
template <>
class DMANNWorkloadCreator<DMANNWorkload_NPU40_RESERVED> {
public:
    static inline DMANNWorkload_NPU40_RESERVED create_workload(const DMATransfer1D& dma) {
        // check if memory direction is representable
        if (dma.memory_direction == MemoryDirection::__size) {
            throw std::runtime_error(
                    "Cannot create a DMANNWorkload_NPU40_RESERVED from a DMAWorkload : unknown memory direction");
        }

        // safe to try representation

        DMANNWorkload_NPU40_RESERVED equivalentWorkload{
                dma.device,                 // VPUDevice device;  ///< NPU device
                dma.transfer_length_bytes,  // int src_width;
                dma.transfer_length_bytes,  // int dst_width;
                0,                          // int num_dim;
                {{{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}}},
                Num_DMA_Engine::Num_Engine_1,
                dma.memory_direction  // MemoryDirection transfer_direction;
        };

        return equivalentWorkload;  // hoping for ReturnValueOptimisation
    }
};

inline std::ostream& operator<<(std::ostream& stream, const VPUNN::DMANNWorkload_NPU27& d) {
    stream << "\nDMANNWorkload_NPU27: \n"                                                                          //
           << "device: \t" << (int)d.device << " : " << VPUDevice_ToText.at(static_cast<int>(d.device)) << " ;\n"  //
           << "num_planes: \t" << d.num_planes << " ;\n"
           << "length: \t" << d.length << " ;\n"
           << "src_width: \t" << d.src_width << " ;\n"
           << "dst_width: \t" << d.dst_width << " ;\n"
           << "src_stride: \t" << d.src_stride << " ;\n"
           << "dst_stride: \t" << d.dst_stride << " ;\n"
           << "src_plane_stride: \t" << d.src_plane_stride << " ;\n"
           << "dst_plane_stride: \t" << d.dst_plane_stride << " ;\n"

           << "\ndirection: \t" << (int)d.transfer_direction << " : "
           << MemoryDirection_ToText.at(static_cast<int>(d.transfer_direction)) << " ;\n"

           << out_terminator() << "DMANNWorkload_NPU27 "  // terminator
            ;
    return stream;
}

inline std::ostream& operator<<(std::ostream& stream, const VPUNN::DMANNWorkload_NPU40_RESERVED& d) {
    stream << "DMANNWorkload_NPU40_RESERVED: \n"                                                                         //
           << "device: \t" << (int)d.device << " : " << VPUDevice_ToText.at(static_cast<int>(d.device)) << " ;\n"  //
           << "src_width: \t" << d.src_width << " ;\n"
           << "dst_width: \t" << d.dst_width << " ;\n"
           << "num_dim: \t" << d.num_dim << " ;\n";

    for (int i = 1; i <= static_cast<int>(d.e_dim.size()); ++i) {
        const auto& dim{d.e_dim[i - 1]};
        stream << "  src_stride_" << i << "  : \t" << dim.src_stride << " ;\n"
               << "  dst_stride_" << i << "  : \t" << dim.dst_stride << " ;\n"
               << "  src_dim_size_" << i << ": \t" << dim.src_dim_size << " ;\n"
               << "  dst_dim_size_" << i << ": \t" << dim.dst_dim_size << " ;\n";
    }

    stream << "\nNum_DMA_Engine: \t" << (int)d.num_engine << " : "
           << Num_DMA_Engine_ToText.at(static_cast<int>(d.num_engine)) << " ;"

           << "\ndirection: \t" << (int)d.transfer_direction << " : "
           << MemoryDirection_ToText.at(static_cast<int>(d.transfer_direction)) << " ;\n"

           << out_terminator() << "DMANNWorkload_NPU40_RESERVED ";  // terminator
    return stream;
}

/// @brief Convert a DMATransfer1D to a DMAWorkload to be used by DMA theoretical model.
/// @deprecated will be removed with DMAWorkload
static inline DMAWorkload convert_dma1d_2_dmawl(const DMATransfer1D& dma) {
    const VPUTensor ten{{static_cast<unsigned int>(dma.transfer_length_bytes), 1, 1, 1}, DataType::UINT8};

    MemoryLocation from{MemoryLocation::__size};
    MemoryLocation to{MemoryLocation::__size};

    switch (dma.memory_direction) {
    case MemoryDirection::DDR2DDR:
        from = MemoryLocation::DRAM;
        to = MemoryLocation::DRAM;
        break;
    case MemoryDirection::DDR2CMX:
        from = MemoryLocation::DRAM;
        to = MemoryLocation::CMX;
        break;
    case MemoryDirection::CMX2DDR:
        from = MemoryLocation::CMX;
        to = MemoryLocation::DRAM;
        break;
    case MemoryDirection::CMX2CMX:
        from = MemoryLocation::CMX;
        to = MemoryLocation::CMX;
        break;
    default:
        throw std::runtime_error("Cannot create a DMAWorkload from a DMATransfer1D : unknown memory direction");
        break;
    }

    return DMAWorkload{
            dma.device,  // device
            ten,         // input
            ten,         // output
            from,        // input_location
            to,          // output_location
    };
}

}  // namespace VPUNN

#endif  // DMA_WORKLOAD_H
