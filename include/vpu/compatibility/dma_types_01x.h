// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_DMA_TYPES_01X_H
#define VPUNN_DMA_TYPES_01X_H

#include <algorithm>
#include <array>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <vector>
#include "../dma_types.h"
#include "../utils.h"
#include "inference/dma_preprocessing.h"

#include <cmath>
#include <iostream>
#include <map>
#include <sstream>  //
#include <string>

namespace VPUNN {

/** @brief interface for original inputs, named 01. This is a convention on what to contain the VPUNN's input descriptor
 * in this namespace all the types will be stored exactly as they are required by this interface
 */
namespace intf_dma_01x {

// when making a new interface version, start copying from here

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

// when making a new interface version, Stop copying here

/**
 * @brief converts the present day interface value to the value corresponding to this interface version
 * requires the existence of mapToText and mapFromText services for the subjected enums
 * @throws out_of_range if the conversion is not possible
 */
template <typename CompatibleEnum, typename PresentEnum>
CompatibleEnum convert(PresentEnum present_day_value_type) {
    // return static_cast<VPUNN::intf_00::VPUDevice>(present_day_value_type);

    // search the text of the passed value, according to present day enum
    std::string text_val{""};
    try {
        text_val = VPUNN::mapToText<PresentEnum>().at(static_cast<int>(present_day_value_type));  // new to txt
    } catch (const std::exception& e) {
        std::stringstream buffer;
        buffer << "[ERROR]:could not map enum value: " << static_cast<int>(present_day_value_type)
               << " to enum text!. Value unmapped in the following defined map:" << VPUNN::mapToText<PresentEnum>()
               << " Original exception: " << e.what() << " File: " << __FILE__ << " Line: " << __LINE__;
        std::string details = buffer.str();
        throw std::out_of_range(details);
    }

    int old_id = -1;
    try {
        // old_id = VPUNN::intf_00::VPUDevice_FromText.at(text_val);//txt to old
        old_id = mapFromText<CompatibleEnum>().at(text_val);
    } catch (const std::exception& e) {
        std::stringstream buffer;
        buffer << "[ERROR]:could not map enum from text: " << text_val
               << " to an enum value in target required interface. This value might not be supported.\n Value unmapped "
                  "in the defined map:"
               << mapFromText<CompatibleEnum>() << " Initial exception: " << e.what() << " File: " << __FILE__
               << " Line: " << __LINE__;
        std::string details = buffer.str();
        throw std::out_of_range(details);
    }

    return static_cast<CompatibleEnum>(old_id);
}

}  // namespace intf_dma_01x

/**
 * @brief For interface of NPU2.7 . Descriptor and interface datatype are dedicated
 */
template <typename T>
class Preprocessing_Interface01_DMA :
        public PreprocessingInserterDMA<T, Preprocessing_Interface01_DMA<T>, DMANNWorkload_NPU27> {
private:
protected:
    friend class PreprocessingInserterDMA<T, Preprocessing_Interface01_DMA<T>, DMANNWorkload_NPU27>;

    /**
     * @brief Transform a DPUWorkload into a DPUWorkload descriptor
     * Here the concrete descriptor is created/populated according to established convention/interface
     *
     * @param workload a DPUWorkload
     * @param debug_offset [out] is the offset where a new value can be written. interpreted as how many positions were
     * written
     * @tparam only_simulate, if true then no data is actually written, only the offset is computed
     * @return std::vector<T>& a DPUWorkload descriptor
     */
    template <bool only_simulate>
    const std::vector<T>& transformOnly(const DMANNWorkload_NPU27& workload, size_t& debug_offset) {
        // Build the vector from the inputs
        size_t offset = 0;
        // offset = this->insert<only_simulate>(workload.device, offset);
        offset = this->template insert<only_simulate>(workload.num_planes, offset);
        offset = this->template insert<only_simulate>(workload.length, offset);

        offset = this->template insert<only_simulate>(workload.src_width, offset);
        offset = this->template insert<only_simulate>(workload.dst_width, offset);
        offset = this->template insert<only_simulate>(workload.src_stride, offset);
        offset = this->template insert<only_simulate>(workload.dst_stride, offset);

        offset = this->template insert<only_simulate>(workload.src_plane_stride, offset);
        offset = this->template insert<only_simulate>(workload.dst_plane_stride, offset);

        offset = this->template insert<only_simulate>(
                intf_dma_01x::convert<intf_dma_01x::MemoryDirection>(workload.transfer_direction), offset);

        debug_offset = offset;

        // Return the output as a pointer to the data
        return this->processed_output;
    }

    static inline constexpr size_t size_of_descriptor{12};  ///< how big the descriptor is, fixed at constructor

public:
    /// @brief the descriptor interface that this type was designed to fill/comply with
    static int getInterfaceVersion() {
        return static_cast<std::underlying_type_t<NNVersionsDMA>>(NNVersionsDMA::VERSION_01_27);
    }

    ///@brief Ctor , inits the content with expected size
    Preprocessing_Interface01_DMA() {
        this->set_size(size_of_descriptor);
    };

    ///@brief default virtual destructor
    virtual ~Preprocessing_Interface01_DMA() = default;
};

/**
 * @brief For interface of NPU4.0+ . Descriptor and interface datatype are dedicated
 */
template <typename T>
class Preprocessing_Interface02_DMA :
        public PreprocessingInserterDMA<T, Preprocessing_Interface02_DMA<T>, DMANNWorkload_NPU40_RESERVED> {
private:
protected:
    friend class PreprocessingInserterDMA<T, Preprocessing_Interface02_DMA<T>, DMANNWorkload_NPU40_RESERVED>;

    /**
     * @brief Transform a DPUWorkload into a DPUWorkload descriptor
     * Here the concrete descriptor is created/populated according to established convention/interface
     *
     * @param workload a DPUWorkload
     * @param debug_offset [out] is the offset where a new value can be written. interpreted as how many positions were
     * written
     * @tparam only_simulate, if true then no data is actually written, only the offset is computed
     * @return std::vector<T>& a DPUWorkload descriptor
     */
    template <bool only_simulate>
    const std::vector<T>& transformOnly(const DMANNWorkload_NPU40_RESERVED& workload, size_t& debug_offset) {
        // Build the vector from the inputs
        size_t offset = 0;

        offset = this->template insert<only_simulate>(workload.dst_width, offset);
        offset = this->template insert<only_simulate>(workload.src_width, offset);

        offset = this->template insert<only_simulate>(workload.num_dim, offset);

        // strides for dims 1,2,3,4,5
        for (const auto& d : workload.e_dim) {
            offset = this->template insert<only_simulate>(d.dst_stride, offset);
            offset = this->template insert<only_simulate>(d.src_stride, offset);

            offset = this->template insert<only_simulate>(d.dst_dim_size, offset);
            offset = this->template insert<only_simulate>(d.src_dim_size, offset);
        }

        offset = this->template insert<only_simulate>(
                intf_dma_01x::convert<intf_dma_01x::Num_DMA_Engine>(workload.num_engine),
                offset);  // enum 2
        offset = this->template insert<only_simulate>(
                intf_dma_01x::convert<intf_dma_01x::MemoryDirection>(workload.transfer_direction), offset);  // enum 4

        debug_offset = offset;

        // Return the output as a pointer to the data
        return this->processed_output;
    }

    static inline constexpr size_t size_of_descriptor{29};  ///< how big the descriptor is, fixed at constructor

public:
    /// @brief the descriptor interface that this type was designed to fill/comply with
    static int getInterfaceVersion() {
        return static_cast<std::underlying_type_t<NNVersionsDMA>>(NNVersionsDMA::VERSION_02_40);
    }

    ///@brief Ctor , inits the content with expected size
    Preprocessing_Interface02_DMA() {
        this->set_size(size_of_descriptor);
    };

    ///@brief default virtual destructor
    virtual ~Preprocessing_Interface02_DMA() = default;
};

/**
 * @brief For interface of NPU4.0+ . Descriptor and interface datatype are dedicated
 */
template <typename T>
class Preprocessing_Interface03_DMA :
        public PreprocessingInserterDMA<T, Preprocessing_Interface03_DMA<T>, DMANNWorkload_NPU40_RESERVED> {
private:
protected:
    friend class PreprocessingInserterDMA<T, Preprocessing_Interface03_DMA<T>, DMANNWorkload_NPU40_RESERVED>;

    /**
     * @brief Transform a DPUWorkload into a DPUWorkload descriptor
     * Here the concrete descriptor is created/populated according to established convention/interface
     * "config_num_dim",
     * "width_cfg_dst",
     * "width_cfg_src",
     * "dim_size_1_dst",
     * "dim_size_2_dst",
     * "dim_size_dst_3",
     * "dim_size_dst_4",
     * "dim_size_dst_5",
     * "dim_size_1_src",
     * "dim_size_2_src",
     * "dim_size_src_3",
     * "dim_size_src_4",
     * "dim_size_src_5",
     * "stride_dst_1",
     * "stride_dst_2",
     * "stride_dst_3",
     * "stride_dst_4",
     * "stride_dst_5",
     * "stride_src_1",
     * "stride_src_2",
     * "stride_src_3",
     * "stride_src_4",
     * "stride_src_5",
     * "direction_Direction.DDR2CMX",
     * "direction_Direction.CMX2CMX",
     * "direction_Direction.CMX2DDR",
     * "direction_Direction.DDR2DDR"
     *
     * @param workload a DPUWorkload
     * @param debug_offset [out] is the offset where a new value can be written. interpreted as how many positions were
     * written
     * @tparam only_simulate, if true then no data is actually written, only the offset is computed
     * @return std::vector<T>& a DPUWorkload descriptor
     */
    template <bool only_simulate>
    const std::vector<T>& transformOnly(const DMANNWorkload_NPU40_RESERVED& workload, size_t& debug_offset) {
        // Build the vector from the inputs
        size_t offset = 0;

        offset = this->template insert<only_simulate>(workload.num_dim, offset);

        offset = this->template insert<only_simulate>(workload.dst_width, offset);
        offset = this->template insert<only_simulate>(workload.src_width, offset);

        // sizes dst for dims 1,2,3,4,5
        for (const auto& d : workload.e_dim) {
            offset = this->template insert<only_simulate>(d.dst_dim_size, offset);
        }
        // sizes src for dims 1,2,3,4,5
        for (const auto& d : workload.e_dim) {
            offset = this->template insert<only_simulate>(d.src_dim_size, offset);
        }

        // strides dst for dims 1,2,3,4,5
        for (const auto& d : workload.e_dim) {
            offset = this->template insert<only_simulate>(d.dst_stride, offset);
        }
        // strides src for dims 1,2,3,4,5
        for (const auto& d : workload.e_dim) {
            offset = this->template insert<only_simulate>(d.src_stride, offset);
        }

        offset = this->template insert<only_simulate>(
                intf_dma_01x::convert<intf_dma_01x::MemoryDirection>(workload.transfer_direction), offset);  // enum 4

        debug_offset = offset;

        // Return the output as a pointer to the data
        return this->processed_output;
    }

    static inline constexpr size_t size_of_descriptor{27};  ///< how big the descriptor is, fixed at constructor

public:
    /// @brief the descriptor interface that this type was designed to fill/comply with
    static int getInterfaceVersion() {
        return static_cast<std::underlying_type_t<NNVersionsDMA>>(NNVersionsDMA::VERSION_03_RESERVED_v1);
    }

    ///@brief Ctor , inits the content with expected size
    Preprocessing_Interface03_DMA() {
        this->set_size(size_of_descriptor);
    };

    ///@brief default virtual destructor
    virtual ~Preprocessing_Interface03_DMA() = default;
};

}  // namespace VPUNN

#endif  // VPUNN_TYPES_H
