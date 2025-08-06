// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_TYPES_14_H
#define VPUNN_TYPES_14_H

#include <algorithm>
#include <array>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <vector>
#include "../types.h"  // need to know the present day types for conversion
#include "../utils.h"
#include "inference/nn_descriptor_versions.h"
#include "inference/preprocessing_inserter.h"

#include "preprocessing_adapters.h"

#include <map>
#include <sstream>
#include <string>
#include <type_traits>
#include "inference/preprocessing_inserter_basics.h"
#include "vpu/validation/dpu_operations_validator.h"

#include "types12.h"  //based on 12
// #include "types13.h"  //based on 13 maybe

namespace VPUNN {

/** @brief type interface forNPU_RESERVED v3 named 14. This is a convention on what to contain the VPUNN's input
 * descriptor in this namespace all the types will be stored exactly like they are required by this interface
 * vs intf13: UINT4 added to Dtypes
 */
namespace intf_14 {

// when making a new interface version, start copying from here

/// gives the EnumMap for a E enum type
/// has to be fully implemented for each type we want to cover
template <typename E>
inline const EnumMap& mapToText();

template <typename E>
inline const EnumTextLogicalMap& mapToLogicalText();

/// creates the  EnumInverseMap for a particular E enum type
/// @pre the EnumMap<E> must exists
template <typename E>
inline const EnumInverseMap& mapFromText() {
    static auto m = createInverseMap(mapToText<E>());
    return m;
}

/**
 * @brief Supported Datatypes in the Descriptor as One hot representation
 *
 */
enum class DataType {
    UINT8,    ///< all (u)int 8 expected to be in the same runtime performance
    FLOAT16,  ///< all F16 expected to be in the same runtime performance
    HF8,      ///< all 8 bit Float expected to be around I8, except Elmwise :around FP16
    FLOAT32,  ///< 32bit float
    UINT4,    ///< sub 8 bit types are used for palletization
    __size
};
static const EnumMap DataType_ToText{
        link(DataType::UINT8, "UINT8"),      //
        link(DataType::FLOAT16, "FLOAT16"),  //
        link(DataType::HF8, "HF8"),          //
        link(DataType::FLOAT32, "FLOAT32"),  //
        link(DataType::UINT4, "UINT4"),      //
};
template <>
inline const EnumMap& mapToText<DataType>() {
    return DataType_ToText;
}

static const EnumTextLogicalMap dtype_logical_map{
        link_logical("UINT8", "UINT8"),       // same
        link_logical("INT8", "UINT8"),        // mapped
        link_logical("FLOAT16", "FLOAT16"),   // same
        link_logical("BFLOAT16", "FLOAT16"),  // same
        link_logical("HF8", "HF8"),           // same
        link_logical("BF8", "HF8"),           // same
        link_logical("UINT4", "UINT4"),       // same, to be profiled as UINT4 palletized, only for wts
        link_logical("INT4", "UINT4"),        // mapped, to be profiled as UINT4 palletized, only for wts
        link_logical("UINT2", "UINT4"),       // mapped
        link_logical("INT2", "UINT4"),        // mapped
        link_logical("UINT1", "UINT8"),       // not supported
        link_logical("INT1", "UINT8"),        // not supported
        link_logical("FLOAT32", "FLOAT32"),   // same
        link_logical("INT32", "FLOAT16"),     // not supported
};
template <>
inline const EnumTextLogicalMap& mapToLogicalText<DataType>() {
    return dtype_logical_map;
}

/**
 * @brief converts the present day interface value to the value corresponding to this interface version
 * requires the existence of mapToText and mapFromText services for the subjected enums
 * @throws out_of_range if the conversion is not possible
 */
template <typename CompatibleEnum, typename PresentEnum>
CompatibleEnum convert(PresentEnum present_day_value_type) {
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
        std::string logically_mapped_text{""};
        try {
            logically_mapped_text = mapToLogicalText<CompatibleEnum>().at(text_val);
        } catch (const std::exception& e) {
            std::stringstream buffer;
            buffer << "[ERROR]:could not map enum value: " << text_val << " to a logical enum text!. "
                   << " Initial exception: " << e.what() << " File: " << __FILE__ << " Line: " << __LINE__;
            std::string details = buffer.str();
            throw std::out_of_range(details);
        }

        old_id = mapFromText<CompatibleEnum>().at(logically_mapped_text);
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

}  // namespace intf_14

// interface class

/// Inserts different datatypes into a descriptor buffer
template <class T, typename DeviceAdapter>
class Inserter_Interface14 : Inserter<T> {
public:
    using Inserter<T>::insert;  ///< exposes the non virtual insert methods
    Inserter_Interface14(std::vector<T>& output): Inserter<T>(output) {
    }

    /// @brief insert specialization for VPUTensor
    template <bool only_simulate>
    size_t insert(const VPUTensor& data, size_t offset) {
        offset = this->insert<only_simulate>(data.get_shape()[0], offset);
        offset = this->insert<only_simulate>(data.get_shape()[1], offset);
        offset = this->insert<only_simulate>(data.get_shape()[2], offset);
        offset = this->insert<only_simulate>(data.get_shape()[3], offset);

        // datatype is particular here
        offset = this->insert<only_simulate>(intf_14::convert<intf_14::DataType>(data.get_dtype()), offset);
        return offset;
    }
};

/**
 * @brief Based on  almost same datatype as intf12, added SUperdense and 2x inplace flags
 *
 * DATA CHANGES:
 *
 */
template <class T, typename DeviceAdapter, NNVersions V>
class Preprocessing_Interface14_Archetype :
        public PreprocessingInserter<T, Preprocessing_Interface14_Archetype<T, DeviceAdapter, V>> {
private:
    inline static const DPU_OperationValidator workload_validator{};  ///< sanitizer mechanisms
protected:
    friend class PreprocessingInserter<T, Preprocessing_Interface14_Archetype<T, DeviceAdapter, V>>;

    /**
     * @brief Transform a DPUWorkload into a DPUWorkload descriptor
     *
     * @param workload a DPUWorkload
     * @param debug_offset [out] is the offset where a new value can be written. interpreted as how many positions were
     * written
     * @tparam only_simulate, if true then no data is actually written, only the offset is computed
     * @return std::vector<T>& a DPUWorkload descriptor
     */
    template <bool only_simulate>
    void transformOnly(const DPUWorkload& workload, size_t& debug_offset,
                       std::vector<T>& destination_descriptor) const {
        Inserter_Interface14<T, DeviceAdapter> ins(destination_descriptor);

        // Build the vector from the inputs
        size_t offset = 0;

        // for enums we must put here the equivalent version  from the target interface, not latest types

        {
            const auto operation{DeviceAdapter::mock_replace_operations(workload.op)};
            offset = ins.template insert<only_simulate>(intf_12::convert<intf_12::Operation>(operation), offset);
        }

        auto insert_autopadded = [&](const VPUTensor& tensor, size_t offset,
                                     const std::optional<bool> autopad) -> size_t {
            if (autopad && tensor.channels() < 16) {
                const auto padded_tensor{VPUTensor({tensor.width(), tensor.height(), 16, tensor.batches()}, tensor)};
                offset = ins.template insert<only_simulate>(padded_tensor, offset);
            } else {
                offset = ins.template insert<only_simulate>(tensor, offset);
            }
            return offset;
        };

        offset = insert_autopadded(workload.inputs[0], offset, workload.input_autopad);

        // input 1 _type has special source
        offset = ins.template insert<only_simulate>(
                intf_14::convert<intf_14::DataType>(workload.weight_type.value_or(workload.inputs[0].get_dtype())),
                offset);

        offset = insert_autopadded(workload.outputs[0], offset, workload.output_autopad);

        offset = ins.template insert<only_simulate>(workload.kernels[0], offset);
        offset = ins.template insert<only_simulate>(workload.kernels[1], offset);

        offset = ins.template insert<only_simulate>(workload.strides[0], offset);
        offset = ins.template insert<only_simulate>(workload.strides[1], offset);

        offset = ins.template insert<only_simulate>(workload.padding[0], offset);
        offset = ins.template insert<only_simulate>(workload.padding[1], offset);
        offset = ins.template insert<only_simulate>(workload.padding[2], offset);
        offset = ins.template insert<only_simulate>(workload.padding[3], offset);

        offset = ins.template insert<only_simulate>(intf_12::convert<intf_12::ExecutionMode>(workload.execution_order),
                                                    offset);

        {
            // normalize value as it have been read from a csv (limited precision) to match the generated cache
            const float act_sprs{std::stof(std::to_string(workload.act_sparsity))};
            const float wts_sprs{std::stof(std::to_string(workload.weight_sparsity))};

            offset = ins.template insert<only_simulate>(act_sprs, offset);
            offset = ins.template insert<only_simulate>(wts_sprs, offset);
        }

        {
            const auto modified_fields{DeviceAdapter::avoid_untrained_space(workload)};

            const auto owt{modified_fields.owt};
            offset = ins.template insert<only_simulate>(owt, offset);
        }

        offset = ins.template insert<only_simulate>(intf_12::convert<intf_12::Layout>(workload.outputs[0].get_layout()),
                                                    offset);  // odu_permute

        // new fields for interface 13

        offset = ins.template insert<only_simulate>(workload.is_inplace_output_memory(), offset);
        offset = ins.template insert<only_simulate>(workload.is_weightless_operation(), offset);

        offset = ins.template insert<only_simulate>(workload.is_superdense(), offset);

        debug_offset = offset;
    }

    inline static constexpr size_t size_of_descriptor{44 + 3 + 3 + 3};  ///< how big the descriptor is, fixed at type.

public:
    /// @brief the descriptor interface that this type was designed to fill/comply with
    static int getInterfaceVersion() {
        return static_cast<std::underlying_type_t<NNVersions>>(V);
    }

    /// @brief Ctor , inits the content with expected size
    Preprocessing_Interface14_Archetype()
            : PreprocessingInserter<T, Preprocessing_Interface14_Archetype<T, DeviceAdapter, V>>(size_of_descriptor) {};

    /// @brief default virtual destructor
    virtual ~Preprocessing_Interface14_Archetype() = default;
};

//---------------------------------------------------------
template <class T>
using Preprocessing_Interface14 =
        Preprocessing_Interface14_Archetype<T, NN5XInputAdapter, NNVersions::VERSION_14_NPU_RESERVED>;

}  // namespace VPUNN

#endif  // VPUNN_TYPES_H
