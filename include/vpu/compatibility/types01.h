// Copyright © 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_TYPES_01_H
#define VPUNN_TYPES_01_H

#include <algorithm>
#include <array>
#include <cstdlib>
#include <numeric>
#include <vector>
#include "../types.h"  // need to know the present day types for conversion
#include "../utils.h"
#include "inference/preprocessing.h"

#include <map>
#include <sstream>
#include <string>
#include <type_traits>

namespace VPUNN {

/** @brief interface for original inputs, named 01. This is a convention on what to contain the VPUNN's input descriptor
 * in this namespace all the types will be stored exactly as they are required by this interface
 */
namespace intf_01 {

template <typename E>
inline const EnumMap& mapToText();

template <typename E>
inline const EnumInverseMap& mapFromText() {
    static auto m = createInverseMap(mapToText<E>());
    return m;
}

/**
 * @brief VPU IP generations
 *
 */
enum class VPUDevice { VPU_2_0, VPU_2_1, VPU_2_7, VPU_RESERVED, __size };
static const EnumMap VPUDevice_ToText{
        link(VPUDevice::VPU_2_0, "VPU_2_0"),
        link(VPUDevice::VPU_2_1, "VPU_2_1"),
        link(VPUDevice::VPU_2_7, "VPU_2_7"),
        link(VPUDevice::VPU_RESERVED, "VPU_RESERVED"),
};
template <>
inline const EnumMap& mapToText<VPUDevice>() {
    return VPUDevice_ToText;
}

/**
 * @brief Supported Datatypes
 *
 */
enum class DataType { UINT8, INT8, FLOAT16, BFLOAT16, __size };
static const EnumMap DataType_ToText{
        link(DataType::UINT8, "UINT8"),
        link(DataType::INT8, "INT8"),
        link(DataType::FLOAT16, "FLOAT16"),
        link(DataType::BFLOAT16, "BFLOAT16"),
};
template <>
inline const EnumMap& mapToText<DataType>() {
    return DataType_ToText;
}

/**
 * @brief HW operations
 *
 */
enum class Operation { CONVOLUTION, DW_CONVOLUTION, ELTWISE, MAXPOOL, AVEPOOL, CM_CONVOLUTION, __size };
static const EnumMap Operation_ToText{
        link(Operation::CONVOLUTION, "CONVOLUTION"), link(Operation::DW_CONVOLUTION, "DW_CONVOLUTION"),
        link(Operation::ELTWISE, "ELTWISE"),         link(Operation::MAXPOOL, "MAXPOOL"),
        link(Operation::AVEPOOL, "AVEPOOL"),         link(Operation::CM_CONVOLUTION, "CM_CONVOLUTION"),
};
template <>
inline const EnumMap& mapToText<Operation>() {
    return Operation_ToText;
}

/**
 * @brief Supported activation functions
 *
 */
enum class ActivationFunction { NONE, RELU, LRELU, ADD, SUB, MULT, __size };
static const EnumMap ActivationFunction_ToText{
        link(ActivationFunction::NONE, "NONE"),   link(ActivationFunction::RELU, "RELU"),
        link(ActivationFunction::LRELU, "LRELU"), link(ActivationFunction::ADD, "ADD"),
        link(ActivationFunction::SUB, "SUB"),     link(ActivationFunction::MULT, "MULT"),
};
template <>
inline const EnumMap& mapToText<ActivationFunction>() {
    return ActivationFunction_ToText;
}

/**
 * @brief Swizzling keys
 *
 */
enum class Swizzling { KEY_0, KEY_1, KEY_2, KEY_3, KEY_4, KEY_5, __size };
static const EnumMap Swizzling_ToText{
        link(Swizzling::KEY_0, "KEY_0"), link(Swizzling::KEY_1, "KEY_1"), link(Swizzling::KEY_2, "KEY_2"),
        link(Swizzling::KEY_3, "KEY_3"), link(Swizzling::KEY_4, "KEY_4"), link(Swizzling::KEY_5, "KEY_5"),
};
template <>
inline const EnumMap& mapToText<Swizzling>() {
    return Swizzling_ToText;
}
/**
 * @brief DPU execution modes
 *
 */
enum class ExecutionMode { VECTOR, MATRIX, VECTOR_FP16, CUBOID_16x16, CUBOID_8x16, CUBOID_4x16, __size };
static const EnumMap ExecutionMode_ToText{
        link(ExecutionMode::VECTOR, "VECTOR"),           link(ExecutionMode::MATRIX, "MATRIX"),
        link(ExecutionMode::VECTOR_FP16, "VECTOR_FP16"), link(ExecutionMode::CUBOID_16x16, "CUBOID_16x16"),
        link(ExecutionMode::CUBOID_8x16, "CUBOID_8x16"), link(ExecutionMode::CUBOID_4x16, "CUBOID_4x16"),
};
template <>
inline const EnumMap& mapToText<ExecutionMode>() {
    return ExecutionMode_ToText;
}

/**
 * @brief Data layout
 *
 */
enum class Layout { ZMAJOR, CMAJOR, __size };
static const EnumMap Layout_ToText{
        link(Layout::ZMAJOR, "ZMAJOR"),
        link(Layout::CMAJOR, "CMAJOR"),
};
template <>
inline const EnumMap& mapToText<Layout>() {
    return Layout_ToText;
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
 * @brief VPU Hw subsystem
 *
 */
enum class VPUSubsystem { VPU_DPU, VPU_SHV, VPU_DMA, VPU_CPU, VPU_CMX, __size };
static const EnumMap VPUSubsystem_ToText{
        link(VPUSubsystem::VPU_DPU, "VPU_DPU"), link(VPUSubsystem::VPU_SHV, "VPU_SHV"),
        link(VPUSubsystem::VPU_DMA, "VPU_DMA"), link(VPUSubsystem::VPU_CPU, "VPU_CPU"),
        link(VPUSubsystem::VPU_CMX, "VPU_CMX"),
};
template <>
inline const EnumMap& mapToText<VPUSubsystem>() {
    return VPUSubsystem_ToText;
}

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

}  // namespace intf_01

// interface class

/**
 * @brief Preprocessing for original  input interface
 * Has 71 bytes input and populates 67
 * It is used by VPU 2.0, and VPU 2.7 from beginning 2022
 * Its NN model has only the name VPUNN and no other info
 */
template <class T>
class Preprocessing_Interface01 : public PreprocessingInserter<T, Preprocessing_Interface01<T>> {
protected:
    using PreprocessingInserter<T, Preprocessing_Interface01<T>>::insert;  ///< exposes the non virtual insert methods
    friend class PreprocessingInserter<T, Preprocessing_Interface01<T>>;

    /// @brief insert specialization for VPUTensor
    template <bool only_simulate>
    size_t insert(const VPUTensor& data, size_t offset) {
        offset = this->insert<only_simulate>(data.get_shape()[0], offset);
        offset = this->insert<only_simulate>(data.get_shape()[1], offset);
        offset = this->insert<only_simulate>(data.get_shape()[2], offset);
        offset = this->insert<only_simulate>(data.get_shape()[3], offset);

        offset = this->insert<only_simulate>(intf_01::convert<intf_01::DataType>(data.get_dtype()), offset);
        return offset;
    }

    /**
     * @brief Transform a DPUWorkload into a DPUWorkload descriptor
     *
     * @tparam only_simulate, if true then no data is actually written, only the offset is computed
     *
     * @param workload a DPUWorkload
     * @param debug_offset [out] is the offset where a new value can be written. interpreted as how many positions were
     * written
     * @return std::vector<T>& a DPUWorkload descriptor
     */
    template <bool only_simulate>
    const std::vector<T>& transformOnly(const DPUWorkload& workload, size_t& debug_offset) {
        // Build the vector from the inputs
        size_t offset = 0;

        // for enums we must put here the equivalent version  from the target interface, not latest types

        offset = this->insert<only_simulate>(intf_01::convert<intf_01::VPUDevice>(workload.device), offset);
        offset = this->insert<only_simulate>(intf_01::convert<intf_01::Operation>(workload.op), offset);

        offset = this->insert<only_simulate>(workload.inputs[0], offset);
        offset = this->insert<only_simulate>(workload.outputs[0], offset);

        offset = this->insert<only_simulate>(workload.kernels[0], offset);
        offset = this->insert<only_simulate>(workload.kernels[1], offset);

        offset = this->insert<only_simulate>(workload.strides[0], offset);
        offset = this->insert<only_simulate>(workload.strides[1], offset);

        offset = this->insert<only_simulate>(workload.padding[0], offset);
        offset = this->insert<only_simulate>(workload.padding[1], offset);
        offset = this->insert<only_simulate>(workload.padding[2], offset);
        offset = this->insert<only_simulate>(workload.padding[3], offset);

        offset =
                this->insert<only_simulate>(intf_01::convert<intf_01::ExecutionMode>(workload.execution_order), offset);
        offset = this->insert<only_simulate>(
                intf_01::convert<intf_01::ActivationFunction>(workload.activation_function), offset);
        offset = this->insert<only_simulate>(workload.act_sparsity, offset);
        offset = this->insert<only_simulate>(workload.weight_sparsity, offset);

        offset = this->insert<only_simulate>(intf_01::convert<intf_01::Swizzling>(workload.input_swizzling[0]), offset);
        offset = this->insert<only_simulate>(intf_01::convert<intf_01::Swizzling>(workload.input_swizzling[1]), offset);

        offset =
                this->insert<only_simulate>(intf_01::convert<intf_01::Swizzling>(workload.output_swizzling[0]), offset);

        offset = this->insert<only_simulate>(workload.output_write_tiles, offset);

        debug_offset = offset;

        // Return the output as a pointer to the data
        return this->processed_output;
    }

    const size_t size_of_descriptor{
            71};  ///< how big the descriptor is, fixed at constructor. This interface has 71 but writes only 67

public:
    /// @brief the descriptor interface that this type was designed to fill/comply with
    static int getInterfaceVersion() {
        return static_cast<std::underlying_type_t<NNVersions>>(NNVersions::VERSION_01_BASE);
    }

    /// @brief Ctor , inits the content with expected size
    Preprocessing_Interface01() {
        this->set_size(size_of_descriptor);
    };

    /// @brief default virtual destructor
    virtual ~Preprocessing_Interface01() = default;
};

/**
 * @brief Same enums as v01, (not with extra input)
 * Has 67 elements in buffer
 *
 * Its NN model : VPUNN-10-X
 * Since this interface version uses exactly the types of v01 no need to have dedicated new enum conversion
 */
template <class T>
class Preprocessing_Interface10 : public PreprocessingInserter<T, Preprocessing_Interface10<T>> {
protected:
    using PreprocessingInserter<T, Preprocessing_Interface10<T>>::insert;  ///< exposes the non virtual insert methods
    friend class PreprocessingInserter<T, Preprocessing_Interface10<T>>;

    /// @brief insert specialization for VPUTensor
    template <bool only_simulate>
    size_t insert(const VPUTensor& data, size_t offset) {
        offset = this->insert<only_simulate>(data.get_shape()[0], offset);
        offset = this->insert<only_simulate>(data.get_shape()[1], offset);
        offset = this->insert<only_simulate>(data.get_shape()[2], offset);
        offset = this->insert<only_simulate>(data.get_shape()[3], offset);

        offset = this->insert<only_simulate>(intf_01::convert<intf_01::DataType>(data.get_dtype()), offset);

        // offset = insert(intf_01::convert<intf_01::Layout>(data.get_layout()), offset);

        // offset = insert(data.get_sparsity(), offset);
        return offset;
    }

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
    const std::vector<T>& transformOnly(const DPUWorkload& workload, size_t& debug_offset) {
        // Build the vector from the inputs
        size_t offset = 0;

        // for enums we must put here the equivalent version  from the target interface, not latest types

        offset = this->insert<only_simulate>(intf_01::convert<intf_01::VPUDevice>(workload.device), offset);
        offset = this->insert<only_simulate>(intf_01::convert<intf_01::Operation>(workload.op), offset);

        offset = this->insert<only_simulate>(workload.inputs[0], offset);
        // offset = insert(workload.inputs_1[0], offset);

        offset = this->insert<only_simulate>(workload.outputs[0], offset);

        offset = this->insert<only_simulate>(workload.kernels[0], offset);
        offset = this->insert<only_simulate>(workload.kernels[1], offset);

        offset = this->insert<only_simulate>(workload.strides[0], offset);
        offset = this->insert<only_simulate>(workload.strides[1], offset);

        offset = this->insert<only_simulate>(workload.padding[0], offset);
        offset = this->insert<only_simulate>(workload.padding[1], offset);
        offset = this->insert<only_simulate>(workload.padding[2], offset);
        offset = this->insert<only_simulate>(workload.padding[3], offset);

        offset =
                this->insert<only_simulate>(intf_01::convert<intf_01::ExecutionMode>(workload.execution_order), offset);
        offset = this->insert<only_simulate>(
                intf_01::convert<intf_01::ActivationFunction>(workload.activation_function), offset);
        offset = this->insert<only_simulate>(workload.act_sparsity, offset);
        offset = this->insert<only_simulate>(workload.weight_sparsity, offset);

        offset = this->insert<only_simulate>(intf_01::convert<intf_01::Swizzling>(workload.input_swizzling[0]), offset);
        offset = this->insert<only_simulate>(intf_01::convert<intf_01::Swizzling>(workload.input_swizzling[1]), offset);

        offset =
                this->insert<only_simulate>(intf_01::convert<intf_01::Swizzling>(workload.output_swizzling[0]), offset);

        offset = this->insert<only_simulate>(workload.output_write_tiles, offset);

        debug_offset = offset;

        // Return the output as a pointer to the data
        return this->processed_output;
    }

    const size_t size_of_descriptor;  ///< how big the descriptor is, fixed at constructor

public:
    /// @brief the descriptor interface that this type was designed to fill/comply with
    static int getInterfaceVersion() {
        return static_cast<std::underlying_type_t<NNVersions>>(NNVersions::VERSION_10_ENUMS_SAME);
    }
    /// @brief Ctor , inits the content with expected size
    Preprocessing_Interface10(): size_of_descriptor(this->calculate_size()) {
        this->set_size(size_of_descriptor);
    };

    /// @brief default virtual destructor
    virtual ~Preprocessing_Interface10() = default;
};

}  // namespace VPUNN

#endif  // VPUNN_TYPES_H
