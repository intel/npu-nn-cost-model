// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_TYPES_12_H
#define VPUNN_TYPES_12_H

#include <algorithm>
#include <array>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <vector>
#include "../types.h"  // need to know the present day types for conversion
#include "../utils.h"
#include "inference/nn_descriptor_versions.h"
#include "inference/preprocessing.h"

#include "preprocessing_adapters.h"

#include <map>
#include <sstream>
#include <string>
#include <type_traits>
#include "vpu/validation/dpu_operations_validator.h"

namespace VPUNN {

/** @brief type interface for VPU2.7 Beta inputs, named 11. This is a convention on what to contain the VPUNN's input
 * descriptor in this namespace all the types will be stored exactly like they are required by this interface
 */
namespace intf_12 {

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
 * @brief VPU IP generations
 *
 */
enum class VPUDevice { VPU_2_0, VPU_2_1, VPU_2_7, VPU_4_0, NPU_RESERVED1, NPU_RESERVED1_W, __size };
static const EnumMap VPUDevice_ToText{link(VPUDevice::VPU_2_0, "VPU_2_0"),
                                      link(VPUDevice::VPU_2_1, "VPU_2_1"),
                                      link(VPUDevice::VPU_2_7, "VPU_2_7"),
                                      link(VPUDevice::VPU_4_0, "VPU_4_0"),
                                      link(VPUDevice::NPU_RESERVED1, "NPU_RESERVED1"),
                                      link(VPUDevice::NPU_RESERVED1_W, "NPU_RESERVED1_W")};
template <>
inline const EnumMap& mapToText<VPUDevice>() {
    return VPUDevice_ToText;
}

/**
 * @brief Supported Datatypes
 *
 */
enum class DataType {
    UINT8,     ///< all (u)int 8 expected to be in the same runtime performance
    INT8,      ///< all (u)int 8 expected to be in the same runtime performance
    FLOAT16,   ///< all F16 expected to be in the same runtime performance
    BFLOAT16,  ///< all F16 expected to be in the same runtime performance
    BF8,       ///< all 8 bit Float expected to be around I8, except Elmwise :around FP16
    HF8,       ///< all 8 bit Float expected to be around I8, except Elmwise :around FP16
    UINT4,  ///< sub 8 bit types are used for palletization  (2 INT4 in a byte), but for computation they used at least
            ///< 8 bit datatypes
    INT4,
    UINT2,
    INT2,
    UINT1,
    INT1,
    __size
};
static const EnumMap DataType_ToText{
        link(DataType::UINT8, "UINT8"),       link(DataType::INT8, "INT8"),   link(DataType::FLOAT16, "FLOAT16"),
        link(DataType::BFLOAT16, "BFLOAT16"), link(DataType::BF8, "BF8"),     link(DataType::HF8, "HF8"),
        link(DataType::UINT4, "UINT4"),       link(DataType::INT4, "INT4"),   link(DataType::UINT2, "UINT2"),
        link(DataType::INT2, "INT2"),         link(DataType::UINT1, "UINT1"), link(DataType::INT1, "INT1"),
};
template <>
inline const EnumMap& mapToText<DataType>() {
    return DataType_ToText;
}

/**
 * @brief HW operations
 *
 */
enum class Operation {
    CONVOLUTION,     //
    DW_CONVOLUTION,  //
    ELTWISE,         //
    MAXPOOL,         //
    AVEPOOL,         //
    CM_CONVOLUTION,  //
    LAYER_NORM,      //
    ELTWISE_MUL,     //
    // Other new ops?
    __size
};
static const EnumMap Operation_ToText{
        link(Operation::CONVOLUTION, "CONVOLUTION"), link(Operation::DW_CONVOLUTION, "DW_CONVOLUTION"),
        link(Operation::ELTWISE, "ELTWISE"),         link(Operation::MAXPOOL, "MAXPOOL"),
        link(Operation::AVEPOOL, "AVEPOOL"),         link(Operation::CM_CONVOLUTION, "CM_CONVOLUTION"),
        link(Operation::LAYER_NORM, "LAYER_NORM"),   link(Operation::ELTWISE_MUL, "ELTWISE_MUL"),
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
enum class Swizzling { KEY_0 /*disabled*/, KEY_1, KEY_2, KEY_3, KEY_4, KEY_5, __size };
static const EnumMap Swizzling_ToText{
        link(Swizzling::KEY_0, "KEY_0"), link(Swizzling::KEY_1, "KEY_1"), link(Swizzling::KEY_2, "KEY_2"),
        link(Swizzling::KEY_3, "KEY_3"), link(Swizzling::KEY_4, "KEY_4"), link(Swizzling::KEY_5, "KEY_5"),
};
template <>
inline const EnumMap& mapToText<Swizzling>() {
    return Swizzling_ToText;
}
/**
 * @brief DPU execution modes , see SAS/HAS
 */
enum class ExecutionMode {
    VECTOR,
    MATRIX,
    VECTOR_FP16,   //
    CUBOID_16x16,  // from 27, 40 :  NTHW/NTK = 16/4,   50 : NTHW/NTK = 16/2
    CUBOID_8x16,   // from 27, 40 :  NTHW/NTK = 8/8     50 : NTHW/NTK = 8/4
    CUBOID_4x16,   // from 27, 40 :  NTHW/NTK = 4/16    50 : NTHW/NTK = 4/8
    __size
};
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
 * ZMAJOR and CMAJOR are coming from VPU2.0, legacy layouts
 *
 *  XYZ, XZY, YXZ, YZX, ZXY, ZYX  were introduced for 2.7
 * They are to interpreted as from  innermost(contiguous) to outermost dimension of the tensor
 * eg: XYZ  is NCHW;   N=Batch is always outermost,  then channels (Z), height (Y), width (X)
 *
 * INVALID is first usage is exposure to VPUNN in some cases where Layout does not matter, is neither good Like (for
 * input_1 when MAXPOOL).
 *
 * Equivalence legacy to xyz permutations:
 * ZMAJOR is Z,X,Y
 * CMAJOR is X,Y,Z
 *
 */
enum class Layout { ZMAJOR, CMAJOR, XYZ, XZY, YXZ, YZX, ZXY, ZYX, INVALID, __size };
static const EnumMap Layout_ToText{link(Layout::ZMAJOR, "ZMAJOR"),  link(Layout::CMAJOR, "CMAJOR"),  // legacy
                                   link(Layout::XYZ, "XYZ"),        link(Layout::XZY, "XZY"),
                                   link(Layout::YXZ, "YXZ"),        link(Layout::YZX, "YZX"),
                                   link(Layout::ZXY, "ZXY"),        link(Layout::ZYX, "ZYX"),
                                   link(Layout::INVALID, "INVALID")};
template <>
inline const EnumMap& mapToText<Layout>() {
    return Layout_ToText;
}

/// @brief ISI_Strategy
enum class ISIStrategy { CLUSTERING, SPLIT_OVER_H, SPLIT_OVER_K, __size };
static const EnumMap ISIStrategy_ToText{
        link(ISIStrategy::CLUSTERING, "Clustering"),
        link(ISIStrategy::SPLIT_OVER_H, "SplitOverH"),
        link(ISIStrategy::SPLIT_OVER_K, "SplitOverK"),
};
template <>
inline const EnumMap& mapToText<ISIStrategy>() {
    return ISIStrategy_ToText;
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

}  // namespace intf_12

// interface class

/**
 * @brief Preprocessing for HALO , VPU27/NPU40 and newer
 * Has 118  floats input
 * Mocks BF8 and HS8 to UINT8
 * Mocks NPU_RESERVED1 ops to something available in NPU4
 * removed ISI
 * removed device
 * changed swizzle descriptor from one_hot enum to a enabled/disabled bool
 * Added HALO
 *
 * DATA CHANGES:
 * 1) mock BF8 and HF8 to uint8
 * 3) mock_replace_operations : operation mock for NPU_RESERVED1: LAYER_NORM & ELTWISE_MUL mapped to ELTWISE;
 * 4) establishUniqueSwizzling    : ON for all except ELMWISE where OFF is also accepted.   All should be the same, if
 * at least one is different than OFF than we consider it to be all ON
 *
 */
template <class T, typename DeviceAdapter, NNVersions V>
class Preprocessing_Interface12_Archetype :
        public PreprocessingInserter<T, Preprocessing_Interface12_Archetype<T, DeviceAdapter, V>> {
private:
    const DPU_OperationValidator workload_validator{};  ///< sanitizer mechanisms
protected:
    using PreprocessingInserter<T, Preprocessing_Interface12_Archetype<T, DeviceAdapter, V>>::
            insert;  ///< exposes the non virtual insert methods
    friend class PreprocessingInserter<T, Preprocessing_Interface12_Archetype<T, DeviceAdapter, V>>;

    /// @brief insert specialization for VPUTensor
    template <bool only_simulate>
    size_t insert(const VPUTensor& data, size_t offset) {
        offset = this->insert<only_simulate>(data.get_shape(), offset);

        {  // mock BF8 and HF8 to uint8
            const auto datatype{DeviceAdapter::mock_replace_datatypes(data.get_dtype())};
            offset = this->insert<only_simulate>(intf_12::convert<intf_12::DataType>(datatype), offset);
        }
        offset = this->insert<only_simulate>(intf_12::convert<intf_12::Layout>(data.get_layout()), offset);
        offset = this->insert<only_simulate>(data.get_sparsity(), offset);
        return offset;
    }

    // specialization for latest swizzling. Each version type should have its own if wants special treatment
    // instead of enum one hot style, use a boolean enabled/disabled
    template <bool only_simulate>
    size_t insert(intf_12::Swizzling data, size_t offset) {
        // 1 for enabled :Key_n, zero for disabled: KEY_0
        const bool is_swizzling_enabled{(data != intf_12::Swizzling::KEY_0)};
        return this->insert<only_simulate>(is_swizzling_enabled, offset);
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

        // no more device
        {  // operation mock for NPU_RESERVED1
            const auto operation{DeviceAdapter::mock_replace_operations(workload.op)};
            offset = this->insert<only_simulate>(intf_12::convert<intf_12::Operation>(operation), offset);
        }

        offset = this->insert<only_simulate>(workload.inputs[0], offset);
        // input 1 tensor to be generated in place here!
        {
            auto input_1 = workload_validator.construct_input_1(workload);
            // wts type follow the computation on act types.
            // INT4/UINT4 or other type dedicated for weights is ignored and replaced with the data type from input_0
            const auto wts_established{workload.inputs[0].get_dtype()};

            const VPUTensor wts{
                    VPUTensor(input_1.get_shape(), wts_established, input_1.get_layout(), input_1.get_sparsity())};

            offset = this->insert<only_simulate>(wts, offset);
        }

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
                this->insert<only_simulate>(intf_12::convert<intf_12::ExecutionMode>(workload.execution_order), offset);

        offset = this->insert<only_simulate>(workload.act_sparsity, offset);
        offset = this->insert<only_simulate>(workload.weight_sparsity, offset);

        {
            const auto one_swizling{DeviceAdapter::establishUniqueSwizzling(workload.input_swizzling[0],
                                                                            workload.input_swizzling[1],
                                                                            workload.output_swizzling[0], workload.op)};

            offset = this->insert<only_simulate>(intf_12::convert<intf_12::Swizzling>(std::get<0>(one_swizling)),
                                                 offset);  // for input 0

            offset = this->insert<only_simulate>(intf_12::convert<intf_12::Swizzling>(std::get<1>(one_swizling)),
                                                 offset);  // for input 1

            offset = this->insert<only_simulate>(intf_12::convert<intf_12::Swizzling>(std::get<2>(one_swizling)),
                                                 offset);  // for output 0
        }

        offset = this->insert<only_simulate>(workload.output_write_tiles, offset);

        // rem isi_strategy

        offset = this->insert<only_simulate>(workload.halo, offset);  // new in v12

        debug_offset = offset;

        // Return the output as a pointer to the data
        return this->processed_output;
    }

    inline static const size_t size_of_descriptor{124};  ///< how big the descriptor is, fixed at constructor.

public:
    /// @brief the descriptor interface that this type was designed to fill/comply with
    static int getInterfaceVersion() {
        return static_cast<std::underlying_type_t<NNVersions>>(V);
    }

    /// @brief Ctor , inits the content with expected size
    Preprocessing_Interface12_Archetype() {
        this->set_size(size_of_descriptor);
    };

    /// @brief default virtual destructor
    virtual ~Preprocessing_Interface12_Archetype() = default;
};

//---------------------------------------------------------
template <class T>
using Preprocessing_Interface12 = Preprocessing_Interface12_Archetype<T, NN40InputAdapter, NNVersions::VERSION_12_HALO>;

}  // namespace VPUNN

#endif  // VPUNN_TYPES_H
