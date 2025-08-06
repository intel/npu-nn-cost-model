// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_TYPES_11_H
#define VPUNN_TYPES_11_H

#include <algorithm>
#include <array>
#include <cstdlib>
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

namespace VPUNN {

/** @brief type interface for VPU2.7 Beta inputs, named 11. This is a convention on what to contain the VPUNN's input
 * descriptor in this namespace all the types will be stored exactly like they are required by this interface
 */
namespace intf_11 {

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
enum class VPUDevice { VPU_2_0, VPU_2_1, VPU_2_7, VPU_4_0, __size };
static const EnumMap VPUDevice_ToText{link(VPUDevice::VPU_2_0, "VPU_2_0"), link(VPUDevice::VPU_2_1, "VPU_2_1"),
                                      link(VPUDevice::VPU_2_7, "VPU_2_7"), link(VPUDevice::VPU_4_0, "VPU_4_0")};
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
 * VECTOR, MATRIX, VECTOR_FP16,are DELETED for VPU2.7
 *
 */
enum class ExecutionMode { CUBOID_16x16, CUBOID_8x16, CUBOID_4x16, __size };
static const EnumMap ExecutionMode_ToText{
        link(ExecutionMode::CUBOID_16x16, "CUBOID_16x16"),
        link(ExecutionMode::CUBOID_8x16, "CUBOID_8x16"),
        link(ExecutionMode::CUBOID_4x16, "CUBOID_4x16"),
};
template <>
inline const EnumMap& mapToText<ExecutionMode>() {
    return ExecutionMode_ToText;
}

/**
 * @brief Data layout
 *
 * ZMAJOR and CMAJOR are coming from VPU2.0, were DELETED!
 *
 * XYZ, XZY, YXZ, YZX, ZXY, ZYX  were introduced for 2.7
 * They are to interpreted as from  innermost to outermost dimension of the tensor
 * eg: XYZ is NCHW; N=Batch is always outermost, then channels (Z), height (Y), width (X)
 *
 */
enum class Layout { XYZ, XZY, YXZ, YZX, ZXY, ZYX, INVALID, __size };
static const EnumMap Layout_ToText{link(Layout::XYZ, "XYZ"),        link(Layout::XZY, "XZY"), link(Layout::YXZ, "YXZ"),
                                   link(Layout::YZX, "YZX"),        link(Layout::ZXY, "ZXY"), link(Layout::ZYX, "ZYX"),
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

}  // namespace intf_11

// interface class

/// Inserts different datatypes into a descriptor buffer
template <class T, typename DeviceAdapter>
class Inserter_Interface11 : Inserter<T> {
public:
    using Inserter<T>::insert;  ///< exposes the non virtual insert methods
    Inserter_Interface11(std::vector<T>& output): Inserter<T>(output) {
    }

    /// @brief insert specialization for VPUTensor
    template <bool only_simulate>
    size_t insert(const VPUTensor& data, size_t offset) {
        offset = this->insert<only_simulate>(data.get_shape()[0], offset);
        offset = this->insert<only_simulate>(data.get_shape()[1], offset);
        offset = this->insert<only_simulate>(data.get_shape()[2], offset);
        offset = this->insert<only_simulate>(data.get_shape()[3], offset);

        {  // mock BF8 and HF8 to uint8
            const auto datatype{DeviceAdapter::mock_replace_datatypes(data.get_dtype())};
            offset = this->insert<only_simulate>(intf_11::convert<intf_11::DataType>(datatype), offset);
        }
        offset = this->insert<only_simulate>(intf_11::convert<intf_11::Layout>(data.get_layout()), offset);
        offset = this->insert<only_simulate>(data.get_sparsity(), offset);
        return offset;
    }
};

/**
 * @brief Preprocessing for VPU2.7 BEta  input interface
 * Has 93  floats input
 *
 * DATA CHANGES:
 * 1) mock BF8 and HF8 to uint8
 * 2) mock_replace_devices:   all > 2.7 to 2.7
 * 3) mock_replace_operations : LAYER_NORM & ELTWISE_MUL mapped to ELTWISE;
 * 4) establishUniqueSwizzling    : 5 for all except ELMWISE where 0 is also accepted.   All should be the same, if at
 * least one is different than zero than we consider it to be all 5
 * 5) owt and ISI:  avoid_untrained_space, order of calls: c, a, b
 *
 *  5.a) CLUSTERING + OWT=2+ : not possible,           :replaced with SOK+OWT=2+ (both do no use input HALO),  filter
 * with step b) next
 *
 *  5.b) SOK + ELEMENTWISE   : not possible to profile : replace with CLU+OWT=1  (slightly smaller then real due to
 * owt=1),
 *
 *  5.c) SOH + Kernel vertical is 1: no reason to use it, no input halo necessary: replace  with CLU , , filter with a)
 * next.
 *
 * @TODO experiment: isi SOH  to put into the descriptor the memory tensor , not the compute one. This is how it was
 * trained!? HOw are the runtime evolving: smaller or bigger. The Memory Tensors  are normally smaller.   Think if all
 * memo tensors are a subject? or only ones that shrink on VPU2.7?  See some examples in Layer.cpp ion SOHH fake!
 */
template <class T, typename DeviceAdapter, NNVersions V>
class Preprocessing_Interface11_Archetype :
        public PreprocessingInserter<T, Preprocessing_Interface11_Archetype<T, DeviceAdapter, V>> {
private:
    inline static const DPU_OperationValidator workload_validator{};  ///< sanitizer mechanisms
protected:
    friend class PreprocessingInserter<T, Preprocessing_Interface11_Archetype<T, DeviceAdapter, V>>;

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
        Inserter_Interface11<T, DeviceAdapter> myIns(destination_descriptor);

        // Build the vector from the inputs
        size_t offset = 0;

        // for enums we must put here the equivalent version  from the target interface, not latest types

        {  // device 4.0 is not supported for now we are mocking VPU_4_0 with 2.7. This has to be removed when we have a
            // VPU4.0 trained NN
            const auto device{DeviceAdapter::mock_replace_devices(workload.device)};
            offset = myIns.template insert<only_simulate>(intf_11::convert<intf_11::VPUDevice>(device), offset);
        }
        {
            const auto operation{DeviceAdapter::mock_replace_operations(workload.op)};
            offset = myIns.template insert<only_simulate>(intf_11::convert<intf_11::Operation>(operation), offset);
        }

        {  // this is a special case: VPU2.7 NN for SOHHalo(SPLIT_OVER_H) splits was trained only on memory tensor
           // (smaller than input tensor). SO we need to generate the descriptor using the reduced  memory tensor for W
           // and H.
            offset = myIns.template insert<only_simulate>(DeviceAdapter::alternative_input0_spatial_memory(workload),
                                                          offset);
        }
        // input 1 tensor to be generated in place here!
        // NOTE : IN case the INPUT was changed have to be considered?Only for changes that we want to impact weights
        {
            DPUWorkload wl_adapted{workload};  // need this because eg CM_CONVto CONV changed the input size!
            wl_adapted.device = DeviceAdapter::mock_replace_devices(
                    workload.device);  // maybe changed use the Config from target device
            wl_adapted.op = DeviceAdapter::mock_replace_operations(workload.op);  // maybe changed
            wl_adapted.inputs[0] = DeviceAdapter::input0_OperationBasedReplace(workload, workload.inputs[0]);

            const auto input_1 = workload_validator.construct_input_1(wl_adapted);
            // wts type follow the computation on act types.
            // INT4/UINT4 or other type dedicated for weights is ignored and replaced with the data type from input_0
            const auto wts_established{wl_adapted.inputs[0].get_dtype()};

            const VPUTensor wts{
                    VPUTensor(input_1.get_shape(), wts_established, input_1.get_layout(), input_1.get_sparsity())};

            offset = myIns.template insert<only_simulate>(wts, offset);
        }

        offset = myIns.template insert<only_simulate>(workload.outputs[0], offset);

        offset = myIns.template insert<only_simulate>(workload.kernels[0], offset);
        offset = myIns.template insert<only_simulate>(workload.kernels[1], offset);

        offset = myIns.template insert<only_simulate>(workload.strides[0], offset);
        offset = myIns.template insert<only_simulate>(workload.strides[1], offset);

        offset = myIns.template insert<only_simulate>(workload.padding[0], offset);
        offset = myIns.template insert<only_simulate>(workload.padding[1], offset);
        offset = myIns.template insert<only_simulate>(workload.padding[2], offset);
        offset = myIns.template insert<only_simulate>(workload.padding[3], offset);

        offset = myIns.template insert<only_simulate>(
                intf_11::convert<intf_11::ExecutionMode>(workload.execution_order), offset);

        {
            // normalize value as it have been read from a csv (limited precision) to match the generated cache
            const float act_sprs{std::stof(std::to_string(workload.act_sparsity))};
            const float wts_sprs{std::stof(std::to_string(workload.weight_sparsity))};

            offset = myIns.template insert<only_simulate>(act_sprs, offset);
            offset = myIns.template insert<only_simulate>(wts_sprs, offset);
        }

        {
            const auto swizz{DeviceAdapter::establishUniqueSwizzling(workload.input_swizzling[0],
                                                                     workload.input_swizzling[1],
                                                                     workload.output_swizzling[0], workload.op)};

            offset = myIns.template insert<only_simulate>(intf_11::convert<intf_11::Swizzling>(std::get<0>(swizz)),
                                                          offset);  // for in 0

            offset = myIns.template insert<only_simulate>(intf_11::convert<intf_11::Swizzling>(std::get<1>(swizz)),
                                                          offset);  // for input 1

            offset = myIns.template insert<only_simulate>(intf_11::convert<intf_11::Swizzling>(std::get<2>(swizz)),
                                                          offset);  // for output 0
        }

        {
            const auto modified_fields{DeviceAdapter::avoid_untrained_space(workload)};

            const auto owt{modified_fields.owt};
            offset = myIns.template insert<only_simulate>(owt, offset);

            const auto isi{modified_fields.isi};
            offset = myIns.template insert<only_simulate>(intf_11::convert<intf_11::ISIStrategy>(isi), offset);
        }

        debug_offset = offset;
    }

    inline static constexpr size_t size_of_descriptor{93};  ///< how big the descriptor is, fixed at type.

public:
    /// @brief the descriptor interface that this type was designed to fill/comply with
    static int getInterfaceVersion() {
        return static_cast<std::underlying_type_t<NNVersions>>(V);
    }

    /// @brief Ctor , inits the content with expected size
    Preprocessing_Interface11_Archetype()
            : PreprocessingInserter<T, Preprocessing_Interface11_Archetype<T, DeviceAdapter, V>>(size_of_descriptor) {};

    /// @brief default virtual destructor
    virtual ~Preprocessing_Interface11_Archetype() = default;
};

//---------------------------------------------------------
template <class T>
using Preprocessing_Interface11 =
        Preprocessing_Interface11_Archetype<T, NN27InputAdapter, NNVersions::VERSION_11_VPU27_BETA>;

//--------------------------------------------------------------

template <class T>
using Preprocessing_Interface4011 =
        Preprocessing_Interface11_Archetype<T, /*NN27InputAdapter*/ NN40InputAdapter, NNVersions::VERSION_11_NPU40>;

template <class T>
using Preprocessing_Interface15911 = Preprocessing_Interface11_Archetype<T, /*NN27InputAdapter*/ NN27_159_InputAdapter,
                                                                         NNVersions::VERSION_11_V89_COMPTBL>;

template <class T>
using Preprocessing_Interface4111 =
        Preprocessing_Interface11_Archetype<T, /*NN27InputAdapter*/ NN41InputAdapter, NNVersions::VERSION_11_NPU41>;

}  // namespace VPUNN

#endif  // VPUNN_TYPES_H
