// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_DPU_TYPES_H
#define VPUNN_DPU_TYPES_H

#include <iostream>  //y
#include <map>       //y
#include <sstream>   //
#include <string>    //y
#include <type_traits>
#include <utility>  //y
#include <vector>

namespace VPUNN {

/// @brief mapping an enum value to its string/text representation
using EnumMap = std::map<int, const std::string>;

/// @brief creates a pair to be added to the EnumMap
/// @tparam T , the enumeration type
template <class T>
EnumMap::value_type link(const T& enum_val, const char* name) {
    return EnumMap::value_type{static_cast<int>(enum_val), std::string(name)};
}

/// @brief reverse of EnumMap (from string to value)
using EnumInverseMap = std::map<const std::string, const int>;

/// @brief creates and inverse map given a direct map (EnumMap)
inline const EnumInverseMap createInverseMap(const EnumMap& direct_map) {
    EnumInverseMap inverse_map;
    for (const auto& elem : direct_map) {
        inverse_map.emplace(std::make_pair(elem.second, elem.first));
    }
    return inverse_map;
}

/// @brief Holds mappings between names (usually used for direct mapping between interfaces)
using EnumTextLogicalMap = std::map<std::string, std::string>;
/// @brief creates a pair to be added to the EnumTextLogicalMap
inline EnumTextLogicalMap::value_type link_logical(std::string name, std::string mapped_name) {
    return EnumTextLogicalMap::value_type{name, mapped_name};
}

/// @brief helper EnumMap << operator
inline std::ostream& operator<<(std::ostream& os, const EnumMap& m) {
    os << "[ ";
    for (const auto& p : m) {
        os << "( " << p.first << ":" << p.second << " )";
    }
    os << " ]";
    return os;
}
/// @brief helper EnumInverseMap << operator
inline std::ostream& operator<<(std::ostream& os, const EnumInverseMap& m) {
    os << "[ ";
    for (const auto& p : m) {
        os << "( " << p.first << ":" << p.second << " )";
    }
    os << " ]";
    return os;
}

// when making a new interface version, start copying from here

/// gives the EnumMap for a E enum type
/// has to be fully implemented for each type we want to cover
template <typename E>
inline typename std::enable_if<std::is_enum_v<E>, const EnumMap&>::type mapToText();

template <typename E>
inline typename std::enable_if<std::is_enum_v<E>, std::string>::type enumName();

template <typename T, typename = void>
struct has_mapToText : std::false_type {};

template <typename T>
struct has_mapToText<T, std::void_t<decltype(mapToText<T>())>> :
        std::is_same<decltype(mapToText<T>()), const EnumMap&> {};

template <typename T, typename = void>
struct has_enumName : std::false_type {};

template <typename T>
struct has_enumName<T, std::void_t<decltype(enumName<T>())>> : std::is_same<decltype(enumName<T>()), std::string> {};

/// creates the  EnumInverseMap for a particular E enum type
/// @pre the EnumMap<E> must exists
template <typename E>
inline typename std::enable_if<has_mapToText<E>::value, const EnumInverseMap&>::type mapFromText() {
    static auto m = createInverseMap(mapToText<E>());
    return m;
}

/**
 * @brief VPU IP generations
 *
 */
enum class VPUDevice { VPU_2_0, VPU_2_1, VPU_2_7, VPU_4_0, NPU_RESERVED, NPU_RESERVED_W, __size };
static const EnumMap VPUDevice_ToText{link(VPUDevice::VPU_2_0, "VPU_2_0"), link(VPUDevice::VPU_2_1, "VPU_2_1"),
                                      link(VPUDevice::VPU_2_7, "VPU_2_7"), link(VPUDevice::VPU_4_0, "VPU_4_0"),
                                      link(VPUDevice::NPU_RESERVED, "NPU_RESERVED"), link(VPUDevice::NPU_RESERVED_W, "NPU_RESERVED_W")};
template <>
inline const EnumMap& mapToText<VPUDevice>() {
    return VPUDevice_ToText;
}

template <>
inline std::string enumName<VPUDevice>() {
    return "VPUDevice";
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
    INT32,    ///< 32bit integer
    FLOAT32,  ///< 32bit float
    __size    ///< last element, its value  equals number of useful enum values
};
static const EnumMap DataType_ToText{
        link(DataType::UINT8, "UINT8"),       link(DataType::INT8, "INT8"),       link(DataType::FLOAT16, "FLOAT16"),
        link(DataType::BFLOAT16, "BFLOAT16"), link(DataType::BF8, "BF8"),         link(DataType::HF8, "HF8"),
        link(DataType::UINT4, "UINT4"),       link(DataType::INT4, "INT4"),       link(DataType::UINT2, "UINT2"),
        link(DataType::INT2, "INT2"),         link(DataType::UINT1, "UINT1"),     link(DataType::INT1, "INT1"),
        link(DataType::INT32, "INT32"),       link(DataType::FLOAT32, "FLOAT32"),
};
template <>
inline const EnumMap& mapToText<DataType>() {
    return DataType_ToText;
}

template <>
inline std::string enumName<DataType>() {
    return "DataType";
}

/**
 * @brief HW operations
 *
 */
enum class Operation {
    CONVOLUTION,     //
    DW_CONVOLUTION,  //
    ELTWISE,         // ADD + SUB
    MAXPOOL,         //
    AVEPOOL,         //
    CM_CONVOLUTION,  //
    LAYER_NORM,      // new LayerNorm (Sum of Squares)??
    ELTWISE_MUL,     // MUL
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

template <>
inline std::string enumName<Operation>() {
    return "Operation";
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

template <>
inline std::string enumName<ActivationFunction>() {
    return "ActivationFunction";
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

template <>
inline std::string enumName<Swizzling>() {
    return "Swizzling";
}

/**
 * @brief DPU execution modes
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

template <>
inline std::string enumName<ExecutionMode>() {
    return "ExecutionMode";
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

template <>
inline std::string enumName<Layout>() {
    return "Layout";
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

template <>
inline std::string enumName<ISIStrategy>() {
    return "ISIStrategy";
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

template <>
inline std::string enumName<VPUSubsystem>() {
    return "VPUSubsystem";
}

inline std::istream& operator>>(std::istream& is, std::string& dt) {
    std::getline(is, dt);
    return is;
}

template <typename T, typename = void>
struct is_mapFromText_callable : std::false_type {};

template <typename T>
struct is_mapFromText_callable<T, std::void_t<decltype(mapFromText<T>())>> :
        std::is_same<decltype(mapFromText<T>()), const EnumInverseMap&> {};

template <typename E>
typename std::enable_if<is_mapFromText_callable<E>::value, std::istream&>::type operator>>(std::istream& is, E& dt) {
    std::string raw_value;
    is >> raw_value;

    // the Enum will be in the form of "<EnumName>.<EnumStringValue>" we need only the <EnumStringValue>
    std::vector<std::string> tokens;
    std::stringstream ss(raw_value);
    std::string token;

    while (std::getline(ss, token, '.')) {
        tokens.push_back(token);
    }

    if (!tokens.empty()) {
        dt = static_cast<E>(mapFromText<E>().at(tokens.back()));
    }

    return is;
}

// when making a new interface version, Stop copying here

}  // namespace VPUNN

#endif
