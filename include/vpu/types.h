// Copyright © 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_TYPES_H
#define VPUNN_TYPES_H

#include <algorithm>
#include <array>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <vector>
#include "utils.h"

#include <cmath>
#include <iostream>
#include <map>
#include <sstream>  //
#include <string>

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
enum class VPUDevice { VPU_2_0, VPU_2_1, VPU_2_7, VPU_RESERVED, __size };
static const EnumMap VPUDevice_ToText{link(VPUDevice::VPU_2_0, "VPU_2_0"), link(VPUDevice::VPU_2_1, "VPU_2_1"),
                                      link(VPUDevice::VPU_2_7, "VPU_2_7"), link(VPUDevice::VPU_RESERVED, "VPU_RESERVED")};
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
 * ZMAJOR and CMAJOR are coming from VPU2.0, legacy layouts
 *
 *  XYZ, XZY, YXZ, YZX, ZXY, ZYX  were introduced for 2.7
 * They are to interpreted as from  innermost to outermost dimension of the tensor
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

//
namespace Dim {
enum Grid { W, H };
enum Act { X, Y, Z, B };
enum Wt { K, C, Ky, Kx };
enum Padding { TOP, BOTTOM, LEFT, RIGHT };
}  // namespace Dim

/** @brief Get the size of the dtype
 *
 * @param dtype a DataType object
 * @return size in bytes
 */
constexpr unsigned int dtype_to_bytes(DataType dtype) noexcept {
    switch (dtype) {
    case DataType::FLOAT16:
    case DataType::BFLOAT16:
        return 2;
    default:
        return 1;
    }
}

/// @brief default Layout that is equivalent with legacy ZMAJOR
constexpr Layout getDefaultLayout() {
    return Layout::ZXY;
}

/**
 * @brief Get the tensor serial order given a layout
 *
 * @param layout a Tensor Layout
 * @return std::array<unsigned int, 4>, order of dimensions from innermost to outermost. values represent Dim::Act
 *
 * Invalid will be mapped to the default one : ZMAJOR/ZXY
 */
constexpr std::array<unsigned int, 4> layout_to_order(Layout layout) noexcept {
    switch (layout) {
    case Layout::CMAJOR:
        return {Dim::Act::X, Dim::Act::Y, Dim::Act::Z, Dim::Act::B};  // X,Y,Z,B  from innermost to outermost dimensions
    case Layout::ZMAJOR:
        return {Dim::Act::Z, Dim::Act::X, Dim::Act::Y, Dim::Act::B};  // Z,X,Y,B  from innermost to outermost dimensions

    case Layout::XYZ:
        return {Dim::Act::X, Dim::Act::Y, Dim::Act::Z, Dim::Act::B};  // X,Y,Z, B
    case Layout::XZY:
        return {Dim::Act::X, Dim::Act::Z, Dim::Act::Y, Dim::Act::B};  // X,Z,Y, B

    case Layout::YXZ:
        return {Dim::Act::Y, Dim::Act::X, Dim::Act::Z, Dim::Act::B};  // Y,X,Z, B
    case Layout::YZX:
        return {Dim::Act::Y, Dim::Act::Z, Dim::Act::X, Dim::Act::B};  // Y,Z,X, B

    case Layout::ZXY:
        return {Dim::Act::Z, Dim::Act::X, Dim::Act::Y, Dim::Act::B};  // Z,X,Y, B
    case Layout::ZYX:
        return {Dim::Act::Z, Dim::Act::Y, Dim::Act::X, Dim::Act::B};  // Z,Y,X, B

    case Layout::INVALID:                                             // fall-through
    default:                                                          // ZMajor like, this is the first in the enum list
        return {Dim::Act::Z, Dim::Act::X, Dim::Act::Y, Dim::Act::B};  // Z,X,Y,B  from innermost to outermost dimensions
    }
}

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

/**
 * @brief Return grid in X, Y, Z, B format
 *
 * @param mode a DPUWorkload ExecutionMode
 * @return std::vector<unsigned int>
 */
inline std::vector<unsigned int> mpe_mode_to_grid(ExecutionMode mode) {
    switch (mode) {
    case ExecutionMode::VECTOR:
        return {16, 1, 16, 1};
    case ExecutionMode::VECTOR_FP16:
        return {4, 1, 16, 1};
    default:
        return {4, 4, 16, 1};
    }
}

/**
 * @brief Return the NTHW/NTK grid in X, Y, Z, B format
 *
 * @param mode a DPUWorkload ExecutionMode
 * @return std::vector<unsigned int>
 */
inline std::vector<unsigned int> mpe_mode_to_nthw_ntk_grid(ExecutionMode mode) {
    switch (mode) {
    case ExecutionMode::CUBOID_4x16:
        return {8, 8, 256, 1};
    case ExecutionMode::CUBOID_8x16:
        return {16, 8, 128, 1};
    case ExecutionMode::CUBOID_16x16:
        return {16, 16, 64, 1};
    default:
        return {1, 1, 1, 1};
    }
}

/**
 * @brief Cost model tensor class
 *
 */
class VPUTensor {
private:
    std::array<unsigned int, 4> shape;  ///< the 4 dimensions of the real tensor in the order Dim::Act XYZC  WHCB
    DataType dtype;                     ///< datatatype of the described tensor
    Layout layout;                      ///< memory organization of the tensor
    bool sparsity;                      ///< is sparsity present?
    std::array<unsigned int, 4> strides{0, 0, 0, 0};  ///< strides of the tensor's dimensions. order is the same as for
                                                      ///< shape. Strides are computed in constructor.

    void compute_strides() {
        auto size = dtype_to_bytes(dtype);
        const auto order = layout_to_order(layout);
        for (long unsigned int idx = 0; idx < order.size(); idx++) {
            strides[order[idx]] = size;
            size *= shape[order[idx]];
        }
    }

public:
    /**
     * @brief Construct a new VPUTensor object
     *
     * @param shape VPUTensor shape in width, height, channels, batch format
     * @param dtype VPUTensor datatype
     * @param layout VPUTensor layout (default:  Layout::ZXY , ZMAJOR equivalent)
     * @param sparsity true if sparsity is present
     */
    explicit VPUTensor(const std::array<unsigned int, 4>& shape = {1, 1, 1, 1}, DataType dtype = DataType::UINT8,
                       Layout layout = Layout::ZXY /*ZMAJOR equivalent*/, bool sparsity = false)
            : shape(shape), dtype(dtype), layout(layout), sparsity(sparsity) {
        compute_strides();
    };

    /**
     * @brief Construct a new VPUTensor object
     *
     * @param width VPUTensor width
     * @param height VPUTensor height
     * @param channels VPUTensor channels
     * @param batch VPUTensor batch
     * @param dtype VPUTensor datatype
     * @param layout VPUTensor layout (default:  Layout::ZXY , ZMAJOR equivalent)
     * @param sparsity true if sparsity is present
     */
    explicit VPUTensor(unsigned int width, unsigned int height, unsigned int channels, unsigned int batch,
                       DataType dtype, Layout layout = Layout::ZXY /*ZMAJOR equivalent*/, bool sparsity = false)
            : VPUTensor({width, height, channels, batch}, dtype, layout, sparsity){};

    /**
     * @brief Construct a new VPUTensor object based on a shape , and taken the rest of attributes from another tensor
     *
     * @param shape_ VPUTensor shape in width, height, channels, batch format
     * @param rest a reference to a tensor that provides all info besides shape
     */
    explicit VPUTensor(const std::array<unsigned int, 4>& shape_, const VPUTensor& rest)
            : VPUTensor(shape_, rest.get_dtype(), rest.get_layout(), rest.get_sparsity()){};

    /// @brief Get the x dimension
    unsigned int x() const noexcept {
        return shape[Dim::Act::X];
    };

    /// @brief Get the y dimension
    unsigned int y() const noexcept {
        return shape[Dim::Act::Y];
    };

    /// @brief Get the z dimension
    unsigned int z() const noexcept {
        return shape[Dim::Act::Z];
    };

    /// @brief Get the batch dimension
    unsigned int b() const noexcept {
        return shape[Dim::Act::B];
    };

    /// @brief Get the x dimension stride
    unsigned int sx() const noexcept {
        return strides[Dim::Act::X];
    };

    /// @brief Get the y dimension stride
    unsigned int sy() const noexcept {
        return strides[Dim::Act::Y];
    };

    /// @brief Get the z dimension stride
    unsigned int sz() const noexcept {
        return strides[Dim::Act::Z];
    };

    /// @brief Get the batch dimension stride
    unsigned int sb() const noexcept {
        return strides[Dim::Act::B];
    };

    /// @brief Get the height
    unsigned int height() const noexcept {
        return y();
    };

    /// @brief Get the width
    unsigned int width() const noexcept {
        return x();
    };

    /// @brief Get the channels
    unsigned int channels() const noexcept {
        return z();
    };

    /// @brief Get the batches dimension
    unsigned int batches() const noexcept {
        return b();
    };

    /// @brief Get the size in bytes
    /// @return size in bytes
    unsigned int size() const {
        return multiply_vector(shape) * dtype_to_bytes(dtype);
    }

    /// @brief Check if the tensor is floating point
    /// @return true if floating point type
    bool is_float() const noexcept {
        switch (dtype) {
        case DataType::FLOAT16:
        case DataType::BFLOAT16:
            return true;
        default:
            return false;
        }
    }

    /// @brief Check if the tensor is integer
    /// @return true if integer type
    bool is_int() const noexcept {
        return !is_float();
    }

    /// @brief Get the shape
    /// @return a 4 vector containing the shape in convention XYZB
    const std::array<unsigned int, 4>& get_shape() const noexcept {
        return shape;
    }

    /// @brief Set the VPUTensor shape
    /// @param in_shape in convention XYZB
    void set_shape(std::array<unsigned int, 4> in_shape) {
        shape = in_shape;
        compute_strides();
    }

    /// @brief Get the datatype
    DataType get_dtype() const noexcept {
        return dtype;
    }

    /// @brief changes the underlying data type only if same size new vs old
    /// @returns newly set type.
    DataType change_datatype_superficial(DataType new_datatype) {
        const auto size = dtype_to_bytes(get_dtype());
        if (size == dtype_to_bytes(new_datatype)) {
            dtype = new_datatype;
        }
        return get_dtype();
    }

    /// @brief Get the layout
    Layout get_layout() const noexcept {
        return layout;
    }

    /// @brief changes the layout type if new one has the same structure as old
    /// this change must not affect the shape or strides
    /// @param new_layout the desired layout
    /// @returns true if new layout set, false otherwise
    bool set_if_same_layout(Layout new_layout) noexcept {
        const auto order_now = layout_to_order(layout);
        const auto order_next = layout_to_order(new_layout);
        if (order_now == order_next) {
            layout = new_layout;
            return true;
        }
        return false;  // no change
    }

    /// @brief Get the sparsity flag
    bool get_sparsity() const noexcept {
        return sparsity;
    }

    /// equality test operator
    bool operator==(const VPUTensor& b) const {
        bool r{true};
        r = r && (shape == b.shape);
        r = r && (dtype == b.dtype);
        r = r && (layout == b.layout);
        r = r && (sparsity == b.sparsity);

        return r;
    }
};

inline constexpr Swizzling default_init_swizzling() {
    return Swizzling::KEY_0;
}

/// @brief The base structure that encodes a DPU workloads
struct DPUWorkload {
    VPUDevice device;  ///< device family, VPU2_0, 2_7, ...
    Operation op;      ///< operation, like convolution, etc

    std::array<VPUTensor, 1> inputs;  ///< input0 tensors, the data/activation tensor details

    std::array<VPUTensor, 1> outputs;  ///<   output tensors

    std::array<unsigned int, 2> kernels;  ///< kernel sizes WH
    std::array<unsigned int, 2> strides;  ///< kernel strides WH
    std::array<unsigned int, 4> padding;  ///< kernel padding  Top, Bottom, Left,  Right

    ExecutionMode execution_order;  ///< execution mode

    ActivationFunction activation_function = ActivationFunction::NONE;  ///< operation activation function

    float act_sparsity = 0;     ///< input activation sparsity
    float weight_sparsity = 0;  ///< weight sparsity

    std::array<Swizzling, 2> input_swizzling = {default_init_swizzling(),
                                                default_init_swizzling()};   ///< input tensors swizzling
    std::array<Swizzling, 1> output_swizzling = {default_init_swizzling()};  ///< output tensors swizzling

    /// @brief broadcast policy, Split Over K situation , In the SOK tiling strategy, weights are split across
    /// the tiles over the K dimension. The DPU in each tile compute a K-slice of the output tensors and
    /// then broadcast the result in each CMX tile, implicitly concatenating the results and having then
    /// all activations completely replicated
    unsigned int output_write_tiles{1};

    std::array<unsigned int, 4> offsets = {0, 0, 0, 0};  ///< offsets relative to the parent DPULayer, L2 API

    ISIStrategy isi_strategy{ISIStrategy::CLUSTERING};  ///< inter slice interconnect strategy , from 2.7 onwards
    bool weight_sparsity_enabled{false};  ///< is sparsity enabled for input_1(weights)? This cannot be deduced,is
                                          ///< independent(can be true for sparsity rate =0)

    /// equality test operator
    bool operator==(const DPUWorkload& b) const {
        bool r{true};
        r = r && (device == b.device);
        r = r && (op == b.op);
        r = r && (inputs == b.inputs);
        r = r && (outputs == b.outputs);

        r = r && (kernels == b.kernels);
        r = r && (strides == b.strides);
        r = r && (padding == b.padding);

        r = r && (execution_order == b.execution_order);
        r = r && (activation_function == b.activation_function);

        const float EPSILON{0.00001f};
        auto is_equal = [&EPSILON](float a, float b) {
            return (std::fabs(a - b) < EPSILON);  // very simple since vals around zero
        };
        r = r && (is_equal(act_sparsity, b.act_sparsity));
        r = r && (is_equal(weight_sparsity, b.weight_sparsity));

        r = r && (input_swizzling == b.input_swizzling);
        r = r && (output_swizzling == b.output_swizzling);

        r = r && (output_write_tiles == b.output_write_tiles);
        r = r && (isi_strategy == b.isi_strategy);
        r = r && (weight_sparsity_enabled == b.weight_sparsity_enabled);

        return r;
    }
};

inline std::ostream& operator<<(std::ostream& stream, const VPUNN::VPUTensor& d) {
    stream << "VPUTensor: \n"                                //
           << " shape: \t{" << d.x() << "," << d.y() << ","  //
           << d.z() << "," << d.b() << "} ;\n"               //
           << " dtype: \t" << (int)d.get_dtype() << " : " << DataType_ToText.at(static_cast<int>(d.get_dtype()))
           << " ;\n"  //
           << " layout: \t" << (int)d.get_layout() << " : " << Layout_ToText.at(static_cast<int>(d.get_layout()))
           << " ;\n"                                                              //
           << " sparsity: \t" << (d.get_sparsity() ? "true" : "false") << " ;\n"  //
           << " strides: \t{" << d.sx() << "," << d.sy() << ","                   //
           << d.sz() << "," << d.sb() << "} ;\n"                                  //
            ;
    return stream;
}

inline std::ostream& operator<<(std::ostream& stream, const VPUNN::DPUWorkload& d) {
    stream << "Workload: \n"                                                                                        //
           << " device: \t" << (int)d.device << " : " << VPUDevice_ToText.at(static_cast<int>(d.device)) << " ;\n"  //
           << " Operation: \t" << (int)d.op << " : " << Operation_ToText.at(static_cast<int>(d.op))
           << " ;\n"  //

           // inputs and oytputs tensors
           << " input: \t{\n"
           << d.inputs[0] << " } ;\n"  //
           << " output: \t{\n"
           << d.outputs[0] << " } ;\n"  //

           << " kernels: [W,H]  \t{" << d.kernels[Dim::Grid::W] << "," << d.kernels[Dim::Grid::H] << "} ;\n"  //
           << " strides: [W,H]  \t{" << d.strides[Dim::Grid::W] << "," << d.strides[Dim::Grid::H] << "} ;\n"  //
           << " padding: [TBLR] \t{" << d.padding[Dim::TOP] << "," << d.padding[Dim::BOTTOM] << ","           //
           << d.padding[Dim::LEFT] << "," << d.padding[Dim::RIGHT] << "} ;\n"                                 //

           << " execution_order: \t" << (int)d.execution_order << " : "
           << ExecutionMode_ToText.at(static_cast<int>(d.execution_order)) << " ;\n"  //
           << " activation_function: \t" << (int)d.activation_function << " : "
           << ActivationFunction_ToText.at(static_cast<int>(d.activation_function)) << " ;\n"  //

           << " act_sparsity: \t" << d.act_sparsity << " ;\n"        //
           << " weight_sparsity: \t" << d.weight_sparsity << " ;\n"  //

           << " input_swizzling: \t{" << (int)d.input_swizzling[0] << "," << (int)d.input_swizzling[1] << "}"
           << " :  {" << Swizzling_ToText.at(static_cast<int>(d.input_swizzling[0])) << ","
           << Swizzling_ToText.at(static_cast<int>(d.input_swizzling[1])) << "} ;\n"  //

           << " output_swizzling: \t{" << (int)d.output_swizzling[0] << "}"
           << " :  {" << Swizzling_ToText.at(static_cast<int>(d.output_swizzling[0])) << "} ;\n"  //

           << " output_write_tiles: \t" << d.output_write_tiles << " ;\n"    //
           << " offsets: \t{" << d.offsets[0] << "," << d.offsets[1] << ","  //
           << d.offsets[2] << "," << d.offsets[3] << "} ;\n"                 //
           << " isi_strategy: \t" << (int)d.isi_strategy << " : "
           << ISIStrategy_ToText.at(static_cast<int>(d.isi_strategy)) << " ;\n"  //
           << " weight_sparsity_enabled: \t" << (int)d.weight_sparsity_enabled << " : "
           << (d.weight_sparsity_enabled ? "true" : "false") << " ;\n"  //
            ;
    return stream;
}

/**
 * @brief The base structure that encodes a DMA workloads
 *
 */
struct DMAWorkload {
    VPUDevice device;  ///< VPU device

    VPUTensor input;   ///< input tensor
    VPUTensor output;  ///<  output tensor

    MemoryLocation input_location;   ///< Input memory location
    MemoryLocation output_location;  ///<  Output memory location

    unsigned int output_write_tiles = 1;  ///< number of CMX tiles to broadcast. NOT USED!

    /**
     * @brief This function computes the size of the DMAWorkload features to feed to the NN
     *
     * @return unsigned int
     */
    static unsigned int sizeTODELETEME() {
        unsigned int size = 1;                        // output_write_tiles size
        size += static_cast<int>(VPUDevice::__size);  // Size of Device
        size += 2 * (4 + static_cast<int>(DataType::__size) +
                     static_cast<int>(Layout::__size));  // Input + output tensor size
        return size;
    }
};

inline std::ostream& operator<<(std::ostream& stream, const VPUNN::DMAWorkload& d) {
    stream << "DMA Workload: \n"  //
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
            ;
    return stream;
}

/**
 * @brief The base structure that encodes a Software layer
 *
 */
struct SWOperation {
    VPUDevice device;                      ///< The VPU device
    const std::vector<VPUTensor> inputs;   ///< The input tensors
    const std::vector<VPUTensor> outputs;  ///< The output tensors

    /// @brief ctor must exist since we have aggregate initialization possible on this type (abstract type)
    SWOperation(const VPUDevice& device, const std::vector<VPUTensor>& inputs, const std::vector<VPUTensor>& outputs)
            : device{device}, inputs{inputs}, outputs{outputs} {
    }

    /**
     * @brief Return the number of cycles of the sw operation
     *
     * @return unsigned int
     */
    virtual unsigned int cycles() const = 0;

    /**
     * @brief Destroy the SWOperation object
     *
     */
    virtual ~SWOperation(){};
};

/**
 * @brief describes a Software layer (SHAVE) request
 */
class SHAVEWorkload {
    std::string name;  ///<  the name of the SW operation. We have a very flexible range of them.
    VPUDevice device;  ///< The VPU device. There will be different methods/calibrations/profiling depending on device

    // input and output tensors number and content must be correlated with the operation and among themselves. Not all
    // combinations are possible
    std::vector<VPUTensor> inputs;   ///< The input tensors. Mainly shape and datatype are used
    std::vector<VPUTensor> outputs;  ///< The output tensors. Mainly shape and datatype are used

public:
    /// @brief ctor must exist since we have aggregate initialization possible on this type (abstract type)
    SHAVEWorkload(const std::string& operation_name, const VPUDevice& device, const std::vector<VPUTensor>& inputs,
                  const std::vector<VPUTensor>& outputs)
            : name(operation_name), device{device}, inputs{inputs}, outputs{outputs} {
    }

    SHAVEWorkload(const SHAVEWorkload&) = default;
    SHAVEWorkload& operator=(const SHAVEWorkload&) = default;

    // accessors

    std::string get_name() const {
        return name;
    };
    VPUDevice get_device() const {
        return device;
    };
    const std::vector<VPUTensor>& get_inputs() const {
        return inputs;
    };
    const std::vector<VPUTensor>& get_outputs() const {
        return outputs;
    };
    std::string toString() const {
        std::stringstream stream;
        stream << "SHAVEWorkload: \n"                                                                                //
               << " Operation: \t" << name << " ;\n"                                                                 //
               << " device: \t" << (int)device << " : " << VPUDevice_ToText.at(static_cast<int>(device)) << " ;\n";  //

        // inputs and outputs tensors
        {
            stream << " inputs: \t{\n";
            for (size_t i = 0; i < inputs.size(); i++) {
                stream << " input[" << i << "]: \t{\n" << inputs[i] << " } ;\n";
            }
            stream << "\t}inputs \n";
        }
        {
            stream << " outputs: \t{\n";
            for (size_t i = 0; i < outputs.size(); i++) {
                stream << " output[" << i << "]: \t{\n" << outputs[i] << " } ;\n";
            }
            stream << "\t}outputs \n";
        }

        return stream.str();
    };
};

inline std::ostream& operator<<(std::ostream& stream, const VPUNN::SHAVEWorkload& d) {
    stream << d.toString();
    return stream;
}

}  // namespace VPUNN

#endif  // VPUNN_TYPES_H
