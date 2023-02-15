// Copyright © 2022 Intel Corporation
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
#include <numeric>
#include <vector>
#include "utils.h"

#include <iostream>
#include <map>
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

/// @creates and inverse map given a direct map (EnumMap)
inline const EnumInverseMap createInverseMap(const EnumMap& direct_map) {
    EnumInverseMap inverse_map;
    for (auto elem : direct_map) {
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

/// gives the EnumMap for a E enum type
/// has to be fully implemented for each type we want to cover
template <typename E>
inline const EnumMap& mapToText();

/// creates the  EnumInverseMap for a particular E enum type
/// @precondition the EnumMap<E> must exists
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

//
namespace Dim {
enum Grid { W, H };
enum Act { X, Y, Z, B };
enum Wt { K, C, Ky, Kx };
enum Padding { TOP, BOTTOM, LEFT, RIGHT };
}  // namespace Dim

/**
 * @brief Get the size of the dtype
 *
 * @param dtype a DataType object
 * @return unsigned int
 */
inline unsigned int dtype_to_bytes(DataType dtype) {
    switch (dtype) {
    case DataType::FLOAT16:
    case DataType::BFLOAT16:
        return 2;
    default:
        return 1;
    }
}

/**
 * @brief Get the tensor serial order given a layour
 *
 * @param layout a Tensor Layout
 * @return std::array<unsigned int, 4>
 */
inline std::array<unsigned int, 4> layout_to_order(Layout layout) {
    switch (layout) {
    case Layout::CMAJOR:
        return {0, 1, 2, 3};
    default:
        return {2, 0, 1, 3};
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
    std::array<unsigned int, 4> shape;  ///< the 4 dimensions of the real tensor
    DataType dtype;                     ///< datatatype of the described tensor
    Layout layout;                      ///< memory organization of the tensor
    bool sparsity;                      ///< is sparsity present?
    std::array<unsigned int, 4> strides;

    void compute_strides() {
        auto size = dtype_to_bytes(dtype);
        auto order = layout_to_order(layout);
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
     * @param layout VPUTensor layout
     */
    VPUTensor(const std::array<unsigned int, 4>& shape = {1, 1, 1, 1}, DataType dtype = DataType::UINT8,
              Layout layout = Layout::ZMAJOR, bool sparsity = false)
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
     * @param layout VPUTensor layout
     */
    VPUTensor(unsigned int width, unsigned int height, unsigned int channels, unsigned int batch, DataType dtype,
              Layout layout = Layout::ZMAJOR, bool sparsity = false)
            : VPUTensor({width, height, channels, batch}, dtype, layout, sparsity){};

    /**
     * @brief Get the VPUTensor x dimension
     *
     * @return unsigned int
     */
    unsigned int x() const {
        return shape[Dim::Act::X];
    };

    /**
     * @brief Get the VPUTensor y dimension
     *
     * @return unsigned int
     */
    unsigned int y() const {
        return shape[Dim::Act::Y];
    };

    /**
     * @brief Get the VPUTensor z dimension
     *
     * @return unsigned int
     */
    unsigned int z() const {
        return shape[Dim::Act::Z];
    };

    /**
     * @brief Get the VPUTensor batch dimension
     *
     * @return unsigned int
     */
    unsigned int b() const {
        return shape[Dim::Act::B];
    };

    /**
     * @brief Get the VPUTensor x dimension stride
     *
     * @return unsigned int
     */
    unsigned int sx() const {
        return strides[Dim::Act::X];
    };

    /**
     * @brief Get the VPUTensor y dimension stride
     *
     * @return unsigned int
     */
    unsigned int sy() const {
        return strides[Dim::Act::Y];
    };

    /**
     * @brief Get the VPUTensor z dimension stride
     *
     * @return unsigned int
     */
    unsigned int sz() const {
        return strides[Dim::Act::Z];
    };

    /**
     * @brief Get the VPUTensor batch dimension stride
     *
     * @return unsigned int
     */
    unsigned int sb() const {
        return strides[Dim::Act::B];
    };

    /**
     * @brief Get the VPUTensor height
     *
     * @return unsigned int
     */
    unsigned int height() const {
        return y();
    };

    /**
     * @brief Get the VPUTensor width
     *
     * @return unsigned int
     */
    unsigned int width() const {
        return x();
    };

    /**
     * @brief Get the VPUTensor channels
     *
     * @return unsigned int
     */
    unsigned int channels() const {
        return z();
    };

    /**
     * @brief Get the VPUTensor batches
     *
     * @return unsigned int
     */
    unsigned int batches() const {
        return b();
    };

    /**
     * @brief Get the VPUTensor size
     *
     * @return unsigned int
     */
    unsigned int size() const {
        return multiply_vector(shape) * dtype_to_bytes(dtype);
    }

    /**
     * @brief Check if the tensor is floating point
     *
     * @return true
     * @return false
     */
    bool is_float() const {
        switch (dtype) {
        case DataType::FLOAT16:
        case DataType::BFLOAT16:
            return true;
        default:
            return false;
        }
    }

    /**
     * @brief Check if the tensor is integer
     *
     * @return true
     * @return false
     */
    bool is_int() const {
        return !is_float();
    }

    /**
     * @brief Get the VPUTensor shape
     *
     * @return const std::array<unsigned int, 4>&
     */
    const std::array<unsigned int, 4>& get_shape() const {
        return shape;
    }

    /**
     * @brief Set the VPUTensor shape
     *
     * @param in_shape
     */
    void set_shape(std::array<unsigned int, 4> in_shape) {
        shape = in_shape;
        compute_strides();
    }

    /**
     * @brief Get the VPUTensor datatype
     *
     * @return DataType
     */
    DataType get_dtype() const {
        return dtype;
    }

    /**
     * @brief Get the VPUTensor layout
     *
     * @return Layout
     */
    Layout get_layout() const {
        return layout;
    }

    /**
     * @brief Get the VPUTensor sparsity flag
     *
     * @return sparsity flag
     */
    bool get_sparsity() const {
        return sparsity;
    }
};

/**
 * @brief The base structure that encodes a DPU workloads
 *
 */
struct DPUWorkload {
    /**
     * @brief The DPUWorkload device
     *
     */
    VPUDevice device;
    /**
     * @brief The DPUWorkload operation
     *
     */
    Operation op;

    /// @brief The DPUWorkload input0 tensors
    std::array<VPUTensor, 1> inputs;
    /// @brief The DPUWorkload input1 tensors
    // std::array<VPUTensor, 1> inputs_1; //postponed the introduction of weigths

    /**
     * @brief The DPUWorkload output tensors
     *
     */
    std::array<VPUTensor, 1> outputs;
    /**
     * @brief The DPUWorkload kernel sizes
     *
     */
    std::array<unsigned int, 2> kernels;
    /**
     * @brief The DPUWorkload kernel strides
     *
     */
    std::array<unsigned int, 2> strides;
    /**
     * @brief The DPUWorkload kernel padding
     *
     */
    std::array<unsigned int, 4> padding;
    /**
     * @brief The DPUWorkload execution mode
     *
     */
    ExecutionMode execution_order;
    /**
     * @brief The DPUWorkload operation activation function
     *
     */
    ActivationFunction activation_function = ActivationFunction::NONE;
    /**
     * @brief The DPUWorkload input activation sparsity
     *
     */
    float act_sparsity = 0;
    /**
     * @brief The DPUWorkload weight sparsity
     *
     */
    float weight_sparsity = 0;
    /**
     * @brief The DPUWorkload input tensors swizzling
     *
     */
    std::array<Swizzling, 2> input_swizzling = {Swizzling::KEY_0, Swizzling::KEY_0};
    /**
     * @brief The DPUWorkload output tensors swizzling
     *
     */
    std::array<Swizzling, 1> output_swizzling = {Swizzling::KEY_0};
    /**
     * @brief The DPUWorkload broadcast policy
     *
     */
    unsigned int output_write_tiles = 1;
    /**
     * @brief The DPUWorkload offsets relative to the parent DPULayer
     *
     */
    std::array<unsigned int, 4> offsets = {0, 0, 0, 0};
};

/**
 * @brief The base structure that encodes a DMA workloads
 *
 */
struct DMAWorkload {
    /**
     * @brief The VPU device
     *
     */
    VPUDevice device;
    /**
     * @brief The input tensor
     *
     */
    VPUTensor input;
    /**
     * @brief The output tensor
     *
     */
    VPUTensor output;
    /**
     * @brief Input memory location
     *
     */
    MemoryLocation input_location;
    /**
     * @brief Output memory location
     *
     */
    MemoryLocation output_location;
    /**
     * @brief The number of CMX tiles to broadcast
     *
     */
    unsigned int output_write_tiles = 1;

    /**
     * @brief This function computes the size of the DMAWorkload features to feed to the NN
     *
     * @return unsigned int
     */
    static unsigned int size() {
        // output_write_tiles size
        unsigned int size = 1;

        // Size of Device
        size += static_cast<int>(VPUDevice::__size);

        // Input + output tensor size
        size += 2 * (4 + static_cast<int>(DataType::__size) + static_cast<int>(Layout::__size));

        return size;
    }
};

/**
 * @brief The base structure that encodes a Software layer
 *
 */
struct SWOperation {
    VPUDevice device;                ///< The VPU device
    std::vector<VPUTensor> inputs;   ///< The input tensors
    std::vector<VPUTensor> outputs;  ///< The output tensors

    /// @brief ctor must exist since we have aggregate initialization possible on this type (abstract type)
    SWOperation(const VPUDevice& device, const std::vector<VPUTensor>& inputs, const std::vector<VPUTensor>& outputs)
            : device{device}, inputs{inputs}, outputs{outputs} {
    }

    /**
     * @brief Return the number of cycles of the sw operation
     *
     * @return unsigned int
     */
    virtual unsigned int cycles() = 0;

    /**
     * @brief Destroy the SWOperation object
     *
     */
    virtual ~SWOperation() = default;
};

/**
 * @brief Radom sample from the vector with uniform distribution
 *
 * @tparam T usually a numerics type
 * @param vector a std::vector of T
 * @return T a random sample from the vector
 */
template <class T>
T sample(const std::vector<T>& vector) {
    auto idx = std::rand() % vector.size();
    return vector[idx];
}

/**
 * @brief Random sample from a class enum type with uniform distribution
 *
 * @tparam T an enum type
 * @return T a random sample from the enum
 */
template <class T>
T randomEnum() {
    auto idx = std::rand() % static_cast<int>(T::__size);
    return static_cast<T>(idx);
}

/**
 * @brief A structure to generate random DPU workloads
 * @details Useful to generate random DPU workloads
 * Example: VPUNN::randDPUWorkload(VPUNN::VPUDevice::VPU_2_0)
 *
 */
struct randDPUWorkload {
    /**
     * @brief randDPUWorkload VPUDevice
     *
     */
    VPUNN::VPUDevice device;

public:
    /**
     * @brief Construct a new random DPUWorkload object
     *
     * @param device a VPUDevice object
     */
    randDPUWorkload(VPUNN::VPUDevice device): device(device) {
    }

    /**
     * @brief overloaded operator () to enable calling the struct
     *
     * @return VPUNN::DPUWorkload
     */
    VPUNN::DPUWorkload operator()() {
        std::vector<unsigned int> random_dim = {1, 4, 7, 16, 32, 64, 128};
        std::vector<unsigned int> random_channels = {1, 16, 32, 64, 128};
        std::vector<unsigned int> random_kernels = {1, 3};
        std::vector<unsigned int> random_strides = {1, 3};
        std::vector<ExecutionMode> vpu_2_0_modes = {ExecutionMode::MATRIX, ExecutionMode::VECTOR};
        std::vector<ExecutionMode> vpu_2_7_modes = {ExecutionMode::CUBOID_16x16, ExecutionMode::CUBOID_4x16,
                                                    ExecutionMode::CUBOID_8x16};

        auto op = randomEnum<Operation>();
        unsigned int ic, oc, kx, ky;
        if (op == Operation::ELTWISE || op == Operation::DW_CONVOLUTION || op == Operation::MAXPOOL ||
            op == Operation::AVEPOOL) {
            ic = oc = sample(random_channels);
        } else {
            ic = sample(random_channels);
            oc = sample(random_channels);
        }
        auto width = sample(random_channels), height = sample(random_channels);

        if (op == Operation::ELTWISE) {
            kx = ky = 1;
        } else {
            kx = sample(random_kernels);
            ky = sample(random_kernels);
        }

        auto mode = device == VPUDevice::VPU_2_0 ? sample(vpu_2_0_modes) : sample(vpu_2_7_modes);

        auto input_tensor = VPUTensor({width, height, ic, 1}, randomEnum<DataType>());
        // auto input_1_tensor = VPUTensor({width, height, ic, 1}, randomEnum<DataType>());
        auto output_tensor = VPUTensor({width, height, oc, 1}, randomEnum<DataType>());

        return DPUWorkload({device,
                            op,
                            {input_tensor},
                            //{input_1_tensor},
                            {output_tensor},
                            {kx, ky},
                            {1, 1},
                            {kx / 2, kx / 2, ky / 2, ky / 2},
                            mode});
    }
};

}  // namespace VPUNN

#endif  // VPUNN_TYPES_H
