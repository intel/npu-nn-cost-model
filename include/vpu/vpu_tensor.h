// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_VPU_TENSOR_H
#define VPUNN_VPU_TENSOR_H

#include <array>
#include <iostream>
#include <map>
#include <sstream>  //
#include <string>

#include "dpu_types.h"
#include "dpu_types_info.h"
#include "utils.h"
#include "vpu/datatype_collection_size.h"

namespace VPUNN_unit_tests {
class VPUTensorTest;  // forward declaration
}
namespace VPUNN {

/**
 * @brief Cost model tensor class
 *
 */
class VPUTensor {
private:
    std::array<unsigned int, 4> shape;  ///< the 4 dimensions of the real tensor in the order Dim::Act XYZB  WHCB
    DataType dtype;                     ///< datatatype of the described tensor
    Layout layout;                      ///< memory organization of the tensor
    bool sparsity;                      ///< is sparsity present?

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
            : shape(shape),
              dtype(dtype),
              layout(layout),
              sparsity(sparsity){
                      // throw_if_invalid();
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

    /// @brief Get the volume in number of samples (values)
    /// @return number of samples
    unsigned int volume() const {
        return multiply_vector(shape);
    }

    /// @brief Check if the tensor is floating point
    /// @return true if floating point type
    bool is_float() const noexcept {
        switch (dtype) {
        case DataType::FLOAT32:
        case DataType::FLOAT16:
        case DataType::BFLOAT16:
        case DataType::BF8:
        case DataType::HF8:
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
    }

    /// @brief Get the datatype
    DataType get_dtype() const noexcept {
        return dtype;
    }

    /// @brief changes the underlying data type only if same size new vs old
    /// @returns newly set type.
    DataType change_datatype_superficial(DataType new_datatype) {
        if (is_same_datatype_footprint(new_datatype, get_dtype())) {
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

    /// @brief sets the sparsity flag
    void set_sparsity(bool sparsity_new) noexcept {
        sparsity = sparsity_new;
    }

    /// @brief Get the size in bytes based on packmode
    /// @return size in bytes
    unsigned int size() const {
        if constexpr (PackMode::pack_mode_0 == packmode) {
            return size_packmode_0();
        } else if constexpr (PackMode::pack_mode_1 == packmode) {
            return size_packmode_1();
        } else if constexpr (PackMode::pack_mode_2 == packmode) {
            return size_packmode_2();
        } else if constexpr (PackMode::pack_mode_3 == packmode) {
            return size_packmode_3();
        }
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
    friend bool operator<(const VPUNN::VPUTensor& lhs, const VPUNN::VPUTensor& rhs);

private:
    /// overflowBits can take values in interval: [1, 8] and represents the number of bits occupied by a data type in
    /// the last byte
    ///
    /// overflowBits can not be smaller that 1 (have value of 0) because that would mean the data type doesn't occupy
    /// any bits in the next byte of memory
    /// example: an INT8 occupies a whole byte, so we do not move to the next byte, but an INT9 occupies a whole byte
    /// plus one bit from the next byte, meaning overflowBits will be 1
    ///
    /// overflowBits can not be greater than 8, as 8 bits equal one byte
    /// example: an INT8 occupies a whole byte, since overflowBits checks how many bits are occupied in the last byte, x
    /// will be 8, same if type is FLOAT16, it occupies 2 bytes in memory, meaning the last byte will also have all 8
    /// bits occupied 
    /// @return true if overflowBits is in interval [1, 8] , false if not 
    bool isOverflowBitsValid(const int overflowBits) const noexcept {
        if (overflowBits < 1 || overflowBits > 8)
            return false;
        return true;
    }

    /**
     * @brief we compute the size in bytes and number of tensor's elements that we can place in memory one after the
     * other so that an element cannot span more bytes than the number indicated by its size in bytes
     *
     * example:
     * 1) if dtype is INT13 we can place correctly one element in memory, still remain 3 bites unoccupied from the
     * last byte, but when we try to place another element right after the first one, this element will occupy the 3
     * bits left from the first element (the first byte on which this element extends), then a whole byte (the second
     * byte on which the element extends) and needs 2 more bits that it will try to occupies them in the third byte, but
     * the size type of this element is 2B => we cannot place several such elements one after the other in memory
     *
     * 2) if dtype is INT12 we can place 2 elements one after the other in memory, because such an element occupies a
     * whole byte and 4 bits from another byte, the next element placed in memory will occupy the 4 bits left from the
     * previous element and another whole byte => the second element does not span more than 2B as indicated by its size
     * type
     *
     * @param elem represent the innermost dimension or the total number of elements for a tensor
     *
     * @return the number and size in bytes of elements that can be placed one after the other in memory without an
     * element exceeding its type dimension in bytes
     */
    std::pair<const int, const int> computeContiguousElementCountAndSize() const {
        // the minimum number of bytes in which we can store the dtype
        const int type_dimension_B = static_cast<int>(dtype_to_bytes(dtype));

        // number of bits occupied in the last byte of the type dimension
        // example: if type_dim=1 and dtype=INT3 => there is 1 byte and only 3 bits occupied
        //          if type_dim=2 and dtype=INT12 => there is 2 bytes and only 4 bits occupied from the second byte
        const int overflowBits = dtype_to_bits(dtype) - 8 * (type_dimension_B - 1);

         if (!isOverflowBitsValid(overflowBits))
            return std::make_pair(-1, -1);

        // the number of elements that can be placed in memory one after the other without an element spanning more
        // bytes than type_dimension_B indicates
        const int contiguousSeq_elem = 8 / overflowBits;

        // the size in bytes for such a contiguous sequence
        const int bytes_per_completeSeq = contiguousSeq_elem * (type_dimension_B - 1) + 1;

        return std::make_pair(contiguousSeq_elem, bytes_per_completeSeq);
    }

    /*
     * @brief compute the size in bytes of the total number of sequences formed by elements that can be placed one after
     * the other in memory without an element extending over more bytes than it would normally occupy
     *
     * @param elem represent the innermost dimension or the total number of elements for a tensor
     *
     * @return the size in bytes of all aligned sequences that can be formed
     */
    int computeAlignedSequencesSize_B(const int elem) const {
        std::pair<const int, const int> seq_info = computeContiguousElementCountAndSize();
        const int contiguousSeq_elem = seq_info.first;
        const int bytes_per_completeSeq = seq_info.second;

        // how many complete sequences do we have
        const int completeSequences = elem / contiguousSeq_elem;

        // size in bytes
        const int completeSequences_Bytes = completeSequences * bytes_per_completeSeq;

        return completeSequences_Bytes;
    }

    /*
     * @brief compute the size in bytes of remaining elements that do not form a complete sequence
     *
     * Complete sequence meaning:
     * 1. A complete sequence refers to the maximum number of elements that can be arranged
     * contiguously in memory such that no element within this sequence spans more bytes
     * than it would normally occupy.
     * 2. A complete sequence must contain a number of elements that maximizes the utilization
     * of available memory space.
     *
     * example: if dtype is INT3 or INT12 a complete sequence is formed by 2 elements
     *          if dtype is INT13 or INT7 a complete sequence is formed by 1 element
     *          id dtype is INT2 a complete sequence is formed by 4 elements
     *
     *
     * @param elem represent the innermost dimension or the total number of elements for a tensor
     * @param contiguousSeq_elm is the number of elements that can be placed in memory one after the other without an
     * element spanning more bytes than normally occupies
     *
     * @return size of remaining elements in bytes
     */
    int sizeOfRemaininElem_B(const int elem, const int contiguousSeq_elm) const {
        const int remainingElem = elem % contiguousSeq_elm;
        const int size_of_remainingElem_Bytes =
                ((remainingElem == 0) ? (0) : ((remainingElem * dtype_to_bits(dtype)) / 8 + 1));

        return size_of_remainingElem_Bytes;
    }

    /// a sample cannot span a number of bytes larger than what it normally occupies, a byte cannot
    /// contain 2 dimensions
    unsigned int size_packmode_0() const {
        const auto order = layout_to_order(layout);
        const auto innermost_dimension{static_cast<int>(shape[order[0]])};

        std::pair<const int, const int> seq_info = computeContiguousElementCountAndSize();
        const int contiguousSeq_elm = seq_info.first;
        const int completeSequences_Bytes = computeAlignedSequencesSize_B(innermost_dimension);

        // remaining elements that do not form a complete sequence
        const int size_of_remainingElem_Bytes = sizeOfRemaininElem_B(innermost_dimension, contiguousSeq_elm);

        // innermost dimension size in bytes
        const int innmermost_dim_size_B = completeSequences_Bytes + size_of_remainingElem_Bytes;

        // size for entire tensor
        const int size = innmermost_dim_size_B * shape[order[1]] * shape[order[2]] * shape[order[3]];

        return static_cast<unsigned int>(size);
    }

    /// a sample cannot span a number of bytes larger than what it normally occupies, a byte can
    /// contain 2 dimensions
    unsigned int size_packmode_1() const {
        const int elements_count = static_cast<int>(multiply_vector(shape));

        std::pair<const int, const int> seq_info = computeContiguousElementCountAndSize();
        const int contiguousSeq_elm = seq_info.first;
        const int completeSequences_Bytes = computeAlignedSequencesSize_B(elements_count);

        // remaining elements that do not form a complete sequence
        const int size_of_remainingElem_Bytes = sizeOfRemaininElem_B(elements_count, contiguousSeq_elm);

        // size for entire tensor
        const int size = completeSequences_Bytes + size_of_remainingElem_Bytes;

        return static_cast<unsigned int>(size);
    }

    int transform_bits_to_bytes(const int elem) const {
        const int type_in_bits = static_cast<int>(dtype_to_bits(dtype));

        const int size_in_bits = elem * type_in_bits;
        const int size = (size_in_bits % 8 == 0) ? size_in_bits / 8 : (size_in_bits / 8) + 1;

        return size;
    }

    /// a sample can span a number of bytes larger than what it normally occupies, a byte cannot
    /// contain 2 dimensions
    unsigned int size_packmode_2() const {
        const auto order = layout_to_order(layout);
        const auto innermost_dimension{shape[order[0]]};

        // size in bytes for innermost dimension
        const int innermost_dim_Bytes = transform_bits_to_bytes(innermost_dimension);

        // size in bytes for entire tensor
        const int size = innermost_dim_Bytes * shape[order[1]] * shape[order[2]] * shape[order[3]];

        return static_cast<unsigned int>(size);
    }

    /// a sample can span a number of bytes larger than what it normally occupies, a byte can
    /// contain 2 dimensions
    unsigned int size_packmode_3() const {
        ;
        const int elements_count = static_cast<int>(multiply_vector(shape));

        // size in bytes for entire tensor
        const int size = transform_bits_to_bytes(elements_count);
        return static_cast<unsigned int>(size);
    }

    bool is_tensor_valid_packmode_0() const {
        const auto order = layout_to_order(layout);

        // if the data type is less than 8 bits, the elements from innermost dimension should be able to occupy the
        // maximum number of bytes used
        // @example:
        // BAD CASE: Dtype: INT4 innermost dim: 1 => 4 bits of a byte would be occupied, leaving space for another
        // element of this type in the byte => the space has not been maximally occupied with the number of possible
        // elements
        //
        // GOOD CASE: Dtype: INT4 innermost dim: 2 => 8 bits of a byte would be occupied BAD CASE: Dtype: INT4
        // innermost dim: 3 => 1 byte would be fully occupied, but just 4 bits of the second one would be occupied
        //
        //
        // BAD CASE: Dtype: INT3 innermost dim: 1 => 3 bits of a byte would be occupied, but 2 elements of this type fit
        // in one byte (5 bits remain unoccupied, allowing for another element of this type to fit)
        //
        // GOOD CASE: Dtype: INT3 innermost dim: 2 => 6 bits would be occupied, 2 bits remain unoccupied, but another
        // element of this type cannot fit in the byte, a maximum of 2 elements can fit in one byte BAD CASE: Dtype:
        // INT3 innermost dim: 3 => 6 bits of the first byte are occupied, 2 remain unoccupied, but just 3 bits of the
        // second byte are occupied, there still is enough space for another element
        //
        // if data type is larger than 8 bits, then only one element of this type, fits in 2 bytes
        // @example:
        // INT12 occupies 12 bits out of 2 bytes, leaving 6 bits free, but another element of this type cannot fit
        // =>types larger than 8 bits will always occupy 2 bytes in memory
        const auto innermost_dimension{shape[order[0]]};
        const int type_dimension{dtype_to_bytes(dtype)};

        if (type_dimension < 1)
            return false;

        // number of bits occupied in the last byte of the type dimension
        // example: if type_dim=1 and dtype=INT3 => there is 1 byte and only 3 bits occupied
        //          if type_dim=2 and dtype=INT12 => there is 2 bytes and only 4 bits occupied from the second byte
        const int overflowBits =
                dtype_to_bits(dtype) - 8 * (type_dimension - 1);

         if (!isOverflowBitsValid(overflowBits))
            return false;

        // number of elements of that type that you can place in memory one after one without exceeding the allowed
        // number of bytes an element can span example: INT4, INT2, INT1, INT3 cannot fit in 2 bytes, max 1 INT12,
        // INT13, INT16 cannot fit in 3 bytes, max 2
        const int contiguousCapacity = 8 / overflowBits;

        //  remaining elements that do not form a complete cycle
        const int remainingElem = innermost_dimension % contiguousCapacity;
        if (remainingElem != 0)
            return false;

        return true;
    }

    // @example:
    // GOOD CASE: Dtype: INT4, tensor size: 1x2x2x1 => 4 elements that will occupy 2bytes
    //
    // BAD CASE: Dtype: INT4, tensor size: 3x5x1x1 => 15 elements that will occupy 7.5 bytes there is still enough space
    // for another INT4 element => in this case, we did not maximize the space
    //
    //
    // GOOD CASE: Dtype: INT3, tensor size: 1x1x4x1  4 elements that will occupy 2bytes, still remain 2 bites from each
    // byte unoccupied, because in 1 byte fits 2 INT3 elements
    //
    // BAD CASE: Dtype: INT3, tensor size: 1x1x3x1  3 elements that will occupy an entire byte and 3 bits from the next
    // one, but in the second byte would still be enough space for another element => the space has not been fully
    // occupied
    //
    // throw an exception in the constructor if the condition is not met
    bool is_tensor_valid_packmode_1() const {
        const auto elements_count = multiply_vector(shape);
        const int type_dimension = dtype_to_bytes(dtype);

        if (type_dimension == -1)
            return false;

        // number of bits occupied in the last byte of the type dimension
        // example: if type_dim=1 and dtype=INT3 => there is 1 byte and only 3 bits occupied
        //          if type_dim=2 and dtype=INT12 => there is 2 bytes and only 4 bits occupied from the second byte
        const int overflowBits = dtype_to_bits(dtype) - 8 * (type_dimension - 1); 

         if (!isOverflowBitsValid(overflowBits))
            return false;

        // number of elements of that type that you can place in memory one after one without exceeding the allowed
        // number of bytes an element can span example: INT4, INT2, INT1, INT3 cannot fit in 2 bytes, max 1 INT12,
        // INT13, INT16 cannot fit in 3 bytes, max 2
        const int contiguousCapacity = 8 / overflowBits;

        //  remaining elements that do not form a complete cycle
        const int remainingElem = elements_count % contiguousCapacity;
        if (remainingElem != 0)
            return false;

        return true;
    }

    // @example:
    // layout in example will be ZXY
    // GOOD CASE: Dtype: INT4, tensor size: 1x2x2x1 => innermost dim is 2 => 2 elements will occupy 1byte
    //
    // BAD CASE: Dtype: INT4, tensor size: 1x2x3x1 => innermost dim is 3 => 3 elements will occupy 1 byte and 4 bits
    // from the next one, but there are still 4 bits unoccupied => enough space to put another INT4 element
    //
    //
    // GOOD CASE: Dtype: INT3, tensor size: 1x2x5x1 => innermost dim is 5 => the first 2 elements will occupy 6 bits of
    // the first byte, the 3rd will occupy 2 bites of the first byte and 1 bit of the second byte, and the last 2
    // elements will occupy 6 bits of the second byte, but there is no enough space to place another INT3 element
    //
    // BAD CASE: Dtype: INT3, tensor size: 1x2x4x1 => innermost dim is 4 => the first 2 elements will occupy 6 bits of
    // the first byte, the 3rd will occupy 2 bites of the first byte and 1 bit of the second byte, and the last
    // element will occupy 3 bits of the second byte, but there is enough space (4 bits) to place another INT3 element
    bool is_tensor_valid_packmode_2() const {
        const auto order = layout_to_order(layout);
        const auto innermost_dimension{shape[order[0]]};

        const auto occupied_bits =
                (innermost_dimension * dtype_to_bits(dtype)) % 8;  // occupied bits from the last byte
        const auto unoccupied_bits =
                (occupied_bits == 0) ? 0 : (8 - occupied_bits);  // unoccupied bits from the last byte

        // we verify if there is still enough space in the last byte to put another element
        if ((unoccupied_bits / dtype_to_bits(dtype)) != 0) {
            return false;
        }

        return true;
    }

    bool is_tensor_valid_packmode_3() const {
        const auto elements_count = multiply_vector(shape);

        const auto occupied_bits = (elements_count * dtype_to_bits(dtype)) % 8;  // occupied bits from the last byte
        const auto unoccupied_bits =
                (occupied_bits == 0) ? 0 : (8 - occupied_bits);  // unoccupied bits from the last byte

        // we verify if there is still enough space in the last byte to put another element
        if ((unoccupied_bits / dtype_to_bits(dtype)) != 0) {
            return false;
        }

        return true;
    }

    void throw_if_invalid(void) const {
        if constexpr (PackMode::pack_mode_0 == packmode) {
            if (!is_tensor_valid_packmode_0()) {
                throw std::invalid_argument("Invalid tensor, Packmode 0");
            }
        }
        if constexpr (PackMode::pack_mode_1 == packmode) {
            if (!is_tensor_valid_packmode_1()) {
                throw std::invalid_argument("Invalid tensor, Packmode 1");
            }
        }
        if constexpr (PackMode::pack_mode_2 == packmode) {
            if (!is_tensor_valid_packmode_2()) {
                throw std::invalid_argument("Invalid tensor, Packmode 2");
            }
        }
        if constexpr (PackMode::pack_mode_3 == packmode) {
            if (!is_tensor_valid_packmode_3()) {
                throw std::invalid_argument("Invalid tensor, Packmode 3");
            }
        }
    }

    enum class PackMode {
        pack_mode_0,  // a sample cannot span a number of bytes larger than what it normally occupies, a byte cannot
                      // contain 2 dimensions
        pack_mode_1,  // a sample cannot span a number of bytes larger than what it normally occupies, a byte can
                      // contain 2 dimensions
        pack_mode_2,  // a sample can span a number of bytes larger than what it normally occupies, a byte cannot
                      // contain 2 dimensions
        pack_mode_3,  // a sample can span a number of bytes larger than what it normally occupies, a byte can contain 2
                      // dimensions
    };

    static constexpr PackMode packmode{
            PackMode::pack_mode_0};  // 0 means no sample can go across byte boundary. eg INT3 is 2 per byte and 2 bits
                                     // free.Also a dimension cannot share a byte with another dimension (lines must be
                                     // full)
    friend class ::VPUNN_unit_tests::VPUTensorTest;
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
            ;
    return stream;
}

inline bool operator<(const VPUNN::VPUTensor& lhs, const VPUNN::VPUTensor& rhs) {
    // lexicographical_compare style
    {  // shape
        if (lhs.shape < rhs.shape)
            return true;
        if (rhs.shape < lhs.shape)
            return false;
    }

    {  // dtype
        if (lhs.dtype < rhs.dtype)
            return true;
        if (rhs.dtype < lhs.dtype)
            return false;
    }
    {  // sparsity
        if (lhs.sparsity < rhs.sparsity)
            return true;
        if (rhs.sparsity < lhs.sparsity)
            return false;
    }
    return false;  // all are  no smaller or no larger than other
}

}  // namespace VPUNN

#endif  // VPUNN_TYPES_H
