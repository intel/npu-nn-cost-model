// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef SHAVEMODEL1to1_H
#define SHAVEMODEL1to1_H

#include <iostream>
#include <list>
#include <random>
#include <vector>

#include "shave_equations.h"
#include "shave_model_basics.h"
#include "vpu/cycles_interface_types.h"
#include "vpu/types.h"
#include "vpu/datatype_collection_size.h"


namespace VPUNN {

class ShaveModel1to1 : public ShaveCyclesProvider<ShaveModel1to1> {
private:
    const DataType dtype_;  ///< the data type of the output ()

    FirstDegreeEquation unroll;  ///< input for this eq is size in bytes,formed out of a slope and intercept that gives
                                 ///< the time in us for sw op based on the size at output

    const float offset_scalar_;  ///< inside the block, measures the height between the the %vector_size_ and the
                                 ///< following points in us

    const float offset_unroll_;  ///< special for first block having an offset from the rest of the blocks of data.
                                 ///< measured in us

    const unsigned int vector_size_;  ///< The number of ops for a VPUDevice Ex. 8 operations for VPU2.7
    const unsigned int unroll_size_;  ///< The number to determine what is the size of a block

    const bool unroll_loop_;  ///< check if unroll exists
    const int block_size_;    ///< how many operations we have in a block, size in operations

    friend std::ostream& operator<<(std::ostream& stream, const VPUNN::ShaveModel1to1& d);

public:
    /**
     * @brief Construct a new Shave Model 1 to 1 object
     */
    ShaveModel1to1(DataType dtype, float slope, float intercept, float offset_scalar, float offset_unroll,
                   unsigned int VectorSize, unsigned int UnrollSize, unsigned int DpuFreq, unsigned int ShvFreq)
            : ShaveCyclesProvider<ShaveModel1to1>{DpuFreq, ShvFreq},
              dtype_(dtype),
              unroll{slope, intercept},
              offset_scalar_(offset_scalar),
              offset_unroll_(offset_unroll),
              vector_size_(VectorSize),
              unroll_size_(UnrollSize),
              unroll_loop_(unroll_size_ > NOUNROLL),
              block_size_(unroll_loop_ ? vector_size_ * unroll_size_ : unroll_size_) {
    }

protected:
    /**
     * @brief Checks if the op_count is in the first block or not
     *
     * @param op_count numbers of ops required
     * @return rather is in the first block or not
     */
    bool is_in_first_block_of_operations(const int& op_count) const {
        if (unroll_loop_ && (op_count < block_size_)) {
            return true;
        }
        return false;
    }

    /**
     * @brief A rule that determines if we have to add scalar, when the given value is in the interior of the block
     * based on the number of operations
     *
     * @param op_count numbers of ops required
     * @return rather is in the first block or not
     */
    bool is_scalar_value(const int& op_count) const {
        if (op_count % vector_size_ != 0) {
            return true;
        }
        return false;
    }

    /**
     * @brief Determines if the given op_count is the first in the block or not
     * x_value = block_size first in block for loop unroll
     *
     * @param op_count numbers of ops required
     * @return rather is in the first block or not
     */
    bool is_first_value_in_block(const int& op_count) const {
        if (unroll_loop_ && ((op_count % block_size_) == 0)) {
            return true;
        }
        return false;
    }

public:
    /**
     * @brief Get the time in us for the the activation based on the output size in bytes
     *
     * @return the time in us
     */
    float getMicroSeconds(const int& output_size_bytes) const {
        float y_model = unroll(output_size_bytes);  // size in bytes

        // commuting from bytes to operations
        // if i have a number of bytes and we know the data type then we know how many ops we made
        //
        // there can be situations where the number of elements is not exact because, in the case of types that are
        // submultiples of 8 bits (e.g., INT1, INT2, INT4), a byte can hold multiple elements of these types (e.g., one
        // byte can store 4 INT2 elements or 2 INT4 elements). When the size in bytes is an odd number, it is possible
        // that the number of elements may not be exact
        int output_size_operations = compute_elements_count_from_bytes(output_size_bytes, dtype_);

        // Checks the rules for the special elements
        if (is_in_first_block_of_operations(output_size_operations)) {
            y_model -= offset_unroll_;
        }

        if (is_scalar_value(output_size_operations)) {
            y_model += offset_scalar_;
        }

        if (is_first_value_in_block(output_size_operations)) {
            y_model += offset_scalar_;
        }

        return y_model;
    }
};

inline std::ostream& operator<<(std::ostream& stream, const VPUNN::ShaveModel1to1& d) {
    stream << "ShaveModel1to1: \n"
           << " dtype: \t" << (int)d.dtype_ << " : " << DataType_ToText.at(static_cast<int>(d.dtype_)) << " ;\n"  //
           << " 1st deg equations: [slope,intercept]  \t{" << d.unroll.slope_ << "," << d.unroll.intercept_
           << "} ;\n"                                             //
           << " offset_scalar: \t" << d.offset_scalar_ << " ;\n"  //
           << " offset_unroll: \t" << d.offset_unroll_ << " ;\n"  //

           << " vector_size_: \t" << d.vector_size_ << " ;\n"  //
           << " unroll_size_: \t" << d.unroll_size_ << " ;\n"  //
           << " block_size_: \t" << d.block_size_ << " ;\n"    //

           << d.converter                            //
           << out_terminator() << "ShaveModel1to1 "  // terminator
            ;
    return stream;
}

/// TODO: Refactor idea in here, quite similar code
/// For NPU40 we observed a different behaviour for our simple models
/// This class have the role to modelate the new discovered behaviour
class ShaveModel1to1NPU40 : public ShaveCyclesProvider<ShaveModel1to1NPU40> {
private:
    const DataType dtype_;  ///< the data type of the output ()

    FirstDegreeEquation unroll;  ///< input for this eq is size in bytes,formed out of a slope and intercept that gives
                                 ///< the time in us for sw op based on the size at output

    const float offset_unroll_;  ///< special for first block having an offset from the rest of the blocks of data.
                                 ///< measured in us

    const float intra_block_offset_;  ///< inside a block each block has a base slope. This is basically de delta
                                      ///< between the lowest and the highest point in a block [(block_size -
                                      ///< vector_size + 1) + (block_size + 1)] measured in us

    const float vector_offset_;  ///< inside a block we have vector slopes for elements that are not fully in a vector
                                 ///< measured in us

    const int displacement_size_;  ///< This displacement shifts the vector blocks such as the lowest point in them to
                                   ///< be at a vector_size multiplier
    const unsigned int vector_size_;  ///< The number of ops for a VPUDevice Ex. 32 operations for NPU40
    const unsigned int unroll_size_;  ///< The number to determine what is the size of a block

    const bool unroll_loop_;  ///< check if unroll exists
    const int block_size_;    ///< how many operations we have in a block, size in operations

    friend std::ostream& operator<<(std::ostream& stream, const VPUNN::ShaveModel1to1NPU40& d);

public:
    /**
     * @brief Construct a new Shave Model 1 to 1 object
     */
    ShaveModel1to1NPU40(DataType dtype, float slope, float intercept, float offset_unroll, float intra_block_offset,
                        float vector_offset, unsigned int displacement_size, unsigned int VectorSize,
                        unsigned int UnrollSize, unsigned int DpuFreq, unsigned int ShvFreq)
            : ShaveCyclesProvider<ShaveModel1to1NPU40>{DpuFreq, ShvFreq},
              dtype_(dtype),
              unroll{slope, intercept},
              offset_unroll_(offset_unroll),
              intra_block_offset_(intra_block_offset),
              vector_offset_(vector_offset),
              displacement_size_(displacement_size),
              vector_size_(VectorSize),
              unroll_size_(UnrollSize),
              unroll_loop_(unroll_size_ >= NOUNROLL),
              block_size_(unroll_loop_ ? vector_size_ * unroll_size_ : unroll_size_) {
    }

protected:
    /**
     * @brief Checks if the op_count is in the first block or not
     *
     * @param op_count numbers of ops required
     * @return rather is in the first block or not
     */
    bool is_in_first_block_of_operations(const int& op_count) const {
        if (unroll_loop_ && (op_count <= block_size_)) {
            return true;
        }
        return false;
    }

    /**
     * @brief Calculate the time to be added based on the vector block position inside a block
     *
     * @param op_count numbers of ops required
     * @return time to be added based on position
     */
    float calculate_intra_block_offset(const int& op_count) const {
        float additional_time = 0;
        
        // taking the position in the block
        // Position is determined by the op count and in case of a displacement it is shifted to left
        // by the displacement size
        int block_position = (op_count - displacement_size_) % block_size_;

        // base position inside a block
        int intra_block_point = block_position / vector_size_;

        // this is the time addition based on the location in block
        // in case that we encounter division by 0 the additional time should remain 0
        if(unroll_size_ - 1 != 0)
            additional_time = (intra_block_offset_ * intra_block_point) / (unroll_size_ - 1);

        return additional_time;
    }

    /**
     * @brief Calculate the time to be added based on the position in vector block
     *
     * @param op_count numbers of ops required
     * @return time to be added based on position
     */
    float calculate_vector_offset(const int& op_count) const {
        float additional_time = 0;

        // taking the position in the block
        // Position is determined by the op count and in case of a displacement it is shifted to left
        // by the displacement size
        int block_position = (op_count - displacement_size_) % block_size_;

        // position inside a vectorial calculation
        int intra_vector_point = block_position % vector_size_;

        // this is the time addition based on the location in block
        // in case that we encounter division by 0 the additional time should remain 0
        if(vector_size_ - 1 != 0)
            additional_time = (vector_offset_ * intra_vector_point) / (vector_size_ - 1);

        return additional_time;
    }

public:
    /**
     * @brief Get the time in us for the the activation based on the output size in bytes
     *
     * @return the time in us
     */
    float getMicroSeconds(const int& output_size_bytes) const {
        float y_model = unroll(output_size_bytes);  // size in bytes

        // commuting from bytes to operations
        // if i have a number of bytes and we know the data type then we know how many ops we made
        //
        // there can be situations where the number of elements is not exact because, in the case of types that are
        // submultiples of 8 bits (e.g., INT1, INT2, INT4), a byte can hold multiple elements of these types (e.g., one
        // byte can store 4 INT2 elements or 2 INT4 elements). When the size in bytes is an odd number, it is possible
        // that the number of elements may not be exact
        int output_size_operations = compute_elements_count_from_bytes(output_size_bytes, dtype_);

        // Checks the rules for the special elements
        if (is_in_first_block_of_operations(output_size_operations)) {
            y_model -= offset_unroll_;
        }

        // Additioning the position in block
        // might have to make some checks in here, some ops dont have unroll
        if (unroll_loop_)
            y_model += calculate_intra_block_offset(output_size_operations);

        if (vector_size_ > 0)
            y_model += calculate_vector_offset(output_size_operations);

        return y_model;
    }
};

inline std::ostream& operator<<(std::ostream& stream, const VPUNN::ShaveModel1to1NPU40& d) {
    stream << "ShaveModel1to1NPU40: \n"
           << " dtype: \t" << (int)d.dtype_ << " : " << DataType_ToText.at(static_cast<int>(d.dtype_)) << " ;\n"  //
           << " 1st deg equations: [slope,intercept]  \t{" << d.unroll.slope_ << "," << d.unroll.intercept_
           << "} ;\n"                                                       //
           << " offset_unroll: \t" << d.offset_unroll_ << " ;\n"            //
           << " intra_block_offset: \t" << d.intra_block_offset_ << " ;\n"  //
           << " vector_offset: \t" << d.vector_offset_ << " ;\n"            //
           << " displacement: \t" << d.displacement_size_ << " ;\n"         //
           << " vector_size_: \t" << d.vector_size_ << " ;\n"               //
           << " unroll_size_: \t" << d.unroll_size_ << " ;\n"               //
           << " block_size_: \t" << d.block_size_ << " ;\n"                 //

           << d.converter                                 //
           << out_terminator() << "ShaveModel1to1NPU40 "  // terminator
            ;
    return stream;
}

}  // namespace VPUNN

#endif /* SHAVEMODEL1to1_H*/
