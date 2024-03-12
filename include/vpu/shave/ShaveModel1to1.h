// Copyright © 2023 Intel Corporation
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

#include "vpu/cycles_interface_types.h"
#include "vpu/types.h"

namespace VPUNN {

constexpr int NOUNROLL{1};

/**
 * @brief Defines the structure of the first degree equation of the line for each shave operation.
 * The ecuation is slope_ * size + intercept_
 * @param slope_ is defined by time in us divided by size of output bytes
 * @param intercept_ is defined  by time in us
 */

struct FirstDegreeEquation {
    float slope_;      ///< Represents us/bytes
    float intercept_;  ///< Represents us
    /**
     * @brief Overload for operator() which calculates the time based on the first degree equation of the Shave
     * activation and the size in bytes of the output
     *
     * @param size of output in bytes
     * @return the time in us
     */
    float operator()(const int& size) const {
        return slope_ * size + intercept_;
    }
};

class ShaveModel1to1 {
private:
    const DataType dtype_;  ///< the data type of the input

    FirstDegreeEquation unroll;  ///< input for this eq is size in bytes,formed out of a slope and intercept that gives
                                 ///< the time in us for sw op based on the size at output

    const float offset_scalar_;  ///< inside the block, measures the height between the the %vector_size_ and the
                                 ///< following points in us

    const float offset_unroll_;  ///< special for first block having an offset from the rest of the blocks of data.
                                 ///< measured in us

    const unsigned int vector_size_;  ///< The number of ops for a VPUDevice Ex. 8 operations for VPU2.7
    const unsigned int unroll_size_;  ///< The number to determine what is the size of a block
    const unsigned int dpu_freq_;     ///< Measured in MHz, the Frequency at which the profiling was made.
    const unsigned int shv_freq_;     ///< Measured in MHz, the Frequency of a Activation Shave. Ex. Is the VPU_Freq in
                                      ///< case of the  VPU2.7

    const bool unroll_loop_;  ///< check if unroll exists
    const int block_size_;    ///< how many operations we have in a block, size in operations
public:
    /**
     * @brief Construct a new Shave Model 1 to 1 object
     */
    ShaveModel1to1(DataType dtype, float slope, float intercept, float offset_scalar, float offset_unroll,
                   unsigned int VectorSize, unsigned int UnrollSize, unsigned int DpuFreq, unsigned int ShvFreq)
            : dtype_(dtype),
              unroll{slope, intercept},
              offset_scalar_(offset_scalar),
              offset_unroll_(offset_unroll),
              vector_size_(VectorSize),
              unroll_size_(UnrollSize),
              dpu_freq_(DpuFreq),
              shv_freq_(ShvFreq),
              unroll_loop_(unroll_size_ > NOUNROLL),
              block_size_(unroll_loop_ ? vector_size_ * unroll_size_ : unroll_size_) {
    }

public:
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
    /**
     * @brief Get the time in us for the the activation based on the output size in bytes
     *
     * @return the time in us
     */
    float getMicroSeconds(const int& output_size_bytes) const {
        float y_model = unroll(output_size_bytes);  // size in bytes

        // commuting from bytes to operations
        // if i have a number of bytes and we know the data type then we know how many ops we made
        int output_size_operations = output_size_bytes / dtype_to_bytes(dtype_);

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
    /**
     * @brief Determines the number of cycles related to the profiling DPU freq, based on the size of output
     *
     * @return the number of cycles required based on CyclesInterfaceType
     */
    CyclesInterfaceType getDPUCycles(const int output_size_bytes) const {
        const float us = getMicroSeconds(output_size_bytes);
        const float raw_cycles = us * dpu_freq_;
        const auto cycles = Cycles::toCycleInterfaceType(raw_cycles);
        return cycles;
    }
    /**
     * @brief Determines the number of cycles related to the input value of a DPU freq given as a parameter based on the
     * size of output
     *
     * @return the number of cycles required based on CyclesInterfaceType
     */
    CyclesInterfaceType getDPUCycles(const int output_size_bytes, const int present_dpu_frq) const {
        const float us = getMicroSeconds(output_size_bytes);
        const float raw_cycles = us * present_dpu_frq;
        const auto cycles = Cycles::toCycleInterfaceType(raw_cycles);
        return cycles;
    }
    /**
     * @brief Determines the number of cycles related to the input value of a DPU freq and SHV freq given as a parameter
     * based on the size of output. In order to get accurate numbers we use a change factor based on the freq we made
     * profile and the given value for the profiling
     *
     * @return the number of cycles required based on CyclesInterfaceType
     */
    CyclesInterfaceType getDPUCycles(const int output_size_bytes, const int present_dpu_frq,
                                     const int present_shv_frq) const {
        const float us = getMicroSeconds(output_size_bytes);
        const float frq_change_factor = static_cast<float>(shv_freq_) / present_shv_frq;
        const float raw_cycles = (us * frq_change_factor) * present_dpu_frq;
        const auto cycles = Cycles::toCycleInterfaceType(raw_cycles);
        return cycles;
    }
};

}  // namespace VPUNN

#endif /* SHAVEMODEL1to1_H*/