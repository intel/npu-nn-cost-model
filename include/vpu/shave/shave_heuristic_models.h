// Copyright © 2025 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the "Software Package")
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the "third-party-programs.txt" or other similarly-named text file included with the
// Software Package for additional details.

#ifndef SHAVE_HEURISTIC_MODELS_H
#define SHAVE_HEURISTIC_MODELS_H

#include <iostream>
#include <cmath>
#include "vpu/shave/shave_model_basics.h"
#include "vpu/cycles_interface_types.h"
#include "vpu/types.h"
#include "vpu/datatype_collection_size.h"

namespace VPUNN {

/**
 * @brief Simple heuristic model for non-vectorizable SHAVE operations
 * 
 * This model provides a basic cost estimation for inherently non-vectorizable kernels
 * (e.g., interpolate, elementwise gather, gridSample) using a simple 1 element/cycle
 * throughput with derates and entry cost.
 * 
 * Cost formula: cycles = entry_cost + (output_elements * cycles_per_element * overall_derate)
 * ATTENTION: Currently this model return SHAVE Cycles only, It will need adaptation to return DPU cycles.
 * Any negative values are handled at ctor level.
 */
class ShaveSimpleHeuristicModel {
private:
    const float elements_per_cycle_;      ///< Base cycles per element (default: 1.0)
    const float code_derate_;          ///< Overall kernel efficiency derate (default: 0.8 for 20% derate)
    const float bw_derate_;            ///< Bandwidth derate (default: 0.7 for 30% derate)
    const float entry_cost_cycles_;       ///< Entry cost in cycles (default: 2000)

    friend std::ostream& operator<<(std::ostream& stream, const VPUNN::ShaveSimpleHeuristicModel& d);

public:
    /**
     * @brief Construct a Simple Heuristic Model for non-vectorizable operations
     * 
     * @param elements_per_cycle Base throughput (default: 1 element/cycle)
     * @param overall_derate Overall efficiency derate multiplier (default: 0.8 for 20% derate)
     * @param entry_cost_cycles Fixed entry cost in cycles (default: 2000)

     */
    ShaveSimpleHeuristicModel(float elements_per_cycle = 1.0f,
                              float code_derate = 0.8f,
                              float bw_derate = 0.7f,
                              float entry_cost_cycles = 2000.0f)
              : elements_per_cycle_(elements_per_cycle > 0.0f ? elements_per_cycle : 1.0f),
              code_derate_(code_derate > 0.0f ? code_derate : 0.8f),
              bw_derate_(bw_derate > 0.0f ? bw_derate : 0.7f),
              entry_cost_cycles_(entry_cost_cycles > 0.0f ? entry_cost_cycles : 2000.0f) {
    }
public:
    /**
     * @brief Get the time in microseconds for the operation based on total elements
     * 
     * @param total_elements Number of total elements
     * @return Number of cycles
     */
    CyclesInterfaceType getCycles(const long long& total_elements) const {        
        // Calculate cycles: entry_cost + (elements * cycles_per_element * overall_derate)
        // Apply the overall derate multiplier directly as a factor on per-element cost
        const float total_cycles = entry_cost_cycles_ + (static_cast<float>(total_elements) / (elements_per_cycle_ * code_derate_ * bw_derate_));
        
        return static_cast<CyclesInterfaceType>(std::ceil(total_cycles));
    }
};

inline std::ostream& operator<<(std::ostream& stream, const VPUNN::ShaveSimpleHeuristicModel& d) {
    stream << "ShaveSimpleHeuristicModel: \n"
           << " elements_per_cycle: \t" << d.elements_per_cycle_ << " ;\n"
           << " code_derate: \t" << d.code_derate_ << " ;\n"
           << " bw_derate: \t" << d.bw_derate_ << " ;\n"
           << " entry_cost_cycles: \t" << d.entry_cost_cycles_ << " ;\n"
           << out_terminator() << "ShaveSimpleHeuristicModel ";
    return stream;
}

/**
 * @brief Advanced heuristic model for vectorizable SHAVE operations
 * 
 * This model provides cost estimation based on theoretical maximum throughput
 * (512 bits/cycle = 32 fp16 elements/cycle) with various bandwidth and compute
 * bound considerations.
 * 
 * Features:
 * - Bandwidth bound-ness reduction for broadcast and multi-input scenarios
 * - Compute throughput based on arithmetic operations per output
 * - BW derate of 30% (0.7 multiplier)
 * - Overall efficiency derate of 20% (1.2 multiplier)
 * - Entry cost of 2000 cycles
 * - Unaligned shape derate (2x for non-aligned inner dimensions)
 * ATTENTION: Currently this model return SHAVE Cycles only, It will need adaptation to return DPU cycles.
 * 
 * If negative values or zero are provided then it will call the simple heuristic model.
 * Any negative values are handled internally or at ctor level.
 */
class ShaveRooflineHeuristicModel {
private:
    const float arithmetic_ops_per_32_outputs_;     ///< Number of arithmetic operations per 32 outputs
    const float memory_ops_per_32_outputs_;         ///< Number of memory operations per 32 outputs
    const bool unaligned_by_nature_;                ///< Whether the kernel has inherently unaligned access
    const float bw_derate_;                         ///< Bandwidth derate (default: 0.7 for 30% derate)
    const float code_derate_;                       ///< Code efficiency derate (default: 0.8 for 20% derate)
    const float entry_cost_cycles_;                 ///< Entry cost in cycles (default: 2000)
    const float scalar_cost_per_channel_;           ///< Additional scalar cost per channel
    const float unalignment_derate_;                ///< Derate for unaligned shapes (default: 2.0)

    friend std::ostream& operator<<(std::ostream& stream, const VPUNN::ShaveRooflineHeuristicModel& d);

protected:
    /**
     * @brief Calculate cycles using simple heuristic (1 elem/cycle with derates)
     * 
     * @param total_size_in_bits Total size in bits
     * @param num_channels Number of channels for scalar cost
     * @param dtype_bits Bits per element (default: 16 for fp16)
     * @return Cycles using simple heuristic approach
     */
    CyclesInterfaceType calculateSimpleHeuristic(const long long& total_number_of_elements, 
                                                  const int num_channels) const {
        const float computation_cycles = total_number_of_elements / (code_derate_ * bw_derate_);
        const float scalar_cycles = scalar_cost_per_channel_ * num_channels;
        const float total_cycles = entry_cost_cycles_ + computation_cycles + scalar_cycles;
        return static_cast<CyclesInterfaceType>(std::ceil(total_cycles));
    }

    /**
     * @brief Normalize ops parameters handling zero/negative values
     * 
     * @param memory_ops Memory operations per 32 outputs (will be modified)
     * @param arithmetic_ops Arithmetic operations per 32 outputs (will be modified)
     * @return True if both were zero/negative (simple heuristic should be used)
     */
    bool normalizeOpsParameters(float& memory_ops, float& arithmetic_ops) const {
        // If both are zero or negative, signal to use simple heuristic
        if (memory_ops <= 0.0f && arithmetic_ops <= 0.0f) {
            return true;
        }
        
        // If one is zero but the other isn't, set both to the same value
        if (memory_ops <= 0.0f && arithmetic_ops > 0.0f) {
            memory_ops = arithmetic_ops;
        } else if (arithmetic_ops <= 0.0f && memory_ops > 0.0f) {
            arithmetic_ops = memory_ops;
        }
        
        return false;
    }

    /**
     * @brief Calculate effective throughput in bits/cycle using roofline model
     * 
     * @param memory_ops Memory operations per 32 outputs
     * @param arithmetic_ops Arithmetic operations per 32 outputs
     * @return Effective throughput in bits per cycle
     */
    float calculateEffectiveThroughput(float memory_ops, float arithmetic_ops) const {
        // Calculate BW and compute throughputs in fp16 elements per cycle
        const float bw_throughput = 32.0f / memory_ops;
        const float compute_throughput = 32.0f / arithmetic_ops;
        
        // Roofline: select minimum (bottleneck)
        const float effective_throughput = std::min(bw_throughput, compute_throughput);
        float effective_throughput_in_bits = effective_throughput * 16.0f; // bits per cycle
        
        // Apply unalignment derate if applicable
        if (unaligned_by_nature_) {
            effective_throughput_in_bits /= unalignment_derate_;
        }
        
        // Apply overall code efficiency and bandwidth derates
        effective_throughput_in_bits *= code_derate_ * bw_derate_;
        
        return effective_throughput_in_bits;
    }

    /**
     * @brief Calculate total cycles including entry cost, computation, and scalar costs
     * 
     * @param total_size_in_bits Total size in bits
     * @param effective_throughput Effective throughput in bits/cycle
     * @param num_channels Number of channels for scalar cost
     * @return Total cycles (rounded up)
     */
    CyclesInterfaceType calculateTotalCycles(const long long& total_size_in_bits,
                                              const float effective_throughput,
                                              const int num_channels) const {
        const float computation_cycles = static_cast<float>(total_size_in_bits) / effective_throughput;
        const float scalar_cycles = scalar_cost_per_channel_ * num_channels;
        const float total_cycles = entry_cost_cycles_ + computation_cycles + scalar_cycles;
        return static_cast<CyclesInterfaceType>(std::ceil(total_cycles));
    }

public:
    /**
     * @brief Construct a Roofline Heuristic Model for vectorizable operations
     * 
     * @param arithmetic_ops_per_32_outputs Number of vector arithmetic operations per 32 outputs
     * @param memory_ops_per_32_outputs Number of memory operations per 32 outputs
     * @param unaligned_by_nature Whether kernel has inherently unaligned access patterns
     * @param is_broadcast_scenario True if output shape = input shape (reduces to 16 elem/cc)
     * @param is_two_for_one_scenario True if 2 inputs → 1 output or 1 input → 2 outputs (512/3 bits/cc)
     * @param bw_derate Bandwidth derate factor (default: 0.7 for 30% reduction)
     * @param code_derate Overall code efficiency (default: 0.8 for 20% derate)
     * @param entry_cost_cycles Fixed entry cost (default: 2000 cycles)
     * @param scalar_cost_per_channel Additional per-channel scalar cost (default: 0)
     * @param unalignment_derate Derate for unaligned inner shape (default: 2.0)
     */
    ShaveRooflineHeuristicModel(float arithmetic_ops_per_32_outputs,
                                float memory_ops_per_32_outputs,
                                bool unaligned_by_nature = false,
                                float bw_derate = 0.7f,
                                float code_derate = 0.8f,
                                float entry_cost_cycles = 2000.0f,
                                float scalar_cost_per_channel = 0.0f,
                                float unalignment_derate = 2.0f)
            : arithmetic_ops_per_32_outputs_(arithmetic_ops_per_32_outputs),
              memory_ops_per_32_outputs_(memory_ops_per_32_outputs),
              unaligned_by_nature_(unaligned_by_nature),
              bw_derate_(bw_derate > 0.0f ? bw_derate : 0.7f),
              code_derate_(code_derate > 0.0f ? code_derate : 0.8f),
              entry_cost_cycles_(entry_cost_cycles > 0.0f ? entry_cost_cycles : 2000.0f),
              scalar_cost_per_channel_(scalar_cost_per_channel > 0.0f ? scalar_cost_per_channel : 0.0f),
              unalignment_derate_(unalignment_derate > 0.0f ? unalignment_derate : 2.0f) {
    }

    /**
     * @brief Get the number of cycles for the operation based on total size in bits
     * 
     * Uses roofline model: min(BW throughput, compute throughput) with derates.
     * Falls back to simple heuristic (1 elem/cycle) if both ops parameters are zero.
     * If only one parameter is zero, uses the other parameter's value for both.
     * 
     * @param total_size_in_bits Size of the output (or processed data) in bits
     * @param num_channels Number of channels (for per-channel scalar cost)
     * @param dtype_bits Bits per element (default: 16 for fp16)
     * @return Number of cycles (rounded up)
     */
    CyclesInterfaceType getCycles(const long long& total_size_in_bits,
                                  const long long& total_number_of_elements,
                                  const int num_channels = 0) const {
        float memory_ops = memory_ops_per_32_outputs_;
        float arithmetic_ops = arithmetic_ops_per_32_outputs_;
        
        // Check if we should use simple heuristic (both ops zero/negative)
        if (normalizeOpsParameters(memory_ops, arithmetic_ops)) {
            return calculateSimpleHeuristic(total_number_of_elements, num_channels);
        }

        // Calculate effective throughput using roofline model
        const float effective_throughput = calculateEffectiveThroughput(memory_ops, arithmetic_ops);

        // Calculate and return total cycles
        return calculateTotalCycles(total_size_in_bits, effective_throughput, num_channels);
    }
};

inline std::ostream& operator<<(std::ostream& stream, const VPUNN::ShaveRooflineHeuristicModel& d) {
    stream << "ShaveRooflineHeuristicModel: \n"
           << " arithmetic_ops_per_32_outputs: \t" << d.arithmetic_ops_per_32_outputs_ << " ;\n"
           << " memory_ops_per_32_outputs: \t" << d.memory_ops_per_32_outputs_ << " ;\n"
           << " unaligned_by_nature: \t" << d.unaligned_by_nature_ << " ;\n"
           << " bw_derate: \t" << d.bw_derate_ << " (BW efficiency: " 
           << (d.bw_derate_ * 100.0f) << "%) ;\n"
           << " code_derate: \t" << d.code_derate_ << " (code efficiency: " 
           << (d.code_derate_ * 100.0f) << "%) ;\n"
           << " entry_cost_cycles: \t" << d.entry_cost_cycles_ << " ;\n"
           << " scalar_cost_per_channel: \t" << d.scalar_cost_per_channel_ << " ;\n"
           << " unalignment_derate: \t" << d.unalignment_derate_ << " ;\n"
           << out_terminator() << "ShaveRooflineHeuristicModel ";
    return stream;
}

}  // namespace VPUNN

#endif  // SHAVE_HEURISTIC_MODELS_H
