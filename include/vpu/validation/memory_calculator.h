// Copyright © 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_VPU_VALIDATOR_MEMORY_CALCULATOR_H
#define VPUNN_VPU_VALIDATOR_MEMORY_CALCULATOR_H

#include <algorithm>
#include <iostream>
#include <sstream>  // for error formating

#include "data_dpu_operation.h"
#include "interface_valid_values.h"
#include "vpu/types.h"

namespace VPUNN {

/// @brief size of a DPU tensors in memory
struct MemorySize {
    long long cmx{0};             ///<  all inputs and outputs aligned
    long long input_0{0};         ///< activators size aligned
    long long input_1{0};         ///< weight size aligned
    long long output_0{0};        ///< output size aligned
    bool inplace_output{false};   ///< output_is in-place, does not contribute to the total memo need
    long long cmx_overhead{0};    ///< cmx overall overhead. e.g: runtime size (80K),hardware profile block (hwp) 16K
    bool ignore_overhead{false};  ///< overhead is not added to the total memory
};

// if placed outside namespace and into the test file it will not be observed by clang compiler
inline std::ostream& operator<<(std::ostream& stream, const VPUNN::MemorySize& d) {
    stream << "WL CMX MemorySize (aligned): \n"                                           //
           << " total: \t" << d.cmx << " ;\n"                                             //
           << " input_0: \t" << d.input_0 << " ;\n"                                       //
           << " input_1: \t" << d.input_1 << " ;\n"                                       //
           << " output_0: \t" << d.output_0 << " ;\n"                                     //
           << " inplace_output: \t" << (d.inplace_output ? "true" : "false") << " ;\n"    //
           << " cmx overhead: \t" << d.cmx_overhead << " ;\n"                             //
           << " ignore_overhead: \t" << (d.ignore_overhead ? "true" : "false") << " ;\n"  //
            ;
    return stream;
}

class WorkloadMemorySizeCalculator {
private:
    bool ignore_cmx_overhead{
            true};  ///< if true the cmx_memory_aligned_overhead will not be added to the total cmx memory

public:
    /// changes the state of ignore_cmx_overhead, that controls if overhead is added or not to final memory
    void set_ignore_cmx_overhead(bool new_state) {
        ignore_cmx_overhead = new_state;
    }

    /// @brief cmx memory in bytes , not considering broadcasting
    ///
    /// @param w is the workload for which the memory to be computed
    /// @param config knows device configurations and restrictions
    /// @returns memory information.
    MemorySize compute_memory(const DPUOperation& w, const IDeviceValidValues& config) const {
        const IOperationDynamicConstraints& operation_behaviour{
                config.get_specific_behaviour(w.operation)};  // might throw

        const auto in_0_size = operation_behaviour.input_0_volume(w.input_0);
        const auto in_0_aligned_size = config.compute_size_aligned(in_0_size, w.input_0.datatype);

        const auto in_1_size = operation_behaviour.input_1_volume(w.input_1);
        // const auto in_1_aligned_size = config.compute_size_aligned(in_1_size, w.input_1.datatype);
        const auto in_1_aligned_size = operation_behaviour.input_1_aligned_size_bytes(in_1_size, config, w);

        const auto out_0_size = operation_behaviour.output_0_volume(w.output_0);
        const auto out_0_aligned_size = config.compute_size_aligned(out_0_size, w.output_0.datatype);

        bool inplace_output{false};  //< ignore or no the output to the total memory sum

        // special case for in-place-output for ELEMENTWISE
        if (w.operation == Operation::ELTWISE) {
            inplace_output = true;
            // is this valid for all cases of ISI_strategy?
        }

        const long long cmx_sum = (ignore_cmx_overhead ? 0 : config.cmx_memory_aligned_overhead) +  //
                                  (in_0_aligned_size) +                                             //
                                  (in_1_aligned_size) +                                             //
                                  (inplace_output ? 0 : out_0_aligned_size);

        return MemorySize{cmx_sum,
                          in_0_aligned_size,
                          in_1_aligned_size,
                          out_0_aligned_size,
                          inplace_output,
                          config.cmx_memory_aligned_overhead,
                          ignore_cmx_overhead};
    }
};

}  // namespace VPUNN

#endif  //
