// Copyright © 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_DPU_OPERATIONS_VALIDATOR_H
#define VPUNN_DPU_OPERATIONS_VALIDATOR_H

#include <algorithm>

#include <sstream>  // for error formating
#include <stdexcept>

#include <iostream>

#include "vpu/types.h"

#include <tuple>
#include "behaviors_and_devices_containers.h"
#include "device_valid_values.h"
#include "dpu_operations_valid_behaviours.h"
#include "interface_valid_values.h"
#include "memory_calculator.h"
#include "sanity_report.h"

namespace VPUNN {

/// @brief reunites the dynamic behaviors for known operations
/// workload level operations dynamic behavior
using OperationsBehaviour = Behaviours<CONVOLUTION_Constraints, DW_CONVOLUTION_Constraints, CM_CONVOLUTION_Constraints,
                                       ELTWISE_Constraints, MAXPOOL_Constraints>;

template <class BehaviorsContext>
class ContextualMemoryCalculator {
protected:
    BehaviorsContext& context_ref;                     ///< reference to configuration
    WorkloadMemorySizeCalculator memory_calculator{};  ///< used for memory calculation, ignore cmx overhead by default

    ContextualMemoryCalculator(BehaviorsContext& external_context): context_ref(external_context) {
    }

public:
    /// @brief cmx memory in bytes
    ///
    /// @param w is the workload for which the memory to be computed, has to be completely populated (also input_1)
    /// @throws runtime_error if the operation from the workload is not supported or known
    /// @returns memory information. In case the device is not supported will contain -1 or negative values in structure
    MemorySize compute_wl_memory(const DPUOperation& w) const {
        if (!context_ref.is_supported(w.device)) {
            return MemorySize{-1};  // or maybe throw?
        }

        const auto& config = context_ref.get_config(w.device);
        return memory_calculator.compute_memory(w, config);
    }

    /// @brief cmx memory in bytes
    ///
    /// @param wl is the workload for which the memory to be computed
    /// @throws runtime_error if the operation from the workload is not supported or known
    /// @returns memory information. In case the device is not supported will contain -1 or negative values in structure
    MemorySize compute_wl_memory(const DPUWorkload& wl) const {
        DPUOperation op(wl);

        if (!context_ref.is_supported(op.device)) {
            return MemorySize{-1};  // or maybe throw?
        }
        const auto& config = context_ref.get_config(op.device);
        const auto& operation_behaviour = config.get_specific_behaviour(op.operation);  // might throw

        operation_behaviour.deduce_input_1(op.input_0, op.output_0, config, op.kernel, op.input_1);

        return memory_calculator.compute_memory(op, config);
    }
};

/// @brief provides services for Workload validation,
template <class OperationsConfiguration>
class DPU_ConfigurableOperationValidator :
        public OperationsConfiguration,
        public ContextualMemoryCalculator<OperationsConfiguration> {
protected:
    //    ContextualMemoryCalculator<OperationsConfiguration> memo{*(this)};

public:
    DPU_ConfigurableOperationValidator()
            : ContextualMemoryCalculator<OperationsConfiguration>((static_cast<OperationsConfiguration&>(
                      *this)))  // only dereferencing derived will give errors , null in context
    {
    }

    /// @brief constructs the VPUTensor for input_1, deduced from workload context
    /// assumes the workload is sanitized, will not do extra checks , but will throw
    /// @param wl the workload for which the input 1 (weights) will be computed
    /// @throws runtime_error if the operation is not supported or known
    ///
    /// @returns the VPUTensor describing input_1
    VPUTensor construct_input_1(const DPUWorkload& wl) const {
        DPUOperation op(wl);

        const auto& config = this->get_config(op.device);
        auto& operation_behaviour = config.get_specific_behaviour(op.operation);  // may throw

        operation_behaviour.deduce_input_1(op.input_0, op.output_0, config, op.kernel, op.input_1);
        // sparsity and swizzling set in constructor
        // weights sparsity should not be present  if not enabled (assumed sanitation)

        auto& in = op.input_1;
        const VPUTensor weights =
                VPUTensor({static_cast<unsigned int>(in.width), static_cast<unsigned int>(in.height),
                           static_cast<unsigned int>(in.channels), static_cast<unsigned int>(in.batch)},
                          config.restrict_datatype(in.datatype), in.layout, in.sparsity_enabled);

        return weights;
    }

    /// validates if this workload (DPUOperation) is valid, results are provided in results param.
    /// contextual rules are provided from outside
    void check_workload_consistency(const DPUOperation& w, const IDeviceValidValues& config,
                                    const IOperationDynamicConstraints& operation_behaviour,
                                    SanityReport& result) const {
        result.resetOK();  // all OK

        Checker checker;
        try {
            checker.check_is_in_list(w.device, config.devices, "Device");
            checker.check_is_in_list(w.operation, config.get_valid_operations_range(), "Operation");

            checker.check_is_in_list(w.output_write_tiles, config.output_write_tile_options, "output_write_tiles");
            // dep on out tile and op
            checker.check_is_in_list(w.isi_strategy, config.get_ISI_Strategy_Range(w), "ISI_strategy");

            {  // kernel aspects
                const auto kernel_options{config.get_kernel_range(w)};
                checker.check_is_in_list(w.kernel.width, kernel_options, "kernel.width");
                checker.check_is_in_list(w.kernel.height, kernel_options, "kernel.height");

                auto k = w.kernel;  // check if equal (SOH + DW_CONV)
                if (operation_behaviour.normalize_kernel_dimension(w.isi_strategy, k)) {
                    checker.add_check_failed("Kernel dimension are not normalized properly!(maybe not equal?)");
                }
            }
            {  // padding aspects
                const auto kernel_pad_horz_options{config.get_pad_horz_range(w)};
                const auto kernel_pad_vert_options{config.get_pad_vert_range(w)};
                const auto k{w.kernel};

                checker.check_is_in_list(k.pad_left, kernel_pad_horz_options, "kernel.pad_left");
                checker.check_is_in_list(k.pad_top, kernel_pad_vert_options, "kernel.pad_top");
                checker.check_is_in_list(k.pad_right, kernel_pad_horz_options, "kernel.pad_right");
                checker.check_is_in_list(k.pad_bottom, kernel_pad_vert_options, "kernel.pad_bottom");
            }
            {  // input/activation dimensions and tensor properties
                const auto& in0{w.input_0};
                // what to do with batch??
                checker.check_is_in_list((int)in0.height, config.get_input_height_range(w), "input_0.height");
                checker.check_is_in_list((int)in0.width, config.get_input_width_range(w), "input_0.width");

                checker.check_is_in_list((int)in0.channels, config.get_input_channels_range(w), "input_0.channels");

                checker.check_is_in_list(in0.datatype, config.valid_datatypes, "input_0.datatype");
                checker.check_is_in_list(in0.layout, config.valid_layouts, "input_0.layout");
                checker.check_is_in_list(in0.swizzling, config.valid_swizzlings, "input_0.swizzling");
            }
            {  // stride , depends on input zero, and operation sometimes
                const auto k{w.kernel};
                const auto stride_options{config.get_strides_range(w)};

                checker.check_is_in_list(k.stride_width, stride_options.first, "kernel.stride_width");
                checker.check_is_in_list(k.stride_height, stride_options.second, "kernel.stride_height");
            }
            {  // output dims, non random,  depend on input, padding, kernel, stride
               // batch in out to be equal
                if (w.output_0.batch != w.input_0.batch) {
                    checker.add_check_failed("Output.batch different than input_0.batch!");
                }

                const auto expected_out_width =
                        config.compute_output_dim((int)w.input_0.width, w.kernel.pad_left, w.kernel.pad_right,
                                                  w.kernel.width, w.kernel.stride_width);
                const auto expected_out_height =
                        config.compute_output_dim((int)w.input_0.height, w.kernel.pad_top, w.kernel.pad_bottom,
                                                  w.kernel.height, w.kernel.stride_height);

                checker.check_is_in_list(w.output_0.width, {expected_out_width}, "output_0.width");
                checker.check_is_in_list(w.output_0.height, {expected_out_height}, "output_0.height");

                // channels, do not check? (maybe in op specific area)

                //
                {  // layout and types for output_0
                    const auto& out0{w.output_0};
                    checker.check_is_in_list(out0.datatype, config.valid_datatypes, "output_0.datatype");
                    checker.check_is_in_list(out0.layout, config.valid_layouts, "output_0.layout");
                    checker.check_is_in_list(out0.swizzling, config.valid_swizzlings, "output_0.swizzling");
                }
            }
            {  // sparsity check on all channels

                // checker.check_is_in_list(w.input_0.sparsity_enabled, config.boolean_datatypes,
                //                          "input_0.sparsity_enabled");
                if ((w.input_0.sparsity < 0.0F) || (w.input_0.sparsity > 1.0F)) {
                    checker.add_check_failed("input_0.sparsity not in interval [0.0, 1.0] !");
                }

                // checker.check_is_in_list(w.input_1.sparsity_enabled, config.boolean_datatypes,
                //                          "output_1.sparsity_enabled");
                if ((w.input_1.sparsity < 0.0F) || (w.input_1.sparsity > 1.0F)) {
                    checker.add_check_failed("output_1.sparsity not in interval [0.0, 1.0] !");
                }
                {
                    std::string info_out{};
                    if (!operation_behaviour.check_sparsity_rules(config, w, info_out))
                        checker.add_check_failed(info_out);
                }
            }
            { checker.check_is_in_list(w.execution_order, config.valid_execution_order, "Execution_Order"); }
            // no padding optimization checked

            {  // check correlation between in-out tensors
                std::string info_out{};
                if (!operation_behaviour.check_input_output_tensor_corelation(config, w, info_out))
                    checker.add_check_failed(info_out);
            }

        } catch (const std::exception& e) {
            checker.add_check_failed(e.what());
        }
        // draw a final conclusion based on what was accumulated into the checker
        if (!checker.is_clean()) {
            result.mark_invalid_DPU_workload();
            result.info = checker.findings();
        }
    }

protected:
};

/// configuration bundle for Workloads at the most atomic level. workloads that are to be subjected to DPU
using OperationsContext = Behavior_Device_Mapping<OperationsBehaviour,  // operations
                                                  VPU2_0_WorkloadValidValues, VPU2_7_WorkloadValidValues>;

using DPU_OperationValidator = DPU_ConfigurableOperationValidator<OperationsContext>;

}  // namespace VPUNN

#endif  //
