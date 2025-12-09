// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_DPU_OPERATIONS_SANITIZER_H
#define VPUNN_DPU_OPERATIONS_SANITIZER_H

#include <numeric>

#include "sanity_report.h"
#include "vpu/types.h"

// #include "device_valid_values.h"
#include "dpu_operations_valid_behaviours.h"
#include "dpu_operations_validator.h"

namespace VPUNN {

/// @brief Sanitizes a Workload  based on device rules
class DPU_OperationSanitizer : private DPU_OperationValidator {
public:
    /// @brief Checks if the workload is usable, makes sanitization changes , and lets the user know the sanitized
    /// workload and its conclusion
    ///
    /// Sanitization means that some parameters of the workload are automatically adjusted, but still the new WL is
    /// relevant (or equivalent regarding cost) to the original one. Sanitization performed:
    /// - input and output tensors data type are restricted to one type per category. UINT8 for ints on 8 bit, FLOAT16
    /// for all16 bit floats. @see IDeviceValidValues around valid_datatypes usage
    ///  - ...
    ///
    /// Usability checks will check general characteristics:
    /// - if the device is supported
    /// - if workload fits in CMX memory
    /// - if the operation is supported
    /// - ...
    ///
    /// @param wl [in, out] the workload to analyze and sanitize
    /// @param result [out] status of check. if not OK (not NO_ERROR) the wl cannot be used
    void check_and_sanitize(DPUWorkload& wl, SanityReport& result) const {
        result.resetOK();  // all OK

        if (!is_supported(wl.device)) {
            result.mark_unknown_device();
            return;
        }

        const auto& config = get_config(wl.device);

        // force execution mode when dCIM engine is selected
        if (wl.mpe_engine == MPEEngine::DCIM) {
            wl.execution_order = ExecutionMode::dCIM_32x128;
        }

        // check ahead
        if (!config.is_valid_operation(wl.op)) {
            result.mark_unknown_operation();
            return;
        }

        // apply data restriction for input and output tensor types. device/config dependent
        {
            const auto intype_0{wl.inputs[0].get_dtype()};
            wl.inputs[0].change_datatype_superficial(config.restrict_datatype(intype_0));

            const auto outtype_0{wl.outputs[0].get_dtype()};
            wl.outputs[0].change_datatype_superficial(config.restrict_datatype(outtype_0));
        }
        // change layouts types from VPU2.0 defaults to VPU 2.7 complete set
        {
            wl.inputs[0].set_if_same_layout(config.adapt_device_comaptible_tensor_layout(wl.inputs[0].get_layout()));
            wl.outputs[0].set_if_same_layout(config.adapt_device_comaptible_tensor_layout(wl.outputs[0].get_layout()));
        }
        // change swizzlings in case we are in a device that has less/no swizzlings
        {
            wl.input_swizzling[0] = config.adapt_device_comaptible_swizzling(wl.input_swizzling[0]);
            wl.input_swizzling[1] = config.adapt_device_comaptible_swizzling(wl.input_swizzling[1]);

            wl.output_swizzling[0] = config.adapt_device_comaptible_swizzling(wl.output_swizzling[0]);
        }

        try {                                  // operation might be problematic
                                               // check memory
            const DPUOperation w(wl, config);  // workload internal representation

            const auto cmx_memory = memory_calculator.compute_memory(w, config);

            const int avaialable_cmx_memo{config.get_cmx_size(wl.device)};
            const auto necesarry_cmx_memo = cmx_memory.cmx;

            if (avaialable_cmx_memo < necesarry_cmx_memo) {
                result.mark_size_too_big();
                return;
            }

            // here is a good place to do the validation checking. All was OK until now
            auto& operation_behaviour = config.get_specific_behaviour(wl.op);
            DPU_OperationValidator::check_workload_consistency(w, config, operation_behaviour, result);

        } catch (const std::runtime_error& e) {
            result.mark_unknown_operation();
            std::cerr << "Exception detected during sanitization: " << e.what() << std::endl;
            return;
        }
    }

    void check_data_consistency(DPUWorkload& wl, SanityReport& result) const {
        result.resetOK();  // all OK
        if (!is_supported(wl.device)) {
            result.mark_unknown_device();
            return;
        }
        const auto& config = get_config(wl.device);  // previous if prevents throwing

        // check ahead
        if (!config.is_valid_operation(wl.op)) {
            result.mark_unknown_operation();
            return;
        }

        try {
            const DPUOperation w(wl, config);                                  // workload internal representation
            auto& operation_behaviour = config.get_specific_behaviour(wl.op);  // will throw if unknown op
            DPU_OperationValidator::check_workload_consistency(w, config, operation_behaviour, result);

        } catch (const std::runtime_error&) {
            result.mark_unknown_operation();
        }
    }

    const IDeviceValidValues& getDeviceConfiguration(VPUDevice device) const {
        return get_config(device);  // for now
    }
};

}  // namespace VPUNN

#endif  //
