// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef L1_COST_SERIALIZATION_WRAPPER_H
#define L1_COST_SERIALIZATION_WRAPPER_H

#include "vpu/serialization/serialization_wrapper.h"
#include "performance_mode.h"
namespace VPUNN {

class L1CostSerializationWrap : public CostSerializationWrap {
private:
    const DPU_OperationSanitizer& sanitizer;  ///< sanitizer mechanisms

private:

    /// @brief Turns OFF the swizzling
    ///
    /// @param workload [in, out] that will be changed in case the conditions are met
    void swizzling_turn_OFF(DPUWorkload& workload) const {
        if constexpr (false == PerformanceMode::allowLegacySwizzling_G5) {
            // only for some devices
            if (workload.device >= VPUDevice::NPU_RESERVED) {
                workload.set_all_swizzlings(Swizzling::KEY_0);
            }
        }
    }

    void serialize_info_and_compute_workload_uid(const DPUWorkload& wl) {
        auto wl_op = DPUOperation(wl, sanitizer.getDeviceConfiguration(wl.device));
        serializer_operation_uid = wl_op.hash();
        serializer.serialize(wl_op,
                             SerializableField<std::string>{"workload_uid", std::to_string(serializer_operation_uid)},
                             SerializableField<std::string>{"info", wl.get_layer_info()});
    }

public:
    L1CostSerializationWrap(CSVSerializer& ser, const DPU_OperationSanitizer& sanitizer_, bool inhibit = false,
                            size_t the_uid = 0)
            : CostSerializationWrap(ser, inhibit, the_uid),  // initialize the base class
              sanitizer(sanitizer_)

    {
    }

    ~L1CostSerializationWrap() = default;

    void serializeInfoAndComputeWorkloadUid(const DPUWorkload& wl, bool close_line=false) {
        if (!is_serialization_enabled() || is_error_present_during_serialization())
            return;
        try {
            serialize_info_and_compute_workload_uid(wl);

            if (close_line)
            {
                serializer.end();
            }
        } catch (const std::exception& e) {
            Logger::warning() << "Encountered invalid workload while serialization: " << e.what() << "\n";
            set_error();  // mark the error, so next operations will know something is not OK
            serializer.clean_buffers();
            serializer_operation_uid = 0;  // reset the wl uid in case of error
        }
    }

    void serializeCyclesAndCostInfo_closeLine(const CyclesInterfaceType cycles, const std::string cost_source,
                                              const std::string& info) {
        if (!is_serialization_enabled() || is_error_present_during_serialization())
            return;

        try {
            if (!serializer.is_write_buffer_clean()) {
                serializer.serialize(SerializableField<decltype(cycles)>{"vpunn_cycles", cycles});
                serializer.serialize(SerializableField<std::string>{"cost_source", cost_source});
                auto trimmed_info = trim_csv_str(info);
                serializer.serialize(SerializableField<std::string>{"error_info", trimmed_info});
                serializer.end();
            }
        } catch (const std::exception& e) {
            Logger::warning() << "Encountered invalid workload while serialization: " << e.what() << "\n";
            set_error();  // mark the error, so next operations will know something is not OK
            serializer.clean_buffers();
        }
        serializer.clean_buffers();
    }

    void serializeCyclesAndComputeWorkloadUid_closeLine(std::vector<DPUWorkload> serializer_orig_wls,
        const std::vector<CyclesInterfaceType> cycles_vector,
        const std::string& dpu_nickname)
    {
        if (!is_serialization_enabled() || is_error_present_during_serialization())
            return;

        for (unsigned int idx = 0; idx < serializer_orig_wls.size(); ++idx) {
            try {
                swizzling_turn_OFF(serializer_orig_wls[idx]);  // swizz guard sanitization
                auto wl_op = DPUOperation(serializer_orig_wls[idx],
                                          sanitizer.getDeviceConfiguration(serializer_orig_wls[idx].device));
                serializer_operation_uid = wl_op.hash();
                serializer.serialize(wl_op, SerializableField<std::string>{"workload_uid",
                                                                           std::to_string(serializer_operation_uid)});

                serializer.serialize(SerializableField<decltype(cycles_vector[idx])>{
                        dpu_nickname, cycles_vector[idx]});
                serializer.end();
            } catch (const std::exception& e) {
                Logger::warning() << "Encountered invalid workload while serialization: " << e.what() << "\n";
                set_error();  // mark the error, so next operations will know something is not OK
                serializer.clean_buffers();
            }
        }
    }
};
}  // namespace VPUNN

#endif  // VPUNN_SERIALIZATION_WRAPPER_H
