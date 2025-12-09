// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef SHAVE_COST_SERIALIZATION_WRAPPER_H
#define SHAVE_COST_SERIALIZATION_WRAPPER_H

#include "vpu/serialization/serialization_wrapper.h"
namespace VPUNN {

class SHAVECostSerializationWrap : public CostSerializationWrap {
private:
    void serialize_shave(const SHAVEWorkload& shave_wl) const {
        const auto& inputs = shave_wl.get_inputs();
        const auto& outputs = shave_wl.get_outputs();
        const auto& params = shave_wl.get_params();
        const auto& extra_params = shave_wl.get_extra_params();

        serializer.serialize(SerializableField<VPUDevice>{"device", shave_wl.get_device()});
        serializer.serialize(SerializableField<std::string>{"operation", shave_wl.get_name()});

        for (size_t i = 0; i < inputs.size(); i++) {
            serializer.serialize(
                    SerializableField<unsigned int>{"input_" + std::to_string(i) + "_batch", inputs[i].batches()});
            serializer.serialize(
                    SerializableField<unsigned int>{"input_" + std::to_string(i) + "_channels", inputs[i].channels()});
            serializer.serialize(
                    SerializableField<unsigned int>{"input_" + std::to_string(i) + "_height", inputs[i].height()});
            serializer.serialize(
                    SerializableField<unsigned int>{"input_" + std::to_string(i) + "_width", inputs[i].width()});
            serializer.serialize(
                    SerializableField<DataType>{"input_" + std::to_string(i) + "_datatype", inputs[i].get_dtype()});
            serializer.serialize(
                    SerializableField<Layout>{"input_" + std::to_string(i) + "_layout", inputs[i].get_layout()});
            serializer.serialize(SerializableField<bool>{"input_" + std::to_string(i) + "_sparsity_enabled",
                                                         inputs[i].get_sparsity()});
        }

        for (size_t i = 0; i < outputs.size(); i++) {
            serializer.serialize(
                    SerializableField<unsigned int>{"output_" + std::to_string(i) + "_batch", outputs[i].batches()});
            serializer.serialize(SerializableField<unsigned int>{"output_" + std::to_string(i) + "_channels",
                                                                 outputs[i].channels()});
            serializer.serialize(
                    SerializableField<unsigned int>{"output_" + std::to_string(i) + "_height", outputs[i].height()});
            serializer.serialize(
                    SerializableField<unsigned int>{"output_" + std::to_string(i) + "_width", outputs[i].width()});
            serializer.serialize(
                    SerializableField<DataType>{"output_" + std::to_string(i) + "_datatype", outputs[i].get_dtype()});
            serializer.serialize(
                    SerializableField<Layout>{"output_" + std::to_string(i) + "_layout", outputs[i].get_layout()});
            serializer.serialize(SerializableField<bool>{"output_" + std::to_string(i) + "_sparsity_enabled",
                                                         outputs[i].get_sparsity()});
        }

        int param_idx = 0;
        for (const auto& param : params) {
            std::visit(
                    [&](auto&& arg) {
                        using T = std::decay_t<decltype(arg)>;

                        serializer.serialize(SerializableField<T>{"param_" + std::to_string(param_idx), arg});
                    },
                    param);
            param_idx++;
        }
        int extra_param_idx = 0;
        for (const auto& [key, value] : extra_params) {
            std::string combined = key + "/";
            std::visit([&](auto&& arg) {
                using T = std::decay_t<decltype(arg)>;
                if constexpr (std::is_same_v<T, std::string>) {
                    combined += arg;
                } else if constexpr (std::is_same_v<T, bool>) {
                    combined += arg ? "True" : "False";
                } else {
                    combined += std::to_string(arg);
                }
            }, value);
            serializer.serialize(SerializableField<std::string>{"extra_param_" + std::to_string(extra_param_idx), combined});
            extra_param_idx++;
        }
    }


public:
    SHAVECostSerializationWrap(CSVSerializer& ser, bool inhibit = false, size_t the_uid = 0)
            : CostSerializationWrap(ser, inhibit, the_uid)  // initialize the base class

    {
    }

    ~SHAVECostSerializationWrap() = default;

    void serializeShaveWorkloadWithCycles(const SHAVEWorkload& swl, const std::string& shave_model_kind, const CyclesInterfaceType cycles) {
        if (!is_serialization_enabled())
            return;
        try {
            serializer.serialize(SerializableField<VPUDevice>{"device", swl.get_device()});
            serializer.serialize(SerializableField<std::string>{"loc_name", swl.get_loc_name()});
            serializer.serialize(SerializableField<std::string>{"shave_model_kind", shave_model_kind});
            serializer.serialize(SerializableField<decltype(cycles)>{"cycles", cycles});
            serialize_shave(swl);
            serializer.end();  // new line in csv!
        } catch (const std::exception& e) {
            Logger::warning() << "Encountered invalid workload while serialization: " << e.what() << "\n";
            set_error();  // mark the error, so next operations will know something is not OK
            serializer.clean_buffers();
        }
        serializer.clean_buffers();
    }
};

/**
 * @brief Utility class for SHAVE serialization operations
 * 
 * This class provides static utility functions for SHAVE serialization
 * without requiring object instantiation.
 */
class ShaveSerializerUtils {
public:
    /**
     * @brief Get the field names for SHAVE serialization
     * 
     * Returns a vector of field names used for SHAVE workload serialization.
     * This function can be called directly without creating an instance of the class.
     * 
     * @param max_num_params Maximum number of parameters to include in the field names (default: 10)
     * @return std::vector<std::string> Vector containing all field names for SHAVE serialization
     */
    static std::vector<std::string> get_names_for_shave_serializer(const int max_num_params = 10) {
        auto fields = std::vector<std::string>({"device", "operation"});
        
        // Input fields for up to 8 inputs
        for (int i = 0; i < 8; i++) {
            fields.emplace_back("input_" + std::to_string(i) + "_batch");
            fields.emplace_back("input_" + std::to_string(i) + "_channels");
            fields.emplace_back("input_" + std::to_string(i) + "_height");
            fields.emplace_back("input_" + std::to_string(i) + "_width");
            fields.emplace_back("input_" + std::to_string(i) + "_sparsity_enabled");
            fields.emplace_back("input_" + std::to_string(i) + "_datatype");
            fields.emplace_back("input_" + std::to_string(i) + "_layout");
        }

        // Output fields (assuming single output)
        fields.emplace_back("output_0_batch");
        fields.emplace_back("output_0_channels");
        fields.emplace_back("output_0_height");
        fields.emplace_back("output_0_width");
        fields.emplace_back("output_0_sparsity_enabled");
        fields.emplace_back("output_0_datatype");
        fields.emplace_back("output_0_layout");

        // Model and cycles fields
        fields.emplace_back("shave_model_kind");
        fields.emplace_back("cycles");

        // Parameter fields
        for (int i = 0; i <= max_num_params; i++) {
            fields.emplace_back("param_" + std::to_string(i));
        }

        // Extra parameter fields
        for (int i = 0; i <= 8; i++) {
            fields.emplace_back("extra_param_" + std::to_string(i));
        }

        // Additional fields
        fields.emplace_back("loc_name");
        fields.emplace_back("info");
        fields.emplace_back("workload_uid");

        return fields;
    }
};

}  // namespace VPUNN

#endif  // VPUNN_SERIALIZATION_WRAPPER_H
