// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_SHAVE_WORKLOAD_H
#define VPUNN_SHAVE_WORKLOAD_H

#include <array>
#include <iostream>
#include <map>
#include <sstream>  //
#include <string>
#include <variant>
#include <vector>

#include "dpu_types.h"
#include "vpu_tensor.h"
#include "dpu_defaults.h"
#include "core/utils.h"

namespace VPUNN {

/**
 * @brief describes a Software layer (SHAVE) request
 */
class SHAVEWorkload {
private:
    std::string name{};  ///<  the name of the SW operation. We have a very flexible range of them.
    VPUDevice device{};  ///< The VPU device. There will be different methods/calibrations/profiling depending on device

    // input and output tensors number and content must be correlated with the operation and among themselves. Not all
    // combinations are possible
    std::vector<VPUTensor> inputs{};  ///< The input tensors. Mainly shape and datatype are used
    std::vector<VPUTensor> outputs{};  ///< The output tensors. Mainly shape and datatype are used
    std::string loc_name{};            ///< The location name

public:
    using Param = std::variant<int, float>;
    using Parameters = std::vector<SHAVEWorkload::Param>;

private:
    Parameters call_params{};  ///<  can emulate call parameters

public:
    /// @brief ctor must exist since we have aggregate initialization possible on this type (abstract type)
    SHAVEWorkload(const std::string& operation_name, const VPUDevice& device, const std::vector<VPUTensor>& inputs,
                  const std::vector<VPUTensor>& outputs, const Parameters& params = {}, const std::string& loc_name = "")
            : name(operation_name), device{device}, inputs{inputs}, outputs{outputs}, loc_name(loc_name), call_params{params} {
    }

    SHAVEWorkload(const SHAVEWorkload&) = default;
    SHAVEWorkload& operator=(const SHAVEWorkload&) = default;

    /// @brief Default constructor
    SHAVEWorkload() = default;


    // accessors

    std::string get_name() const {
        return name;
    };
    VPUDevice get_device() const {
        return device;
    };
    const std::vector<VPUTensor>& get_inputs() const {
        return inputs;
    };
    const std::vector<VPUTensor>& get_outputs() const {
        return outputs;
    };
    const Parameters& get_params() const {
        return call_params;
    };

    const std::string& get_loc_name() const {
        return loc_name;
    };

    uint32_t hash() const {
        std::stringstream ss;
        ss << static_cast<int>(get_device()) << ",";
        ss << get_name() << ",";
        for (const auto& input : get_inputs()) {
            ss << input.batches() << "," << input.channels() << "," << input.height() << "," << input.width() << ",";
        }
        for (const auto& output : get_outputs()) {
            ss << output.batches() << "," << output.channels() << "," << output.height() << "," << output.width()
               << ",";
        }

        for (const auto& param : get_params()) {
            std::visit(
                    [&ss](auto&& arg) {
                        ss << arg << ",";
                    },
                    param);
        }

        return fnv1a_hash(ss.str());
    }

    std::string toString() const {
        std::stringstream stream;
        stream << "SHAVEWorkload: \n"                                                                                //
               << " Operation: \t" << name << " ;\n"                                                                 //
               << " device: \t" << (int)device << " : " << VPUDevice_ToText.at(static_cast<int>(device)) << " ;\n";  //

        // inputs and outputs tensors
        {
            stream << " inputs: \t{\n";
            for (size_t i = 0; i < inputs.size(); i++) {
                stream << " input[" << i << "]: \t{\n" << inputs[i] << " } ;\n";
            }
            stream << "\t}inputs \n";
        }
        {
            stream << " outputs: \t{\n";
            for (size_t i = 0; i < outputs.size(); i++) {
                stream << " output[" << i << "]: \t{\n" << outputs[i] << " } ;\n";
            }
            stream << "\t}outputs \n";
        }

        auto toStream = [&stream](const Param& v) {
            if (const int* pvali = std::get_if<int>(&v)) {
                stream << *pvali;
            } else if (const float* pvalf = std::get_if<float>(&v)) {
                stream << *pvalf;
            }
        };
        {  // parameters
            stream << " parameters: \t{\n";
            for (size_t i = 0; i < call_params.size(); i++) {
                stream << " param[" << i << "]: \t{ ";  //
                toStream(call_params[i]);
                stream << " } ;\n";  //
            }
            stream << "\t} \n";
        }
        stream << out_terminator() << "SHAVEWorkload ";  // terminator

        return stream.str();
    };
    friend bool operator<(const VPUNN::SHAVEWorkload& lhs, const VPUNN::SHAVEWorkload& rhs);
};

inline std::ostream& operator<<(std::ostream& stream, const VPUNN::SHAVEWorkload& d) {
    stream << d.toString();
    return stream;
}

inline bool operator<(const VPUNN::SHAVEWorkload& lhs, const VPUNN::SHAVEWorkload& rhs) {
    // lexicographical_compare style
    {  // name
        if (lhs.name < rhs.name)
            return true;
        if (rhs.name < lhs.name)
            return false;
    }

    {  // device
        if (lhs.device < rhs.device)
            return true;
        if (rhs.device < lhs.device)
            return false;
    }
    {  // inputs
        if (lhs.inputs < rhs.inputs)
            return true;
        if (rhs.inputs < lhs.inputs)
            return false;
    }
    {  // outputs
        if (lhs.outputs < rhs.outputs)
            return true;
        if (rhs.outputs < lhs.outputs)
            return false;
    }
    {  // call_params
        if (lhs.call_params < rhs.call_params)
            return true;
        if (rhs.call_params < lhs.call_params)
            return false;
    }
    return false;  // all are  no smaller or no larger than other
}

}  // namespace VPUNN

#endif  // VPUNN_TYPES_H
