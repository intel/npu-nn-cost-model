// Copyright © 2026 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the "Software Package")
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the "third-party-programs.txt" or other similarly-named text file included with the
// Software Package for additional details.

#include "vpu/shave_workload.h"
#include "core/primitive_hash.h"
#include "vpu/workload_hash.h"

#include "core/utils.h"
#include <iostream>
#include <sstream>  

namespace VPUNN {

uint32_t SHAVEWorkload::hash() const {
    uint32_t h = fnv_offset_basis;

    // Hash device
    h = PrimitiveHash::hash_enum(h, get_device());

    // Hash operation name
    h = PrimitiveHash::hash_string(h, get_name());

    // Hash input and output tensors
    for (const auto& input : get_inputs()) {
        h = WorkloadHash::hash_tensor(h, input);
    }
    for (const auto& output : get_outputs()) {
        h = WorkloadHash::hash_tensor(h, output);
    }

    // Hash call parameters
    for (const auto& param : get_params()) {
        h = WorkloadHash::hash_param(h, param);
    }

    // Hash extra parameters (sorted by key since std::map is ordered)
    for (const auto& [key, value] : get_extra_params()) {
        h = PrimitiveHash::hash_string(h, key);
        h = WorkloadHash::hash_param(h, value);
    }

    return h;
}


std::string SHAVEWorkload::toString() const {
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
        } else if (const std::string* pvals = std::get_if<std::string>(&v)) {
            stream << *pvals;
        } else if (const bool* pvalb = std::get_if<bool>(&v)) {
            stream << (*pvalb ? "true" : "false");
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
    {  // extra parameters
        stream << " extra parameters: \t{\n";
        for (const auto& [key, value] : extra_params) {
            stream << " key: " << key << " -> value: ";
            toStream(value);
            stream << " ;\n";
        }
        stream << "\t} \n";
    }
    stream << out_terminator() << "SHAVEWorkload ";  // terminator

    return stream.str();
}

bool operator<(const VPUNN::SHAVEWorkload& lhs, const VPUNN::SHAVEWorkload& rhs) {
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

std::ostream& operator<<(std::ostream& stream, const VPUNN::SHAVEWorkload& d) {
    stream << d.toString();
    return stream;
}

}  // namespace VPUNN
