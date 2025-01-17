// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_BASIC_LAYER_CONTEXT_H
#define VPUNN_BASIC_LAYER_CONTEXT_H

// #include <array>
#include <iostream>
// #include <map>
// #include <optional>
#include <sstream>  //
#include <string>

namespace VPUNN {

/// @brief layer information, for logging and debugging purposes
class BasicLayerContextInformation {
public:
    std::string layer_info{
            ""};  ///< textual information about the belonging of this workload. NOt mandatory, used only for logging
                  // members
public:
    std::string get_layer_info() const {
        return layer_info;
    }
    void set_layer_info(const std::string& layer_info_name) {
        layer_info = layer_info_name;
    }

    // operations/methods
public:
    /// equality test operator
    bool operator==(const BasicLayerContextInformation& b) const {
        bool r{true};
        r = r && (layer_info == b.layer_info);
        return r;
    }
};

inline std::ostream& operator<<(std::ostream& stream, const VPUNN::BasicLayerContextInformation& d) {
    stream << "layer_info:" << d.layer_info << " ;\n";
    return stream;
}

}  // namespace VPUNN

#endif
