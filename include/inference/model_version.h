// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef MODEL_VERSION_H
#define MODEL_VERSION_H

#include <stdio.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>

namespace VPUNN {

/// Extracts and keeps  the version out of a NN raw name
class ModelVersion {
public:
    /// @brief value of input interface version, the one for the input descriptor
    int get_input_interface_version() const {
        return input_version;
    };
    /// @brief value of output interface version, the one for the NN provided value(s)
    int get_output_interface_version() const {
        return output_version;
    };

    /// @brief name of the NN, without version info
    std::string get_NN_name() const {
        return name;
    };

    /// @brief initial, raw name of the VPUNN model. contains version info
    std::string get_raw_name() const {
        return full_raw_name;
    };

    /// @brief parsed the name and extracts the information
    /// get the name based on separator "-", template: NNNNNNN-VI-VO[and here to be a string with nickname&serialNo
    /// without impacting the crt implementation]
    /// Template extension: NNNNNNN-VI-VO $v0000.0000 Nickname26chars$
    /// with 26 chars between $ signs
    /// VI and VO must be integers, and only the integers part will be
    /// considered when converting NNNNNN cannot be empty, it will be replaced with "none" if empty Only first three
    /// parts of the name are considered, rest are ignored. Missing pars will be considered default: none-1-1
    ///
    /// @throws invalid_argument, out_of_range
    void parse_name(const std::string& raw_NN_name) {
        // get the name based on separator "-", template NNNNNNN-VI-VO
        // VI and VO must be integers, and only the integers part will be considered when converting
        reset();  // all are on default
        full_raw_name = raw_NN_name;
        std::istringstream input;
        input.str(full_raw_name);
        std::vector<std::string> parts;
        for (std::string part{""}; std::getline(input, part, version_separator);) {
            parts.push_back(part);
        }

        // fist position is the full name, mandatory
        if (parts.size() >= 1) {
            if (parts[0].length() > 0) {
                name = parts[0];
            }
        } else {  // empty name
            // use latest
            input_version = def_latest_special_version;
        }

        // in version
        if (parts.size() >= 2) {
            size_t c;
            int v{std::stoi(parts[1], &c)};  // might throw invalid_argument, out_of_range
            input_version = v;
        }

        // out version
        if (parts.size() >= 3) {
            size_t c;
            int v{std::stoi(parts[2], &c)};  // might throw invalid_argument, out_of_range
            output_version = v;
        }
    }

    ModelVersion() {
        reset();
    }

private:
    std::string full_raw_name;
    std::string name;
    int input_version;
    int output_version;

    static constexpr char version_separator = '-';
    static constexpr int def_input_version = 1;
    static constexpr int def_output_version = 1;
    static constexpr int def_latest_special_version = 0;

    void reset() {
        full_raw_name = "none";
        name = "none";
        input_version = 1;
        output_version = 1;
    }
};

}  // namespace VPUNN
#endif
