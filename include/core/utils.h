// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_CORE_UTILS_H
#define VPUNN_CORE_UTILS_H

#include <iostream>
#include <string>
#include <unordered_map>
#include <sstream>
#include <vector>
#include <thread>

namespace VPUNN {
	
static inline void set_env_var(const std::string& name, const std::string& value) {
#ifdef _WIN32
    std::string envString = name + "=" + value;
    _putenv(envString.c_str());  // Use _putenv on Windows
#else
    // POSIX-specific code (Linux/macOS)
    setenv(name.c_str(), value.c_str(), 1);  // 1 means overwrite the variable if it exists
#endif
}

static inline std::unordered_map<std::string, std::string> get_env_vars(const std::vector<std::string>& variables) {
    std::unordered_map<std::string, std::string> envMap;

    for (const auto& var : variables) {
#ifdef _WIN32  // Windows-specific code
        char* buffer = nullptr;
        size_t bufferSize = 0;

        if (_dupenv_s(&buffer, &bufferSize, var.c_str()) == 0 && buffer != nullptr) {
            envMap[var] = buffer;
            free(buffer);  // Free the allocated buffer after use
        } else {
            envMap[var] = "";  // If the environment variable is not found, store an empty string
        }
#else  // Unix-like systems
        const char* value = std::getenv(var.c_str());
        if (value) {
            envMap[var] = value;
        } else {
            envMap[var] = "";  // If the environment variable is not found, store an empty string
        }
#endif
    }

    return envMap;
}

static inline std::string generate_uid() {
    const auto file_name_postfix = get_env_vars({"VPUNN_FILE_NAME_POSTFIX"}).at("VPUNN_FILE_NAME_POSTFIX");
    std::ostringstream ss;
    ss << std::this_thread::get_id() << "_" << file_name_postfix;
    //std::string uid = ss.str();  // std::to_string(std::chrono::system_clock::now().time_since_epoch().count());

    return ss.str();
}

}

#endif  // VPUNN_CORE_UTILS_H