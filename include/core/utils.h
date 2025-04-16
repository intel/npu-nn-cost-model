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

#include <cmath>
#include <iostream>
#include <mutex>
#include <shared_mutex>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <vector>

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
    const std::string file_name_postfix = get_env_vars({"VPUNN_FILE_NAME_POSTFIX"}).at("VPUNN_FILE_NAME_POSTFIX");
    std::ostringstream ss;
    ss << std::this_thread::get_id() << "_" << file_name_postfix;
    // std::string uid = ss.str();  // std::to_string(std::chrono::system_clock::now().time_since_epoch().count());

    return ss.str();
}

template <typename KeyType, typename ValueType>
class ThreadSafeMap {
protected:
    using MapType = std::map<KeyType, ValueType>;  ///< no hash
    MapType _map;
    mutable std::shared_mutex _mutex;

public:
    ThreadSafeMap() = default;
    ThreadSafeMap(const ThreadSafeMap& other) {
        std::shared_lock lock(other._mutex);
        _map = other._map;
    }
    ThreadSafeMap& operator=(const ThreadSafeMap& other) {
        if (this == &other) {
            return *this;
        }
        std::unique_lock lock(_mutex, std::defer_lock);
        std::unique_lock other_lock(other._mutex, std::defer_lock);
        std::lock(lock, other_lock);
        _map = other._map;

        return *this;
    }
    virtual ~ThreadSafeMap() = default;

    void insert(const KeyType& key, const ValueType& value) {
        std::unique_lock lock(_mutex);
        _map[key] = value;
    }
    bool contains(const KeyType& key) const {
        std::shared_lock lock(_mutex);
        return _map.find(key) != _map.end();
    }
    bool find(const KeyType& key, ValueType& value) const {
        std::shared_lock lock(_mutex);
        auto it = _map.find(key);
        if (it != _map.end()) {
            value = it->second;
            return true;
        }
        return false;
    }
    const ValueType* find(const KeyType& key) const {
        std::shared_lock lock(_mutex);
        auto it = _map.find(key);
        if (it != _map.end()) {
             return &it->second;
        }
        return nullptr;
    }
    bool remove(const KeyType& key) {
        std::unique_lock lock(_mutex);
        auto it = _map.find(key);
        if (it != _map.end()) {
            _map.erase(it);
            return true;
        }
        return false;
    }
    void clear() {
        std::unique_lock lock(_mutex);
        _map.clear();
    }
    size_t size() const {
        std::shared_lock lock(_mutex);
        return _map.size();
    }
    bool empty() const {
        std::shared_lock lock(_mutex);
        return _map.empty();
    }
    std::vector<KeyType> keys() const {
        std::shared_lock lock(_mutex);
        std::vector<KeyType> keys;
        for (const auto& [key, value] : _map) {
            keys.push_back(key);
        }
        return keys;
    }
    std::vector<ValueType> values() const {
        std::shared_lock lock(_mutex);
        std::vector<ValueType> values;
        for (const auto& [key, value] : _map) {
            values.push_back(value);
        }
        return values;
    }
};

template <typename T>
inline const std::string vec2int_str(const std::vector<T>& vec);

template <>
inline const std::string vec2int_str(const std::vector<float>& vec) {
    std::string result;
    constexpr size_t max_digits = 10;  // Maximum number of digits in the number
    result.reserve(vec.size() * max_digits);

    auto format_number = [&](auto value) -> std::string_view {
        thread_local char buffer[10];  // Thread local to avoid multithreading issues, max 10 characters numbers
        int len = std::snprintf(buffer, sizeof(buffer), "%d:", static_cast<int>(std::floor(value)));
        return std::string_view(buffer, len);
    };

    for (const auto& elem : vec) {
        result += format_number(elem);
    }

    return result;  // RVO
}

template <>
inline const std::string vec2int_str(const std::vector<int>& vec) {
    std::string result;
    constexpr size_t max_digits = 10;  // Maximum number of digits in the number
    result.reserve(vec.size() * max_digits);
    auto format_number = [&](auto value) -> std::string_view {
        thread_local char buffer[10];  // Thread local to avoid multithreading issues, max 10 characters numbers
        int len = std::snprintf(buffer, sizeof(buffer), "%d:", value);
        return std::string_view(buffer, len);
    };
    for (const auto& elem : vec) {
        result += format_number(elem);
    }
    return result;  // RVO
}

// 32b Fowler-Noll-Vo hash function with string input
inline uint32_t fnv1a_hash(const std::string& str) {
    uint32_t h = 0x811c9dc5;  // FNV-1a base
    for (char c : str) {
        h ^= c;
        h *= static_cast<uint32_t>(0x01000193);  // FNV-1a prime
    }
    return h;
}

// Define the has_hash trait for this namespace
template <typename, typename = std::void_t<>>
struct has_hash : std::false_type {};

template <typename T>
struct has_hash<T, std::void_t<decltype(std::declval<T>().hash())>> : std::true_type {};

template <typename T>
inline constexpr bool has_hash_v = has_hash<T>::value;

/// An ugly mechanism to generate a hash for a float vector descriptor
template <typename T>
struct NNDescriptor {
    const std::vector<T>& _desc;

    NNDescriptor(const std::vector<T>& desc): _desc(desc) {
    }

    uint32_t hash() const {
        return fnv1a_hash(vec2int_str(_desc));
    }
};

}  // namespace VPUNN

#endif  // VPUNN_CORE_UTILS_H