// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_PERSISTENT_CACHE
#define VPUNN_PERSISTENT_CACHE

#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <optional>
#include <stdexcept>
#include <vector>

#include "core/logger.h"

namespace VPUNN {
// A counter to measure the accesses and hits plus misses for a cache
// It is used to measure the efficiency of the cache
// Has a reset mechanism , to restart all counters, and a print method to show the results
/* coverity[rule_of_three_violation:FALSE] */
class AccessCounter {
private:
    // mutable because the reset is const
    mutable size_t accesses{0};
    mutable size_t hits{0};
    mutable size_t misses{0};

public:
    AccessCounter() {
    }

    // destructor with status printing
    ~AccessCounter() {
        try {
            printToLog(" At destructor: ");
            // std::cout << "\n At destructor: " << printString();
        } catch (...) {
            // ignore
        }
    }

    // copy constructor
    AccessCounter(const AccessCounter&) = delete;

    // copy assignment operator
    AccessCounter& operator=(const AccessCounter&) = delete;

    void access(bool hit = true) {
        ++accesses;
        ++(hit ? hits : misses);
    }

    void hit() {
        access(true);
    }

    void miss() {
        access(false);
    }

    void reset() const {
        accesses = 0;
        hits = 0;
        misses = 0;
    }

    void printToLog(const std::string prefix = "") const {
        Logger::info() << "\n" << prefix << printString() << "\n";
    }

    void printToCout() const {
        std::cout << "\n" << printString() << "\n";
    }

    // print to a string all the details, including the hit and miss ratios in percentage
    std::string printString() const {
        std::string result;
        result += "Cache Object: " + std::to_string((unsigned long long int)this) +
                  " stats: Accesses: " + std::to_string(accesses) + ", Hits: " + std::to_string(hits) +
                  ", Misses: " + std::to_string(misses);
        result += "\t, Hit ratio: " + std::to_string(getHitRatio() * 100) +
                  "%, Miss ratio: " + std::to_string(getMissRatio() * 100) + "%";
        return result;
    }

    size_t getAccesses() const {
        return accesses;
    }

    size_t getHits() const {
        return hits;
    }

    size_t getMisses() const {
        return misses;
    }

    // calc hit ratio
    double getHitRatio() const {
        return (accesses == 0) ? 0.0 : (static_cast<double>(hits) / accesses);
    }
    // calc miss ratio
    double getMissRatio() const {
        return (accesses == 0) ? 0.0 : (static_cast<double>(misses) / accesses);
    }
};

template <class T>
class FixedCache {
public:
    typedef std::map<std::vector<T>, T> PreloadedMap;

private:
    /// loaded from file, must be loaded from a file with the same descriptor signature
    PreloadedMap deserialized_table;
    const size_t interface_size_active{0};

    mutable AccessCounter counter{};  // mutable to allow "get" const methods to update it

public:
    FixedCache(size_t interface_size): FixedCache(interface_size, "") {
    }

    FixedCache(size_t interface_size = 0, const std::string& filename = ""): interface_size_active{interface_size} {
        deserializeCacheFromFile(filename);
    }

    FixedCache(size_t interface_size = 0, const char* file_data = nullptr, size_t file_data_length = 0)
            : interface_size_active{interface_size} {
        deserializeCacheFromData(file_data, file_data_length);
    }

    /// test for presence
    bool contains(const std::vector<T>& wl) const {
        return deserialized_table.find(wl) != deserialized_table.end();
    }

    /// getter
    std::optional<T> get(const std::vector<T>& wl) const {
        auto it = deserialized_table.find(wl);
        if (it != deserialized_table.end()) {
            counter.hit();
            return it->second;
        } else {
            counter.miss();
            return std::nullopt;
        }
    }

    /// setter
    void set(const std::vector<T>& wl, const T& value) {
        if (wl.size() != interface_size_active) {
            std::cerr << "Workload is with different size:  " << wl.size() << ", expecting : " << interface_size_active
                      << ", ignoring" << std::endl;
            return;
        }
        deserialized_table[wl] = value;
    }

    const T* get_pointer(const std::vector<T>& wl) const {
        auto it = deserialized_table.find(wl);
        if (it != deserialized_table.end()) {
            counter.hit();
            return &(it->second);
        } else {
            counter.miss();
            return nullptr;
        }
    }

    // for debug mainly
    const PreloadedMap& getMap() const {
        return deserialized_table;
    }

    /**
     * @brief Deserialize the cache from a binary file
     *
     * @param filename the name of the binary file
     * @param interface_size the size of the workload descriptor
     */
    bool deserializeCacheFromFile(const std::string& filename) {
        if (filename == "") {
            return false;
        }

        if (!std::filesystem::exists(filename)) {
            std::cerr << "Cache file does NOT exists: " << filename << ", ignoring..." << std::endl;
            return false;
        }

        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "Failed to open cache file for reading: " << filename << ", ignoring..." << std::endl;
            return false;
        }

        const size_t interface_size{interface_size_active};

        while (file.peek() != EOF) {
            std::vector<T> key(interface_size);
            T value{};

            // Read the key (interface_size values)
            file.read(reinterpret_cast<char*>(key.data()), interface_size * sizeof(T));

            // Read the value (interface_size+1-th value)
            file.read(reinterpret_cast<char*>(&value), sizeof(T));

            // Add the item to the deserialized table
            deserialized_table[key] = value;
        }

        file.close();
        return true;
    }

    bool deserializeCacheFromData(const char* file_data = nullptr, size_t file_data_length = 0) {
        if (file_data == nullptr) {  // null
            return false;
        }

        if (file_data_length == 0) {
            std::cerr << "Failed to open cache file for reading: size is zero "
                      << ", ignoring..." << std::endl;
            return false;
        }

        const size_t interface_size{interface_size_active};
        const size_t key_size{interface_size * sizeof(T)};  // bytes
        const size_t value_size{sizeof(T)};                 // bytes
        const size_t element_size{key_size + value_size};   // bytes

        {
            std::vector<T> key(interface_size);
            T value{};

            size_t crt_pos = 0;
            while ((crt_pos + element_size) <= file_data_length) {
                // copy to vector
                std::memcpy(reinterpret_cast<char*>(key.data()), file_data + crt_pos, key_size);
                // Read the value (interface_size+1-th value)
                std::memcpy(reinterpret_cast<char*>(&value), file_data + crt_pos + key_size, value_size);

                // Add the item to the deserialized table
                deserialized_table[key] = value;

                crt_pos += element_size;  // next element
            }
        }

        return true;
    }

    void readElement(std::ifstream& file, std::vector<T>& key, T& value) const {
        key.resize(interface_size_active);  // ensure space
        file.read(reinterpret_cast<char*>(key.data()), interface_size_active * sizeof(T));
        file.read(reinterpret_cast<char*>(&value), sizeof(T));
    }

    bool writeElement(std::ofstream& file, const std::vector<T>& key, const T& value) const {
        if (key.size() != interface_size_active) {
            std::cerr << "Workload is with different size:  " << key.size() << ", expecting : " << interface_size_active
                      << ", ignoring" << std::endl;
            return false;
        }

        file.write(reinterpret_cast<const char*>(key.data()), interface_size_active * sizeof(T));
        file.write(reinterpret_cast<const char*>(&value), sizeof(T));

        return true;
    }

    /// serialize the cache to a binary file
    bool serializeCacheToFile(const std::string& filename, bool appendIfExists = false) const {
        if (filename == "") {
            return false;
        }

        if (std::filesystem::exists(filename) && !appendIfExists) {
            std::cerr << "Cache file already exists: " << filename << ", ignoring..." << std::endl;
            return false;
        }

        const auto appendOrNew{appendIfExists ? std::ios::app : std::ios::trunc};

        std::ofstream file(filename, std::ios::binary | std::ios::out | appendOrNew);
        if (!file.is_open()) {
            std::cerr << "Failed to open cache file for writing: " << filename << ", ignoring..." << std::endl;
            return false;
        }

        // iterate and save
        for (auto const& [key, value] : deserialized_table) {
            writeElement(file, key, value);
        }

        file.close();
        return true;
    }

    // get debug access to the counter
    const AccessCounter& getCounter() const {
        return counter;
    }
};

}  // namespace VPUNN

#endif  // VPUNN_CACHE