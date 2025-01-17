// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_CACHE
#define VPUNN_CACHE

#include <list>
#include <map>
#include <stdexcept>
#include <vector>

#include "persistent_cache.h"

namespace VPUNN {

/**
 * @brief a workload cache using LRU (least recent used) replacement policy
 * @tparam T workload datatype
 */
template <class T>
class LRUCache {
private:
    typedef std::list<std::pair<std::vector<T>, T>> List;
    typedef typename List::iterator List_Iter;
    typedef std::map<std::vector<T>, List_Iter> Map;
    typedef typename Map::iterator Map_Iter;
    List workloads;
    Map m_table;
    size_t max_size, size;

    /// loaded from file, must be loaded from a file with the same descriptor signature
    const FixedCache<T> deserialized_table;

    static std::string decideCacheFilename(const std::string& filenamePrio1,
                                           const std::string& loadThisCSVIfPairedCacheExists) {
        auto selected_filename{filenamePrio1};  // can be empty or nonexistent
        if (filenamePrio1.empty()) {            // prio 1 dropped
            std::filesystem::path pairC{loadThisCSVIfPairedCacheExists};
            pairC.replace_extension(".cache_bin");
            if (!loadThisCSVIfPairedCacheExists.empty() && std::filesystem::exists(pairC)) {
                selected_filename = pairC.string();  // new selection
            }
        }
        return selected_filename;
    }

public:
    /**
     * @brief Construct a new LRUCache object
     *
     * @param max_size the maximum size of the LRUCache
     */
    explicit LRUCache(size_t max_size, size_t interface_size = 0, const std::string& filename = "",
                      const std::string& loadIfPairedCacheExists = "")
            : max_size(max_size),
              size(0),
              deserialized_table{interface_size, decideCacheFilename(filename, loadIfPairedCacheExists)} {
    }

    // const char* model_data, size_t model_data_length, bool copy_model_data
    explicit LRUCache(size_t max_size, size_t interface_size = 0, const char* file_data = nullptr,
                      size_t file_data_length = 0)
            : max_size(max_size), size(0), deserialized_table{interface_size, file_data, file_data_length} {
    }

    /**
     * @brief Add a new workload descriptor to the cache
     *
     * @param wl the workload descriptor (key)
     * @param value the workload value
     */
    void add(const std::vector<T>& wl, const T& value) {
        // If max_size == 0 we effectively disable the cache
        if (max_size == 0)
            return;

        // Check if the workload is already in the deserialized table
        if (deserialized_table.contains(wl))
            return;

        // Insert items in the list and map
        workloads.push_front({wl, value});
        auto it = workloads.begin();
        m_table.insert({wl, it});
        size++;

        while (size > max_size) {
            // Get the last element.
            auto last_item = workloads.end();
            --last_item;

            // Remove the last element
            remove(last_item->first);
        }
    }

private:
    /**
     * @brief Remove a workload from the cache
     *
     * @param wl the workload descriptor
     */
    void remove(std::vector<T>& wl) {
        // Remove the workload from the cache
        auto it = m_table.find(wl);

        if (it != m_table.end()) {
            // Erase the element from the map
            m_table.erase(it->first);
        } else {
            throw std::out_of_range("VPUNN Cache out of range");
        }

        // Remove the last element from the list
        workloads.pop_back();

        // Update the size
        size--;
    }

public:
    /**
     * @brief Get a workload from the cache.
     *
     * @param wl the workload descriptor
     * @return T* a pointer to the workload value stored in the cache, or nullptr if not available
     */
    const T* get(const std::vector<T>& wl) {
        // Check if the workload is in the deserialized table
        const T* elementInPreloadedCache{deserialized_table.get_pointer(wl)};
        if (elementInPreloadedCache) {
            return elementInPreloadedCache;  // ret the pointer to the element in the preloaded cache
        }

        // Check if the workload is in the main table
        Map_Iter it = m_table.find(wl);
        if (it != m_table.end()) {
            // Move the workload to the beginning of the list
            workloads.splice(workloads.begin(), workloads, it->second);
            // Return the value
            return &(it->second->second);
        } else {
            return nullptr;
        }
    }

    const AccessCounter& getPreloadedCacheCounter() const {
        return deserialized_table.getCounter();
    }
};

}  // namespace VPUNN

#endif  // VPUNN_CACHE