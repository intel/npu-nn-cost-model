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

#include "core/utils.h"
#include "core/persistent_cache.h"

namespace VPUNN {

/**
 * @brief a workload cache using LRU (least recent used) replacement policy
 * @tparam K is the Key type
 * @tparam V is the Value type
 */
template <typename K, typename V>
class LRUCache {
private:
    typedef std::list<std::pair<K, V>> List;
    typedef typename List::const_iterator List_Iter_cnst;

    typedef std::map<K, List_Iter_cnst> Map;
    typedef typename Map::const_iterator Map_Iter_cnst;

    List workloads;  ///< list with first being the most recently used key
    Map m_table;     ///< table for fast searching of keys  (contains pointers to list objects)
    const size_t max_size;
    size_t size{0};

    /// loaded from file, must be loaded from a file with the same descriptor signature
    /// @note this is a draft implementation
    /// This datatype knows it is a float Value and uint32 key. this beats the K, V template
    const FixedCache deserialized_table;  // maybe send it as template, OR reuse V and hashable K?

    /// @brief Decide which cache file to load  (DRAFT
    /// @param filenamePrio1 the first filename to try to load. must be a valid name and extension.Must be empty to go
    /// and try to select the second option
    /// @param loadThisCSVIfPairedCacheExists the second filename to try to load if the first does not exist. Takes only
    /// the name, and replaces the extension with the default extension for cache. Is selected only if exists on disk!
    static std::string decideCacheFilename(const std::string& filenamePrio1,
                                           const std::string& loadThisCSVIfPairedCacheExists) {
        auto selected_filename{filenamePrio1};  // can be empty or nonexistent
        if (filenamePrio1.empty()) {            // prio 1 dropped, does not exist
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
    explicit LRUCache(size_t max_size, const std::string& filename = "",
                      const std::string& prio2_loadIfPairedCacheExists = "")
            : max_size(max_size), deserialized_table{decideCacheFilename(filename, prio2_loadIfPairedCacheExists)} {
    }

    // const char* model_data, size_t model_data_length, bool copy_model_data
    explicit LRUCache(size_t max_size, const char* file_data = nullptr, size_t file_data_length = 0)
            : max_size(max_size), deserialized_table{file_data, file_data_length} {
    }

    /**
     * @brief Add a new workload descriptor to the cache
     *
     * @param wl the workload descriptor (key)
     * @param value the workload value
     */
    void add(const K& wl, const V& value) {
        // If max_size == 0 we effectively disable the cache!
        if (max_size == 0)
            return;

        // Check if the workload is already in the deserialized table
        {
            if constexpr (has_hash_v<K>) {
                if (deserialized_table.contains(wl.hash()))
                    return;
            } else {
                if (deserialized_table.contains(NNDescriptor<float>(wl).hash()))
                    return;
            }
        }

        // Insert items in the list and map
        workloads.push_front({wl, value});
        m_table.insert({wl, workloads.cbegin()});
        size++;

        // delete the oldest ones that occupy more space than allowed
        while (size > max_size) {
            const auto& last_item{workloads.back()};
            remove(last_item.first);  // key is first in pair
        }
    }

private:
    /**
     * @brief Remove a workload from the cache
     *
     * @param wl the workload descriptor
     */
    void remove(const K& wl) {
        Map_Iter_cnst it = m_table.find(wl);

        if (it != m_table.cend()) {
            m_table.erase(it->first);  // key is first
        } else {
            throw std::out_of_range("VPUNN Cache out of range");
        }

        workloads.pop_back();  // Remove the last element from the list
        size--;                // Update the size
    }

public:
    /**
     * @brief Get a workload from the cache.
     *
     * @param wl the workload descriptor
     * @return T* a pointer to the workload value stored in the cache, or nullptr if not available
     */
    const V* get(const K& wl) {
        // Check if the workload is in the deserialized table
        {
            uint32_t wlhash{0};
            if constexpr (has_hash_v<K>) {
                wlhash = wl.hash();
            } else {
                wlhash = NNDescriptor<float>(wl).hash();
            }

            const V* elementInPreloadedCache{deserialized_table.get_pointer(wlhash)};
            if (elementInPreloadedCache) {
                return elementInPreloadedCache;  // ret the pointer to the element in the preloaded cache
            }
        }

        // Check if the workload is in the main table
        Map_Iter_cnst it = m_table.find(wl);
        if (it != m_table.cend()) {
            // Move the workload to the beginning of the list
            workloads.splice(workloads.cbegin(), workloads, it->second);
            return &(it->second->second);  // second is the list iterator
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