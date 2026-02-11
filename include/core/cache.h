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
#include <unordered_map>
#include <optional>
#include <stdexcept>
#include <vector>
#include <thread>
#include <filesystem>
#include <shared_mutex>

#include <cassert>

#include "core/persistent_cache.h"
#include "core/utils.h"

namespace VPUNN {

template <typename K, typename V>
class FixedCacheAddON {
protected:
    // FixedCacheAddON(): FixedCacheAddON("", "") {
    // }
    FixedCacheAddON(const std::string& filename, const std::string& prio2_loadIfPairedCacheExists)
            : deserialized_table{[&]() {
                  auto env_override = check_if_env_path_override();
                  if (!env_override.empty()) {
                      return FixedCache(env_override);
                  }
                  return FixedCache(decideCacheFilename(filename, prio2_loadIfPairedCacheExists));
              }()} {
    }

    FixedCacheAddON(const char* file_data, size_t file_data_length)
            : deserialized_table{[&]() {
                  auto env_override = check_if_env_path_override();
                  if (!env_override.empty()) {
                      return FixedCache(env_override);
                  }
                  return FixedCache(file_data, file_data_length);
              }()} {
    }

protected:
    bool contains(const K& wl) const {
        if constexpr (has_hash_v<K>) {
            if (deserialized_table.contains(wl.hash()))
                return true;
        } else {
            if (deserialized_table.contains(NNDescriptor<float>(wl).hash()))
                return true;
        }
        return false;
    }

    std::optional<V> get(const K& wl) const {
        // Check if the workload is in the deserialized table
        uint32_t wlhash{0};
        if constexpr (has_hash_v<K>) {
            wlhash = wl.hash();
        } else {
            wlhash = NNDescriptor<float>(wl).hash();
        }

        return deserialized_table.get(wlhash);
    }

private:
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

    static std::string check_if_env_path_override() {
        auto env_cache_path = get_env_vars({"VPUNN_CACHE_PATH"}).at("VPUNN_CACHE_PATH");
        if (!env_cache_path.empty() && std::filesystem::exists(env_cache_path)) {
            return env_cache_path;
        }
        return {};
    }

public:
    const AccessCounter& getPreloadedCacheCounter() const {
        return deserialized_table.getCounter();
    }
};

// Custom hasher for std::vector<float> using FNV-1a
struct VectorFloatHasher {
    std::size_t operator()(const std::vector<float>& vec) const noexcept {
        return static_cast<std::size_t>(fnv1a_hash(vec));
    }
};

// Type trait to select appropriate map type based on key
// Default: use std::map for types without custom hasher (O(log n) lookup)
// Specialize this template in the header where your key type is defined
// to use std::unordered_map with a custom hasher for O(1) lookup
template<typename K>
struct MapTypeSelector {
    template<typename V>
    using type = std::map<K, V>;
};

// Specialization for std::vector<float>: use std::unordered_map
template<>
struct MapTypeSelector<std::vector<float>> {
    template<typename V>
    using type = std::unordered_map<std::vector<float>, V, VectorFloatHasher>;
};

/**
 * @brief a workload cache using LRU (least recent used) replacement policy
 * @tparam K is the Key type
 * @tparam V is the Value type
 * 
 * Uses std::unordered_map for O(1) average lookup when specialized (e.g., DPUWorkload, std::vector<float>)
 * Falls back to std::map for other types (O(log n) lookup)
 */
template <typename K, typename V>
class LRUCache : public FixedCacheAddON<K, V> {
private:
    typedef std::list<std::pair<K, V>> List;
    typedef typename List::const_iterator List_Iter_cnst;

    // Select map type based on key type
    // Uses unordered_map with custom hasher when MapTypeSelector is specialized (O(1) lookup)
    // Falls back to std::map for other types (O(log n) lookup)
    typedef typename MapTypeSelector<K>::template type<List_Iter_cnst> Map;
    typedef typename Map::const_iterator Map_Iter_cnst;

    mutable List workloads;  ///< list with first being the most recently used key.
    Map m_table;             ///< table for fast searching of keys  (contains pointers to list objects, as iterators)

    const size_t max_size;

    mutable std::shared_mutex mtx;  ///< Mutex to protect shared resources.

public:
    /**
     * @brief Construct a new LRUCache object
     *
     * @param max_size the maximum size of the LRUCache
     */
    explicit LRUCache(size_t max_size, const std::string& filename = "",
                      const std::string& prio2_loadIfPairedCacheExists = "")
            : FixedCacheAddON<K, V>(filename, prio2_loadIfPairedCacheExists), max_size(max_size) {
    }

    // const char* model_data, size_t model_data_length, bool copy_model_data
    explicit LRUCache(size_t max_size, const char* file_data, size_t file_data_length)
            : FixedCacheAddON<K, V>(file_data, file_data_length), max_size(max_size) {
    }

    /**
     * @brief Add a new workload descriptor to the cache. If the key exists is does NOT replace the old value with the
     * new one
     *
     * @param wl the workload descriptor (key)
     * @param value the workload value
     */
    void add(const K& wl, const V& value) {
        // If max_size == 0 we effectively disable the cache!
        if (max_size == 0)
            return;

        std::unique_lock<std::shared_mutex> lock(mtx);  // Exclusive lock for write

        // Check if the workload is already in the deserialized table
        if (FixedCacheAddON<K, V>::contains(wl))
            return;

        const Map_Iter_cnst& map_it{m_table.find(wl)};
        if (map_it == m_table.cend()) {
            // Insert items in the list and map
            workloads.push_front({wl, value});         // adds a new element
            m_table.insert({wl, workloads.cbegin()});  // would not add a new element if wl is already inside

            clean_up_excess_elements();  // if size is exceeded
        } else {
            // wl already in table, keep old value, move to first position
            mark_as_most_recently_used(map_it);
        }

        if (!check_consistency()) {
            throw std::runtime_error("Cache consistency check failed after adding workload");
        }
    }

public:
    /**
     * @brief Get a value from the cache.
     *
     * @param wl the workload(key) descriptor
     * @return std::optional<V> the value stored in the cache, or nothing if not available
     */
    std::optional<V> get(const K& wl, std::string* source = nullptr) const {
        // Check if the workload is in the deserialized table
        {
            const std::optional<V> found{FixedCacheAddON<K, V>::get(wl)};
            if (found) {
                if (source) *source = "fixed_cache";
                return found;
            }
        }

        // First, try to find the key with a shared lock
        {
            std::shared_lock<std::shared_mutex> lock(mtx);
            auto map_it = m_table.find(wl);
            if (map_it == m_table.cend()) {
                return std::nullopt;
            }
        }

        // If found, acquire a unique lock and do the mutation
        std::unique_lock<std::shared_mutex> lock(mtx);
        auto map_it = m_table.find(wl);
        if (map_it != m_table.cend()) {
            mark_as_most_recently_used(map_it);
            if (source) *source = "dyn_cache";
            return (map_it->second->second);
        } else {
            return std::nullopt;
        }
    }

private:
    /**
     * @brief Remove a workload from the cache
     *
     * @param wl the workload descriptor
     */
    void remove(const K& wl) {
        const Map_Iter_cnst& it{m_table.find(wl)};

        if (it != m_table.cend()) {
            m_table.erase(it->first);  // key is first
        } else {
            throw std::out_of_range("VPUNN Cache out of range, an element was not in table");
        }

        workloads.pop_back();  // Remove the last element from the list

        if (!check_consistency()) {
            throw std::runtime_error("Cache consistency check failed after removing workload");
        }
    }

    /// deletes what exceeds the size
    void clean_up_excess_elements() {
        // delete the oldest ones that occupy more space than allowed
        while (m_table.size() > max_size) {
            const auto& oldest_item{workloads.back()};  // last
            remove(oldest_item.first);                  // key is first in pair
        }
    }
    
    void mark_as_most_recently_used(const Map_Iter_cnst& map_it) const {
        if (map_it != m_table.cend()) {                      // Move the workload to the beginning of the list
            const List_Iter_cnst& list_it = map_it->second;  // second is the list iterator
            workloads.splice(workloads.cbegin(), workloads, list_it);
        }
    }

protected:

    /// @brief Check if the cache is consistent, i.e., the number of workloads matches the size of the table
    bool check_consistency() const {
        if (workloads.size() != m_table.size()) {
            return false;
        }

        return true;
    }
};

}  // namespace VPUNN

#endif  // VPUNN_CACHE
