// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_CACHE_DESCRIPTORS
#define VPUNN_CACHE_DESCRIPTORS

#include <list>
#include <map>
#include <stdexcept>
// #include <unordered_map>
// #include <unordered_set>
#include <vector>

namespace VPUNN {

/**
 * @brief a workload cache using LRU (least recent used) replacement policy
 *
 * @tparam T value datatype, like a runtime value
 * @tparam K key datatype, normally a descriptor
 */
template <class V, typename K>
class LRUKeyCache {
private:
    typedef std::list<std::pair<K, V>> List;
    // typedef typename List::iterator List_Iter;
    typedef typename List::const_iterator List_Iter_cnst;
    typedef std::map<K, List_Iter_cnst> Map;
    typedef typename Map::const_iterator Map_Iter_cnst;
    List workloads;
    Map m_table;
    const size_t max_size;
    size_t size{0};

public:
    /**
     * @brief Construct a new LRUCache object
     *
     * @param max_size the maximum size of the LRUKeyCache
     */
    LRUKeyCache(size_t max_size): max_size(max_size) {
    }

    /**
     * @brief Add a new workload descriptor to the cache
     *
     * @param wl the workload descriptor (key)
     * @param value the workload value
     */
    void add(const K& wl, const V& value) {
        // If max_size == 0 we effectively disable the cache
        if (max_size == 0)
            return;
        // Insert items in the list and map
        workloads.push_front({wl, value});
        const auto it{workloads.begin()};
        m_table.insert({wl, it});
        size++;

        while (size > max_size) {
            // Get the last element.
            auto last_item{workloads.end()};
            --last_item;

            // Remove the last element
            remove(last_item->first);  // key is first in pair
        }
    }

private:
    /**
     * @brief Remove a workload from the cache
     *
     * @param wl the workload descriptor
     */
    void remove(const K& wl) {
        // Remove the workload from the cache
        const auto it{m_table.find(wl)};

        if (it != m_table.cend()) {
            // Return the value
            m_table.erase(it->first);  // key is first
        } else {
            throw std::out_of_range("VPUNN Cache Descriptor out of range");
        }

        workloads.pop_back();  // Remove the last element
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
        Map_Iter_cnst it{m_table.find(wl)};
        if (it != m_table.cend()) {
            // Move the workload to the beginning of the list
            workloads.splice(workloads.cbegin(), workloads, it->second);  // second is the list iterator
            // Return the value
            return &(it->second->second);
        } else {
            return nullptr;
        }
    }
};

/**
 * @brief a workload cache using LRU (least recent used) replacement policy
 *
 * @tparam T value datatype, like a runtime value
 * @tparam K key datatype, normally a descriptor
 */
template <class V, typename K>
class SimpleLUTKeyCache {
private:
    typedef std::list<std::pair<K, V>> List;
    // typedef typename List::iterator List_Iter;
    typedef typename List::const_iterator List_Iter_cnst;
    typedef std::map<K, List_Iter_cnst> Map;
    typedef typename Map::const_iterator Map_Iter_cnst;
    List workloads;
    Map m_table;
    const size_t max_size;
    size_t size{0};

public:
    /**
     * @brief Construct a new LRUCache object
     *
     * @param max_size the maximum size of the LRUKeyCache
     */
    SimpleLUTKeyCache(size_t max_size): max_size(max_size) {
    }

    /**
     * @brief Add a new workload descriptor to the cache
     *
     * @param wl the workload descriptor (key)
     * @param value the workload value
     */
    void add(const K& wl, const V& value) {
        // If max_size == 0 we effectively disable the cache
        if (max_size == 0)
            return;
        // Insert items in the list and map
        workloads.push_front({wl, value});
        const auto it{workloads.begin()};
        m_table.insert({wl, it});
        size++;

        // while (size > max_size) {
        //     // Get the last element.
        //     auto last_item{workloads.end()};
        //     --last_item;

        //    // Remove the last element
        //    remove(last_item->first);  // key is first in pair
        //}
    }

private:
    /**
     * @brief Remove a workload from the cache
     *
     * @param wl the workload descriptor
     */
    void remove(const K& wl) {
        // Remove the workload from the cache
        const auto it{m_table.find(wl)};

        if (it != m_table.cend()) {
            // Return the value
            m_table.erase(it->first);  // key is first
        } else {
            throw std::out_of_range("VPUNN Cache Descriptor out of range");
        }

        workloads.pop_back();  // Remove the last element
        size--;                // Update the size
    }

public:
    /**
     * @brief Get a workload from the cache.
     *
     * @param wl the workload descriptor
     * @return T* a pointer to the workload value stored in the cache, or nullptr if not available
     */
    const V* get(const K& wl) const {
        Map_Iter_cnst it{m_table.find(wl)};
        if (it != m_table.cend()) {
            //// Move the workload to the beginning of the list
            // workloads.splice(workloads.cbegin(), workloads, it->second);  // second is the list iterator
            //  Return the value
            return &(it->second->second);
        } else {
            return nullptr;
        }
    }
};

}  // namespace VPUNN

#endif  // VPUNN_CACHE