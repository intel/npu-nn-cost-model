// Copyright © 2023 Intel Corporation
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

namespace VPUNN {

/**
 * @brief a workload cache using LRU (least recent used) replacement policy
 *
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

public:
    /**
     * @brief Construct a new LRUCache object
     *
     * @param max_size the maximum size of the LRUCache
     */
    LRUCache(size_t max_size): max_size(max_size), size(0) {
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
            // Return the value
            m_table.erase(it->first);
        } else {
            throw std::out_of_range("VPUNN Cache out of range");
            ;
        }

        // Remove the last element
        workloads.pop_back();

        // Update the size
        size--;
    }

public:
    /**
     * @brief Get a workload from the cache.
     *
     * @param wl the workload descriptor
     * @return T* a pointer to the workload value stored in the cache, or NULL if not available
     */
    T* get(const std::vector<T>& wl) {
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
};

}  // namespace VPUNN

#endif  // VPUNN_CACHE