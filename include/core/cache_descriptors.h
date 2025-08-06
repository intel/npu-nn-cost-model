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

// OBSOLETE/TENTATIVE DEVELOPMENT NOW

#include <list>
#include <map>
#include <optional>
#include <stdexcept>
// #include <unordered_map>
// #include <unordered_set>
#include <vector>

#include <cassert>

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
    typedef typename List::const_iterator List_Iter_cnst;

    typedef std::map<K, List_Iter_cnst> Map;
    typedef typename Map::const_iterator Map_Iter_cnst;

    mutable List workloads;  ///< list with first being the most recently used key.
    Map m_table;             ///< table for fast searching of keys  (contains pointers to list objects, as iterators)

    const size_t max_size;
    size_t size{0};  // current size of list (and table)

public:
    /**
     * @brief Construct a new LRUCache object
     *
     * @param max_size the maximum size of the LRUKeyCache
     */
    LRUKeyCache(size_t max_size): max_size(max_size) {
    }

    /**
     * @brief Add a new workload descriptor to the cache. If the key exists is does NOT replace the old value with the
     * new one
     * @param wl the workload descriptor (key)
     * @param value the workload value
     */
    void add(const K& wl, const V& value) {
        // If max_size == 0 we effectively disable the cache
        if (max_size == 0)
            return;

        const Map_Iter_cnst& map_it{m_table.find(wl)};
        if (map_it == m_table.cend()) {
            // Insert items in the list and map
            workloads.push_front({wl, value});         // adds a new element
            m_table.insert({wl, workloads.cbegin()});  // would not add a new element if wl is already inside
            size++;

            clean_up_excess_elements();  // if size is exceeded
        } else {
            // wl already in table, keep old value, move to first position
            mark_as_most_recently_used(map_it);
        }

        assert_invariants();
    }

public:
    /**
     * @brief Get a value from the cache.
     *
     * @param wl the workload(key) descriptor
     * @return std::optional<V> the value stored in the cache, or nothing if not available
     */
    std::optional<V> get(const K& wl) const {
        // Check if the workload is in the main table
        const Map_Iter_cnst& map_it{m_table.find(wl)};
        if (map_it != m_table.cend()) {
            mark_as_most_recently_used(map_it);
            return (map_it->second->second);  //  second is the value stored in list, now first one
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
        size--;                // Update the size

        assert_invariants();
    }
    /// deletes what exceeds the size
    void clean_up_excess_elements() {
        // delete the oldest ones that occupy more space than allowed
        while (size > max_size) {
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

    void assert_invariants() const {
        assert(workloads.size() == size);
        assert(m_table.size() == size);
    }
};
//
///**
// * @brief a workload cache without capacity and replace policy
// *
// * @tparam T value datatype, like a runtime value
// * @tparam K key datatype, normally a descriptor
// */
//template <class V, typename K>
//class SimpleLUTKeyCache {
//private:
//    typedef std::list<std::pair<K, V>> List;
//    typedef typename List::const_iterator List_Iter_cnst;
//
//    typedef std::map<K, List_Iter_cnst> Map;
//    typedef typename Map::const_iterator Map_Iter_cnst;
//
//    mutable List workloads;  ///< list with first being the most recently used key.
//    Map m_table;             ///< table for fast searching of keys  (contains pointers to list objects, as iterators)
//
//    //const size_t max_size;
//    size_t size{0};  // current size of list (and table)
//
//public:
//    /**
//     * @brief Construct a new LRUCache object
//     *
//     * @param max_size the maximum size of the LRUKeyCache
//     */
//    SimpleLUTKeyCache(size_t /*max_size*/)/*: max_size(max_size)*/ {
//    }
//
//    /**
//     * @brief Add a new workload descriptor to the cache. If the key exists is does NOT replace the old value with the
//     * new one
//     *
//     * @param wl the workload descriptor (key)
//     * @param value the workload value
//     */
//    void add(const K& wl, const V& value) {
//        // If max_size == 0 we effectively disable the cache
//        //if (max_size == 0)
//        //    return;
//
//        const Map_Iter_cnst& map_it{m_table.find(wl)};
//        if (map_it == m_table.cend()) {
//            // Insert items in the list and map
//            workloads.push_front({wl, value});         // adds a new element
//            m_table.insert({wl, workloads.cbegin()});  // would not add a new element if wl is already inside
//            size++;
//
//         //   clean_up_excess_elements();  // if size is exceeded
//        }
//        assert_invariants();
//    }
//
//public:
//    /**
//     * @brief Get a value from the cache.
//     *
//     * @param wl the workload(key) descriptor
//     * @return std::optional<V> the value stored in the cache, or nothing if not available
//     */
//    std::optional<V> get(const K& wl) const {
//        // Check if the workload is in the main table
//        const Map_Iter_cnst& map_it{m_table.find(wl)};
//        if (map_it != m_table.cend()) {
//            // mark_as_most_recently_used(map_it);
//            return (map_it->second->second);  //  second is the value stored in list
//        } else {
//            return std::nullopt;
//        }
//    }
//
//private:
//    /**
//     * @brief Remove a workload from the cache
//     *
//     * @param wl the workload descriptor
//     */
//    void remove(const K& wl) {
//        // Remove the workload from the cache
//        const auto it{m_table.find(wl)};
//
//        if (it != m_table.cend()) {
//            // Return the value
//            m_table.erase(it->first);  // key is first
//        } else {
//            throw std::out_of_range("VPUNN Cache Descriptor out of range");
//        }
//
//        workloads.pop_back();  // Remove the last element
//        size--;                // Update the size
//    }
//
//    /// deletes what exceeds the size
//    void clean_up_excess_elements() {
//        // delete the oldest ones that occupy more space than allowed
//        while (size > max_size) {
//            const auto& oldest_item{workloads.back()};  // last
//            remove(oldest_item.first);                  // key is first in pair
//        }
//    }
//
//    //void mark_as_most_recently_used(const Map_Iter_cnst& map_it) const {
//    //    if (map_it != m_table.cend()) {                      // Move the workload to the beginning of the list
//    //        const List_Iter_cnst& list_it = map_it->second;  // second is the list iterator
//    //        workloads.splice(workloads.cbegin(), workloads, list_it);
//    //    }
//    //}
//
//    void assert_invariants() const {
//        assert(workloads.size() == size);
//        assert(m_table.size() == size);
//    }
//};

}  // namespace VPUNN

#endif  // VPUNN_CACHE
