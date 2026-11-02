// Copyright © 2026 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_NN_COST_CACHE_MANAGER_H_
#define VPUNN_NN_COST_CACHE_MANAGER_H_

#include <cassert>
#include <optional>
#include <string>
#include <type_traits>  // for std::is_same_v
#include <utility>      // for std::forward
#include <variant>      // for std::variant

#include "core/cache.h"
#include "core/dpu_map_type_selector.h"
#include "core/vecdescriptor_map_type_selector.h"  // for LRUCache<std::vector<float>, float>
#include "vpu/types.h"

namespace VPUNN {

/// @brief Owns and selects between the two cache backends used to memorize NN inference results.
///
/// Two backends exist: keying by descriptor (`DescriptorCache`) and keying directly on `DPUWorkload`
/// (`WorkloadCache`). Only one backend is ever engaged for a given instance, decided once at construction time from
/// the constructor flag (see uses_workload_hash_cache()).
///
/// Since only one backend is ever engaged, both live in a single `std::variant`, so the storage reserved is the max of
/// the two backend sizes.
class NNCostCacheManager {
private:
    using DescriptorCache = LRUCache<std::vector<float>, float>;
    using WorkloadCache = LRUCache<DPUWorkload, float>;
    using ActiveCache = std::variant<DescriptorCache, WorkloadCache>;

public:
    NNCostCacheManager(const bool use_workload_hash_cache, const unsigned int cache_size,
                       const std::string& cache_filename, const std::string& paired_cache_filename)
            : active_cache(
                      build_active_cache(use_workload_hash_cache, cache_size, cache_filename, paired_cache_filename)) {
    }

    NNCostCacheManager(const bool use_workload_hash_cache, const unsigned int cache_size,
                       const char* cache_data = nullptr, const size_t cache_data_length = 0)
            : active_cache(build_active_cache(use_workload_hash_cache, cache_size, cache_data, cache_data_length)) {
    }

    const AccessCounter& getPreloadedCacheCounter() const {
        return uses_workload_hash_cache() ? std::get<WorkloadCache>(active_cache).getPreloadedCacheCounter()
                                          : std::get<DescriptorCache>(active_cache).getPreloadedCacheCounter();
    }

    bool uses_workload_hash_cache() const noexcept {
        return has_workload_cache();
    }

    /// @brief exposes which cache backend was actually constructed, for testing the construction invariant
    bool has_descriptor_cache() const noexcept {
        return std::holds_alternative<DescriptorCache>(active_cache);
    }

    /// @brief exposes which cache backend was actually constructed, for testing the construction invariant
    bool has_workload_cache() const noexcept {
        return std::holds_alternative<WorkloadCache>(active_cache);
    }

    /// @brief looks up the cache entry for `key`. `Key` selects the backend at compile time:
    /// `DPUWorkload` queries the workload-hash cache, `std::vector<float>` queries the descriptor cache.
    /// The caller decides which one to use, normally guided by uses_workload_hash_cache().
    template <typename Key>
    std::optional<float> get(const Key& key, std::string* source = nullptr) const {
        static_assert(std::is_same_v<Key, DPUWorkload> || std::is_same_v<Key, std::vector<float>>,
                      "NNCostCacheManager key must be DPUWorkload (workload-hash cache) or "
                      "std::vector<float> (descriptor cache)");
        using ExpectedBackend = std::conditional_t<std::is_same_v<Key, DPUWorkload>, WorkloadCache, DescriptorCache>;

        check_active_backend<ExpectedBackend>();
        return std::get<ExpectedBackend>(active_cache).get(key, source);
    }

    /// @brief adds `value` under `key`. `Key` selects the backend at compile time:
    /// `DPUWorkload` queries the workload-hash cache, `std::vector<float>` queries the descriptor cache.
    /// The caller decides which one to use, normally guided by uses_workload_hash_cache().
    template <typename Key>
    void add(const Key& key, const float value) {
        static_assert(std::is_same_v<Key, DPUWorkload> || std::is_same_v<Key, std::vector<float>>,
                      "NNCostCacheManager key must be DPUWorkload (workload-hash cache) or "
                      "std::vector<float> (descriptor cache)");
        using ExpectedBackend = std::conditional_t<std::is_same_v<Key, DPUWorkload>, WorkloadCache, DescriptorCache>;

        check_active_backend<ExpectedBackend>();
        return std::get<ExpectedBackend>(active_cache).add(key, value);
    }

private:
    /// @brief guards against using a key type that does not match the active backend
    template <typename ExpectedBackend>
    void check_active_backend() const {
        assert(std::holds_alternative<ExpectedBackend>(active_cache));
    }

    /// @brief constructs the active backend directly in place, in the alternative selected by
    /// `use_workload_hash_cache`; the other alternative is never constructed.
    template <typename... Args>
    static ActiveCache build_active_cache(const bool use_workload_hash_cache_flag, Args&&... args) {
        if (use_workload_hash_cache_flag) {
            return ActiveCache{std::in_place_type<WorkloadCache>, std::forward<Args>(args)...};
        }
        return ActiveCache{std::in_place_type<DescriptorCache>, std::forward<Args>(args)...};
    }

private:
    ActiveCache active_cache;  ///< holds whichever backend is active; the other is never constructed
};

}  // namespace VPUNN

#endif  // VPUNN_NN_COST_CACHE_MANAGER_H_
