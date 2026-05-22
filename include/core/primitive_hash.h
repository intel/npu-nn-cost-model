// Copyright © 2026 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the "Software Package")
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the "third-party-programs.txt" or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_PRIMITIVE_HASH_H
#define VPUNN_PRIMITIVE_HASH_H

#include "utils.h"

#include <optional>
#include <string>
#include <type_traits>

namespace VPUNN {

/**
 * @brief Static utility class providing FNV-1a based hash functions for primitive data types.
 *
 * PrimitiveHash offers a collection of static methods for hashing primitive values (integers,
 * floats, strings, booleans, enums, and optionals) using the 32-bit FNV-1a (Fowler-Noll-Vo)
 * hash algorithm. All methods follow an incremental hashing pattern: they accept a running
 * hash state @p h and fold in the new value, returning the updated hash.
 *
 * This class serves as the foundational hashing layer used by higher-level hashers such as
 * WorkloadHash to compute composite hashes for VPU workload descriptors (DPU, SHAVE, DMA).
 *
 * The class is not instantiable; all methods are static.
 *
 * @note The FNV-1a constants (fnv_prime = 0x01000193, fnv_offset_basis = 0x811c9dc5) are
 *       defined in utils.h.
 */
class PrimitiveHash {
private:
    PrimitiveHash() = default;  // prevent instantiation
    
public:
    /**
     * @brief Hash a single uint32_t value byte-by-byte using FNV-1a.
     *
     * Processes each of the four bytes of @p value from least-significant to most-significant,
     * XOR-ing each byte into the running hash and multiplying by the FNV-1a prime.
     *
     * @param h     Current running hash state.
     * @param value  The 32-bit unsigned integer to fold into the hash.
     * @return Updated hash after incorporating all four bytes of @p value.
     */
    static uint32_t hash_uint32(uint32_t h, uint32_t value) {
        h = (h ^ (value & 0xFF)) * fnv_prime;
        h = (h ^ ((value >> 8) & 0xFF)) * fnv_prime;
        h = (h ^ ((value >> 16) & 0xFF)) * fnv_prime;
        h = (h ^ (value >> 24)) * fnv_prime;
        return h;
    }

    /**
     * @brief Hash a float value with fractional rescaling for sub-unit precision.
     *
     * Values in the range (-1, 1) (excluding zero) are scaled by 100 before truncation to
     * uint32_t, preserving two decimal digits of precision. Values outside that range (or zero)
     * are cast directly to uint32_t. The resulting integer is then hashed via hash_uint32().
     * @note This approach is only viable for floats that are expected to have small precision
     * And is not intended for general-purpose float hashing, as it will cause collisions for values
     *
     * @param h  Current running hash state.
     * @param c  The float value to fold into the hash.
     * @return Updated hash after incorporating @p c.
     */
    static uint32_t hash_float(uint32_t h, float c) {
        float scaled_value = c * 100.0f;
        uint32_t value =
                (c < 1.0f && c > -1.0f && c != 0.0f) ? static_cast<uint32_t>(scaled_value) : static_cast<uint32_t>(c);
        return hash_uint32(h, value);
    }

    /**
     * @brief Hash a string character-by-character using the FNV-1a scheme.
     *
     * Each character of @p s is XOR-ed into the running hash and multiplied by the
     * FNV-1a prime, producing an order-dependent hash of the string contents.
     *
     * @param h  Current running hash state.
     * @param s  The string to fold into the hash.
     * @return Updated hash after incorporating all characters of @p s.
     */
    static uint32_t hash_string(uint32_t h, const std::string& s) {
        for (const char c : s) {
            h ^= static_cast<uint32_t>(static_cast<unsigned char>(c));
            h *= fnv_prime;
        }
        return h;
    }

    /**
     * @brief Hash an enum value by casting it to uint32_t.
     *
     * @tparam T  An enumeration type (scoped or unscoped).
     * @param h      Current running hash state.
     * @param value  The enum value to fold into the hash.
     * @return Updated hash after incorporating the numeric representation of @p value.
     */
    template <typename T>
    static uint32_t hash_enum(uint32_t h, T value) {
        return hash_uint32(h, static_cast<uint32_t>(value));
    }

    /**
     * @brief Hash a boolean value (true maps to 1, false to 0).
     *
     * @param h      Current running hash state.
     * @param value  The boolean value to fold into the hash.
     * @return Updated hash after incorporating @p value.
     */
    static uint32_t hash_bool(uint32_t h, bool value) {
        return hash_uint32(h, value ? 1 : 0);
    }

    /**
     * @brief Hash an optional value, distinguishing engaged from disengaged state.
     *
     * A presence flag (true/false) is hashed first so that a disengaged optional always
     * produces a different hash than an engaged one. When the optional holds a value,
     * the appropriate type-specific hasher (enum, bool, or uint32) is dispatched via
     * constexpr-if.
     *
     * @tparam T  The contained type; must be an enum, bool, or integer-convertible type.
     * @param h    Current running hash state.
     * @param opt  The optional value to fold into the hash.
     * @return Updated hash after incorporating the presence flag and, if engaged, the value.
     */
    template <typename T>
    static uint32_t hash_optional(uint32_t h, const std::optional<T>& opt) {
        if (opt.has_value()) {
            h = hash_bool(h, true);
            if constexpr (std::is_enum_v<T>) {
                h = hash_enum(h, opt.value());
            } else if constexpr (std::is_same_v<T, bool>) {
                h = hash_bool(h, opt.value());
            } else {
                h = hash_uint32(h, static_cast<uint32_t>(opt.value()));
            }
        } else {
            h = hash_bool(h, false);
        }
        return h;
    }

};

} // namespace VPUNN

#endif // VPUNN_PRIMITIVE_HASH_H
