// Copyright @ 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the "Software Package")
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the "third-party-programs.txt" or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_RANGES
#define VPUNN_RANGES

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <iterator>
#include <limits>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "vpu/utils.h"

namespace VPUNN {

/// @brief Policy for defining a range based on divisibility by a number
///
/// @details SmartRanges is intentionally immutable after construction:
/// - All configuration fields are const.
/// - As a consequence, SmartRanges is copyable/movable via constructors, but it is not assignable
///   (operator= is deleted). Assignment would require overwriting const members.
/// - Any modification is expressed by creating a new instance, for example via the helper methods that return a new
///   range (multiply_upper, multiply_lower, add_upper, add_lower) or by constructing a new SmartRanges directly.
/// - Construction normalizes divisors: if divisor or second_divisor are provided as 0, they will be defaulted as 1.
///
/// SmartRanges is designed for non-negative domains:
/// - All usage involves non-negative bounds (channel counts, spatial dimensions, batch sizes, etc.).
/// - Negative bounds are normalized to 0 at construction time via clamp_negative_to_zero().
/// - Negative divisors are normalized to their absolute value at construction (divisibility is sign-independent).

/* coverity[rule_of_five_violation:FALSE] */
class SmartRanges {
private:
    const int lowerBound;  ///<  has to be smaller or equal to upperBound. Always >= 0 (clamped at construction).
    const int upperBound;  ///<  has to be greater or equal to lowerBound. Always >= 0 (clamped at construction).
    const int divisor;     ///<  values have to be divisible with this number. Always > 0 (abs applied at construction).

    /// If present, values have to be divisible with this number too only if
    /// they are larger or equal than its value (it acts as both a divisor and a threshold).
    /// Always > 0 when set (abs applied at construction).
    const std::optional<int> second_divisor;

    /// Cached least common multiple (lcm) of divisor and second_divisor, used as stepping value for the
    /// iterator when second_divisor is enabled.
    /// lcm.has_value() is always true when second_divisor.has_value() is true
    const std::optional<int> lcm;

    /// @brief Clamps a negative value to 0 to enforce the non-negative domain invariant.
    /// @details SmartRanges models non-negative domains (hardware dimensions, channel counts, etc.).
    /// Any negative bound provided at construction is silently normalized to 0.
    /// @param value The value to clamp.
    /// @return The original value if >= 0, otherwise 0.
    static constexpr int clamp_negative_to_zero(int value) {
        return (value < 0) ? 0 : value;
    }

    /// @brief Converts 0 divisor to 1 to keep SmartRanges invariants and avoid modulo-by-zero UB.
    ///  Negative divisors are accepted but treated as their absolute value (divisibility is sign-independent).
    static int normalize_divisor(int divisor_) {
        return (divisor_ == 0) ? 1 : std::abs(divisor_);
    }

    /// @brief Converts an optional second divisor (if present) from 0 to 1 to keep SmartRanges invariants.
    static std::optional<int> normalize_divisor(const std::optional<int>& second_divisor_) {
        const std::optional<int> result{second_divisor_.has_value()
                                                ? std::optional<int>{normalize_divisor(*second_divisor_)}  // normalized
                                                : std::nullopt};
        return result;
    }

    /// @brief Computes the least common multiple (lcm) of divisor and second_divisor if second_divisor is set,
    /// otherwise returns std::nullopt.
    /// @param divisor_ The first divisor, always required by the SmartRanges invariants.
    /// @param second_divisor_ The second divisor, optional according to SmartRanges, but required to compute lcm.
    ///
    /// @return std::optional<int> containing the lcm of divisor and second_divisor if second_divisor is set, or
    /// std::nullopt if second_divisor is not set.
    static std::optional<int> compute_lcm(int divisor_, const std::optional<int>& second_divisor_) {
        const std::optional<int> result{
                second_divisor_.has_value() ? std::optional<int>{std::lcm(divisor_, *second_divisor_)}  // normalized
                                            : std::nullopt};
        return result;
    }

    /// @brief Determines the appropriate stepping value (divisor or LCM) for a given value based on where in the
    /// SmartRange it falls.
    /// @param value The value for which to determine the stepping unit.
    ///
    /// @return The stepping unit (divisor or LCM) that should be used for alignment and iteration based on the value
    /// and the SmartRange configuration.
    int getStepForDomain(int value) const {
        int step{divisor};  // the mandatory  first one

        if (second_divisor.has_value() && value >= *second_divisor) {
            step = *lcm;
        }

        return step;
    }

    /// @brief Checks if advancing from first_value to second_value crosses the second_divisor threshold
    /// @param first_value The starting value before the potential crossing.
    /// @param second_value The value after the potential crossing.
    ///
    /// @return true if the transition from first_value to second_value crosses the second_divisor threshold, false
    /// otherwise.
    bool isAcrossSecondDivisorBorder(int first_value, int second_value) const {
        if (second_divisor.has_value()            //
            && (first_value < *second_divisor)    // starting below
            && (second_value >= *second_divisor)  // jumping above
        ) {
            return true;
        }
        return false;
    }

    /// @brief Aligns the given value up to the next valid value that is greater than or equal to the given value in the
    /// SmartRange, taking into account the current domain (divisor-only or LCM) and the second_divisor threshold.
    /// @param value The value to align up to the next valid value in the SmartRange.
    ///
    /// @return The next valid value in the SmartRange that is greater than or equal to the given value, properly
    /// aligned according to the current domain and second_divisor threshold.
    int alignUnboundedToNextValid(int anyValue) const {
        int candidate = round_up(anyValue, getStepForDomain(anyValue));

        // If we were below threshold but round_up pushed us to or past it,
        // we've left the divisor-only domain and must realign to the LCM grid.
        if (isAcrossSecondDivisorBorder(anyValue, candidate)) {
            candidate = round_up(candidate, *lcm);
        }

        return candidate;
    }

public:
    /// @brief Advances valid_value to the next valid value in the range, or sets it to std::nullopt if past the end.
    /// @pre The current value must already be a valid member of the range (asserted).
    /// @details Because the current value is known to be aligned, the next candidate is computed by adding the
    /// appropriate step size (divisor or LCM) directly, avoiding a redundant round_up in the common case.
    /// When adding a step crosses the second_divisor threshold, the next valid value is exactly the LCM itself,
    /// because the crossing point is always below or at the first LCM multiple (all values are non-negative and
    /// the LCM is the smallest positive common multiple of both divisors).
    std::optional<int> incrementAssumingIsValid(std::optional<int> valid_value) const {
        if (valid_value.has_value()) {
            const int current = *valid_value;
            assert(is_in(current));

            int next = current + getStepForDomain(current);

            // If we crossed the threshold, the next valid value that is divisible by both divisors
            // and greater than or equal to second_divisor is exactly *lcm, by definition of the
            // least common multiple (current is aligned to divisor and we are entering the domain
            // where both divisibility constraints apply).
            if (isAcrossSecondDivisorBorder(current, next)) {
                next = *lcm;
            }

            if (next <= upperBound) {
                valid_value = next;
            } else {
                valid_value.reset();
            }
        }
        return valid_value;
    }

    using value_type = int;

    // Forward declare the iterator class before using it
    class const_iterator;

    /// @brief Constructs a SmartRanges with the given parameters.
    /// @details lowerBound must be <= upperBound if the range is expected to be valid.
    /// Negative bounds are clamped to 0 (SmartRanges models non-negative domains).
    /// @param lowerBound_ Lower bound of the range. Clamped to 0 if negative.
    /// @param upperBound_ Upper bound of the range. Clamped to 0 if negative.
    /// @param divisor_ If set to 0 it is normalized to 1.
    /// @param second_div If set to 0 it is normalized to 1.
    SmartRanges(int lowerBound_, int upperBound_, int divisor_ = 1, std::optional<int> second_div = {})
            : lowerBound(clamp_negative_to_zero(lowerBound_)),
              upperBound(clamp_negative_to_zero(upperBound_)),
              divisor(normalize_divisor(divisor_)),
              second_divisor(normalize_divisor(second_div)),
              lcm(compute_lcm(divisor, second_divisor)) {
    }

    SmartRanges(int lowerBound_, int upperBound_, int divisor_, int second_div)
            : SmartRanges(lowerBound_, upperBound_, divisor_, std::optional<int>{second_div}) {
    }

    SmartRanges(const SmartRanges& other)
            : lowerBound(other.lowerBound),
              upperBound(other.upperBound),
              divisor(other.divisor),
              second_divisor(other.second_divisor),
              lcm(other.lcm) {
    }

    /// SmartRanges is immutable (const members), so moving is equivalent to copying; defaulting avoids misleading
    /// std::move.
    SmartRanges(SmartRanges&& other) noexcept = default;

    SmartRanges(): lowerBound(0), upperBound(0), divisor(1), second_divisor(std::nullopt), lcm(std::nullopt) {
    }

    SmartRanges& operator=(const SmartRanges& other) = delete;
    SmartRanges& operator=(SmartRanges&& other) noexcept = delete;

    ~SmartRanges() {};

    /// @brief: here we verify if a value respect all the range requirements
    /// @param value: the value we want to verify
    /// @param text: a string with information when value does not respect all the requirements, will be cleaned up
    /// initially
    /// @return true if value respect all the requirements, false if not
    bool is_in(int value, std::string& text) const {
        const bool part_of_range{
                is_in(value)};  // initial check without text, to avoid doing extra work when value is in the range

        if (!part_of_range) {
            const bool belongs{(value >= lowerBound) && (value <= upperBound)};
            const bool divisible{belongs ? ((value % divisor) == 0 ? true : false) : true};  // test only if belongs

            const bool second_check_enabled{(belongs && divisible) ? second_divisor.has_value()
                                                                   : false};  // test only if belongs and divisible
            const bool second_divisible_OK{
                    (second_check_enabled && (value >= *second_divisor))  // cases when we are interested
                            ? ((value % (*second_divisor)) == 0 ? true
                                                                : false)  // has to be divisible by second_divisor
                            : true  // OK because either not enabled or other conditions not met
            };
            // error handling only if error, otherwise do not waste time
            text = "";

            if (!belongs) {
                text = " Value :" + std::to_string(value) + " is not in interval [" + std::to_string(lowerBound) +
                       ", " + std::to_string(upperBound) + "]";
            }

            if (!divisible) {
                text += " Value :" + std::to_string(value) + " is not divisible by " + std::to_string(divisor) + "!";
            }
            if (!second_divisible_OK) {
                text += " Value :" + std::to_string(value) + " is not second divisible by " +
                        std::to_string(*second_divisor) + "!";
            }
        }

        return part_of_range;
    }
    bool is_in(int value) const {
        const bool belongs{(value >= lowerBound) && (value <= upperBound)};
        const bool divisible{belongs ? ((value % divisor) == 0 ? true : false) : true};  // test only if belongs

        const bool second_check_enabled{(belongs && divisible) ? second_divisor.has_value()
                                                               : false};  // test only if belongs and divisible
        const bool second_divisible_OK{
                (second_check_enabled && (value >= *second_divisor))         // cases when we are interested
                        ? ((value % (*second_divisor)) == 0 ? true : false)  // has to be divisible by second_divisor
                        : true  // OK because either not enabled or other conditions not met
        };

        const bool part_of_range{belongs && divisible && second_divisible_OK};

        return part_of_range;
    }

    /// @brief Finds the smallest valid value in the range that is greater than or equal to the given value, or
    /// std::nullopt if none.
    /// @param value The value to compare against the valid values in the range.
    ///
    /// @return The smallest valid value in the range that is greater than or equal to the given value, or
    /// std::nullopt if none.
    std::optional<int> roundToNextLarger(int value) const {
        std::optional<int> result{std::nullopt};

        if (betweenLimits(value)) {
            const int candidate = alignUnboundedToNextValid(value);  // mathematical candidate
            if (candidate <= upperBound) {
                result = candidate;
            }
        }

        return result;
    }

    /// @brief Checks whether a value falls within [lowerBound, upperBound].
    /// @param value The value to check.
    ///
    /// @return true if lowerBound <= value <= upperBound, false otherwise
    bool betweenLimits(int value) const {
        return (value >= lowerBound) && (value <= upperBound);
    }

    /// @brief multiplies the upper bound of the range by the given value
    /// @param multiplier used to adjust the range upper bound
    ///
    /// @return a new SmartRanges with the upper bound of the range updated based on the multiplier
    SmartRanges multiply_upper(int multiplier) const {
        SmartRanges newRange{getLowerBound(), getUpperBound() * multiplier, divisor, second_divisor};
        return newRange;
    }

    /// @brief multiplies the lower bound of the range by the given value
    /// @param multiplier used to adjust the range lower bound
    ///
    /// @return a new SmartRanges with the lower bound of the range updated based on the multiplier
    SmartRanges multiply_lower(int multiplier) const {
        SmartRanges newRange{getLowerBound() * multiplier, getUpperBound(), divisor, second_divisor};
        return newRange;
    }

    /// @brief increase or decrease (by adding a negative value) the upper bound of the range by the given value
    /// @param added_term used to adjust the range upper bound
    ///
    /// @return a new SmartRanges with the upper bound of the range updated based on the added_term
    SmartRanges add_upper(int added_term) const {
        SmartRanges newRange{getLowerBound(), getUpperBound() + added_term, divisor, second_divisor};
        return newRange;
    }

    /// @brief increase or decrease (by adding a negative value) the lower bound of the range by the given value
    /// @param added_term used to adjust the range lower bound
    ///
    /// @return a new SmartRanges with the lower bound of the range updated based on the added_term
    SmartRanges add_lower(int added_term) const {
        SmartRanges newRange{getLowerBound() + added_term, getUpperBound(), divisor, second_divisor};
        return newRange;
    }

    int getUpperBound() const {
        return this->upperBound;
    }

    int getLowerBound() const {
        return this->lowerBound;
    }

    /// @brief Returns an iterator to the first valid element in the range, or end() if no valid elements exist.
    /// @details Range-based for loops require begin() / end() to obtain the iterator pair for traversal.
    /// Providing cbegin() / cend() alone is not enough, because the language does not use
    /// them for range-for desugaring. Keeping begin() / end() also makes SmartRanges behave like an STL
    /// container and enables algorithms that expect range.begin() / range.end().
    const_iterator begin() const {
        if (lowerBound > upperBound) {
            return end();
        }
        return const_iterator(*this);
    }

    /// @brief Returns an iterator representing the end position (one-past-the-last) of the range, represented by
    /// std::nullopt in the iterator's current value.
    /// @details Required counterpart to begin() for range-based for loops and standard algorithms.
    const_iterator end() const {
        return const_iterator(*this, std::nullopt);
    }

    /// @brief Returns a const iterator to the first valid element. Equivalent to begin().
    const_iterator cbegin() const {
        return begin();
    }

    /// @brief Returns a const iterator to the end position. Equivalent to end().
    const_iterator cend() const {
        return end();
    }

    /// @brief Generate a vector containing all valid values within this range
    /// @tparam T The type to cast the values to (default: int)
    /// @return std::vector<T> containing all values that satisfy the range constraints, cast to type T
    template <typename T = int>
    std::vector<T> transformSmartRangetoVector() const {
        std::vector<T> result;

        // Uses the iterator to traverse all valid values in the SmartRanges and adds them to the result vector
        for (int value : *this) {
            result.push_back(static_cast<T>(value));
        }

        return result;
    }

    static constexpr int max_limit{std::numeric_limits<int>::max()};  ///< max value accepted for a SmartRange bound

    /// @brief Const input iterator over the valid values defined by a SmartRanges instance
    ///
    /// @details The iterator is intentionally nested so it can be used like STL containers:
    /// SmartRanges::const_iterator it = range.cbegin();
    ///
    /// This is a const-iterator: it yields int values by value and holds a reference to const SmartRanges
    ///
    /// Iterator lifetime/invalidation:
    /// - The iterator stores a reference to the SmartRanges instance that created it.
    /// - Iterators are only valid as long as that SmartRanges object remains alive.
    /// - Moving or destroying the SmartRanges object invalidates all iterators obtained from it.
    ///
    /// Copy assignability:
    /// - This iterator provides a custom copy/move assignment operator that enforces same-range identity.
    /// - Assignment only updates the position (current value); the range reference is never reseated.
    /// - Cross-range assignment throws std::logic_error, preventing silent rebinding to a different range.
    /// - This enables STL algorithms that internally assign iterators (e.g. std::find, std::copy, std::count)
    ///   as long as both iterators originate from the same SmartRanges instance (which is always the case
    ///   for begin/end pairs passed to standard algorithms).
    ///
    /// Post-increment (it++) is intentionally deleted. Use pre-increment (++it) instead.
    class const_iterator {
    public:
        /// Iterator traits for compatibility with STL algorithms. Values are generated on the fly (no backing storage),
        /// and we intentionally expose the weakest standard category needed by algorithms, which is input iterator.
        using iterator_category = std::input_iterator_tag;
        using value_type = SmartRanges::value_type;
        using difference_type = std::ptrdiff_t;
        using pointer = void;
        using reference = value_type;

        /// @brief Defaulted copy/move constructors. Explicitly declared because the user-defined
        /// operator= suppresses the implicit move constructor, and we want both to remain available.
        const_iterator(const const_iterator&) = default;
        const_iterator(const_iterator&&) = default;
        ~const_iterator() = default;

        /// @brief Copy assignment that only updates position, never reseats the range reference.
        /// @throws std::logic_error if the source iterator refers to a different SmartRanges instance.
        /// @details STL algorithms (e.g. std::find via MSVC's _Seek_wrapped) require iterators to be
        /// copy-assignable. Since all iterator pairs passed to such algorithms originate from the same
        /// range (begin/end of the same SmartRanges), this same-range precondition is naturally satisfied
        /// in correct usage. Cross-range assignment is a programming error surfaced as an exception.
        const_iterator& operator=(const const_iterator& other) {
            if (this == &other) {
                return *this;
            }
            if (&range != &(other.range)) {
                throw std::logic_error("Cannot assign iterators from different SmartRanges instances");
            }
            current = other.current;
            return *this;
        }

        /// @brief Move assignment that only updates position, never reseats the range reference.
        /// @throws std::logic_error if the source iterator refers to a different SmartRanges instance.
        const_iterator& operator=(const_iterator&& other) {
            if (this == &other) {
                return *this;
            }
            if (&range != &(other.range)) {
                throw std::logic_error("Cannot assign iterators from different SmartRanges instances");
            }
            current = std::move(other.current);
            return *this;
        }

        /// @brief The dereference operator returns the current value of the iterator. It is only valid to call this
        /// operator if the iterator is not at the end position.
        ///
        /// @details Dereference returns by value (not by reference) because SmartRanges generates values as it goes and
        /// does not own element storage like an STL container. Returning a reference would refer to iterator-owned
        /// state (current) and risks dangling/unstable references after increment (e.g. storing const int& r = *it;
        /// ++it;). Returning by value avoids implying address/stability guarantees that aren't provided by this
        /// generated range.
        ///
        /// This is similar to the STL implementation, where dereferencing end() is undefined behavior
        value_type operator*() const {
            assert(current.has_value());
            return *current;
        }

        /// @brief Pre-increment operator advances the iterator to the next valid value and returns a reference to this
        /// iterator.
        const_iterator& operator++() {
            // incrementAssumingIsValid() assumes the value is already valid, and will advance it to the next valid
            // value (or std::nullopt if past upperBound).
            current = range.incrementAssumingIsValid(current);
            return *this;
        }

        /// @brief Post-increment is intentionally deleted.
        /// Pre-increment (++it) is preferred for input iterators where the previous position is not needed.
        const_iterator operator++(int) = delete;

        /// @brief Equality operators compare iterators for equality/inequality. Two iterators are considered equal if
        /// they refer to the same position in the same SmartRanges instance.
        ///
        /// @details Iterator equality is only meaningful for iterators that refer to the same range instance.
        /// Two iterators at the same "position" (same current value / end state) but produced from different
        /// SmartRanges must not compare equal, otherwise algorithms could mix unrelated iterator domains (e.g.
        /// comparing begin() from one range to end() from another) and terminate early or behave incorrectly.
        bool operator==(const const_iterator& other) const {
            return (&range == &(other.range)) && (current == other.current);
        }

        bool operator!=(const const_iterator& other) const {
            return !(*this == other);
        }

    private:
        /// Only SmartRanges should access the constructor of the iterator, otherwise might cause issues
        friend class SmartRanges;

        /// Stored as a reference (not a pointer) to enforce that the iterator cannot be rebound
        /// to a different SmartRanges instance. The custom operator= verifies same-range identity
        /// and only updates the position, throwing on cross-range assignment.
        const SmartRanges& range;
        std::optional<value_type> current;

        /// @brief begin() constructor
        explicit const_iterator(const SmartRanges& owner)
                : range{owner}, current{owner.roundToNextLarger(owner.getLowerBound())} {
        }

        /// @brief end() constructor
        explicit const_iterator(const SmartRanges& owner, std::nullopt_t): range(owner), current{std::nullopt} {
        }
    };
};

/// @brief holds multiple SmartRanges and allows to check if a value respects at least one of them
///
/// @details MultiSmartRanges is intentionally non-assignable:
/// - It owns a std::vector<SmartRanges>.
/// - SmartRanges is immutable and non-assignable (its operator= is deleted due to const members).
/// - Therefore, std::vector<SmartRanges> cannot be assigned in a well-formed way, so MultiSmartRanges::operator=
///   is deleted too.
/// - Usage consequence: any update must be expressed by constructing a new MultiSmartRanges instance (copy/move
///   construction) or by using methods that return a new object (multiply_lower, multiply_upper).
class MultiSmartRanges {
private:
    std::vector<SmartRanges> ranges;

public:
    using value_type = int;
    MultiSmartRanges(const std::vector<SmartRanges>& ranges_): ranges(ranges_) {
    }

    // Copy constructor
    MultiSmartRanges(const MultiSmartRanges& other): ranges(other.ranges) {
    }

    // Move constructor
    MultiSmartRanges(MultiSmartRanges&& other) noexcept: ranges(std::move(other.ranges)) {
    }

    MultiSmartRanges(): ranges() {
    }

    MultiSmartRanges& operator=(const MultiSmartRanges& other) = delete;
    MultiSmartRanges& operator=(MultiSmartRanges&& other) noexcept = delete;

    ~MultiSmartRanges() {
    }

    /// @brief: here we verify if a value respect all the range requirements
    /// @param value: the value we want to verify
    /// @param text: a string with information when value does not respect all the requirements
    /// @param mask: a vector of bools that indicates which ranges to check
    ///             - If empty: all ranges are checked (mask is set to all true).
    ///             - If smaller than the number of ranges: mask is extended with false for missing entries.
    ///             - If larger than the number of ranges: extra mask entries are ignored.
    /// @return true if value respect at least one range, false if not
    bool is_in(int value, std::string& text, std::vector<bool> mask = {}) const {
        if (mask.empty()) {
            mask.resize(ranges.size(), true);  // all ranges are checked
        } else if (mask.size() < ranges.size()) {
            mask.resize(ranges.size(), false);  // missing entries are not checked
        }
        std::string all_failed_message{""};
        std::string one_range_message{""};
        for (size_t i = 0; i < ranges.size(); ++i) {
            if (mask[i]) {
                if (ranges[i].is_in(value, one_range_message)) {
                    // at least one match found, no message needed
                    return true;  // at least one match found,  EXIT
                } else {
                    // accumulate messages for all failed ranges
                    all_failed_message += " Range " + std::to_string(i) + " check failed: " + one_range_message + "\n";
                }
            }
        }
        text = std::move(all_failed_message);  // all failed ranges
        return false;                          // no match found
    }

    bool is_in(int value, std::vector<bool> mask = {}) const {
        std::string text;
        return is_in(value, text, std::move(mask));
    }

    /// @brief: get a specific range from the vector of ranges
    /// @param index: the index of the range we want to get
    /// @return the range at the given index
    SmartRanges get_range(size_t index) const {
        if (index >= ranges.size()) {
            throw std::out_of_range("Index out of range");
        }
        return ranges[index];
    }

    int getUpperBound() const {
        return std::accumulate(ranges.begin(), ranges.end(), 0, [](int max, const SmartRanges& range) {
            return std::max(max, range.getUpperBound());
        });
    }

    int getLowerBound() const {
        return std::accumulate(ranges.begin(), ranges.end(), 0, [](int min, const SmartRanges& range) {
            return std::min(min, range.getLowerBound());
        });
    }

    MultiSmartRanges multiply_lower(int multiplier) const {
        std::vector<SmartRanges> new_ranges;
        new_ranges.reserve(ranges.size());
        for (const auto& range : ranges) {
            new_ranges.push_back(range.multiply_lower(multiplier));
        }
        return MultiSmartRanges(new_ranges);
    }

    MultiSmartRanges multiply_upper(int multiplier) const {
        std::vector<SmartRanges> new_ranges;
        new_ranges.reserve(ranges.size());
        for (const auto& range : ranges) {
            new_ranges.push_back(range.multiply_upper(multiplier));
        }
        return MultiSmartRanges(new_ranges);
    }
};

}  // namespace VPUNN

#endif  //
