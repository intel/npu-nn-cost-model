// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_VPU_CHECKER_UTILS_H
#define VPUNN_VPU_CHECKER_UTILS_H

#include "vpu/types.h"

#include <iostream>  // std::cout
#include <sstream>   // std::stringstream
#include <string>    // std::string

#include "data_dpu_operation.h"
#include "vpu/ranges.h"
#include "vpu/validation/interface_valid_values.h"

namespace VPUNN {

/// @brief simple checker mechanism that logs textual info and remembers if a error was recorded.
class Checker {
private:
    bool clean_status{true};       ///< true if no problems were found since reset
    std::string acc_findings{""};  ///< textual info gathered since last reset

    static bool print_tags;  ///< set to false to avoid the addition of [CHECK] tags. true by default

public:
    /// sets a new mode and returns the current mode
    static bool set_print_tags(bool new_mode) {
        const auto old_mode = print_tags;
        print_tags = new_mode;
        return old_mode;
    }

    /// cleans  up the history
    /// @returns the state before reset
    bool reset() {
        auto prev_state{clean_status};

        clean_status = true;
        acc_findings = "";

        return prev_state;
    }
    /// @returns true if the checker has no error recorded
    bool is_clean() const {
        return clean_status;
    }
    /// marks the checker with error and ads a textual info
    /// @param info the string with information
    void add_check_failed(const std::string& info) {
        clean_status = false;  // at least one problem

        if (Checker::print_tags) {
            acc_findings += ("\n[CHECK FAILED]: " + info + " [END CHECK]");
        } else {
            acc_findings += info;
        }
    }

    /// @returns the string containing the textual information that was logged (since reset).
    std::string findings() const {
        return acc_findings;
    }

    /// functor (class with one func) for showing a value of type T
    template <class T, class Enable = void>
    struct Show {
        static std::string show_value(const T& item) {
            std::stringstream buffer;
            buffer << item;
            return buffer.str();
        }
    };

    /// Specialisation of Show for Enums that have also the mapp to text operation
    template <class Enum>
    struct Show<Enum, typename std::enable_if<std::is_enum<Enum>::value>::type> {
        static std::string show_value(const Enum& item) {
            std::stringstream buffer;
            buffer << static_cast<int>(item);
            try {
                const auto& enum_text{mapToText<Enum>().at(static_cast<int>(item))};
                buffer << " {" << enum_text << "}";
            } catch (const std::exception&) {  // no extra info
            }

            return buffer.str();
        }
    };

    /// checks if the item belongs to a container. If not present it will record an error/finding
    template <class T>
    bool check_is_in_list(const T& item, const Values<T>& container, const std::string& what) {
        const auto found = std::find(container.begin(), container.end(), item) != container.end();

        if (!found) {
            std::stringstream buffer;
            buffer << what << " with value: " << Show<T>::show_value(item)
                   << " is not found in allowed list: " << show_compact(container);
            const std::string details = buffer.str();

            add_check_failed(details);
        }

        return found;
    }

    /// checks if the item belongs to a closed interval. If not present it will record an error/finding
    template <class T>
    bool check_is_in_interval(const T& item, const std::pair<T, T>& interval, const std::string& what) {
        const bool belongs{(item >= interval.first) && (item <= interval.second)};

        if (!belongs) {
            std::stringstream buffer;
            buffer << what << " with value: " << Show<T>::show_value(item) << " is not found in interval: [ "
                   << interval.first << " , " << interval.second << " ]";
            const std::string details = buffer.str();

            add_check_failed(details);
        }

        return belongs;
    }

    /// checks if the item respects all the rules of a smart range. If not it will record an error/finding
    bool check_is_in_requirements(const int item, const MultiSmartRanges& range, const std::string& what) {
        std::string message{""};
        bool result{range.is_in(item, message)};

        if (!result) {
            std::stringstream buffer;
            buffer << what << " is not ok because: " << message;
            const std::string details = buffer.str();
            add_check_failed(details);
        }
        return result;
    }

    /// checks for equality. If not present it will record an error/finding
    template <class T>
    bool check_is_equal(const T& item, const T& right_side, const std::string what) {
        const auto equal{item == right_side};
        if (!equal) {
            std::stringstream buffer;
            buffer << what << " with value: " << Show<T>::show_value(item)
                   << " is not equal to : " << Show<T>::show_value(right_side);
            const std::string details = buffer.str();

            add_check_failed(details);
        }

        return equal;
    }

private:
    template <class T>
    std::string show_compact(const Values<T>& container) const {
        std::stringstream buffer;
        buffer << "[ ";
        if (container.size() > 5U) {  // brief
            auto elem = container.cbegin();
            buffer << Show<T>::show_value(*(elem + 0)) << " , ";
            buffer << Show<T>::show_value(*(elem + 1)) << " , ";
            buffer << " ... , " << Show<T>::show_value(*(container.cend() - 1));

        } else {  // full
            for (const auto& elem : container) {
                buffer << Show<T>::show_value(elem) << " , ";
            }
        }
        buffer << "]";

        return buffer.str();
    }
};

}  // namespace VPUNN

#endif  //
