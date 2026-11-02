// Copyright © 2026 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_UTEST_LOGGER_STREAM_H
#define VPUNN_UTEST_LOGGER_STREAM_H

#include "logger/logger_level.h"

#include <ios>
#include <ostream>
#include <sstream>

namespace VPUNN_unit_tests {
namespace utest_log {

/**
 * @brief A lightweight stream buffer that conditionally emits log messages.
 *
 * The stream stores user content in an internal `std::ostringstream` until the
 * object is destroyed. At destruction time, it emits the accumulated message
 * to the shared logger only when the current logger threshold allows the
 * represented severity level.
 */
class Stream {
public:
    /**
     * @brief Constructs a stream for a specific log severity.
     *
     * @param level Severity associated with the messages emitted through this
     *              stream instance.
     */
    explicit Stream(Level::level_enum level);

    /**
     * @brief Flushes the buffered message to the shared logger when enabled.
     */
    ~Stream();

    /**
     * Copying is disabled because this object owns a temporary buffer that is flushed once
     * in the destructor. A copy would try to log the same message twice.
     */
    Stream(const Stream&) = delete;
    /**
     * Copy assignment is disabled because the buffer is a one-shot message. Assigning it would
     * duplicate ownership of the same pending log entry.
     */
    Stream& operator=(const Stream&) = delete;
    /**
     * Moving is disabled because the destructor is the single place where the message is flushed.
     * A moved object would make ownership unclear.
     */
    Stream(Stream&&) = delete;
    /**
     * Move assignment is disabled for the same reason: one message must have one owner, and the
     * flush happens in exactly one destructor. This also avoids the compiler warning caused by
     * the class storing const members (`level_` and `enabled_`), which cannot be assigned by the
     * implicitly generated move assignment operator.
     */
    Stream& operator=(Stream&&) = delete;

    /**
     * @brief Appends a value to the buffered message if logging is enabled.
     *
     * @tparam T Type of the value being streamed.
     * @param value Value to append to the internal message buffer.
     * @return Reference to the same stream for chaining.
     */
    template <typename T>
    Stream& operator<<(const T& value) {
        buffer_ << value;
        return *this;
    }

    /**
     * @brief Handles stream manipulators such as `std::endl` or `std::flush`.
     *
     * @param manipulator Std stream manipulator to process on the local buffer.
     * @return Reference to the same stream for chaining.
     */
    Stream& operator<<(std::ostream& (*manipulator)(std::ostream&));

    /**
     * @brief Handles `std::ios_base` manipulators.
     *
     * @param manipulator Std stream manipulator to process on the local buffer.
     * @return Reference to the same stream for chaining.
     */
    Stream& operator<<(std::ios_base& (*manipulator)(std::ios_base&));

private:
    /// Severity associated with this stream instance.
    const Level::level_enum level_;

    /// Accumulates the message payload until the stream is destroyed.
    std::ostringstream buffer_{};
};

}  // namespace utest_log
}  // namespace VPUNN_unit_tests

#endif  // VPUNN_UTEST_LOGGER_STREAM_H
