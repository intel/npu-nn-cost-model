// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_LOGGER_H
#define VPUNN_LOGGER_H

#include <iostream>
#include <sstream>  // for error formating
#include <string>
#include "vpu/types.h"
#include "vpu/utils.h"

#include <mutex>

namespace VPUNN {

/**
 * @brief Logger verbosity levels (same as VPUX)
 *
 */
enum class LogLevel {
    None = 0,     // Logging is disabled
    Fatal = 1,    // Used for very severe error events that will most probably
                  // cause the application to terminate
    Error = 2,    // Reporting events which are not expected during normal
                  // execution, containing probable reason
    Warning = 3,  // Indicating events which are not usual and might lead to
                  // errors later
    Info = 4,     // Short enough messages about ongoing activity in the process
    Debug = 5,    // More fine-grained messages with references to particular data
                  // and explanations
    Trace = 6,    // Involved and detailed information about execution, helps to
                  // trace the execution flow, produces huge output
};

/**
 * @brief Convert a LegLevel to an uppercase string
 *
 * @param level
 * @return const std::string
 */
const std::string toString(LogLevel level);

/**
 * @brief A class that implements a cout-enabled interface for logging
 *
 */
/* coverity[rule_of_three_violation:FALSE] */
class LoggerStream {
    bool _enabled;
    LogLevel _logLevel;
    std::ostringstream  _buffer;
    std::ostringstream* pout{nullptr};  ///< second output
    static std::mutex cout_mutex;       ///< Mutex for protecting std::cout

public:
    /**
     * @brief Construct a new LoggerStream object
     *
     * @param level verbosity level
     * @param enabled
     */
    LoggerStream(LogLevel level, bool enabled, std::ostringstream* buff = nullptr)
            : _enabled(enabled), _logLevel(level), pout(buff) {
        _buffer << "[VPUNN " << toString(_logLevel) << "]: ";
    }

    /**
     * @brief overload of operator <<
     *
     * @tparam T
     * @param msg the message to print
     * @return LoggerStream&
     */
    template <typename T>
    LoggerStream& operator<<(const T& msg) {
        _buffer << msg;
        return *this;
    }

    /**
     * @brief Destroy the LoggerStream object
     *
     */
    ~LoggerStream() {
        if (_enabled || pout) {
            std::lock_guard<std::mutex> lock(cout_mutex);
            if (_enabled) {
                std::cout << _buffer.str() << std::endl;
            }
            if (pout) {
                *pout << _buffer.str() << std::endl;
            }
        }
    }
};

/**
 * @brief Logger class
 *
 */
class Logger final {
private:
    static LogLevel _logLevel;  // = LogLevel::None;

    static std::ostringstream buffer;
    static std::ostringstream* active_second_logger;  ///< logs into a string , deactivated by default
    static std::mutex log_mutex;                      // Mutex to protect shared resources

public:
    static void clear2ndlog() {
        std::lock_guard<std::mutex> lock(log_mutex);
        buffer.str("");
    }
    static std::string get2ndlog() {
        std::lock_guard<std::mutex> lock(log_mutex);
        return buffer.str();
    }
    static void activate2ndlog() {
        std::lock_guard<std::mutex> lock(log_mutex);
        active_second_logger = &buffer;
    }
    static void deactivate2ndlog() {
        std::lock_guard<std::mutex> lock(log_mutex);
        active_second_logger = nullptr;
    }
    /**
     * @brief Initialize the Logger
     *
     * @param level verbosity
     */
    static void initialize(LogLevel level = LogLevel::Warning) {
        setLevel(level);
    }

public:
    /**
     * @brief Return the verbosity level
     *
     * @return auto
     */
    static auto level() {
        std::lock_guard<std::mutex> lock(log_mutex);
        return _logLevel;
    }

    /**
     * @brief Set the verbosity level
     *
     * @param level
     */
    static void setLevel(LogLevel level) {
        std::lock_guard<std::mutex> lock(log_mutex);
#ifdef VPUNN_ENABLE_LOGGING
        _logLevel = level;
#else
        UNUSED(level);
        _logLevel = LogLevel::None;
#endif
    }

    /**
     * @brief get if the Logger is enabled or not
     *
     * @return true
     * @return false
     */
    static bool enabled() {
#ifdef VPUNN_ENABLE_LOGGING
        return true;
#else
        return false;
#endif
    }

private:
    static auto log(LogLevel level) {
        std::lock_guard<std::mutex> lock(log_mutex);
        bool enabled = level <= _logLevel;
        return LoggerStream(level, enabled, active_second_logger);
    }

public:
    /**
     * @brief Fatal error
     *
     * @return auto
     */
    static auto fatal() {
        return log(LogLevel::Fatal);
    }

    /**
     * @brief Error message
     *
     * @return auto
     */
    static auto error() {
        return log(LogLevel::Error);
    }
    /**
     * @brief Warning message
     *
     * @return auto
     */
    static auto warning() {
        return log(LogLevel::Warning);
    }
    /**
     * @brief Info message
     *
     * @return auto
     */
    static auto info() {
        return log(LogLevel::Info);
    }
    /**
     * @brief Debug message
     *
     * @return auto
     */
    static auto debug() {
        return log(LogLevel::Debug);
    }
    /**
     * @brief Trace message
     *
     * @return auto
     */
    static auto trace() {
        return log(LogLevel::Trace);
    }
};

/**
 * @brief Throw an error message
 *
 * @tparam T
 * @param msg
 */
template <class T>
void throw_error(std::string msg) {
    VPUNN::Logger::error() << msg;
    throw T(msg);
}

}  // namespace VPUNN

#endif  // VPUNN_LOGGER_H
