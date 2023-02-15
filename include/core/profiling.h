// Copyright © 2022 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef CORE_PROFILING_H
#define CORE_PROFILING_H

#include <chrono>

namespace VPUNN {

/**
 * @brief Get the current timestamp
 *
 * @return auto
 */
inline auto tick() {
    return std::chrono::high_resolution_clock::now();
}

/**
 * @brief Get the current timestamp and return the elapsed time with the previous timestamp
 *
 * @param t1 the previous timestamp
 * @return double the elapsed time in millisecond
 */
inline double tock(std::chrono::high_resolution_clock::time_point t1) {
    auto t2 = tick();
    /* Getting number of milliseconds as a double. */
    std::chrono::duration<double, std::milli> ms_double = t2 - t1;
    return ms_double.count();
}

/**
 * @brief Syncronous timeout class
 *
 * @tparam T a duration type (example std::milli)
 */
template <class T>
class SyncStopWatch {
private:
    bool running = false;
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time;

public:
    /**
     * @brief Start the SyncStopWatch
     *
     * @return SyncStopWatch&
     */
    SyncStopWatch& start() {
        start_time = tick();
        running = true;
        return *this;
    }

    /**
     * @brief Stop the SyncStopWatch
     *
     * @return SyncStopWatch&
     */
    SyncStopWatch& stop() {
        running = false;
        return *this;
    }

    /**
     * @brief Reset the SyncStopWatch
     *
     * @return SyncStopWatch&
     */
    SyncStopWatch& reset() {
        start_time = tick();
        running = false;
        return *this;
    }

    /**
     * @brief Compute the elapsed time from the start
     *
     * @return unsigned int
     */
    unsigned int interval() {
        auto now = tick();
        std::chrono::duration<float, T> elapsed_time = now - start_time;
        return static_cast<unsigned int>(elapsed_time.count());
    }
};

}  // namespace VPUNN

#endif  // CORE_PROFILING_H
