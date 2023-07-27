// Copyright © 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_UTILS_H
#define VPUNN_UTILS_H

#define UNUSED(expr) (void)(expr)

#include <cassert>
#include <queue>
#include "types.h"

namespace VPUNN {

/**
 * @brief Compute the execution time of a vector of workloads on N processors
 *
 * @tparam T datatype
 * @param n_procesors number of processors (parallelism level)
 * @param tasks_cost a vector of workload costs in cycles
 * @param runtime_overhead per-workload runtime overhead
 * @return T overall execution cycle, e.g. longest thread
 */
template <class T>
T dpu_schedule(const unsigned int n_procesors, const std::vector<T>& tasks_cost, T runtime_overhead = 0) {
    const auto initializer = std::vector<T>(n_procesors, 0);
    // MIN priority queue
    auto queue = std::priority_queue<T, std::vector<T>, std::greater<T>>(initializer.begin(), initializer.end());

    for (long unsigned int idx = 0; idx < tasks_cost.size(); idx++) {
        auto smallest_time = queue.top();
        queue.pop();
        queue.push(smallest_time + tasks_cost[idx] + runtime_overhead);  //@todo: overflow protection mechanism
    }

    // Return the max of the queue -> the execution critical path (last in queue)
    T result = static_cast<T>(0);
    while (!queue.empty()) {
        result = queue.top();
        queue.pop();
    }
    return result;
}

/**
 * @brief Multiply and accumulate an array
 *
 * @tparam T datatype
 * @tparam SIZE array size
 * @param vec the array
 * @return T
 */
template <class T, size_t SIZE>
T multiply_vector(const std::array<T, SIZE>& vec) {
    return std::accumulate(std::begin(vec), std::end(vec), 1, std::multiplies<T>());
}

/**
 * @brief Multiply and accumulate a vector
 *
 * @tparam T datatype
 * @param vec the vector
 * @return T
 */
template <class T>
T multiply_vector(const std::vector<T>& vec) {
    return std::accumulate(std::begin(vec), std::end(vec), 1, std::multiplies<T>());
}

/**
 * @brief Integer ceil division
 *
 * @tparam T datatype
 * @param a numerator
 * @param b denominator
 * @return T
 */
template <class T>
T ceil_division(T a, T b) {
    return (a + b - 1) / b;
}

/**
 * @brief Rounding up a by b
 *
 * @tparam T datatype
 * @param a the value to round
 * @param b the rounding factor
 * @return T
 */
template <class T>
T round_up(T a, T b) {
    return ((a + b - 1) / b) * b;
}

/**
 * @brief Perform an elementwise ceil division and then multiply the results together
 *
 * @param v1
 * @param v2
 * @return unsigned int
 */
inline unsigned int divide_and_multiply_vectors(const std::vector<unsigned int>& v1,
                                                const std::vector<unsigned int>& v2) {
    assert(v1.size() == v2.size());
    unsigned int result = 1;
    for (unsigned int idx = 0; idx < v1.size(); idx++) {
        result *= ceil_division(v1[idx], v2[idx]);
    }
    return result;
}

/**
 * @brief Compute an input tensor dimension from the output dimension and the operation parameters
 *
 * @param output the output dimension
 * @param kernel the kernel size
 * @param total_padding the total padding
 * @param stride the kernel stride
 * @return unsigned int the input dimension
 */
inline unsigned int helper_input_dim(unsigned int output, unsigned int kernel, unsigned int total_padding,
                                     unsigned int stride) {
    // output = floor((input + total_padding - kernel) / stride)) + 1
    int input = ((int)output - 1) * (int)stride - (int)total_padding + (int)kernel;
    if (input < 0) {
        input = 0;
    }
    if (output > 0) {
        assert(output == (input + total_padding - kernel) / stride + 1);
    } else {
        assert(output == (unsigned int)input);  // zero
    }
    return input;
}
}  // namespace VPUNN

#endif  // VPUNN_UTILS_H
