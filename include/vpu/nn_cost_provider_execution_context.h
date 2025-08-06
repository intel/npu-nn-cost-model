// Copyright © 2025 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef NN_COST_PROVIDER_EXECUTION_CONTEXT_H_
#define NN_COST_PROVIDER_EXECUTION_CONTEXT_H_

#include "inference/inference_execution_data.h"

#include <thread>

#include <sstream>
#include <vector>

namespace VPUNN {

struct NNExecutionContext {
    InferenceExecutionData runtime_buffer_data;   ///< buffer data for the inference execution
    std::vector<float> workloads_results_buffer;  ///< buffer for the results of the BATCH inference

    const std::thread::id thread_id;                        ///< thread id for the context
    static inline constexpr size_t prealloc_results{1000};  ///< how much results buffer to pre-alloc

    explicit NNExecutionContext(InferenceExecutionData&& execution_data_specific)
            : runtime_buffer_data(std::move(execution_data_specific)),
              workloads_results_buffer{},
              thread_id(std::this_thread::get_id()) {
        workloads_results_buffer.reserve(prealloc_results);  // reserve space for the results
    };
};

}  // namespace VPUNN

#endif  // NN_COST_PROVIDER_EXECUTION_CONTEXT_H_
