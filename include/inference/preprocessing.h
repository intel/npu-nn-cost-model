// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef PREPROCESSING_H
#define PREPROCESSING_H

#include <vpu/types.h>
#include <sstream>  // for error formating
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

#include <unordered_set>

namespace VPUNN {

/**
 * @brief Class to transform DPUWorkload objects into NN descriptors that can be used by the NN
 *
 * @tparam T datatype of the descriptor
 */
template <class T>
class Preprocessing {
private:
    const size_t size_of_current_descriptor;  ///< size of the descriptor, used to initialize the processed_output

protected:
    std::vector<T> makeSizedContainer() const {
        std::vector<T> processed_output_;
        processed_output_.resize(output_size(), static_cast<T>(0.0));
        return processed_output_;
    }

    /** @brief verifies that the space available to produce results is at least expected_size
     * @param expected_size minimum size to be OK
     * @returns true, otherwise will throw
     * @throws out_of_range if the internal space is not sufficient
     */
    inline bool check_and_throw_size(size_t expected_size) const {
        if (output_size() < expected_size) {
            std::stringstream buffer;
            buffer << "[ERROR]:Transformation aborted!, buffer to fill has size:  " << output_size()
                   << " and is smaller than the size it will be written: " << expected_size
                   << ". Potential error in interface or the loaded NN model expects wrong data! "
                   << " File: " << __FILE__ << " Line: " << __LINE__;
            std::string details = buffer.str();
            throw std::out_of_range(details);
        }
        return true;
    }

    /**
     * @brief Transform a DPUWorkload into a DPUWorkload descriptor
     *
     * @param workload a DPUWorkload to be transformed
     * @param debug_offset [out] will store how many elements were actually written
     * @return std::vector<T>& a DPUWorkload descriptor
     */
    virtual const std::vector<T> generate_descriptor(const DPUWorkload& workload, size_t& debug_offset) const = 0;

public:
    /// @brief provides the interface number this instance implements
    /// @returns the interface version
    virtual int interface_version() const = 0;

    /// @brief Constructor, initializes the processed_output with the size of the descriptor
    Preprocessing(size_t size_of_descriptor): size_of_current_descriptor{size_of_descriptor} {};

    /**
     * @brief Return the size of the DPUWorkload descriptors
     *
     * @return unsigned int
     */
    unsigned int output_size() const {
        return static_cast<unsigned int>(size_of_current_descriptor);
    }

public:
    /**
     * @brief Transform a DPUWorkload into a DPUWorkload descriptor
     *
     * @param workload the DPUWorkloadto transform
     * @return std::vector<T> a DPUWorkload descriptor
     */
    const std::vector<T> transformSingle(const DPUWorkload& workload) const {
        size_t unused_output_written_offset{};
        return generate_descriptor(workload, unused_output_written_offset);
    };

    /** @brief default virtual destructor, we need this because this class is abstract
     */
    virtual ~Preprocessing() = default;

    /**
     * @brief Transform DPUWorkloads into a DPUWorkload descriptors
     *
     * @param workloads a vector of DPUWorkloads
     * @param pad the amount of padding to add (default 1), the batch size
     * @return  DPUWorkload descriptors, vector . RVO expected
     */
    const std::vector<T> transformBatch(const std::vector<DPUWorkload>& workloads, unsigned int pad = 1) const {
        assert(pad > 0);  // pad must be at least 1
        const auto total_workloads{round_up(static_cast<unsigned int>(workloads.size()), pad)};

        std::vector<T> batch_processed_output;  ///< descriptor like processed_output, but for batch.
        batch_processed_output.reserve(total_workloads * output_size());

        // zero filled variable with the same size as processed_output
        const std::vector<T> zero_vector = [](const auto elements_count) {
            std::vector<T> zero;
            zero.resize(elements_count, static_cast<T>(0.0));
            return zero;
        }(this->output_size());

        for (long unsigned int idx = 0; idx < total_workloads; ++idx) {
            const std::vector<T> one_descriptor{(idx < workloads.size()) ? transformSingle(workloads[idx])
                                                                         : zero_vector};
            assert(one_descriptor.size() == output_size());
            // copy this result into the big batch descriptor
            std::copy(one_descriptor.cbegin(), one_descriptor.cend(), std::back_inserter(batch_processed_output));
        }
        return batch_processed_output;  // RVO
    }
};

}  // namespace VPUNN
#endif
