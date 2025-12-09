// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef DMA_PREPROCESSING_H
#define DMA_PREPROCESSING_H

// #include <math.h>
#include <vpu/types.h>
#include "vpu/dma_types.h"

#include <sstream>  // for error formating
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

// #include "vpu/validation/dpu_operations_validator.h"

namespace VPUNN {

/**
 * @brief Class to transform DMANNWorkload objects into NN descriptors that can be used by the NN
 *
 * @tparam T datatype of the descriptor
 */
template <class T, class DMADesc>
class IPreprocessingDMA {
protected:
    const size_t size_of_current_descriptor;  ///< size of the descriptor, used to initialize the processed_output

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
     * @brief Transform a DMANNWorkload_NPU27 into a DMANNWorkload descriptor
     *
     * @param workload a DMANNWorkload to be transformed
     * @param debug_offset [out] will store how many elements were actually written
     * @return std::vector<T>& a DMANNWorkload descriptor
     */
    virtual const std::vector<T> generate_descriptor(const DMADesc& workload, size_t& debug_offset) const = 0;

public:
    /// @brief provides the interface number this instance implements
    /// @returns the interface version
    virtual int interface_version() const = 0;

    /// @brief Construct a new Preprocessing object
    IPreprocessingDMA(size_t size_of_descriptor) : size_of_current_descriptor{size_of_descriptor} {};

    /**
     * @brief Return the size of the DPUWorkload descriptors
     *
     * @return unsigned int
     */
    unsigned int output_size() const {
        return static_cast<unsigned int>(size_of_current_descriptor);
    }

    /**
     * @brief Transform a DMANNWorkload into a DMANNWorkload descriptor
     *
     * @param workload the DMANNWorkload to transform
     * @return std::vector<T>& a DMANNWorkload descriptor
     */
    const std::vector<T> transformSingle(const DMADesc& workload) const {
        size_t unsused_output_written_offset{};
        return generate_descriptor(workload, unsused_output_written_offset);
    };

    /// @brief default virtual destructor, we need this because this class is abstract
    virtual ~IPreprocessingDMA() = default;

    /**
     * @brief Transform DMAWorkloads into a DMAWorkload descriptors
     *
     * @param workloads a vector of DMAWorkloads
     * @param pad the amount of padding to add (default 1), the batch size
     * @return  DPUWorkload descriptors, vector . RVO expected
     */
    const std::vector<T> transformBatch(const std::vector<DMADesc>& workloads, unsigned int pad = 1) const {
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
