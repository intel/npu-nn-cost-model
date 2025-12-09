// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef DMA_PREPROCESSING_INSERTER_H
#define DMA_PREPROCESSING_INSERTER_H

#include <vpu/types.h>
#include <sstream>  // for error formating
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

#include <unordered_set>
#include "inference/dma_preprocessing.h"

namespace VPUNN {

/**
 * @brief Intermediate base class for a real Preprocessing
 * Using  curiously recurring template pattern (CRTP) provides static polymorphism
 *
 *
 * @tparam T datatype of the descriptor
 * @tparam D the derived class. Real/Final processors are derived from this
 */
template <class T, class D, class DMADesc>
class PreprocessingInserterDMA : public IPreprocessingDMA<T, DMADesc> {
protected:
    /**
     * @brief This function computes the size of the DMANNWorkload features to feed to the NN
     * This is the size of a particular descriptor for a [particular] cost NN input
     *
     * @return unsigned int  size of the descriptor for this workload according to NN input expectations
     */
    size_t calculate_size() const {
        const DMADesc dummy_wl{};
        size_t size_required = 0;
        // use the derived class to run a mock of transform only for finding how much it fills in
        static_cast<const D*>(this)->template transformOnly<true>(dummy_wl, size_required);
        return size_required;
    }

protected:
public:
    using IPreprocessingDMA<T, DMADesc>::transformSingle;  ///< exposes the non virtual transform for  workloads vector

    PreprocessingInserterDMA(size_t size_of_descriptor): IPreprocessingDMA<T, DMADesc>(size_of_descriptor) {};
    virtual ~PreprocessingInserterDMA() = default;

    int interface_version() const override {
        return D::getInterfaceVersion();  // expected static method
    }

    /**
     * @brief Transform a DMANNWorkload into a DMANNWorkload descriptor
     *
     * @param workload a dedicated workload type
     * @param debug_offset [out] is the offset where a new value can be written. interpreted as how many positions were
     * written
     * @return std::vector<T>& a DMANNWorkload descriptor
     */
    const std::vector<T> generate_descriptor(const DMADesc& workload, size_t& debug_offset) const override {
        std::vector<T> descriptor{this->makeSizedContainer()};
        this->check_and_throw_size(static_cast<const D*>(this)->size_of_descriptor);  // will throw in case not matching

        static_cast<const D*>(this)->template transformOnly<false>(workload, debug_offset, descriptor);
        return descriptor;
    };
};

}  // namespace VPUNN
#endif
