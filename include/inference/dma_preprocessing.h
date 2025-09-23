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
    std::vector<T> processed_output;  ///< descriptor, represents the input to the cost model NN.
    std::vector<T> zero_vector;       ///< zero filled variable with the same size as processed_output

    size_t probable_batch_size{1};          ///< most probable batch size
    std::vector<T> batch_processed_output;  ///< descriptor like processed_output, but for batch.

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
    virtual const std::vector<T>& generate_descriptor(const DMADesc& workload, size_t& debug_offset) = 0;

public:
    /// @brief provides the interface number this instance implements
    /// @returns the interface version
    virtual int interface_version() const = 0;

    /// @brief Construct a new Preprocessing object
    IPreprocessingDMA() {};

    /**
     * @brief Return the size of the DPUWorkload descriptors
     *
     * @return unsigned int
     */
    unsigned int output_size() const {
        return (unsigned int)processed_output.size();
    }
    /**
     * @brief Set the size of the DPUWorkload descriptors
     *
     * @param size
     */
    void set_size(size_t size) {
        processed_output.resize(size, static_cast<T>(0.0));
        zero_vector.resize(size, static_cast<T>(0.0));
    }
    /**
     * @brief reset the DPUWorkload descriptor
     *
     */
    void reset() {
        std::fill(processed_output.begin(), processed_output.end(), static_cast<T>(0.0));
    }

    /**
     * @brief Set the most probable value for batch. Will ensure cache memory is present according to this
     *
     * @param batch_size how many workloads are in a batch
     */
    void set_probable_batch(size_t batch_size) {
        probable_batch_size = batch_size;
        const auto requested_size = batch_size * output_size();
        batch_processed_output.reserve(requested_size);
        batch_processed_output.resize(0);  // keep empty
    }

    /**
     * @brief Transform a DMANNWorkload into a DMANNWorkload descriptor
     *
     * @param workload the DMANNWorkload to transform
     * @return std::vector<T>& a DMANNWorkload descriptor
     */
    const std::vector<T>& transformSingle(const DMADesc& workload) {
        size_t unsused_output_written_offset;
        return generate_descriptor(workload, unsused_output_written_offset);
    };

    /// @brief default virtual destructor, we need this because this class is abstract
    virtual ~IPreprocessingDMA() = default;
};

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
    /** @brief fills in the positions corresponding to an enum.
     * All on zero except the position corresponding to the active enum value is put on 1
     * Enum values must start at zero and be contiguous
     *
     * @param data the enum value
     * @param offset m the index where writing starts
     * @param category_size how many values this enum has
     *
     * @tparam only_simulate, a boolean, if true, the method will do no writing, just computing the size it needs to do
     * the writing
     * @tparam E,  enum type
     *
     * @return the offset of next available position. Equivalent to how many positions are filled in the output vector.
     */
    template <bool only_simulate, class E>
    size_t one_hot(E data, size_t offset, size_t category_size) {
        size_t idx = static_cast<size_t>(data);  // assuming the enums are contiguous and from zero
        if (idx < category_size || category_size == 0) {
            insert<only_simulate>(T(1.0), idx + offset);
        }
        return offset + category_size;  // this enum occupies category_size positions
    }

    template <bool only_simulate = false, class E, size_t SIZE>
    size_t insert(const std::array<E, SIZE>& vec, size_t offset) {
        for (long unsigned int idx = 0; idx < vec.size(); idx++) {
            offset = insert<only_simulate>(vec[idx], offset);
        }
        return offset;
    }

    template <bool only_simulate = false, class E>
    size_t insert(const std::vector<E>& vec, size_t offset) {
        for (auto idx = 0; idx < vec.size(); idx++) {
            offset = insert<only_simulate>(vec[idx], offset);
        }
        return offset;
    }

    ///// @brief inserter for VPUTensor
    ///// required because of array and vector inserter
    // template <bool only_simulate>
    // size_t insert(const VPUTensor& data, size_t offset) {
    //     // calls the specialization from the derived class, known at compile time
    //     return static_cast<D*>(this)->template insert<only_simulate>(data, offset);
    // }

    template <bool only_simulate = false, class E>
    size_t insert(E data, size_t offset) {
        return one_hot<only_simulate>(data, offset, static_cast<int>(E::__size));
    }

    template <bool only_simulate = false>
    size_t insert(unsigned int data, size_t offset) {
        return insert<only_simulate>(T(data), offset);
    }
    template <bool only_simulate = false>
    size_t insert(int data, size_t offset) {
        return insert<only_simulate>(T(data), offset);
    }

    template <bool only_simulate = false>
    size_t insert(bool data, size_t offset) {
        return insert<only_simulate>(T(data), offset);
    }

    template <bool only_simulate = false>
    size_t insert(const T& data, size_t offset) {
        if (!only_simulate) {
            if (offset >= this->processed_output.size()) {
                std::stringstream buffer;
                buffer << "[ERROR] PreprocessingInserter.insert(),"
                       << " out of range write (offset) when building VPUNN descriptor, offset:" << offset
                       << ", this is beyond the allocated descriptor size:  " << this->processed_output.size()
                       << " Cannot continue! "
                       << " File: " << __FILE__ << " Line: " << __LINE__;
                const std::string details = buffer.str();
                throw std::out_of_range(details);
            }
            this->processed_output[offset] = data;
        }
        return offset + 1;
    }

    /**
     * @brief This function computes the size of the DMANNWorkload features to feed to the NN
     * This is the size of a particular descriptor for a [particular] cost NN input
     *
     * @return unsigned int  size of the descriptor for this workload according to NN input expectations
     */
    size_t calculate_size() {
        const DMADesc dummy_wl{};
        size_t size_required = 0;
        // use the derived class to run a mock of transform only for finding how much it fills in
        static_cast<D*>(this)->template transformOnly<true>(dummy_wl, size_required);
        return size_required;
    }

public:
    using IPreprocessingDMA<T, DMADesc>::transformSingle;  ///< exposes the non virtual transform for  workloads vector

    PreprocessingInserterDMA() {};
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
    const std::vector<T>& generate_descriptor(const DMADesc& workload, size_t& debug_offset) override {
        this->reset();  // all on zero
        D* derived{static_cast<D*>(this)};
        this->check_and_throw_size(derived->size_of_descriptor);  // will throw in case not enough space

        return derived->template transformOnly<false>(workload, debug_offset);
    };
};

/// @brief enum for NN descriptor versions (input versions)
enum class NNVersionsDMA : int {
    VERSION_00_LATEST_NONE = 0,  ///< no version OR last version
    VERSION_01_27 = 1,           ///< initial version, first one for 2.7
    VERSION_02_40 = 2,           ///< 6D dedicated to 4.0+
    VERSION_03_RESERVED_v1 = 3,        ///< 6D dedicated to RESERVED iteration 1 . Probably supports max 2D
};

}  // namespace VPUNN
#endif
