// Copyright © 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef PREPROCESSING_H
#define PREPROCESSING_H

#include <math.h>
#include <vpu/types.h>
#include <sstream>  // for error formating
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "vpu/validation/dpu_operations_validator.h"

namespace VPUNN {

/**
 * @brief Class to transform DPUWorkload objects into NN descriptors that can be used by the NN
 *
 * @tparam T datatype of the descriptor
 */
template <class T>
class Preprocessing {
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
     * @brief Transform a DPUWorkload into a DPUWorkload descriptor
     *
     * @param workload a DPUWorkload to be transformed
     * @param debug_offset [out] will store how many elements were actually written
     * @return std::vector<T>& a DPUWorkload descriptor
     */
    virtual const std::vector<T>& generate_descriptor(const DPUWorkload& workload, size_t& debug_offset) = 0;

public:
    /// @brief provides the interface number this instance implements
    /// @returns the interface version
    virtual int interface_version() const = 0;

    /**
     * @brief Construct a new Preprocessing object
     *
     */
    Preprocessing(){};

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
     * @brief Set the most probable value for batch. WIll ensure cache memory is present according to this
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
     * @brief Transform a DPUWorkload into a DPUWorkload descriptor
     *
     * @param workload the DPUWorkloadto transform
     * @return std::vector<T>& a DPUWorkload descriptor
     */
    const std::vector<T>& transform(const DPUWorkload& workload) {
        size_t unsused_output_written_offset;
        return generate_descriptor(workload, unsused_output_written_offset);
    };

    /** @brief default virtual destructor, we need this because this class is abstract
     */
    virtual ~Preprocessing() = default;

    /**
     * @brief Transform DPUWorkloads into a DPUWorkload descriptors
     *
     * @param workloads a vector of DPUWorkloads
     * @param pad the amount of padding to add (default 1), the batch size
     * @return reference to DPUWorkload descriptors
     */
    const std::vector<T>& transform(const std::vector<DPUWorkload>& workloads, unsigned int pad = 1) {
        //<all workloads, normalized
        const auto total_workloads{round_up(static_cast<unsigned int>(workloads.size()), pad)};

        // ensure output is big enough, this will not decrease capacity, but make it empty
        set_probable_batch(total_workloads);

        for (long unsigned int idx = 0; idx < total_workloads; ++idx) {
            const auto& one_descriptor{(idx < workloads.size()) ? transform(workloads[idx]) : zero_vector};
            assert(one_descriptor.size() == output_size());
            // copy this result into the big batch descriptor
            std::copy(one_descriptor.begin(), one_descriptor.end(), std::back_inserter(batch_processed_output));
        }
        return batch_processed_output;
    }
};

/**
 * @brief Intermediate base class for a real Preprocessing
 * Using  curiously recurring template pattern (CRTP) provides static polymorphism
 *
 *
 * @tparam T datatype of the descriptor
 * @tparam D the derived class. Real/Final processors are derived from this
 */
template <class T, class D>
class PreprocessingInserter : public Preprocessing<T> {
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

    /// @brief inserter for VPUTensor
    /// required because of array and vector inserter
    template <bool only_simulate>
    size_t insert(const VPUTensor& data, size_t offset) {
        // calls the specialization from the derived class, known at compile time
        return static_cast<D*>(this)->template insert<only_simulate>(data, offset);
    }

    template <bool only_simulate = false, class E>
    size_t insert(E data, size_t offset) {
        return one_hot<only_simulate>(data, offset, static_cast<int>(E::__size));
    }

    template <bool only_simulate = false>
    size_t insert(unsigned int data, size_t offset) {
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
     * @brief This function computes the size of the DPUWorklads features to feed to the NN
     * This is the size of a particular descriptor for a [particular] cost NN input
     *
     * @return unsigned int  size of the descriptor for this workload according to NN input expectations
     */
    size_t calculate_size() {
        const DPUWorkload dummy_wl{
                VPUNN::VPUDevice::VPU_2_7,
                VPUNN::Operation::CONVOLUTION,
                {VPUNN::VPUTensor(1, 1, 1, 1, VPUNN::DataType::UINT8)},  // input dimensions
                {VPUNN::VPUTensor(1, 1, 1, 1, VPUNN::DataType::UINT8)},  // output dimensions
                {1, 1},                                                  // kernels
                {1, 1},                                                  // strides
                {0, 0, 0, 0},                                            // padding
                VPUNN::ExecutionMode::CUBOID_16x16                       // execution mode
        };
        size_t size_required = 0;
        // use the derived class to run a mock of transform only for finding how much it fills in
        static_cast<D*>(this)->template transformOnly<true>(dummy_wl, size_required);
        return size_required;
    }

public:
    using Preprocessing<T>::transform;  ///< exposes the non virtual transform for  workloads vector

    PreprocessingInserter(){};
    virtual ~PreprocessingInserter() = default;

    int interface_version() const override {
        return D::getInterfaceVersion();  // expected static method
    }

    /**
     * @brief Transform a DPUWorkload into a DPUWorkload descriptor
     *
     * @param workload a DPUWorkload
     * @param debug_offset [out] is the offset where a new value can be written. interpreted as how many positions were
     * written
     * @return std::vector<T>& a DPUWorkload descriptor
     */
    const std::vector<T>& generate_descriptor(const DPUWorkload& workload, size_t& debug_offset) override {
        this->reset();                                                          // all on zero
        this->check_and_throw_size(static_cast<D*>(this)->size_of_descriptor);  // will throw in case not enough space

        return static_cast<D*>(this)->template transformOnly<false>(workload, debug_offset);
    };
};
/// @brief enum for NN descriptor versions (input versions)
enum class NNVersions : int {
    VERSION_00_LATEST_NONE = 0,  ///< no version OR last version
    VERSION_01_BASE = 1,         ///< base version, the unnamed one
    VERSION_10_ENUMS_SAME = 10,  ///< evo of v01, with correct size. est November 2022 VPU2.7 alpha release
    VERSION_11_VPU27_BETA = 11,  ///< input 1 generated, isi strategy, layouts. est Jan 2023 VPU2.7 beta release
};

/**
 * @brief Latest evolution of the Preprocessing for latest interfaces
 */
template <class T>
class PreprocessingLatest : public PreprocessingInserter<T, PreprocessingLatest<T>> {
private:
    const DPU_OperationValidator workload_validator{};  ///< sanitizer mechanisms
protected:
    using PreprocessingInserter<T, PreprocessingLatest<T>>::insert;  ///< exposes the non virtual insert methods
    friend class PreprocessingInserter<T, PreprocessingLatest<T>>;

    /// @brief insert specialization for VPUTensor
    template <bool only_simulate>
    size_t insert(const VPUTensor& data, size_t offset) {
        offset = this->insert<only_simulate>(data.get_shape(), offset);
        offset = this->insert<only_simulate>(data.get_dtype(), offset);
        offset = this->insert<only_simulate>(data.get_layout(), offset);
        offset = this->insert<only_simulate>(data.get_sparsity(), offset);
        return offset;
    }

    /**
     * @brief Transform a DPUWorkload into a DPUWorkload descriptor
     * Here the concrete descriptor is created/populated according to established convention/interface
     *
     * @param workload a DPUWorkload
     * @param debug_offset [out] is the offset where a new value can be written. interpreted as how many positions were
     * written
     * @tparam only_simulate, if true then no data is actually written, only the offset is computed
     * @return std::vector<T>& a DPUWorkload descriptor
     */
    template <bool only_simulate>
    const std::vector<T>& transformOnly(const DPUWorkload& workload, size_t& debug_offset) {
        // Build the vector from the inputs
        size_t offset = 0;
        offset = this->insert<only_simulate>(workload.device, offset);
        offset = this->insert<only_simulate>(workload.op, offset);

        offset = this->insert<only_simulate>(workload.inputs, offset);

        // input 1 tensor to be generated in place here!
        {
            auto input_1 = workload_validator.construct_input_1(workload);
            offset = this->insert<only_simulate>(input_1, offset);
        }
        offset = this->insert<only_simulate>(workload.outputs, offset);

        offset = this->insert<only_simulate>(workload.kernels, offset);
        offset = this->insert<only_simulate>(workload.strides, offset);
        offset = this->insert<only_simulate>(workload.padding, offset);

        offset = this->insert<only_simulate>(workload.execution_order, offset);
        // offset = this->insert<only_simulate>(workload.activation_function, offset);

        offset = this->insert<only_simulate>(workload.act_sparsity, offset);
        offset = this->insert<only_simulate>(workload.weight_sparsity, offset);

        offset = this->insert<only_simulate>(workload.input_swizzling, offset);   // 2 elements
        offset = this->insert<only_simulate>(workload.output_swizzling, offset);  // 1 element

        offset = this->insert<only_simulate>(workload.output_write_tiles, offset);

        offset = this->insert<only_simulate>(workload.isi_strategy, offset);

        // weight_sparsity_enabled  contributes to the input_1 deduced tensor

        debug_offset = offset;

        // Return the output as a pointer to the data
        return this->processed_output;
    }

    const size_t size_of_descriptor;  ///< how big the descriptor is, fixed at constructor

public:
    /// @brief the descriptor interface that this type was designed to fill/comply with
    static int getInterfaceVersion() {
        return static_cast<std::underlying_type_t<NNVersions>>(
                NNVersions::VERSION_00_LATEST_NONE);  // 0 is a reserved interface type for latest evolution, draft
                                                      // changes
    }

    /**
     * @brief Tells what other interface is equal to this latest version, use with care
     * A latest interface might have already a defined interface version (a specialization with version), or a new
     * version is planned and this latest will become that version.
     * If this latest is equal with a particular version it is worth to use the latests since it is faster at execution.
     *
     * @return the version of compatible/equal interface version, 0 otherwise
     */
    static int implements_also_interface() {
        return static_cast<std::underlying_type_t<NNVersions>>(NNVersions::VERSION_00_LATEST_NONE);
    }

    /**
     * @brief Ctor , inits the content with expected size
     */
    PreprocessingLatest(): size_of_descriptor(this->calculate_size()) {
        this->set_size(size_of_descriptor);
    };

    /**
     * @brief default virtual destructor
     */
    virtual ~PreprocessingLatest() = default;
};

}  // namespace VPUNN
#endif
