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

    /// @brief insert specialization for HaloInfoHW
    template <bool only_simulate>
    size_t insert(const HaloWorkload::HaloInfoHW& data, size_t offset) {
        offset = this->insert<only_simulate>(data.top, offset);
        offset = this->insert<only_simulate>(data.bottom, offset);
        offset = this->insert<only_simulate>(data.left, offset);
        offset = this->insert<only_simulate>(data.right, offset);
        return offset;
    }

    /// @brief insert specialization for HaloInfoHWC
    template <bool only_simulate>
    size_t insert(const HaloWorkload::HaloInfoHWC& data, size_t offset) {
        offset = this->insert<only_simulate>(data.top, offset);
        offset = this->insert<only_simulate>(data.bottom, offset);
        offset = this->insert<only_simulate>(data.left, offset);
        offset = this->insert<only_simulate>(data.right, offset);
        offset = this->insert<only_simulate>(data.front, offset);
        offset = this->insert<only_simulate>(data.back, offset);
        return offset;
    }

    /// @brief insert specialization for HaloWorkload
    template <bool only_simulate>
    size_t insert(const HaloWorkload& data, size_t offset) {
        offset = this->insert<only_simulate>(data.input_0_halo, offset);
        offset = this->insert<only_simulate>(data.output_0_halo, offset);
        offset = this->insert<only_simulate>(data.output_0_halo_broadcast_cnt, offset);
        return offset;
    }

    // specialization for latest swizzling. Each version type should have its own if wants special treatment
    // instead of enum one hot style, use a boolean enabled/disabled
    template <bool only_simulate>
    size_t insert(Swizzling data, size_t offset) {
        // 1 for enabled :Key_n, zero for disabled: KEY_0
        const bool is_swizzling_enabled{(data != Swizzling::KEY_0)};
        return this->insert<only_simulate>(is_swizzling_enabled, offset);
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

    ///// COmputes the memory  input tensor , regarding  the spatial W and H dimensions. It uses only real HALO specific
    ///// to VPU2.7, (positive HALO).
    ///// Scope is to generate the input tensor that was used when training the NPU2.7 NN.
    /////
    ///// @returns the new input tensor, only H and W are changed, and only by becoming smaller.
    // VPUTensor computeActualSpatialMemoryNoHaloTensor(const VPUTensor& origT, const HaloWorkload& halo) const {
    //     const auto& in_halo{halo.input_0_halo};
    //     //  extension will be negative(memory reduction) if halo(positive halo),
    //     // or positive (memory increase) ,memory is larger, if negative halo, but we consume less (prev layer wrote
    //     //  more): DO NOT CARE

    //    auto newDimension = [](const long long crtDimension, const int oneEndHalo, const int otherEndHalo) {
    //        const int oneExt{oneEndHalo > 0 ? -oneEndHalo : 0};  // only if halo memory
    //        const int twoExt{otherEndHalo > 0 ? -otherEndHalo : 0};

    //        const long long newDim = crtDimension + (oneExt + twoExt);
    //        return (newDim > 0 ? newDim : 0);  // limit to zero
    //    };
    //    const auto h{newDimension(origT.height(), in_halo.top, in_halo.bottom)};
    //    const auto w{newDimension(origT.width(), in_halo.left, in_halo.right)};

    //    const std::array<unsigned int, 4> newshape{static_cast<unsigned int>(w), static_cast<unsigned int>(h),  //
    //                                               origT.channels(), origT.batches()};                          //
    //                                               whcb
    //    const VPUTensor ret(newshape, origT);
    //    return ret;
    //}

protected:
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
     * @param debug_offset [out] is the offset where a new value can be written. interpreted as how many positions
     * were written
     * @return std::vector<T>& a DPUWorkload descriptor
     */
    const std::vector<T>& generate_descriptor(const DPUWorkload& workload, size_t& debug_offset) override {
        this->reset();                                                          // all on zero
        this->check_and_throw_size(static_cast<D*>(this)->size_of_descriptor);  // will throw in case not enough space

        return static_cast<D*>(this)->template transformOnly<false>(workload, debug_offset);
    };
};

}  // namespace VPUNN
#endif
