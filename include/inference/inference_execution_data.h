// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_INFERENCE_EXECUTION_DATA_H
#define VPUNN_INFERENCE_EXECUTION_DATA_H

#include <stdio.h>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>

#include "core/logger.h"
#include "core/tensors.h"
// #include "core/vpunn_api.h"

#include "kernels/bias.h"

#include "vpunn_generated.h"  //for flabuffer model

namespace VPUNN {
/// holds the RW memory that reflects INference model and is used to execute the Model on it.
/* coverity[rule_of_five_violation:FALSE] */
class InferenceExecutionData {
public:
    InferenceExecutionData(const unsigned int batch, const VPUNN_SCHEMA::Model* theModel)
            : input_buffer_cached_IDX{theModel ? theModel->inputs()->Get(0) : -1},
              output_buffer_cached_IDX{theModel ? theModel->outputs()->Get(0) : -1} {
        if (theModel) {
            allocate_tensorsMapAndBias(batch, theModel);  // default batch size is 1
            check_in_out_cardinality(theModel);           // check if the model has one input and one output//throws
        }
    }
    InferenceExecutionData(const InferenceExecutionData&) = delete;             ///< no copy allowed
    InferenceExecutionData(InferenceExecutionData&&) = default;                 ///< move is allowed
    InferenceExecutionData& operator=(const InferenceExecutionData&) = delete;  ///< no copy allowed
    InferenceExecutionData& operator=(InferenceExecutionData&&) = delete;       ///< move is implicitly deleted
    virtual ~InferenceExecutionData() = default;                                ///< destructor

protected:
    std::vector<Tensor<float>>
            tensor_map;  ///< this is the memory where actual inputs and inter layer data is kept.
                         ///< You need one instance
                         ///< of this to execute an inference. If multiple inferences have to be run in parallel in the
                         ///< name model, this data placeholder has to be unique per execution thread.

    BiasOpBuffer bias;  ///< the bias operation support. Contains the bias buffer.

    // maybe we can store also shortcuts to overal input and oputput ?
    const int32_t input_buffer_cached_IDX;
    const int32_t output_buffer_cached_IDX;

    /// @ brief returns the maximum value for dimension zero (the batch size) among all tensors
    int max_batch_in_tensors(std::vector<Tensor<float>>& tensor_map_) const {
        int max_dim_zero = 0;
        for (auto it = tensor_map_.cbegin(); it != tensor_map_.cend(); ++it) {
            const int dim_zero = (*it).shape()[0];
            max_dim_zero = std::max(max_dim_zero, dim_zero);
        }
        return max_dim_zero;
    }

    template <typename T>
    std::vector<T> parse_vector(const flatbuffers::Vector<T>* vector, const unsigned int forced_first_dim) const {
        std::vector<T> vv;
        const bool change_allowed{forced_first_dim > 1};  //< mechanism is ON only for some values
        for (auto it = vector->begin(); it != vector->end(); ++it) {
            const auto parsed_value{*it};
            const bool is_first_dim{it == vector->begin()};  //< are we at first component
            // Change the batch dimension only for activations and if old_batch == 1 and new_batch > 1
            const auto dim{((parsed_value == 1) && is_first_dim && change_allowed) ? forced_first_dim : parsed_value};
            vv.push_back(dim);
        }
        return vv;
    }

    void allocate_tensorsMapAndBias(const unsigned int batch, const VPUNN_SCHEMA::Model* theModel);

    friend class InferenceModel;

public:
    // Get the tensors from tensor_map from the schema
    template <typename T>
    std::vector<const Tensor<float>*> get_ro_tensors_from_index(const flatbuffers::Vector<T>* tensors) const {
        std::vector<const Tensor<float>*> result;
        for (auto it = tensors->cbegin(); it != tensors->cend(); ++it) {
            if (*it >= 0 && *it < (int)tensor_map.size())
                result.push_back(&(tensor_map[*it]));
        }
        return result;
    }
    // Get the tensors from tensor_map from the schema
    template <typename T>
    std::vector<Tensor<float>*> get_rw_tensors_from_index(const flatbuffers::Vector<T>* tensors) {
        std::vector<Tensor<float>*> result;
        for (auto it = tensors->cbegin(); it != tensors->cend(); ++it) {
            if (*it >= 0 && *it < (int)tensor_map.size())
                result.push_back(&(tensor_map[*it]));
        }
        return result;
    }

protected:
    /**
     * @brief Get the VPUNN input tensors
     *
     * @return std::vector<Tensor<float>*> the NN input tensors
     */
    std::vector<const Tensor<float>*> input_tensors() const {
        std::vector<const Tensor<float>*> ret{&(tensor_map[input_buffer_cached_IDX])};
        return ret;
    }

    /**
     * @brief  Get the VPUNN output tensors
     *
     * @return std::vector<Tensor<float>*> the NN output tensors
     */
    std::vector<const Tensor<float>*> output_tensors() const {
        std::vector<const Tensor<float>*> ret{&(tensor_map[output_buffer_cached_IDX])};
        return ret;
    }

public:
    /**
     * @brief Get the model input tensors shapes
     *
     * @return std::vector<std::vector<unsigned int>>
     */
    std::vector<std::vector<unsigned int>> input_shapes() const {
        std::vector<std::vector<unsigned int>> shapes;
        for (const auto& tensor : input_tensors()) {
            shapes.push_back(tensor->shape());
        }
        return shapes;
    }

    /**
     * @brief Get the model output tensors shapes
     *
     * @return std::vector<std::vector<unsigned int>>
     */
    std::vector<std::vector<unsigned int>> output_shapes() const {
        std::vector<std::vector<unsigned int>> shapes;
        for (const auto& tensor : output_tensors()) {
            shapes.push_back(tensor->shape());
        }
        return shapes;
    }

    /**
     * @brief Set the network inputs tensors values
     *
     * @tparam T data buffer datatype
     * @param inputs a data buffer
     * @param size the data buffer size
     */
    template <typename T>
    void set_inputs(const T* inputs, const unsigned int size) {
        tensor_map[input_buffer_cached_IDX].assign(inputs, sizeof(T) * size);
    }

    // get network outputs
    /**
     * @brief Get the outputs tensor values
     *
     * @tparam T the output tensor datatype
     * @return T* a pointer to the output buffer
     */
    template <typename T>
    const T* get_outputs() const {
        return tensor_map[output_buffer_cached_IDX].c_ptr();
    }

    /**
     * @brief Get a copy of the outputs tensor as a std::vector
     *
     * @tparam T the output tensor datatype
     * @return std::vector<T> the output buffer
     */
    template <typename T>
    const std::vector<T> get_outputs_copy_as_vector() const {
        return tensor_map[output_buffer_cached_IDX].data_vector();
    }

    /// now we must have one input and one output
    void check_in_out_cardinality(const VPUNN_SCHEMA::Model* theModel) const {
        const auto in_cnt{theModel->inputs()->size()};
        const auto out_cnt{theModel->outputs()->size()};
        if ((in_cnt != 1) || (out_cnt != 1)) {
            Logger::error() << "Only single input and output model is valid! Inputs count: " << in_cnt
                            << ", Outputs count: " << out_cnt << " !";
            throw std::runtime_error(
                    "Invalid model: Only single input and output are supported. This one fails, see log!");
        }
    }
};
}  // namespace VPUNN
#endif
