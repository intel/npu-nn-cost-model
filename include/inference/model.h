// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef SCHEMA_PARSER_H
#define SCHEMA_PARSER_H

#include <stdio.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>

#include "core/logger.h"
#include "core/tensors.h"
#include "core/vpunn_api.h"
#include "kernels/bias.h"
#include "kernels/fully_connected.h"
#include "kernels/kNN.h"
#include "kernels/l2_normalization.h"
#include "kernels/sigmoid.h"

#include "vpunn_generated.h"

namespace VPUNN {
/**
 * @brief VPUNN inference model
 * After creation can be used only if the model was initialized, otherwise will crash
 *
 */
class VPUNN_API(InferenceModel) {
private:
    std::vector<char> buf;
    const VPUNN_SCHEMA::Model* model{nullptr};  //< the flatbuffer model. todo: make it reference?
    std::vector<Tensor<float>> tensor_map;
    bool initialized;
    BiasOp bias;  ///< the bias operation support. Contains the bias buffer.

    /// @brief Parse a flatbuffer vector into a std vectors with tensor dimensions
    /// First position is special and treated as batch. may be changed
    /// @param vector to be parsed
    /// @param forced_first_dim , a new batch (first dimension )to be set . Must be greater than 1 to be considered
    /// @return the computed shape
    template <typename T>
    std::vector<T> parse_vector(const flatbuffers::Vector<T>* vector, const unsigned int forced_first_dim) {
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

    // Get the tensors from tensor_map from the schema
    template <typename T>
    std::vector<Tensor<float>*> get_tensors_from_index(const flatbuffers::Vector<T>* tensors) {
        std::vector<Tensor<float>*> result;
        for (auto it = tensors->begin(); it != tensors->end(); ++it) {
            if (*it >= 0 && *it < (int)tensor_map.size())
                result.push_back(&(tensor_map[*it]));
        }
        return result;
    }

    // Run an individual layer
    void run_layer(const VPUNN_SCHEMA::Layer* layer);

    /// @ brief returns the maximum value for dimension zero (the batch size) among all tensors
    int max_batch_in_tensors() const {
        int max_dim_zero = 0;
        for (auto it = tensor_map.begin(); it != tensor_map.end(); ++it) {
            const int dim_zero = (*it).shape()[0];
            max_dim_zero = std::max(max_dim_zero, dim_zero);
        }
        return max_dim_zero;
    }

public:
    /**
     * @brief Name of the network stored in the model
     *
     * @return unaltered name of the model as it is stored in the source/filename of the loaded model
     */
    std::string network_name() {
        return (model != nullptr) ? model->name()->c_str() : "";
    }

    /**
     * @brief Construct a new Inference Model object
     *
     * @param filename .vpunn file
     */
    explicit InferenceModel(const char* filename);
    /**
     * @brief Construct a new Inference Model object
     *
     * @param data a pointer to a const char buffer containing the .vpunn model
     * @param length the data buffer length
     * @param with_copy enable/disable memcopy of the original data buffer
     */
    InferenceModel(const char* data, size_t length, bool with_copy);

    /**
     * @brief Check if the NN model is initialized
     *
     * @return true if the NN model is initialized
     * @return false if the NN model is not initialized
     */
    bool is_initialized() const {
        return initialized;
    }

    /**
     * @brief Creates/Allocates the activation and weights buffer in memory
     *
     * @param batch the VPUNN inference batch size
     * @throws runtime_error in case tensors and buffers have a mismatch problem
     */
    void allocate_tensors(const unsigned int batch);

    /**
     * @brief Get the VPUNN input tensors
     *
     * @return std::vector<Tensor<float>*> the NN input tensors
     */
    std::vector<Tensor<float>*> input_tensors() {
        return get_tensors_from_index(model->inputs());
    }

    /**
     * @brief  Get the VPUNN output tensors
     *
     * @return std::vector<Tensor<float>*> the NN output tensors
     */
    std::vector<Tensor<float>*> output_tensors() {
        return get_tensors_from_index(model->outputs());
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
        if (model->inputs()->size() > 1) {
            Logger::error() << "Only single input model is valid";
            throw std::runtime_error("This model has more inputs.Only single input model is valid.");
        }

        tensor_map[model->inputs()->Get(0)].assign(inputs, sizeof(T) * size);
    }

    // get network outputs
    /**
     * @brief Get the outputs tensor values
     *
     * @tparam T the output tensor datatype
     * @return T* a pointer to the output buffer
     */
    template <typename T>
    const T* get_outputs() {
        if (model->outputs()->size() > 1) {
            Logger::error() << "Only single output model is valid";
            return nullptr;
        }
        return tensor_map[model->outputs()->Get(0)].data();
    }

    /**
     * @brief Get a copy of the outputs tensor as a std::vector
     *
     * @tparam T the output tensor datatype
     * @return std::vector<T> the output buffer
     */
    template <typename T>
    const std::vector<T> get_outputs_vector() {
        if (model->outputs()->size() > 1) {
            Logger::error() << "Only single output model is valid";
            return {};
        }
        return tensor_map[model->outputs()->Get(0)].data_vector();
    }

    /**
     * @brief Run the inference
     *
     */
    void predict();
};

}  // namespace VPUNN
#endif
