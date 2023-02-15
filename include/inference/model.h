// Copyright © 2022 Intel Corporation
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
    std::vector<std::shared_ptr<Tensor<float>>> tensor_map;
    bool initialized;

    // Parse a flatbuffer vector into a std vectors
    template <typename T>
    std::vector<T> parse_vector(const flatbuffers::Vector<T>* vector, const unsigned int batch) {
        std::vector<T> vv;
        for (auto it = vector->begin(); it != vector->end(); ++it) {
            // Change the batch dimension only for activations and if old_batch == 1 and new_batch > 1
            const auto dim = (*it == 1 && it == vector->begin() && batch > 1) ? batch : *it;
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
                result.push_back(tensor_map[*it].get());
        }
        return result;
    }

    // Run an individual layer
    void run_layer(const VPUNN_SCHEMA::Layer* layer);

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
    bool is_initialized() {
        return initialized;
    }

    /**
     * @brief Create the activation and weights buffer in memory
     *
     * @param batch the VPUNN inference batch size
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
    void set_inputs(T* inputs, unsigned int size) {
        if (model->inputs()->Length() > 1) {
            Logger::error() << "Only single input model is valid";
            return;
        }

        tensor_map[model->inputs()->Get(0)]->assign(inputs, sizeof(T) * size);
    }

    // get network outputs
    /**
     * @brief Get the outputs tensor values
     *
     * @tparam T the output tensor datatype
     * @return T* a pointer to the output buffer
     */
    template <typename T>
    T* get_outputs() {
        if (model->outputs()->Length() > 1) {
            Logger::error() << "Only single output model is valid";
            return nullptr;
        }
        return tensor_map[model->outputs()->Get(0)]->data();
    }

    /**
     * @brief Run the inference
     *
     */
    void predict();
};

/// Extracts and keeps  the version out of a NN raw name
class ModelVersion {
public:
    /// @brief value of input interface version, the one for the input descriptor
    int get_input_interface_version() const {
        return input_version;
    };
    /// @brief value of output interface version, the one for the NN provided value(s)
    int get_output_interface_version() const {
        return output_version;
    };

    /// @brief name of the NN, without version info
    std::string get_NN_name() const {
        return name;
    };

    /// @brief initial, raw name of the VPUNN model. contains version info
    std::string get_raw_name() const {
        return full_raw_name;
    };

    /// @brief parsed the name and extracts the information
    /// get the name based on separator "-", template: NNNNNNN-VI-VO
    /// VI and VO must be integers, and only the integers part will be considered when converting
    /// NNNNNN cannot be empty, it will be replaced with "none" if empty
    /// Only first three parts of the name are considered, rest are ignored.
    /// Missing pars will be considered default: none-1-1
    ///
    /// @throws invalid_argument, out_of_range
    void parse_name(const std::string& raw_NN_name) {
        // get the name based on separator "-", template NNNNNNN-VI-VO
        // VI and VO must be integers, and only the integers part will be considered when converting
        reset();  // all are on default
        full_raw_name = raw_NN_name;
        std::istringstream input;
        input.str(full_raw_name);
        std::vector<std::string> parts;
        for (std::string part{""}; std::getline(input, part, version_separator);) {
            parts.push_back(part);
        }

        // fist position is the full name, mandatory
        if (parts.size() >= 1) {
            if (parts[0].length() > 0) {
                name = parts[0];
            }
        } else {  // empty name
            // use latest
            input_version = def_latest_special_version;
        }

        // in version
        if (parts.size() >= 2) {
            size_t c;
            int v{std::stoi(parts[1], &c)};  // might throw invalid_argument, out_of_range
            input_version = v;
        }

        // out version
        if (parts.size() >= 3) {
            size_t c;
            int v{std::stoi(parts[2], &c)};  // might throw invalid_argument, out_of_range
            output_version = v;
        }
    }

    ModelVersion() {
        reset();
    }

private:
    std::string full_raw_name;
    std::string name;
    int input_version;
    int output_version;

    static constexpr char version_separator = '-';
    static constexpr int def_input_version = 1;
    static constexpr int def_output_version = 1;
    static constexpr int def_latest_special_version = 0;

    void reset() {
        full_raw_name = "none";
        name = "none";
        input_version = 1;
        output_version = 1;
    }
};

}  // namespace VPUNN
#endif
