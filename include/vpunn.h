// Copyright © 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_H
#define VPUNN_H

#include <string>
#include "core/profiling.h"
#include "inference/model.h"

/// @brief top namespace for VPUNN cost model library
namespace VPUNN {

/**
 * @brief VPUNN runtime model
 *
 */
class Runtime {
private:
    InferenceModel model;  ///< the NN loaded from a file/buffer (flatbuffer)
    const bool profile;
    ModelVersion model_version;  ///< holds the version info about the loaded model

public:
    /**
     * @brief Construct a new Runtime object
     *
     * @param filename .vpunn model
     * @param batch model batch size
     * @param profile enable/disable profiling
     */
    explicit Runtime(const std::string& filename, const unsigned int batch = 1, bool profile = false)
            : model(filename.c_str()), profile(profile), model_version() {
        if (initialized()) {
            model.allocate_tensors(batch);  // might throw
            model_version.parse_name(model.network_name());
        }
    }

    /**  @brief provides version info for loaded model
     * The provided reference is always up to date with the loaded model
     * The info are conditioned(otherwise default) by the successful loading of the model.
     *
     * @return a long lived reference to the version information
     */
    const ModelVersion& model_version_info() {
        return model_version;
    }

    /**
     * @brief Construct a new Runtime object
     *
     * @param model_data .vpunn model buffer
     * @param model_data_length buffer size
     * @param copy_model_data enable/disable memcopy of the module buffer
     * @param batch model batch size
     * @param profile enable/disable profiling
     */
    explicit Runtime(const char* model_data, size_t model_data_length, bool copy_model_data,
                     const unsigned int batch = 1, bool profile = false)
            : model(model_data, model_data_length, copy_model_data), profile(profile), model_version() {
        if (initialized()) {
            model.allocate_tensors(batch);  // might throw
            model_version.parse_name(model.network_name());
        }
    }

    /**
     * @brief Check if the NN model is initialized
     *
     * @return true if the NN model is initialized
     * @return false if the NN model is not initialized
     */
    bool initialized() const {
        return model.is_initialized();
    }

    /**
     * @brief Get the model input tensors
     *
     * @return std::vector<Tensor<float>*>
     */
    std::vector<Tensor<float>*> input_tensors() {
        return model.input_tensors();
    }

    /**
     * @brief Get the model output tensors
     *
     * @return std::vector<Tensor<float>*>
     */
    std::vector<Tensor<float>*> output_tensors() {
        return model.output_tensors();
    }

    /**
     * @brief Get the model input tensors shapes
     *
     * @return std::vector<std::vector<unsigned int>>
     */
    std::vector<std::vector<unsigned int>> input_shapes() {
        std::vector<std::vector<unsigned int>> shapes;
        for (auto tensor : input_tensors()) {
            shapes.push_back(tensor->shape());
        }
        return shapes;
    }

    /**
     * @brief Get the model output tensors shapes
     *
     * @return std::vector<std::vector<unsigned int>>
     */
    std::vector<std::vector<unsigned int>> output_shapes() {
        std::vector<std::vector<unsigned int>> shapes;
        for (auto tensor : output_tensors()) {
            shapes.push_back(tensor->shape());
        }
        return shapes;
    }

    /**
     * @brief Run the inference
     *
     * @tparam T input input_array datatype
     * @param input_array input data
     * @param input_size input data size
     * @return pointer to one or more values containing the inference result
     */
    template <class T>
    const T* predict(const T* input_array, const unsigned int input_size) {
        model.set_inputs(input_array, input_size);  // might throw if input mismatch
        auto t1 = tick();
        model.predict();
        if (profile) {
            auto delta = tock(t1);
            Logger::info() << "Execution time: " << delta << " ms";
        }
        return model.get_outputs<T>();
    }

    /**
     * @brief Run the inference
     *
     * @tparam T input input_array datatype
     * @param input_tensor input data
     * @return one or more values containing the inference result
     */
    template <class T>
    const std::vector<T> predict(const std::vector<T>& input_tensor) {
        model.set_inputs(input_tensor.data(),
                         static_cast<unsigned int>(input_tensor.size()));  // might throw if input mismatch
        auto t1 = tick();
        model.predict();
        if (profile) {
            auto delta = tock(t1);
            Logger::info() << "Execution time: " << delta << " ms";
        }
        return model.get_outputs_vector<T>();
    }
};

}  // namespace VPUNN

#endif  // VPUNN_H
