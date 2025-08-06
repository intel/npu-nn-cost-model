// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_INFERENCE_MODEL_H
#define VPUNN_INFERENCE_MODEL_H

#include <stdio.h>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>

#include "core/logger.h"
#include "core/tensors.h"
#include "core/vpunn_api.h"

// #include "kernels/bias.h"

#include "inference/inference_execution_data.h"
#include "vpunn_generated.h"  //for flabuffer model

namespace VPUNN {

/**
 * @brief VPUNN inference model
 * After creation can be used only if the model was initialized, otherwise will crash
 *
 */
class VPUNN_API InferenceModel {
private:
    std::vector<char> buffer_for_model;         // holds the flabuffer content if stored in this class
    const VPUNN_SCHEMA::Model* model{nullptr};  ///< the flatbuffer model. It is just a view/interpretation of the
                                                ///< buffer_for_model or of another external buffer

    bool initialized;

    // Run an individual layer, memory passed from outside
    void run_layer(const VPUNN_SCHEMA::Layer* layer, const std::vector<const Tensor<float>*>& inputs,
                   const std::vector<Tensor<float>*>& outputs, BiasOpBuffer& biasBuf) const;

public:
    const VPUNN_SCHEMA::Model* get_model() const {
        return is_initialized() ? model : nullptr;
    }
    /**
     * @brief Name of the network stored in the model
     *
     * @return unaltered name of the model as it is stored in the source/filename of the loaded model
     */
    std::string network_name() const {
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
     * @brief Run the inference
     *
     */
    void predict(InferenceExecutionData& execution_memory) const;
};

}  // namespace VPUNN
#endif
