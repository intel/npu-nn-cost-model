// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#include "inference/inference_execution_data.h"

#include "vpunn_generated.h"  //for flabuffer model

// #include "inference/model.h"

#include "kernels/bias.h"
// #include "kernels/fully_connected.h"
// #include "kernels/kNN.h"
// #include "kernels/l2_normalization.h"
// #include "kernels/sigmoid.h"

namespace VPUNN {

void InferenceExecutionData::allocate_tensorsMapAndBias(const unsigned int batch, const VPUNN_SCHEMA::Model* theModel) {
    const auto tensors = theModel->tensors();
    const auto buffers = theModel->buffers();

    for (auto flatbuffer_tensor = tensors->cbegin(); flatbuffer_tensor != tensors->cend(); ++flatbuffer_tensor) {
        const uint32_t buffer_ID = flatbuffer_tensor->buffer();
        constexpr uint32_t NOT_EXISTING{0};
        const bool buffer_is_present{buffer_ID != NOT_EXISTING};
        // Batch only activations
        // we use the batch only for buffers that are not stored in model (dynamic tensors)
        const uint32_t forced_batch{(buffer_is_present) ? 1 : batch};

        const auto tensor_shape{parse_vector(flatbuffer_tensor->shape(), forced_batch)};

        {                                        // Create/Fill the new tensor structure
            Tensor<float> tensor{tensor_shape};  // allocates on heap!
            if (buffer_is_present) {  // in this case, copy the existing data(the buffer) into tensor's memory
                const auto array = buffers->Get(buffer_ID)->data();
                tensor.assign((const float*)(array->data()),
                              array->size());  // will throw if buffer mismatch with tensor shape
            } else {
                tensor.fill(0);  // Fill with zeros the tensor's memory
            }
            tensor_map.emplace_back(std::move(tensor));  // store the created memory!
        }
    }

    bias.reserve_bias_space(this->max_batch_in_tensors(tensor_map));
}

}  // namespace VPUNN
