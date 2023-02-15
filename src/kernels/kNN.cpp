// Copyright © 2022 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#include "kernels/kNN.h"
#include <queue>
#include <vector>
#include "kernels/vpunn_blas.h"

// Find the smallest N items in an array
void n_index(int n_index, float* data, unsigned int* index, unsigned int size) {
    std::priority_queue<std::pair<float, unsigned int>, std::vector<std::pair<float, unsigned int>>,
                        std::greater<std::pair<float, unsigned int>>>
            q;
    for (unsigned int i = 0; i < size; ++i) {
        q.push(std::pair<float, unsigned int>(data[i], i));
    }
    for (int i = 0; i < n_index; ++i) {
        int ki = q.top().second;
        index[i] = ki;
        q.pop();
    }
}

void neighbours_idx(VPUNN::Tensor<float>* weights, VPUNN::Tensor<float>* activations, int n_neighbours,
                    VPUNN::Tensor<unsigned int>& indexes, VPUNN::Tensor<float>& distances) {
    // activations is of the shape [Batch, embedding]
    // weights is of the shape [items, embedding]
    unsigned int embedding_shape = weights->shape()[1];
    unsigned int batch_size = activations->shape()[0];
    unsigned int items = weights->shape()[0];

    // compute the matmul between activations and weights: 1 - A * W.T
    // A is of shape [batch, embeddings] => shape: m, k
    // W.t is of shape [embeddings, items] => shape: k, n
    // output is of shape [batch, items] => shape: m, n
    // m = batch, k = embeddings, n = items
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans, batch_size, items, embedding_shape, -1.0, activations->c_ptr(),
                batch_size, weights->c_ptr(), items, 1.0, distances.c_ptr(), batch_size);

    // This piece of code is hideous, I think it can be parallelized at least for batch exploration
    for (unsigned int b_idx = 0; b_idx < batch_size; b_idx++) {
        n_index(n_neighbours, &(distances[b_idx * items]), indexes.c_ptr() + b_idx * n_neighbours, items);
    }
}

void VPUNN::kNN(VPUNN::Tensor<float>* weights, VPUNN::Tensor<float>* targets, VPUNN::Tensor<float>* activations,
                VPUNN::Tensor<float>* output, unsigned int n_neighbours) {
    unsigned int batch_size = activations->shape()[0];
    unsigned int items = weights->shape()[0];

    // Tensor containing all the index of
    auto indexes = VPUNN::Tensor<unsigned int>({batch_size, n_neighbours});
    // Initialize the distance vector to be all 1's
    auto distances = VPUNN::Tensor<float>({batch_size, items}, 1.0);
    neighbours_idx(weights, activations, n_neighbours, indexes, distances);

    // Compute the weighted average of the distances
    for (unsigned int b_idx = 0; b_idx < batch_size; b_idx++) {
        float sum = 0, prediction = 0;
        for (unsigned int idx = 0; idx < n_neighbours; idx++) {
            float weight = 1.0f / (distances[indexes[idx]] + 1e-12f);
            prediction += targets->c_ptr()[indexes[idx]] * weight;
            sum += weight;
        }
        output->c_ptr()[b_idx * n_neighbours] = prediction / sum;
    }
}
