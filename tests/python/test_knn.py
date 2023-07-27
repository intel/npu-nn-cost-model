# Copyright © 2023 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
# LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
# is subject to the terms and conditions of the software license agreements for the Software Package,
# which may also include notices, disclaimers, or license terms for third party or open source software
# included in or with the Software Package, and your use indicates your acceptance of all such terms.
# Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
# Software Package for additional details.


from vpunn.kNN import kNearestNeighbors
import numpy as np
import pytest
import random


def normalize(X, axis=0):
    for idx in range(X.shape[axis]):
        X[idx, :] = X[idx, :] / np.linalg.norm(X[idx, :])


@pytest.mark.parametrize("n_items", [10, 15, 20, 30, 200, 1000])
@pytest.mark.parametrize("embeddings", [5, 10, 20])
@pytest.mark.parametrize("n_neighbors", [1, 5])
@pytest.mark.parametrize("batch", [1, 2, 4])
def test_kNN(n_items, embeddings, n_neighbors, batch):

    # Tolerances
    rtol, atol = 1e-03, 1e-03

    model = kNearestNeighbors(n_neighbors)

    if embeddings is None:
        embeddings = n_items

    selected_item = random.randint(0, n_items - batch)

    X = np.random.uniform(-1, 1, (n_items, embeddings)).astype(np.float32)
    normalize(X)

    # Test that the input has unitary L2 norm
    for idx in range(n_items):
        norm = np.linalg.norm(X[idx, :])
        assert np.allclose(norm, 1.0, rtol, atol)

    targets = np.random.uniform(-1, 1, (n_items)).astype(np.float32)

    selected_target = targets[selected_item : selected_item + batch]

    model.fit(X, targets)
    model_input = X[selected_item : selected_item + batch, :]

    reference_output = model.predict_reference(model_input).flatten()
    output = model.predict(model_input).flatten()

    print(f"\nModel input: {model_input}")
    print(f"Selected item: {selected_item}")
    print(f"Selected target: {selected_target}")
    print(f"Reference output: {reference_output}")
    print(f"Model output: {output}")

    # Equal with tolerance

    assert np.allclose(selected_target, reference_output, rtol, atol)
    assert np.allclose(reference_output, output, rtol, atol)
