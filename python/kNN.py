# Copyright © 2022 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
# LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
# is subject to the terms and conditions of the software license agreements for the Software Package,
# which may also include notices, disclaimers, or license terms for third party or open source software
# included in or with the Software Package, and your use indicates your acceptance of all such terms.
# Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
# Software Package for additional details.

from vpunn.builder import serialize_model, kNNDatabase
from vpunn.model import VPUNN
import numpy as np
import tempfile


class kNearestNeighbors:
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
        self.W = None
        self.targets = None
        self.model = None
        self.db = None

    def fit(self, X, y):

        self.W = X
        self.targets = y

        if len(y.shape) < 2:
            db_targets = np.expand_dims(y, -1)
        else:
            db_targets = y

        self.db = kNNDatabase(self.W, db_targets, self.n_neighbors)
        with tempfile.NamedTemporaryFile(suffix=".vpunn") as tmp:
            serialize_model(None, tmp.name, database=self.db, verbose=False)
            self.model = VPUNN(tmp.name)
        return self

    def predict(self, X):
        batch = []
        for batch_idx in range(X.shape[0]):
            val = self.model.run_inference(X[batch_idx, :])
            batch.append(val.copy())
        return np.vstack(batch)

    def transform(self, X):
        return self.predict(X)

    def metric(self, X):
        # X is of the shape [Batch, embedding]
        # W is of the shape [items, embedding]
        return 1 - np.matmul(X, self.W.T)

    def neighbors(self, X):
        # Distances should be of shape [batch, items]
        distances = self.metric(X)

        # Duplicate the targets for batch addressing
        targets = np.tile(self.targets, (X.shape[0], 1))

        # Get the topK index across the items
        topK_idx = np.argpartition(distances, self.n_neighbors, axis=-1)[
            :, : self.n_neighbors
        ]

        # this should be of size [batch, self.n_neighbors]
        top_distances = np.take_along_axis(distances, topK_idx, axis=-1)

        # this should be of size [batch, self.n_neighbors]
        top_target = np.take_along_axis(targets, topK_idx, axis=-1)

        return top_distances, top_target

    def prediction_error(self, weights, targets, target):
        # Compute target pdf from the weights
        pdf = weights / np.sum(weights, axis=-1)

        # Standard deviation
        std = np.sqrt(np.einsum("bk,bk->b", np.power(targets - target, 2), pdf))

        return std

    def predict_reference(self, X, prediction_error=False):

        # Get the top distances and targets
        distances, targets = self.neighbors(X)

        # The weights are the inverse of the distance
        weights = 1 / (distances + 1e-12)

        # Compute the dot product between b and k index
        # The expected shape is [batch, 1]
        target = np.einsum("bk,bk->b", targets, weights) / np.sum(weights, axis=-1)

        if prediction_error:
            target_prediction_error = self.prediction_error(weights, targets, target)
            return target, target_prediction_error
        return target

    def uniques(self):
        self.W, idx = np.unique(self.W, return_index=True, axis=0)
        self.targets = self.targets[idx]
