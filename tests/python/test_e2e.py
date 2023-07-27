# Copyright © 2023 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
# LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
# is subject to the terms and conditions of the software license agreements for the Software Package,
# which may also include notices, disclaimers, or license terms for third party or open source software
# included in or with the Software Package, and your use indicates your acceptance of all such terms.
# Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
# Software Package for additional details.


import os

# Disable GPU when running this test
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # noqa

import pytest  # noqa
import shutil  # noqa
import tempfile  # noqa
import numpy as np  # noqa
import tensorflow as tf  # noqa


# Append python path

from vpunn.model import VPUNN  # noqa
from vpunn.builder import serialize_model  # noqa

# Seed for reproducibility
np.random.seed(42)
if int(tf.__version__.split(".")[0]) < 2:
    tf.compat.v1.enable_eager_execution()
    tf.random.set_random_seed(42)
else:
    tf.random.set_seed(42)


def generate_model(layers, input_tensor, normalize):

    input_shape = (input_tensor.shape[-1],)
    input_layer = tf.keras.Input(shape=input_shape, name="input")

    output_layer = input_layer
    for layer in layers:
        print(output_layer.shape, input_layer.shape)
        output_layer = layer(output_layer)
    # L2 normalizse the layer
    if normalize:

        def norm(x):
            return tf.math.l2_normalize(x, axis=-1)

        output_layer = tf.keras.layers.Lambda(norm)(output_layer)

    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

    # Compile the model with an input
    output = model(input_tensor)

    model.summary()

    dirpath = tempfile.mkdtemp()

    model_path = os.path.join(dirpath, "model.vpunn")

    serialize_model(model, model_path)

    return model_path, np.array(output)


@pytest.mark.parametrize("batch", [1, 2, 4, 8])
@pytest.mark.parametrize("input_channels", [1, 4, 10, 16, 32, 64, 128])
@pytest.mark.parametrize("output_channels", [1, 4, 10, 16, 32, 64, 128])
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("activation", [None, "relu", "sigmoid"])
@pytest.mark.parametrize("layers", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("normalize", [False, True])
def test_arithmetics(
    batch, input_channels, output_channels, bias, activation, layers, normalize
):

    input_tensor = np.random.uniform(-100, 100, (batch, input_channels)).astype(
        np.float32
    )

    ones = tf.constant_initializer(1.0)
    value = tf.constant_initializer(0.22883666)

    if normalize and activation == "sigmoid":
        pytest.skip("Unsupported configuration")

    fc_layers = [
        tf.keras.layers.Dense(
            output_channels,
            #  kernel_initializer=value,
            #  bias_initializer=value,
            activation=activation,
            use_bias=bias,
        )
        for _ in range(layers)
    ]
    model_path, reference_output = generate_model(fc_layers, input_tensor, normalize)

    model = VPUNN(model_path, batch=batch)

    print(input_tensor)
    print(reference_output)

    # First inference
    output = model.run_inference(input_tensor)

    # Second inference
    output2 = model.run_inference(input_tensor)

    print(output)
    print(output2)

    shutil.rmtree(os.path.split(model_path)[0])

    # Equal with tolerance
    rtol, atol = 1e-03, 1e-03
    assert np.allclose(reference_output, output, rtol, atol)
    assert np.allclose(reference_output, output2, rtol, atol)


# test_arithmetics(batch=2, input_channels=2, output_channels=2,
#                  bias=False, activation="sigmoid", layers=2, normalize=False)
