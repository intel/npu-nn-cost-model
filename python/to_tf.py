# Copyright © 2022 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
# LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
# is subject to the terms and conditions of the software license agreements for the Software Package,
# which may also include notices, disclaimers, or license terms for third party or open source software
# included in or with the Software Package, and your use indicates your acceptance of all such terms.
# Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
# Software Package for additional details.

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)  # noqa

import tensorflow as tf
import numpy as np
import argparse

import VPUNN_SCHEMA.ActivationFunctionType  # noqa
import VPUNN_SCHEMA.FullyConnectedLayer  # noqa
import VPUNN_SCHEMA.TensorType  # noqa
import VPUNN_SCHEMA.LayerType  # noqa
import VPUNN_SCHEMA.Model  # noqa
import VPUNN_SCHEMA.Buffer  # noqa
import VPUNN_SCHEMA.Layer  # noqa
import VPUNN_SCHEMA.Tensor  # noqa


def get_activation(activation):
    if activation == VPUNN_SCHEMA.ActivationFunctionType.ActivationFunctionType.RELU:
        return "relu"
    return None


def create_tf_model(vpunn_model, remove_head=False):

    print(f"Model: {vpunn_model}")

    buf = open(vpunn_model, "rb").read()
    buf = bytearray(buf)
    vpunn = VPUNN_SCHEMA.Model.Model.GetRootAsModel(buf, 0)

    print(f"Model name: {vpunn.Name().decode('ascii')}")
    print(f"\t inputs: {vpunn.InputsLength()}")
    print(f"\t outputs: {vpunn.OutputsLength()}")
    print(f"\t tensors: {vpunn.TensorsLength()}")
    print(f"\t operations: {vpunn.OperatorsLength()}")

    tensors = [vpunn.Tensors(idx) for idx in range(vpunn.TensorsLength())]
    buffers = [vpunn.Buffers(idx) for idx in range(vpunn.BuffersLength())]
    operators = [vpunn.Operators(idx) for idx in range(vpunn.OperatorsLength())]
    inputs = [vpunn.Inputs(idx) for idx in range(vpunn.InputsLength())]
    outputs = [vpunn.Outputs(idx) for idx in range(vpunn.OutputsLength())]

    assert len(inputs) == 1
    assert len(outputs) == 1

    input_tensor = tensors[inputs[0]]

    input_layer = tf.keras.Input(shape=(input_tensor.ShapeAsNumpy()[-1],), name="input")
    out = input_layer
    for idx, op in enumerate(operators):
        if idx + 1 == len(operators) and remove_head:
            print("Skip last layer")
            continue
        name = op.Name().decode("ascii")
        activation = get_activation(op.ActivationFunction())
        inputs = [tensors[idx] for idx in list(op.InputsAsNumpy())]
        outputs = [tensors[idx] for idx in list(op.OutputsAsNumpy())]

        if (
            VPUNN_SCHEMA.LayerType.LayerType.FullyConnectedLayer
            == op.ImplementationType()
        ):
            kernel_shape = inputs[1].ShapeAsNumpy()
            weights = [
                buffers[wt.Buffer()]
                .DataAsNumpy()
                .view(np.float32)
                .reshape(wt.ShapeAsNumpy())
                for wt in inputs[1:]
            ]

            has_bias = len(weights) > 1
            weight = tf.constant_initializer(weights[0])
            bias = tf.constant_initializer(weights[1]) if has_bias else None
            layer = tf.keras.layers.Dense(
                kernel_shape[-1],
                use_bias=has_bias,
                kernel_initializer=weight,
                bias_initializer=bias,
                name=name,
                activation=activation,
            )
            # set weights and bias
            print(name, "FC", activation, kernel_shape)
            out = layer(out)
        elif (
            VPUNN_SCHEMA.LayerType.LayerType.L2NormalizationLayer
            == op.ImplementationType()
        ):
            print(name, "L2", activation)
            out = tf.math.l2_normalize(out, axis=-1)
        else:
            print(name, "None", activation, inputs, outputs)

    model = tf.keras.Model(inputs=input_layer, outputs=out)

    model.summary()

    return model


def define_and_parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str, help="VPUNN model")

    return parser.parse_args()


if __name__ == "__main__":
    args = define_and_parse_args()
    create_tf_model(args.model)
