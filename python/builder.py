# Copyright © 2023 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
# LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
# is subject to the terms and conditions of the software license agreements for the Software Package,
# which may also include notices, disclaimers, or license terms for third party or open source software
# included in or with the Software Package, and your use indicates your acceptance of all such terms.
# Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
# Software Package for additional details.

from dataclasses import dataclass
import tensorflow as tf
import numpy as np
import flatbuffers
import argparse

import VPUNN_SCHEMA.ActivationFunctionType  # noqa
import VPUNN_SCHEMA.FullyConnectedLayer  # noqa
import VPUNN_SCHEMA.kNNLayer  # noqa
import VPUNN_SCHEMA.TensorType  # noqa
import VPUNN_SCHEMA.LayerType  # noqa
import VPUNN_SCHEMA.Model  # noqa
import VPUNN_SCHEMA.Buffer  # noqa
import VPUNN_SCHEMA.Layer  # noqa
import VPUNN_SCHEMA.Tensor  # noqa


def FinishFileID(builder, rootTable, fid):
    """
    This is a temporary function to workaround a bug in the
    current release of flatbuffers. When it is resolved by
    the maintainers, this can be removed.
    - Writes some header fields that were missing including
    the magic number identfier
    @param builder - flat buffer builder object
    @param rootTable - flat buffer current object
    @param fid - magic number for identification, must be 4 chars and ascii encoded
    """
    N = flatbuffers.number_types
    encode = flatbuffers.encode
    flags = N.Uint8Flags
    prepSize = N.Uint8Flags.bytewidth * len(fid)
    builder.Prep(builder.minalign, prepSize + len(fid))
    for i in range(len(fid) - 1, -1, -1):
        builder.head = builder.head - flags.bytewidth
        encode.Write(flags.packer_type, builder.Bytes, builder.Head(), fid[i])
    builder.Finish(rootTable)


class SerialModel:
    def __init__(self, name, tensors, inputs, outputs, layers, buffers):
        self.name = name
        self.tensors = tensors
        self.inputs = inputs
        self.outputs = outputs
        self.layers = layers
        self.buffers = buffers

    def build(self, fbb):

        built_tensors = [tensor._fb(fbb) for tensor in self.tensors]
        built_layers = [layer._fb(fbb) for layer in self.layers]
        built_buffers = [buffer._fb(fbb) for buffer in self.buffers]

        VPUNN_SCHEMA.Model.ModelStartTensorsVector(fbb, len(built_tensors))
        for tensor in reversed(built_tensors):
            fbb.PrependUOffsetTRelative(tensor)
        serial_tensors = fbb.EndVector(len(built_tensors))

        VPUNN_SCHEMA.Model.ModelStartOperatorsVector(fbb, len(built_layers))
        for layer in reversed(built_layers):
            fbb.PrependUOffsetTRelative(layer)
        serial_layers = fbb.EndVector(len(built_layers))

        VPUNN_SCHEMA.Model.ModelStartInputsVector(fbb, len(self.inputs))
        for input in reversed(self.inputs):
            fbb.PrependUint32(input)
        serial_inputs = fbb.EndVector(len(self.inputs))

        VPUNN_SCHEMA.Model.ModelStartOutputsVector(fbb, len(self.outputs))
        for input in reversed(self.outputs):
            fbb.PrependUint32(input)
        serial_outputs = fbb.EndVector(len(self.outputs))

        VPUNN_SCHEMA.Model.ModelStartBuffersVector(fbb, len(built_buffers))
        for buffer in reversed(built_buffers):
            fbb.PrependUOffsetTRelative(buffer)
        serial_buffers = fbb.EndVector(len(built_buffers))

        serial_name = fbb.CreateString(self.name)

        VPUNN_SCHEMA.Model.ModelStart(fbb)
        VPUNN_SCHEMA.Model.ModelAddName(fbb, serial_name)
        VPUNN_SCHEMA.Model.ModelAddTensors(fbb, serial_tensors)
        VPUNN_SCHEMA.Model.ModelAddOperators(fbb, serial_layers)
        VPUNN_SCHEMA.Model.ModelAddInputs(fbb, serial_inputs)
        VPUNN_SCHEMA.Model.ModelAddOutputs(fbb, serial_outputs)
        VPUNN_SCHEMA.Model.ModelAddBuffers(fbb, serial_buffers)
        return VPUNN_SCHEMA.Model.ModelEnd(fbb)


class SerialTensor:
    def __init__(self, tensor, buffer_map):
        self.name = tensor.name
        if type(tensor.shape) in [list, tuple]:
            self.shape = tensor.shape
        else:
            self.shape = tensor.shape.as_list()
        self.dtype = tensor.dtype

        if self.name in buffer_map:
            self.buffer = buffer_map.index(self.name)
        else:
            self.buffer = 0

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return (
            isinstance(other, SerialTensor)
            and self.name == other.name
            and self.shape == other.shape
            and self.dtype == other.dtype
        )

    def _fb(self, fbb):

        VPUNN_SCHEMA.Tensor.TensorStartShapeVector(fbb, len(self.shape))
        for dim in reversed(self.shape):
            dim = dim if dim is not None else 1
            fbb.PrependUint32(dim)
        serial_shape = fbb.EndVector(len(self.shape))

        serial_name = fbb.CreateString(self.name)

        VPUNN_SCHEMA.Tensor.TensorStart(fbb)
        VPUNN_SCHEMA.Tensor.TensorAddName(fbb, serial_name)
        VPUNN_SCHEMA.Tensor.TensorAddShape(fbb, serial_shape)
        VPUNN_SCHEMA.Tensor.TensorAddBuffer(fbb, self.buffer)
        return VPUNN_SCHEMA.Tensor.TensorEnd(fbb)


class SerialLayer:
    def __init__(self, layer, tensor_map):
        self.name = layer.name
        self.input_tensors = [
            resolve_tensor_index(tensor, tensor_map)
            for tensor in get_input_tensors(layer)
        ]
        self.output_tensors = [
            resolve_tensor_index(tensor, tensor_map)
            for tensor in get_output_tensors(layer)
        ]

        activation_dict = {
            "relu": VPUNN_SCHEMA.ActivationFunctionType.ActivationFunctionType.RELU,
            "sigmoid": VPUNN_SCHEMA.ActivationFunctionType.ActivationFunctionType.SIGMOID,
            "linear": VPUNN_SCHEMA.ActivationFunctionType.ActivationFunctionType.NOOP,
        }
        if hasattr(layer, "activation"):
            self.activation = activation_dict[layer.activation.__name__]
        else:
            self.activation = (
                VPUNN_SCHEMA.ActivationFunctionType.ActivationFunctionType.NOOP
            )

        if isinstance(layer, tf.keras.layers.Dense):
            self.implementation = VPUNN_SCHEMA.LayerType.LayerType.FullyConnectedLayer
        elif isinstance(layer, kNNDatabase):
            self.implementation = VPUNN_SCHEMA.LayerType.LayerType.kNNLayer
            self.n_neighbors = layer.n_neighbors
        else:
            self.implementation = VPUNN_SCHEMA.LayerType.LayerType.L2NormalizationLayer

    def _fb(self, fbb):

        VPUNN_SCHEMA.Layer.LayerStartInputsVector(fbb, len(self.input_tensors))
        for dim in reversed(self.input_tensors):
            dim = dim if dim is not None else 0
            fbb.PrependUint32(dim)
        serial_input_tensors = fbb.EndVector(len(self.input_tensors))

        VPUNN_SCHEMA.Layer.LayerStartOutputsVector(fbb, len(self.output_tensors))
        for dim in reversed(self.output_tensors):
            dim = dim if dim is not None else 0
            fbb.PrependUint32(dim)
        serial_output_tensors = fbb.EndVector(len(self.output_tensors))

        serial_name = fbb.CreateString(self.name)

        self.impl_type = None
        if self.implementation == VPUNN_SCHEMA.LayerType.LayerType.kNNLayer:
            VPUNN_SCHEMA.kNNLayer.kNNLayerStart(fbb)
            VPUNN_SCHEMA.kNNLayer.kNNLayerAddNNeighbors(fbb, self.n_neighbors)
            self.impl_type = VPUNN_SCHEMA.kNNLayer.kNNLayerEnd(fbb)

        VPUNN_SCHEMA.Layer.LayerStart(fbb)
        VPUNN_SCHEMA.Layer.LayerAddName(fbb, serial_name)
        VPUNN_SCHEMA.Layer.LayerAddInputs(fbb, serial_input_tensors)
        VPUNN_SCHEMA.Layer.LayerAddOutputs(fbb, serial_output_tensors)
        VPUNN_SCHEMA.Layer.LayerAddActivationFunction(fbb, self.activation)

        VPUNN_SCHEMA.Layer.LayerAddImplementationType(fbb, self.implementation)
        if self.impl_type:
            VPUNN_SCHEMA.Layer.LayerAddImplementation(fbb, self.impl_type)
        return VPUNN_SCHEMA.Layer.LayerEnd(fbb)


class SerialBuffer:
    def __init__(self, tensor=None):
        if tensor is not None:
            self.data = tensor.numpy()
            self.name = tensor.name
        else:
            self.data = np.zeros(1).astype(np.float32)
            self.name = None

    def _fb(self, fbb):

        if len(self.data.shape) == 2:
            # Transpose to (output_channels, input_channels) format
            self.data = np.transpose(self.data, (1, 0))

        packed_view = self.data.flatten().view(dtype=np.uint8)

        VPUNN_SCHEMA.Buffer.BufferStartDataVector(fbb, len(packed_view))
        for val in reversed(packed_view):
            fbb.PrependUint8(val)
        packed_data = fbb.EndVector(len(packed_view))

        VPUNN_SCHEMA.Buffer.BufferStart(fbb)
        VPUNN_SCHEMA.Buffer.BufferAddData(fbb, packed_data)
        return VPUNN_SCHEMA.Buffer.BufferEnd(fbb)


def resolve_tensor_index(tensor, tensor_map):
    return tensor_map.index(tensor.name)


def get_input_tensors(layer):
    return [layer.input] + layer.weights


def get_output_tensors(layer):
    return [layer.output]


def get_tensors(layer):
    return get_input_tensors(layer) + get_output_tensors(layer)


def preprocess_model(model):
    # Get the dense feature layers
    dense_features = model.get_layer("dense_features")

    # Extract the preprocessing
    _ = tf.keras.Model(inputs=model.inputs, outputs=dense_features.output)

    # Get the remaining layers
    layers = model.layers[model.layers.index(dense_features) + 1 :]

    compute_model = tf.keras.Sequential(layers, name="VPUNN")

    compute_model.build(dense_features.output.shape)

    # # Print the compute module
    compute_model.summary()

    return compute_model


def convert_model(tflite_model_path, vpunn_model_path):

    print(f"model_name: {tflite_model_path}")

    # Load the model
    model = tf.keras.models.load_model(tflite_model_path)

    model = preprocess_model(model)

    # Serialize the model
    serialize_model(model, vpunn_model_path)


@dataclass
class kNNTensor:
    data: np.array
    name: str

    @property
    def shape(self):
        return self.data.shape

    def numpy(self):
        return self.data

    @property
    def dtype(self):
        return self.data.dtype


class kNNDatabase:
    def __init__(self, W, targets, n_neighbors):
        self.W = kNNTensor(W.astype(np.float32), "kNN_W")
        self.targets = kNNTensor(targets.astype(np.float32), "kNN_t")
        self.n_neighbors = n_neighbors
        self.name = "kNN"

    @property
    def weights(self):
        return [self.W, self.targets]

    @property
    def output(self):
        return kNNTensor(np.zeros((1, self.targets.shape[-1])), "kNN_out")


def serialize_model(model, output_path, name="VPUNN", database=None, verbose=True):

    # List all the layers in
    layers = model.layers if model else []
    if database is not None:
        if model:
            database.input = layers[-1].output
        else:
            database.input = kNNTensor(np.zeros((1, database.W.shape[-1])), "kNN_in")
        layers.append(database)

    buffers = sum(
        [[SerialBuffer(tensor) for tensor in layer.weights] for layer in layers], []
    )

    buffers = [SerialBuffer()] + buffers

    # Build the buffer map
    buffer_map = [tensor.name for tensor in buffers]

    tensors = list(
        set(
            sum(
                [
                    [SerialTensor(tensor, buffer_map) for tensor in get_tensors(layer)]
                    for layer in layers
                ],
                [],
            )
        )
    )

    tensor_map = [tensor.name for tensor in tensors]

    if model:
        # Remove the first layer in case of TF network
        layers = layers[1:]
    layers = [SerialLayer(layer, tensor_map) for layer in layers]

    # Only the activations
    input_tensors = [layers[0].input_tensors[0]]
    output_tensors = layers[-1].output_tensors

    if verbose:
        print("Generating model....")

    model = SerialModel(name, tensors, input_tensors, output_tensors, layers, buffers)

    fbb = flatbuffers.Builder(1024)
    serial_model = model.build(fbb)
    FinishFileID(fbb, serial_model, "VPUN".encode())

    with open(output_path, "wb") as output:
        output.write(fbb.Output())

    if verbose:
        print(f"Model written to {output_path}")


def define_and_parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a VPUNN model from a tensorflow one"
    )

    parser.add_argument("--name", required=True, type=str, help="Model name")
    parser.add_argument(
        "--output",
        type=str,
        default="./model.vpunn",
        help="Output model (default model.vpunn)",
    )

    return parser.parse_args()


def main():
    args = define_and_parse_args()
    convert_model(args.name, args.output)


if __name__ == "__main__":

    main()
