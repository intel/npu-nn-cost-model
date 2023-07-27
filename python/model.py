# Copyright © 2023 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
# LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
# is subject to the terms and conditions of the software license agreements for the Software Package,
# which may also include notices, disclaimers, or license terms for third party or open source software
# included in or with the Software Package, and your use indicates your acceptance of all such terms.
# Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
# Software Package for additional details.

from array import array
import numpy as np
import argparse

from vpunn import VPUNN_lib  # noqa


class VPUNN:
    def __init__(self, filename, profile=False, batch=1):
        self.model = VPUNN_lib.Runtime(filename, batch, profile)

        assert len(self.model.input_shapes()) == 1
        assert len(self.model.output_shapes()) == 1

    def input_shapes(self):
        return [tuple(shape) for shape in self.model.input_shapes()]

    def output_shapes(self):
        return [tuple(shape) for shape in self.model.output_shapes()]

    def run_inference(self, features):
        feature_array = array("f", features.tobytes())

        result_shape = self.output_shapes()[0]
        arr = self.model.predict(feature_array)

        data = np.array(
            arr,
            dtype=features.dtype,
        )

        return data.reshape(result_shape)


def main():
    parser = argparse.ArgumentParser(description="VPUNN runtime example")

    parser.add_argument("model", type=str, help="Model path")
    parser.add_argument("--profile", action="store_true", help="Enable profiling")
    parser.add_argument(
        "--batch", type=int, default=1, help="Set a specific batch size"
    )

    args = parser.parse_args()

    # Load the model
    model = VPUNN(args.model, profile=args.profile, batch=args.batch)

    # Some random features
    features = np.random.randint(0, 2, model.input_shapes()[0])

    # Get the cycles
    cycles = model.run_inference(features.astype(np.float32))

    print(f"Cycles: {cycles}")


if __name__ == "__main__":
    # Running the main function
    main()
