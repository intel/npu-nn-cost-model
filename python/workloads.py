# Copyright © 2023 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
# LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
# is subject to the terms and conditions of the software license agreements for the Software Package,
# which may also include notices, disclaimers, or license terms for third party or open source software
# included in or with the Software Package, and your use indicates your acceptance of all such terms.
# Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
# Software Package for additional details.

import json
import hashlib
import numpy as np
from typing import List
from math import ceil
from dataclasses import dataclass, asdict

sample = False
verbosity = 2
safety_factor = 1.0

vpu2_0_devices = {"VPUDevice.VPU_2_0": 1, "VPUDevice.VPU_2_1": 1}
vpu2_7_devices = {"VPUDevice.VPU_2_7": 2}

cmx_sizes = {}
cmx_sizes.update(vpu2_0_devices)
cmx_sizes.update(vpu2_7_devices)


@dataclass(eq=True, unsafe_hash=True)
class DPULayer:
    """
    DPULayer class models a valid layer that can be computed by the VPU. A layer can be split in multiple DPU workloads.
    A DPU Layer needs to satisfy different constraints compared to a DPU workload, for example:
    - it can have a memory footprint bigger that the actual memory size
    - the number of channels required by SOK must be bigger than 32 (as 16 cannot be split over 2 workloads)
    - the layouts, swizzling and execution order can be left blank as they can be decided later by the compiler
    """

    device: str = None
    operation: str = None
    input_0_batch: int = None
    input_0_channels: int = None
    input_0_height: int = None
    input_0_width: int = None
    input_1_batch: int = None
    input_1_channels: int = None
    input_1_height: int = None
    input_1_width: int = None
    input_sparsity_enabled: bool = None
    weight_sparsity_enabled: bool = None
    input_sparsity_rate: float = None
    weight_sparsity_rate: float = None
    execution_order: str = None
    activation_function: str = None
    kernel_height: int = None
    kernel_width: int = None
    kernel_pad_bottom: int = None
    kernel_pad_left: int = None
    kernel_pad_right: int = None
    kernel_pad_top: int = None
    kernel_stride_height: int = None
    kernel_stride_width: int = None
    output_0_batch: int = None
    output_0_channels: int = None
    output_0_height: int = None
    output_0_width: int = None
    input_0_datatype: str = None
    input_0_layout: str = None
    input_0_swizzling: str = None
    input_1_datatype: str = None
    input_1_layout: str = None
    input_1_swizzling: str = None
    output_0_datatype: str = None
    output_0_layout: str = None
    output_0_swizzling: str = None
    isi_strategy: str = None
    output_write_tiles: int = None
    output_sparsity_enabled: bool = None

    def print_error(self, string):
        if verbosity >= 1:
            print("\033[91mError: " + string + "\033[0m")

    def print_warning(self, string):
        if verbosity >= 2:
            print("\033[33mWarning: " + string + "\033[0m")

    def replace_none(self, field_name, default_value, _range=None):
        """
        Replace self.<field_name> with a default value if it is None
        Checks if the specified value is in _range.
        Args:
            field_name: one of the field of the dataclass
            default_value: default value of the field
            _range: set of numbers
        Returns:
            None
        """
        if getattr(self, field_name) is None:
            if not sample and default_value is None:
                if not (type(_range) == list and None in _range):
                    raise ValueError(
                        f"{field_name} was not specified when creating the workload."
                    )
            setattr(self, field_name, default_value)
        elif _range is not None:
            nl = "\n"
            assert (
                getattr(self, field_name) in _range
            ), f"Selected {field_name} ({getattr(self, field_name)}) is not in the set { str(_range).replace(nl, '')}"

    def safe_assign(self, field_name, expected_value):
        """
        Ensure that self.<field_name> is equals to a certain expected value
        Args:
            field_name: one of the field of the dataclass
            expected_value: the expected value
        Returns:
            None
        """
        if getattr(self, field_name) is not None:
            assert (
                getattr(self, field_name) == expected_value
            ), f"Optional field '{field_name}' was assigned to {getattr(self, field_name)}, which is different from the expected value ({expected_value})"
        else:
            setattr(self, field_name, expected_value)

    @staticmethod
    def sample_list(elements, uniform=True):
        """
        Selects an element from a list of elements based on their position and returns the selected element.

        Args:
        - elements (list): A list of elements to sample from.
        - uniform (bool): If True, samples elements uniformly from the list. If False, samples elements
            with a probability distribution that increases linearly with the element's position in the list.

        Returns:
        - A tuple containing the selected element and the original list of elements.

        Raises:
        - TypeError: If elements is not a list or uniform is not a boolean.
        - ValueError: If elements is an empty list or contains only None values.
        """
        global sample
        if not sample:
            return None, elements

        if None in elements:
            elements.remove(None)
        gend = np.vectorize(lambda x, a: 1 / (x + 1 - (x * a)))
        prob_distributions = np.array(gend(np.arange(len(elements)), uniform))
        prob_distributions /= sum(prob_distributions)
        return np.random.choice(elements, p=prob_distributions).item(), elements

    def align_to(self, x, multiple):
        return x - (x % -multiple)

    # since not all sparsity levels are valid we round to a valid value
    def sanitize_sparsity(self, tensor_size, sparsity_level):
        # here we use 32 instead of 16 to take into account SOK sparsity
        number_of_zeroes = round((tensor_size / 32) * sparsity_level) * 32

        # The blob stores the density rate as a float32 (which is not always enough)
        # in order to match the generated density rate with the one in the blob we cast this value
        # the alternative would be to never use DensityRate() but to compute it starting from the weight data
        return 1 - np.float32(1 - number_of_zeroes / tensor_size).item()

    def __post_init__(self):

        sample_list = DPUWorkload.sample_list

        valid_devices = [*vpu2_0_devices.keys(), *vpu2_7_devices.keys()]
        valid_operations = [
            "Operation.CONVOLUTION",
            "Operation.DW_CONVOLUTION",
            "Operation.CM_CONVOLUTION",
            "Operation.ELTWISE",
            "Operation.MAXPOOL",
        ]

        valid_isi_strategy = [
            "ISIStrategy.CLUSTERING",
            "ISIStrategy.SPLIT_OVER_H",
            "ISIStrategy.SPLIT_OVER_K",
            None,
        ]

        valid_activation_function = [
            "ActivationFunction.NONE",
            "ActivationFunction.RELU",
            "ActivationFunction.LRELU",
            "ActivationFunction.ADD",
            "ActivationFunction.SUB",
            "ActivationFunction.MULT",
        ]

        self.replace_none("device", "VPUDevice.VPU_2_7", valid_devices)

        if self.operation == "Operation.ELTWISE":
            del valid_isi_strategy[2]  # ELTWISE not possible on SOK
        self.replace_none("output_write_tiles", 1)

        if self.output_write_tiles > 1:
            self.replace_none("isi_strategy", *sample_list(valid_isi_strategy))
        else:
            self.safe_assign(
                "isi_strategy", "ISIStrategy.CLUSTERING"
            )  # no split if output_write_tiles =1
        if self.isi_strategy == "ISIStrategy.SPLIT_OVER_K":
            startidx = 1
        else:
            startidx = 0

        if self.device in vpu2_0_devices:
            valid_execution_order = [
                "ExecutionMode.VECTOR_FP16",  # 4x1
                "ExecutionMode.VECTOR",  # 16x1
                "ExecutionMode.MATRIX",  # 4x4
                None,
            ]
            # should be Swizzling.INVALID
            valid_swizzlings = ["Swizzling.KEY_0"]
            valid_layouts = ["Layout.ZXY", "Layout.XYZ", None]  # -> ZMajor  # -> CMajor
            valid_input_channels = {
                "Operation.CONVOLUTION": np.arange(1, 512) * 16,
                "Operation.DW_CONVOLUTION": np.arange(1, 512) * 16,
                "Operation.CM_CONVOLUTION": np.arange(2, 16),
                "Operation.ELTWISE": np.arange(1, 512) * 16,
                "Operation.MAXPOOL": np.arange(1, 512) * 16,
            }
        else:  # not 2.0
            valid_execution_order = [
                "ExecutionMode.CUBOID_4x16",
                "ExecutionMode.CUBOID_8x16",
                "ExecutionMode.CUBOID_16x16",
                None,
            ]
            valid_swizzlings = [
                "Swizzling.KEY_0",
                "Swizzling.KEY_1",
                "Swizzling.KEY_2",
                "Swizzling.KEY_3",
                "Swizzling.KEY_4",
                "Swizzling.KEY_5",
                None,
            ]

            valid_layouts = [
                "Layout.ZXY",
                "Layout.XZY",
                "Layout.YXZ",
                "Layout.YZX",
                "Layout.ZYX",
                "Layout.XYZ",
                "Layout.INVALID",
                None,
            ]

            valid_input_channels = {
                "Operation.CONVOLUTION": np.arange(1, 512) * 16,
                "Operation.DW_CONVOLUTION": [16, 32, 64][startidx:],
                "Operation.CM_CONVOLUTION": np.arange(2, 16),
                "Operation.ELTWISE": np.arange(1, 512)[startidx:] * 16,
                "Operation.MAXPOOL": [16, 32, 64][startidx:],
            }

        quantized_datatypes = ["DataType.UINT8", "DataType.INT8"]

        float_datatypes = ["DataType.FLOAT16", "DataType.BFLOAT16"]

        valid_datatypes = [*quantized_datatypes, *float_datatypes]

        if self.isi_strategy == "ISIStrategy.SPLIT_OVER_K":
            del valid_operations[3]  # SOK not compatible with ELEMENTWISE
        self.replace_none("operation", *sample_list(valid_operations))

        maximum_kernel_size = 1 if self.operation == "Operation.ELTWISE" else 11

        self.replace_none(
            "kernel_height", *sample_list(np.arange(1, maximum_kernel_size + 1))
        )
        self.replace_none(
            "kernel_width", *sample_list(np.arange(1, maximum_kernel_size + 1))
        )

        self.safe_assign("input_0_batch", 1)
        self.replace_none(
            "input_0_height",
            *sample_list(
                np.arange(
                    self.kernel_height
                    * (
                        2 if self.isi_strategy == "ISIStrategy.SPLIT_OVER_H" else 1
                    ),  # only for Layer
                    8192,
                ),
                False,
            ),
        )
        self.replace_none(
            "input_0_width", *sample_list(np.arange(self.kernel_width, 8192), False)
        )
        self.replace_none(
            "input_0_channels",
            *sample_list(valid_input_channels[self.operation], False),
        )

        if (
            self.isi_strategy == "ISIStrategy.SPLIT_OVER_H"
            and self.operation == "Operation.DW_CONVOLUTION"
            and self.kernel_height != self.kernel_width
        ):
            raise ValueError(
                "self.kernel_height != self.kernel_width not supported in SOH because of vpu2.x_generic_rules.yaml#L395-L407"
            )

        self.replace_none(
            "kernel_pad_left", *sample_list([*range(0, (self.kernel_width + 1) // 2)])
        )
        if self.kernel_pad_right is None:
            # artificial constraint only if generated
            self.safe_assign("kernel_pad_right", self.kernel_pad_left)
        else:  # purely for checking it has a valid value
            self.replace_none(
                "kernel_pad_right",
                *sample_list([*range(0, (self.kernel_width + 1) // 2)]),
            )

        self.replace_none(
            "kernel_pad_top", *sample_list([*range(0, (self.kernel_height + 1) // 2)])
        )
        if self.kernel_pad_bottom is None:
            # artificial constraint
            self.safe_assign("kernel_pad_bottom", self.kernel_pad_top)
        else:  # purely for checking it has a valid value
            self.replace_none(
                "kernel_pad_bottom",
                *sample_list([*range(0, (self.kernel_height + 1) // 2)]),
            )

        if self.operation != "Operation.ELTWISE":
            s_sizes = np.arange(1, 8)
        else:
            s_sizes = np.arange(1, 2)
        self.replace_none(
            "kernel_stride_height",
            *sample_list(
                s_sizes[s_sizes <= min(self.input_0_height, self.input_0_width)], False
            ),
        )
        # artificial constraint always
        self.safe_assign("kernel_stride_width", self.kernel_stride_height)

        self.safe_assign("output_0_batch", self.input_0_batch)

        self.safe_assign(
            "output_0_height",
            int(
                (
                    (
                        self.input_0_height
                        + (self.kernel_pad_top + self.kernel_pad_bottom)
                        - (self.kernel_height - 1)
                        - 1
                    )
                    // self.kernel_stride_height
                )
                + 1
            ),
        )
        self.safe_assign(
            "output_0_width",
            int(
                (
                    (
                        self.input_0_width
                        + (self.kernel_pad_left + self.kernel_pad_right)
                        - (self.kernel_width - 1)
                        - 1
                    )
                    // self.kernel_stride_width
                )
                + 1
            ),
        )

        def check_trailing_padding(in_dim, out_dim, stride, kernel_radix, leading_pad):
            # Adapt right and bottom padding
            # Reference: https://theano-pymc.readthedocs.io/en/latest/tutorial/conv_arithmetic.html#zero-padding-non-unit-strides
            pads = stride * (out_dim - 1) + kernel_radix - leading_pad - in_dim
            return max(pads, 0)

        self.kernel_pad_left = check_trailing_padding(
            self.input_0_width,
            self.output_0_width,
            self.kernel_stride_width,
            self.kernel_width,
            self.kernel_pad_right,
        )
        self.kernel_pad_top = check_trailing_padding(
            self.input_0_height,
            self.output_0_height,
            self.kernel_stride_height,
            self.kernel_height,
            self.kernel_pad_bottom,
        )

        if self.isi_strategy == "ISIStrategy.SPLIT_OVER_H":
            if self.output_0_height == 1:
                raise ValueError("can't do SPLIT_OVER_H if output_0_height is 1")
            if self.kernel_height > 1 and self.device == "VPUDevice.VPU_2_7":
                upper_seg = ceil(self.input_0_height / 2)
                lower_seg = self.input_0_height // 2
                se_sp_size_se_size = upper_seg * self.input_0_width // 4
                se_sp_size1_se_size = lower_seg * self.input_0_width // 4
                if se_sp_size_se_size == 1 and se_sp_size1_se_size == 1:
                    raise ValueError(
                        'as per HAS: "SE_SEG_SIZE must be configured to be a multiple of 4"'
                    )
                if (
                    se_sp_size_se_size == 3
                    and se_sp_size1_se_size == 3
                    and self.kernel_pad_bottom == 1
                    and self.kernel_pad_top == 0
                ):
                    raise ValueError("possible bug")

        self.replace_none("input_0_datatype", valid_datatypes[0], valid_datatypes)
        self.replace_none("input_0_layout", valid_layouts[0], valid_layouts)
        self.replace_none("input_0_swizzling", valid_swizzlings[0], valid_swizzlings)
        if self.operation in [
            "Operation.CONVOLUTION",
            "Operation.DW_CONVOLUTION",
            "Operation.CM_CONVOLUTION",
        ]:
            self.replace_none("output_0_datatype", *sample_list(valid_datatypes))
        else:
            self.safe_assign("output_0_datatype", self.input_0_datatype)
        self.replace_none("output_0_layout", valid_layouts[0], valid_layouts)
        self.replace_none("output_0_swizzling", valid_swizzlings[0], valid_swizzlings)

        if self.operation in [
            "Operation.CONVOLUTION",
            "Operation.DW_CONVOLUTION",
            "Operation.CM_CONVOLUTION",
        ]:
            # TODO: add support for act fp16 and weight i8
            self.safe_assign("input_1_datatype", self.input_0_datatype)
            self.replace_none(
                "input_1_swizzling", valid_swizzlings[0], valid_swizzlings
            )
            self.safe_assign("input_1_layout", self.input_0_layout)

            if self.operation == "Operation.CONVOLUTION":
                self.replace_none(
                    "output_0_channels",
                    *sample_list(np.arange(1, 512)[startidx:] * 16, False),
                )
                self.safe_assign(
                    "input_1_channels",
                    self.input_0_channels * self.kernel_width * self.kernel_height,
                )
            elif self.operation == "Operation.DW_CONVOLUTION":
                self.safe_assign("output_0_channels", self.input_0_channels)
                self.safe_assign(
                    "input_1_channels",
                    self.align_to(self.kernel_width * self.kernel_height, 16),
                )
            elif self.operation == "Operation.CM_CONVOLUTION":
                # if input_0_channels==1 Fathom converts it to DW_CONVOLUTION
                self.replace_none(
                    "output_0_channels",
                    *sample_list(np.arange(1, 512)[startidx:] * 16, False),
                )
                self.safe_assign(
                    "input_1_channels",
                    self.align_to(
                        self.input_0_channels * self.kernel_width * self.kernel_height,
                        16 if self.input_1_datatype in quantized_datatypes else 8,
                    ),
                )

            self.safe_assign("input_1_batch", self.output_0_channels)
            self.safe_assign("input_1_height", 1)
            self.safe_assign("input_1_width", 1)

            if (
                self.isi_strategy == "ISIStrategy.SPLIT_OVER_K"
                and self.output_0_channels % 32 != 0
            ):
                self.weight_sparsity_enabled = False
                self.weight_sparsity_rate = 0.0

            if self.operation == "Operation.CONVOLUTION":
                if not sample:
                    if self.weight_sparsity_enabled is None:
                        if self.weight_sparsity_rate is not None:
                            if self.weight_sparsity_rate > 0.0:
                                self.weight_sparsity_enabled = True
                            else:
                                self.weight_sparsity_enabled = False
                        else:
                            self.weight_sparsity_enabled = 0.0
                            self.weight_sparsity_rate = 0.0
                    if self.weight_sparsity_rate is None:
                        self.weight_sparsity_rate = 0.0
                    if not self.weight_sparsity_enabled:
                        assert (
                            self.weight_sparsity_rate == 0.0
                        ), "Weight sparsity disabled, can't set sparsity rate"
                    self.weight_sparsity_rate = self.sanitize_sparsity(
                        self.input_1_size, self.weight_sparsity_rate
                    )
                else:
                    if (
                        self.weight_sparsity_rate is not None
                        and self.weight_sparsity_rate > 0.0
                    ):
                        self.safe_assign("weight_sparsity_enabled", True)
                    else:
                        self.replace_none(
                            "weight_sparsity_enabled", *sample_list([True, False])
                        )
                    if self.weight_sparsity_enabled:
                        if self.weight_sparsity_rate is None:
                            self.weight_sparsity_rate = self.sanitize_sparsity(
                                self.input_1_size, np.random.uniform(low=0.0, high=1.0)
                            )
                        else:
                            self.weight_sparsity_rate = self.sanitize_sparsity(
                                self.input_1_size, self.weight_sparsity_rate
                            )
                    else:
                        self.safe_assign("weight_sparsity_rate", 0.0)
            else:
                self.safe_assign("weight_sparsity_enabled", False)
                self.safe_assign("weight_sparsity_rate", 0.0)
                self.safe_assign("input_sparsity_enabled", False)
                self.safe_assign("input_sparsity_rate", 0.0)

        elif self.operation == "Operation.ELTWISE":
            self.safe_assign("input_1_datatype", self.input_0_datatype)
            self.replace_none(
                "input_1_swizzling", valid_swizzlings[0], valid_swizzlings
            )
            self.safe_assign("input_1_layout", self.input_0_layout)

            self.safe_assign("output_0_channels", self.input_0_channels)

            self.safe_assign("input_1_batch", self.input_0_batch)
            self.safe_assign("input_1_channels", self.input_0_width)
            self.safe_assign("input_1_height", self.input_0_channels)
            self.safe_assign("input_1_width", self.input_0_height)

            self.safe_assign("weight_sparsity_enabled", False)
            self.safe_assign("weight_sparsity_rate", 0.0)
            self.safe_assign("input_sparsity_enabled", False)
            self.safe_assign("input_sparsity_rate", 0.0)

        elif self.operation == "Operation.MAXPOOL":
            self.safe_assign("input_1_datatype", self.input_0_datatype)
            self.replace_none(
                "input_1_swizzling", valid_swizzlings[0], valid_swizzlings
            )
            self.safe_assign("input_1_layout", "Layout.INVALID")

            self.safe_assign("output_0_channels", self.input_0_channels)

            self.safe_assign("input_1_batch", 0)
            self.safe_assign("input_1_channels", 0)
            self.safe_assign("input_1_height", 0)
            self.safe_assign("input_1_width", 0)

            self.safe_assign("input_sparsity_enabled", False)
            self.safe_assign("input_sparsity_rate", 0.0)
            self.safe_assign("weight_sparsity_enabled", False)
            self.safe_assign("weight_sparsity_rate", 0.0)

        self.replace_none("execution_order", *sample_list(valid_execution_order))
        self.replace_none(
            "activation_function", "ActivationFunction.NONE", valid_activation_function
        )

        if self.input_sparsity_rate is None:
            self.input_sparsity_rate = 0.0
        else:
            self.input_sparsity_rate = self.sanitize_sparsity(
                self.input_0_size, self.input_sparsity_rate
            )
        if self.input_sparsity_enabled is None:
            self.input_sparsity_enabled = self.input_sparsity_rate != 0.0
        if self.input_sparsity_enabled:
            assert self.input_sparsity_rate != None
        if not self.input_sparsity_enabled:
            assert (
                self.input_sparsity_rate == 0.0
            ), "Input sparsity disabled but a rate bigger than 0.0 was provided"

        assert (
            self.input_sparsity_rate >= 0.0 and self.input_sparsity_rate <= 1.0
        ), "input sparsity rate must be in [0.0, 1.0]"
        assert (
            self.weight_sparsity_rate >= 0.0 and self.weight_sparsity_rate <= 1.0
        ), "weight sparsity rate must be in [0.0, 1.0]"

        self.safe_assign("output_sparsity_enabled", False)

        def restrict_datatype(current):
            if current in quantized_datatypes:
                return "DataType.UINT8"
            else:
                return "DataType.FLOAT16"

        self.input_0_datatype = restrict_datatype(self.input_0_datatype)
        self.input_1_datatype = restrict_datatype(self.input_1_datatype)
        self.output_0_datatype = restrict_datatype(self.output_0_datatype)

    @property
    def input_0_size(self):
        if self.operation == "Operation.CM_CONVOLUTION":
            num_elements = (
                self.input_0_height
                * self.input_0_width
                * (4 if self.input_0_channels < 5 else 16)
            )
        else:
            num_elements = (
                self.input_0_height * self.input_0_width * self.input_0_channels
            )

        return num_elements

    @property
    def input_0_size_aligned(self):
        input_0_quantized = self.input_0_datatype in ["DataType.UINT8", "DataType.INT8"]
        # align to 16KB chunks
        return self.align_to(self.input_0_size * (1 if input_0_quantized else 2), 16384)

    @property
    def input_1_size(self):
        if self.operation in [
            "Operation.CONVOLUTION",
            "Operation.DW_CONVOLUTION",
            "Operation.CM_CONVOLUTION",
        ]:
            num_elements = (
                self.input_1_height
                * self.input_1_width
                * self.input_1_channels
                * self.input_1_batch
            )
        elif self.operation in ["Operation.ELTWISE"]:
            num_elements = (
                self.input_1_height * self.input_1_width * self.input_1_channels
            )
        elif self.operation == "Operation.MAXPOOL":
            num_elements = 0
        return num_elements

    @property
    def input_1_size_aligned(self):
        # in this function we also take into account sparsity map and weight table (if any)
        input_1_quantized = self.input_1_datatype in ["DataType.UINT8", "DataType.INT8"]
        input_1_size_total = self.input_1_size * (1 if input_1_quantized else 2)
        if self.weight_sparsity_enabled:
            # add sparsity map size
            input_1_size_total = round(
                input_1_size_total
                - (input_1_size_total * self.weight_sparsity_rate)
            )
            input_1_size_total += self.output_0_channels * self.align_to(
                self.input_0_channels // 8 * self.kernel_height * self.kernel_width, 16
            )
        if self.operation in [
            "Operation.CONVOLUTION",
            "Operation.DW_CONVOLUTION",
            "Operation.CM_CONVOLUTION",
            "Operation.MAXPOOL",
        ]:
            # add weight table size
            input_1_size_total += (
                self.output_0_channels * 16
            )  # 16 bytes for each output channel
        return self.align_to(input_1_size_total, 16384)  # align to 16KB chunks

    @property
    def weight_table_size(self):
        return self.output_0_batch * 16

    @property
    def output_0_size(self):
        return self.output_0_height * self.output_0_width * self.output_0_channels

    @property
    def output_0_size_aligned(self):
        output_0_quantized = self.output_0_datatype in [
            "DataType.UINT8",
            "DataType.INT8",
        ]
        # align to 16KB chunks
        return self.align_to(
            self.output_0_size * (1 if output_0_quantized else 2), 16384
        )

    def __str__(self):
        to_return = f"{self.__class__.__name__}("
        for attr in self.__dict__:
            if attr in [
                "input_1_batch",
                "input_1_channels",
                "input_1_height",
                "input_1_width",
                "output_0_batch",
                "output_0_height",
                "output_0_width",
                "cycles",
            ]:
                continue
            if "_size" in attr:
                continue
            if self.operation == "Operation.ELTWISE" and attr == "input_1_swizzling":
                continue
            if self.operation == "Operation.MAXPOOL" and attr in [
                "input_1_datatype",
                "input_1_layout",
                "input_1_swizzling",
            ]:
                continue
            if (
                self.operation
                not in ["Operation.CONVOLUTION", "Operation.CM_CONVOLUTION"]
                and attr == "output_0_channels"
            ):
                continue
            if attr.startswith("__"):
                pass
            else:
                to_return += f"{attr}={repr(getattr(self, attr))}, "
        return to_return[:-2] + ")"

    def nn_cmx_usage(self):
        return sum(
            [
                (
                    self.input_0_size_aligned
                    // (2 if self.isi_strategy == "ISIStrategy.SPLIT_OVER_H" else 1)
                ),
                (
                    self.input_1_size_aligned
                    // (2 if self.isi_strategy == "ISIStrategy.SPLIT_OVER_K" else 1)
                ),
                (
                    self.output_0_size_aligned
                    if self.operation != "Operation.ELTWISE"
                    else 0
                ),
                (80 * 1024),  # runtime size
                (16 * 1024),  # hardware profile block (hwp)
            ]
        )

    def to_dict(self):
        return asdict(self)

    def md5hash(self):
        return hashlib.md5(
            json.dumps(asdict(self), sort_keys=True).encode("utf-8")
        ).hexdigest()


@dataclass(eq=True, unsafe_hash=True)
class DPUWorkload(DPULayer):
    """
    DPUWorkload class models a concrete DPU workload, all the fields must be filled and it needs to fit in the CMX memory
    """

    def __post_init__(self):
        super().__post_init__()

        assert (
            self.execution_order != None
        ), "A valid instance of a workload should have execution_order set"

        assert (
            self.input_0_layout != None
        ), "A valid instance of a workload should have layout set"
        assert (
            self.input_0_swizzling != None
        ), "A valid instance of a workload should have swizzling set"
        assert (
            self.input_1_layout != None
        ), "A valid instance of a workload should have layout set"
        assert (
            self.input_1_swizzling != None
        ), "A valid instance of a workload should have swizzling set"
        assert (
            self.output_0_layout != None
        ), "A valid instance of a workload should have layout set"
        assert (
            self.output_0_swizzling != None
        ), "A valid instance of a workload should have swizzling set"

        assert (
            self.isi_strategy != None
        ), "A valid instance of a workload should have an isi strategy set"

        if self.nn_cmx_usage() > cmx_sizes[self.device] * 1024 * 1024 * safety_factor:
            raise ValueError(
                f"workload doesn't fit in cmx ({self.nn_cmx_usage()}/{cmx_sizes[self.device]*1024*1024})"
            )


def md5_workload_list(workloads: List[DPUWorkload]):
    return hashlib.md5(
        json.dumps([wl.md5hash() for wl in workloads]).encode("utf-8")
    ).hexdigest()


def get_nn_cmx_peak(workloads: List[DPUWorkload]):  # Heuristic

    peak = 0
    for wl in workloads:
        size = wl.nn_cmx_usage
        peak = max(size, peak)
    return peak


def get_ddr_heap_size(workloads: List[DPUWorkload]):
    size = 0
    for wl in workloads:
        if wl.operation == "Operation.ELTWISE":
            size += wl.input_0_size_aligned + wl.input_1_size_aligned
        else:
            size += wl.input_0_size_aligned
    return size


@dataclass(eq=True, unsafe_hash=True)
class DMAWorkload:
    def __init__(self):
        pass
