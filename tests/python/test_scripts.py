# Copyright © 2023 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
# LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
# is subject to the terms and conditions of the software license agreements for the Software Package,
# which may also include notices, disclaimers, or license terms for third party or open source software
# included in or with the Software Package, and your use indicates your acceptance of all such terms.
# Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
# Software Package for additional details.

import subprocess
import pytest
import vpunn
import os


def run(command, path="."):
    print(command)
    return subprocess.check_output(command, cwd=path, shell=True)


@pytest.mark.parametrize("mode", ["DPU"])
@pytest.mark.parametrize("target", ["cycles", "power", "utilization"])
@pytest.mark.parametrize("device", ["VPU_2_0"])
@pytest.mark.parametrize("operation", ["CONVOLUTION"])
@pytest.mark.parametrize("width", [56])
@pytest.mark.parametrize("height", [56])
@pytest.mark.parametrize("input_channels", [64])
@pytest.mark.parametrize("output_channels", [64])
@pytest.mark.parametrize("kernel", [1])
@pytest.mark.parametrize("padding", [0])
@pytest.mark.parametrize("strides", [1])
@pytest.mark.parametrize("input_dtype", ['UINT8', 'INT8', 'FLOAT16', 'BFLOAT16'])
@pytest.mark.parametrize("output_dtype", ['UINT8', 'INT8', 'FLOAT16', 'BFLOAT16'])
@pytest.mark.parametrize("execution_order", ['VECTOR_FP16', 'VECTOR', 'MATRIX',])
def test_vpu_cost_model_2_0(
    mode,
    target,
    device,
    operation,
    width,
    height,
    input_channels,
    output_channels,
    kernel,
    padding,
    strides,
    input_dtype,
    output_dtype,
    execution_order,
):

    cmd = (
        f"vpu_cost_model "
        f"--target {target} "
        f"--device {device} "
        f"{mode} "
        f"--operation {operation} "
        f"--width {width} "
        f"--height {height} "
        f"--input-channels {input_channels} "
        f"--output-channels {output_channels} "
        f"--kw {kernel} "
        f"--kh {kernel} "
        f"--pad-bottom {padding} "
        f"--pad-left {padding} "
        f"--pad-right {padding} "
        f"--pad-top {padding} "
        f"--stride-height {strides} "
        f"--stride-width {strides} "
        f"--input-datatype {input_dtype} "
        f"--output-datatype {output_dtype} "
        f"--execution-order {execution_order}"
    )
    run(cmd)

@pytest.mark.parametrize("mode", ["DPU"])
@pytest.mark.parametrize("target", ["cycles", "power", "utilization"])
@pytest.mark.parametrize("device", ["VPU_2_7"])
@pytest.mark.parametrize("operation", ["CONVOLUTION"])
@pytest.mark.parametrize("width", [56])
@pytest.mark.parametrize("height", [56])
@pytest.mark.parametrize("input_channels", [64])
@pytest.mark.parametrize("output_channels", [64])
@pytest.mark.parametrize("kernel", [1])
@pytest.mark.parametrize("padding", [0])
@pytest.mark.parametrize("strides", [1])
@pytest.mark.parametrize("input_dtype", ['UINT8', 'INT8', 'FLOAT16', 'BFLOAT16'])
@pytest.mark.parametrize("output_dtype", ['UINT8', 'INT8', 'FLOAT16', 'BFLOAT16'])
@pytest.mark.parametrize("execution_order", ['CUBOID_4x16', 'CUBOID_8x16', 'CUBOID_16x16'])
def test_vpu_cost_model_2_7(
    mode,
    target,
    device,
    operation,
    width,
    height,
    input_channels,
    output_channels,
    kernel,
    padding,
    strides,
    input_dtype,
    output_dtype,
    execution_order,
):

    cmd = (
        f"vpu_cost_model "
        f"--target {target} "
        f"--device {device} "
        f"{mode} "
        f"--operation {operation} "
        f"--width {width} "
        f"--height {height} "
        f"--input-channels {input_channels} "
        f"--output-channels {output_channels} "
        f"--kw {kernel} "
        f"--kh {kernel} "
        f"--pad-bottom {padding} "
        f"--pad-left {padding} "
        f"--pad-right {padding} "
        f"--pad-top {padding} "
        f"--stride-height {strides} "
        f"--stride-width {strides} "
        f"--input-datatype {input_dtype} "
        f"--output-datatype {output_dtype} "
        f"--execution-order {execution_order}"
    )
    run(cmd)


@pytest.mark.parametrize("nDPU", [1, 5])
@pytest.mark.parametrize("nTiles", [1, 2, 4])
@pytest.mark.parametrize("device", ["VPU_2_0", "VPU_2_7"])
@pytest.mark.parametrize("operation", ["CONVOLUTION"])
@pytest.mark.parametrize("width", [56])
@pytest.mark.parametrize("height", [56])
@pytest.mark.parametrize("input_channels", [64])
@pytest.mark.parametrize("output_channels", [64])
@pytest.mark.parametrize("kernel", [1])
@pytest.mark.parametrize("padding", [0])
@pytest.mark.parametrize("strides", [1])
@pytest.mark.parametrize("input_dtype", ["UINT8"])
@pytest.mark.parametrize("output_dtype", ["UINT8"])
@pytest.mark.parametrize("input_ddr", [True, False])
@pytest.mark.parametrize("output_ddr", [True, False])
@pytest.mark.parametrize("prefetch", [True, False])
def test_vpu_layer_cost_model(
    nDPU,
    nTiles,
    device,
    operation,
    width,
    height,
    input_channels,
    output_channels,
    kernel,
    padding,
    strides,
    input_dtype,
    output_dtype,
    input_ddr,
    output_ddr,
    prefetch,
):

    if device == "VPU_2_7" and (nDPU == 5 or nTiles == 4):
        # Invalid configurations
        return

    cmd = (
        f"vpu_layer_cost_model --nDPU {nDPU} "
        f"--nTiles {nTiles} "
        f"--device {device} "
        f"--operation {operation} "
        f"--width {width} "
        f"--height {height} "
        f"--input_channels {input_channels} "
        f"--output_channels {output_channels} "
        f"--kernel {kernel} "
        f"--padding {padding} "
        f"--strides {strides} "
        f"--input_dtype {input_dtype} "
        f"--output_dtype {output_dtype} "
    )

    if input_ddr:
        cmd += " --input-ddr "
    if output_ddr:
        cmd += " --output-ddr "
    if prefetch:
        cmd += " --prefetch "

    run(cmd)
