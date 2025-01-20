# Copyright © 2024 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
# LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
# is subject to the terms and conditions of the software license agreements for the Software Package,
# which may also include notices, disclaimers, or license terms for third party or open source software
# included in or with the Software Package, and your use indicates your acceptance of all such terms.
# Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
# Software Package for additional details.


from skbuild import setup
import subprocess
import shutil
import os


def get_version():

    this_dir = os.path.dirname(os.path.realpath(__file__))

    git_describe = subprocess.check_output(
        ["git", "describe", "--tags"], cwd=this_dir
    ).decode("utf-8")

    sections = git_describe.split("-")
    version = sections[0]

    return version


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt") as fh:
    requirements = fh.readlines()

build_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "_skbuild")
if os.path.isdir(build_dir):
    print(f"Clean {build_dir}")
    shutil.rmtree(build_dir)

setup(
    name="vpunn_cost_model",
    version=get_version(),
    author="Alessandro Palla",
    description="VPUNN cost model",
    license="Apache License 2.0",
    cmake_install_target="vpunn-install-bindings",
    cmake_args=[
        "-DVPUNN_BUILD_EXAMPLES=OFF",
        "-DVPUNN_BUILD_TESTS=OFF",
        "-DVPUNN_BUILD_SHARED_LIB=OFF",
    ],
    long_description=long_description,
    long_description_content_type="text/markdown",
 
    project_urls={
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    package_dir={"vpunn": "python"},
    packages=["vpunn"],
    entry_points={
        "console_scripts": [
            "vpunn_to_json=vpunn.to_json:main",
            "vpunn_builder=vpunn.builder:main",
            "vpu_cost_model=vpunn.cost:main",
            "vpu_layer_cost_model=vpunn.layer:main",
        ],
    },
    python_requires=">=3.6",
    install_requires=requirements,
)
