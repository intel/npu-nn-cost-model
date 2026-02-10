# Copyright © 2024 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
# LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
# is subject to the terms and conditions of the software license agreements for the Software Package,
# which may also include notices, disclaimers, or license terms for third party or open source software
# included in or with the Software Package, and your use indicates your acceptance of all such terms.
# Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
# Software Package for additional details.

import subprocess
import shutil
import os
from setuptools import setup

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

setup(
    name="vpunn_cost_model",
    version=get_version(),
    description="VPUNN cost model",
    license="Apache License 2.0",
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
    python_requires=">=3.6",
    install_requires=requirements,
)
