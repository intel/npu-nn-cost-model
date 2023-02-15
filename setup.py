# Copyright © 2022 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
# LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
# is subject to the terms and conditions of the software license agreements for the Software Package,
# which may also include notices, disclaimers, or license terms for third party or open source software
# included in or with the Software Package, and your use indicates your acceptance of all such terms.
# Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
# Software Package for additional details.


from distutils.command.install_data import install_data
from distutils.dir_util import copy_tree
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from setuptools.command.install_lib import install_lib
from setuptools.command.install_scripts import install_scripts
import subprocess
import pathlib
import shutil
import struct
import os

BITS = struct.calcsize("P") * 8
PACKAGE_NAME = "python"
SOURCE_DIR = "."


def get_version():

    fathom_root = os.environ.get("FATHOM_ROOT")

    if fathom_root is None:
        this_dir = os.path.dirname(os.path.realpath(__file__))
    else:
        this_dir = os.path.join(fathom_root, "lib", "nn_cost_model")

    git_describe = subprocess.check_output(
        ["git", "describe", "--tags"], cwd=this_dir
    ).decode("utf-8")

    sections = git_describe.split("-")
    version = sections[0]

    return version


class CMakeExtension(Extension):
    """
    An extension to run the cmake build

    This simply overrides the base extension class so that setuptools
    doesn't try to build your sources for you
    """

    def __init__(self, name, sources=[]):

        super().__init__(name=name, sources=sources)


class InstallCMakeLibsData(install_data):
    """
    Just a wrapper to get the install data into the egg-info

    Listing the installed files in the egg-info guarantees that
    all of the package files will be uninstalled when the user
    uninstalls your package through pip
    """

    def run(self):
        """
        Outfiles are the libraries that were built using cmake
        """

        # There seems to be no other way to do this; I tried listing the
        # libraries during the execution of the InstallCMakeLibs.run() but
        # setuptools never tracked them, seems like setuptools wants to
        # track the libraries through package data more than anything...
        # help would be appriciated

        self.outfiles = self.distribution.data_files


class InstallCMakeLibs(install_lib):
    """
    Get the libraries from the parent distribution, use those as the outfiles

    Skip building anything; everything is already built, forward libraries to
    the installation step
    """

    def add(self, src, dest):
        source = os.path.join(self.distribution.bin_dir, src)
        destination = os.path.join(self.build_dir, "vpunn", dest)
        self.announce(f"Move {source} to {destination}", level=3)
        if os.path.isdir(source):
            copy_tree(source, destination)
        else:
            shutil.copy(source, destination)
        self.distribution.data_files.append(destination)

    def run(self):
        """
        Copy libraries from the bin directory and place them as appropriate
        """

        self.announce("Moving library files", level=3)

        # We have already built the libraries in the previous build_ext step

        self.skip_build = True

        bin_dir = self.distribution.bin_dir

        # Depending on the files that are generated from your cmake
        # build chain, you may need to change the below code, such that
        # your files are moved to the appropriate location when the installation
        # is run
        self.distribution.data_files = []

        os.makedirs(os.path.join(self.build_dir, "vpunn/lib"), exist_ok=True)
        os.makedirs(os.path.join(self.build_dir, "VPUNN_SCHEMA"), exist_ok=True)
        os.makedirs(os.path.join(self.build_dir, "vpunn/include"), exist_ok=True)

        self.announce(f"Binary dir is {bin_dir}", level=3)

        self.add("lib/libinference.so", "lib/libinference.so")
        self.add("lib/libvpunn_optimization.so", "lib/libvpunn_optimization.so")
        self.add("include/VPUNN_SCHEMA", "../VPUNN_SCHEMA")
        self.add("_deps/flatbuffers-src/include", "include")
        self.add("include/vpunn_generated.h", "include/vpunn_generated.h")
        self.add("include/vpu", "include/vpu")
        self.add("../../include", "include")
        self.add("../../models", "models")
        self.add("../../src/schema", "schema")

        for ff in self.distribution.data_files:
            self.announce(f"Installed library in {ff}", level=3)

        # Must be forced to run after adding the libs to data_files
        self.distribution.run_command("install_data")

        super().run()


class InstallCMakeScripts(install_scripts):
    """
    Install the scripts in the build dir
    """

    def run(self):
        """
        Copy the required directory to the build directory and super().run()
        """

        self.announce("Moving scripts files", level=3)

        # Scripts were already built in a previous step

        self.skip_build = True

        bin_dir = os.path.join(self.distribution.bin_dir, "include")

        scripts_dirs = [
            os.path.join(bin_dir, _dir)
            for _dir in os.listdir(bin_dir)
            if os.path.isdir(os.path.join(bin_dir, _dir))
        ]

        for scripts_dir in scripts_dirs:
            dest_folder = os.path.join(self.build_dir, os.path.basename(scripts_dir))
            self.announce(f"Moving {scripts_dir} to {dest_folder}", level=3)
            shutil.move(scripts_dir, dest_folder)

        # Mark the scripts for installation, adding them to
        # distribution.scripts seems to ensure that the setuptools' record
        # writer appends them to installed-files.txt in the package's egg-info

        self.distribution.scripts = scripts_dirs

        super().run()


class BuildCMakeExt(build_ext):
    """
    Builds using cmake instead of the python setuptools implicit build
    """

    def run(self):
        """
        Perform build_cmake before doing the 'normal' stuff
        """

        for extension in self.extensions:

            if extension.name == "libinference":

                self.build_cmake(extension)

    def build_cmake(self, extension: Extension):
        """
        The steps required to build the extension
        """

        self.announce("Preparing the build environment", level=3)

        build_dir = pathlib.Path(self.build_temp)

        extension_path = pathlib.Path(self.get_ext_fullpath(extension.name))

        os.makedirs(build_dir, exist_ok=True)
        os.makedirs(extension_path.parent.absolute(), exist_ok=True)

        # Now that the necessary directories are created, build

        self.announce("Configuring cmake project", level=3)

        # Change your cmake arguments below as necessary
        # Below is just an example set of arguments for building the NN Cost Model as a Python module

        cc = os.environ.get("CC")
        cxx = os.environ.get("CXX")

        if cc is None:
            default_cc = os.popen("which gcc").read().strip()
            self.announce(
                "Warning: $CC is not set, defaulting to {}".format(default_cc), level=3
            )
            cc = default_cc

        if cxx is None:
            default_cxx = os.popen("which g++").read().strip()
            self.announce(
                "Warning: $CXX is not set, defaulting to {}".format(default_cxx),
                level=3,
            )
            cxx = default_cxx

        self.spawn(
            [
                "cmake",
                "-H" + SOURCE_DIR,
                "-B" + self.build_temp,
                "-DCMAKE_C_COMPILER=" + cc,
                "-DCMAKE_CXX_COMPILER=" + cxx,
            ]
        )

        self.announce("Building binaries", level=3)

        self.spawn(["cmake", "--build", self.build_temp, "--", "python", "-j"])

        # Build finished, now copy the files into the copy directory
        # The copy directory is the parent directory of the extension (.pyd)

        self.announce("Moving built python module", level=3)

        self.distribution.bin_dir = build_dir


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt") as fh:
    requirements = fh.readlines()

setup(
    name="vpunn_cost_model",
    version=get_version(),
    author="Alessandro Palla",
    author_email="alessandro.palla@intel.com",
    description="VPUNN cost model",
    license="Apache License 2.0",
    ext_modules=[CMakeExtension(name="libinference")],
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/intel-sandbox/libraries.performance.modeling.vpu.nn_cost_model",
    project_urls={
        "Bug Tracker": "https://github.com/intel-sandbox/libraries.performance.modeling.vpu.nn_cost_model/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    package_dir={"vpunn": "python"},
    # packages=setuptools.find_packages(),
    packages=["vpunn"],
    package_data={"vpunn": ["include/vpunn.h"]},
    entry_points={
        "console_scripts": [
            "vpunn_to_json=vpunn.to_json:main",
            "vpunn_builder=vpunn.builder:main",
            "vpu_cost_model=vpunn.cost:main",
            "vpu_layer_cost_model=vpunn.layer:main",
        ],
    },
    cmdclass={
        "build_ext": BuildCMakeExt,
        "install_data": InstallCMakeLibsData,
        "install_lib": InstallCMakeLibs,
        # 'install_scripts': InstallCMakeScripts
    },
    python_requires=">=3.6",
    install_requires=requirements,
)
