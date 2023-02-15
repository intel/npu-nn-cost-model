#! /usr/bin/env python3
# Copyright © 2022 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
# LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
# is subject to the terms and conditions of the software license agreements for the Software Package,
# which may also include notices, disclaimers, or license terms for third party or open source software
# included in or with the Software Package, and your use indicates your acceptance of all such terms.
# Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
# Software Package for additional details.

import argparse
from shutil import which
import os
import subprocess
import sys
import ntpath


this_file_location = os.path.dirname(os.path.abspath(__file__))
schema = os.path.join(this_file_location, "schema/vpunn.fbs ")
exec = "flatc "


def cmd_exists(name):
    # Source: https://stackoverflow.com/questions/11210104/check-if-a-program-exists-from-a-python-script
    """Check whether `name` is on PATH and marked as executable."""
    return which(name) is not None


def define_and_parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "file",
        type=str,
        help="graphFile to deserialize to json OR an already deserialized json",
    )

    return parser.parse_args()


def blob_to_json(file, args):

    if not cmd_exists("flatc"):
        print("flatc command not found in system path (or not marked executable).")
        print("Your FlatBuffer setup may not be fully complete")
        print("Exiting..")
        sys.exit(2)

    output_location = os.getcwd()
    out_flag = f" -o {output_location}"
    json_arguments = "--defaults-json --strict-json --force-defaults"
    json_flag = "--json"

    cmd = f"{exec} {out_flag} {schema} {json_flag} -- {file} {json_arguments}"

    subprocess.call(cmd, shell=True)
    return os.path.join(output_location, ntpath.basename(file).replace("vpunn", "json"))


def json_to_blob(file, args):

    if not cmd_exists("flatc"):
        print("flatc command not found in system path (or not marked executable).")
        print("Your FlatBuffer setup may not be fully complete")
        print("Exiting..")
        sys.exit(2)

    output_location = os.getcwd()
    out_flag = " -o " + output_location
    binary_flag = " --binary "

    cmd = f"{exec} {out_flag} {schema} {binary_flag} {file}"

    subprocess.call(cmd, shell=True)
    return os.path.join(output_location, ntpath.basename(file).replace("json", "blob"))


def command(args):
    f = args.file
    f = os.path.expanduser(f)
    f = os.path.abspath(f)
    ext = os.path.splitext(f)[1]

    if ext == ".vpunn":
        f = blob_to_json(f, args)

    elif ext == ".json":
        f = json_to_blob(f, args)

    else:
        print("Unsupported input format")
        sys.exit(1)

    print(f"File written to {f}")


def main():
    args = define_and_parse_args()
    command(args)


if __name__ == "__main__":
    main()
