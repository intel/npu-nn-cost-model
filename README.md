# VPUNN cost model

A NN-Based Cost Model for VPU Devices. For additional information about model setup and training, please refer [this paper](https://arxiv.org/abs/2205.04586)

If you find this work useful, please cite the following paper:

```
@article{DBLP:journals/corr/abs-2205-04586,
  doi = {10.48550/ARXIV.2205.04586},
  url = {https://arxiv.org/abs/2205.04586},
  author = {Hunter, Ian Frederick Vigogne Goodbody and Palla, Alessandro and Nagy, Sebastian Eusebiu and Richmond, Richard and McAdoo, Kyle},
  title = {Towards Optimal VPU Compiler Cost Modeling by using Neural Networks to Infer Hardware Performances},
  publisher = {arXiv},
  year = {2022},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```

## Setup

GCC version should be > 9. You can check your GCC version by running `gcc --version` and `g++ --version`

If you do not set CC and CXX environment variables, `which gcc` and `which g++` are used by default.

Compile the library by typing `cmake -H. -Bbuild && cmake --build build`

@TODO: environment compatible with newer compiler versions (gcc>=10, clang >10 )  

### Use Intel oneAPI MKL

Install oneAPI base Toolkit ([instructions](https://software.intel.com/content/www/us/en/develop/tools/oneapi/base-toolkit/download.html)). oneAPI is massive so feel free to install only the Math Kernel Library library.

If you have troubles with proxy, please export `no_proxy=127.0.0.1` in order to bypass any no_proxy env vs `*.intel.com` urls

To enable MKL you need to source this file `/opt/intel/oneapi/setvars.sh` to set the appropriate environment variables. Look [here](https://software.intel.com/content/www/us/en/develop/documentation/get-started-with-intel-oneapi-base-linux/top/run-a-sample-project-with-vscode.html) on how to get started with VSC

### Select BLAS library

You can select which BLAS library to use (assume you have MKL installed) and the threading mode by using the following cmake variables

- `-DCBLAS_LIB=<value>` (options: `mkl` for oneMKL and `openblas` for OpenBLAS)
- `-DMKL_THREADING=<value>` (options: `tbb` for oneAPI Threading Building Blocks and `sequential` for no threading)

## Using the cost model: C++

To use the VPUN cost model in a cmake project is quite simple. An example of a CMakeLists.txt file is shown below

```cmake
include_directories(${CMAKE_BINARY_DIR}/include)
include_directories(${FLATBUFFERS_SRC_DIR}/include)

...

target_link_libraries(<your exe or lib> inference)
```

The following example code explains how to instantiate the cost model and how to run a simple query for a 3x3s1 convolution

```c++
#include "vpu_cost_model.h"

auto model = VPUNN::VPUCostModel(model_path);

auto dpu_cycles = model.DPU({VPUNN::VPUDevice::VPU_2_7,
                             VPUNN::Operation::CONVOLUTION,
                             {VPUNN::VPUTensor(56, 56, 16, 1, VPUNN::DataType::UINT8)}, // input dimensions
                             {VPUNN::VPUTensor(56, 56, 16, 1, VPUNN::DataType::UINT8)}, // output dimensions
                             {3, 3}, //kernels
                             {1, 1}, //strides
                             {1, 1}, //padding
                             VPUNN::ExecutionMode::CUBOID_16x16} // execution mode
                            );
```

The `example` folder contains few examples on how to build and use the cost model in a C++ project. The following list is a WIP of the supported example:

- `workload_mode_selection`:
  - Selecting the optimal MPE mode for a VPU_2_0 workload
  - Choosing the optimal workload split strategy amound multiple ones

## Using the cost model: Python

You can install the library by typing `pip install .`

Do this in a python virtual environment.

### Cost models

Run the `vpu_cost_model` script to evaluate workloads from the command line

```bash
usage: vpu_cost_model [-h] --model MODEL [-t {cycles,power,utilization}] {VPU_2_7,VPU_4_0} ...

VPU cost model

positional arguments:
  {VPU_2_7,VPU_4_0}

options:
  -h, --help            show this help message and exit
  --model MODEL, -m MODEL
                        Model path
```

there are two possible VPU versions, each version has a DPU and DMA model. It is possible to bring up the help menu in the following ways:

```
vpu_cost_model VPU_2_7 DPU -h
vpu_cost_model VPU_2_7 DMA -h
vpu_cost_model VPU_4_0 DPU -h
vpu_cost_model VPU_4_0 DMA -h
```

minimal example usage:
```
vpu_cost_model VPU_2_7 DPU -o CONVOLUTION --inch 64 --outch 64 --height 16 --width 16 --kh 3 --kw 3 --indt UINT8 --outdt UINT8 --mpe-mode CUBOID_16x16
vpu_cost_model VPU_2_7 DMA -l 1024 --sw 1024 --dw 1024 -d DDR2CMX
vpu_cost_model VPU_4_0 DPU -o CONVOLUTION --inch 64 --outch 64 --height 16 --width 16 --kh 3 --kw 3 --indt UINT8 --outdt UINT8 --mpe-mode CUBOID_16x16
vpu_cost_model VPU_4_0 DMA 1024 --sw 1024 --dw 1024 -d DDR2CMX
```

#### VPUNN builder

Generate a VPUNN model from a tensorflow one

```bash
optional arguments:
  -h, --help       show this help message and exit
  --name NAME      Model name
  --output OUTPUT  Output model (default model.vpunn)
```

### VPUNN to JSON

Convert a VPUNN model into json for debugging purpose

```bash
usage: vpunn_to_json [-h] file

positional arguments:
  file        graphFile to deserialize to json OR an already deserialized json

optional arguments:
  -h, --help  show this help message and exit
```

## Javascript (WASM) support

To compile the Web Assembly (WASM) version of the library, follow the steps below:

1. Install Emscripten (link [here](https://emscripten.org/docs/getting_started/downloads.html))
2. Configure Emscripten with cmake by typing `emmake cmake ..`
3. Build the Javascript interface `emmake make vpunn_js -j`

The build command produces an `npm` package that can be later installed in any js project by doing `npm install <path to build folder>/dist/vpunn-*.tgz`

## Developer guide

### Git hooks

All developers should install the git hooks that are tracked in the .githooks directory. We use the pre-commit framework for hook management. The recommended way of installing it is using pip:

```bash
pip install pre-commit
```

The hooks can then be installed into your local clone using:

```bash
pre-commit install --allow-missing-config
```

--allow-missing-config is an optional argument that will allow users to have the hooks installed and be functional even if using an older branch that does not have them tracked. A warning will be displayed for such cases when the hooks are ran.

If you want to manually run all pre-commit hooks on a repository, run `pre-commit run --all-files`. To run individual hooks use `pre-commit run <hook_id>`.

Uninstalling the hooks can be done using

```bash
pre-commit uninstall
```

## Testing the library

### Cost model test (C++)

Tests uses [Google test suite](https://github.com/google/googletest) for automatizing tests
To run the test suite: `ctest --test-dir build/tests/cpp/`

Example: running only cost model integration test: `./tests/cpp/test_cost_model`

### E2E Python test

`pytest tests/python/test_e2e.py -v`

### WASM test

Assuming you build VPUNN WASM library in `build_wasm`, install VPUNN locally with all its dependencies.

```bash
npm install --prefix tests/js
npm install --save-optional build_wasm/dist/vpunn-*.tgz --prefix tests/js
```

Start testing by running

`npm run test --prefix=tests/js`

### Code coverage

To generate Code coverage report you need to enable it in CMake

```shell
cmake -DCMAKE_BUILD_TYPE=Coverage  .. && make coverage -j
```

This commands generate a `coverage` folder into the build one with all the coverage information

Dependencies:

- Gcov-9 and Gcovr tools are needed in order to generate the report
- Only GCC is supported (no WASM/Visual Studio)

## Notice about configurations not covered by training, or with greater errors.
### NPU2.0
Not Available
### NPU2.7
- ISI=CLUSTERING + OWT=2    : replaced at runtime with SOK. runtime should be the same, no input halo used
- Elementwise + ISI=SOK     : replaced at runtime with clustering + owt=1,  time is a little undervalued, but its the best approximation available
- CM_CONV (compress convolution) + InputChannels=1
- SOH (HALO) split with Kernel =1 has probably not been part of training, doesn't make sense to have kernel=1 and input halo.NN predictions are problematic. :   replaced at runtime with Clustering.
- SOH Halo split , at least when H is small, K small, produces much bigger results than SOH Overlapped. This is not realistic, might be a NN limitation. See VPULayerCostModelTest.Unet_perf_SOH_SOK_after_SOHO
- Output write tiles is limited to 2. EG also when used as mock for NPU4.0 where more than 2 tiles are present and used for split.

- NPU2.7 splits by H with Halo  were trained  to NN using the memory tensor instead of the general rule for compute tensor (memory tensor is smaller  with half a kernel in general). Calling NN with compute tensor introduces errors by reporting smaller values. To get corrected values (closer to Ground Truth) when generating the descriptor for NNs with interface 11 and SOH isi strategy, we are using not the input tensor, but a computed memory input tensor that mimics the one used at training

### NPU4.0 (in development)
Reusing:when using the 2.7 trained version as mock please read the NPU2.7 section above.
  - DW_CONV (depthwise convolution)with kernel 3x3 is optimized in NPU4.0, but not in NPU2.7. The NN reported runtime is adjusted with a factor depending on datatype, channels and kernel size
Trained NN for 4.0: 
  - WIP

### Known problems:
- NPU2.7: NN was not trained to discriminate the sporadic high runtime for swizzling. EISXW-98656 not solved (ELt wise add with big profiled CLUSTERING, but small SOH) Test: RuntimeELT_CONV_SOH_SOK_EISXW_98656. 
Elementwise accepts (at NN run) SWizzling ON or OFF but has to be the same for all in/out/wts  all 0 (OFF), all 5(ON) combinations not trained. *To consider:* training of NN with swizzlings combinations (profiling shows runtime is different)



## SHAVE operators available

Shave version interface 1 (the old one) will be deleted in the near future, do not use it.
SHAVE v2 interface is active. 

Details of any operator can be obtained by  calling: ShaveOpExecutor::toString() method. 

For most updated list of operators and their details see also the unit tests: TestSHAVE.SHAVE_v2_ListOfOperators, TestSHAVE.SHAVE_v2_ListOfOperatorsDetails_27,... .

For information about the profiled operators and extraparameters you can consult this [document](src/shave/Readme.md#shave-current-operators)

## Cost providers

The cost model is designed to be extensible. The cost providers are the classes that implement the cost model for a specific device. The cost providers are selected at runtime based on the device type. The following cost providers are available:
- NN based cost provider - is a learned performance model.
- Theoretical cost provider - is a simple mathematical model.
- "Oracle" cost provider - a LUT of measured performance for specific workloads.
- Profiled cost provider - it's an http service that can be queried to get the measured performance of a specific workload.
    - Currently it supports only DPU costs and it can be configured using the following env. variables
        - `ENABLE_VPUNN_PROFILING_SERVICE` -- `TRUE` to enable the profiling service
        - `VPUNN_PROFILING_SERVICE_BACKEND` -- `silicon` to use the RVP for profiling, `vpuem` to use VPUEM as a cost provider.
        - `VPUNN_PROFILING_SERVICE_HOST` -- address of the profiling service host, default is `irlccggpu04.ir.intel.com`
        - `VPUNN_PROFILING_SERVICE_PORT` -- port of the profiling service, default is `5000`

To see a list of all queried workloads and which cost provider was used for each, set the environment variable `ENABLE_VPUNN_DATA_SERIALIZATION` to `TRUE`.
This will generate a couple of `csv` files in the directory where vpunn is used.
