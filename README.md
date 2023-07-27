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

Do this in a Python virtual environment.

### VPU cost model
Run the `vpu_cost_model` script to evaluate workloads from the command line

```bash
usage: vpu_cost_model [-h] [--model MODEL] [--mode {DPU,DMA}] [--target {cycles,power,utilization}]
                      [--device {VPU_2_0,VPU_2_1,VPU_2_7,VPU_4_0}]
                      [--operation {CONVOLUTION,DW_CONVOLUTION,ELTWISE,MAXPOOL,CM_CONVOLUTION}]
                      [--mpe_mode {4x4,16x1,4x1}] 
                      [--nthw-ntk {4x16,8x8,16x4}]
                      [--activation {NONE,RELU,MULT,LRELU,ADD,SUB}] 
                      [--width WIDTH] [--height HEIGHT]
                      [--input_channels INPUT_CHANNELS]
                      [--output_channels OUTPUT_CHANNELS] 
                      [--batch BATCH]
                      [--kernel KERNEL] [--padding PADDING] [--strides STRIDES]
                      [--input_dtype {UINT8,INT8,FLOAT16,BFLOAT16}] 
                      [--output_dtype {UINT8,INT8,FLOAT16,BFLOAT16}]
                      [--output_layout {ZXY,XZY,YXZ,YZX,ZYX,XYZ}]
                      [--isi_strategy {clustering,split_over_h,split_over_k}]
                      [--act-sparsity ACT_SPARSITY]
                      [--param-sparsity-enabled PARAM_SPARSITY_ENABLED]
                      [--param-sparsity PARAM_SPARSITY]
                      [--input-swizzling INPUT_SWIZZLING]
                      [--param-swizzling PARAM_SWIZZLING]
                      [--output-swizzling OUTPUT_SWIZZLING]
                      [--output-write-tiles OUTPUT_WRITE_TILES]

VPU cost model

options:
  -h, --help            show this help message and exit
  --model MODEL, -m MODEL
                        Model path
  --mode {DPU,DMA}      Profiling mode
  --target {cycles,power,utilization}
                        Target type
  --device {VPU_2_0,VPU_2_1,VPU_2_7,VPU_4_0}, -d {VPU_2_0,VPU_2_1,VPU_2_7,VPU_4_0}
                        The VPU IP device
  --operation {CONVOLUTION,DW_CONVOLUTION,ELTWISE,MAXPOOL,CM_CONVOLUTION}, -op {CONVOLUTION,DW_CONVOLUTION,ELTWISE,MAXPOOL,CM_CONVOLUTION}
                        The operation
  --mpe_mode {4x4,16x1,4x1}
                        DPU MPE mode
  --nthw-ntk {4x16,8x8,16x4}
                        DPU nthw-ntk mode
  --activation {NONE,RELU,MULT,LRELU,ADD,SUB}, -act {NONE,RELU,MULT,LRELU,ADD,SUB}
                        The operation activation function
  --width WIDTH, -x WIDTH
                        Tensor width
  --height HEIGHT, -y HEIGHT
                        Tensor height
  --input_channels INPUT_CHANNELS, -ic INPUT_CHANNELS
                        Tensor input channels
  --output_channels OUTPUT_CHANNELS, -oc OUTPUT_CHANNELS
                        Tensor output channels
  --batch BATCH, -b BATCH
                        Tensor batch
  --kernel KERNEL, -k KERNEL
                        Operation Kernel
  --padding PADDING, -p PADDING
                        Operation padding
  --strides STRIDES, -s STRIDES
                        Operation strides
  --input_dtype {UINT8,INT8,FLOAT16,BFLOAT16}
                        The input datatype
  --output_dtype {UINT8,INT8,FLOAT16,BFLOAT16}
                        The output datatype
  --output_layout {ZXY,XZY,YXZ,YZX,ZYX,XYZ}
                        The odu layout
  --isi_strategy {clustering,split_over_h,split_over_k}
                        ISI Strategy
  --act-sparsity ACT_SPARSITY
                        Activation tensor sparsity
  --param-sparsity-enabled PARAM_SPARSITY_ENABLED
                        Weight tensor sparsity enabled
  --param-sparsity PARAM_SPARSITY
                        Weight tensor sparsity
  --input-swizzling INPUT_SWIZZLING
                        Input tensor swizzling
  --param-swizzling PARAM_SWIZZLING
                        Weight tensor swizzling
  --output-swizzling OUTPUT_SWIZZLING
                        output tensor swizzling
  --output-write-tiles OUTPUT_WRITE_TILES
                        Controls on how many tiles the DPU broadcast (1 = no broadcast)
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
Tests use [Google test suite](https://github.com/google/googletest).s
To run the test suite: `ctest --test-dir build/tests/cpp/`

Example: running only cost model integration test: `./tests/cpp/test_cost_model`

### E2E Python test
To run the Python tests you will need to install pytest. There is a further dependency on tensorflow for some tests.

Install both in the same Python virtual environment in which you have installed the cost model (as per the instructions above). 
Example: running only end-to-end tests: `pytest tests/python/test_e2e.py -v`

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



