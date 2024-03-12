# NPUNN cost model

A NN-Based Cost Model for NPU Devices. For additional information about model setup and training, please refer [this paper](https://arxiv.org/abs/2205.04586)

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

To use the NPUNN cost model in a cmake project is quite simple. An example of a CMakeLists.txt file is shown below

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

### NPU cost model

Run the `vpu_cost_model` script to evaluate workloads from the command line

```bash
usage: vpu_cost_model [-h] [--model MODEL] [-t {cycles,power,utilization}] -d {VPU_2_0,VPU_2_1,VPU_2_7} {DPU,DMA} ...

NPU cost model

positional arguments:
  {DPU,DMA}

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL, -m MODEL
                        Model path
  -t {cycles,power,utilization}, --target {cycles,power,utilization}
                        The target type
  -d {VPU_2_0,VPU_2_1,VPU_2_7}, --device {VPU_2_0,VPU_2_1,VPU_2_7}
                        The NPU IP device
```

DPU arguments:
```
usage: vpu_cost_model DPU [-h] -o {convolution,dw_convolution,eltwise,maxpool,avepool,cm_convolution} --inch INPUT_0_CHANNELS [--outch OUTPUT_0_CHANNELS] --height INPUT_0_HEIGHT --width INPUT_0_WIDTH
                          [--input-sparsity-enabled] [--weight-sparsity-enabled] [--input-sparsity-rate INPUT_SPARSITY_RATE] [--weight-sparsity-rate WEIGHT_SPARSITY_RATE] --mpe-mode
                          {VECTOR_FP16,VECTOR,MATRIX,CUBOID_4x16,CUBOID_8x16,CUBOID_16x16} [--af {none,relu,lrelu,add,sub,mult}] --kh KERNEL_HEIGHT --kw KERNEL_WIDTH [--pb KERNEL_PAD_BOTTOM]
                          [--pl KERNEL_PAD_LEFT] [--pr KERNEL_PAD_RIGHT] [--pt KERNEL_PAD_TOP] [--sh KERNEL_STRIDE_HEIGHT] [--sw KERNEL_STRIDE_WIDTH] --indt {uint8,int8,float16,bfloat16} --outdt
                          {uint8,int8,float16,bfloat16} [--input-layout {zxy,xzy,yxz,yzx,zyx,xyz,invalid}] [--output-layout {zxy,xzy,yxz,yzx,zyx,xyz,invalid}] [--input_swizzling INPUT_0_SWIZZLING]
                          [--weight_swizzling INPUT_1_SWIZZLING] [--output_swizzling OUTPUT_0_SWIZZLING] [--isi {clustering,split_over_h,split_over_k}] [--owt OUTPUT_WRITE_TILES]
                          [--output-sparsity-enabled]

optional arguments:
  -h, --help            show this help message and exit
  -o {convolution,dw_convolution,eltwise,maxpool,avepool,cm_convolution}, --op {convolution,dw_convolution,eltwise,maxpool,avepool,cm_convolution}, --operation {convolution,dw_convolution,eltwise,maxpool,avepool,cm_convolution}
                        Operation type
  --inch INPUT_0_CHANNELS, --input-channels INPUT_0_CHANNELS, --input-0-channels INPUT_0_CHANNELS
                        Number of input channels
  --outch OUTPUT_0_CHANNELS, --output-channels OUTPUT_0_CHANNELS, --output-0-channels OUTPUT_0_CHANNELS
                        Number of output channels
  --height INPUT_0_HEIGHT, --input-height INPUT_0_HEIGHT, --input-0-height INPUT_0_HEIGHT
                        Input activation height
  --width INPUT_0_WIDTH, --input-width INPUT_0_WIDTH, --input-0-width INPUT_0_WIDTH
                        Input activation width
  --input-sparsity-enabled
                        The flag to enable input sparsity
  --weight-sparsity-enabled
                        The flag to enable weight sparsity
  --input-sparsity-rate INPUT_SPARSITY_RATE
                        The rate of input sparsity (only valid when enabling input sparsity)
  --weight-sparsity-rate WEIGHT_SPARSITY_RATE
                        The rate of weight sparsity (only valid when enabling weight sparsity)
  --mpe-mode {VECTOR_FP16,VECTOR,MATRIX,CUBOID_4x16,CUBOID_8x16,CUBOID_16x16}, --execution-order {VECTOR_FP16,VECTOR,MATRIX,CUBOID_4x16,CUBOID_8x16,CUBOID_16x16}, --execution-mode {VECTOR_FP16,VECTOR,MATRIX,CUBOID_4x16,CUBOID_8x16,CUBOID_16x16}
                        For KMB device set the MPE mode, for later devices it sets the Execution Order (nthw)
  --af {none,relu,lrelu,add,sub,mult}, --activation-function {none,relu,lrelu,add,sub,mult}
                        The activation function that follow the operation (only valid for KMB)
  --kh KERNEL_HEIGHT, --kernel-height KERNEL_HEIGHT
                        The kernel height
  --kw KERNEL_WIDTH, --kernel-width KERNEL_WIDTH
                        The kernel width
  --pb KERNEL_PAD_BOTTOM, --pad-bottom KERNEL_PAD_BOTTOM
                        The bottom padding
  --pl KERNEL_PAD_LEFT, --pad-left KERNEL_PAD_LEFT
                        The left padding
  --pr KERNEL_PAD_RIGHT, --pad-right KERNEL_PAD_RIGHT
                        The right padding
  --pt KERNEL_PAD_TOP, --pad-top KERNEL_PAD_TOP
                        The top padding
  --sh KERNEL_STRIDE_HEIGHT, --stride-height KERNEL_STRIDE_HEIGHT
                        The stride height
  --sw KERNEL_STRIDE_WIDTH, --stride-width KERNEL_STRIDE_WIDTH
                        The stride width
  --indt {uint8,int8,float16,bfloat16}, --input-datatype {uint8,int8,float16,bfloat16}
                        The input datatype
  --outdt {uint8,int8,float16,bfloat16}, --output-datatype {uint8,int8,float16,bfloat16}
                        The output datatype
  --input-layout {zxy,xzy,yxz,yzx,zyx,xyz,invalid}
                        The input layout
  --output-layout {zxy,xzy,yxz,yzx,zyx,xyz,invalid}
                        The output layout
  --input_swizzling INPUT_0_SWIZZLING, --input-0-swizzling INPUT_0_SWIZZLING
                        The input swizzling
  --weight_swizzling INPUT_1_SWIZZLING, --input-1-swizzling INPUT_1_SWIZZLING
                        The weight swizzling
  --output_swizzling OUTPUT_0_SWIZZLING, --output-0-swizzling OUTPUT_0_SWIZZLING
                        The output swizzling
  --isi {clustering,split_over_h,split_over_k}, --isi-strategy {clustering,split_over_h,split_over_k}
                        The ISI Strategy
  --owt OUTPUT_WRITE_TILES, --output-write-tiles OUTPUT_WRITE_TILES
                        Controls on how many tiles the DPU broadcast (1 = no broadcast)
  --output-sparsity-enabled
                        The flag to enable output sparsity
```

minimal example:

```
vpu_cost_model --device VPU_2_7 DPU  -o CONVOLUTION --inch 64 --outch 64 --height 16 --width 16 --kh 3 --kw 3 --indt UINT8 --outdt UINT8 --mpe-mode CUBOID_16x16
```

DMA arguments:
```
usage: vpu_cost_model DMA [-h] --height HEIGHT --width WIDTH --kernel KERNEL --padding PADDING --strides STRIDES --device DEVICE --input_channels INPUT_CHANNELS --output_channels OUTPUT_CHANNELS
                          --input_dtype INPUT_DTYPE --output_dtype OUTPUT_DTYPE

optional arguments:
  -h, --help            show this help message and exit
  --height HEIGHT
  --width WIDTH
  --kernel KERNEL
  --padding PADDING
  --strides STRIDES
  --device DEVICE
  --input_channels INPUT_CHANNELS
  --output_channels OUTPUT_CHANNELS
  --input_dtype INPUT_DTYPE
  --output_dtype OUTPUT_DTYPE
```

#### NPUNN builder

Generate a NPUNN model from a tensorflow one

```bash
optional arguments:
  -h, --help       show this help message and exit
  --name NAME      Model name
  --output OUTPUT  Output model (default model.vpunn)
```

### NPUNN to JSON

Convert a NPUNN model into json for debugging purpose

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

Assuming you build NPUNN WASM library in `build_wasm`, install NPUNN locally with all its dependencies.

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
