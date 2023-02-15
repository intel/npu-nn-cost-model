# VPU NN Cost Model Versions
Please document here details about each model available. Update at each new version.

## VPU 2.0
filename: vpu_2_0.vpunn
This neural network checkpoint has been trained using ArchBench data.

### Release:

### Features: 
- Input interface/descriptor expects 67 values in a 71 vector, (default original/01 interface)
- Output is hw_overhead (interface version 1), always limited to be >= 1.0

### Performance:
- Untested against ground truth from VPU2.0 since no such dataset exists.

### Interface required: 
- input: `01`
- output: `1`

## VPU 2.7

filename: vpu_2_7.vpunn

The dataset was built with (mainly) FPGA measurements and it is composed of about 500k workloads distributed over the following operations:

| Operation  | % |
| ------------- | - |
| CONVOLUTION  | 40%  |
| CM_CONVOLUTION  | 25% | 
| DW_CONVOLUTION  | 15%  |
| MAXPOOL  | 15% |
| ELTWISE  | 5% |

And the following data types:

| DataType  | % |
| ------------- | - |
| UINT8  | 66%  |
| FLOAT16  | 33%  |

### Features: 
- Input interface/descriptor expects 67 values in a 67 vector, same as interface 01 but with correct size. Input interface version: 10.  
- Output is directly cycles (interface version 2)

### Performance:

| Data set | average error | median error | max error | < 20% error |   
| -------- | ------------- | ------------ | --------- | ----------- |
| *Training* | 1.72% | 0.58% | 75.2% | 98.1% |    
| *Validation* | 4.03% | 2.42% | 80.8% | 97.7% |    

### Model structure:

```python
from torch import nn
nn.Sequential(
    nn.Linear(67,256),
    nn.ReLU(),
    nn.Linear(256,512),
    nn.ReLU(),
    nn.Linear(512,1024),
    nn.ReLU(),
    nn.Linear(1024,512),
    nn.ReLU(),
    nn.Linear(512,1),
)
```

### Interface required: 
- input: `10`
- output: `2`

# Model `name` rules and versioning
The following naming convention specifying the interfaces of the model is applied to the `name` field of the model schema.

**`VPUNN-NN1-NN2`**

**`VPUNN`**: a name, currently `VPUNN` is used. (Must not contain `-`.)

**`NN1`**: a number, represents the version of the input interface, may be:
- nothing (missing)
- **`00`**: Use the latest development implementation. Use with caution, normally for draft development in branch. Default in case no name is present in NN
- **`01`**: (default if missing). Legacy mode for VPU2.0 and VPU2.7 before October 2022. (input vector size 71, but only 67 filled in)
- **`10`**: From Nov 2022. (input vector size 67, 67 filled in), i.e. a a patch on input interface `01`. (No extra input_1, no sparsity, no layout in descriptor)
	
**`NN2`**: a number, represents the version of the output, may be:
- nothing (missing)
- **`1`**: (default if mising), Legacy mode, *hardware overhead*. In some situations the overhead cannot be smaller than 1.0, and the user should limit it
- **`2`**: DPU clock cycles
- **`3`**: Hardware overhead, unbounded values. Correct even if values are smaller than one, no user guard is required


