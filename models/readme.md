# VPU NN Cost Model Versions
## VPU 2.0
Filenames of available cost model networks:
* vpu_2_0.vpunn   (default for python interface)
* vpu_2_0.fast.vpunn

These neural network checkpoints have been trained using ArchBench data. The dataset is composed of 10M workloads distributed over the following operations:

| Operation  | % |
| ------------- | - |
| CONVOLUTION  | 20%  |
| CM_CONVOLUTION  | 20% | 
| DW_CONVOLUTION  | 20%  |
| MAXPOOL  | 20% |
| ELTWISE  | 20% |

And the following data types:

| DataType  | % |
| ------------- | - |
| UINT8  | 50%  |
| FLOAT16  | 50%  |

### Features: 
- *vpu_2_0.vpunn*: Input interface/descriptor expects 67 values in a 67 vector (interface 10). This was  retrained in the beginning of 2023
- *vpu_2_0.fast.vpunn*: Input interface/descriptor expects 67 values in a 67 vector (interface 10). This was  retrained in the beginning of 2023
- Output is:
    * vpu_2_0.vpunn: output is directly cycles (interface version 2)
    * vpu_2_0.fast.vpunn: output is directly cycles (interface version 2)

### Performance:
- Untested against ground truth from VPU2.0 since no such dataset exists.
- Performance on the training/validation dataset:

* vpu_2_0.vpunn
    | Data set | average error | median error | max error | > 20% error |
    | -------- | ------------- | ------------ | --------- | ----------- |
    | *Test* | 1.02% | 0.40% | 53.86% | 0.16% |
* vpu_2_0.fast.vpunn
    | Data set | average error | median error | max error | > 20% error |
    | -------- | ------------- | ------------ | --------- | ----------- |
    | *Test* | 2.70% |  1.34% | 131.65% | 1.63% |

vpu_2_0.vpunn vs vpu_2_0.fast.vpunn
ERRORS                  22.65%
CORRELATION             0.741

### Model structure:
* vpu_2_0.vpunn
    ```Kernels: [256,256,256,256,1]
    Activation Funtion: Relu
    ```
* vpu_2_0.fast.vpunn
    ```Kernels: [64,64,64,64,1]
    Activation Funtion: Relu
    ```

### Interface required:
| model | Input | Output |  
| -------------------- | ------------- | ------------ |
| *vpu_2_0.vpunn* | 10 | 2 |   
| *vpu_2_0.fast.vpunn* | 10 | 2|


## VPU 2.7
Filenames of available cost model networks:
* vpu_2_7.vpunn  (default for python interface)
* vpu_2_7.fast.vpunn

The training dataset was built with (mainly) FPGA measurements and it is composed of about **2.5 million workloads** including both dense and sparse, with a range of sparsity levels, and distributed over the following operations:

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
- Input: version 11 (see the model `name` rules and versioning section below)  
- Output: version 2 (DPU cycles)

### Performance:

* vpu_2_7.vpunn
    | Data set | average error | median error | max error | < 20% error |   
    | -------- | ------------- | ------------ | --------- | ----------- |
    | *Training* | 4.89% | 3.26% | 97.89% | 148.11% |
    | *Validation* | 4.88% | 3.25% | 97.91% | 89.80% |

* vpu_2_7.fast.vpunn
    | Data set | average error | median error | max error | < 20% error |   
    | -------- | ------------- | ------------ | --------- | ----------- |
    | *Training* | 5.94% | 4.17% | 96.85% | 104.66% |
    | *Validation* | 5.94% | 4.17% | 96.83% | 82.20% |


### Model structure:
* vpu_2_7.vpunn
    ```python
    from torch import nn
    nn.Sequential(
        nn.Linear(93,256),
        nn.ReLU(),
        nn.Linear(256,256),
        nn.ReLU(),
        nn.Linear(256,256),
        nn.ReLU(),
        nn.Linear(256,256),
        nn.ReLU(),
        nn.Linear(256,1),
    )
    ```
* vpu_2_7.fast.vpunn
    ```python
    from torch import nn
    nn.Sequential(
        nn.Linear(93,64),
        nn.ReLU(),
        nn.Linear(64,64),
        nn.ReLU(),
        nn.Linear(64,64),
        nn.ReLU(),
        nn.Linear(64,64),
        nn.ReLU(),
        nn.Linear(64,1),
    )
    ```
### Interface required: 
| model | Input | Output |  
| -------------------- | ------------- | ------------ |
| *vpu_2_7.vpunn* | 11 | 2 |   
| *vpu_2_7.fast.vpunn* | 11 | 2|

## Inference latency
Time is in microseconds

DPU computation, Executing workload by workload (batch is 1)
| model                 | Average   | Min       |  Average  | Min       | Average   | Min |
| --------------------  | --------- | --------- | --------- | --------- | ----------| --------- |
| *vpu_2_0.vpunn*       | 50        | 49        | 63        | 62        |   33      | 30 
| *vpu_2_0.fast.vpunn*  | 8         | 8         | 11        | 10        |   7       | 6
| *vpu_2_7.vpunn*       | 67        | 66        | 84        | 83        |   60      | 54 
| *vpu_2_7.fast.vpunn*  | 19        | 18        | 24        | 23        |   20      | 17 
|| GCC || clang14 || MSVC ||
|| Xeon E5-2699|2.3GHz| Xeon E5-2699|2.2GHz|  i7-1185G7 (win11) |3GHz|

The times are highly dependent on CPU load/situation (2x increase can be seen in bad cases).
Profiling was done on 1000 random workloads.


# Model `name` rules and versioning
The following naming convention specifying the interfaces of the model is applied to the `name` field of the model schema.

**`VPUNN-NN1-NN2`**

**`VPUNN`**: a name, currently `VPUNN` is used. (Must not contain `-`.)

**`NN1`**: a number, represents the version of the input interface, may be:
- nothing (missing)
- **`00`**: Use the latest development implementation. Use with caution, normally for draft development in branch. Default in case no name is present in NN
- **`01`**: (default if missing). Legacy mode for VPU2.0 and VPU2.7 before October 2022. (input vector size 71, but only 67 filled in)
- **`10`**: From Nov 2022. (input vector size 67, 67 filled in), i.e. a a patch on input interface `01`. (No extra input_1, no sparsity, no layout in descriptor)
- **`11`**: From Feb 2023.For Beta VPU2.7 (input vector size 93). Contains changes versus prev:  
	- added input_1 as deduced values(except w sparsity enabled flag that is obtained from DPUWorkload )
	- added the Layout in the Tensor descriptor (all 3 tensors). enum with xyz permutations + INVALID
	- added sparsity enabled flag in Tensor descriptor (all 3 tensors)
	- removed activation function
	- aded isi_strategy enum with 3 values : CLUSTERING, SPLIT_OVER_H, SPLIT_OVER_K
	
**`NN2`**: a number, represents the version of the output, may be:
- nothing (missing)
- **`1`**: (default if mising), Legacy mode, *hardware overhead*. In some situations the overhead cannot be smaller than 1.0, and the user should limit it
- **`2`**: DPU clock cycles
- **`3`**: Hardware overhead, unbounded values. Correct even if values are smaller than one, no user guard is required


