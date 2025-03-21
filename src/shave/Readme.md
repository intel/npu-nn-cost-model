# SHAVE code generation and data format information

## NPU2.7
### CSV Information 
In the [CSV](shave_layers_vpu_2_7.csv) are modeled Shaves that fall into the activation and elementwise category, that operations doesnt require a additional parameters.

Line format: 

- kernel_type
- kernel_name
- slope
- intercept 
- scalar_offset
- unroll_offset
- unroll_size
- vector_size
- DPU_frequency
- SHV_frequency

## NPU4.0
### CSV Information 
In the [CSV](shave_layers_vpu_40.csv) are modeled Shaves that fall into the activation and elementwise category, that operations doesnt require a additional parameters.

Line format: 

- kernel_type
- kernel_name
- slope
- intercept 
- unroll_offset
- intra_block_offset
- vector_offset
- displacement_size
- unroll_size
- vector_size
- DPU_frequency
- SHV_frequency

# Shave current operators

#### Here is all the information about the current profiled operators. **In case that your shave operator does not exist in this list, you can use the default shave operator.** The **default operator** will return the number of DPUCycles equal to the number of elements in the tensor.

## NPU2.7 operators (trained)

### DEVICE : VPU_2_7  has # SHAVE operators : 74 
-  interpolatewh_1     : just a mock/tentative for interpolate, for interpolate where WH is changed (only), and layout is WHCB (strict)
-  MVN                 : params: 1 int, represents the number of axes selected (from innermost to outermost), here it is important also the layout of the tensor. For example if we want to select 2 axes then if the selected ones are N and C we expect them to be innermost ones in order to use MVN6, otherwise if it will be H and W it will make use of Simple MVN.
-  MVN6                : params: 1 int, represents the number of axes selected (from innermost to outermost)
-  MVN6_fourAx         : Does not take any extra parameters. All the axes are selected by default.
-  MVN6_oneAx          : Does not take any extra parameters. By default the innermost dimension is considered selected 
-  MVN6_onlyOneAx      : Does not take any extra parameters. By default the innermost dimension is considered selected 
-  MVN6_threeAx        : Does not take any extra parameters. By default the first three innermost dimensions are considered selected 
-  MVN6_twoAx          : Does not take any extra parameters. By default the first two innermost dimension are considered selected 
-  MVN_2Ax             : Does not take any extra parameters. By default the first two innermost dimension are considered selected. This is the specific case in which we selected H and W, they should be the innermost dimensions.
-  MVN_3Ax             : Does not take any extra parameters. By default the first three innermost dimension are considered selected. This is the specific case in which we selected C, H and W, they should be the innermost dimensions.

These operators below are not taking any extra parameters, The represent the Activation and Elementwise operators profiled:
-  abs
-  acos
-  acosh
-  asin
-  asinh
-  atan
-  atanh
-  ceiling
-  clamp
-  cos
-  cosh
-  cumsum
-  default
-  div
-  elu
-  equal
-  erf
-  exp
-  floor
-  floormod
-  gelu
-  greater
-  greatereq
-  hardsigmoid
-  hardswish
-  hsigmoid
-  hswish
-  less
-  lesseq
-  log
-  logicaland
-  logicalnot
-  logicalor
-  logicalxor
-  max
-  min
-  mish
-  notequal
-  power
-  round
-  select
-  selu
-  sigmoid
-  sign
-  sin
-  sinh
-  sqrt
-  swish
-  tan
-  tanh
-  default :special dummy implementation (like in the old shave ) for not profiled operators. It is the first bisector line, return value in DPU cycles is equal to the number of elements in the output tensor.

## NPU4.0 operators (trained)

### DEVICE : VPU_4_0  has # SHAVE operators : 80
-  gather              : params: 2 int, first represents the axis(dimension index for gathering), second parameter is batch_dims(leading number of dimensions being batches). Both parameters are restricted to value 1,(the only profiled selection)
-  normalizel2onlyc    : params: 1 int, represents the selected dimension(C(1) is the only one supported). Layout restrictions: If Layout different from XYZ will retrieve ERROR_SHAVE_LAYOUT
-  softmax             : params: 1 int, represents the selected dimension(N(0), C(1), H(2), W(3)), Layout restrictions: If Layout different from XYZ will retrieve ERROR_SHAVE_LAYOUT, Input restrictions: the batch size over 1 is not supported
-  **MVN**                 : params: 1 int, represents the number of axes selected (from innermost to outermost), here it is important also the layout of the tensor. **MVN** it is more flexible, it is the top MVN implementation that will select its optimized sub model based on layout and selected dimensions.
    - **MVN** (top implementation) breaks down into:
        - **MVN_2Ax** - This is a specific scenario when we select 2 axes (top implementation gets 2 as a parameter) and the layout has W as the innermost dimension and H as the second innermost dimension. 
        - **MVN_3Ax** - This is a specific scenario when we select 3 axes (top implementation gets 3 as a parameter) and the layout has W as the innermost dimension, H as the second innermost dimension and C as the third dimension. 
        - **MVN6** The above implementations are optimized cases. If it doesnt fall into any of above, it will use the MVN6 generic implementation.
-  **MVN6**                : params: 1 int, represents the number of axes selected (from innermost to outermost). This is the model for the generic implementation of MVN6 with the flexiblity of selecting 1 to 4 axes. This implementation will **not** treat the optimization in case of selecting the H or W.
    - **MVN6** (top implementation) breaks down into:
        - **MVN6_oneAx** - In case that we have only one selected axes it will use the MVN6 model for only one axes.  
        - **MVN6_twoAx** - In case that we have only one selected axes it will use the MVN6 model for only two axes.
        - **MVN6_threeAx** - In case that we have only one selected axes it will use the MVN6 model for only three axes.
        - **MVN6_fourAx** - In case that we have only one selected axes it will use the MVN6 model for all axes.
-  **MVN6_fourAx**         : Does not take any extra parameters. All the axes are selected by default. This is the model for the generic implementation of MVN6 with the restriction of selecting all axes.
-  **MVN6_oneAx**          : Does not take any extra parameters. By default the innermost dimension is considered selected. This is the model for the generic implementation of MVN6 with the restriction of selecting one axes.
-  **MVN6_onlyOneAx**      : Does not take any extra parameters. By default the innermost dimension is considered selected. This is the model for the generic implementation of MVN6 with the restriction of selecting one axes.
-  **MVN6_threeAx**        : Does not take any extra parameters. By default the first three innermost dimensions are considered selected. This is the model for the generic implementation of MVN6 with the restriction of selecting three axes.
-  **MVN6_twoAx**          : Does not take any extra parameters. By default the first two innermost dimension are considered selected. This is the model for the generic implementation of MVN6 with the restriction of selecting two axes. 
-  **MVN_2Ax**             : Does not take any extra parameters. By default the first two innermost dimension are considered selected. This is the specific case which represents the optimized model where we selected H and W, they should be the innermost dimensions. 
-  **MVN_3Ax**             : Does not take any extra parameters. By default the first three innermost dimension are considered selected. This is the specific case which represents the optimized model where we selected C, H and W, they should be the innermost dimensions.

These operators below are not taking any extra parameters, The represent the Activation and Elementwise operators profiled:
-  abs
-  acos
-  acosh
-  add
-  asin
-  asinh
-  atan
-  atanh
-  ceiling
-  clamp
-  cos
-  cosh
-  cumsum
-  default
-  div
-  elu
-  erf
-  exp
-  fakequantize
-  floor
-  floormod
-  gelu
-  greater
-  greatereq
-  hardsigmoid
-  hsigmoid
-  hswish
-  less
-  lesseq
-  log
-  logicaland
-  logicalnot
-  logicalor
-  logicalxor
-  max
-  min
-  mish
-  mul
-  negative
-  notequal
-  power
-  prelu
-  relu
-  round
-  select
-  selu
-  sigmoid
-  sign
-  sin
-  sinh
-  softplus
-  sqrt
-  squaredifference
-  sub
-  swish
-  tan
-  tanh
-  equal the time is the same despite the size (2952 DPU cycles). The only thing that gives equal a slope was the complementary convert operations before and after the equal operation. Since it is a special case it will be treated as a constant and it will give a constant time. The operations of convert appear in case that we use the ReferenceSW pipeline but in the real case will Convert run on SW or on DMA?
-  default :special dummy implementation (like in the old shave ) for not profiled operators. It is the first bisector line, return value in DPU cycles is equal to the number of elements in the output tensor.

## VPUEM Operators Usage
  
All these operation are available for VPU2.7, VPU4., it is only necessary to change the values of the parameters corresponding to the generation of the device. 
The name convention is achieved by concatenating the specific prefix and the name of each operation to highlight that the computing logic from VPUEM is used.

To check the parameters use toString() function to display it. 

The method `dpuCycles` return the number of computed cycles at the given DPU Frequency. The values of computed cycles are closed to the effective cycles, the difference is insignificant. 
For Piecewise Operation, the tensor is split in subblocks and for each blocks is computed the ideal cycles and sum together to get the computed cycles.

For Softamx Operation is necessary to give the hight, product betweeh hight and width, and the channels to compute the compute the ideal cycles. The operations are incomplet, the user can get just the idel cycles. It is restricted for C innermost ( XYZ layout).

For Spatial Operation is necessary to give the entire tensor size to compute the ideal cycles The operations are incomplete, it is available just the method for the ideal cycles.

## VPUEM Operators VPU2.7

A Piecewise Operation is a simple operation based on a 3 slopes equation. Available Piecewise Operations from VPUEM:
* vpuem.add
* vpuem.gelu
* vpuem.gptq
* vpuem.hswish
* vpuem.log
* vpuem.mul
* vpuem.mvn
* vpuem.mvn_cw
* vpuem.mvn_fused
* vpuem.sigmoid
* vpuem.softmax - no need for extra params
* vpuem.softmax_x
* vpuem.swish
* vpuem.tanh

Available Softmax Operation from VPUEM:
* vpuem.softmax_x (draft) - it is restricted for C innermost ( XYZ layout).

Available Spatial Operations from VPUEM:
* vpuem.mvn (draft) - depends on the layout, the model does not require normalization axes

## VPUEM Operators VPU4.0

A Piecewise Operation is a simple operation based on a 3 slopes equation. Available Piecewise Operations from VPUEM:
* vpuem.add
* vpuem.gelu
* vpuem.hswish
* vpuem.log
* vpuem.mul
* vpuem.mvn
* vpuem.sigmoid
* vpuem.softmax - no need for extra param
* vpuem.softmax_x
* vpuem.swish
* vpuem.tanh

Available Softmax Operation from VPUEM:
* vpuem.softmax_x (draft) - it is restricted for C innermost ( XYZ layout).

Available Spatial Operations from VPUEM:
* vpuem.mvn (draft) - depends on the layout, the model does not require normalization axes
