# Python Bindings

The Python bindings for VPUNN are implemented using `pybind11` and are located in `src/python/VPUNN.cpp`.

## Installation

To install the package with bindings, run the following command from the root of the repository:

```bash
pip install .
```

This will use `scikit-build` to compile the C++ extension and install the Python package.

Alternatively, for development, the extension can be built in-tree:

```bash
cmake -S . -B build -DENABLE_PYTHON_BINDING=ON
cmake --build build --target vpunn_python_bindings -j
export PYTHONPATH=$PWD/build/lib   # holds _VPUNN*.so
```

## SHAVE (software kernel) cost queries

`SHAVECostModel` mirrors the C++ API: the same workload returns the same cycles in both languages.

```python
import _VPUNN as VPUNN  # or: import vpunn as VPUNN, when installed as a package

model = VPUNN.SHAVECostModel()  # optional: SHAVECostModel(cache_filename, cache_size)

# operators modelled for a device (the static query needs no model instance)
operations = VPUNN.SHAVECostModel.queryDeviceMappedSupportedOperations(VPUNN.VPUDevice.NPU_5_0)

tensor = VPUNN.VPUTensor(56, 56, 16, 1, VPUNN.DataType.FLOAT16, VPUNN.Layout.ZXY)
workload = VPUNN.SHAVEWorkload("MVN", VPUNN.VPUDevice.NPU_5_0, [tensor], [tensor])

cycles = model.computeCycles(workload)
cycles, info = model.computeCyclesMsg(workload)   # same value, plus diagnostic text

if VPUNN.Cycles.isErrorCode(cycles):
    print("query failed:", VPUNN.Cycles.toErrorText(cycles))
```

Scope and behaviour of this interface:

- **NPU 5.0 and newer only.** Older devices (`VPU_2_0` ... `VPU_4_0`) raise `ValueError`, both when
  building a `SHAVEWorkload` and when querying the supported operators.
- **A workload is only name, device, input and output tensors.** Call parameters, extra parameters,
  the location name and the profiling backend hint are deliberately not exposed.
- **Code prefetch is off by default.** `workload.include_code_prefetch = True` models a cold (first)
  invocation; `model.getCodePrefetchCost(workload)` returns that cold start cost on its own.
- **Errors are returned in band**, as in C++: always check them with `Cycles.isErrorCode()`
  (for example `ERROR_SHAVE_OPERATOR_MISSING` when the operator is not modelled for that device).
- `VPUCostModel.SHAVE()` / `VPUCostModel.SHAVEMsg()` expose the same query through the generic cost model.

Other useful entry points: `getShaveSupportedOperations(device)`, `getShaveInstanceInfo(name, device)`,
`workload.total_number_of_elements()`, `workload.total_size_in_bits()`, `workload.hash()`.

## Development

If you need to modify the bindings:

1.  Edit `src/python/VPUNN.cpp` or `src/python/binding.h`.
2.  Reinstall the package to rebuild the extension:
    ```bash
    pip install .
    ```

