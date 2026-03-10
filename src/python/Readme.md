# Python Bindings

The Python bindings for VPUNN are implemented using `pybind11` and are located in `src/python/VPUNN.cpp`.

## Installation

To install the package with bindings, run the following command from the root of the repository:

```bash
pip install .
```

This will use `scikit-build` to compile the C++ extension and install the Python package.

## Development

If you need to modify the bindings:

1.  Edit `src/python/VPUNN.cpp` or `src/python/binding.h`.
2.  Reinstall the package to rebuild the extension:
    ```bash
    pip install .
    ```

