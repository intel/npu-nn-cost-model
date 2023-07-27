# Update new python bindings

In order to update new python bindings you need to run the following commands

```bash
cmake -DENABLE_PYTHON_BINDING=ON -DGENERATE_PYTHON_BINDING=ON ..
make vpunn_python_bindings -j
```

This will use Binder and pybind11 to build python bindings for the library

Other approach is to do in the root folder: 

```
cmake -H. -Bbuild -DENABLE_PYTHON_BINDING=ON -DGENERATE_PYTHON_BINDING=ON 
cmake --build build --target vpunn_python_bindings
```

