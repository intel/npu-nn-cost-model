// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// Minimal hand-written Python bindings for VPUNN - DPUMsg functionality only

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <vpu_cost_model.h>
#include <vpu_dma_cost_model.h>
#include <vpu/types.h>
#include <vpu/dma_types.h>

namespace py = pybind11;

PYBIND11_MODULE(VPUNN, m) {
    m.doc() = "Minimal VPUNN Python bindings for DPU and DMA cost modeling";

    // Core enums needed for DPUWorkload
    py::enum_<VPUNN::VPUDevice>(m, "VPUDevice")
        .value("VPU_2_0", VPUNN::VPUDevice::VPU_2_0)
        .value("VPU_2_1", VPUNN::VPUDevice::VPU_2_1)
        .value("VPU_2_7", VPUNN::VPUDevice::VPU_2_7)
        .value("VPU_4_0", VPUNN::VPUDevice::VPU_4_0)
        .value("NPU_5_0", VPUNN::VPUDevice::NPU_5_0)
        .value("NPU_RESERVED", VPUNN::VPUDevice::NPU_RESERVED)
        .value("NPU_RESERVED_1", VPUNN::VPUDevice::NPU_RESERVED_1);

    py::enum_<VPUNN::DataType>(m, "DataType")
        .value("UINT8", VPUNN::DataType::UINT8)
        .value("INT8", VPUNN::DataType::INT8)
        .value("FLOAT16", VPUNN::DataType::FLOAT16)
        .value("BFLOAT16", VPUNN::DataType::BFLOAT16)
        .value("BF8", VPUNN::DataType::BF8)
        .value("HF8", VPUNN::DataType::HF8)
        .value("UINT4", VPUNN::DataType::UINT4)
        .value("INT4", VPUNN::DataType::INT4)
        .value("UINT2", VPUNN::DataType::UINT2)
        .value("INT2", VPUNN::DataType::INT2)
        .value("UINT1", VPUNN::DataType::UINT1)
        .value("INT1", VPUNN::DataType::INT1)
        .value("INT32", VPUNN::DataType::INT32)
        .value("FLOAT32", VPUNN::DataType::FLOAT32)
        .value("UINT16", VPUNN::DataType::UINT16)
        .value("INT16", VPUNN::DataType::INT16)
        .value("FLOAT4", VPUNN::DataType::FLOAT4);

    py::enum_<VPUNN::Operation>(m, "Operation")
        .value("CONVOLUTION", VPUNN::Operation::CONVOLUTION)
        .value("DW_CONVOLUTION", VPUNN::Operation::DW_CONVOLUTION)
        .value("ELTWISE", VPUNN::Operation::ELTWISE)
        .value("MAXPOOL", VPUNN::Operation::MAXPOOL)
        .value("AVEPOOL", VPUNN::Operation::AVEPOOL)
        .value("CM_CONVOLUTION", VPUNN::Operation::CM_CONVOLUTION)
        .value("LAYER_NORM", VPUNN::Operation::LAYER_NORM)
        .value("ELTWISE_MUL", VPUNN::Operation::ELTWISE_MUL);

    py::enum_<VPUNN::ActivationFunction>(m, "ActivationFunction")
        .value("NONE", VPUNN::ActivationFunction::NONE)
        .value("RELU", VPUNN::ActivationFunction::RELU)
        .value("LRELU", VPUNN::ActivationFunction::LRELU)
        .value("ADD", VPUNN::ActivationFunction::ADD)
        .value("SUB", VPUNN::ActivationFunction::SUB)
        .value("MULT", VPUNN::ActivationFunction::MULT);

    py::enum_<VPUNN::ExecutionMode>(m, "ExecutionMode")
        .value("VECTOR", VPUNN::ExecutionMode::VECTOR)
        .value("MATRIX", VPUNN::ExecutionMode::MATRIX)
        .value("VECTOR_FP16", VPUNN::ExecutionMode::VECTOR_FP16)
        .value("CUBOID_16x16", VPUNN::ExecutionMode::CUBOID_16x16)
        .value("CUBOID_8x16", VPUNN::ExecutionMode::CUBOID_8x16)
        .value("CUBOID_4x16", VPUNN::ExecutionMode::CUBOID_4x16)
        .value("dCIM_32x128", VPUNN::ExecutionMode::dCIM_32x128);

    py::enum_<VPUNN::Layout>(m, "Layout")
        .value("ZMAJOR", VPUNN::Layout::ZMAJOR)
        .value("CMAJOR", VPUNN::Layout::CMAJOR)
        .value("XYZ", VPUNN::Layout::XYZ)
        .value("XZY", VPUNN::Layout::XZY)
        .value("YXZ", VPUNN::Layout::YXZ)
        .value("YZX", VPUNN::Layout::YZX)
        .value("ZXY", VPUNN::Layout::ZXY)
        .value("ZYX", VPUNN::Layout::ZYX)
        .value("INVALID", VPUNN::Layout::INVALID);

    py::enum_<VPUNN::Swizzling>(m, "Swizzling")
        .value("KEY_0", VPUNN::Swizzling::KEY_0)
        .value("KEY_1", VPUNN::Swizzling::KEY_1)
        .value("KEY_2", VPUNN::Swizzling::KEY_2)
        .value("KEY_3", VPUNN::Swizzling::KEY_3)
        .value("KEY_4", VPUNN::Swizzling::KEY_4)
        .value("KEY_5", VPUNN::Swizzling::KEY_5);

    py::enum_<VPUNN::ISIStrategy>(m, "ISIStrategy")
        .value("CLUSTERING", VPUNN::ISIStrategy::CLUSTERING)
        .value("SPLIT_OVER_H", VPUNN::ISIStrategy::SPLIT_OVER_H)
        .value("SPLIT_OVER_K", VPUNN::ISIStrategy::SPLIT_OVER_K);

    // DMA-related enums
    py::enum_<VPUNN::MemoryLocation>(m, "MemoryLocation")
        .value("DRAM", VPUNN::MemoryLocation::DRAM)
        .value("CMX", VPUNN::MemoryLocation::CMX)
        .value("CSRAM", VPUNN::MemoryLocation::CSRAM)
        .value("UPA", VPUNN::MemoryLocation::UPA);

    py::enum_<VPUNN::MemoryDirection>(m, "MemoryDirection")
        .value("DDR2CMX", VPUNN::MemoryDirection::DDR2CMX)
        .value("CMX2CMX", VPUNN::MemoryDirection::CMX2CMX)
        .value("CMX2DDR", VPUNN::MemoryDirection::CMX2DDR)
        .value("DDR2DDR", VPUNN::MemoryDirection::DDR2DDR);

    py::enum_<VPUNN::Num_DMA_Engine>(m, "Num_DMA_Engine")
        .value("Num_Engine_1", VPUNN::Num_DMA_Engine::Num_Engine_1)
        .value("Num_Engine_2", VPUNN::Num_DMA_Engine::Num_Engine_2);

    // VPUTensor class
    py::class_<VPUNN::VPUTensor>(m, "VPUTensor")
        .def(py::init<>())
        .def(py::init<unsigned int, unsigned int, unsigned int, unsigned int, VPUNN::DataType>(),
             py::arg("width"), py::arg("height"), py::arg("channels"), py::arg("batch"), py::arg("dtype"))
        .def(py::init<unsigned int, unsigned int, unsigned int, unsigned int, VPUNN::DataType, VPUNN::Layout>(),
             py::arg("width"), py::arg("height"), py::arg("channels"), py::arg("batch"), py::arg("dtype"), py::arg("layout"))
        .def(py::init<unsigned int, unsigned int, unsigned int, unsigned int, VPUNN::DataType, VPUNN::Layout, bool>(),
             py::arg("width"), py::arg("height"), py::arg("channels"), py::arg("batch"), py::arg("dtype"), py::arg("layout"), py::arg("sparsity"))
        .def("width", &VPUNN::VPUTensor::width)
        .def("height", &VPUNN::VPUTensor::height)
        .def("channels", &VPUNN::VPUTensor::channels)
        .def("batches", &VPUNN::VPUTensor::batches)
        .def("get_dtype", &VPUNN::VPUTensor::get_dtype)
        .def("get_layout", &VPUNN::VPUTensor::get_layout)
        .def("get_sparsity", &VPUNN::VPUTensor::get_sparsity)
        .def("set_sparsity", &VPUNN::VPUTensor::set_sparsity)
        .def("volume", &VPUNN::VPUTensor::volume)
        .def("size", &VPUNN::VPUTensor::size);

    // DPUWorkload structure
    py::class_<VPUNN::DPUWorkload>(m, "DPUWorkload")
        .def(py::init<>())
        .def_readwrite("device", &VPUNN::DPUWorkload::device)
        .def_readwrite("op", &VPUNN::DPUWorkload::op)
        .def_readwrite("inputs", &VPUNN::DPUWorkload::inputs)
        .def_readwrite("outputs", &VPUNN::DPUWorkload::outputs)
        .def_readwrite("kernels", &VPUNN::DPUWorkload::kernels)
        .def_readwrite("strides", &VPUNN::DPUWorkload::strides)
        .def_readwrite("padding", &VPUNN::DPUWorkload::padding)
        .def_readwrite("execution_order", &VPUNN::DPUWorkload::execution_order)
        .def_readwrite("activation_function", &VPUNN::DPUWorkload::activation_function)
        .def_readwrite("act_sparsity", &VPUNN::DPUWorkload::act_sparsity)
        .def_readwrite("weight_sparsity", &VPUNN::DPUWorkload::weight_sparsity)
        .def_readwrite("input_swizzling", &VPUNN::DPUWorkload::input_swizzling)
        .def_readwrite("output_swizzling", &VPUNN::DPUWorkload::output_swizzling)
        .def_readwrite("output_write_tiles", &VPUNN::DPUWorkload::output_write_tiles)
        .def_readwrite("offsets", &VPUNN::DPUWorkload::offsets)
        .def_readwrite("isi_strategy", &VPUNN::DPUWorkload::isi_strategy)
        .def_readwrite("weight_sparsity_enabled", &VPUNN::DPUWorkload::weight_sparsity_enabled)
        .def("get_layer_info", &VPUNN::DPUWorkload::get_layer_info)
        .def("set_layer_info", &VPUNN::DPUWorkload::set_layer_info)
        .def("is_inplace_output", &VPUNN::DPUWorkload::is_inplace_output)
        .def("is_weightless_operation", &VPUNN::DPUWorkload::is_weightless_operation)
        .def("get_weight_type", &VPUNN::DPUWorkload::get_weight_type);

    // DMAWorkload structure (legacy)
    py::class_<VPUNN::DMAWorkload>(m, "DMAWorkload")
        .def(py::init<>())
        .def_readwrite("device", &VPUNN::DMAWorkload::device)
        .def_readwrite("input", &VPUNN::DMAWorkload::input)
        .def_readwrite("output", &VPUNN::DMAWorkload::output)
        .def_readwrite("input_location", &VPUNN::DMAWorkload::input_location)
        .def_readwrite("output_location", &VPUNN::DMAWorkload::output_location)
        .def_readwrite("output_write_tiles", &VPUNN::DMAWorkload::output_write_tiles);

    // DMANNWorkload_NPU27 structure
    py::class_<VPUNN::DMANNWorkload_NPU27>(m, "DMANNWorkload_NPU27")
        .def(py::init<>())
        .def(py::init<VPUNN::VPUDevice, int, int, int, int, int, int, int, int, VPUNN::MemoryDirection>(),
             py::arg("device"), py::arg("num_planes"), py::arg("length"), py::arg("src_width"), py::arg("dst_width"),
             py::arg("src_stride"), py::arg("dst_stride"), py::arg("src_plane_stride"), py::arg("dst_plane_stride"),
             py::arg("transfer_direction"))
        .def(py::init<VPUNN::VPUDevice, int, int, int, int, int, int, int, int, VPUNN::MemoryDirection, std::string>(),
             py::arg("device"), py::arg("num_planes"), py::arg("length"), py::arg("src_width"), py::arg("dst_width"),
             py::arg("src_stride"), py::arg("dst_stride"), py::arg("src_plane_stride"), py::arg("dst_plane_stride"),
             py::arg("transfer_direction"), py::arg("loc_name"))
        .def_readwrite("device", &VPUNN::DMANNWorkload_NPU27::device)
        .def_readwrite("num_planes", &VPUNN::DMANNWorkload_NPU27::num_planes)
        .def_readwrite("length", &VPUNN::DMANNWorkload_NPU27::length)
        .def_readwrite("src_width", &VPUNN::DMANNWorkload_NPU27::src_width)
        .def_readwrite("dst_width", &VPUNN::DMANNWorkload_NPU27::dst_width)
        .def_readwrite("src_stride", &VPUNN::DMANNWorkload_NPU27::src_stride)
        .def_readwrite("dst_stride", &VPUNN::DMANNWorkload_NPU27::dst_stride)
        .def_readwrite("src_plane_stride", &VPUNN::DMANNWorkload_NPU27::src_plane_stride)
        .def_readwrite("dst_plane_stride", &VPUNN::DMANNWorkload_NPU27::dst_plane_stride)
        .def_readwrite("transfer_direction", &VPUNN::DMANNWorkload_NPU27::transfer_direction)
        .def_readwrite("loc_name", &VPUNN::DMANNWorkload_NPU27::loc_name)
        .def("getAccessedBytes", &VPUNN::DMANNWorkload_NPU27::getAccessedBytes);

    // DMANNWorkload_NPU40_50 structure
    py::class_<VPUNN::DMANNWorkload_NPU40_50>(m, "DMANNWorkload_NPU40_50")
        .def(py::init<>())
        .def(py::init<VPUNN::VPUDevice>(), py::arg("device"))
        .def_readwrite("device", &VPUNN::DMANNWorkload_NPU40_50::device)
        .def_readwrite("src_width", &VPUNN::DMANNWorkload_NPU40_50::src_width)
        .def_readwrite("dst_width", &VPUNN::DMANNWorkload_NPU40_50::dst_width)
        .def_readwrite("num_dim", &VPUNN::DMANNWorkload_NPU40_50::num_dim)
        .def_readwrite("num_engine", &VPUNN::DMANNWorkload_NPU40_50::num_engine)
        .def_readwrite("transfer_direction", &VPUNN::DMANNWorkload_NPU40_50::transfer_direction)
        .def_readwrite("loc_name", &VPUNN::DMANNWorkload_NPU40_50::loc_name);

    // VPUCostModel class - only DPUMsg method
    py::class_<VPUNN::VPUCostModel>(m, "VPUCostModel")
        .def(py::init<>())
        .def(py::init<const std::string&>(), py::arg("model_path"))
        .def("DPUMsg", &VPUNN::VPUCostModel::DPUMsg, 
             "Calculate DPU cost and return cycles with error message",
             py::arg("workload"))
       .def("DPU", static_cast<VPUNN::CyclesInterfaceType (VPUNN::VPUCostModel::*)(VPUNN::DPUWorkload) const>(&VPUNN::VPUCostModel::DPU),
           "Calculate DPU cost and return cycles only",
           py::arg("workload"))
       ;

    // DMACostModel classes for different DMA workload types
    py::class_<VPUNN::DMACostModel<VPUNN::DMANNWorkload_NPU27>>(m, "DMACostModel_NPU27")
        .def(py::init<const std::string&>(), py::arg("model_path"))
        .def("computeCycles", 
             static_cast<VPUNN::CyclesInterfaceType (VPUNN::DMACostModel<VPUNN::DMANNWorkload_NPU27>::*)(const VPUNN::DMANNWorkload_NPU27&)>(&VPUNN::DMACostModel<VPUNN::DMANNWorkload_NPU27>::computeCycles),
             "Calculate DMA cycles for NPU27 workload",
             py::arg("workload"))
        .def("computeCyclesMsg", &VPUNN::DMACostModel<VPUNN::DMANNWorkload_NPU27>::computeCyclesMsg,
             "Calculate DMA cycles with error message for NPU27 workload",
             py::arg("workload"));

    py::class_<VPUNN::DMACostModel<VPUNN::DMANNWorkload_NPU40_50>>(m, "DMACostModel_NPU40_50")
        .def(py::init<const std::string&>(), py::arg("model_path"))
        .def("computeCycles", 
             static_cast<VPUNN::CyclesInterfaceType (VPUNN::DMACostModel<VPUNN::DMANNWorkload_NPU40_50>::*)(const VPUNN::DMANNWorkload_NPU40_50&)>(&VPUNN::DMACostModel<VPUNN::DMANNWorkload_NPU40_50>::computeCycles),
             "Calculate DMA cycles for NPU40/50 workload",
             py::arg("workload"))
        .def("computeCyclesMsg", &VPUNN::DMACostModel<VPUNN::DMANNWorkload_NPU40_50>::computeCyclesMsg,
             "Calculate DMA cycles with error message for NPU40/50 workload",
             py::arg("workload"));

}
