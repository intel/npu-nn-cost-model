// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.
#include "vpu/validation/interface_valid_values.h"

#include <gtest/gtest.h>

#include "common/common_helpers.h"
#include "vpu/validation/dpu_operations_validator.h"  //for mock

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>

namespace VPUNN_unit_tests {
using namespace VPUNN;

class IntfDeviceValidValuesTest : public ::testing::Test {
public:
    // dummy class to differentiate constructor
    class WeightsAlignment_class {};

protected:
    void SetUp() override {
    }

    class DeviceValidValuesMock : public VPUNN::IDeviceValidValues {
    public:
        const VPUNN::OperationsBehaviour specific_behaviours{};  ///< known behaviors

        /// @brief non public constructor for initializing the reference
        DeviceValidValuesMock(const int input_heigth_start_factor_SOH_)
                : VPUNN::IDeviceValidValues(specific_behaviours,
                                            valid_execution_order_map_default,  //
                                            valid_swizzlings_def,               //
                                            valid_layouts_def,                  //
                                            devices_def,                        //
                                            cmx_KB_sizes_def,                   //
                                            output_write_tile_options_def,      //
                                            isi_stategy_options_def,            //
                                            weigths_alignment_B_def,            //
                                            input_heigth_start_factor_SOH_,     // special
                                            valid_datatypes_map_default,        //
                                            valid_operations_default,           //
                                            alignement_size_bytes_def,          //
                                            out_innermost_dim_alignment_def,    //
                                            legacy_samples_alignment_weights_def) {
        }
        DeviceValidValuesMock()
                : VPUNN::IDeviceValidValues(specific_behaviours,
                                            valid_execution_order_map_default,  //
                                            valid_swizzlings_def,               //
                                            valid_layouts_def,                  //
                                            devices_def,                        //
                                            cmx_KB_sizes_def,                   //
                                            output_write_tile_options_def,      //
                                            isi_stategy_options_def,            //
                                            weigths_alignment_B_def,            //
                                            input_heigth_start_factor_SOH_def,  // special
                                            valid_datatypes_map_default,        //
                                            valid_operations_default,           //
                                            alignement_size_bytes_def,          //
                                            out_innermost_dim_alignment_def,    //
                                            legacy_samples_alignment_weights_def) {
        }

        /// constructor that helps to test memory behavior for different  weights alignment
        /// @param weigths_alignment_B_ the weights alignment to be used
        /// @param WeightsAlignment_class& just to differentiate from other constructors
        DeviceValidValuesMock(const int weights_alignment_B_, const WeightsAlignment_class&)
                : VPUNN::IDeviceValidValues(specific_behaviours,
                                            valid_execution_order_map_default,  //
                                            valid_swizzlings_def,               //
                                            valid_layouts_def,                  //
                                            devices_def,                        //
                                            cmx_KB_sizes_def,                   //
                                            output_write_tile_options_def,      //
                                            isi_stategy_options_def,            //
                                            weights_alignment_B_,               //
                                            input_heigth_start_factor_SOH_def,  // special
                                            valid_datatypes_map_default,        //
                                            valid_operations_default,           //
                                            alignement_size_bytes_def,          //
                                            out_innermost_dim_alignment_def,    //
                                            legacy_samples_alignment_weights_def) {
        }

        // const VPUNN::Channels& get_output_channels_range(const VPUNN::DPUOperation&) const override {
        //     return channels_1;
        // };

        // const VPUNN::Channels& get_input_channels_range(const VPUNN::DPUOperation&) const override {
        //     return channels_1;
        // };

        MultiSmartRanges get_output_channels_restriction(const DPUOperation&) const override {
            return channels_1_range;  // nothing
        }
        MultiSmartRanges get_input_channels_restriction(const DPUOperation&) const override {
            return channels_1_range;  // nothing
        }

        MultiSmartRanges get_batch_restrictions() const override {
            return batch_restrictions;
        }

        // const Values<DataType>& get_input_valid_datatypes(const DPUOperation&) const override {
        //     return valid_datatypes_subset;
        // }

        // const Values<DataType>& get_output_valid_datatypes(const DPUOperation&) const override {
        //     return valid_datatypes_subset;
        // }

        // const Values<DataType>& get_weights_valid_datatypes(const DPUOperation&) const override {
        //     return valid_datatypes_subset;
        // }

        VPUNN::Layout adapt_device_comaptible_tensor_layout(VPUNN::Layout layout) const override {
            return layout;
        };

        VPUNN::Swizzling adapt_device_comaptible_swizzling(VPUNN::Swizzling swizz) const override {
            return swizz;
        };

        const VPUNN::Channels channels_1{1};
        const MultiSmartRanges channels_1_range{{SmartRanges{1, 1, 1}}};

        const MultiSmartRanges batch_restrictions{{SmartRanges(1, SmartRanges::max_limit)}};

        int get_input_heigth_start_factor_SOH() const {
            return get_spatial_range_start_factor_HW();
        }

        // setup content

        inline static const Values<ExecutionMode> valid_execution_order_def{
                ExecutionMode::CUBOID_4x16,
                ExecutionMode::CUBOID_8x16,
                ExecutionMode::CUBOID_16x16,

        };  // 4x1, 16x1, 4x4
        inline static const Values<Swizzling> valid_swizzlings_def{Swizzling::KEY_0, Swizzling::KEY_1,
                                                                   Swizzling::KEY_2, Swizzling::KEY_3,
                                                                   Swizzling::KEY_4, Swizzling::KEY_5};
        inline static const Values<Layout> valid_layouts_def{
                Layout::ZXY /*default one, ZMAJOR like*/,
                Layout::XYZ,
                Layout::XZY,
                Layout::YXZ,
                Layout::YZX,
                Layout::ZYX,
        };
        inline static const Values<VPUDevice> devices_def{
                VPUDevice::VPU_2_7,
        };

        inline static const std::unordered_map<VPUDevice, int> cmx_KB_sizes_def{
                {devices_def[0], (2 * 1024 * 100) / 100}};  // memory increased with 0%

        inline static const Values<int> output_write_tile_options_def{1, 2};
        inline static const Values<ISIStrategy> isi_stategy_options_def{
                ISIStrategy::CLUSTERING,
                ISIStrategy::SPLIT_OVER_H,
                ISIStrategy::SPLIT_OVER_K,
        };

        inline static const int weigths_alignment_B_def{16};
        inline static const bool legacy_samples_alignment_weights_def{false};  // no alignment (samples)
        inline static const int out_innermost_dim_alignment_def{1};  // bytes
        inline static const int input_heigth_start_factor_SOH_def{1};

        inline static const int alignement_size_bytes_def{16 * 1024};

        inline static const Values<DataType> valid_datatypes_subset{
                {DataType::UINT8, DataType::INT8, DataType::FLOAT16, DataType::BFLOAT16}};

        inline static const IDeviceValidValues::ValidDatatypes valid_datatypes_map_default{
                // valid data types based on operations
                {
                        {Operation::CONVOLUTION, valid_datatypes_subset},     //
                        {Operation::DW_CONVOLUTION, valid_datatypes_subset},  //
                        {Operation::CM_CONVOLUTION, valid_datatypes_subset},  //
                        {Operation::ELTWISE, valid_datatypes_subset},         //
                        {Operation::MAXPOOL, valid_datatypes_subset},         //
                },
                {
                        {Operation::CONVOLUTION, valid_datatypes_subset},     //
                        {Operation::DW_CONVOLUTION, valid_datatypes_subset},  //
                        {Operation::CM_CONVOLUTION, valid_datatypes_subset},  //
                        {Operation::ELTWISE, valid_datatypes_subset},         //
                        {Operation::MAXPOOL, valid_datatypes_subset},         //
                },
                {
                        {Operation::CONVOLUTION, valid_datatypes_subset},     //
                        {Operation::DW_CONVOLUTION, valid_datatypes_subset},  //
                        {Operation::CM_CONVOLUTION, valid_datatypes_subset},  //
                        {Operation::ELTWISE, valid_datatypes_subset},         //
                        {Operation::MAXPOOL, valid_datatypes_subset},         //
                },
        };
        inline static const Values<Operation> valid_operations_default{
                Operation::CONVOLUTION,     //
                Operation::DW_CONVOLUTION,  //
                Operation::CM_CONVOLUTION,  //
                Operation::ELTWISE,         //
                Operation::MAXPOOL,         //
        };

        inline static const IDeviceValidValues::ValidExecutionModes valid_execution_order_map_default{
                // valid execution modes based on operations
                {
                        {Operation::CONVOLUTION, valid_execution_order_def},         //
                        {Operation::DW_CONVOLUTION, {ExecutionMode::CUBOID_16x16}},  //
                        {Operation::CM_CONVOLUTION, {ExecutionMode::CUBOID_16x16}},  //
                        {Operation::MAXPOOL, {ExecutionMode::CUBOID_16x16}},         //
                        {Operation::ELTWISE, {ExecutionMode::CUBOID_8x16}},          //
                }};
    };

private:
};

TEST_F(IntfDeviceValidValuesTest, Input1_memory_computation_test) {
    auto mk_wl = [](const Operation op, const DataType dtype, const unsigned int ch_in, const unsigned int ch_out,
                    const unsigned int k_w, const unsigned int k_h, bool weights_sparsity = false) -> DPUWorkload {
        const HaloWorkload zeroHalo;
        const SEPModeInfo sepInfo{};

        // this test doesn't depend on device, any device can be used here
        return VPUNN::DPUWorkload{
                VPUDevice::VPU_2_7,
                op,
                {VPUNN::VPUTensor(27, 18, ch_in, 1, DataType::UINT8)},   // input dimensions
                {VPUNN::VPUTensor(27, 18, ch_out, 1, DataType::UINT8)},  // output dimensions
                {k_w, k_h},                                              // kernels
                {1, 1},                                                  // strides
                {0, 0, 0, 0},                                            // padding
                VPUNN::ExecutionMode::CUBOID_16x16,                      // execution mode
                VPUNN::ActivationFunction::NONE,                         // activation
                0.0F,                                                    // act_sparsity
                0.0F,                                                    // weight_sparsity
                {swz_def, swz_def},                                      // input_swizzling
                {swz_def},                                               // output_swizzling
                1,                                                       // output_write_tiles
                {0, 0, 0, 0},                                            // offsets
                VPUNN::ISIStrategy::CLUSTERING,                          // isi_strategy
                weights_sparsity,                                        // weight_sparsity_enabled
                zeroHalo,                                                // halo
                sepInfo,                                                 // sep
                dtype                                                    // datatype for weights
        };
    };

    struct TestInput {
        VPUNN::DPUWorkload wl;  // the wl for which we compute memory
        int weights_alignment;  // weights alignment for this test
    };

    struct TestExpectation {
        long long in1_mem_size_exp;  // memory expected; it depends on test input
        bool expect_throw = false;  // whether we expect an exception or not, exemple: if operation is ELTWISE and dtype
                                    // is INT1, INT2, INT4 will throw because these dtypes are not supported for input0
                                    // and input1 is just a copy of input0
    };

    struct TestCase {
        TestInput t_in;
        TestExpectation t_exp;
    };

    using TestsVector = std::vector<TestCase>;

    auto test_info = [](const DPUWorkload& wl) -> std::string {
        std::string info =
                " device: \t" + VPUDevice_ToText.at(static_cast<int>(wl.device)) + " Operation: \t" +
                Operation_ToText.at(static_cast<int>(wl.op)) + " input1 dtype: \t" +
                (wl.weight_type.has_value() ? DataType_ToText.at(static_cast<int>(wl.weight_type.value())) : "Same") +
                " channels: \t" + std::to_string(wl.inputs[0].z()) + " weight_sparsity_enabled: \t" +
                (wl.weight_sparsity_enabled ? "true" : "false");
        return info;
    };

    // this lambda function verify if the input1 memory is computed correctly for different workloads
    auto verify_input1_memory = [&test_info](const TestsVector& tests) {
        int i = 1;  // index of test cases
        for (const auto& t : tests) {
            VPUNN::DPUWorkload wl_ref{t.t_in.wl};
            std::cout << "Test case "
                      << " " << i << ": " << test_info(wl_ref) << "\n";

            {
                long long mem = 0;
                auto wl{wl_ref};

                DeviceValidValuesMock config{t.t_in.weights_alignment, WeightsAlignment_class()};
                const DPUOperation op(wl, config);
                const IOperationDynamicConstraints& operation_behaviour{config.get_specific_behaviour(op.operation)};
                if (t.t_exp.expect_throw) {
                    EXPECT_THROW(mem = operation_behaviour.input_1_contiguous_size_bytes(config, op),
                                 std::runtime_error)
                            << "Test : " << i << " ," << wl << std::endl;
                    i++;
                } else {
                    EXPECT_NO_THROW(mem = operation_behaviour.input_1_contiguous_size_bytes(config, op))
                            << "Test : " << i << " ," << wl << std::endl;
                    EXPECT_EQ(mem, t.t_exp.in1_mem_size_exp)
                            << "Test case "
                            << " " << i << ": " << test_info(std::move(wl_ref))
                            << " weights alignment: " << t.t_in.weights_alignment << "\n";
                }
            }
            i++;
        }
    };

    {
        const TestsVector tests_dw_conv = {
                // clang-format off
            /************************************************ TABLE HEADER ********************************************************/  
            /*  ||                 workload              || weights alignment || input1 memory expected ||    */
            {{mk_wl(Operation::DW_CONVOLUTION, DataType::UINT1,  1, 1, 3, 3), 1},{18}}, //1*1*(3*3)*1   /* 9ch and UINT1 => 2B for channels should be aligned to 1B => 2B*/ + 1*16 /*weights table*/
            {{mk_wl(Operation::DW_CONVOLUTION, DataType::UINT1,  1, 2, 3, 3), 1},{36}}, //1*1*(3*3)*2   /* 9ch and UINT1 => 2B for channels should be aligned to 1B => 2B*/ + 2*16 /*weights table*/
            {{mk_wl(Operation::DW_CONVOLUTION, DataType::UINT1,  1, 1, 3, 1), 1},{17}}, //1*1*(3*1)*1   /* 3ch and UINT1 => 1B for channels should be aligned to 1B => 1B*/ + 1*16 /*weights table*/
            {{mk_wl(Operation::DW_CONVOLUTION, DataType::UINT1,  1, 2, 3, 1), 1},{34}}, //1*1*(3*1)*2   /* 3ch and UINT1 => 1B for channels should be aligned to 1B => 1B*/ + 2*16 /*weights table*/

            {{mk_wl(Operation::DW_CONVOLUTION, DataType::UINT1,  1, 1, 3, 3), 9},{25}}, //1*1*(3*3)*1   /* 9ch and UINT1  => 2B for channels  should be aligned to 9B => 9B  */ + 1*16 /*weights table*/
            {{mk_wl(Operation::DW_CONVOLUTION, DataType::UINT1,  1, 2, 3, 3), 9},{50}}, //1*1*(3*3)*2   /* 9ch and UINT1  => 2B for channels  should be aligned to 9B => 9B  */ + 2*16 /*weights table*/
            {{mk_wl(Operation::DW_CONVOLUTION, DataType::UINT2,  1, 1, 3, 3), 9},{25}}, //1*1*(3*3)*1   /* 9ch and UINT2  => 3B for channels  should be aligned to 9B => 9B  */ + 1*16 /*weights table*/
            {{mk_wl(Operation::DW_CONVOLUTION, DataType::UINT4,  1, 1, 3, 3), 9},{25}}, //1*1*(3*3)*1   /* 9ch and UINT4  => 5B for channels  should be aligned to 9B => 9B  */ + 1*16 /*weights table*/
            {{mk_wl(Operation::DW_CONVOLUTION, DataType::UINT4,  1, 1, 3, 3), 8},{24}}, //1*1*(3*3)*1   /* 9ch and UINT4  => 5B for channels  should be aligned to 8B => 8B  */ + 1*16 /*weights table*/
            {{mk_wl(Operation::DW_CONVOLUTION, DataType::UINT8,  1, 1, 3, 3), 9},{25}}, //1*1*(3*3)*1   /* 9ch and UINT8  => 9B for channels  should be aligned to 9B => 9B  */ + 1*16 /*weights table*/
            {{mk_wl(Operation::DW_CONVOLUTION, DataType::UINT16, 1, 1, 3, 3), 9},{34}}, //1*1*(3*3)*1*2 /* 9ch and UINT16 => 18B for channels should be aligned to 9B => 18B */ + 1*16 /*weights table*/

            {{mk_wl(Operation::DW_CONVOLUTION, DataType::UINT1,  1, 1, 3, 3), 5},{21}}, //1*1*(3*3)*1   /* 9ch and UINT1  => 2B for channels  should be aligned to 5B => 5B */  + 1*16 /*weights table*/
            {{mk_wl(Operation::DW_CONVOLUTION, DataType::UINT2,  1, 1, 3, 3), 5},{21}}, //1*1*(3*3)*1   /* 9ch and UINT2  => 3B for channels  should be aligned to 5B => 5B */  + 1*16 /*weights table*/
            {{mk_wl(Operation::DW_CONVOLUTION, DataType::UINT4,  1, 1, 3, 3), 5},{21}}, //1*1*(3*3)*1   /* 9ch and UINT4  => 5B for channels  should be aligned to 5B => 5B */  + 1*16 /*weights table*/
            {{mk_wl(Operation::DW_CONVOLUTION, DataType::UINT8,  1, 1, 3, 3), 5},{26}}, //1*1*(3*3)*1   /* 9ch and UINT8  => 9B for channels  should be aligned to 5B => 10B */ + 1*16 /*weights table*/
            {{mk_wl(Operation::DW_CONVOLUTION, DataType::UINT16, 1, 1, 3, 3), 5},{36}}, //1*1*(3*3)*1*2 /* 9ch and UINT16 => 18B for channels should be aligned to 5B => 20B */ + 1*16 /*weights table*/
            
            {{mk_wl(Operation::DW_CONVOLUTION, DataType::UINT1,  1, 1, 1, 3), 7},{23}}, //1*1*(1*3)*1   /* 3ch and UINT1  => 1B for channels should be aligned to 7B => 7B */ + 1*16 /*weights table*/
            {{mk_wl(Operation::DW_CONVOLUTION, DataType::UINT2,  1, 1, 1, 3), 7},{23}}, //1*1*(1*3)*1   /* 3ch and UINT2  => 1B for channels should be aligned to 7B => 7B */ + 1*16 /*weights table*/
            {{mk_wl(Operation::DW_CONVOLUTION, DataType::UINT4,  1, 1, 1, 3), 7},{23}}, //1*1*(1*3)*1   /* 3ch and UINT4  => 2B for channels should be aligned to 7B => 7B */ + 1*16 /*weights table*/
            {{mk_wl(Operation::DW_CONVOLUTION, DataType::UINT8,  1, 1, 1, 3), 7},{23}}, //1*1*(1*3)*1   /* 3ch and UINT8  => 3B for channels should be aligned to 7B => 7B */ + 1*16 /*weights table*/
            {{mk_wl(Operation::DW_CONVOLUTION, DataType::UINT16, 1, 1, 1, 3), 7},{23}}, //1*1*(1*3)*1*2 /* 3ch and UINT16 => 6B for channels should be aligned to 7B => 7B */ + 1*16 /*weights table*/
            
            {{mk_wl(Operation::DW_CONVOLUTION, DataType::UINT1,  1, 1, 5, 5), 12},{28}}, //1*1*(5*5)*1   /* 25ch and UINT1  => 4B  for channels should be aligned to 12B => 12B */ + 1*16 /*weights table*/
            {{mk_wl(Operation::DW_CONVOLUTION, DataType::UINT2,  1, 1, 5, 5), 12},{28}}, //1*1*(5*5)*1   /* 25ch and UINT2  => 7B  for channels should be aligned to 12B => 12B */ + 1*16 /*weights table*/
            {{mk_wl(Operation::DW_CONVOLUTION, DataType::UINT4,  1, 1, 5, 5), 12},{40}}, //1*1*(5*5)*1   /* 25ch and UINT4  => 13B for channels should be aligned to 12B => 24B */ + 1*16 /*weights table*/
            {{mk_wl(Operation::DW_CONVOLUTION, DataType::UINT8,  1, 1, 5, 5), 12},{52}}, //1*1*(5*5)*1   /* 25ch and UINT8  => 25B for channels should be aligned to 12B => 36B */ + 1*16 /*weights table*/
            {{mk_wl(Operation::DW_CONVOLUTION, DataType::UINT16, 1, 1, 5, 5), 12},{76}}, //1*1*(5*5)*1*2 /* 25ch and UINT16 => 50B for channels should be aligned to 12B => 60B */ + 1*16 /*weights table*/
                // clang-format on
        };

        verify_input1_memory(tests_dw_conv);
    }

    {
        const TestsVector tests_conv = {
                // clang-format off
            /************************************************ TABLE HEADER ********************************************************/
            /*  || workload || input1 memory expected ||    */
            {{mk_wl(Operation::CONVOLUTION, DataType::UINT1,  1, 1, 3, 3), 5},{21}}, //1*1*(3*3*1)*1   /* 9ch and UINT1  => 2B  for channels should be aligned to 5B => 5B */  + 1*16  /*weights table*/
            {{mk_wl(Operation::CONVOLUTION, DataType::UINT2,  1, 1, 3, 3), 5},{21}}, //1*1*(3*3*1)*1   /* 9ch and UINT2  => 3B  for channels should be aligned to 5B => 5B */  + 1*16  /*weights table*/
            {{mk_wl(Operation::CONVOLUTION, DataType::UINT4,  1, 1, 3, 3), 5},{21}}, //1*1*(3*3*1)*1   /* 9ch and UINT4  => 5B  for channels should be aligned to 5B => 5B */  + 1*16  /*weights table*/
            {{mk_wl(Operation::CONVOLUTION, DataType::UINT8,  1, 1, 3, 3), 5},{26}}, //1*1*(3*3*1)*1   /* 9ch and UINT8  => 9B  for channels should be aligned to 5B => 10B */ + 1*16 /*weights table*/
            {{mk_wl(Operation::CONVOLUTION, DataType::UINT16, 1, 1, 3, 3), 5},{36}}, //1*1*(3*3*1)*1*2 /* 9ch and UINT16 => 18B for channels should be aligned to 5B => 20B */ + 1*16 /*weights table*/
                                                              
            {{mk_wl(Operation::CONVOLUTION, DataType::UINT1,  1, 1, 1, 3), 2},{18}}, //1*1*(1*3*1)*1   /* 3ch and UINT1  => 1B for channels should be aligned to 2B => 2B */ + 1*16 /*weights table*/
            {{mk_wl(Operation::CONVOLUTION, DataType::UINT2,  1, 1, 1, 3), 2},{18}}, //1*1*(1*3*1)*1   /* 3ch and UINT2  => 1B for channels should be aligned to 2B => 2B */ + 1*16 /*weights table*/
            {{mk_wl(Operation::CONVOLUTION, DataType::UINT4,  1, 1, 1, 3), 2},{18}}, //1*1*(1*3*1)*1   /* 3ch and UINT4  => 2B for channels should be aligned to 2B => 2B */ + 1*16 /*weights table*/
            {{mk_wl(Operation::CONVOLUTION, DataType::UINT8,  1, 1, 1, 3), 2},{20}}, //1*1*(1*3*1)*1   /* 3ch and UINT8  => 3B for channels should be aligned to 2B => 4B */ + 1*16 /*weights table*/
            {{mk_wl(Operation::CONVOLUTION, DataType::UINT16, 1, 1, 1, 3), 2},{22}}, //1*1*(1*3*1)*1*2 /* 3ch and UINT16 => 6B for channels should be aligned to 2B => 6B */ + 1*16 /*weights table*/
                                                            
            {{mk_wl(Operation::CONVOLUTION, DataType::UINT1,  1, 1, 5, 3), 17},{33}}, //1*1*(5*3*1)*1   /* 15ch and UINT1  => 2B  for channels should be aligned to 17B => 17B */ + 1*16 /*weights table*/
            {{mk_wl(Operation::CONVOLUTION, DataType::UINT2,  1, 1, 5, 3), 17},{33}}, //1*1*(5*3*1)*1   /* 15ch and UINT2  => 4B  for channels should be aligned to 17B => 17B */ + 1*16 /*weights table*/
            {{mk_wl(Operation::CONVOLUTION, DataType::UINT4,  1, 1, 5, 3), 17},{33}}, //1*1*(5*3*1)*1   /* 15ch and UINT4  => 8B  for channels should be aligned to 17B => 17B */ + 1*16 /*weights table*/
            {{mk_wl(Operation::CONVOLUTION, DataType::UINT8,  1, 1, 5, 3), 17},{33}}, //1*1*(5*3*1)*1   /* 15ch and UINT8  => 15B for channels should be aligned to 17B => 17B */ + 1*16 /*weights table*/
            {{mk_wl(Operation::CONVOLUTION, DataType::UINT16, 1, 1, 5, 3), 17},{50}}, //1*1*(5*3*1)*1*2 /* 15ch and UINT16 => 30B for channels should be aligned to 17B => 34B */ + 1*16 /*weights table*/
                                                           
            //WEIGHT SPARSITY = TRUE                     
            {{mk_wl(Operation::CONVOLUTION, DataType::UINT1,  1, 1, 7, 4, true), 13},{45}}, //1*1*(7*4*1)*1   /* 28ch and UINT1  => 4B  for channels should be aligned to 13B => 13B */ + 1*16 /*weights table*/  + 16*1 /*sparsity map*/
            {{mk_wl(Operation::CONVOLUTION, DataType::UINT2,  1, 1, 7, 4, true), 13},{45}}, //1*1*(7*4*1)*1   /* 28ch and UINT2  => 7B  for channels should be aligned to 13B => 13B */ + 1*16 /*weights table*/  + 16*1 /*sparsity map*/
            {{mk_wl(Operation::CONVOLUTION, DataType::UINT4,  1, 1, 7, 4, true), 13},{58}}, //1*1*(7*4*1)*1   /* 28ch and UINT4  => 14B for channels should be aligned to 13B => 26B */ + 1*16 /*weights table*/  + 16*1 /*sparsity map*/
            {{mk_wl(Operation::CONVOLUTION, DataType::UINT8,  1, 1, 7, 4, true), 13},{71}}, //1*1*(7*4*1)*1   /* 28ch and UINT8  => 28B for channels should be aligned to 13B => 39B */ + 1*16 /*weights table*/  + 16*1 /*sparsity map*/
            {{mk_wl(Operation::CONVOLUTION, DataType::UINT16, 1, 1, 7, 4, true), 13},{97}}, //1*1*(7*4*1)*1*2 /* 28ch and UINT16 => 56B for channels should be aligned to 13B => 65B */ + 1*16 /*weights table*/  + 16*1 /*sparsity map*/

                // clang-format on
        };

        verify_input1_memory(tests_conv);
    }

    {
        const TestsVector tests_cm_conv = {
                // clang-format off
            /************************************************ TABLE HEADER ********************************************************/
            /*  || workload || input1 memory expected ||    */
            {{mk_wl(Operation::CM_CONVOLUTION, DataType::UINT1,  10, 1, 3, 3), 11},{38}}, //1*1*(3*3*10)*1 /* 90ch and dtype UINT1  => 12B  for channels should be aligned to 11B =>22B  */ + 16*1 /*weights table*/
            {{mk_wl(Operation::CM_CONVOLUTION, DataType::UINT2,  10, 1, 3, 3), 11},{49}}, //1*1*(3*3*10)*1 /* 90ch and dtype UINT2  => 23B  for channels should be aligned to 11B =>33B  */ + 16*1 /*weights table*/
            {{mk_wl(Operation::CM_CONVOLUTION, DataType::UINT4,  10, 1, 3, 3), 11},{71}}, //1*1*(3*3*10)*1 /* 90ch and dtype UINT4  => 45B  for channels should be aligned to 11B =>55B  */ + 16*1 /*weights table*/
            {{mk_wl(Operation::CM_CONVOLUTION, DataType::UINT8,  10, 1, 3, 3), 11},{115}},//1*1*(3*3*10)*1 /* 90ch and dtype UINT8  => 90B  for channels should be aligned to 11B =>99B  */ + 16*1 /*weights table*/
            {{mk_wl(Operation::CM_CONVOLUTION, DataType::UINT16, 10, 1, 3, 3), 11},{203}},//1*1*(3*3*10)*1 /* 90ch and dtype UINT16 => 180B for channels should be aligned to 11B =>187B */ + 16*1 /*weights table*/

            {{mk_wl(Operation::CM_CONVOLUTION, DataType::UINT1,  1, 1, 1, 3), 5},{21}}, //1*1*(1*3*1)*1 /* 3ch and dtype UINT1  => 1B for channels should be aligned to 5B =>5B  */ + 16*1 /*weights table*/
            {{mk_wl(Operation::CM_CONVOLUTION, DataType::UINT2,  1, 1, 1, 3), 5},{21}}, //1*1*(1*3*1)*1 /* 3ch and dtype UINT2  => 1B for channels should be aligned to 5B =>5B  */ + 16*1 /*weights table*/
            {{mk_wl(Operation::CM_CONVOLUTION, DataType::UINT4,  1, 1, 1, 3), 5},{21}}, //1*1*(1*3*1)*1 /* 3ch and dtype UINT4  => 2B for channels should be aligned to 5B =>5B  */ + 16*1 /*weights table*/
            {{mk_wl(Operation::CM_CONVOLUTION, DataType::UINT8,  1, 1, 1, 3), 5},{21}}, //1*1*(1*3*1)*1 /* 3ch and dtype UINT8  => 3B for channels should be aligned to 5B =>5B  */ + 16*1 /*weights table*/
            {{mk_wl(Operation::CM_CONVOLUTION, DataType::UINT16, 1, 1, 1, 3), 5},{26}}, //1*1*(1*3*1)*1 /* 3ch and dtype UINT16 => 6B for channels should be aligned to 5B =>10B */ + 16*1 /*weights table*/

            {{mk_wl(Operation::CM_CONVOLUTION, DataType::UINT1,  1, 1, 7, 7), 9},{25}}, //1*1*(1*7*7)*1 /* 49ch and dtype UINT1  => 7B  for channels should be aligned to 9B =>9B  */ + 16*1 /*weights table*/
            {{mk_wl(Operation::CM_CONVOLUTION, DataType::UINT2,  1, 1, 7, 7), 9},{34}}, //1*1*(1*7*7)*1 /* 49ch and dtype UINT2  => 13B for channels should be aligned to 9B =>18B */ + 16*1 /*weights table*/
            {{mk_wl(Operation::CM_CONVOLUTION, DataType::UINT4,  1, 1, 7, 7), 9},{43}}, //1*1*(1*7*7)*1 /* 49ch and dtype UINT4  => 25B for channels should be aligned to 9B =>27B */ + 16*1 /*weights table*/
            {{mk_wl(Operation::CM_CONVOLUTION, DataType::UINT8,  1, 1, 7, 7), 9},{70}}, //1*1*(1*7*7)*1 /* 49ch and dtype UINT8  => 49B for channels should be aligned to 9B =>54B */ + 16*1 /*weights table*/
            {{mk_wl(Operation::CM_CONVOLUTION, DataType::UINT16, 1, 1, 7, 7), 9},{115}},//1*1*(1*7*7)*1 /* 49ch and dtype UINT16 => 98B for channels should be aligned to 9B =>99B */ + 16*1 /*weights table*/

                // clang-format on
        };

        verify_input1_memory(tests_cm_conv);
    }
}

/// Test make list method
TEST_F(IntfDeviceValidValuesTest, makeListTest) {
    DeviceValidValuesMock dut;

    {
        const auto res{dut.makeList(1, 10)};
        VPUNN::Values<int> exp{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

        ASSERT_EQ(res.size(), exp.size());

        for (unsigned int i = 0; i < exp.size(); ++i) {
            EXPECT_EQ(res[i], exp[i]) << "i: " << i;
        }
    }
    {
        const auto res{dut.makeList(1, 10, 100)};
        VPUNN::Values<int> exp{100, 200, 300, 400, 500, 600, 700, 800, 900, 1000};

        ASSERT_EQ(res.size(), exp.size());

        for (unsigned int i = 0; i < exp.size(); ++i) {
            EXPECT_EQ(res[i], exp[i]) << "i: " << i;
        }
    }

    {
        const auto res{dut.makeList(25, 25, 100)};
        VPUNN::Values<int> exp{2500};

        ASSERT_EQ(res.size(), exp.size());

        for (unsigned int i = 0; i < exp.size(); ++i) {
            EXPECT_EQ(res[i], exp[i]) << "i: " << i;
        }
    }
}

/// Test default swizzling for devices
TEST_F(IntfDeviceValidValuesTest, defaultSwizz) {
    VPUNN::DPUWorkload wl2_0 = {
            VPUNN::VPUDevice::VPU_2_0,
            VPUNN::Operation::CONVOLUTION,
            {VPUNN::VPUTensor(16, 16, 64, 1, VPUNN::DataType::INT8, VPUNN::Layout::ZMAJOR)},  // input dimensions
            {VPUNN::VPUTensor(16, 16, 64, 1, VPUNN::DataType::INT8, VPUNN::Layout::ZMAJOR)},  // output dimensions
            {1, 1},                                                                           // kernels
            {1, 1},                                                                           // strides
            {0, 0, 0, 0},                                                                     // padding
            VPUNN::ExecutionMode::VECTOR                                                      // execution mode
    };

    VPUNN::DPUWorkload wl2_7 = {
            VPUNN::VPUDevice::VPU_2_7,
            VPUNN::Operation::ELTWISE,
            {VPUNN::VPUTensor(1600, 1600, 64, 1, VPUNN::DataType::INT8)},  // input dimensions
            {VPUNN::VPUTensor(1600, 1600, 64, 1, VPUNN::DataType::INT8)},  // output dimensions
            {1, 1},                                                        // kernels
            {1, 1},                                                        // strides
            {0, 0, 0, 0},                                                  // padding
            VPUNN::ExecutionMode::CUBOID_16x16                             // execution mode
    };

    VPUNN::DPUWorkload wl4_0 = {
            VPUNN::VPUDevice::VPU_4_0,
            VPUNN::Operation::CONVOLUTION,
            {VPUNN::VPUTensor(16, 16, 64, 1, VPUNN::DataType::BFLOAT16)},  // input dimensions
            {VPUNN::VPUTensor(16, 16, 64, 1, VPUNN::DataType::FLOAT16)},   // output dimensions
            {1, 1},                                                        // kernels
            {1, 1},                                                        // strides
            {0, 0, 0, 0},                                                  // padding
            VPUNN::ExecutionMode::CUBOID_16x16                             // execution mode
    };
    struct TestInput {
        VPUNN::DPUWorkload wl;
    };

    struct TestExpectation {
        // input swizzling
        Swizzling swizIn0;
        Swizzling swizIn1;

        // output swizzling
        Swizzling swizOut0;
    };

    struct TestCase {
        TestInput t_in;
        TestExpectation t_exp;
        std::string text = "";
    };

    // variables of type TestExpectation representing the expected  values for swizzling
    // const TestExpectation allZero{Swizzling::KEY_0, Swizzling::KEY_0, Swizzling::KEY_0};
    const TestExpectation allFive{Swizzling::KEY_5, Swizzling::KEY_5, Swizzling::KEY_5};

    using TestsVector = std::vector<TestCase>;

    // this lambda function check if swizzling is initialized correctly
    // for VPU2.0 swizzling=0 is the only one possible, but by default=5 (in constructor swizzling is set to 5)
    // swizzling=5 by default for VPU2.7 and newer
    auto verify_swizz = [](const TestsVector& tests) {
        int i = 1;  // test case index
        for (const auto& t : tests) {
            std::cout << "Test case: " << i << "\n";
            EXPECT_EQ(t.t_in.wl.input_swizzling[0], t.t_exp.swizIn0);    // input0 swizzling
            EXPECT_EQ(t.t_in.wl.input_swizzling[1], t.t_exp.swizIn1);    // input1 swizzling
            EXPECT_EQ(t.t_in.wl.output_swizzling[0], t.t_exp.swizOut0);  // output0 swizzling
            i++;
        }
    };

    const TestsVector tests = {
            {{std::move(wl2_0)},
             allFive,
             "VPU2_0, swizzling should be 5 by default (swizzling is set to 5 in constructor)"},
            {{std::move(wl2_7)},
             allFive,
             "VPU2_7, swizzling should be 5 by default (swizzling is set to 5 in constructor)"},
            {{std::move(wl4_0)},
             allFive,
             "VPU4_0, swizzling should be 5 by default (swizzling is set to 5 in constructor)"},
    };

    verify_swizz(tests);
}

/// Test get_padMax method
TEST_F(IntfDeviceValidValuesTest, padMaxTest) {
    DeviceValidValuesMock dut;

    ASSERT_EQ(dut.get_padMax(7), 3);
    ASSERT_EQ(dut.get_padMax(8), 4);

    ASSERT_EQ(dut.get_padMax(1), 0);
    ASSERT_EQ(dut.get_padMax(2), 1);
    ASSERT_EQ(dut.get_padMax(0), 0);

    ASSERT_EQ(dut.get_padMax(3), 1);
    ASSERT_EQ(dut.get_padMax(4), 2);

    ASSERT_EQ(dut.get_padMax(5), 2);
    ASSERT_EQ(dut.get_padMax(6), 3);
}

/// Test compute_output_dim method
TEST_F(IntfDeviceValidValuesTest, outputDimTest) {
    DeviceValidValuesMock dut;

    ASSERT_EQ(dut.compute_output_dim(7, 0, 0, 3, 1), 5);
    ASSERT_EQ(dut.compute_output_dim(7, 1, 1, 3, 1), 7);
    ASSERT_EQ(dut.compute_output_dim(7, 1, 0, 3, 1), 6);
    ASSERT_EQ(dut.compute_output_dim(7, 0, 1, 3, 1), 6);

    ASSERT_EQ(dut.compute_output_dim(7, 0, 0, 3, 2), 3);

    ASSERT_EQ(dut.compute_output_dim(6, 0, 0, 3, 2), 2);

    ASSERT_EQ(dut.compute_output_dim(3, 0, 0, 3, 1), 1);
    ASSERT_EQ(dut.compute_output_dim(3, 0, 0, 3, 2), 1);
    ASSERT_EQ(dut.compute_output_dim(3, 1, 1, 3, 1), 3);

    ASSERT_EQ(dut.compute_output_dim(10, 2, 2, 5, 2), 5);
    ASSERT_EQ(dut.compute_output_dim(10, 1, 1, 5, 2), 4);
    ASSERT_EQ(dut.compute_output_dim(10, 1, 2, 5, 2), 5);

    ASSERT_EQ(dut.compute_output_dim(10, 3, 3, 9, 5), 2);
    ASSERT_EQ(dut.compute_output_dim(10, 3, 3, 9, 4), 2);

    ASSERT_EQ(dut.compute_output_dim(10, 2, 2, 9, 4), 2);
    ASSERT_EQ(dut.compute_output_dim(10, 2, 3, 9, 4), 2);

    ASSERT_EQ(dut.compute_output_dim(10, 3, 3, 7, 4), 3);
    ASSERT_EQ(dut.compute_output_dim(10, 2, 3, 7, 4), 3);
    ASSERT_EQ(dut.compute_output_dim(10, 2, 2, 7, 4), 2);

    ASSERT_EQ(dut.compute_output_dim(7, 0, 0, 1, 1), 7);
    ASSERT_EQ(dut.compute_output_dim(7, 0, 0, 1, 2), 4);
    ASSERT_EQ(dut.compute_output_dim(7, 0, 0, 1, 3), 3);

    ASSERT_EQ(dut.compute_output_dim(7, 1, 0, 1, 3), 3);

    // errors examples
    ASSERT_EQ(dut.compute_output_dim(73, 1, 1, 7, 6), 12);

    EXPECT_EQ(dut.compute_output_dim(13, 1, 1, 7, 6), 2);
    EXPECT_EQ(dut.compute_output_dim(15, 1, 1, 7, 6), 2);

    EXPECT_EQ(dut.compute_output_dim(12, 1, 1, 7, 6), 2);
    EXPECT_EQ(dut.compute_output_dim(12, 0, 1, 7, 6), 2);
    EXPECT_EQ(dut.compute_output_dim(12, 1, 0, 7, 6), 2);
    EXPECT_EQ(dut.compute_output_dim(12, 0, 0, 7, 6), 1);
}

/// Test pad_to_next_multiple method
TEST_F(IntfDeviceValidValuesTest, pad_to_next_multipleTest) {
    DeviceValidValuesMock dut;

    ASSERT_EQ(dut.align_to(7, 16), 16);
    ASSERT_EQ(dut.align_to(7, 20), 20);
    ASSERT_EQ(dut.align_to(0, 16), 0);

    ASSERT_EQ(dut.align_to(17, 16), 32);
    ASSERT_EQ(dut.align_to(33, 16), 48);
}

TEST_F(IntfDeviceValidValuesTest, check_trailing_padding_dimTest) {
    DeviceValidValuesMock dut;

    ASSERT_EQ(dut.check_trailing_padding(7, 5, 0, 3, 1), 0);
    ASSERT_EQ(dut.check_trailing_padding(7, 7, 1, 3, 1), 1);

    ASSERT_EQ(dut.check_trailing_padding(10, 5, 2, 5, 2), 1);  // can be

    ASSERT_EQ(dut.check_trailing_padding(7, 7, 1, 3, 1), 1);  // huge padding

    ASSERT_EQ(dut.check_trailing_padding(7, 3, 0, 1, 3), 0);  // k=1

    ASSERT_EQ(dut.check_trailing_padding(7, 7, 0, 1, 1), 0);  // k=1

    // errors examples
    EXPECT_EQ(dut.check_trailing_padding(73, 12, 1, 7, 6), 0);

    EXPECT_EQ(dut.check_trailing_padding(13, 2, 1, 7, 6), 0);
    EXPECT_EQ(dut.check_trailing_padding(14, 2, 1, 7, 6), 0);
    EXPECT_EQ(dut.check_trailing_padding(15, 2, 1, 7, 6), 0);

    EXPECT_EQ(dut.check_trailing_padding(12, 2, 1, 7, 6), 0);
    EXPECT_EQ(dut.check_trailing_padding(12, 2, 0, 7, 6), 1);
    EXPECT_EQ(dut.check_trailing_padding(12, 2, 1, 7, 6), 0);
}
using namespace VPUNN;

// checks what is the minim input height that can be used with this kernel & padding, and STrategy if unsplit
TEST_F(IntfDeviceValidValuesTest, get_input_height_range) {
    DeviceValidValuesMock dut(1);   // extension is 1 , like for split and below
    DeviceValidValuesMock dut2(2);  // extension 2, like for top Layer
    const DPUWorkload tst_refH16P1K3x3{
            VPUDevice::VPU_2_7,
            Operation::CONVOLUTION,
            {VPUTensor(60, 16, 128, 1, VPUNN::DataType::UINT8)},  // input dimensions
            {VPUTensor(60, 16, 128, 1, VPUNN::DataType::UINT8)},  // output dimensions
            {1, 3},                                               // kernels
            {1, 1},                                               // strides
            {1, 1, 1, 1},                                         // padding
            ExecutionMode::CUBOID_16x16,                          // execution mode
    };
    const DPUWorkload tst_refH16P1K5x5{
            VPUDevice::VPU_2_7,
            Operation::CONVOLUTION,
            {VPUTensor(60, 16, 128, 1, VPUNN::DataType::UINT8)},  // input dimensions
            {VPUTensor(60, 16, 128, 1, VPUNN::DataType::UINT8)},  // output dimensions
            {1, 5},                                               // kernels
            {1, 1},                                               // strides
            {1, 1, 1, 1},                                         // padding
            ExecutionMode::CUBOID_16x16,                          // execution mode
    };
    const DPUWorkload tst_refH16P1K2x2{
            VPUDevice::VPU_2_7,
            Operation::CONVOLUTION,
            {VPUTensor(60, 16, 128, 1, VPUNN::DataType::UINT8)},  // input dimensions
            {VPUTensor(60, 16, 128, 1, VPUNN::DataType::UINT8)},  // output dimensions
            {1, 2},                                               // kernels
            {1, 1},                                               // strides
            {1, 1, 1, 1},                                         // padding
            ExecutionMode::CUBOID_16x16,                          // execution mode
    };
    auto check_it = [&dut](DPUOperation& wl, std::array<int, 2> pad, int minExp, const std::string& h = "") {
        wl.kernel.pad_top = pad[0];
        wl.kernel.pad_bottom = pad[1];
        auto r = dut.get_input_height_range(wl);
        EXPECT_EQ(r[0], minExp) << h << "paddings: " << pad[0] << " , " << pad[1] << "\n" << wl;
    };
    auto check_it2 = [&dut2](DPUOperation& wl, std::array<int, 2> pad, int minExp, const std::string& h = "") {
        wl.kernel.pad_top = pad[0];
        wl.kernel.pad_bottom = pad[1];
        auto r = dut2.get_input_height_interval(wl, true);
        EXPECT_EQ(r.first, minExp) << h << "paddings: " << pad[0] << " , " << pad[1] << "\n" << wl;
    };

    {  // 3x3
        std::string info{" x3 "};
        DPUOperation wl{tst_refH16P1K3x3};
        check_it(wl, {1, 1}, 1, info);
        check_it(wl, {1, 0}, 2, info);
        check_it(wl, {0, 1}, 2, info);
        check_it(wl, {0, 0}, 3, info);
    }
    {  // x5
        std::string info{" x5 "};
        DPUOperation wl{tst_refH16P1K5x5};
        check_it(wl, {2, 2}, 1, info);
        check_it(wl, {1, 0}, 4, info);
        check_it(wl, {1, 1}, 3, info);
        check_it(wl, {0, 0}, 5, info);

        check_it(wl, {10, 10}, 1, info + "pathologic pad! ");
    }
    {  // x2
        std::string info{" x2 "};
        DPUOperation wl{tst_refH16P1K2x2};

        check_it(wl, {1, 1}, 1, info);
        check_it(wl, {1, 0}, 1, info);
        check_it(wl, {0, 1}, 1, info);
        check_it(wl, {0, 0}, 2, info);
    }

    {  // 3x3 and SOH special:
        DPUOperation wl{tst_refH16P1K3x3};
        {
            std::string info{" x3SOH1x "};
            EXPECT_EQ(dut.get_input_heigth_start_factor_SOH(), 1);
            wl.isi_strategy = ISIStrategy::SPLIT_OVER_H;
            EXPECT_EQ(dut.get_input_heigth_start_factor_SOH(), 1);

            check_it(wl, {1, 1}, 1, info);
            check_it(wl, {1, 0}, 2, info);
            check_it(wl, {0, 1}, 2, info);
            check_it(wl, {0, 0}, 3, info);
        }
        {
            // 3x3 and SOH special :
            std::string info{" x3SOH2x "};
            // now we need t least 2 rows in output to be guaranteed
            EXPECT_EQ(dut2.get_input_heigth_start_factor_SOH(), 2);
            wl.isi_strategy = ISIStrategy::CLUSTERING;
            EXPECT_EQ(dut2.get_input_heigth_start_factor_SOH(), 2);

            check_it2(wl, {1, 1}, 2, info);
            check_it2(wl, {1, 0}, 3, info);
            check_it2(wl, {0, 1}, 3, info);
            check_it2(wl, {0, 0}, 4, info);
        }
    }
}

// checks what is the minim input height that can be used with this kernel & padding
TEST_F(IntfDeviceValidValuesTest, get_input_width_range) {
    DeviceValidValuesMock dut(1);   // extension is 1 , like for split and below
    DeviceValidValuesMock dut2(2);  // extension 2, like for top Layer
    const DPUWorkload tst_refH16P1K3x3{
            VPUDevice::VPU_2_7,
            Operation::CONVOLUTION,
            {VPUTensor(16, 60, 128, 1, VPUNN::DataType::UINT8)},  // input dimensions
            {VPUTensor(16, 60, 128, 1, VPUNN::DataType::UINT8)},  // output dimensions
            {3, 1},                                               // kernels
            {1, 1},                                               // strides
            {1, 1, 1, 1},                                         // padding
            ExecutionMode::CUBOID_16x16,                          // execution mode
    };
    const DPUWorkload tst_refH16P1K5x5{
            VPUDevice::VPU_2_7,
            Operation::CONVOLUTION,
            {VPUTensor(16, 60, 128, 1, VPUNN::DataType::UINT8)},  // input dimensions
            {VPUTensor(16, 60, 128, 1, VPUNN::DataType::UINT8)},  // output dimensions
            {5, 1},                                               // kernels
            {1, 1},                                               // strides
            {1, 1, 1, 1},                                         // padding
            ExecutionMode::CUBOID_16x16,                          // execution mode
    };
    const DPUWorkload tst_refH16P1K2x2{
            VPUDevice::VPU_2_7,
            Operation::CONVOLUTION,
            {VPUTensor(16, 60, 128, 1, VPUNN::DataType::UINT8)},  // input dimensions
            {VPUTensor(16, 60, 128, 1, VPUNN::DataType::UINT8)},  // output dimensions
            {2, 1},                                               // kernels
            {1, 1},                                               // strides
            {1, 1, 1, 1},                                         // padding
            ExecutionMode::CUBOID_16x16,                          // execution mode
    };
    auto check_it = [&dut](DPUOperation& wl, std::array<int, 2> pad, int minExp, const std::string& h = "") {
        wl.kernel.pad_left = pad[0];
        wl.kernel.pad_right = pad[1];
        auto r = dut.get_input_width_range(wl);
        EXPECT_EQ(r[0], minExp) << h << "paddings: " << pad[0] << " , " << pad[1] << "\n" << wl;
    };
    auto check_it2 = [&dut2](DPUOperation& wl, std::array<int, 2> pad, int minExp, const std::string& h = "") {
        wl.kernel.pad_left = pad[0];
        wl.kernel.pad_right = pad[1];
        auto r = dut2.get_input_width_interval(wl, true);
        EXPECT_EQ(r.first, minExp) << h << "paddings: " << pad[0] << " , " << pad[1] << "\n" << wl;
    };

    {  // 3x3
        std::string info{" x3 "};
        DPUOperation wl{tst_refH16P1K3x3};
        check_it(wl, {1, 1}, 1, info);
        check_it(wl, {1, 0}, 2, info);
        check_it(wl, {0, 1}, 2, info);
        check_it(wl, {0, 0}, 3, info);
    }
    {  // x5
        std::string info{" x5 "};
        DPUOperation wl{tst_refH16P1K5x5};
        check_it(wl, {2, 2}, 1, info);
        check_it(wl, {1, 0}, 4, info);
        check_it(wl, {1, 1}, 3, info);
        check_it(wl, {0, 0}, 5, info);

        check_it(wl, {10, 10}, 1, info + "pathologic pad! ");
    }
    {  // x2
        std::string info{" x2 "};
        DPUOperation wl{tst_refH16P1K2x2};

        check_it(wl, {1, 1}, 1, info);
        check_it(wl, {1, 0}, 1, info);
        check_it(wl, {0, 1}, 1, info);
        check_it(wl, {0, 0}, 2, info);
    }

    {  // 3x3 and SOH special:
        DPUOperation wl{tst_refH16P1K3x3};
        {
            std::string info{" x3SOH1x "};
            EXPECT_EQ(dut.get_input_heigth_start_factor_SOH(), 1);
            wl.isi_strategy = ISIStrategy::SPLIT_OVER_H;
            EXPECT_EQ(dut.get_input_heigth_start_factor_SOH(), 1);

            check_it(wl, {1, 1}, 1, info);
            check_it(wl, {1, 0}, 2, info);
            check_it(wl, {0, 1}, 2, info);
            check_it(wl, {0, 0}, 3, info);
        }
        {
            // 3x3 and SOH special :
            std::string info{" x3SOH2x "};
            // NO change /influence
            EXPECT_EQ(dut2.get_input_heigth_start_factor_SOH(), 2);
            wl.isi_strategy = ISIStrategy::CLUSTERING;
            EXPECT_EQ(dut2.get_input_heigth_start_factor_SOH(), 2);

            check_it2(wl, {1, 1}, 2, info);
            check_it2(wl, {1, 0}, 3, info);
            check_it2(wl, {0, 1}, 3, info);
            check_it2(wl, {0, 0}, 4, info);
        }
    }
}

TEST_F(IntfDeviceValidValuesTest, Restrict_DataTypes_Test) {
    DeviceValidValuesMock dut;

    EXPECT_EQ(dut.restrict_datatype(DataType::UINT8), DataType::UINT8);
    EXPECT_EQ(dut.restrict_datatype(DataType::INT8), DataType::UINT8);

    EXPECT_EQ(dut.restrict_datatype(DataType::UINT4), DataType::UINT4);
    EXPECT_EQ(dut.restrict_datatype(DataType::INT4), DataType::UINT4);

    EXPECT_EQ(dut.restrict_datatype(DataType::UINT2), DataType::UINT2);
    EXPECT_EQ(dut.restrict_datatype(DataType::INT2), DataType::UINT2);

    EXPECT_EQ(dut.restrict_datatype(DataType::UINT1), DataType::UINT1);
    EXPECT_EQ(dut.restrict_datatype(DataType::INT1), DataType::UINT1);

    EXPECT_EQ(dut.restrict_datatype(DataType::FLOAT16), DataType::FLOAT16);
    EXPECT_EQ(dut.restrict_datatype(DataType::BFLOAT16), DataType::FLOAT16);

    EXPECT_EQ(dut.restrict_datatype(DataType::HF8), DataType::HF8);
    EXPECT_EQ(dut.restrict_datatype(DataType::BF8), DataType::HF8);
}

class DPUOperationTest : public ::testing::Test {
public:
protected:
    void SetUp() override {
    }
    const VPUNN::DPUWorkload wl_ref1{
            VPUNN::VPUDevice::VPU_2_7,
            VPUNN::Operation::ELTWISE,
            {VPUNN::VPUTensor(56, 56, 256, 1, VPUNN::DataType::UINT8)},  // input dimensions
            {VPUNN::VPUTensor(56, 56, 256, 1, VPUNN::DataType::UINT8)},  // output dimensions
            {1, 1},                                                      // kernels
            {1, 1},                                                      // strides
            {0, 0, 0, 0},                                                // padding
            VPUNN::ExecutionMode::CUBOID_16x16,                          // execution mode
            VPUNN::ActivationFunction::NONE,                             // activation
            0.0F,                                                        // act_sparsity
            0.0F,                                                        // weight_sparsity
            {swz_def, swz_def},                                          // input_swizzling
            {swz_def},                                                   // output_swizzling
            1,                                                           // output_write_tiles
            {0, 0, 0, 0},                                                // offsets
            VPUNN::ISIStrategy::CLUSTERING,                              // isi_strategy
            false,                                                       // weight_sparsity_enabled
    };

    const VPUNN::DPUWorkload wl_ref2{
            VPUNN::VPUDevice::VPU_2_7,
            VPUNN::Operation::CONVOLUTION,
            {VPUNN::VPUTensor(14, 14, 512, 1, VPUNN::DataType::INT8, VPUNN::Layout::YXZ, true)},  // input dimensions
            {VPUNN::VPUTensor(7, 7, 512, 1, VPUNN::DataType::INT8, VPUNN::Layout::YZX)},          // output dimensions
            {3, 3},                                                                               // kernels
            {2, 2},                                                                               // strides
            {1, 0, 1, 0},                                                                         // padding
            VPUNN::ExecutionMode::CUBOID_16x16,                                                   // execution mode
            VPUNN::ActivationFunction::NONE,                                                      // activation
            0.8F,                                                                                 // act_sparsity
            0.2F,                                                                                 // weight_sparsity
            {VPUNN::Swizzling::KEY_1, VPUNN::Swizzling::KEY_2},                                   // input_swizzling
            {VPUNN::Swizzling::KEY_4},                                                            // output_swizzling
            1,                                                                                    // output_write_tiles
            {0, 0, 0, 0},                                                                         // offsets
            VPUNN::ISIStrategy::SPLIT_OVER_H,                                                     // isi_strategy
            true,  // weight_sparsity_enabled
    };
    const float EPSILON{0.00001f};
    bool is_equal(float a, float b) const {
        return (std::fabs(a - b) < EPSILON);  // very simple since vals around zero
    };

    void testEqual(const DPUOperation& dpu, const DPUWorkload& wl) {
        EXPECT_EQ(dpu.device, wl.device);
        EXPECT_EQ(dpu.operation, wl.op);

        // input 0/ act
        EXPECT_EQ(dpu.input_0.height, wl.inputs[0].height());
        EXPECT_EQ(dpu.input_0.width, wl.inputs[0].width());
        EXPECT_EQ(dpu.input_0.channels, wl.inputs[0].channels());
        EXPECT_EQ(dpu.input_0.batch, wl.inputs[0].batches());
        EXPECT_EQ(dpu.input_0.datatype, wl.inputs[0].get_dtype());
        EXPECT_EQ(dpu.input_0.layout, wl.inputs[0].get_layout());

        EXPECT_NEAR(dpu.input_0.sparsity, wl.act_sparsity, EPSILON);

        EXPECT_EQ(dpu.input_0.sparsity_enabled, wl.inputs[0].get_sparsity());
        EXPECT_EQ(dpu.input_0.swizzling, wl.input_swizzling[0]);

        // inout 1 /weights
        EXPECT_EQ(dpu.input_1.height, 0);
        EXPECT_EQ(dpu.input_1.width, 0);
        EXPECT_EQ(dpu.input_1.channels, 0);
        EXPECT_EQ(dpu.input_1.batch, 1);
        EXPECT_EQ(dpu.input_1.datatype, wl.inputs[0].get_dtype());
        EXPECT_EQ(dpu.input_1.layout, Layout::ZXY);

        EXPECT_NEAR(dpu.input_1.sparsity, wl.weight_sparsity, EPSILON);

        EXPECT_EQ(dpu.input_1.sparsity_enabled, wl.weight_sparsity_enabled);
        EXPECT_EQ(dpu.input_1.swizzling, wl.input_swizzling[1]);

        // output
        EXPECT_EQ(dpu.output_0.height, wl.outputs[0].height());
        EXPECT_EQ(dpu.output_0.width, wl.outputs[0].width());
        EXPECT_EQ(dpu.output_0.channels, wl.outputs[0].channels());
        EXPECT_EQ(dpu.output_0.batch, wl.outputs[0].batches());
        EXPECT_EQ(dpu.output_0.datatype, wl.outputs[0].get_dtype());
        EXPECT_EQ(dpu.output_0.layout, wl.outputs[0].get_layout());

        EXPECT_NEAR(dpu.output_0.sparsity, 0.0f, EPSILON);

        EXPECT_EQ(dpu.output_0.sparsity_enabled, wl.outputs[0].get_sparsity());
        EXPECT_EQ(dpu.output_0.swizzling, wl.output_swizzling[0]);

        EXPECT_EQ(dpu.execution_order, wl.execution_order);

        EXPECT_EQ(dpu.kernel.height, wl.kernels[Dim::Grid::H]);
        EXPECT_EQ(dpu.kernel.width, wl.kernels[Dim::Grid::W]);
        EXPECT_EQ(dpu.kernel.pad_bottom, wl.padding[Dim::Padding::BOTTOM]);
        EXPECT_EQ(dpu.kernel.pad_left, wl.padding[Dim::Padding::LEFT]);
        EXPECT_EQ(dpu.kernel.pad_right, wl.padding[Dim::Padding::RIGHT]);
        EXPECT_EQ(dpu.kernel.pad_top, wl.padding[Dim::Padding::TOP]);
        EXPECT_EQ(dpu.kernel.stride_height, wl.strides[Dim::Grid::H]);
        EXPECT_EQ(dpu.kernel.stride_width, wl.strides[Dim::Grid::W]);

        EXPECT_EQ(dpu.output_write_tiles, wl.output_write_tiles);
        EXPECT_EQ(dpu.isi_strategy, wl.isi_strategy);
    }
};

TEST_F(DPUOperationTest, constructor) {
    EXPECT_FALSE(wl_ref1 == wl_ref2);

    {
        DPUOperation dpu(wl_ref2);
        testEqual(dpu, wl_ref2);
    }
    {
        DPUOperation dpu(wl_ref1);
        testEqual(dpu, wl_ref1);
    }
}

TEST_F(DPUOperationTest, toDPUWorkload) {
    {
        DPUOperation dpu(wl_ref2);
        DPUWorkload w{dpu.clone_as_DPUWorkload()};

        EXPECT_FALSE(w == wl_ref2);
        w.weight_type = wl_ref2.weight_type;
        w.weightless_operation = wl_ref2.weightless_operation;
        w.in_place_output_memory = wl_ref2.in_place_output_memory;
        w.superdense_memory = wl_ref2.superdense_memory;
        w.input_autopad = wl_ref2.input_autopad;
        w.output_autopad = wl_ref2.output_autopad;
        EXPECT_EQ(w, wl_ref2);

        EXPECT_FALSE(w == wl_ref1);
    }
    {
        DPUOperation dpu(wl_ref1);
        DPUWorkload w{dpu.clone_as_DPUWorkload()};

        EXPECT_FALSE(w == wl_ref1);
        w.weight_type = wl_ref1.weight_type;
        w.weightless_operation = wl_ref1.weightless_operation;
        w.in_place_output_memory = wl_ref1.in_place_output_memory;
        w.superdense_memory = wl_ref1.superdense_memory;
        w.input_autopad = wl_ref1.input_autopad;
        w.output_autopad = wl_ref1.output_autopad;
        EXPECT_EQ(w, wl_ref1);

        EXPECT_FALSE(w == wl_ref2);
    }
    {
        DPUWorkload wref3{wl_ref1};
        wref3.weight_type = DataType::INT4;  // set a specific value for the optional

        DPUOperation dpu(wref3);
        DPUWorkload w{dpu.clone_as_DPUWorkload()};

        EXPECT_FALSE(w == wref3);
        w.weightless_operation = wref3.weightless_operation;
        w.in_place_output_memory = wref3.in_place_output_memory;
        w.superdense_memory = wref3.superdense_memory;
        w.input_autopad = wref3.input_autopad;
        w.output_autopad = wref3.output_autopad;
        EXPECT_EQ(w, wref3);

        EXPECT_FALSE(w == wl_ref2);
        EXPECT_FALSE(w == wl_ref1);
    }
    {  // add tests where the clone is equal
       // 1 not an elementwise
       // 2 elementwise with weigthles and inplace set up manually (not default mode)
    }
    {  // test for superdense
        {
            DPUWorkload wref3{wl_ref1};
            wref3.set_superdense(true);  // set a specific value for the optional

            DPUOperation dpu(wref3);
            DPUWorkload w{dpu.clone_as_DPUWorkload()};

            EXPECT_FALSE(w == wref3);
            w.weight_type = wref3.weight_type;
            w.weightless_operation = wref3.weightless_operation;
            w.in_place_output_memory = wref3.in_place_output_memory;
            w.input_autopad = wref3.input_autopad;
            w.output_autopad = wref3.output_autopad;
            EXPECT_EQ(w, wref3);

            EXPECT_FALSE(w == wl_ref2);
            EXPECT_FALSE(w == wl_ref1);
        }
        {
            DPUWorkload wref3{wl_ref1};
            wref3.set_superdense(false);  // set a specific value for the optional

            DPUOperation dpu(wref3);
            DPUWorkload w{dpu.clone_as_DPUWorkload()};

            EXPECT_FALSE(w == wref3);
            w.weight_type = wref3.weight_type;
            w.weightless_operation = wref3.weightless_operation;
            w.in_place_output_memory = wref3.in_place_output_memory;
            w.input_autopad = wref3.input_autopad;
            w.output_autopad = wref3.output_autopad;
            EXPECT_EQ(w, wref3);

            EXPECT_FALSE(w == wl_ref2);
            EXPECT_FALSE(w == wl_ref1);
        }
    }
}

}  // namespace VPUNN_unit_tests
