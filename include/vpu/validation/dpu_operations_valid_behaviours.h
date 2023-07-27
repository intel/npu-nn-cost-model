// Copyright © 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_DPU_OPERATIONS_VALID_BEHAVIOURS_H
#define VPUNN_DPU_OPERATIONS_VALID_BEHAVIOURS_H

#include "vpu/types.h"

#include "checker_utils.h"
#include "data_dpu_operation.h"
#include "interface_operations_behavior.h"
#include "interface_valid_values.h"
#include "vpu/sample_generator/sample_generator.h"

namespace VPUNN {

class IOperationDynamicGenerator {
public:
    /// @brief  dynamic establishment of output_0 and input_1
    /// eg input_1 (weights) may depend dynamically on output_0 info (channels)
    virtual void generate_operation_dependent_tensors(Sampler& sampler, const IDeviceValidValues& config,
                                                      DPUOperation& dpu) const = 0;

    /// @brief fills/generates sparsity values
    virtual void generate_sparsity(Sampler& sampler, const IDeviceValidValues& config, DPUOperation& dpu) const = 0;

protected:
    virtual ~IOperationDynamicGenerator() = default;
};

class Base_Constraints : public IOperationDynamicConstraints {
public:
protected:
    virtual long long get_weight_table_size(const long long out_0_channels) const {
        return out_0_channels * 16;
    }

public:
    /// @brief this specialization checks sparsity is turned off
    bool check_sparsity_rules(const IDeviceValidValues&, const DPUOperation& dpu, std::string& info) const override {
        Checker checker;
        checker.check_is_equal(dpu.input_0.sparsity_enabled, false, "input_0 sparsity_enabled  ");
        checker.check_is_equal(dpu.input_0.sparsity, 0.0f, "input_0 sparsity  ");

        checker.check_is_equal(dpu.input_1.sparsity_enabled, false, "input_1 sparsity_enabled  ");
        checker.check_is_equal(dpu.input_1.sparsity, 0.0f, "input_1 sparsity  ");

        checker.check_is_equal(dpu.output_0.sparsity_enabled, false, "output_0 sparsity_enabled  ");

        info = checker.findings();
        return checker.is_clean();
    }

    long long input_1_volume(const TensorInfo& w) const noexcept override {
        return w.height * w.width * w.channels * w.batch;
    }

    /// @brief computes the aligned size in bytes
    long long input_1_aligned_size_bytes(const long long elem_size, const IDeviceValidValues& config,
                                         const DPUOperation& dpu) const noexcept override {
        auto in_1_size = config.compute_size_raw(elem_size, dpu.input_1.datatype);

        if (dpu.input_1.sparsity_enabled) {
            // reducing according to sparsity
            in_1_size -= static_cast<decltype(in_1_size)>(std::floor((in_1_size * dpu.input_1.sparsity)));

            // adding sparsity map
            const auto one_output_sparse_bit_map{
                    config.align_to((dpu.input_0.channels * dpu.kernel.height * dpu.kernel.width) / 8, 16)};
            in_1_size += (dpu.output_0.channels * one_output_sparse_bit_map);
        }

        in_1_size += get_weight_table_size(dpu.output_0.channels);

        return config.align_to(in_1_size, config.alignement_size_bytes);  // # align to 16KB chunks
    }
};

class GenericConvolution_Constraints : public Base_Constraints, public IOperationDynamicGenerator {
public:
protected:
    /// @brief sets initial input_1 content,
    void set_input_1_props_from_input_0(DPUOperation& dpu) const {
        dpu.input_1.datatype = dpu.input_0.datatype;  // TODO: add support for act fp16 and weight i8
        dpu.input_1.swizzling = dpu.input_0.swizzling;
        dpu.input_1.layout = dpu.input_0.layout;
    }

public:
    void generate_sparsity(Sampler&, const IDeviceValidValues&, DPUOperation& dpu) const override {
        dpu.input_0.sparsity = 0.0F;
        dpu.input_0.sparsity_enabled = false;
        dpu.input_1.sparsity = 0.0F;
        dpu.input_1.sparsity_enabled = false;
    }
    /// @ reduces/adjusts sparsity  according to context
    void limit_sparsity(const IDeviceValidValues&, DPUOperation& dpu) const override {
        if ((dpu.isi_strategy == ISIStrategy::SPLIT_OVER_K)  // SOK
            && (dpu.output_0.channels % 32 != 0))            // channels not aligned to 32
        {
            dpu.input_1.sparsity_enabled = false;
            dpu.input_1.sparsity = 0.0F;
        }
    }

    /// @brief if SOK => channels must be aligned to 32 channels
    /// rule to be applied to un-tiled layer, before split on tiles
    bool check_sparsity_layer_SOK(const IDeviceValidValues&, const DPUOperation& dpu, std::string& info) const {
        Checker checker;
        if ((dpu.isi_strategy == ISIStrategy::SPLIT_OVER_K)  // SOK
            && (dpu.output_0.channels % 32 != 0))            // channels not aligned to 32
        {
            checker.check_is_equal(dpu.input_1.sparsity_enabled, false,
                                   " SOK + out.channels not aligned to 32: input_1 sparsity_enabled  ");
            checker.check_is_equal(dpu.input_1.sparsity, 0.0f,
                                   " SOK + out.channels not aligned to 32: input_1 sparsity  ");
        }

        info = checker.findings();
        return checker.is_clean();
    }
};

class CONVOLUTION_Constraints : public GenericConvolution_Constraints {
protected:
    void generate_operation_dependent_tensors(Sampler& sampler, const IDeviceValidValues& config,
                                              DPUOperation& dpu) const override {
        set_input_1_props_from_input_0(dpu);
        dpu.input_1.height = 1;
        dpu.input_1.width = 1;

        dpu.output_0.datatype = sampler.sample_list(config.valid_datatypes);

        // choose range based on isi
        const Channels& out_channels_range{config.get_output_channels_range(dpu)};

        dpu.output_0.channels = sampler.sample_list_decrease_prob(out_channels_range);  //  non uniform
        dpu.input_1.channels = dpu.input_0.channels * (long long)dpu.kernel.height * (long long)dpu.kernel.width;

        dpu.input_1.batch = dpu.output_0.channels;
    }

    bool check_input_output_tensor_corelation(const IDeviceValidValues& config, const DPUOperation& dpu,
                                              std::string& info) const override {
        Checker checker;
        checker.check_is_in_list((int)dpu.output_0.channels, config.get_output_channels_range(dpu),
                                 "output_0.channels");

        info = checker.findings();
        return checker.is_clean();
    };

    void generate_sparsity(Sampler& sampler, const IDeviceValidValues& config, DPUOperation& dpu) const override {
        dpu.input_1.sparsity_enabled = sampler.sample_list(config.boolean_datatypes);  // uniform true/false

        dpu.input_1.sparsity = 0.0F;  // just as default

        if (dpu.input_1.sparsity_enabled)  // provide a rate also
        {
            const auto raw_sparsity = sampler.sample_continuous_uniform();
            dpu.input_1.sparsity = config.sanitize_sparsity(input_1_volume(dpu.input_1), raw_sparsity);
        }

        dpu.input_0.sparsity = 0.0F;
        dpu.input_0.sparsity_enabled = false;
    }

    bool check_sparsity_rules(const IDeviceValidValues&, const DPUOperation& dpu, std::string& info) const override {
        Checker checker;
        // if not enabled then the value should be zero

        if (!dpu.input_0.sparsity_enabled) {
            checker.check_is_equal(dpu.input_0.sparsity, 0.0f, "input_0.sparsity_enabled false and sparsity is ");
        }

        if (!dpu.input_1.sparsity_enabled) {
            checker.check_is_equal(dpu.input_1.sparsity, 0.0f, "input_1.sparsity_enabled false and sparsity is ");
        }

        checker.check_is_equal(dpu.output_0.sparsity_enabled, false, "output_0 sparsity_enabled  ");

        info = checker.findings();
        return checker.is_clean();
    }

    void deduce_input_1(const TensorInfo& in_0, const TensorInfo& out_0, const IDeviceValidValues&,
                        const KernelInfo& kernel, TensorInfo& w) const noexcept override {
        w.datatype = in_0.datatype;  // TODO: add support for act fp16 and weight i8

        w.layout = in_0.layout;
        w.height = 1;
        w.width = 1;
        w.channels = in_0.channels * kernel.height * kernel.width;
        w.batch = out_0.channels;

        // swizzling and sparsity are not deduced.
    }
};

class DW_CONVOLUTION_Constraints : public GenericConvolution_Constraints {
protected:
    void generate_operation_dependent_tensors(Sampler& sampler, const IDeviceValidValues& config,
                                              DPUOperation& dpu) const override {
        set_input_1_props_from_input_0(dpu);
        dpu.input_1.height = 1;
        dpu.input_1.width = 1;

        dpu.output_0.datatype = sampler.sample_list(config.valid_datatypes);

        dpu.output_0.channels = dpu.input_0.channels;  // keep channels
        dpu.input_1.channels = config.align_to(dpu.kernel.height * dpu.kernel.width, 16);

        dpu.input_1.batch = dpu.output_0.channels;
    }
    bool check_input_output_tensor_corelation(const IDeviceValidValues&, const DPUOperation& dpu,
                                              std::string& info) const override {
        Checker checker;
        checker.check_is_in_list((int)dpu.output_0.channels, {(int)dpu.input_0.channels}, "output_0.channels");

        info = checker.findings();
        return checker.is_clean();
    };

    void deduce_input_1(const TensorInfo& in_0, const TensorInfo& out_0, const IDeviceValidValues& config,
                        const KernelInfo& kernel, TensorInfo& w) const noexcept override {
        w.datatype = in_0.datatype;  // TODO: add support for act fp16 and weight i8

        w.layout = in_0.layout;
        w.height = 1;
        w.width = 1;

        w.channels = config.align_to(kernel.height * kernel.width, 16);

        w.batch = out_0.channels;

        // swizzling and sparsity are not deduced.
    }
};

class CM_CONVOLUTION_Constraints : public GenericConvolution_Constraints {
protected:
    void generate_operation_dependent_tensors(Sampler& sampler, const IDeviceValidValues& config,
                                              DPUOperation& dpu) const override {
        set_input_1_props_from_input_0(dpu);
        dpu.input_1.height = 1;
        dpu.input_1.width = 1;

        dpu.output_0.datatype = sampler.sample_list(config.valid_datatypes);

        // # if input_0_channels==1 Fathom converts it to DW_CONVOLUTION

        // choose range based on isi
        const Channels& out_channels_range{config.get_output_channels_range(dpu)};

        dpu.output_0.channels = sampler.sample_list_decrease_prob(out_channels_range);  //  non uniform
        int multiple{config.contains_value(config.quantized_datatypes, dpu.input_1.datatype) ? 16 : 8};
        dpu.input_1.channels = config.align_to(dpu.input_0.channels * dpu.kernel.height * dpu.kernel.width, multiple);

        dpu.input_1.batch = dpu.output_0.channels;
    }

    bool check_input_output_tensor_corelation(const IDeviceValidValues& config, const DPUOperation& dpu,
                                              std::string& info) const override {
        Checker checker;
        checker.check_is_in_list((int)dpu.output_0.channels, config.get_output_channels_range(dpu),
                                 "output_0.channels");

        info = checker.findings();
        return checker.is_clean();
    };

    long long input_0_volume(const TensorInfo& w) const noexcept override {
        long long channels_lim{(w.channels < 5) ? 4 : 16};
        return w.height * w.width * channels_lim;
    };

    void deduce_input_1(const TensorInfo& in_0, const TensorInfo& out_0, const IDeviceValidValues& config,
                        const KernelInfo& kernel, TensorInfo& w) const noexcept override {
        w.datatype = in_0.datatype;  // TODO: add support for act fp16 and weight i8
        w.layout = in_0.layout;
        w.height = 1;
        w.width = 1;

        int multiple{config.contains_value(config.quantized_datatypes, w.datatype) ? 16 : 8};
        w.channels = config.align_to(in_0.channels * kernel.height * kernel.width, multiple);

        w.batch = out_0.channels;

        // swizzling and sparsity are not deduced.
    }
};

class ELTWISE_Constraints : public Base_Constraints, public IOperationDynamicGenerator {
protected:
    void generate_operation_dependent_tensors(Sampler&, const IDeviceValidValues&, DPUOperation& dpu) const override {
        dpu.input_1.datatype = dpu.input_0.datatype;
        dpu.input_1.swizzling = dpu.input_0.swizzling;
        dpu.input_1.layout = dpu.input_0.layout;

        dpu.output_0.datatype = dpu.input_0.datatype;
        dpu.output_0.channels = dpu.input_0.channels;

        dpu.input_1.batch = dpu.input_0.batch;
        dpu.input_1.channels = dpu.input_0.width;
        dpu.input_1.height = dpu.input_0.channels;
        dpu.input_1.width = dpu.input_0.height;
    }

    bool check_input_output_tensor_corelation(const IDeviceValidValues&, const DPUOperation& dpu,
                                              std::string& info) const override {
        Checker checker;
        checker.check_is_in_list((int)dpu.output_0.channels, {(int)dpu.input_0.channels},
                                 "output_0.channels == input_0.channels");
        // checker.check_is_in_list(dpu.output_0.datatype, {dpu.input_0.datatype},
        //                          "output_0.datatype ==  input_0.datatype");

        info = checker.findings();
        return checker.is_clean();
    };

    void generate_sparsity(Sampler&, const IDeviceValidValues&, DPUOperation& dpu) const override {
        dpu.input_0.sparsity = 0.0F;
        dpu.input_0.sparsity_enabled = false;
        dpu.input_1.sparsity = 0.0F;
        dpu.input_1.sparsity_enabled = false;
    }

    long long input_1_volume(const TensorInfo& w) const noexcept override {
        return w.height * w.width * w.channels;
    }

    void deduce_input_1(const TensorInfo& in_0, const TensorInfo&, const IDeviceValidValues&, const KernelInfo&,
                        TensorInfo& w) const noexcept override {
        w.datatype = in_0.datatype;  // TODO: add support for act fp16 and weight i8

        w.layout = in_0.layout;

        w.batch = in_0.batch;
        w.channels = in_0.width;
        w.height = in_0.channels;
        w.width = in_0.height;

        // swizzling and sparsity are not deduced.
    }

    Values<ISIStrategy> filter_ISI_Strategy_Options(const Values<ISIStrategy>& strategies) const override {
        Values<ISIStrategy> v{strategies};

        // Erase–remove idiom
        v.erase(std::remove_if(v.begin(), v.end(),
                               [](const ISIStrategy& x) {
                                   return x == ISIStrategy::SPLIT_OVER_K;  // SOK not allowed for elementwise
                               }),
                v.cend());
        return v;
    }

    /// @returns a output_write_tile container that has the invalid ones eliminated. Operation dependent.
    Values<int> filter_output_write_tile_Options(const Values<int>& output_write_tile_variants) const override {
        Values<int> v{output_write_tile_variants};  // only 1 is allowed

        // Erase–remove idiom
        v.erase(std::remove_if(v.begin(), v.end(),
                               [](const int& x) {
                                   return x != 1;  // only 1 is allowed
                               }),
                v.cend());
        return v;
    }

    long long get_weight_table_size(const long long) const override {
        return 0;
    }
};
class MAXPOOL_Constraints : public Base_Constraints, public IOperationDynamicGenerator {
protected:
    void generate_operation_dependent_tensors(Sampler&, const IDeviceValidValues&, DPUOperation& dpu) const override {
        dpu.input_1.datatype = dpu.input_0.datatype;
        dpu.input_1.swizzling = dpu.input_0.swizzling;
        dpu.input_1.layout = Layout::INVALID;  // INVALID special case for this operation. Post sanitization should
                                               // not change this

        dpu.output_0.datatype = dpu.input_0.datatype;
        dpu.output_0.channels = dpu.input_0.channels;

        dpu.input_1.batch = 0;
        dpu.input_1.channels = 0;
        dpu.input_1.height = 0;
        dpu.input_1.width = 0;
    }

    bool check_input_output_tensor_corelation(const IDeviceValidValues&, const DPUOperation& dpu,
                                              std::string& info) const override {
        Checker checker;
        checker.check_is_in_list((int)dpu.output_0.channels, {(int)dpu.input_0.channels},
                                 "output_0.channels == input_0.channels");
        // checker.check_is_in_list(dpu.output_0.datatype, {dpu.input_0.datatype},
        //                         "output_0.datatype ==  input_0.datatype");

        info = checker.findings();
        return checker.is_clean();
    };

    void generate_sparsity(Sampler&, const IDeviceValidValues&, DPUOperation& dpu) const override {
        dpu.input_0.sparsity = 0.0F;
        dpu.input_0.sparsity_enabled = false;
        dpu.input_1.sparsity = 0.0F;
        dpu.input_1.sparsity_enabled = false;
    }

    long long input_1_volume(const TensorInfo&) const noexcept override {
        return 0;
    }

    void deduce_input_1(const TensorInfo& in_0, const TensorInfo&, const IDeviceValidValues&, const KernelInfo&,
                        TensorInfo& w) const noexcept override {
        w.datatype = in_0.datatype;  // TODO: add support for act fp16 and weight i8

        w.layout = in_0.layout;

        w.batch = 0;
        w.channels = 0;
        w.height = 0;
        w.width = 0;

        // swizzling and sparsity are not deduced.
    }
};

}  // namespace VPUNN

#endif  //
