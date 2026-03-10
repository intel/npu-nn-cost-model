// Copyright © 2024 Intel Corporation
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

template <>
inline SmartRanges::value_type Sampler::sample_list_decrease_prob<SmartRanges>(const SmartRanges& elements) const {
    // lambda function to generate a list of elements for a SmartRanges object
    auto gen_SmartList = [](const SmartRanges& elements) -> std::vector<SmartRanges::value_type> {
        std::string text{""};
        std::vector<SmartRanges::value_type> result;

        for (auto i = elements.getLowerBound(); i <= elements.getUpperBound(); i++) {
            if (elements.is_in(i, text))
                result.emplace_back(i);
        }
        return result;
    };

    auto elem_list{gen_SmartList(elements)};
    return sample_list_decrease_prob(elem_list);
}

template <>
inline MultiSmartRanges::value_type Sampler::sample_list_decrease_prob<MultiSmartRanges>(
        const MultiSmartRanges& elements) const {
    // lambda function to generate a list of elements for a SmartRanges object
    auto gen_SmartList = [](const MultiSmartRanges& elements) -> std::vector<MultiSmartRanges::value_type> {
        std::string text{""};
        std::vector<SmartRanges::value_type> result;

        for (auto i = elements.getLowerBound(); i <= elements.getUpperBound(); i++) {
            if (elements.is_in(i, text))
                result.emplace_back(i);
        }
        return result;
    };

    auto elem_list{gen_SmartList(elements)};
    return sample_list_decrease_prob(elem_list);
}

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
public:
    /// @brief this specialization checks sparsity is turned off for inputs, any state for output
    bool check_sparsity_rules(const IDeviceValidValues&, const DPUOperation& dpu, std::string& info) const override {
        Checker checker;
        checker.check_is_equal(dpu.input_0.sparsity_enabled, false, "input_0 sparsity_enabled  ");
        checker.check_is_equal(dpu.input_0.sparsity, 0.0f, "input_0 sparsity  ");

        checker.check_is_equal(dpu.input_1.sparsity_enabled, false, "input_1 sparsity_enabled  ");
        checker.check_is_equal(dpu.input_1.sparsity, 0.0f, "input_1 sparsity  ");

        // checker.check_is_equal(dpu.output_0.sparsity_enabled, false, "output_0 sparsity_enabled  ");

        info = checker.findings();
        return checker.is_clean();
    }

    /// weights number of elements. SPecific for some operations
    long long input_1_volume(const TensorInfo& w) const noexcept override {
        return w.height * w.width * w.channels * w.batch;
    }

    /// @brief computes the aligned size in bytes for weights
    long long input_1_aligned_size_bytes(const IDeviceValidValues& config,
                                         const DPUOperation& dpu) const noexcept override final {
        const auto size_nonaligned{input_1_contiguous_size_bytes(config, dpu)};
        return config.align_to(size_nonaligned, config.get_page_alignment());  // # align to 16KB chunks
    }

    long long input_0_volume(const TensorInfo& w) const noexcept override {
        return w.height * w.width * w.channels * w.batch;
    }

    /// @brief computes the aligned size in bytes for activators of a workload. the actual memory occupied considering
    /// SEP or sparsity.
    long long input_0_aligned_size_bytes(const IDeviceValidValues& config,
                                         const DPUOperation& dpu) const noexcept override final {
        const auto size_nonaligned{Base_Constraints::input_0_contiguous_size_bytes(config, dpu)};  // non polymorphic
        return config.align_to(size_nonaligned, config.get_page_alignment());  // # align to 16KB chunks
    }

    /// @brief computes the non CMX aligned/contiguous  size in bytes for the activators
    /// if dtype is INT1, INT2, INT4 will throw because these dtypes are not supported for input0 at this moment
    /// this implementation should have the same behavior as function input_1_contiguous_size_bytes from class
    /// ELTWISE_Constrains
    long long input_0_contiguous_size_bytes(const IDeviceValidValues& config,
                                            const DPUOperation& dpu) const noexcept override final {
        // has halo
        const long long default_compute_tensor_samples{input_0_volume(dpu.input_0)};

        TensorInfo t{dpu.input_0_memory_dense};

        // long long data_memory_samples{default_compute_memeory_samples};  //  actual values
        long long sparsity_map_bytes{0};              // sparsity map is in general zero
        long long storage_elements_table_samples{0};  // SEP table is in general not present

        // has SEP
        const SEPModeInfo& sep{dpu.sep_activators};
        if (sep.isEnabled()) {
            t.width = sep.actual_activators_input.width();
            t.height = sep.actual_activators_input.height();
            t.channels = sep.actual_activators_input.channels();
            t.batch = sep.actual_activators_input.batches();

            storage_elements_table_samples = sep.storage_elements_pointers.numberOfElements();
            if (false == sep.no_sparse_map) {  // sparse map is present
                sparsity_map_bytes = config.align_to(default_compute_tensor_samples / 8,
                                                     16);  // one bit per all!!! compute samples, 16 bytes aligned

                // second option would be on compute tensor minus all halo read from others, and NOT adding extra
                // memory that ins not used (neg halo)!

                // third would be on memory dense input representation (all unpacked input memory , used and unused,
                // without halo read from others)

                // Case B , SEP with actual Memory  larger than compute tensor (sampling inside the big memory
                // tensor): OK use compute samples (maybe  option 2 minus halo read ones)

                // is it really possible to have SEP and halo for  extending memory? Rather the extension is part of
                // SEP's actual_activators_input
            }  // SM
        }  // SEP

        // has sparsity,  can be also with SEP or HALO
        if (dpu.input_0.sparsity_enabled) {
            // actual memory data should be reduced with sparsity factor  (all of it, even the unused one!!?)
            // reducing according to sparsity? A(yes); B(no) because the input data is not packed at runtime
            /* data_memory_samples -= static_cast<decltype(data_memory_samples)>(
                     std::floor((data_memory_samples * dpu.input_0.sparsity)));*/

            // storage_elements_table remains as for HALO(0) or SEP, not used in sparsity

            sparsity_map_bytes = config.align_to(default_compute_tensor_samples / 8,
                                                 16);  // one bit per all!!! compute samples, 16 bytes aligned
        }

        const long long innermost_dim_size_B{t.get_tensor_innermost_dim_B()};
        const long long number_of_elem_excluding_innermost_dim{t.numberOfElementsExcludingInnermost()};
        const auto tensor_size_B{
                innermost_dim_size_B *  // not aligning the innermost dim for input0 for 16/32 cmx alignment
                number_of_elem_excluding_innermost_dim};  // tensor size in bytes

        // let's sum in bytes'
        const long long size_nonaligned{tensor_size_B + sparsity_map_bytes +
                                        storage_elements_table_samples * pointer_size};

        return size_nonaligned;
    }

    /// @brief computes the aligned size in bytes for  output activators of a workload. the actual memory occupied
    /// considering sparsity.
    long long output_0_aligned_size_bytes(const IDeviceValidValues& config,
                                          const DPUOperation& dpu) const noexcept override final {
        const auto size_nonaligned{Base_Constraints::output_0_contiguous_size_bytes(config, dpu)};  // non polymorphic
        return config.align_to(size_nonaligned, config.get_page_alignment());  // # align to 16KB chunks
    }

    /// @brief computes the non CMX aligned/contiguous  size in bytes for the output. the actual memory occupied
    /// considering sparsity map
    /// if dtype is INT1, INT2, INT4 will throw because these dtypes are not supported for output0 at this moment
    long long output_0_contiguous_size_bytes(const IDeviceValidValues& config,
                                             const DPUOperation& dpu) const noexcept override final {
        TensorInfo t{dpu.output_0_memory_dense};

        const long long innermost_dim_size_B{t.get_tensor_innermost_dim_B()};  // innermost dimension size in bytes
        const long long number_of_elem_excluding_innermost_dim{
                t.numberOfElementsExcludingInnermost()};  // number of elements excluding innermost dim
        const long long aligned_innermost_dim_size_B{
                dpu.superdense                  // innermost dim is aligned to specific alignment only if not superdense
                        ? innermost_dim_size_B  // if superdense. no alignment
                        : config.align_to(innermost_dim_size_B,
                                          config.get_specific_out_innermost_dim_alignment_B())  // needs alignment
        };

        const auto tensor_size_B{aligned_innermost_dim_size_B *
                                 number_of_elem_excluding_innermost_dim};  // tensor size in bytes

        // has sparsity
        long long sparsity_map_bytes{0};  // sparsity map is in general zero
        if (dpu.output_0.sparsity_enabled) {
            const long long default_compute_memeory_samples{
                    dpu.output_0_memory_dense
                            .numberOfElements()};  // computation was made on real elements, no alignment here
            // no reduction of size due to sparsity
            sparsity_map_bytes = config.align_to(default_compute_memeory_samples / 8,
                                                 16);  // one bit per all!!! compute samples, 16 bytes aligned
        }

        const long long size_nonaligned{tensor_size_B + sparsity_map_bytes};
        // let's sum in bytes'
        return size_nonaligned;
    }
};

class GenericConvolution_Constraints : public Base_Constraints, public IOperationDynamicGenerator {
public:
protected:
    virtual long long get_weight_table_size(const long long out_0_channels) const noexcept {
        return out_0_channels * 16;
    }

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

    long long input_1_contiguous_size_bytes(const IDeviceValidValues& config,
                                            const DPUOperation& dpu) const noexcept override {
        // we assume for all types of CONVS that height and width dimensions are equal with 1, channels dimension is
        // mask size and also considered innermost dimension. In this way layout is considered implicitly channel
        // innermost (ZXY)
        assert(dpu.input_1.width == 1 && dpu.input_1.height == 1);

        // size in bytes
        auto in_1_size = [&] {
            const TensorInfo t{dpu.input_1};

            // we compute the correct number of bytes occupied by just one mask depending on datatype we have
            const auto size_B_of_one_single_mask{t.get_tensor_innermost_dim_B()};  // in bytes

            // we align the mask (which represents innermost dimension for inpu1 tensor) to the correct alignment
            // depending on a specific device . In bytes
            const auto aligned_size_B_of_one_single_mask =
                    config.is_legacy_samples_alignment_weights()
                            ? size_B_of_one_single_mask  // alignment was done in samples. no alignement here in bytes
                            : config.align_to(size_B_of_one_single_mask, config.get_specific_weights_alignment_B());
            return aligned_size_B_of_one_single_mask * dpu.input_1.batch;  // in bytes
        }();

        if (dpu.input_1.sparsity_enabled) {
            // here we calculate how much of the input size is "sparse"(can be skipped/reduced according to sparsity)
            // and subtracts that from the total size
            in_1_size -= static_cast<decltype(in_1_size)>(std::floor((in_1_size * dpu.input_1.sparsity)));

            // adding sparsity map
            const auto one_output_sparse_bit_map{config.align_to(dpu.input_1.channels / 8, 16)};
            in_1_size += (dpu.input_1.batch * one_output_sparse_bit_map);
        }

        in_1_size += get_weight_table_size(dpu.output_0.channels);

        return in_1_size;
    }
};

class CONVOLUTION_Constraints : public GenericConvolution_Constraints {
protected:
    void generate_operation_dependent_tensors(Sampler& sampler, const IDeviceValidValues& config,
                                              DPUOperation& dpu) const override {
        set_input_1_props_from_input_0(dpu);
        dpu.input_1.height = 1;
        dpu.input_1.width = 1;

        dpu.output_0.datatype = sampler.sample_list(config.get_output_valid_datatypes(dpu));

        // choose range based on isi
        const auto out_channels_range{config.get_output_channels_restriction(dpu)};

        dpu.output_0.channels = sampler.sample_list_decrease_prob(out_channels_range);  //  non uniform

        // this rule is here only for Fathom compliance, must be clarified in the future what's actually required
        const auto is_16_align = dtype_to_bytes(dpu.input_1.datatype) > 1 || dpu.device == VPUDevice::VPU_2_7 ||
                                 dpu.input_1.sparsity_enabled;
        dpu.input_1.channels =
                config.align_to(dpu.input_0.channels * (long long)dpu.kernel.height * (long long)dpu.kernel.width,
                                is_16_align ? 16 : 32);

        dpu.input_1.batch = dpu.output_0.channels;
    }

    bool check_input_output_tensor_corelation(const IDeviceValidValues&, const DPUOperation&,
                                              std::string& info) const override {
        Checker checker;
        info = checker.findings();
        return checker.is_clean();
    }

    void generate_sparsity(Sampler& sampler, const IDeviceValidValues& config, DPUOperation& dpu) const override {
        dpu.input_1.sparsity_enabled = sampler.sample_list(config.get_boolean_datatypes());  // uniform true/false

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

        // checker.check_is_equal(dpu.output_0.sparsity_enabled, false, "output_0 sparsity_enabled  ");

        // No sparsity for dCIM
        if (dpu.mpe_engine == MPEEngine::DCIM) {
            checker.check_is_equal(dpu.input_1.sparsity_enabled, false, "dCIM_32x128: input_1 sparsity_enabled  ");
            checker.check_is_equal(dpu.input_1.sparsity, 0.0f, "dCIM_32x128: input_1 sparsity  ");
            checker.check_is_equal(dpu.input_0.sparsity_enabled, false, "dCIM_32x128: input_0 sparsity_enabled  ");
            checker.check_is_equal(dpu.input_0.sparsity, 0.0f, "dCIM_32x128: input_0 sparsity  ");
        }

        info = checker.findings();
        return checker.is_clean();
    }

    /// we assume for all types of CONVS that height and width dimensions are equal with 1, channels dimension is
    /// mask size and also considered innermost dimension and batch dimension represent the number of masks. In this way
    /// layout is considered implicitly channel innermost (ZXY)
    void deduce_input_1_shape_and_layout(const TensorInfo& in_0, const TensorInfo& out_0,
                                         const IDeviceValidValues& config, const KernelInfo& kernel,
                                         TensorInfo& w) const noexcept override {
        const bool is_samples_alignment_on = config.is_legacy_samples_alignment_weights();

        // if legacy alignment flag is on, we keep the old implementation due to regressions
        w.layout = is_samples_alignment_on ? in_0.layout : Layout::ZXY;  // input 1 is an array of masks, innermost
                                                                         // dimension should be channels ( the mask)
        w.height = 1;
        w.width = 1;

        const int multiple = [&] {
            return (is_samples_alignment_on)
                           // legacy alignment on samples
                           ? ((w.sparsity_enabled || (dtype_to_bytes(w.datatype) > 1))          // if multibyte
                                      ? 16                                                      // 16 samples(eg fp16)
                                      : config.get_specific_legacy_weights_alignment_Samples()  // specific alignment in
                                                                                                // samples
                              )
                           : 1;  // 1 means no alignment is done
        }();

        w.channels = config.align_to(in_0.channels * kernel.height * kernel.width, multiple);
        w.batch = out_0.channels;
    }
};

class DW_CONVOLUTION_Constraints : public GenericConvolution_Constraints {
protected:
    void generate_operation_dependent_tensors(Sampler& sampler, const IDeviceValidValues& config,
                                              DPUOperation& dpu) const override {
        set_input_1_props_from_input_0(dpu);
        dpu.input_1.height = 1;
        dpu.input_1.width = 1;

        dpu.output_0.datatype = sampler.sample_list(config.get_output_valid_datatypes(dpu));

        dpu.output_0.channels = dpu.input_0.channels;  // keep channels
        // this rule is here only for Fathom compliance, must be clarified in the future what's actually required.
        // Especially device dependency is a design break
        const auto is_16_align = dtype_to_bytes(dpu.input_1.datatype) > 1 || dpu.device == VPUDevice::VPU_2_7;
        dpu.input_1.channels =
                config.align_to(static_cast<long long>(dpu.kernel.height) * static_cast<long long>(dpu.kernel.width),
                                is_16_align ? 16 : 32);

        dpu.input_1.batch = dpu.output_0.channels;
    }
    bool check_input_output_tensor_corelation(const IDeviceValidValues&, const DPUOperation& dpu,
                                              std::string& info) const override {
        Checker checker;

        if (!dpu.output_autopad) {
            checker.check_is_in_list((int)dpu.output_0.channels, {(int)dpu.input_0.channels},
                                     "output_0.channels == input_0.channels");
        }

        info = checker.findings();
        return checker.is_clean();
    }

    /// we assume for all types of CONVs that height and width dimensions are equal with 1, channels dimension is
    /// mask size and also considered innermost dimension and batch dimension represent the number of masks. In this way
    /// layout is considered implicitly channel innermost (ZXY)
    void deduce_input_1_shape_and_layout(const TensorInfo& in_0, const TensorInfo& out_0,
                                         const IDeviceValidValues& config, const KernelInfo& kernel,
                                         TensorInfo& w) const noexcept override {
        const bool is_samples_alignment_on =
                config.is_legacy_samples_alignment_weights();  // legacy alignment on/off flag

        // if legacy alignment flag is on, we keep the old implementation due to regressions
        w.layout = is_samples_alignment_on ? in_0.layout : Layout::ZXY;  // input 1 is an array of masks, innermost
                                                                         // dimension should be channels ( the mask)
        w.height = 1;
        w.width = 1;

        const int multiple = [&] {
            return (is_samples_alignment_on)
                           // legacy alignment on samples
                           ? ((dtype_to_bytes(w.datatype) > 1)                                  // if multibyte types
                                      ? 16                                                      // 16 samples(eg fp16)
                                      : config.get_specific_legacy_weights_alignment_Samples()  // specific alignment in
                                                                                                // samples
                              )
                           : 1;  // 1 means no alignment is done
        }();

        w.channels =
                config.align_to(static_cast<long long>(kernel.height) * static_cast<long long>(kernel.width), multiple);
        w.batch = out_0.channels;
    }
    friend class AVGPOOL_Constraints;
};

class CM_CONVOLUTION_Constraints : public GenericConvolution_Constraints {
protected:
    void generate_operation_dependent_tensors(Sampler& sampler, const IDeviceValidValues& config,
                                              DPUOperation& dpu) const override {
        set_input_1_props_from_input_0(dpu);
        dpu.input_1.height = 1;
        dpu.input_1.width = 1;

        dpu.output_0.datatype = sampler.sample_list(config.get_output_valid_datatypes(dpu));

        // # if input_0_channels==1 Fathom converts it to DW_CONVOLUTION

        // choose range based on isi
        const auto out_channels_range{config.get_output_channels_restriction(dpu)};

        dpu.output_0.channels = sampler.sample_list_decrease_prob(out_channels_range);  //  non uniform

        const int multiple{wts_mask_alignment_samples(dpu.input_1.datatype)};
        dpu.input_1.channels = config.align_to(dpu.input_0.channels * dpu.kernel.height * dpu.kernel.width, multiple);

        dpu.input_1.batch = dpu.output_0.channels;
    }

    bool check_input_output_tensor_corelation(const IDeviceValidValues&, const DPUOperation&,
                                              std::string& info) const override {
        Checker checker;
        info = checker.findings();
        return checker.is_clean();
    }

    /// CM has special handling , only 4 or 16 channels possible/ WHY we have this affecting memory: UNKNOWN cause
    long long input_0_volume(const TensorInfo& w) const noexcept override {
        long long channels_lim{(w.channels < 5) ? 4 : 16};
        return w.height * w.width * channels_lim * w.batch;
    }

    /// we assume for all types of CONVS that height and width dimensions are equal with 1, channels dimension is
    /// mask size and also considered innermost dimension and batch dimension represent the number of masks. In this way
    /// layout is considered implicitly channel innermost (ZXY)
    void deduce_input_1_shape_and_layout(const TensorInfo& in_0, const TensorInfo& out_0,
                                         const IDeviceValidValues& config, const KernelInfo& kernel,
                                         TensorInfo& w) const noexcept override {
        const bool is_samples_alignment_on =
                config.is_legacy_samples_alignment_weights();  // legacy alignment on/off flag

        // if legacy alignment flag is on, we keep the old implementation due to regressions
        w.layout = is_samples_alignment_on ? in_0.layout : Layout::ZXY;  // input 1 is an array of masks, innermost
                                                                         // dimension should be channels ( the mask)
        w.height = 1;
        w.width = 1;

        // if legacy alignment flag is active the mask will be aligned in samples
        const int multiple{
                (is_samples_alignment_on)
                        ? wts_mask_alignment_samples(
                                  w.datatype)  // if alignment is required, force the alignment to be at least 16 bytes
                        : 1 /* 1 means no alignment is done */};
        w.channels = config.align_to(in_0.channels * kernel.height * kernel.width, multiple);  // samples
        w.batch = out_0.channels;
    }

    ///@brief wts masks must be aligned to minimum 16 bytes!
    /// @returns the alignment in elements
    int wts_mask_alignment_samples(const DataType dtype) const noexcept {
        if (dtype_to_bytes(dtype) > 1) {
            return 8;  // 8 elmnts alignment, => at least 16 bytes alignment
        } else {
            return 16;  // 16 elmnts alignment, => at least 16 bytes alignment
        }
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
        if (!dpu.output_autopad) {
            checker.check_is_in_list((int)dpu.output_0.channels, {(int)dpu.input_0.channels},
                                     "output_0.channels == input_0.channels");
        }

        info = checker.findings();
        return checker.is_clean();
    };

    void generate_sparsity(Sampler&, const IDeviceValidValues&, DPUOperation& dpu) const override {
        dpu.input_0.sparsity = 0.0F;
        dpu.input_0.sparsity_enabled = false;
        dpu.input_1.sparsity = 0.0F;
        dpu.input_1.sparsity_enabled = false;
    }

    long long input_1_volume(const TensorInfo& w) const noexcept override final {
        return w.height * w.width * w.channels;  // batch is expected to be 1
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

        // checker.check_is_equal(dpu.output_0.sparsity_enabled, false, "output_0 sparsity_enabled  ");

        info = checker.findings();
        return checker.is_clean();
    }

    void deduce_input_1_shape_and_layout(const TensorInfo& in_0, const TensorInfo&, const IDeviceValidValues& config,
                                         const KernelInfo&, TensorInfo& w) const noexcept override {
        w.layout = in_0.layout;

        const auto& devices = config.get_devices();  // devices supported by the configuration

        // check if all devices are older than PTL
        bool is_device_older_than_PTL = std::all_of(devices.begin(), devices.end(), [](const VPUDevice dev) {
            return dev < VPUDevice::NPU_5_0;
        });

        w.batch = in_0.batch;

        if (is_device_older_than_PTL) {
            // this permutation is caused by some particular approach in older NN Descriptors
            // that contained the input1 shape and had to match the training set  approach
            // descriptors for NPU5 and newer devices do not contain input1 shape anymore 
            w.channels = in_0.width;
            w.height = in_0.channels;
            w.width = in_0.height;
        } else {
            // for NPU5 and newer devices the permutation does not apply anymore
            w.channels = in_0.channels;
            w.height = in_0.height;
            w.width = in_0.width;
        }
    }

    Values<ISIStrategy> filter_ISI_Strategy_Options(const Values<ISIStrategy>& strategies) const override {
        Values<ISIStrategy> v{strategies};

        // Erase–remove idiom
        // v.erase(std::remove_if(v.begin(), v.end(),
        //                       [](const ISIStrategy& x) {
        //                           return x == ISIStrategy::SPLIT_OVER_K;  // SOK not allowed for elementwise
        //                       }),
        //        v.cend());
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

    /// @brief computes the non CMX aligned/contiguous  size in bytes for the activators
    /// for ELTWISE memory for input1 is the same as input0
    /// this implementation should have the same behavior as Base_Constraints::input_0_contiguous_size_bytes
    long long input_1_contiguous_size_bytes(const IDeviceValidValues& /* config*/,
                                            const DPUOperation& dpu) const noexcept override final {
        const TensorInfo t{dpu.input_1};
        const long long innermost_dim_size_B{t.get_tensor_innermost_dim_B()};
        const long long number_of_elem_excluding_innermost_dim{t.numberOfElementsExcludingInnermost()};
        const long long size_nonaligned{innermost_dim_size_B * number_of_elem_excluding_innermost_dim};

        return size_nonaligned;
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

        if (!dpu.output_autopad) {
            checker.check_is_in_list((int)dpu.output_0.channels, {(int)dpu.input_0.channels},
                                     "output_0.channels == input_0.channels");
        }

        info = checker.findings();
        return checker.is_clean();
    }

    void generate_sparsity(Sampler&, const IDeviceValidValues&, DPUOperation& dpu) const override {
        dpu.input_0.sparsity = 0.0F;
        dpu.input_0.sparsity_enabled = false;
        dpu.input_1.sparsity = 0.0F;
        dpu.input_1.sparsity_enabled = false;
    }

    long long input_1_volume(const TensorInfo&) const noexcept override final {
        return 0;
    }

    void deduce_input_1_shape_and_layout(const TensorInfo&, const TensorInfo&, const IDeviceValidValues&,
                                         const KernelInfo&, TensorInfo& w) const noexcept override {
        w.layout = Layout::INVALID;

        w.batch = 0;
        w.channels = 0;
        w.height = 0;
        w.width = 0;
    }

    /// Maxpool op does not have weights
    long long input_1_contiguous_size_bytes(const IDeviceValidValues& /*config*/,
                                            const DPUOperation& /*dpu*/) const noexcept override final {
        return 0;
    }
};

class LAYERNORM_Constraints : public CONVOLUTION_Constraints {};
class ELTWISE_MUL_Constraints : public ELTWISE_Constraints {};

class REDUCE_Constraints : public Base_Constraints, public IOperationDynamicGenerator {
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

        // reduce will collapse (reduce) the output channels to one value
        checker.check_is_in_list((int)dpu.output_0.width, {(int)dpu.input_0.width}, "output_0.width == input_0.width");
        checker.check_is_in_list((int)dpu.output_0.height, {(int)dpu.input_0.height},
                                 "output_0.height == input_0.height");

        checker.check_is_in_list((int)dpu.output_0.channels, {(int)1}, "output_0.channels == 1");

        checker.check_is_in_list((int)dpu.output_0.batch, {(int)dpu.input_0.batch}, "output_0.batch == input_0.batch");

        // TODO: what happens when a SOK happens, we have only 1 channel!

        info = checker.findings();
        return checker.is_clean();
    };

    void generate_sparsity(Sampler&, const IDeviceValidValues&, DPUOperation& dpu) const override {
        dpu.input_0.sparsity = 0.0F;
        dpu.input_0.sparsity_enabled = false;
        dpu.input_1.sparsity = 0.0F;
        dpu.input_1.sparsity_enabled = false;
    }

    long long input_1_volume(const TensorInfo&) const noexcept override final {
        return 0;  // no weights for REDUCE
    }

    /// no datat and layout invalid!
    void deduce_input_1_shape_and_layout(const TensorInfo&, const TensorInfo&, const IDeviceValidValues&,
                                         const KernelInfo&, TensorInfo& w) const noexcept override {
        w.layout = Layout::INVALID;

        w.batch = 0;
        w.channels = 0;
        w.height = 0;
        w.width = 0;
    }

    /// no weights
    long long input_1_contiguous_size_bytes(const IDeviceValidValues& /*config*/,
                                            const DPUOperation& /*dpu*/) const noexcept override final {
        return 0;
    }
};

/// IS a DW_CONV without weights. BUt we will replace it at the NN descriptor (so some info has to be present)
class AVGPOOL_Constraints :
        public GenericConvolution_Constraints /*, public Base_Constraints,
         public IOperationDynamicGenerator*/

{
protected:
    void generate_operation_dependent_tensors(Sampler& sampler, const IDeviceValidValues& config,
                                              DPUOperation& dpu) const override {
        DW_CONVOLUTION_Constraints().generate_operation_dependent_tensors(sampler, config, dpu);
    }

    bool check_input_output_tensor_corelation(const IDeviceValidValues& config, const DPUOperation& dpu,
                                              std::string& info) const override {
        return DW_CONVOLUTION_Constraints().check_input_output_tensor_corelation(config, dpu, info);
    };

    void generate_sparsity(Sampler&, const IDeviceValidValues&, DPUOperation& dpu) const override {
        dpu.input_0.sparsity = 0.0F;
        dpu.input_0.sparsity_enabled = false;
        dpu.input_1.sparsity = 0.0F;
        dpu.input_1.sparsity_enabled = false;
    }

    void deduce_input_1_shape_and_layout(const TensorInfo& in_0, const TensorInfo& out_0,
                                         const IDeviceValidValues& config, const KernelInfo& kernel,
                                         TensorInfo& w) const noexcept override {
        DW_CONVOLUTION_Constraints().deduce_input_1_shape_and_layout(in_0, out_0, config, kernel, w);
    }
    long long input_1_volume(const TensorInfo&) const noexcept override final {
        return 0;
        // like MAXPOOL , no input_1 wts
    }

    /// no weights
    long long input_1_contiguous_size_bytes(const IDeviceValidValues& /*config*/,
                                            const DPUOperation& /*dpu*/) const noexcept override final {
        return 0;
    }
};

}  // namespace VPUNN

#endif  //
