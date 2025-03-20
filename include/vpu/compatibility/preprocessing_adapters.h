// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef PREPROCESSING_ADAPTERS_H
#define PREPROCESSING_ADAPTERS_H

#include <core/logger.h>
#include <vpu/types.h>

#include <sstream>  // for error formating
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

#include <unordered_set>

namespace VPUNN {

struct FilteredFields {
    ISIStrategy isi{ISIStrategy::CLUSTERING};
    unsigned int owt{1};
};

class PassThroughInputAdapter {
public:
    static DataType mock_replace_datatypes(const DataType in_datatype) {
        return in_datatype;
    }

    static Operation mock_replace_operations(const Operation in_operation) {
        return in_operation;
    }

    static VPUDevice mock_replace_devices(const VPUDevice in_device) {
        return in_device;
    }
    static FilteredFields avoid_untrained_space(const DPUWorkload& w) {
        FilteredFields ff{w.isi_strategy, w.output_write_tiles};
        return ff;
    }

    static VPUTensor alternative_input0_spatial_memory(const DPUWorkload& wl) {
        return wl.inputs[0];
    }
    /// Compute one swizzling based on the 3 in/out individual swizzlings
    static std::tuple<Swizzling, Swizzling, Swizzling> establishUniqueSwizzling(const Swizzling in0,
                                                                                const Swizzling in1,
                                                                                const Swizzling out0, const Operation) {
        std::tuple<Swizzling, Swizzling, Swizzling> resulted_swizz{in0, in1, out0};  // enabled by default
        return resulted_swizz;
    }
};

/// adapting input to VPU2.7
class NN27InputAdapter {
public:
    /// some datatypes are replaced to supported ones
    static DataType mock_replace_datatypes(const DataType in_datatype) {
        // mock BF8 and HF8 to uint8
        const auto datatype{
                (in_datatype == DataType::BF8) || (in_datatype == DataType::HF8)
                        ? DataType::UINT8  // all 8 bit Float expected to be around I8, except Elmwise :around FP16
                        : (((in_datatype == DataType::FLOAT32) || (in_datatype == DataType::INT32))
                                   ? DataType::FLOAT16  // NOt yet supporting 32 bits ODU
                                   : in_datatype)};
        return datatype;
    }

    /// some operations are replaced to supported ones
    static Operation mock_replace_operations(const Operation in_operation) {
        auto operation{in_operation};

        switch (operation) {
        case Operation::LAYER_NORM:  // cascade, would be better a map available?
        case Operation::ELTWISE_MUL:
            operation = Operation::ELTWISE;  // map to elementwise
            break;
        default:
            break;  // nothing
        }

        return operation;
    }

    /// some devices are replaced to supported ones
    static VPUDevice mock_replace_devices(const VPUDevice in_device) {
        // device 4.0 is not supported for now we are mocking VPU_4_0 with 2.7. This has to be removed when we have a
        // VPU4.0 trained NN
        const auto device{(in_device == VPUDevice::VPU_4_0)  // mock 4.0
                                          || ((in_device == VPUDevice::NPU_RESERVED) ||
                                              (in_device == VPUDevice::NPU_RESERVED_W) 
                                              )
                                  ? VPUDevice::VPU_2_7  // all mocked via 2.7
                                  : in_device};

        return device;
    }

    /// ISI Strategy (up to v11) might not be compatible in any combination. Reasons are rather based on data
    /// available for training and training over-fitting.
    ///
    /// a) CLUSTERING + OWT=2+ : not possible,           :replaced with SOK+OWT=2+ (both do no use input HALO),
    /// filter with step b) next
    ///
    /// b) SOK + ELEMENTWISE   : not possible to profile : replace with CLU+OWT=1  (slightly smaller then real
    /// due to owt=1),
    ///
    /// c) SOH + Kernel vertical is 1: no reason to use it, no input halo necessary: replace  with CLU , ,
    /// filter with a) next.
    ///
    /// d) limit owt to 2! THis is to be done only for NPU2.7 trained NNs. For beyond 2.7 we need a new
    /// interface, derived of it!
    ///
    /// order of calls: c, a, b , d
    ///
    /// SOH+OWT>1  was trained. no need to handle
    static FilteredFields avoid_untrained_space(const DPUWorkload& w) {
        FilteredFields ff{w.isi_strategy, w.output_write_tiles};

        auto a_check_CLU_2 = [](FilteredFields& f) {  // a)
            if ((ISIStrategy::CLUSTERING == f.isi) && (1 < f.owt)) {
                f.isi = ISIStrategy::SPLIT_OVER_K;  // SOK is replacing CLU for OWT>1,, next should be check for b)
            }
        };

        auto b_check_SOK_ELM = [&w](FilteredFields& f) {  // b)
            if ((ISIStrategy::SPLIT_OVER_K == f.isi) && (Operation::ELTWISE == w.op)) {
                f.isi = ISIStrategy::CLUSTERING;
                // should go to state a) check
                // a_check_CLU_2(f); do not call A, you will go in circles
                // change  owt to be compatible with  a) rule
                f.owt = 1;
            }
        };

        auto c_check_SOHh_KERNEL = [&w](FilteredFields& f) {  // c)
            if ((ISIStrategy::SPLIT_OVER_H == f.isi) && (1 == w.kernels[Dim::Grid::H])) {
                f.isi = ISIStrategy::CLUSTERING;
                // should go to state a) afterwards
            }
        };

        // limit owt to 2. This was trained only on VPU2.7
        auto d_limit_owt_to_2 = [](FilteredFields& f) {  // d)
            if (2 < f.owt) {
                f.owt = 2;
            }
        };

        c_check_SOHh_KERNEL(ff);
        a_check_CLU_2(ff);
        b_check_SOK_ELM(ff);

        d_limit_owt_to_2(ff);  // only for 2.7

        return ff;
    }

    /// list of operations that were trained to be swizzling sensitive. to 0 and 5.
    /// rest of operations were trained only with key 5
    static const inline std::unordered_set<Operation> trained_for_zero_and_five{Operation::ELTWISE};

    /// weights swizzling is always 0
    static const inline std::unordered_set<Operation> weights_always_zero{Operation::MAXPOOL};

    /// Compute one swizzling based on the 3 in/out individual swizzlings
    static std::tuple<Swizzling, Swizzling, Swizzling> establishUniqueSwizzling(const Swizzling in0,
                                                                                const Swizzling in1,
                                                                                const Swizzling out0,
                                                                                const Operation op) {
        std::tuple<Swizzling, Swizzling, Swizzling> resulted_swizz{Swizzling::KEY_5, Swizzling::KEY_5,
                                                                   Swizzling::KEY_5};  // enabled by default

        if (trained_for_zero_and_five.count(op) != 0) {  // 0 and 5 possible
            // if at least one is different than zero than we consider it to be all 5
            if ((in0 != Swizzling::KEY_0) || (in1 != Swizzling::KEY_0) || (out0 != Swizzling::KEY_0)) {
                resulted_swizz = std::make_tuple(Swizzling::KEY_5, Swizzling::KEY_5, Swizzling::KEY_5);
            } else {  // all 0
                resulted_swizz = std::make_tuple(Swizzling::KEY_0, Swizzling::KEY_0, Swizzling::KEY_0);
            }

        } else {  // only 5 is possible
            resulted_swizz = std::make_tuple(Swizzling::KEY_5, Swizzling::KEY_5, Swizzling::KEY_5);
        }

        if (weights_always_zero.count(op) != 0) {
            std::get<1>(resulted_swizz) = Swizzling::KEY_0;
        }
        return resulted_swizz;
    }

    static VPUTensor alternative_input0_spatial_memory(const DPUWorkload& wl) {
        // this is a special case: VPU2.7 NN for SOHHalo(SPLIT_OVER_H) splits was trained only on memory tensor
        // (smaller than input tensor). SO we need to generate the descriptor using the reduced  memory tensor
        // for W and H.
        if (wl.isi_strategy == VPUNN::ISIStrategy::SPLIT_OVER_H) {
            // hack. Write the actual memory tensor
            const VPUTensor memoryTensor_input0{computeActualSpatialMemoryNoHaloTensor(wl.inputs[0], wl.halo)};
            return memoryTensor_input0;

        } else {
            return wl.inputs[0];  //
        }
    }
    static VPUTensor input0_OperationBasedReplace(const DPUWorkload&, const VPUTensor& input0) {
        const VPUTensor im{input0};
        return im;  // no change
    }

    static VPUTensor computeActualSpatialMemoryNoHaloTensor(const VPUTensor& origT, const HaloWorkload& halo) {
        const auto& in_halo{halo.input_0_halo};
        //  extension will be negative(memory reduction) if halo(positive halo),
        // or positive (memory increase) ,memory is larger, if negative halo, but we consume less (prev layer wrote
        //  more): DO NOT CARE

        auto newDimension = [](const long long crtDimension, const int oneEndHalo, const int otherEndHalo) {
            const int oneExt{oneEndHalo > 0 ? -oneEndHalo : 0};  // only if halo memory
            const int twoExt{otherEndHalo > 0 ? -otherEndHalo : 0};

            const long long newDim = crtDimension + (oneExt + twoExt);
            return (newDim > 0 ? newDim : 0);  // limit to zero
        };
        const auto h{newDimension(origT.height(), in_halo.top, in_halo.bottom)};
        const auto w{newDimension(origT.width(), in_halo.left, in_halo.right)};

        const std::array<unsigned int, 4> newshape{static_cast<unsigned int>(w), static_cast<unsigned int>(h),  //
                                                   origT.channels(), origT.batches()};                          // whcb
        const VPUTensor ret(newshape, origT);
        return ret;
    }
};

/// adapting input to VPU2.7  but in compatibility v159 mode
class NN27_159_InputAdapter : public NN27InputAdapter {
public:
    ///// some datatypes are replaced to supported ones
    // static DataType mock_replace_datatypes(const DataType in_datatype) {
    //     return NN27InputAdapter::mock_replace_datatypes(in_datatype);
    // }

    ///// some operations are replaced to supported ones
    // static Operation mock_replace_operations(const Operation in_operation) {
    //     return NN27InputAdapter::mock_replace_operations(in_operation);
    // }

    ///// some devices are replaced to supported ones
    // static VPUDevice mock_replace_devices(const VPUDevice in_device) {
    //     return NN27InputAdapter::mock_replace_devices(in_device);
    // }

    /// ISI Strategy (up to v11) might not be compatible in any combination. Reasons are rather based on data
    /// available for training and training over-fitting.
    ///
    /// a) CLUSTERING + OWT=2+ : not possible,           :replaced with SOK+OWT=2+ (both do no use input HALO),
    /// filter with step b) next
    ///
    /// b) SOK + ELEMENTWISE   : not possible to profile : replace with CLU+OWT=1  (slightly smaller then real
    /// due to owt=1),
    ///
    /// c) SOH + Kernel vertical is 1: no reason to use it, no input halo necessary: replace  with CLU , ,
    /// filter with a) next.
    ///
    /// d) limit owt to 2! THis is to be done only for NPU2.7 trained NNs. For beyond 2.7 we need a new
    /// interface, derived of it!
    ///
    /// order of calls: c, a, b , d
    ///
    /// SOH+OWT>1  was trained. no need to handle
    static FilteredFields avoid_untrained_space(const DPUWorkload& w) {
        FilteredFields ff{w.isi_strategy, w.output_write_tiles};

        // auto a_check_CLU_2 = [](FilteredFields& f) {  // a)
        //     if ((ISIStrategy::CLUSTERING == f.isi) && (1 < f.owt)) {
        //         f.isi = ISIStrategy::SPLIT_OVER_K;  // SOK is replacing CLU for OWT>1,, next should be check for b)
        //     }
        // };

        // auto b_check_SOK_ELM = [&w](FilteredFields& f) {  // b)
        //     if ((ISIStrategy::SPLIT_OVER_K == f.isi) && (Operation::ELTWISE == w.op)) {
        //         f.isi = ISIStrategy::CLUSTERING;
        //         // should go to state a) check
        //         // a_check_CLU_2(f); do not call A, you will go in circles
        //         // change  owt to be compatible with  a) rule
        //         f.owt = 1;
        //     }
        // };

        // auto c_check_SOHh_KERNEL = [&w](FilteredFields& f) {  // c)
        //     if ((ISIStrategy::SPLIT_OVER_H == f.isi) && (1 == w.kernels[Dim::Grid::H])) {
        //         f.isi = ISIStrategy::CLUSTERING;
        //         // should go to state a) afterwards
        //     }
        // };

        //// limit owt to 2. This was trained only on VPU2.7
        // auto d_limit_owt_to_2 = [](FilteredFields& f) {  // d)
        //     if (2 < f.owt) {
        //         f.owt = 2;
        //     }
        // };

        // c_check_SOHh_KERNEL(ff);
        // a_check_CLU_2(ff);
        // b_check_SOK_ELM(ff);

        // d_limit_owt_to_2(ff);  // only for 2.7

        return ff;
    }

    /// Compute one swizzling based on the 3 in/out individual swizzlings
    static std::tuple<Swizzling, Swizzling, Swizzling> establishUniqueSwizzling(const Swizzling in0,
                                                                                const Swizzling in1,
                                                                                const Swizzling out0,
                                                                                const Operation op) {
        return PassThroughInputAdapter::establishUniqueSwizzling(in0, in1, out0, op);
    }
};

/// adapting input to NPU40
class NN40InputAdapter {
public:
    static const inline std::unordered_set<Operation>& weights_always_zero{NN27InputAdapter::weights_always_zero};
    /// some datatypes are replaced to supported ones
    static DataType mock_replace_datatypes(const DataType in_datatype) {
        return NN27InputAdapter::mock_replace_datatypes(in_datatype);
    }

    /// some operations are replaced to supported ones
    static Operation mock_replace_operations(const Operation in_operation) {
        auto op = NN27InputAdapter::mock_replace_operations(in_operation);
        //// Temporarily, for 4.0 NN, we don't have support for CM_CONV
        //// TODO: maybe scale output value in post processing?
        // if (op == Operation::CM_CONVOLUTION) {
        //     Logger::warning() << "Workload with CM_CONVOLUTION transformed to CONVOLUTION \n";
        //     return Operation::CONVOLUTION;
        //     // THissi incomplete, we need more than replacing the op, we need a valid convolution that has a output
        //     that
        //     // can be corelated with what CM_would be!
        // }
        return op;
    }

    static VPUTensor alternative_input0_spatial_memory(const DPUWorkload& wl) {
        VPUTensor im{NN27InputAdapter::alternative_input0_spatial_memory(wl)};
        return input0_OperationBasedReplace(wl, im);
    }

    static VPUTensor input0_OperationBasedReplace(const DPUWorkload& /*wl*/, const VPUTensor& input0) {
        VPUTensor im{input0};

        // if (wl.op == Operation::CM_CONVOLUTION) {
        //     Logger::warning() << "Workload with CM_CONVOLUTION transformed to CONVOLUTION, and Input channels = 16
        //     \n"; std::array<unsigned int, 4> new_shape{im.get_shape()}; new_shape[2] = 16;  // 16 channels VPUTensor
        //     im_conv{new_shape, im}; return im_conv;
        // }
        return im;  // no change
    }

    /// some devices are replaced to supported ones
    static VPUDevice mock_replace_devices(const VPUDevice in_device) {
        // device RESERVED is not supported for now we are mocking VPU_RESERVED with 4.0. This has to be removed when we
        // have a VPU_RESERVED trained NN
        const auto device{
                (((in_device == VPUDevice::NPU_RESERVED) || (in_device == VPUDevice::NPU_RESERVED_W) 
                  ) ||
                 (in_device > VPUDevice::NPU_RESERVED_W))
                        ? VPUDevice::VPU_4_0  // all mocked via 40
                        : in_device};

        return device;
    }

    /// ISI Strategy (up to v11) might not be compatible in any combination. Reasons are rather based on data
    /// available for training and training over-fitting.
    ///
    /// a) CLUSTERING + OWT=2+ : not possible,           :replaced with SOK+OWT=2+ (both do no use input HALO),
    /// filter with step b) next
    ///
    /// b) SOK + ELEMENTWISE   : not possible to profile : replace with CLU+OWT=1  (slightly smaller then real
    /// due to owt=1),
    ///
    /// c) [SOH] :invalid in : replace  with CLU , ,
    /// filter with a) next.
    ///
    /// d) limit owt to 6! Even if NPU40 supports more
    ///
    /// order of calls: c, a, b , d
    ///
    static FilteredFields avoid_untrained_space(const DPUWorkload& w) {
        FilteredFields ff{w.isi_strategy, w.output_write_tiles};

        // auto a0_SOK_NA_temporarly = [](FilteredFields& f) {  // a0)
        //     if (ISIStrategy::SPLIT_OVER_K == f.isi) {
        //         f.isi = ISIStrategy::CLUSTERING;  // SOK not trained, replace with CLU
        //         f.owt = 1;                        // OWT=1 forcefully
        //     }
        // };

        auto a_check_CLU_2 = [](FilteredFields& f) {  // a)
            if ((ISIStrategy::CLUSTERING == f.isi) && (1 < f.owt)) {
                f.isi = ISIStrategy::SPLIT_OVER_K;  // SOK is replacing CLU for OWT>1,, next should be check for b)
            }
        };

        // NEW ELEMENTWISE SUPPORTS SOK
        // auto b_check_SOK_ELM = [&w](FilteredFields& f) {  // b)
        //     if ((ISIStrategy::SPLIT_OVER_K == f.isi) && (Operation::ELTWISE == w.op)) {
        //         f.isi = ISIStrategy::CLUSTERING;
        //         // should go to state a) check
        //         // a_check_CLU_2(f); do not call A, you will go in circles
        //         // change  owt to be compatible with  a) rule
        //         f.owt = 1;
        //     }
        // };

        auto c_check_SOH_INVALID = [](FilteredFields& f) {  // c)
            if (ISIStrategy::SPLIT_OVER_H == f.isi) {
                f.isi = ISIStrategy::CLUSTERING;
                // should go to state a) afterwards
            }
        };

        // limit owt to 6. This was trained on NPU40
        auto d_limit_owt_to_6 = [](FilteredFields& f) {  // d)
            if (6 < f.owt) {
                f.owt = 6;
            }
        };
        // In practice was observed that OWT=2,3,4,5,6 has the same constant impact

        // ff is the data  holder changed in the next chain of calls

        c_check_SOH_INVALID(ff);
        a_check_CLU_2(ff);
        // b_check_SOK_ELM(ff);  // ELM with broadcast is not allowed/supported . Is this a problem? SHuld be
        // trainable!?

        d_limit_owt_to_6(ff);  // only

        return ff;
    }

    /// Compute one swizzling based on the 3 in/out individual swizzlings
    static std::tuple<Swizzling, Swizzling, Swizzling> establishUniqueSwizzling(const Swizzling in0,
                                                                                const Swizzling in1,
                                                                                const Swizzling out0,
                                                                                const Operation op) {
        std::tuple<Swizzling, Swizzling, Swizzling> resulted_swizz{Swizzling::KEY_5, Swizzling::KEY_5,
                                                                   Swizzling::KEY_5};  // enabled by default

        if (Operation::ELTWISE == op) {  // 0 and 5 possible, and input can be different than output
            // out swizzling  to be normalized to 0 or 5
            const Swizzling out_0_norm{(Swizzling::KEY_0 != out0) ? Swizzling::KEY_5 : Swizzling::KEY_0};

            // first treat input_0 and input_1, they should be the same
            //  if at least one is different than zero than we consider it to be both 5
            if ((in0 != Swizzling::KEY_0) || (in1 != Swizzling::KEY_0)) {
                resulted_swizz = std::make_tuple(Swizzling::KEY_5, Swizzling::KEY_5, out_0_norm);
            } else {  // both 0
                resulted_swizz = std::make_tuple(Swizzling::KEY_0, Swizzling::KEY_0, out_0_norm);
            }

        } else {  // only 5 is possible
            resulted_swizz = std::make_tuple(Swizzling::KEY_5, Swizzling::KEY_5, Swizzling::KEY_5);
        }

        if (NN27InputAdapter::weights_always_zero.count(op) != 0) {
            std::get<1>(resulted_swizz) = Swizzling::KEY_0;
        }
        return resulted_swizz;
    }
};

/// INput adapter for LNL NN, that lets all the swizzlings to pass to NN and to cache.
///  This is needed in order to have in preloaded cache the mixed swizzling information that otherwise are not part of
///  the trained space. disadvantage is that if the preloaded cache does not  hot, the NN prediction is unpredictable
class NN41InputAdapter : public NN40InputAdapter {
public:
    /// Compute one swizzling based on the 3 in/out individual swizzlings
    static std::tuple<Swizzling, Swizzling, Swizzling> establishUniqueSwizzling(const Swizzling in0,
                                                                                const Swizzling in1,
                                                                                const Swizzling out0,
                                                                                const Operation op) {
        std::tuple<Swizzling, Swizzling, Swizzling> resulted_swizz{in0, in1, out0};

        // if (Operation::ELTWISE == op) {  // 0 and 5 possible, and input can be different than output
        //     // out swizzling  to be normalized to 0 or 5
        //     const Swizzling out_0_norm{(Swizzling::KEY_0 != out0) ? Swizzling::KEY_5 : Swizzling::KEY_0};

        //    // first treat input_0 and input_1, they should be the same
        //    //  if at least one is different than zero than we consider it to be both 5
        //    if ((in0 != Swizzling::KEY_0) || (in1 != Swizzling::KEY_0)) {
        //        resulted_swizz = std::make_tuple(Swizzling::KEY_5, Swizzling::KEY_5, out_0_norm);
        //    } else {  // both 0
        //        resulted_swizz = std::make_tuple(Swizzling::KEY_0, Swizzling::KEY_0, out_0_norm);
        //    }

        //} else {  // only 5 is possible
        //    resulted_swizz = std::make_tuple(Swizzling::KEY_5, Swizzling::KEY_5, Swizzling::KEY_5);
        //}

        if (NN27InputAdapter::weights_always_zero.count(op) != 0) {
            std::get<1>(resulted_swizz) = Swizzling::KEY_0;
        }
        return resulted_swizz;
    }
};

/// adapting input to NPU40
class NN5XInputAdapter {
public:
    /// ISI Strategy (up to v11) might not be compatible in any combination. Reasons are rather based on data
    /// available for training and training over-fitting.
    ///
    /// a) CLUSTERING + OWT=2+ : not possible,           :replaced with SOK+OWT=2+ (both do no use input HALO),
    /// filter with step b) next
    ///
    /// b) SOK + ELEMENTWISE   : not possible to profile : replace with CLU+OWT=1  (slightly smaller then real
    /// due to owt=1),
    ///
    /// c) [SOH] :invalid in : replace  with CLU , ,
    /// filter with a) next.
    ///
    /// d) limit owt to 6! Even if NPU40 supports more
    ///
    /// order of calls: c, a, b , d
    ///
    static FilteredFields avoid_untrained_space(const DPUWorkload& w) {
        FilteredFields ff{w.isi_strategy, w.output_write_tiles};

        // auto a0_SOK_NA_temporarly = [](FilteredFields& f) {  // a0)
        //     if (ISIStrategy::SPLIT_OVER_K == f.isi) {
        //         f.isi = ISIStrategy::CLUSTERING;  // SOK not trained, replace with CLU
        //         f.owt = 1;                        // OWT=1 forcefully
        //     }
        // };

        auto a_check_CLU_2 = [](FilteredFields& f) {  // a)
            if ((ISIStrategy::CLUSTERING == f.isi) && (1 < f.owt)) {
                f.isi = ISIStrategy::SPLIT_OVER_K;  // SOK is replacing CLU for OWT>1,, next should be check for b)
            }
        };

        // NEW ELEMENTWISE SUPPORTS SOK
        // auto b_check_SOK_ELM = [&w](FilteredFields& f) {  // b)
        //     if ((ISIStrategy::SPLIT_OVER_K == f.isi) && (Operation::ELTWISE == w.op)) {
        //         f.isi = ISIStrategy::CLUSTERING;
        //         // should go to state a) check
        //         // a_check_CLU_2(f); do not call A, you will go in circles
        //         // change  owt to be compatible with  a) rule
        //         f.owt = 1;
        //     }
        // };

        auto c_check_SOH_INVALID = [](FilteredFields& f) {  // c)
            if (ISIStrategy::SPLIT_OVER_H == f.isi) {
                f.isi = ISIStrategy::CLUSTERING;
                // should go to state a) afterwards
            }
        };

        // limit owt to 6. This was trained on NPU40
        auto d_limit_owt_to_3 = [](FilteredFields& f) {  // d)
            if (3 < f.owt) {
                f.owt = 3;
            }
        };
        // In practice was observed that OWT=2,3,4,5,6 has the same constant impact

        // ff is the data  holder changed in the next chain of calls

        c_check_SOH_INVALID(ff);
        a_check_CLU_2(ff);
        // b_check_SOK_ELM(ff);  // ELM with broadcast is not allowed/supported . Is this a problem? SHuld be
        // trainable!?

        d_limit_owt_to_3(ff);  // only

        return ff;
    }
};

}  // namespace VPUNN
#endif
