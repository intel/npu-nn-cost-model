// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef POSTPROCESSING_MOCKS_H
#define POSTPROCESSING_MOCKS_H

#include "post_process.h"

namespace VPUNN {

class IPostProcess {
public:
    virtual float process(const DPUWorkload&, const float& nn_val) const = 0;
    virtual ~IPostProcess() = default;
    IPostProcess() = default;

protected:
    /// 4 billion, any value higher than this might not be representable on UINT32, and
    /// should be treated like a not in range value given by the NN
    static constexpr float high_threshold{4000000000.0F};

    /// less than this is not representable on UINT32, and has no meanings in
    /// cycles. zero is still left possible to be returned, it might be a special
    /// way of network to communicate something (like no answer)
    static constexpr float low_threshold{0.0F};

public:
    /// @brief checks if the NN returned value is invalid, is outside of usable range
    /// @param nn_output_cycles , the value to be analyzed, this is assumed given by the NN inference
    /// @return true if invalid value
    static bool is_NN_value_invalid(const float nn_output_cycles) noexcept {
        bool validity = false;
        if ((nn_output_cycles > high_threshold) || (nn_output_cycles < low_threshold)) {
            validity = true;
        }
        return validity;
    }
    /// @brief provides the value interval where the NN raw outputs are considered valid and will be used to further
    /// compute information
    ///
    /// @returns a pair containing (minimum_valid_value maximum_valid_value)
    static std::pair<float, float> get_NN_Valid_interval() noexcept {
        return std::make_pair(low_threshold, high_threshold);
    }
};
/// @brief A pass through post processing, does nothing, just returns the NN value
class PassThroughPostP : public IPostProcess {
public:
    PassThroughPostP() = default;
    float process(const DPUWorkload&, const float& nn_val) const override {
        return nn_val;
    }
};

/// @brief post process cycle times obtained from a NPU2.7 model.
/// For now adapts to NPU4.0 and above, by applying a factor to the cycles
class AdaptFromNPU27to40 : public IPostProcess {
public:
    AdaptFromNPU27to40() = default;
    /// @brief scales the cycles based on the workload and the NN value.
    /// Crt implementation treats only DW_CONV optimization of NPU4.0 vs NPU2.7
    float process(const DPUWorkload& w, const float& nn_val) const override {
        // Mock implementation of the process method
        if (w.device >= VPUDevice::VPU_4_0) {
            float processed{nn_val};
            if ((w.op == Operation::DW_CONVOLUTION) && !is_NN_value_invalid(nn_val)) {
                const float factor{getDWFactor(w)};  // neutral
                processed *= factor;
            }
            return processed;
        } else {
            return nn_val;
        }
    }

private:
    constexpr static int ch16{0};  // first index
    constexpr static int ch32{1};  // second index
    constexpr static int ch64{2};  // third index
    constexpr static int chInvalid{-1};

    constexpr static int channelsToIndex(const int ch) {
        switch (ch) {
        case 16:
            return ch16;
        case 32:
            return ch32;
        case 64:
            return ch64;
        default:
            return chInvalid;
        }
    }

    constexpr static int kToIndex0to2(const int kw, const int kh) {
        if ((kw == 3) /* && (kh == 3) */) {  //?x3 was before discovering the Fathom issue. now is 3x?
            return 1;                       // optimized row
        } else if ((kw == 1) || (kh == 1)) {
            // 3x1 yes,
            // 1x3 no, is optimized!?
            return 0;  // first (zeroth) row
        } else {
            return 2;  // rest of the world , non 1 kernel
        }
    }

    constexpr static int contextToIndex(const DPUWorkload& wl) {
        if ((wl.strides[0] == 1) && (wl.strides[1] == 1)) {  // stride is 1x1
            // kernel based
            return kToIndex0to2(wl.kernels[Dim::Grid::W], wl.kernels[Dim::Grid::H]);
        }

        if ((wl.strides[0] == 2) || (wl.strides[1] == 2)) {  // any stride is 2
            return 3;                                        // common data for uncommon stride
        }

        return 4;  // stride is larger than 2 on all dimensions
    }

    using ChannelsFactors = std::array<float, 3>;           ///< factors for each valid channels value 16, 32, 64
    using DWFactorsArray = std::array<ChannelsFactors, 5>;  ///< sets of factors , one per row ,
                                                            /// small(k=1),
                                                            /// OPTIMIZED k=?x3,
                                                            /// NOT OPTIMIZED  k !=3x, stride still 1x1
                                                            /// stride 2x? sau ?x2  NOT OPTIMIZED
                                                            /// stride Larger (NOT 2x?/?x2) NOT OPTIMIZED
                                                            // to be updated

    // static constexpr DWFactorsArray factor_array_8Bit{  //v1.6.5
    //        ChannelsFactors{1.0f, 1.0f, 1.0f},     // k=1, at least one dim, bad
    //        ChannelsFactors{0.6f, 0.5f, 0.74f},    // K =3 optimum
    //        ChannelsFactors{0.83f, 0.75f, 0.77f},  // rest of the world , square kernels, asymmetric kernels
    //                                               // stride !=1 , kernel irrelevant?
    //};

    //// v1.6.6
    // static constexpr DWFactorsArray factor_array_8Bit{
    //         ChannelsFactors{1.0f, 0.8f, 0.75f},    // k=1, at least one dim, bad
    //         ChannelsFactors{0.5f, 0.45f, 0.70f},   // K =3 optimum
    //         ChannelsFactors{0.81f, 0.75f, 0.75f},  // rest of the world , square kernels, asymmetric kernels
    //         ChannelsFactors{0.69f, 0.68f, 0.73f},  // stride !=1
    // };

    // v1.6.98p1
    static constexpr DWFactorsArray factor_array_8Bit{
            ChannelsFactors{1.0f, 0.8f, 0.75f},    // k=1, at least one dim, bad
            ChannelsFactors{0.5f, 0.44f, 0.69f},   //  OPTIMIZED k=?x3 ,
            ChannelsFactors{0.81f, 0.75f, 0.75f},  // NOT OPTIMIZED  k !=3x, stride still 1x1
            ChannelsFactors{0.69f, 0.68f, 0.73f},  // stride 2x? sau ?x2  NOT OPTIMIZED
            ChannelsFactors{0.60f, 0.60f, 0.65f},  // stride Larger (NOT 2x?/?x2) NOT OPTIMIZED
    };

    //// v1.6.5
    // static constexpr DWFactorsArray factor_array_16Bit{
    //         ChannelsFactors{1.0f, 1.0f, 1.0f},     // k=1, at least one dim, bad
    //         ChannelsFactors{0.5f, 0.74f, 0.74f},   // K =3 optimum
    //         ChannelsFactors{0.75f, 0.77f, 0.77f},  // rest of the world, square kernels, asymmetric kernels
    //                                                // stride !=1 , kernel irrelevant?
    // };

    //// v1.6.6
    // static constexpr DWFactorsArray factor_array_16Bit{
    //         ChannelsFactors{1.0f, 1.0f, 1.0f},     // k=1, at least one dim, bad
    //         ChannelsFactors{0.5f, 0.49f, 0.73f},   // K =3 optimum
    //         ChannelsFactors{0.76f, 0.75f, 0.78f},  // rest of the world, square kernels, asymmetric kernels
    //         ChannelsFactors{0.69f, 0.72f, 0.75f},  // stride !=1
    // };

    // v1.6.98p1
    static constexpr DWFactorsArray factor_array_16Bit{
            ChannelsFactors{1.0f, 1.0f, 1.0f},     // k=1, at least one dim, bad
            ChannelsFactors{0.48f, 0.47f, 0.71f},  //  OPTIMIZED k=?x3 ,
            ChannelsFactors{0.76f, 0.74f, 0.77f},  // NOT OPTIMIZED  k !=3x, stride still 1x1
            ChannelsFactors{0.69f, 0.69f, 0.73f},  // stride 2x? sau ?x2  NOT OPTIMIZED
            ChannelsFactors{0.60f, 0.60f, 0.65f},  // stride Larger (NOT 2x?/?x2) NOT OPTIMIZED
    };

    constexpr static const DWFactorsArray& getTypeBasedFactorsArray(const DataType t) {
        if (dtype_to_bytes(t) > 1) {
            return factor_array_16Bit;//or more than 16bit
        } else {
            return factor_array_8Bit;
        }
    }

    /// @brief gets a factor for a DW CONV operation.
    /// The factor is obtained from pre-filled factors arrays depending on type (of input), channels and kernel size
    ///
    /// @precondition the workload is a DW_CONV, well formed
    ///
    /// @param w the workload to be analyzed
    ///
    /// @returns the factor to be applied to the cycles, 1.0 as default
    static float getDWFactor(const DPUWorkload& w) {
        const DWFactorsArray& factors = getTypeBasedFactorsArray(w.inputs[0].get_dtype());
        const auto ch_index = channelsToIndex(w.inputs[0].channels());

        if (ch_index >= 0) {
            const auto k_index = contextToIndex(w);
            return (factors[k_index])[ch_index];
        } else {
            return 1.0f;  // nothing
        }
    }
};

/// @brief post process cycle times obtained from a NPU40model.
/// For now adapts to NPU4.0 and above since we have gaps in trained space.
///  CM conv not covered, replaced by CONV with IC=16
class AdaptFromNPU40to40 : public IPostProcess {
public:
    AdaptFromNPU40to40() = default;
    /// @brief scales the cycles based on the workload and the NN value.
    /// Crt implementation treats only CM_CONV untrained space
    float process(const DPUWorkload& w, const float& nn_val) const override {
        // Mock implementation of the process method
        if (w.device >= VPUDevice::VPU_4_0) {
            float processed{nn_val};
            //if ((w.op == Operation::CM_CONVOLUTION) && !is_NN_value_invalid(nn_val)) {
            //    const float factor{getCMFactor(w)};  // neutral
            //    processed *= factor;
            //}
            return processed;
        } else {
            return nn_val;
        }
    }

private:
    /// @brief gets a factor for a DW CONV operation.
    /// depends only on output channels number (multiple of 16) and input channels number
    ///
    /// @precondition the workload is a CM_CONV
    ///
    /// @param w the workload to be analyzed
    ///
    /// @returns the factor to be applied to the cycles, 1.0 as default
    static float getCMFactor(const DPUWorkload& w) {
        // based on output channels
        const auto oc{w.outputs[0].channels()};
        const auto ic{w.inputs[0].channels()};
        const auto isMoreBytes{dtype_to_bytes(w.outputs[0].get_dtype()) > 1 ? true : false};
        if (ic <= 4) {             // 1,2,3,4 input channels, apply different factors
            if (oc <= 16) {        // 16ch
                return 1.0f;       // float or INT8
            } else if (oc < 64) {  // 32,48 ch
                if (isMoreBytes) {
                    return (2.0f / 3.0f) + 0.30f;
                } else {
                    return 2.0f / 3.0f;
                }
            } else {  // 64 or more  output channels
                if (isMoreBytes) {
                    return (1.0f / 3.0f) + 0.22f;
                } else {
                    return 1.0f / 3.0f;
                }
            }
        } else {  // 5..15 input channels,  factor is 1.0
            return 1.0f;
        }
    }
};

}  // namespace VPUNN
#endif
