// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef SHAVEMODELNORMALIZEL2_H
#define SHAVEMODELNORMALIZEL2_H

#include <iostream>
#include <list>
#include <random>
#include <vector>

#include "shave_equations.h"
#include "shave_model_basics.h"
#include "vpu/cycles_interface_types.h"
#include "vpu/datatype_collection_size.h"
#include "vpu/types.h"

namespace VPUNN {
/**
 * @brief This implementation reflects the NormalizeL2 activation operation time when the selected axes for
 * normalization are only the Channels
 */
class NormalizeL2OnlyC : public ShaveCyclesProvider<NormalizeL2OnlyC> {
private:
    const DataType dtype_;  ///< the data type of the output ()

    FirstDegreeEquation BaseTimeEquation;  ///< This represents the base slope with an 8 unroll hardcoded
                                           ///< in the compiler code from where the composition of slopes begins.
    float BaseVectorOffset;                ///< represents the vector offset of the base unroll. It is calculated
                                           ///< an offset between Points that are note %8

    /*
     represents the time increase of the base slope formed from %16 numbers since we have an unroll for 16 harcoded
     specific for LNL after the elements are not enough to be on a %16 it will try %8(W8Slope) then remaining elements.
     For the remaining elements time is impacted based on being after mod16(mod16 = 1-7) or after mod8(mod16 = 9-15)
    */
    FirstDegreeEquation BaseWTimeEquation;
    float SlopeIncreaseMod1;
    float SlopeIncreaseMod8;
    float SlopeIncreaseMod9;

    float WVectorOffset;  ///< is an vector offset observed on the numbers where mod16 is not 0,1,8 or 9.

    friend std::ostream& operator<<(std::ostream& stream, const VPUNN::NormalizeL2OnlyC& d);

public:
    /**
     * @brief Construct a new NormalizeL2OnlyC object
     */
    NormalizeL2OnlyC(DataType dtype, float baseTimeSlope, float baseTimeIntercept, float baseVectorOffset,
                     float baseTimeSlopeW, float baseTimeInterceptW, float slopeW1, float slopeW8, float slopeW9,
                     float baseVectorOffsetW, unsigned int DpuFreq, unsigned int ShvFreq)
            : ShaveCyclesProvider<NormalizeL2OnlyC>{DpuFreq, ShvFreq},
              dtype_(dtype),
              BaseTimeEquation{baseTimeSlope, baseTimeIntercept},
              BaseVectorOffset(baseVectorOffset),
              BaseWTimeEquation{baseTimeSlopeW, baseTimeInterceptW},
              SlopeIncreaseMod1(slopeW1),
              SlopeIncreaseMod8(slopeW8),
              SlopeIncreaseMod9(slopeW9),
              WVectorOffset(baseVectorOffsetW) {
    }

protected:
    /**
     * @brief Calculates the time for the base slope based on the number of elements located in channels
     *
     * @return the time in us for for the base time
     */
    float baseTimeCalculator(const int& channels_elements) const {
        return BaseTimeEquation(channels_elements) + (channels_elements % 8) * BaseVectorOffset;
    }

    /**
     * @brief Calculates the time that is needed in order to add all the elements based on the optimizations made on the
     * vectorial calculus over Width. The base slope represents the slope of 16 elements. Every other slope has an
     * increase that is also influenced by the number of channels.
     *
     * @return the time in us that needs to be added to the base time.
     */
    float wTimeIncrease(const int& channels_elements, const int& width_elements, const int& rem_elements) const {
        float inc_time = BaseWTimeEquation(channels_elements);

        int mod16W = width_elements % 16;
        int mod8W = width_elements % 8;

        inc_time += (mod16W == 8) ? SlopeIncreaseMod8 * channels_elements : 0;
        inc_time += (mod16W >= 1 && mod16W < 8) ? SlopeIncreaseMod1 * channels_elements : 0;
        inc_time += (mod16W >= 9 && mod16W < 16) ? SlopeIncreaseMod9 * channels_elements : 0;

        float vecTimeW = (mod8W != 0) ? (mod8W - 1) * WVectorOffset * channels_elements : 0;
        float no_of_blocks = ((width_elements / 16) != 0) ? static_cast<float>(width_elements) / 16 : 1.0f;
        return (inc_time * no_of_blocks + vecTimeW) * rem_elements;
    }

public:
    /**
     * @brief Get the time in us for the the activation based on the output size in bytes
     *
     * @return the time in us
     */
    float getMicroSeconds(const int& channels_elements, const int& width_elements, const int& rem_elements) const {
        return baseTimeCalculator(channels_elements) + wTimeIncrease(channels_elements, width_elements, rem_elements);
    }
};

inline std::ostream& operator<<(std::ostream& stream, const VPUNN::NormalizeL2OnlyC& d) {
    stream << "NormalizeL2OnlyC: \n"                                                                              //
           << " dtype: \t" << (int)d.dtype_ << " : " << DataType_ToText.at(static_cast<int>(d.dtype_)) << " ;\n"  //
           << " base time equation: [slope,intercept]  \t{" << d.BaseTimeEquation.slope_ << ","                   //
           << d.BaseTimeEquation.intercept_ << "} ;\n"                                                            //
           << " base vector offset: \t" << d.BaseVectorOffset << " ;\n"                                           //
           << "W base time equation: [slope,intercept]  \t{" << d.BaseWTimeEquation.slope_ << ","                 //
           << d.BaseWTimeEquation.intercept_ << "} ;\n"                                                           //
           << " W vector offset: \t" << d.WVectorOffset << " ;\n"                                                 //
           << " Mod1 Slope: \t" << d.SlopeIncreaseMod1 << " ;\n"                                                  //
           << " Mod8 Slope: \t" << d.SlopeIncreaseMod8 << " ;\n"                                                  //
           << " Mod9 Slope: \t" << d.SlopeIncreaseMod9 << " ;\n"                                                  //
           << d.converter                                                                                         //
           << out_terminator() << "ShaveModel1to1 "  // terminator
            ;
    return stream;
}

}  // namespace VPUNN

#endif /* SHAVEMODEL1to1_H*/