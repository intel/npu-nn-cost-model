// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef SHAVE_OP_EXECUTORS_H
#define SHAVE_OP_EXECUTORS_H

#include <type_traits>
#include "interface_shave_op_executor.h"

#include "GatherModel.h"
#include "MVNModel.h"
#include "NormalizeL2Model.h"
#include "ShaveModel1to1.h"
#include "SoftmaxModel.h"
#include "elementwise.h"
#include "poly_models.h"

#include "vpu/cycles_interface_types.h"
#include "vpu/types.h"

#include <sstream>

namespace VPUNN {

/// @brief Executor around the linear with steps model where the input  variable is the size of output
template <DataType dtype, unsigned int VectorSize, unsigned int UnrollSize, unsigned int DpuFreq, unsigned int ShvFreq>
class ShaveActivation1on1 : public ShaveOpExecutor {
private:
    ShaveModel1to1 model;  ///< model instance

public:
    CyclesInterfaceType dpuCycles(const SHAVEWorkload& w) const override {
        return dpuCycles(w, model.getNominalDPUFrq(), model.getNominalSHVFrq());
    };

    CyclesInterfaceType dpuCycles(const SHAVEWorkload& w, const int present_dpu_frq,
                                  const int present_shv_frq) const override {
        const auto& out = w.get_outputs()[0];  // the only output
        // what happens if datatype is not anymore as the model ??
        const auto cycles = model.getDPUCyclesAnotherFreqDPU_SHV(present_dpu_frq, present_shv_frq, out.size());
        return cycles;
    };

    ShaveActivation1on1(const std::string& name, float slope, float intercept, float offset_scalar, float offset_unroll)
            : ShaveOpExecutor(name),
              model(dtype, slope, intercept, offset_scalar, offset_unroll, VectorSize, UnrollSize, DpuFreq, ShvFreq) {
    }
    std::string toString() const override {
        std::stringstream stream;
        stream << "ShaveActivation1on1: \n"                //
               << " Operation: \t" << getName() << " ;\n"  //
               << " Model    : \t" << model << " ;\n";     //

        return stream.str();
    }
};

/// TODO: Duplicate code, might solve this in future
/// @brief Executor around the linear with steps model where the input  variable is the size of output
template <DataType dtype, unsigned int VectorSize, unsigned int UnrollSize, unsigned int DpuFreq, unsigned int ShvFreq>
class ShaveActivation1on1NPU40 : public ShaveOpExecutor {
private:
    ShaveModel1to1NPU40 model;  ///< model instance

public:
    CyclesInterfaceType dpuCycles(const SHAVEWorkload& w) const override {
        return dpuCycles(w, model.getNominalDPUFrq(), model.getNominalSHVFrq());
    };

    CyclesInterfaceType dpuCycles(const SHAVEWorkload& w, const int present_dpu_frq,
                                  const int present_shv_frq) const override {
        const auto& out = w.get_outputs()[0];  // the only output
        // what happens if datatype is not anymore as the model ??
        const auto cycles = model.getDPUCyclesAnotherFreqDPU_SHV(present_dpu_frq, present_shv_frq, out.size());
        return cycles;
    };

    ShaveActivation1on1NPU40(const std::string& name, float slope, float intercept, float offset_unroll,
                             float intra_block_offset, float vector_offset, int displacement_size)
            : ShaveOpExecutor(name),
              model(dtype, slope, intercept, offset_unroll, intra_block_offset, vector_offset, displacement_size,
                    VectorSize, UnrollSize, DpuFreq, ShvFreq) {
    }
    std::string toString() const override {
        std::stringstream stream;
        stream << "ShaveActivation1on1NPU40: \n"           //
               << " Operation: \t" << getName() << " ;\n"  //
               << " Model    : \t" << model << " ;\n";     //

        return stream.str();
    }
};
/// Initial implementation for Softmax with N=1

template <DataType dtype, unsigned int VectorSize, unsigned int UnrollSize, unsigned int DpuFreq, unsigned int ShvFreq>
class SoftmaxActivationExec : public ShaveOpExecutor {
private:
    SoftmaxModel model;

public:
    CyclesInterfaceType dpuCycles(const SHAVEWorkload& w) const override {
        return dpuCycles(w, model.getNominalDPUFrq(), model.getNominalSHVFrq());
    };

    CyclesInterfaceType dpuCycles(const SHAVEWorkload& w, const int present_dpu_frq,
                                  const int present_shv_frq) const override {
        // Check for the parameter 1 that tells us the selected axis
        if (w.get_params().size() != getNumExpectedParams()) {
            return Cycles::ERROR_SHAVE_PARAMS;
        }
        const int selected_dimension{std::get<int>(w.get_params()[0])};

        // Selected dimension should be C(1), H(2) or W(3)
        // For now Batch(0) is not supported
        if ((selected_dimension <= 0) || (selected_dimension > 3)) {
            return Cycles::ERROR_SHAVE_PARAMS;
        }

        const auto& out = w.get_outputs()[0];  // the only output
        // what happens if datatype is not anymore as the model ??

        // Layout WHCB is the only one supported
        if (out.get_layout() != Layout::XYZ) {
            return Cycles::ERROR_SHAVE_LAYOUT;
        }

        // Supporting only with batch = 1
        if (out.b() > 1) {
            return Cycles::ERROR_SHAVE_INVALID_INPUT;
        }

        // Since the order is reversed we will need to substract from three the value to extract the exact dimension
        const auto& select_index = 3 - selected_dimension;
        const int& selected_dimension_volume = out.get_shape()[select_index];

        int unselected_volume = 1;

        // Since batch is considered always one it is not necessary to go over it
        for (int i = 0; i < 3; i++) {
            if (select_index != i)
                unselected_volume *= out.get_shape()[i];
        }

        const auto cycles = model.getDPUCyclesAnotherFreqDPU_SHV(present_dpu_frq, present_shv_frq,
                                                                 selected_dimension_volume, unselected_volume);

        return cycles;
    };

    SoftmaxActivationExec(const std::string& name, float baseSlope, float baseIntercept, SoftmaxEquationParams e1,
                          SoftmaxEquationParams e2, SoftmaxEquationParams e4, SoftmaxEquationParams e8,
                          SoftmaxEquationParams e16, SoftmaxEquationParams e32)
            : ShaveOpExecutor(name, 1),
              model(dtype, baseSlope, baseIntercept, e1, e2, e4, e8, e16, e32, DpuFreq, ShvFreq) {
    }

    std::string toString() const override {
        std::stringstream stream;
        stream << "SoftmaxActivationExec: \n"                                                            //
               << " Operation: \t" << getName() << " ;\n"                                                //
               << " Shave params: 1 int, represents the selected dimension(N(0), C(1), H(2), W(3))"      //
               << " Layout restrictions: If Layout different from XYZ will retrieve ERROR_SHAVE_LAYOUT"  //
               << " Input restrictions: the batch size over 1 is not supported"                          //
               << " Model    : \t" << model << " ;\n";                                                   //

        return stream.str();
    }
};
/// Implementation for GatherModel 
template <DataType dtype, unsigned int VectorSize, unsigned int DpuFreq, unsigned int ShvFreq>
class GatherActivationExec : public ShaveOpExecutor {
private:
	GatherModel model;

public:
        GatherActivationExec(const std::string& name, float base_slope, float base_intercept, float inter_slope, 
                             float worst_slope, float vector_offset)
                            : ShaveOpExecutor(name, 2),
                              model(dtype, base_slope,base_intercept, worst_slope, inter_slope, vector_offset, 
                              VectorSize, DpuFreq, ShvFreq){
                              }
        
        CyclesInterfaceType dpuCycles(const SHAVEWorkload& w) const override {
            return dpuCycles(w, model.getNominalDPUFrq(), model.getNominalSHVFrq());
        };

        CyclesInterfaceType dpuCycles(const SHAVEWorkload& w, const int present_dpu_frq,
                                      const int present_shv_frq) const override {
            // We need to have 2 parameters, the first one is the axis and the second one is the batch dims
            if (w.get_params().size() != getNumExpectedParams()) {
				return Cycles::ERROR_SHAVE_PARAMS;
			}
            
            const int selected_axis{std::get<int>(w.get_params()[0])};
            const int batch_dims{std::get<int>(w.get_params()[1])};
            
            // The current model was trained just in this way
            if (selected_axis != 1 || batch_dims != 1) {
                return Cycles::ERROR_SHAVE_PARAMS;
            }
            const auto& out = w.get_outputs()[0];
            const auto& out_volume = out.volume();

            const auto layout_dim_order{layout_to_order(out.get_layout())};  // dim from innermost to outermost
            assert(layout_dim_order.size() == 4);

            GatherModel::Dimensions dim_vector{1, 1, 1, 1};
            for (int i = 0; i < 4; ++i) {  // take  the first DimSelected innermost dimensions, they are the selected one
                const auto dimension_index = layout_dim_order[i];
                const auto dimension_size{out.get_shape()[dimension_index]};
                dim_vector[i] = dimension_size;
            }

            const auto cycles = model.getDPUCyclesAnotherFreqDPU_SHV(
                    present_dpu_frq, present_shv_frq, out_volume, dim_vector);

            return cycles;
        }
        std::string toString() const override {
        std::stringstream stream;
        stream << "GatherActivationExec: \n"                                                            //
               << " Operation: \t" << getName() << " ;\n"                                                //
               << " Shave params: 2 int, first represents the axis(dimension index for gathering),"      //
               << " second parameter is batch_dims(leading number of dimensions being batches)"          //
               << " Both parameters are restricted to value 1,(the only profiled selection)"             //
               << " Model    : \t" << model << " ;\n";                                                   //

        return stream.str();
    }
};
/// Implementation for NormalizeL2 only C
template <DataType dtype, unsigned int VectorSize, unsigned int UnrollSize, unsigned int DpuFreq, unsigned int ShvFreq>
class NormalizeL2OnlyCActivationExec : public ShaveOpExecutor {
private:
    NormalizeL2OnlyC model;

public:
    CyclesInterfaceType dpuCycles(const SHAVEWorkload& w) const override {
        return dpuCycles(w, model.getNominalDPUFrq(), model.getNominalSHVFrq());
    };

    CyclesInterfaceType dpuCycles(const SHAVEWorkload& w, const int present_dpu_frq,
                                  const int present_shv_frq) const override {
        // Check for the parameter 1 that tells us the selected axis
        if (w.get_params().size() != getNumExpectedParams()) {
            return Cycles::ERROR_SHAVE_PARAMS;
        }
        const int selected_dimension{std::get<int>(w.get_params()[0])};

        // Selected dimension should be C(1)
        // For now the rest combinations of selected dimensions are not supported
        if ((selected_dimension != 1)) {
            return Cycles::ERROR_SHAVE_PARAMS;
        }

        const auto& out = w.get_outputs()[0];  // the only output
        // what happens if datatype is not anymore as the model ??

        // Layout WHCB is the only one supported
        if (out.get_layout() != Layout::XYZ) {
            return Cycles::ERROR_SHAVE_LAYOUT;
        }

        // Since we have a standard order we can extract the necessary dimension directly by index WHCB
        const int& no_channels_elements = out.get_shape()[2];
        const int& no_width_elements = out.get_shape()[0];
        const int& no_rem_elements = out.get_shape()[1] * out.get_shape()[3];

        const auto cycles = model.getDPUCyclesAnotherFreqDPU_SHV(present_dpu_frq, present_shv_frq, no_channels_elements,
                                                                 no_width_elements, no_rem_elements);

        return cycles;
    };

    NormalizeL2OnlyCActivationExec(const std::string& name, float baseTimeSlope, float baseTimeIntercept,
                                   float baseVectorOffset, float baseTimeSlopeW, float baseTimeInterceptW,
                                   float slopeW1, float slopeW8, float slopeW9, float baseVectorOffsetW)
            : ShaveOpExecutor(name, 1),
              model(dtype, baseTimeSlope, baseTimeIntercept, baseVectorOffset, baseTimeSlopeW, baseTimeInterceptW,
                    slopeW1, slopeW8, slopeW9, baseVectorOffsetW, DpuFreq, ShvFreq) {
    }

    std::string toString() const override {
        std::stringstream stream;
        stream << "NormalizeL2OnlyCExec: \n"                                                                 //
               << " Operation: \t" << getName() << " ;\n"                                                    //
               << " Shave params: 1 int, represents the selected dimension(C(1) is the only one supported)"  //
               << " Layout restrictions: If Layout different from XYZ will retrieve ERROR_SHAVE_LAYOUT"      //
               << " Model    : \t" << model << " ;\n";                                                       //

        return stream.str();
    }
};
/// Initial implementation for only one axis
template <DataType dtype, unsigned int VectorSize, unsigned int UnrollSize, unsigned int DpuFreq, unsigned int ShvFreq>
class MVN6OneAxisActivationExec : public ShaveOpExecutor {
private:
    MVN6OneAxisModel model;

public:
    CyclesInterfaceType dpuCycles(const SHAVEWorkload& w) const override {
        return dpuCycles(w, model.getNominalDPUFrq(), model.getNominalSHVFrq());
    };

    CyclesInterfaceType dpuCycles(const SHAVEWorkload& w, const int present_dpu_frq,
                                  const int present_shv_frq) const override {
        const auto& out = w.get_outputs()[0];  // the only output
        // what happens if datatype is not anymore as the model ??

        const auto output_samples{out.volume()};

        const auto layout_dim_order{layout_to_order(out.get_layout())};  // dim from innermost to outermost
        const auto innermost_dimension_index = layout_dim_order[0];      // innermost dimension
        const auto innermost_dimension_size{out.get_shape()[innermost_dimension_index]};

        const auto cycles = model.getDPUCyclesAnotherFreqDPU_SHV(present_dpu_frq, present_shv_frq, output_samples,
                                                                 innermost_dimension_size);
        return cycles;
    };

    MVN6OneAxisActivationExec(const std::string& name, float slope, float intercept, float alpha,
                              float maxmium_diff_slope /*,float offset_scalar, float offset_unroll*/)
            : ShaveOpExecutor(name),
              model(dtype, slope, intercept, alpha, maxmium_diff_slope,
                    /* offset_scalar, offset_unroll, VectorSize, UnrollSize,*/ DpuFreq, ShvFreq) {
    }

    std::string toString() const override {
        std::stringstream stream;
        stream << "MVN6OneAxisActivationExec: \n"          //
               << " Operation: \t" << getName() << " ;\n"  //
               << " Model    : \t" << model << " ;\n";     //

        return stream.str();
    }
};

struct MVN6Parameters {
    float slope;
    float intercept;
    float alpha;
    float worst_case_slope;
    float slope_delta_diff;
};

template <DataType dtype, unsigned int VectorSize, unsigned int UnrollSize, unsigned int DpuFreq, unsigned int ShvFreq,
          int DimSelected>
class MVN6MultiAxisActivationExec : public ShaveOpExecutor {
    static_assert(DimSelected >= 1, "Dimension must be in range 1,2,3,4, now is too small");
    static_assert(DimSelected <= 4, "Dimension must be in range 1,2,3,4, now is too large");

private:
    MVN6MultiAxisModel model;

public:
    CyclesInterfaceType dpuCycles(const SHAVEWorkload& w) const override {
        return dpuCycles(w, model.getNominalDPUFrq(), model.getNominalSHVFrq());
    };

    CyclesInterfaceType dpuCycles(const SHAVEWorkload& w, const int present_dpu_frq,
                                  const int present_shv_frq) const override {
        const auto& out = w.get_outputs()[0];  // the only output
        // what happens if datatype is not anymore as the model ??

        const auto output_samples{out.volume()};  // total volume!

        const auto layout_dim_order{layout_to_order(out.get_layout())};  // dim from innermost to outermost
        assert(layout_dim_order.size() == 4);

        const int unselectedDim{1};
        MVN6MultiAxisModel::Dimensions selected_axes{unselectedDim, unselectedDim, unselectedDim, unselectedDim};
        for (int i = 0; i < DimSelected;
             ++i) {  // take  the first DimSelected innermost dimensions, they are the selected one
            const auto dimension_index = layout_dim_order[i];
            const auto dimension_size{out.get_shape()[dimension_index]};
            selected_axes[i] = dimension_size;
        }
        // rest of dimension up to 4, remain with default value (0 or 1);

        const auto cycles =
                model.getDPUCyclesAnotherFreqDPU_SHV(present_dpu_frq, present_shv_frq, output_samples, selected_axes);
        return cycles;
    };

    /// ctor with param structure
    MVN6MultiAxisActivationExec(const std::string& name, MVN6Parameters params)
            : ShaveOpExecutor(name),
              model(dtype, params.slope, params.intercept, params.alpha, params.worst_case_slope,
                    params.slope_delta_diff, DpuFreq, ShvFreq) {
    }

    /// ctor with individual params
    MVN6MultiAxisActivationExec(const std::string& name, float slope, float intercept, float alpha,
                                float worst_case_slope, float slope_delta_diff)
            : ShaveOpExecutor(name),
              model(dtype, slope, intercept, alpha, worst_case_slope, slope_delta_diff, DpuFreq, ShvFreq) {
    }

    std::string toString() const override {
        std::stringstream stream;
        stream << "MVN6MultiAxisActivationExec: \n"               //
               << " #Axes selected: \t" << DimSelected << " ;\n"  //
               << " Operation: \t" << getName() << " ;\n"         //
               << " Model    : \t" << model << " ;\n";            //

        return stream.str();
    }
};

template <DataType dtype, unsigned int VectorSize, unsigned int UnrollSize, unsigned int DpuFreq, unsigned int ShvFreq>
class MVN6GenericActivationExec : public ShaveOpExecutor {
private:
    std::array<MVN6MultiAxisModel, 4> models;

public:
    CyclesInterfaceType dpuCycles(const SHAVEWorkload& w) const override {
        return dpuCycles(w, 0, 0, true);  // nominal, ignore the zeros
    };
    CyclesInterfaceType dpuCycles(const SHAVEWorkload& w, const int present_dpu_frq,
                                  const int present_shv_frq) const override {
        return dpuCycles(w, present_dpu_frq, present_shv_frq, false);  // use dpu & shv freq
    };

    CyclesInterfaceType dpuCycles(const SHAVEWorkload& w, const int present_dpu_frq, const int present_shv_frq,
                                  bool nominal_freqs) const {
        // for how many axes? Nr of axes is param 1, as int
        if (w.get_params().size() < getNumExpectedParams()) {
            return Cycles::ERROR_SHAVE_PARAMS;
        }
        const int selected_dim_number{std::get<int>(w.get_params()[0])};

        if ((selected_dim_number <= 0) || (selected_dim_number > 4)) {
            return Cycles::ERROR_SHAVE_PARAMS;
        }

        const auto& out = w.get_outputs()[0];  // the only output
        // what happens if datatype is not anymore as the model ??

        const auto output_samples{out.volume()};  // total volume!

        const auto layout_dim_order{layout_to_order(out.get_layout())};  // dim from innermost to outermost
        assert(layout_dim_order.size() == 4);

        const int unselectedDim{1};
        MVN6MultiAxisModel::Dimensions selected_axes{unselectedDim, unselectedDim, unselectedDim, unselectedDim};
        for (int i = 0; i < selected_dim_number;
             ++i) {  // take  the first selected_dim_number innermost dimensions, they are the selected one
            const auto dimension_index = layout_dim_order[i];
            const auto dimension_size{out.get_shape()[dimension_index]};
            selected_axes[i] = dimension_size;
        }
        // rest of dimension up to 4, remain with default value (0 or 1);

        const auto& sel_model{models[selected_dim_number - 1]};

        const auto cycles = (nominal_freqs ? sel_model.getDPUCycles(output_samples, selected_axes)  // nominal freqs
                                           : sel_model.getDPUCyclesAnotherFreqDPU_SHV(present_dpu_frq, present_shv_frq,
                                                                                      output_samples, selected_axes));
        return cycles;
    };

    MVN6GenericActivationExec(const std::string& name,  //
                              const MVN6Parameters p1,  //
                              const MVN6Parameters p2,  //
                              const MVN6Parameters p3,  //
                              const MVN6Parameters p4   //
                              )
            : ShaveOpExecutor(name, 1),
              models{{
                      {dtype, p1.slope, p1.intercept, p1.alpha, p1.worst_case_slope, p1.slope_delta_diff, DpuFreq,
                       ShvFreq},
                      {dtype, p2.slope, p2.intercept, p2.alpha, p2.worst_case_slope, p2.slope_delta_diff, DpuFreq,
                       ShvFreq},
                      {dtype, p3.slope, p3.intercept, p3.alpha, p3.worst_case_slope, p3.slope_delta_diff, DpuFreq,
                       ShvFreq},
                      {dtype, p4.slope, p4.intercept, p4.alpha, p4.worst_case_slope, p4.slope_delta_diff, DpuFreq,
                       ShvFreq},
              }} {
    }

    std::string toString() const override {
        std::stringstream stream;
        stream << "MVN6GenericActivationExec: \n"          //
               << " Operation: \t" << getName() << " ;\n"  //
               << " Shave params: 1 int, represents the number of axes selected (from innermost to outermost) \n"
               << " Model  1 : \t" << models[0] << " ;\n"   //
               << " Model  2 : \t" << models[1] << " ;\n"   //
               << " Model  3 : \t" << models[2] << " ;\n"   //
               << " Model  4 : \t" << models[3] << " ;\n";  //

        return stream.str();
    }
};
//////////////////////////////////////////////////////////////

/// MVN simple, specific mode for 2 or 3 axes HWC
template <DataType dtype, unsigned int VectorSize, unsigned int UnrollSize, unsigned int DpuFreq, unsigned int ShvFreq,
          int DimSelected>
class MVNSimpleNAxisActivationExec : public ShaveOpExecutor {
    static_assert(DimSelected >= 2, "Dimension must be in range ,2,3, now is too small");
    static_assert(DimSelected <= 3, "Dimension must be in range ,2,3, now is too large");

private:
    MVNSimple2and3AxisModel model;  ///< 2 or 3 axes compatible  (depends on selected dims)
    // TODO: question: should we pass down the 2 constants 64 (unroll size maybe) and 8 (vector size): NO seems to be
    // device independent

public:
    struct MVNSimpleParameters {
        float baseSlope;
        float baseIntercept;
        float baseSupportSlope;
        float mod8SupportSlope;
        float vectorSlope;
    };

    CyclesInterfaceType dpuCycles(const SHAVEWorkload& w) const override {
        return dpuCycles(w, model.getNominalDPUFrq(), model.getNominalSHVFrq());
    };

    CyclesInterfaceType dpuCycles(const SHAVEWorkload& w, const int present_dpu_frq,
                                  const int present_shv_frq) const override {
        const auto& out = w.get_outputs()[0];  // the only output
        // what happens if datatype is not anymore as the model ??

        // const auto output_samples{out.volume()};

        const auto layout_dim_order{layout_to_order(out.get_layout())};  // dim from innermost to outermost
        assert(layout_dim_order.size() == 4);

        const auto& out_shape{out.get_shape()};

        auto selected_dimension_size{1};         // first dimensions
        for (int i = 0; i < DimSelected; ++i) {  // take  the first DimSelected ,they are the selected one
            const auto dimension_index = layout_dim_order[i];
            const auto dimension_size{out_shape[dimension_index]};
            selected_dimension_size *= dimension_size;
        }

        auto unselected_dimension_size{1};  // last dimensions
        const int maxDim{static_cast<int>(layout_dim_order.size())};
        for (int i = DimSelected; i < maxDim; ++i) {  // take  the first DimSelected ,they are the selected one
            const auto dimension_index = layout_dim_order[i];
            const auto dimension_size{out_shape[dimension_index]};
            unselected_dimension_size *= dimension_size;
        }

        const int outermost_dim_size{
                (int)out_shape[3]};  ///< outermost, 4th,  is equal to unselected volume in case of S3

        // thirdmost can be forced to 1 in case of S3dim, there is no thirdmost unselected, but for generalization the
        // setup has to control this via multiplier constants/parameters (the slope)
        const int thirdmost_dim_size{(int)out_shape[2]};  ///< thirdmost, index 2. May be selected or not!!

        const auto cycles =
                model.getDPUCyclesAnotherFreqDPU_SHV(present_dpu_frq, present_shv_frq, selected_dimension_size,
                                                     unselected_dimension_size, outermost_dim_size, thirdmost_dim_size);
        return cycles;
    };

    MVNSimpleNAxisActivationExec(const std::string& name, float baseSlope, float baseIntercept,
                                 float thirdMostSupportSlope, float baseSupportSlope, float mod8SupportSlope,
                                 float vectorSlope)
            : ShaveOpExecutor(name),
              model(dtype, baseSlope, baseIntercept, thirdMostSupportSlope, baseSupportSlope, mod8SupportSlope,
                    vectorSlope, VectorSize, UnrollSize, DpuFreq, ShvFreq) {
    }

    std::string toString() const override {
        std::stringstream stream;
        stream << "MVNSimpleNAxisActivationExec: \n"         //
               << " N        : \t" << DimSelected << " ;\n"  //
               << " Operation: \t" << getName() << " ;\n"    //
               << " Model    : \t" << model << " ;\n";       //

        return stream.str();
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/// COMPOSITE MVN
// template <DataType dtype, unsigned int VectorSize, unsigned int UnrollSize, unsigned int DpuFreq, unsigned int
// ShvFreq>
class MVN_GenericActivationExec : public ShaveOpExecutor {
private:
    const ShaveOpExecutor& mvn6;
    const ShaveOpExecutor& mvn_s2;
    const ShaveOpExecutor& mvn_s3;

public:
    CyclesInterfaceType dpuCycles(const SHAVEWorkload& w) const override {
        return dpuCycles(w, 0, 0, true);  // nominal, ignore the zeros
    };
    CyclesInterfaceType dpuCycles(const SHAVEWorkload& w, const int present_dpu_frq,
                                  const int present_shv_frq) const override {
        return dpuCycles(w, present_dpu_frq, present_shv_frq, false);  // use dpu & shv freq
    };

    CyclesInterfaceType dpuCycles(const SHAVEWorkload& w, const int present_dpu_frq, const int present_shv_frq,
                                  bool nominal_freqs) const {
        // for how many axes? Nr of axes is param 1, as int
        if (w.get_params().size() < getNumExpectedParams()) {
            return Cycles::ERROR_SHAVE_PARAMS;
        }

        const int selected_dim_number{std::get<int>(w.get_params()[0])};


        if ((selected_dim_number <= 0) || (selected_dim_number > 4)) {
            return Cycles::ERROR_SHAVE_PARAMS;
        }

        const auto& out = w.get_outputs()[0];  // the only output
        // what happens if datatype is not anymore as the model ??

        const auto layout{out.get_layout()};                   // enum
        const auto layout_dim_order{layout_to_order(layout)};  // dim from innermost to outermost
        assert(layout_dim_order.size() == 4);

        // select what to call

        // if 2 axes and first 2 are WH-> call special 2 axis
        if ((2 == selected_dim_number) && (Dim::Act::X == layout_dim_order[0]) &&
            (Dim::Act::Y == layout_dim_order[1])) {
            return (nominal_freqs ? mvn_s2.dpuCycles(w) : mvn_s2.dpuCycles(w, present_dpu_frq, present_shv_frq));
        }
        // if 3 axes and first 3 are WHC call special 3 axis
        if ((3 == selected_dim_number) && (Dim::Act::X == layout_dim_order[0]) &&
            (Dim::Act::Y == layout_dim_order[1]) && (Dim::Act::Z == layout_dim_order[2])) {
            return (nominal_freqs ? mvn_s3.dpuCycles(w) : mvn_s3.dpuCycles(w, present_dpu_frq, present_shv_frq));
        }

        // otherwise call MVN6
        return (nominal_freqs ? mvn6.dpuCycles(w) : mvn6.dpuCycles(w, present_dpu_frq, present_shv_frq));
    };

    MVN_GenericActivationExec(const std::string& name,  //
                              const ShaveOpExecutor& r_mvn_s2, const ShaveOpExecutor& r_mvn_s3,
                              const ShaveOpExecutor& r_mvn6)
            : ShaveOpExecutor(name, 1), mvn6{r_mvn6}, mvn_s2{r_mvn_s2}, mvn_s3{r_mvn_s3} {
    }

    std::string toString() const override {
        std::stringstream stream;
        stream << "MVN_GenericActivationExec: \n"          //
               << " Operation: \t" << getName() << " ;\n"  //
               << " Shave params: 1 int, represents the number of axes selected (from innermost to outermost) \n"
               << " Model6   : \t" << mvn6.toString() << " ;\n"     //
               << " Model S2 : \t" << mvn_s2.toString() << " ;\n"   //
               << " Model S3 : \t" << mvn_s3.toString() << " ;\n";  //

        return stream.str();
    }
};

////////////////////////////////////////////////

/// wrapper NPU Mock executor
/// receives ref to base, new DPU freq , new VPU freq, speed up factor (to divide the result )
template <unsigned int DpuFreq, unsigned int ShvFreq>
class NPUMockExecutor : public ShaveOpExecutor {
private:
    const ShaveOpExecutor& mock_source;  ///< model from another NPU that is MOcked
    const float speed_factor;

public:
    CyclesInterfaceType dpuCycles(const SHAVEWorkload& w) const override {
        const auto cycles = mock_source.dpuCycles(w, DpuFreq, ShvFreq);  // at new  DPU , VPU freq point

        if (!Cycles::isErrorCode(cycles)) {
            return Cycles::toCycleInterfaceType((float)cycles / speed_factor);
        }
        return cycles;
    };

    CyclesInterfaceType dpuCycles(const SHAVEWorkload& w, const int present_dpu_frq,
                                  const int present_shv_frq) const override {
        const auto cycles = mock_source.dpuCycles(w, present_dpu_frq, present_shv_frq);  // at new  DPU , VPU freq point

        if (!Cycles::isErrorCode(cycles)) {
            return Cycles::toCycleInterfaceType((float)cycles / speed_factor);
        }
        return cycles;
    };

    NPUMockExecutor(const std::string& name, const ShaveOpExecutor& npu_original, float speed_up)
            : ShaveOpExecutor(name), mock_source(npu_original), speed_factor{speed_up} {
    }
    std::string toString() const override {
        std::stringstream stream;
        stream << "NPUMockExecutor: \n"                                     //
               << " Operation: \t" << getName() << " ;\n"                   //
               << " New DPU freq: \t" << DpuFreq << " ;\n"                  //
               << " New SHV freq: \t" << ShvFreq << " ;\n"                  //
               << " Speed up    : \t" << speed_factor << " ;\n"             //
               << " Mock on top : \t" << mock_source.toString() << " ;\n";  //

        return stream.str();
    }
};

//////////////////////////////////////////////////////////////////////

/// @brief Executor around the simple linear  model where the input  variable is the size of output (tehLegacy/initial
/// model)
/// @tparam KERNEL_NAME is the class name of the legacy model
template <typename KERNEL_NAME, unsigned int efficiencyX1000, unsigned int latency>
class ShaveClassicLinear : public ShaveOpExecutor {
private:
    // SFINAE for filteredInputs,   enablement is done via return value
    template <typename KERNEL_NAME_LOCAL>
    typename std::enable_if<!std::is_base_of<SHVElementwise<efficiencyX1000, latency>, KERNEL_NAME_LOCAL>::value,
                            const VPUTensor&>::type  // one tensor for unary ops
    filteredInputs(const std::vector<VPUTensor>& in) const {
        return in[0];
    }

    template <typename KERNEL_NAME_LOCAL>
    typename std::enable_if<std::is_base_of<SHVElementwise<efficiencyX1000, latency>, KERNEL_NAME_LOCAL>::value,
                            const std::vector<VPUTensor>&>::type  // multiple inputs for element wise
    filteredInputs(const std::vector<VPUTensor>& in) const {
        return in;
    }

public:
    CyclesInterfaceType dpuCycles(const SHAVEWorkload& w) const override {
        const auto& in = filteredInputs<KERNEL_NAME>(w.get_inputs());
        const auto& out = w.get_outputs()[0];

        KERNEL_NAME theInstance(w.get_device(), in, out);  // SHVHardSigmoid for example

        SWOperation& i = theInstance;
        const auto cycles = i.cycles();
        return cycles;
    };

    CyclesInterfaceType dpuCycles(const SHAVEWorkload&, const int, const int) const override {
        return Cycles::ERROR_SHAVE_PARAMS;  // not supported
    };
    ShaveClassicLinear(const std::string& name): ShaveOpExecutor(name){};
    std::string toString() const override {
        std::stringstream stream;
        stream << "ShaveClassicLinear: \n"                 //
               << " Operation: \t" << getName() << " ;\n"  //
               << " efficiencyX1000: \t" << efficiencyX1000 << " ;\n"
               << " latency: \t" << latency << " ;\n";  //

        return stream.str();
    }
};

///// just a POC executor
// class ShaveOPMOckTest : public ShaveOpExecutor {
// public:
//    // ShaveModel1to1 model;
//    CyclesInterfaceType dpuCycles(const SHAVEWorkload& w) const override {
//        const auto& out = w.get_outputs()[0];
//        // what happens if datatype is not anymore as the model ??
//        const auto cycles = out.size();
//        return cycles;
//        // return 1;
//    };
//    ShaveOPMOckTest(const std::string& name): ShaveOpExecutor(name) {
//    }
//};

//////////////////////////////////////////////////////////////////////////////////////

/// Interpolate with polynomial regression
/// for interpolate wher WH is changed (only), and layout is WHCB (strict)
/// INput to model is input W and H plus output volume
/// @tparam MOdel a model that receives 3 inputs (input w and H) and Output number of elements
template <DataType dtype, unsigned int DpuFreq, unsigned int ShvFreq, typename Model>
class InterpolateWH_IWHO_ActExec : public ShaveOpExecutor {
private:
    Model model;

public:
    CyclesInterfaceType dpuCycles(const SHAVEWorkload& w) const override {
        return dpuCycles(w, model.getNominalDPUFrq(), model.getNominalSHVFrq());
    };

    CyclesInterfaceType dpuCycles(const SHAVEWorkload& w, const int present_dpu_frq,
                                  const int present_shv_frq) const override {
        const auto& out = w.get_outputs()[0];  // the only output
        const auto& in = w.get_inputs()[0];    // the activation
        // what happens if datatype is not anymore as the model ??

        {  // this works only if layout is WH, and the interpolate is spatial
            const auto outLayout{out.get_layout()};
            const auto inLayout{in.get_layout()};
            if ((inLayout != outLayout) ||  // keep layout
                (inLayout != Layout::XYZ)   // WH only accepted
                // maybe check Z&B do not change??
            ) {
                return Cycles::ERROR_SHAVE_PARAMS;
            }
        }

        const int output_samples{(int)out.volume()};
        const int w_input{(int)in.width()};
        const int h_input{(int)in.width()};

        const auto cycles = model.getDPUCyclesAnotherFreqDPU_SHV(present_dpu_frq, present_shv_frq, w_input, h_input,
                                                                 output_samples);
        return cycles;
    };

    InterpolateWH_IWHO_ActExec(const std::string& name): ShaveOpExecutor(name), model(dtype, DpuFreq, ShvFreq) {
    }

    std::string toString() const override {
        std::stringstream stream;
        stream << "InterpolateWH_IWHO_ActExec: \n"         //
               << " Operation: \t" << getName() << " ;\n"  //
               << " Model : \t" << model << " ;\n";        //

        return stream.str();
    }
};

}  // namespace VPUNN
#endif