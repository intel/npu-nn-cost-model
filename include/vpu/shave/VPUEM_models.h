#ifndef VPUEM_MODELS_H
#define VPUEM_MODELS_H

#include <iostream>
#include <list>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "VPUEM_cost_function.h"
#include "VPUEM_op_base_dsp.h"
#include "VPUEM_piecewise_calc_subblk_size.h"
#include "shave_equations.h"
#include "shave_model_basics.h"
#include "vpu/cycles_interface_types.h"
#include "vpu/vpuem_models_struct.h"

namespace VPUNN {
class PiecewiseModel : public VPUEMShaveCyclesProvider<PiecewiseModel> {
private:
    const DataType dtype_;
    const std::vector<VPUEM_CostFunction> costFunction_;
    const VPUEMCalcSubblk vpuemCalculator_;
    const float cost_curve_ratio_;
    const int unroll_mode_;

    friend std::ostream& operator<<(std::ostream& stream, const VPUNN::PiecewiseModel& d);

    static std::vector<VPUEM_CostFunction> setCostFunction(
            const std::vector<CostFunction3SlopesDescriptor>& costFunction3SlopesData) {
        std::vector<VPUEM_CostFunction> costFunctionAux;
        costFunctionAux.reserve(costFunction3SlopesData.size());
        for (const auto& slopesData : costFunction3SlopesData) {
            costFunctionAux.push_back(slopesData);
        }
        return costFunctionAux;
    }

public:
    PiecewiseModel(DataType dtype, unsigned int DpuFreq, unsigned int ShvFreq,
                   const std::vector<CostFunction3SlopesDescriptor>& costFunction3SlopesData, bool adaptive_blk_num_en,
                   int max_blk_num, int dspArch, float cost_curve_ratio, int unroll_mode = 0)
            : VPUEMShaveCyclesProvider<PiecewiseModel>{DpuFreq, ShvFreq},
              dtype_(dtype),
              costFunction_(setCostFunction(costFunction3SlopesData)),
              vpuemCalculator_(adaptive_blk_num_en, max_blk_num, dspArch),
              cost_curve_ratio_(cost_curve_ratio),
              unroll_mode_(unroll_mode) {
    }

    std::string getCostFunction() const {
        std::stringstream stream;
        for (const auto& slopesData : costFunction_) {
            stream << " Unroll: \t" << slopesData.costFunction3SlopesData_.unroll_ << " ;\n"
                   << " Offset: \t" << slopesData.costFunction3SlopesData_.offset_ << " ;\n"
                   << " Slopes: { \t";

            for (const auto& slopes : slopesData.costFunction3SlopesData_.slope_) {
                stream << slopes << ", ";
            }
            stream << "};\n";
        }
        return stream.str();
    }

    const VPUEM_CostFunction& find_VPUEM_CostFunctionObject(const int output_size_bytes) const {
        // commuting from bytes to operations
        // if i have a number of bytes and we know the data type then we know how many ops we made
        int output_size_operations = compute_elements_count_from_bytes(output_size_bytes, dtype_);

        std::vector<int> unrollPreferences;
        if (unroll_mode_ != 0) {
            // filter based on the specified unroll_mode
            unrollPreferences.push_back(unroll_mode_);
        } else {
            // check the tensor size
            if (output_size_operations > 256) {
                unrollPreferences = {32, 16, 8, 64};
            } else {
                unrollPreferences = {8, 16, 32, 64};
            }
        }

        for (int preference : unrollPreferences) {
            auto it = std::find_if(costFunction_.begin(), costFunction_.end(),
                                   [preference](const VPUEM_CostFunction& obj) {
                                       return obj.costFunction3SlopesData_.unroll_ == preference;
                                   });

            if (it != costFunction_.end()) {
                // return reference of the found object
                return *it;
            }
        }

        // if unroll_mode is not specified and no object is found, throw exception
        throw std::runtime_error("No VPUEM_CostFunction object found!");
    }

    int getShaveCycles(const SHAVEWorkload& w) const {
        return getComputedCycles(w.get_inputs(), w.get_outputs());
    }

    int getIdealCycles(const int output_size_bytes) const {
        try {
            // find the CostFunction object based on the output size
            const VPUEM_CostFunction& vpuemObject = find_VPUEM_CostFunctionObject(output_size_bytes);
            return vpuemObject.getCycles(dtype_, output_size_bytes, cost_curve_ratio_);
        } catch (const std::runtime_error& e) {
            std::cerr << " Error: " << e.what() << std::endl;
            return 0;
        }
    }

    int calc_blk_cycles(const VPUEM_Subblk_Tensor& odim, const VPUEM_CostFunction& vpuemObject) const {
        return 1 + vpuemObject.getCycles(dtype_, odim.get_output_size(), cost_curve_ratio_);
    }

    int getComputedCycles(const std::vector<VPUTensor>& inputs, const std::vector<VPUTensor>& outputs) const {
        // divide the tensors into subblocks based on the device parameters
        int blk_cycles = 0;
        try {
            const VPUEM_CostFunction& vpuemObject = find_VPUEM_CostFunctionObject(outputs[0].size());
            auto vpuemTuple = vpuemCalculator_.calc_dsp_block_unit(inputs, outputs);
            auto numBlocks = std::get<0>(vpuemTuple);
            auto odim = std::get<3>(vpuemTuple).front();

            for (int idx = 0; idx < numBlocks.back(); idx++) {
                if (idx == (numBlocks.back() - 1)) {
                    auto odim_last = std::get<4>(vpuemTuple).front();
                    blk_cycles += calc_blk_cycles(odim_last, vpuemObject);
                } else if (idx == 0) {
                    blk_cycles +=
                            calc_blk_cycles(odim, vpuemObject) + int(vpuemObject.costFunction3SlopesData_.offset_) - 1;
                } else {
                    blk_cycles += calc_blk_cycles(odim, vpuemObject);
                }
            }

            // std::cout << "block cycles: " << blk_cycles << std::endl;

            return blk_cycles;
        } catch (const std::runtime_error& e) {
            std::cerr << " Error: " << e.what() << std::endl;
            return 0;
        }
    }
};

inline std::ostream& operator<<(std::ostream& stream, const VPUNN::PiecewiseModel& d) {
    stream << "PiecewiseModel: \n"
           << " dtype: \t" << (int)d.dtype_ << " : " << DataType_ToText.at(static_cast<int>(d.dtype_)) << " ;\n"  //
           << d.converter                                                                                         //
           << out_terminator() << "PiecewiseModel"  // terminator
            ;
    return stream;
}

class VPUEMSoftmaxModel : public VPUEMShaveCyclesProvider<VPUEMSoftmaxModel> {
private:
    const DataType dtype_;
    std::vector<VPUEMSoftmax_CostFunction> costFunction_;

    friend std::ostream& operator<<(std::ostream& stream, const VPUNN::VPUEMSoftmaxModel& d);

public:
    VPUEMSoftmaxModel(DataType dtype, const std::vector<CostFunctionSoftmaxDescriptor> costFunctionSoftmaxData,
                      unsigned int DpuFreq, unsigned int ShvFreq)
            : VPUEMShaveCyclesProvider<VPUEMSoftmaxModel>{DpuFreq, ShvFreq}, dtype_(dtype) {
        for (const auto& softData : costFunctionSoftmaxData) {
            costFunction_.push_back(softData);
        }
    }

    int getShaveCycles(const int h_output_size_bytes, const int hw_output_size_bytes,
                       const int c_output_size_bytes) const {
        try {
            return costFunction_[0].getCycles(dtype_, h_output_size_bytes, hw_output_size_bytes, c_output_size_bytes);
        } catch (const std::runtime_error& e) {
            std::cerr << " Error: " << e.what() << std::endl;
            return 0;
        }
    }
};

inline std::ostream& operator<<(std::ostream& stream, const VPUNN::VPUEMSoftmaxModel& d) {
    stream << "VPUEMSOftmaxModel: \n"
           << " dtype: \t" << (int)d.dtype_ << " : " << DataType_ToText.at(static_cast<int>(d.dtype_)) << " ;\n"  //
           << d.converter                                                                                         //
           << out_terminator() << "VPUEMSoftmaxModel"  // terminator
            ;
    return stream;
}

class VPUEMSpatialModel : public VPUEMShaveCyclesProvider<VPUEMSpatialModel> {
private:
    const DataType dtype_;
    const VPUEMSpatial_CostFunction costFunction_;

    friend std::ostream& operator<<(std::ostream& stream, const VPUNN::VPUEMSpatialModel& d);

public:
    VPUEMSpatialModel(DataType dtype, const CostFunctionSpatialDescriptor& costFunctionSpatialData,
                      unsigned int DpuFreq, unsigned int ShvFreq)
            : VPUEMShaveCyclesProvider<VPUEMSpatialModel>{DpuFreq, ShvFreq},
              dtype_(dtype),
              costFunction_(costFunctionSpatialData) {
    }

    int getShaveCycles(int output_size_bytes) const {
        try {
            return costFunction_.getCycles(dtype_, output_size_bytes);
        } catch (const std::runtime_error& e) {
            std::cerr << " Error: " << e.what() << std::endl;
            return 0;
        }
    }
};
inline std::ostream& operator<<(std::ostream& stream, const VPUNN::VPUEMSpatialModel& d) {
    stream << "VPUEMSpatialModel: \n"
           << " dtype: \t" << (int)d.dtype_ << " : " << DataType_ToText.at(static_cast<int>(d.dtype_)) << " ;\n"  //
           << d.converter                                                                                         //
           << out_terminator() << "VPUEMSpatialModel"  // terminator
            ;
    return stream;
}
}  // namespace VPUNN
#endif
