#ifndef SHAVE_VPUEM_EXECUTORS_H
#define SHAVE_VPUEM_EXECUTORS_H

#include "interface_shave_op_executor.h"
#include "vpu/types.h"
#include "VPUEM_models.h"
#include "VPUEM_cost_function.h"

#include <sstream>

#include <map>
#include <string>
#include <vector>

namespace VPUNN {
template <DataType dtype, unsigned int DpuFreq, unsigned int ShvFreq>
class PiecewiseExec : public ShaveOpExecutor {
private:
    PiecewiseModel model;  ///< model instance

public:
    CyclesInterfaceType dpuCycles(const SHAVEWorkload& w) const override {
        return dpuCycles(w, model.getNominalDPUFrq(), model.getNominalSHVFrq());
    };

    CyclesInterfaceType dpuCycles(const SHAVEWorkload& w, const int present_dpu_frq,
                                  const int present_shv_frq) const override {
        
        const auto cycles = model.getDPUCyclesAnotherFreqDPU_SHV(present_dpu_frq, present_shv_frq,w);
        return cycles;
    };

    PiecewiseExec(const std::string& name, std::vector<CostFunction3SlopesDescriptor> costFunction3SlopesData,
                  bool adaptive_blk_num_en, int max_blk_num, int dspArch, float cost_curve_ratio,
                  int unroll_mode = 0)
            : ShaveOpExecutor(name),
              model(dtype, DpuFreq, ShvFreq, costFunction3SlopesData, adaptive_blk_num_en, max_blk_num, dspArch,
                    cost_curve_ratio, unroll_mode) {
    }

    std::string toString() const override {
        std::stringstream stream;
        stream << "PiecewiseExecutor: \n"                
               << " Operation: \t" << getName() << " ;\n"  
               << " Model    : \t" << model << " ;\n"
               << " Slopes Data: \t" << model.getCostFunction() << " ;\n";     
        return stream.str();
    }
};

template <DataType dtype, unsigned int DpuFreq, unsigned int ShvFreq>
class VPUEMSoftmaxExec : public ShaveOpExecutor {
private:
    VPUEMSoftmaxModel model;  ///< model instance

public:
    CyclesInterfaceType dpuCycles(const SHAVEWorkload& w) const override {
        return dpuCycles(w, model.getNominalDPUFrq(), model.getNominalSHVFrq());
    };

    CyclesInterfaceType dpuCycles(const SHAVEWorkload& w, const int present_dpu_frq,
                                  const int present_shv_frq) const override {
        const auto& h_out = w.get_outputs()[0].height();
        const auto& hw_out = w.get_outputs()[0].height() * w.get_outputs()[0].width();  
        const auto& c_out = w.get_outputs()[0].channels();
        const auto cycles = model.getDPUCyclesAnotherFreqDPU_SHV(present_dpu_frq, present_shv_frq, h_out, hw_out, c_out); 
        return cycles;
    };

    VPUEMSoftmaxExec(const std::string& name, std::vector<CostFunctionSoftmaxDescriptor> CostFunctionSoftmaxData)
            : ShaveOpExecutor(name), model(dtype, std::move(CostFunctionSoftmaxData), DpuFreq, ShvFreq) {}

    std::string toString() const override {
        std::stringstream stream;
        stream << "VPUEMSoftmaxExec: \n"
               << " Operation: \t" << getName() << " ;\n"
               << " Model    : \t" << model << " ;\n";

        return stream.str();
    }
};

template <DataType dtype, unsigned int DpuFreq, unsigned int ShvFreq>
class VPUEMSpatialExec : public ShaveOpExecutor {
private:
    VPUEMSpatialModel model;  ///< model instance

public:
    CyclesInterfaceType dpuCycles(const SHAVEWorkload& w) const override {
        return dpuCycles(w, model.getNominalDPUFrq(), model.getNominalSHVFrq());
    };

    CyclesInterfaceType dpuCycles(const SHAVEWorkload& w, const int present_dpu_frq,
                                  const int present_shv_frq) const override {
        const auto& out = w.get_outputs()[0];  // the only output
        const auto cycles = model.getDPUCyclesAnotherFreqDPU_SHV(present_dpu_frq, present_shv_frq, out.size());
        return cycles;
    };

    VPUEMSpatialExec(const std::string& name, CostFunctionSpatialDescriptor CostFunctionSpatialData)
            : ShaveOpExecutor(name), model(dtype, std::move(CostFunctionSpatialData), DpuFreq, ShvFreq) {
    }

    std::string toString() const override {
        std::stringstream stream;
        stream << "VPUEMSpatialExec: \n"
               << " Operation: \t" << getName() << " ;\n"
               << " Model    : \t" << model << " ;\n";

        return stream.str();
    }
};
}  // namespace VPUNN
#endif