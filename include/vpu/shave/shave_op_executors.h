// Copyright © 2023 Intel Corporation
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

#include "ShaveModel1to1.h"
#include "interface_shave_op_executor.h"
#include "vpu/cycles_interface_types.h"
#include "vpu/types.h"

namespace VPUNN {

/// @brief Executor around the linear with steps model where the input  variable is the size of output
template <DataType dtype, unsigned int VectorSize, unsigned int UnrollSize, unsigned int DpuFreq, unsigned int ShvFreq>
class ShaveActivation1on1 : public ShaveOpExecutor {
private:
    ShaveModel1to1 model;  ///< model instance

public:
    CyclesInterfaceType dpuCycles(const SHAVEWorkload& w) const override {
        const auto& out = w.get_outputs()[0];
        // what happens if datatype is not anymore as the model ??
        const auto cycles = model.getDPUCycles(out.size());
        return cycles;
    };

    ShaveActivation1on1(const std::string& name, float slope, float intercept, float offset_scalar, float offset_unroll)
            : ShaveOpExecutor(name),
              model(dtype, slope, intercept, offset_scalar, offset_unroll, VectorSize, UnrollSize, DpuFreq, ShvFreq) {
    }
};

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
    ShaveClassicLinear(const std::string& name): ShaveOpExecutor(name){};
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

}  // namespace VPUNN
#endif