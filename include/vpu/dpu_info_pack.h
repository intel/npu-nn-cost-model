// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_DPUINFOPACK_H
#define VPUNN_DPUINFOPACK_H

#include <sstream>  // for error formating
#include <string>

#include "vpu/cycles_interface_types.h"
#include "vpu/types.h"
#include "vpu/dpu_defaults.h"

namespace VPUNN {
/// @brief L1API info for a DPUWorkload.
/// intention is to obtain all info at once, in a more efficient way.
/// Zero values means either error or value could not be obtained
/// See the original interface method for each field to understand its meaning
struct DPUInfoPack {
    CyclesInterfaceType DPUCycles{0};  ///< DPU()
    std::string errInfo;               ///< error info when doing DPU()

    float energy{0};  ///< DPUEnergy(), uses power_* information

    // for power usage, considers HW optimization like sparsity
    float power_activity_factor{0};  ///< AF, operation adjusted, INT/FLOAT powerVirus as reference  is important
    float power_mac_utilization{0};  ///< hw_utilization, mac only based, Uses estimated dpu Cycles,
    unsigned long int power_ideal_cycles{0};     ///< pure mac, ops considers sparsity
    unsigned long int sparse_mac_operations{0};  ///< how many macs this operation will have on this hardware

    // efficiency part do not consider HW optimization like sparsity
    float efficiency_activity_factor{0};  ///<  operation adjusted, INT/FLOAT powerVirus as reference  is important
    float efficiency_mac_utilization{0};  ///< no op dependency, mac only based, Uses estimated dpu Cycles
    unsigned long int efficiency_ideal_cycles{0};  ///< pure MAC based
    unsigned long int dense_mac_operations{0};     ///< how many macs this operation will have, mathematical maximum

    unsigned long int hw_theoretical_cycles{0};  ///< DPUTheoreticalCycles
};

inline std::ostream& operator<<(std::ostream& stream, const VPUNN::DPUInfoPack& d) {
    stream << "DPUInfoPack: \n"                                                                       //
           << " DPUCycles: \t" << d.DPUCycles << " : " << Cycles::toErrorText(d.DPUCycles) << " ;\n"  //
           << " errInfo: \t" << d.errInfo << " ;\n"                                                   //
           << " energy: \t" << d.energy << " ;\n"                                                     //

           << " power_activity_factor: \t" << d.power_activity_factor << " ;\n"  //
           << " power_mac_utilization: \t" << d.power_mac_utilization << " ;\n"  //
           << " power_ideal_cycles: \t" << d.power_ideal_cycles << " ;\n"        //
           << " sparse_mac_operations: \t" << d.sparse_mac_operations << " ;\n"  //

           << " efficiency_activity_factor: \t" << d.efficiency_activity_factor << " ;\n"  //
           << " efficiency_mac_utilization: \t" << d.efficiency_mac_utilization << " ;\n"  //
           << " efficiency_ideal_cycles: \t" << d.efficiency_ideal_cycles << " ;\n"        //
           << " dense_mac_operations: \t" << d.dense_mac_operations << " ;\n"              //

           << " hw_theoretical_cycles: \t" << d.hw_theoretical_cycles << " ;\n"  //
           << out_terminator() << "DPUInfoPack "                                 // terminator
            ;
    return stream;
}

}  // namespace VPUNN

#endif  //
