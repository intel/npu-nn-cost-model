// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_UT_REGRESSION_H
#define VPUNN_UT_REGRESSION_H

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "common_helpers.h"

#include "vpu/cycles_interface_types.h"

#include "core/logger.h"
#include "vpu/dpu_types.h"
#include "vpu/dpu_workload.h"
#include "vpu/vpu_tensor.h"
#include "vpu_cost_model.h"

#include "cost_model_test.h"

/// @brief namespace for Unit tests of the C++ library
namespace VPUNN_unit_tests {
using namespace VPUNN;

class Regression_Tests : public TestCostModel {
protected:
    bool no_fail{true};// if set on false all tests  that use it (at range definition) will fail. Begavior is like for name:  not_force_failing_all_tests

    // this function takes a workload and returns a 3 elements vector with 3 identical workloads as the one received as
    // parameter, but with a different number of channels 64,32, 16 in this order
    std::array<DPUWorkload, 3> change_wl_channels_64_32_16(const DPUWorkload &wl) {
        unsigned int channels = 64;
        std::array<DPUWorkload, 3> workloads{wl, wl, wl};

        for (int i = 0; i < 3; i++) {
            // input tensor dimensions WHCB
            workloads[i].inputs[0].set_shape(
                    {wl.inputs[0].get_shape()[0], wl.inputs[0].get_shape()[1], channels, wl.inputs[0].get_shape()[3]});
            // output tensor dimensions WHCB
            workloads[i].outputs[0].set_shape({wl.outputs[0].get_shape()[0], wl.outputs[0].get_shape()[1], channels,
                                               wl.outputs[0].get_shape()[3]});

            channels = channels / 2;  // 64 then 32 then 16
        }
        return workloads;
    }

    virtual std::string test_message(const DPUWorkload &wl, std::string tensor_type) const = 0;

    static const int pc = 25;  // default percent (10 fro best), 25 used for release so we have less manual fix to pass
    const int leftSide = -1;
    const int rightSide = 1;
    const int no_gt = -1;  // for error cases when we don't have any gt

    /// @param: value is the value for witch we compute one of the interval's endpoints
    /// @param: p is the value by witch we compute the tolerance
    /// @param: boundarySide is the parameter by which we decide which endpoint of the interval to compute, it's value
    /// can be -1 for the left side, and +1 for the right side
    unsigned int computeIntervalEdge(const int value, const int p, const int boundarySide) {
        int tolerance = (value * p) / 100;
        int interval = value + tolerance * boundarySide;

        return static_cast<unsigned int>(interval);
    }

    /// this function call testcase() function using values for percent and exception_limit that are predefined
    /// if we want to have exception cases where delta belongs to a range we will set the value of the range through the
    /// parameter excep_limit or we will leave it at 0
    GTestCase tc(const DPUWorkload& wl, const int gt, std::string info, int percent = pc, int excep_percent = 0) {
        return testcase(wl, gt, percent, excep_percent, excep_percent, std::move(info));
    };

    GTestCase tc(const DPUWorkload& wl, const int gt, std::string info, int percent, int excep_percent_left, int excep_percent_right) {
        return testcase(wl, gt, percent, excep_percent_left, excep_percent_right, std::move(info));
    };


    /// this function return a test case with specific parameters given as arguments
    GTestCase testcase(const DPUWorkload& wl, const int gt, const int percent, const int excep_lim_l, const int excep_lim_r, std::string info) {
        // clang-format off
        if (gt == no_gt) {
            GTestCase t_err{{wl, gt}, {Cycles::ERROR_INPUT_TOO_BIG, true, computeIntervalEdge(gt, percent, leftSide), computeIntervalEdge(gt, percent, rightSide) * no_fail}, test_message(wl, std::move(info))};
            return t_err;
        }

      unsigned int e_cyc_max=((0 == excep_lim_r) ? 0 : computeIntervalEdge(gt, excep_lim_r, rightSide));
      unsigned int e_cyc_min=((0 == excep_lim_l) ? 0 : computeIntervalEdge(gt, excep_lim_l, leftSide));

             GTestCase t{{wl, gt}, {Cycles::NO_ERROR, true, computeIntervalEdge(gt, percent, leftSide), computeIntervalEdge(gt, percent, rightSide) * no_fail, e_cyc_min, e_cyc_max}, test_message(wl, std::move(info))};

        // clang-format on
        return t;
    }

    void SetUp() override {
        no_fail = true;
    }

    void TearDown() override {
        no_fail = true;
    }

    Regression_Tests()
    {
        //if we want to accept exceptions we should put this variable on true (Put it at its origin!)
        //is_excep_allowed = false;
    }
};


}  // namespace VPUNN_unit_tests
#endif
   // VPUNN_UT_REGRESSIONL_H