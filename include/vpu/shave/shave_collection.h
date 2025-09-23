// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef SHAVE_COLLECTION_H
#define SHAVE_COLLECTION_H

#include <map>
#include <memory>
#include <type_traits>
#include <string>
#include <vector>

#include "elementwise.h"
#include "interface_shave_op_executor.h"
#include "shave_op_executors.h"
#include "shave_vpuem_executors.h"
#include "vpu/cycles_interface_types.h"
#include "vpu/types.h"
#include "vpu/shave/VPUEM_cost_function.h"

namespace VPUNN {

/// @brief list of shaves attached to a device
/// must have access to the destructor of the ShaveOpExecutor.
///     owns the executor concrete instances (creation and destruction is in its responsibility)
/* coverity[rule_of_three_violation:FALSE] */
class DeviceShaveContainer {
private:
    ///@brief in here we can delete a p because we are friends!
    static void deleter(ShaveOpExecutor* p) {
        delete p;
    }

    VPUDevice device;  ///< device intended for this container

    /// the map payload is a unique pointer with delete helper
    /// should always be moved otherwise the pointer will be deleted.
    using map_content_t = std::unique_ptr<ShaveOpExecutor, decltype(&DeviceShaveContainer::deleter)>;

    /// maps names of functions to their model instances (executor). Owns this instances, responsible with destruction
    std::map<std::string, map_content_t> map_shaves;

    /// @brief inserts an executor to the map, full transfer of ownership
    void addOp(map_content_t&& up) {
        if (up != nullptr) {
            // map[up->getName()] = std::move(up);  // overrides previous if existed
            // auto up{map_content_t(p, &deleter)};
            map_shaves.insert({up->getName(), std::move(up)});  // for results checking
        }
    }

public:
    DeviceShaveContainer(VPUDevice device): device(device) {
    }

    DeviceShaveContainer(const DeviceShaveContainer&) = delete;
    DeviceShaveContainer(DeviceShaveContainer&&) = default;
    virtual ~DeviceShaveContainer() = default;

public:
    VPUDevice getDevice() const {
        return device;
    }
    bool existsShave(const std::string sw) const {
        const auto it = map_shaves.find(sw);
        return (it != map_shaves.end());
    }
    std::vector<std::string> getShaveList() const {
        std::vector<std::string> list;
        list.reserve(map_shaves.size());
        for (const auto& i : map_shaves) {
            list.push_back(i.first);
        }

        return list;
    }

    const ShaveOpExecutor& getShaveExecutor(const std::string& sw) const {
        if (existsShave(sw)) {
            return *(map_shaves.at(sw));
        }
        std::stringstream buffer;
        buffer << "[ERROR]:could not find the Shave function with name: " << sw
               << " in the shave list for Device: " << VPUDevice_ToText.at(static_cast<int>(device))
               << " File: " << __FILE__ << " Line: " << __LINE__;
        std::string details = buffer.str();
        throw std::out_of_range(details);
    }

protected:
    // next are helper functions to add a particular instance of concrete  executors, will be used by derived types.

    template <DataType dtype, unsigned int VectorSize, unsigned int UnrollSize, unsigned int DpuFreq,
              unsigned int ShvFreq>
    void Add(const std::string& name, float slope, float intercept, float offset_scalar, float offset_unroll) {
        auto p{new ShaveActivation1on1<dtype, VectorSize, UnrollSize, DpuFreq, ShvFreq>(name, slope, intercept,
                                                                                        offset_scalar, offset_unroll)};
        addOp(map_content_t(p, &DeviceShaveContainer::deleter));
    }

    template <DataType dtype, unsigned int VectorSize, unsigned int UnrollSize, unsigned int DpuFreq,
              unsigned int ShvFreq>
    void Add(const std::string& name, float slope, float intercept, float offset_unroll, float intra_block_offset,
             float vector_offset, unsigned int displacement_size) {
        auto p{new ShaveActivation1on1NPU40<dtype, VectorSize, UnrollSize, DpuFreq, ShvFreq>(
                name, slope, intercept, offset_unroll, intra_block_offset, vector_offset, displacement_size)};
        addOp(map_content_t(p, &DeviceShaveContainer::deleter));
    }

    //// just a mock
    // void Add(const std::string& name) {
    //     addOp(map_content_t((new ShaveOPMOckTest(name)), &DeviceShaveContainer::deleter));
    // }

    // Legacy modes
    template <typename KERNEL_NAME, unsigned int efficiencyX1000, unsigned int latency>
    void Add(const std::string& name) {
        auto p{new ShaveClassicLinear<KERNEL_NAME, efficiencyX1000, latency>(name)};
        addOp(map_content_t(p, &DeviceShaveContainer::deleter));
    }

    template <DataType dtype, unsigned int VectorSize, unsigned int UnrollSize, unsigned int DpuFreq,
              unsigned int ShvFreq>
    void AddMVN6OneAx(const std::string& name, float slope, float intercept, float alpha, float maxmium_diff_slope) {
        auto p{new MVN6OneAxisActivationExec<dtype, VectorSize, UnrollSize, DpuFreq, ShvFreq>(
                name, slope, intercept, alpha, maxmium_diff_slope)};
        addOp(map_content_t(p, &DeviceShaveContainer::deleter));
    }

    template <DataType dtype, unsigned int VectorSize, unsigned int UnrollSize, unsigned int DpuFreq,
              unsigned int ShvFreq, int DimSelected>
    void AddMVN6MultiAx(const std::string& name, const MVN6Parameters& oneParam) {
        auto p{new MVN6MultiAxisActivationExec<dtype, VectorSize, UnrollSize, DpuFreq, ShvFreq, DimSelected>(name,
                                                                                                             oneParam)};
        addOp(map_content_t(p, &DeviceShaveContainer::deleter));
    }
    template <DataType dtype, unsigned int VectorSize, unsigned int UnrollSize, unsigned int DpuFreq,
              unsigned int ShvFreq, int DimSelected>
    void AddMVN6MultiAx(const std::string& name, float slope, float intercept, float alpha, float worst_case_slope,
                        float slope_delta_diff) {
        auto p{new MVN6MultiAxisActivationExec<dtype, VectorSize, UnrollSize, DpuFreq, ShvFreq, DimSelected>(
                name, slope, intercept, alpha, worst_case_slope, slope_delta_diff)};
        addOp(map_content_t(p, &DeviceShaveContainer::deleter));
    }

    // AddMVN_S3Axes
    template <DataType dtype, unsigned int VectorSize, unsigned int UnrollSize, unsigned int DpuFreq,
              unsigned int ShvFreq,
              int DimSelected>
    void AddMVN_SimpleNAx(const std::string& name,  //
                          float baseSlope, float baseIntercept, float thirdMostSupportSlope, float baseSupportSlope,
                          float mod8SupportSlope, float vectorSlope) {
        auto p{new MVNSimpleNAxisActivationExec<dtype, VectorSize, UnrollSize, DpuFreq, ShvFreq, DimSelected>(
                name,  //
                baseSlope, baseIntercept, thirdMostSupportSlope, baseSupportSlope, mod8SupportSlope, vectorSlope)};
        addOp(map_content_t(p, &DeviceShaveContainer::deleter));
    }

    // MULTIPLE PARAMETERS
    template <DataType dtype, unsigned int VectorSize, unsigned int UnrollSize, unsigned int DpuFreq,
              unsigned int ShvFreq>
    void AddMVN6Generic(const std::string& name,  //
                        const MVN6Parameters p1, const MVN6Parameters p2, const MVN6Parameters p3,
                        const MVN6Parameters p4) {
        auto p{new MVN6GenericActivationExec<dtype, VectorSize, UnrollSize, DpuFreq, ShvFreq>(name, p1, p2, p3, p4)};
        addOp(map_content_t(p, &DeviceShaveContainer::deleter));
    }

    // COMPOSITE ELEMENT
    // template <DataType dtype, unsigned int VectorSize, unsigned int UnrollSize, unsigned int DpuFreq,
    //           unsigned int ShvFreq>
    void AddMVN_CompositeGeneric(const std::string& name,  //
                                 const std::string& s2ax, const std::string& s3ax, const std::string& gen6) {
        // let's collect all params
        if (!existsShave(s2ax)) {
            throw std::runtime_error("MVN for simple 2 axes not configured: " + s2ax);
        }
        if (!existsShave(s3ax)) {
            throw std::runtime_error("MVN for simple 3 axes not configured: " + s3ax);
        }
        if (!existsShave(gen6)) {
            throw std::runtime_error("MVN for MVN6 generic not configured: " + gen6);
        }

        auto p{new MVN_GenericActivationExec(name,  //
                                             getShaveExecutor(s2ax), getShaveExecutor(s3ax), getShaveExecutor(gen6))};
        addOp(map_content_t(p, &DeviceShaveContainer::deleter));
    }

    // mock a NPU  with awareness of its present day frequencies
    template <unsigned int DpuFreq, unsigned int ShvFreq>
    void AddMock(const std::string& name, const ShaveOpExecutor& npu_original_mocked, float speed_up) {
        auto p{new NPUMockExecutor<DpuFreq, ShvFreq>(name, npu_original_mocked, speed_up)};
        addOp(map_content_t(p, &DeviceShaveContainer::deleter));
    }

    // WH +outsize input and Layout WHCB
    template <DataType dtype, unsigned int DpuFreq, unsigned int ShvFreq, typename Model>
    void AddPoly_WHO(const std::string& name) {
        auto p{new InterpolateWH_IWHO_ActExec<dtype, DpuFreq, ShvFreq, Model>(name)};
        addOp(map_content_t(p, &DeviceShaveContainer::deleter));
    }

    // Softmax
    template <DataType dtype, unsigned int VectorSize, unsigned int UnrollSize, unsigned int DpuFreq,
              unsigned int ShvFreq>
    void AddSoftmax(const std::string& name, float baseSlope, float baseIntercept, SoftmaxEquationParams e1,
                    SoftmaxEquationParams e2, SoftmaxEquationParams e4, SoftmaxEquationParams e8,
                    SoftmaxEquationParams e16, SoftmaxEquationParams e32) {
        auto p{new SoftmaxActivationExec<dtype, VectorSize, UnrollSize, DpuFreq, ShvFreq>(
                name, baseSlope, baseIntercept, e1, e2, e4, e8, e16, e32)};
        addOp(map_content_t(p, &DeviceShaveContainer::deleter));
    }

    // NormalizeL2OnlyC
    template <DataType dtype, unsigned int VectorSize, unsigned int UnrollSize, unsigned int DpuFreq,
              unsigned int ShvFreq>
    void AddNormalizeL2OnlyC(const std::string& name, float baseTimeSlope, float baseTimeIntercept,
                             float baseVectorOffset, float baseTimeSlopeW, float baseTimeInterceptW, float slopeW1,
                             float slopeW8, float slopeW9, float baseVectorOffsetW) {
        auto p{new NormalizeL2OnlyCActivationExec<dtype, VectorSize, UnrollSize, DpuFreq, ShvFreq>(
                name, baseTimeSlope, baseTimeIntercept, baseVectorOffset, baseTimeSlopeW, baseTimeInterceptW, slopeW1,
                slopeW8, slopeW9, baseVectorOffsetW)};
        addOp(map_content_t(p, &DeviceShaveContainer::deleter));
    }

    // Gather Model
        template <DataType dtype, unsigned int VectorSize, unsigned int DpuFreq, unsigned int ShvFreq>
        void AddGather(const std::string& name, float base_slope, float base_intercept, float inter_slope, 
                       float worst_slope, float vector_offset){
        auto p{new GatherActivationExec<dtype, VectorSize, DpuFreq, ShvFreq>(
                name, base_slope, base_intercept, inter_slope, worst_slope, vector_offset)};
        addOp(map_content_t(p, &DeviceShaveContainer::deleter));
        }

    // VPUEM PieceWise Model
    template <DataType dtype, unsigned int DpuFreq,unsigned int ShvFreq>
    void AddVPUEM_Piecewise(const std::string& name,
                            const std::vector<CostFunction3SlopesDescriptor>& costFunction3SlopesData,
                            bool adaptive_blk_num_en, 
                            int max_blk_num, 
                            int dspArch,
                            float cost_curve_ratio,
                            int unroll_mode = 0) { 
        auto p {new PiecewiseExec<dtype, DpuFreq, ShvFreq>(name, costFunction3SlopesData, adaptive_blk_num_en, max_blk_num, dspArch, cost_curve_ratio, unroll_mode)
        };
        addOp(map_content_t(p, &DeviceShaveContainer::deleter));
    }

     // VPUEM Softmax Model
    template <DataType dtype, unsigned int DpuFreq, unsigned int ShvFreq>
    void AddVPUEM_Softmax(const std::string& name,
                          const std::vector<CostFunctionSoftmaxDescriptor>& CostFunctionSoftmaxData) {
        auto p{new VPUEMSoftmaxExec<dtype, DpuFreq, ShvFreq>(name, CostFunctionSoftmaxData)};
        addOp(map_content_t(p, &DeviceShaveContainer::deleter));
    }

     // VPUEM Spatial Model
    template <DataType dtype, unsigned int DpuFreq, unsigned int ShvFreq>
    void AddVPUEM_Spatial(const std::string& name, const CostFunctionSpatialDescriptor& CostFunctionSpatialData) {
        auto p{new VPUEMSpatialExec<dtype, DpuFreq, ShvFreq>(name, CostFunctionSpatialData)};
        addOp(map_content_t(p, &DeviceShaveContainer::deleter));
    }

};

}  // namespace VPUNN
#endif
