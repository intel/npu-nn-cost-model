// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#include "vpu/shave/shave_collection.h"

#include "vpu/shave/shave_collection_NPU40.h"
#include "vpu/shave/shave_collection_VPU27.h"

#include "vpu/shave/layers.h"
#include "vpu/shave/VPUEM_cost_function.h"
#include "vpu/vpuem_models_struct.h"
#include "vpu/types.h"
#include <array>
#include <vector>
#include <string>

constexpr float VPU27COSTCURVERATIO = 1.0;
constexpr float VPU40COSTCURVERATIO = 2.5;
constexpr int VPU27DSPARCH = 128;
constexpr int VPU40DSPARCH = 512;
constexpr int MAXBLKNUM = 32;

constexpr char opPrefix[] = "vpuem.";

namespace VPUNN {
// this file gets  active when we have a shave lib. Intention is to generate the populate method!

void ShaveInstanceHolder_VPU27::populate() {
    // add concrete instances, generated or by hand

    // clang-format off
         //Add<DataType::FLOAT16, 8, 16, 1300,975>("sigmoid", 8.040653645736843e-05F, 2.3877565530828724F,0.052000000000000046F, 0.6203295622495082F);
         //Add("Mocking1");
        // Add<SHVHardSigmoid, int(0.547F * 1000),4956>("HardSigmoid");//activation
        // Add<SHVTranspose, int(0.1F * 1000),1000>("Transpose");//data movement
        // Add<SHVMinimum, int(0.015F * 1000),11047>("Minimum");//element wise

        AddMVN6OneAx<DataType::FLOAT16, 8, 16, 1300,975>("MVN6_onlyOneAx", 0.199457115f,11.22497677f, 0.068f,   0.369384878f);

        //to do add here MOre MVN's
        //                          slope,        intercept      alpha,   worst_case_slope,  slope_delta_diff
        constexpr MVN6Parameters axes_1{0.199457115f, 11.22497677f, 0.068f,   0.568841993f,      0.0f};
        constexpr MVN6Parameters axes_2{0.222078806f, 11.10391141f, 0.068f,   0.614104853f,      0.067853727f};
        constexpr MVN6Parameters axes_3{0.244695794f, 11.05341681f, 0.068f,   0.693300545f,      0.067856774f};
        constexpr MVN6Parameters axes_4{0.267313214f, 11.02275138f, 0.068f,   0.470888195f,      0.067858486f};

        AddMVN6MultiAx<DataType::FLOAT16, 8, 16, 1300,975,1>("MVN6_oneAx",   axes_1);
        AddMVN6MultiAx<DataType::FLOAT16, 8, 16, 1300,975,2>("MVN6_twoAx",   axes_2);
        AddMVN6MultiAx<DataType::FLOAT16, 8, 16, 1300,975,3>("MVN6_threeAx", axes_3);
        AddMVN6MultiAx<DataType::FLOAT16, 8, 16, 1300,975,4>("MVN6_fourAx",  axes_4);

        //AddMVN6MultiAx<DataType::FLOAT16, 8, 16, 1300,975,-55 >("MVN6_fourAxis",  0.267313214f, 11.02275138f, 0.068f,   0.470888195f,      0.067858486f);

        AddMVN6Generic<DataType::FLOAT16, 8, 16, 1300,975>("MVN6",   axes_1,axes_2,axes_3,axes_4);

        //                                               axis            BaseSlope,   BaseIntercept  ThirdmostSupportSlope  BaseSupportSlope Mod8SupportSlope VectorSlope
        AddMVN_SimpleNAx<DataType::FLOAT16,8,8,1300,975, 3 >("MVN_3Ax", 0.383495706f, 6.636145301f, 0.0f  /*NA*/         , 0.108142857f,    0.03883739f,     0.080619048f);
        AddMVN_SimpleNAx<DataType::FLOAT16,8,8,1300,975, 2 >("MVN_2Ax", 0.449314357f, 11.19068564f, 0.385111764705882f   , 0.109375f,       0.03177f,        0.09636f);

        //composite
         AddMVN_CompositeGeneric("MVN", "MVN_2Ax", "MVN_3Ax",  "MVN6");// depends on previous added models, name wise

         //INterpolatePoly
         AddPoly_WHO<DataType::FLOAT16, 1300,975, InterpolateWHModel_1>("interpolatewh_1");

         // VPUEM PieceWise 
         const std::vector<CostFunction3SlopesDescriptor> costFunction3SlopesDataSigmoid = {
         {8, 124.99999999999993F, {2.782608695652174F, 0.4923076923076922F, 0.04669260700389106F}},
        {16, 140.99999999999991F, {3.657142857142858F, 0.4923076923076922F, 0.04669260700389106F}},
        {32, 140.9999999999998F, {3.1999999999999993F, 0.4923076923076922F, 0.04669260700389106F}},
        {64, 140.9999999999991F, {1.182448036951501F, 0.4923076923076922F, 0.04669260700389106F}}};

         
         AddVPUEM_Piecewise<DataType::FLOAT16,1300,975>(std::string(opPrefix) + "sigmoid", costFunction3SlopesDataSigmoid, true, MAXBLKNUM, VPU27DSPARCH, VPU27COSTCURVERATIO);

        const std::vector<CostFunction3SlopesDescriptor> costFunction3SlopesDataAdd = {
        {8, 124.99999999999993F, {2.782608695652174F, 0.5614035087719298F, 0.08053691275167783F}},
        {16, 140.99999999999986F, {3.047619047619047F, 0.5614035087719298F, 0.08053691275167783F}},
        {32, 140.99999999999997F, {3.324675324675324F, 0.5614035087719298F, 0.08053691275167783F}},
        {64, 140.99999999999875F, {1.3061224489795917F, 0.5614035087719298F, 0.08053691275167783F}}};
         
         AddVPUEM_Piecewise<DataType::FLOAT16,1300,975>(std::string(opPrefix) + "add", costFunction3SlopesDataAdd, true, MAXBLKNUM, VPU27DSPARCH, VPU27COSTCURVERATIO);

        const std::vector<CostFunction3SlopesDescriptor> costFunction3SlopesDataSoftmax = {
        {8, 124.99999999999993F, {2.782608695652174F, 0.4923076923076922F, 0.04669260700389106F}},
        {16, 140.99999999999991F, {3.657142857142858F, 0.4923076923076922F, 0.04669260700389106F}},
        {32, 140.9999999999998F, {3.1999999999999993F, 0.4923076923076922F, 0.04669260700389106F}},
        {64, 140.9999999999991F, {1.182448036951501F, 0.4923076923076922F, 0.04669260700389106F}}};
        
        AddVPUEM_Piecewise<DataType::FLOAT16,1300,975>(std::string(opPrefix) + "softmax", costFunction3SlopesDataSoftmax, true, MAXBLKNUM, VPU27DSPARCH, VPU27COSTCURVERATIO);

        const std::vector<CostFunction3SlopesDescriptor> costFunction3SlopesDataGelu = {
        {8, 139.99999999999974F, {0.5565217391304347F, 0.214765100671141F, 0.025369978858350954F}},
        {16, 140.99999999999935F, {0.5638766519823788F, 0.214765100671141F, 0.025369978858350954F}},
        {32, 149.99999999999906F, {0.6052009456264775F, 0.214765100671141F, 0.025369978858350954F}},
        {64, 140.99999999999787F, {0.43025210084033616F, 0.214765100671141F, 0.025369978858350954F}}};

        AddVPUEM_Piecewise<DataType::FLOAT16,1300,975>(std::string(opPrefix) + "gelu", costFunction3SlopesDataGelu, true, MAXBLKNUM, VPU27DSPARCH, VPU27COSTCURVERATIO);

        const std::vector<CostFunction3SlopesDescriptor> costFunction3SlopesDataHSWISH = {
        {8, 139.99999999999994F, {1.230769230769231F, 0.35955056179775274F, 0.05150214592274678F}},
        {16, 140.99999999999966F, {1.4712643678160917F, 0.35955056179775274F, 0.05150214592274678F}},
        {32, 147.99999999999955F, {1.6953642384105958F, 0.35955056179775274F, 0.05150214592274678F}},
        {64, 141.99999999999818F, {0.7864823348694315F, 0.35955056179775274F, 0.05150214592274678F}}};

        AddVPUEM_Piecewise<DataType::FLOAT16,1300,975>(std::string(opPrefix) + "hswish", costFunction3SlopesDataHSWISH, true, MAXBLKNUM, VPU27DSPARCH, VPU27COSTCURVERATIO);

        const std::vector<CostFunction3SlopesDescriptor> costFunction3SlopesDataLOG = {
        {8, 124.99999999999993F, {2.782608695652174F, 0.4923076923076922F, 0.04669260700389106F}},
        {16, 140.99999999999991F, {3.657142857142858F, 0.4923076923076922F, 0.04669260700389106F}},
        {32, 140.9999999999998F, {3.1999999999999993F, 0.4923076923076922F, 0.04669260700389106F}},
        {64, 140.9999999999991F, {1.182448036951501F, 0.4923076923076922F, 0.04669260700389106F}}};

        AddVPUEM_Piecewise<DataType::FLOAT16,1300,975>(std::string(opPrefix) + "log", costFunction3SlopesDataLOG, true, MAXBLKNUM, VPU27DSPARCH, VPU27COSTCURVERATIO);

        const std::vector<CostFunction3SlopesDescriptor> costFunction3SlopesDataMUL = {
        {8, 124.99999999999993F, {2.782608695652174F, 0.5614035087719298F, 0.08053691275167783F}},
        {16, 140.99999999999986F, {3.047619047619047F, 0.5614035087719298F, 0.08053691275167783F}},
        {32, 140.99999999999997F, {3.324675324675324F, 0.5614035087719298F, 0.08053691275167783F}},
        {64, 140.99999999999875F, {1.3061224489795917F, 0.5614035087719298F, 0.08053691275167783F}}};

        AddVPUEM_Piecewise<DataType::FLOAT16,1300,975>(std::string(opPrefix) + "mul", costFunction3SlopesDataMUL, true, MAXBLKNUM, VPU27DSPARCH, VPU27COSTCURVERATIO);

        const std::vector<CostFunction3SlopesDescriptor> costFunction3SlopesDataSWISH = {
        {8, 200.0F, {1.4222222222222216F, 0.3636363636363636F, 0.08510638297872339F}},
        {16, 200.0F, {1.7534246575342458F, 0.3636363636363636F, 0.08510638297872339F}},
        {32, 208.0F, {1.9104477611940291F, 0.3636363636363636F, 0.08510638297872339F}},
        {64, 201.0F, {0.8101265822784806F, 0.3636363636363636F, 0.08510638297872339F}}};

        AddVPUEM_Piecewise<DataType::FLOAT16,1300,975>(std::string(opPrefix) + "swish", costFunction3SlopesDataSWISH, true, MAXBLKNUM, VPU27DSPARCH, VPU27COSTCURVERATIO);

        const std::vector<CostFunction3SlopesDescriptor> costFunction3SlopesDataTANH = {
        {8, 124.99999999999993F, {2.782608695652174F, 0.4923076923076922F, 0.04669260700389106F}},
        {16, 140.99999999999991F, {3.657142857142858F, 0.4923076923076922F, 0.04669260700389106F}},
        {32, 140.9999999999998F, {3.1999999999999993F, 0.4923076923076922F, 0.04669260700389106F}},
        {64, 140.9999999999991F, {1.182448036951501F, 0.4923076923076922F, 0.04669260700389106F}}};

        AddVPUEM_Piecewise<DataType::FLOAT16,1300,975>(std::string(opPrefix) + "tanh", costFunction3SlopesDataTANH, true, MAXBLKNUM, VPU27DSPARCH, VPU27COSTCURVERATIO);


         // VPUEM Softmax
         const std::vector<CostFunctionSoftmaxDescriptor> costFunctionSoftmaxData = {
             { false, 128,{{8, 283, 57, 158, 245, 29, 46, 120, 244, 26, 44, 119}, { 4, 0, 0, 0, 151, 160, 160, 0, 20, 52, 45, 0}}},
             { false, 512,{{8, 284, 57, 208, 236, 29, 47, 160, 239, 31, 46, 163}, {8, 137, 29, 72, 128, 16, 18, 63, 127, 15, 18, 62}}}
         };

        AddVPUEM_Softmax<DataType::FLOAT16,1300,975>(std::string(opPrefix) + "softmax_x", costFunctionSoftmaxData);

        const std::vector<CostFunctionSoftmaxDescriptor> costFunctionGPTQData = {
             { false, 128,{{8, 258, 26, 121, 213, 38, 30, 106, 219, 36, 22, 82}}}};

        AddVPUEM_Softmax<DataType::FLOAT16,1300,975>(std::string(opPrefix) + "gptq", costFunctionGPTQData);

        const std::vector<CostFunctionSoftmaxDescriptor> costFunctionMVNData = {
             { true, 128,{{8, 252, 68, 100, 266, 41, 32, 114, 0, 0, 0, 0}}},
             { true, 512,{{8, 236, 49, 169, 218, 20, 25, 150, 0, 0, 0, 0}}}
         };

        AddVPUEM_Softmax<DataType::FLOAT16,1300,975>(std::string(opPrefix) + "mvn", costFunctionMVNData);

        const std::vector<CostFunctionSoftmaxDescriptor> costFunctionMVNFusedData = {
             { true, 128,{{8, 270, 93, 134, 265, 47, 38, 129, 0, 0, 0, 0}}},
             { true, 512,{{8, 2172, 57, 2096, 2151, 22, 27, 2075, 0, 0, 0, 0}}}
         };

        AddVPUEM_Softmax<DataType::FLOAT16,1300,975>(std::string(opPrefix) + "mvn_fused", costFunctionMVNFusedData);

        const std::vector<CostFunctionSoftmaxDescriptor> costFunctionMVNcwData = {
            {false, 128, {{4, 303, 22, 176, 254, 46, 38, 127, 253, 37, 30, 126}}},
            {false, 512, {{8, 220, 49, 147, 199, 20, 25, 126, 200, 21, 25, 127}}}};

        AddVPUEM_Softmax<DataType::FLOAT16,1300,975>(std::string(opPrefix) + "mvn_cw", costFunctionMVNcwData);
        // VPUEM Spatial 
 
        const CostFunctionSpatialDescriptor CostFunctionSpatialData = {8, 137, 373, "NHWC", {0.8888888888888889F}};
        AddVPUEM_Spatial<DataType::FLOAT16,1300,975>(std::string(opPrefix) + "mvn", CostFunctionSpatialData);
       // the inline file contains the autogenerated Add method calls. The add inplementations are in DeviceShaveContainer, 
       //should support insertion of all known operators types
         #include "vpu/shave/SHAVE_V27.inl"

    // clang-format on
}

void ShaveInstanceHolder_VPU27CLassic::populate() {
    // adauga 1 instanta concreta
    // clang-format off
         //Add<SHVHardSigmoid, int(0.547F * 1000),4956>("HardSigmoid");//activation
        // Add<SHVTranspose, int(0.1F * 1000),1000>("Transpose");//data movement
         //Add<SHVMinimum, int(0.015F * 1000),11047>("Minimum");//element wise
        
        #include "vpu/shave/SHAVE_V27_Linear.inl"

    // clang-format on
}

//////////////////////////////////////////////

void ShaveInstanceHolder_NPU40::populate() {
    // add concrete instances, generated or by hand
    // clang-format off
   
   // the inline file contains the autogenerated Add method calls. The add implementations are in DeviceShaveContainer, 
   //should support insertion of all known operators types
   
   //Adding MVN6 
   //                          slope,        intercept          alpha,  worst_case_slope,   slope_delta_diff
   constexpr MVN6Parameters axes_1{0.253809942f, 4.65838345114753f, 0.068f, 0.781300211482826f,             0.0f};
   constexpr MVN6Parameters axes_2{0.284761941f, 4.802922262f,      0.068f,       0.873129308f,     0.076349768f};
   constexpr MVN6Parameters axes_3{ 0.31571402f, 4.847094026f,      0.068f,          0.926787f,     0.076348779f};
   constexpr MVN6Parameters axes_4{0.346667233f, 5.005446312f,      0.068f,      0.5757150057f,     0.076348834f};

   AddMVN6MultiAx<DataType::FLOAT16, 32, 16, 1700,971,1>("MVN6_oneAx",   axes_1);
   AddMVN6MultiAx<DataType::FLOAT16, 32, 16, 1700,971,2>("MVN6_twoAx",   axes_2);
   AddMVN6MultiAx<DataType::FLOAT16, 32, 16, 1700,971,3>("MVN6_threeAx", axes_3);
   AddMVN6MultiAx<DataType::FLOAT16, 32, 16, 1700,971,4>("MVN6_fourAx",  axes_4);

   AddMVN6Generic<DataType::FLOAT16, 32, 16, 1700,971>("MVN6",   axes_1,axes_2,axes_3,axes_4);

   //                                                           axis            BaseSlope,   BaseIntercept  ThirdmostSupportSlope  BaseSupportSlope ModVectorSupportSlope   VectorSlope
   AddMVN_SimpleNAx<DataType::FLOAT16, 32, 8, 1700, 971, 3>("MVN_3Ax", 0.951687829916732f, 2.71686787979706f, 0.0f  /*NA*/         , 0.12153333333f,   0.04232083333f,      0.20902833333f);
   AddMVN_SimpleNAx<DataType::FLOAT16, 32, 8, 1700, 971, 2>("MVN_2Ax",       0.995737034f,      2.249996098f, 0.938546667f         ,         0.125f,     0.041261538f,            0.20868f);
    
    //composite
    AddMVN_CompositeGeneric("MVN", "MVN_2Ax", "MVN_3Ax",  "MVN6");// depends on previous added models, name wise

    //Softmax
    constexpr SoftmaxEquationParams e1{ {0.002682965f,  0.009700778f}, { 0.09399437f,  12.44775737f}};
    constexpr SoftmaxEquationParams e2{{0.002430508f,  0.010895337f}, {0.084965949f,	12.6691893f}};
    constexpr SoftmaxEquationParams e4{{0.002367099f,	0.009812789f}, {0.083526381f,	12.45620083f}};
    constexpr SoftmaxEquationParams e8{{0.001992061f,	0.010255307f}, {0.067058858f,	12.47844217f}};
    constexpr SoftmaxEquationParams e16{{0.001258255f,	0.010924353f}, {0.046222578f,	12.42546444f}}; 
    constexpr SoftmaxEquationParams e32{{0.001256785f,	0.011003726f}, {0.000409401f,	8.728951463f}}; 
    constexpr FirstDegreeEquation baseEqSoftmax{0.000783649f, 7.660420549f};

    AddSoftmax<DataType::FLOAT16, 32, 4, 1700, 971>("softmax", baseEqSoftmax.slope_, baseEqSoftmax.intercept_, e1, e2, e4, e8, e16, e32);

    AddGather<DataType::FLOAT16, 8, 1700, 971>("gather", 0.001883862f, 4.790410425f,0.0945429470829462f, 0.219886254182408f, 0.013678893f);
    
         // VPUEM PieceWise 
         const std::vector<CostFunction3SlopesDescriptor> costFunction3SlopesDataSigmoid = {
         {8, 124.99999999999993F, {2.782608695652174F, 0.4923076923076922F, 0.04669260700389106F}},
        {16, 140.99999999999991F, {3.657142857142858F, 0.4923076923076922F, 0.04669260700389106F}},
        {32, 140.9999999999998F, {3.1999999999999993F, 0.4923076923076922F, 0.04669260700389106F}},
        {64, 140.9999999999991F, {1.182448036951501F, 0.4923076923076922F, 0.04669260700389106F}}};

         // Nominal freq from VPUEM: VPU freq = 1057 and DPU freq = 1850, the raport between then is the same with the raport of nominal cost model freqs constante in loc de valori
         AddVPUEM_Piecewise<DataType::FLOAT16, 1700, 971>(std::string(opPrefix) + "sigmoid", costFunction3SlopesDataSigmoid, true, MAXBLKNUM, VPU40DSPARCH, VPU40COSTCURVERATIO);

         const std::vector<CostFunction3SlopesDescriptor> costFunction3SlopesDataAdd = {
        {8, 124.99999999999993F, {2.782608695652174F, 0.5614035087719298F, 0.08053691275167783F}},
        {16, 140.99999999999986F, {3.047619047619047F, 0.5614035087719298F, 0.08053691275167783F}},
        {32, 140.99999999999997F, {3.324675324675324F, 0.5614035087719298F, 0.08053691275167783F}},
        {64, 140.99999999999875F, {1.3061224489795917F, 0.5614035087719298F, 0.08053691275167783F}}};
         
         AddVPUEM_Piecewise<DataType::FLOAT16, 1700, 971>(std::string(opPrefix) + "add", costFunction3SlopesDataAdd, true, MAXBLKNUM, VPU40DSPARCH, VPU40COSTCURVERATIO);

        const std::vector<CostFunction3SlopesDescriptor> costFunction3SlopesDataSoftmax = {
        {8, 124.99999999999993F, {2.782608695652174F, 0.4923076923076922F, 0.04669260700389106F}},
        {16, 140.99999999999991F, {3.657142857142858F, 0.4923076923076922F, 0.04669260700389106F}},
        {32, 140.9999999999998F, {3.1999999999999993F, 0.4923076923076922F, 0.04669260700389106F}},
        {64, 140.9999999999991F, {1.182448036951501F, 0.4923076923076922F, 0.04669260700389106F}}};
        
        AddVPUEM_Piecewise<DataType::FLOAT16, 1700, 971>(std::string(opPrefix) + "softmax", costFunction3SlopesDataSoftmax, true, MAXBLKNUM, VPU40DSPARCH, VPU40COSTCURVERATIO);

        const std::vector<CostFunction3SlopesDescriptor> costFunction3SlopesDataGelu = {
        {8, 139.99999999999974F, {0.5565217391304347F, 0.214765100671141F, 0.025369978858350954F}},
        {16, 140.99999999999935F, {0.5638766519823788F, 0.214765100671141F, 0.025369978858350954F}},
        {32, 149.99999999999906F, {0.6052009456264775F, 0.214765100671141F, 0.025369978858350954F}},
        {64, 140.99999999999787F, {0.43025210084033616F, 0.214765100671141F, 0.025369978858350954F}}};

        AddVPUEM_Piecewise<DataType::FLOAT16, 1700, 971>(std::string(opPrefix) + "gelu", costFunction3SlopesDataGelu, true, MAXBLKNUM, VPU40DSPARCH, VPU40COSTCURVERATIO);

        const std::vector<CostFunction3SlopesDescriptor> costFunction3SlopesDataHSWISH = {
        {8, 139.99999999999994F, {1.230769230769231F, 0.35955056179775274F, 0.05150214592274678F}},
        {16, 140.99999999999966F, {1.4712643678160917F, 0.35955056179775274F, 0.05150214592274678F}},
        {32, 147.99999999999955F, {1.6953642384105958F, 0.35955056179775274F, 0.05150214592274678F}},
        {64, 141.99999999999818F, {0.7864823348694315F, 0.35955056179775274F, 0.05150214592274678F}}};

        AddVPUEM_Piecewise<DataType::FLOAT16, 1700, 971>(std::string(opPrefix) + "hswish", costFunction3SlopesDataHSWISH, true, MAXBLKNUM, VPU40DSPARCH, VPU40COSTCURVERATIO);

        const std::vector<CostFunction3SlopesDescriptor> costFunction3SlopesDataLOG = {
        {8, 124.99999999999993F, {2.782608695652174F, 0.4923076923076922F, 0.04669260700389106F}},
        {16, 140.99999999999991F, {3.657142857142858F, 0.4923076923076922F, 0.04669260700389106F}},
        {32, 140.9999999999998F, {3.1999999999999993F, 0.4923076923076922F, 0.04669260700389106F}},
        {64, 140.9999999999991F, {1.182448036951501F, 0.4923076923076922F, 0.04669260700389106F}}};

        AddVPUEM_Piecewise<DataType::FLOAT16, 1700, 971>(std::string(opPrefix) + "log", costFunction3SlopesDataLOG, true, MAXBLKNUM, VPU40DSPARCH, VPU40COSTCURVERATIO);

        const std::vector<CostFunction3SlopesDescriptor> costFunction3SlopesDataMUL = {
        {8, 124.99999999999993F, {2.782608695652174F, 0.5614035087719298F, 0.08053691275167783F}},
        {16, 140.99999999999986F, {3.047619047619047F, 0.5614035087719298F, 0.08053691275167783F}},
        {32, 140.99999999999997F, {3.324675324675324F, 0.5614035087719298F, 0.08053691275167783F}},
        {64, 140.99999999999875F, {1.3061224489795917F, 0.5614035087719298F, 0.08053691275167783F}}};

        AddVPUEM_Piecewise<DataType::FLOAT16, 1700, 971>(std::string(opPrefix) + "mul", costFunction3SlopesDataMUL, true, MAXBLKNUM, VPU40DSPARCH, VPU40COSTCURVERATIO);

        const std::vector<CostFunction3SlopesDescriptor> costFunction3SlopesDataSWISH = {
        {8, 200.0F, {1.4222222222222216F, 0.3636363636363636F, 0.08510638297872339F}},
        {16, 200.0F, {1.7534246575342458F, 0.3636363636363636F, 0.08510638297872339F}},
        {32, 208.0F, {1.9104477611940291F, 0.3636363636363636F, 0.08510638297872339F}},
        {64, 201.0F, {0.8101265822784806F, 0.3636363636363636F, 0.08510638297872339F}}};

        AddVPUEM_Piecewise<DataType::FLOAT16, 1700, 971>(std::string(opPrefix) + "swish", costFunction3SlopesDataSWISH, true, MAXBLKNUM, VPU40DSPARCH, VPU40COSTCURVERATIO);

        const std::vector<CostFunction3SlopesDescriptor> costFunction3SlopesDataTANH = {
        {8, 124.99999999999993F, {2.782608695652174F, 0.4923076923076922F, 0.04669260700389106F}},
        {16, 140.99999999999991F, {3.657142857142858F, 0.4923076923076922F, 0.04669260700389106F}},
        {32, 140.9999999999998F, {3.1999999999999993F, 0.4923076923076922F, 0.04669260700389106F}},
        {64, 140.9999999999991F, {1.182448036951501F, 0.4923076923076922F, 0.04669260700389106F}}};

        AddVPUEM_Piecewise<DataType::FLOAT16, 1700, 971>(std::string(opPrefix) + "tanh", costFunction3SlopesDataTANH, true, MAXBLKNUM, VPU40DSPARCH, VPU40COSTCURVERATIO);

         

         

         // VPUEM Softmax
         const std::vector<CostFunctionSoftmaxDescriptor> costFunctionSoftmaxData = {
             { false, 128,{{8, 283, 57, 158, 245, 29, 46, 120, 244, 26, 44, 119}, { 4, 0, 0, 0, 151, 160, 160, 0, 20, 52, 45, 0}}},
             { false, 512,{{8, 284, 57, 208, 236, 29, 47, 160, 239, 31, 46, 163}, {8, 137, 29, 72, 128, 16, 18, 63, 127, 15, 18, 62}}}
         };

        AddVPUEM_Softmax<DataType::FLOAT16, 1700, 971>(std::string(opPrefix) + "softmax_x", costFunctionSoftmaxData);

        // VPUEM Spatial 
 
        const CostFunctionSpatialDescriptor CostFunctionSpatialData = {8, 137, 373, "NHWC", {0.8888888888888889F}};
        AddVPUEM_Spatial<DataType::FLOAT16, 1700, 971>(std::string(opPrefix) + "mvn", CostFunctionSpatialData);

    //NormalizeL2OnlyC                                                                  Base Slope,    Base Intercept, Base Vector Offset, W Base Slope,   W Base Intercept,  Slope Mod 1,  Slope mod 8,  Slope mod 9,  W Vector Offset
    AddNormalizeL2OnlyC<DataType::FLOAT16, 32, 16, 1700, 971>("normalizel2onlyc", 0.0010080518479f, 3.94000845803864f,             0.039f, 0.010424209f, 0.311680777926308f, 0.006735642f, 0.003232719f, 0.013133901f,        0.004891f);

   // Commenting out until we have new CSV PARSER in cmake for NPU40
   #include "vpu/shave/SHAVE_V40.inl"

    // clang-format on
}
}  // namespace VPUNN