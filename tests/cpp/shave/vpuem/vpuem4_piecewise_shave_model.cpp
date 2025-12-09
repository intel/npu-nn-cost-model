#include "vpu/shave/VPUEM_models.h"

#include <gtest/gtest.h>
#include "vpu/types.h"
#include "vpu/vpuem_types.h"

#include <fstream>
#include <iostream>
#include <vector>

namespace VPUNN_unit_tests {
using namespace VPUNN;

struct OperationTestData {
    const VPUNN::PiecewiseModel& operation;
    std::vector<int> expected_cycles;
};


class VPUEM4PiecewiseShvTest : public ::testing::TestWithParam<OperationTestData> {
protected:
    std::vector<SHAVEWorkload> test_vectors = {{
                                                       "test_op",
                                                       VPUDevice::VPU_4_0,
                                                       {VPUTensor(32, 32, 32, 1, DataType::FLOAT16, Layout::ZXY)},
                                                       {VPUTensor(32, 32, 32, 1, DataType::FLOAT16, Layout::ZXY)},
                                               },

                                               {
                                                       "test_op",
                                                       VPUDevice::VPU_4_0,
                                                       {VPUTensor(15, 32, 15, 1, DataType::FLOAT16, Layout::ZXY)},
                                                       {VPUTensor(15, 32, 15, 1, DataType::FLOAT16, Layout::ZXY)},
                                               },

                                               {
                                                       "test_op",
                                                       VPUDevice::VPU_4_0,
                                                       {VPUTensor(8, 32, 8, 1, DataType::FLOAT16, Layout::ZXY)},
                                                       {VPUTensor(8, 32, 8, 1, DataType::FLOAT16, Layout::ZXY)},
                                               },
                                               {
                                                       "test_op",
                                                       VPUDevice::VPU_4_0,
                                                       {VPUTensor(17, 32, 17, 1, DataType::FLOAT16, Layout::ZXY)},
                                                       {VPUTensor(17, 32, 17, 1, DataType::FLOAT16, Layout::ZXY)},
                                               },

                                               {
                                                       "test_op",
                                                       VPUDevice::VPU_4_0,
                                                       {VPUTensor(29, 32, 29, 1, DataType::FLOAT16, Layout::ZXY)},
                                                       {VPUTensor(29, 32, 29, 1, DataType::FLOAT16, Layout::ZXY)},
                                               }};

    inline static const std::vector<CostFunction3SlopesDescriptor> costFunction3SlopesDataSigmoid = {
            {8, 124.99999999999993F, {2.782608695652174F, 0.4923076923076922F, 0.04669260700389106F}},
            {16, 140.99999999999991F, {3.657142857142858F, 0.4923076923076922F, 0.04669260700389106F}},
            {32, 140.9999999999998F, {3.1999999999999993F, 0.4923076923076922F, 0.04669260700389106F}},
            {64, 140.9999999999991F, {1.182448036951501F, 0.4923076923076922F, 0.04669260700389106F}}};

    inline static const VPUNN::PiecewiseModel vpuem_sigmoid{
            VPUNN::DataType::FLOAT16, 1700, 971, costFunction3SlopesDataSigmoid, true, 32, 512, 2.5F};

    inline static const std::vector<CostFunction3SlopesDescriptor> costFunction3SlopesDataAdd = {
            {8, 124.99999999999993F, {2.782608695652174F, 0.5614035087719298F, 0.08053691275167783F}},
            {16, 140.99999999999986F, {3.047619047619047F, 0.5614035087719298F, 0.08053691275167783F}},
            {32, 140.99999999999997F, {3.324675324675324F, 0.5614035087719298F, 0.08053691275167783F}},
            {64, 140.99999999999875F, {1.3061224489795917F, 0.5614035087719298F, 0.08053691275167783F}}};

    inline static const VPUNN::PiecewiseModel vpuem_add{
            VPUNN::DataType::FLOAT16, 1700, 971, costFunction3SlopesDataAdd, true, 32, 512, 2.5F};

    inline static const std::vector<CostFunction3SlopesDescriptor> costFunction3SlopesDataSoftmax = {
            {8, 124.99999999999993F, {2.782608695652174F, 0.4923076923076922F, 0.04669260700389106F}},
            {16, 140.99999999999991F, {3.657142857142858F, 0.4923076923076922F, 0.04669260700389106F}},
            {32, 140.9999999999998F, {3.1999999999999993F, 0.4923076923076922F, 0.04669260700389106F}},
            {64, 140.9999999999991F, {1.182448036951501F, 0.4923076923076922F, 0.04669260700389106F}}};

    inline static const VPUNN::PiecewiseModel vpuem_softmax{
            VPUNN::DataType::FLOAT16, 1700, 971, costFunction3SlopesDataSoftmax, true, 32, 512, 2.5F};

    inline static const std::vector<CostFunction3SlopesDescriptor> costFunction3SlopesDataGelu = {
            {8, 139.99999999999974F, {0.5565217391304347F, 0.214765100671141F, 0.025369978858350954F}},
            {16, 140.99999999999935F, {0.5638766519823788F, 0.214765100671141F, 0.025369978858350954F}},
            {32, 149.99999999999906F, {0.6052009456264775F, 0.214765100671141F, 0.025369978858350954F}},
            {64, 140.99999999999787F, {0.43025210084033616F, 0.214765100671141F, 0.025369978858350954F}}};

    inline static const VPUNN::PiecewiseModel vpuem_gelu{
            VPUNN::DataType::FLOAT16, 1700, 971, costFunction3SlopesDataGelu, true, 32, 512, 2.5F};

    inline static const std::vector<CostFunction3SlopesDescriptor> costFunction3SlopesDataHSWISH = {
            {8, 139.99999999999994F, {1.230769230769231F, 0.35955056179775274F, 0.05150214592274678F}},
            {16, 140.99999999999966F, {1.4712643678160917F, 0.35955056179775274F, 0.05150214592274678F}},
            {32, 147.99999999999955F, {1.6953642384105958F, 0.35955056179775274F, 0.05150214592274678F}},
            {64, 141.99999999999818F, {0.7864823348694315F, 0.35955056179775274F, 0.05150214592274678F}}};

    inline static const VPUNN::PiecewiseModel vpuem_hswish{
            VPUNN::DataType::FLOAT16, 1700, 971, costFunction3SlopesDataHSWISH, true, 32, 512, 2.5F};

    inline static const std::vector<CostFunction3SlopesDescriptor> costFunction3SlopesDataLOG = {
            {8, 124.99999999999993F, {2.782608695652174F, 0.4923076923076922F, 0.04669260700389106F}},
            {16, 140.99999999999991F, {3.657142857142858F, 0.4923076923076922F, 0.04669260700389106F}},
            {32, 140.9999999999998F, {3.1999999999999993F, 0.4923076923076922F, 0.04669260700389106F}},
            {64, 140.9999999999991F, {1.182448036951501F, 0.4923076923076922F, 0.04669260700389106F}}};

    inline static const VPUNN::PiecewiseModel vpuem_log{
            VPUNN::DataType::FLOAT16, 1700, 971, costFunction3SlopesDataLOG, true, 32, 512, 2.5F};

    inline static const std::vector<CostFunction3SlopesDescriptor> costFunction3SlopesDataMUL = {
            {8, 124.99999999999993F, {2.782608695652174F, 0.5614035087719298F, 0.08053691275167783F}},
            {16, 140.99999999999986F, {3.047619047619047F, 0.5614035087719298F, 0.08053691275167783F}},
            {32, 140.99999999999997F, {3.324675324675324F, 0.5614035087719298F, 0.08053691275167783F}},
            {64, 140.99999999999875F, {1.3061224489795917F, 0.5614035087719298F, 0.08053691275167783F}}};

    inline static const VPUNN::PiecewiseModel vpuem_mul{
            VPUNN::DataType::FLOAT16, 1700, 971, costFunction3SlopesDataMUL, true, 32, 512, 2.5F};

    inline static const std::vector<CostFunction3SlopesDescriptor> costFunction3SlopesDataSWISH = {
            {8, 200.0F, {1.4222222222222216F, 0.3636363636363636F, 0.08510638297872339F}},
            {16, 200.0F, {1.7534246575342458F, 0.3636363636363636F, 0.08510638297872339F}},
            {32, 208.0F, {1.9104477611940291F, 0.3636363636363636F, 0.08510638297872339F}},
            {64, 201.0F, {0.8101265822784806F, 0.3636363636363636F, 0.08510638297872339F}}};

    inline static const VPUNN::PiecewiseModel vpuem_swish{
            VPUNN::DataType::FLOAT16, 1700, 971, costFunction3SlopesDataSWISH, true, 32, 512, 2.5F};

    inline static const std::vector<CostFunction3SlopesDescriptor> costFunction3SlopesDataTANH = {
            {8, 124.99999999999993F, {2.782608695652174F, 0.4923076923076922F, 0.04669260700389106F}},
            {16, 140.99999999999991F, {3.657142857142858F, 0.4923076923076922F, 0.04669260700389106F}},
            {32, 140.9999999999998F, {3.1999999999999993F, 0.4923076923076922F, 0.04669260700389106F}},
            {64, 140.9999999999991F, {1.182448036951501F, 0.4923076923076922F, 0.04669260700389106F}}};

    inline static const VPUNN::PiecewiseModel vpuem_tanh{
            VPUNN::DataType::FLOAT16, 1700, 971, costFunction3SlopesDataTANH, true, 32, 512, 2.5F};


    void SetUp() override {
    }

public:
    static const VPUNN::PiecewiseModel& sigmoid_model() {
        return vpuem_sigmoid;
    }

    static const VPUNN::PiecewiseModel& add_model() {
        return vpuem_add;
    }

    static const VPUNN::PiecewiseModel& softmax_model() {
        return vpuem_softmax;
    }

    static const VPUNN::PiecewiseModel& gelu_model() {
        return vpuem_gelu;
    }

    static const VPUNN::PiecewiseModel& hswish_model() {
        return vpuem_hswish;
    }

    static const VPUNN::PiecewiseModel& log_model() {
        return vpuem_log;
    }

    static const VPUNN::PiecewiseModel& mul_model() {
        return vpuem_mul;
    }
    
    static const VPUNN::PiecewiseModel& swish_model() {
            return vpuem_swish;
    }

    static const VPUNN::PiecewiseModel& tanh_model() {
        return vpuem_tanh;
    }

private:
};

TEST_P(VPUEM4PiecewiseShvTest, OperationTest) {
    OperationTestData operation_param = GetParam();
    for (long unsigned int idx = 0; idx < test_vectors.size(); idx++) {
        EXPECT_EQ(operation_param.operation.getShaveCycles(test_vectors[idx]), operation_param.expected_cycles[idx]);
    }
}

INSTANTIATE_TEST_SUITE_P(VPUEM4PiecewiseShvTests, VPUEM4PiecewiseShvTest,
                         ::testing::Values(OperationTestData{VPUEM4PiecewiseShvTest::sigmoid_model(),
                                                             {4300, 1097, 404, 1361, 3585}},
                          OperationTestData{VPUEM4PiecewiseShvTest::add_model(), {4108, 1039, 388, 1287, 3426}},
                          OperationTestData{VPUEM4PiecewiseShvTest::softmax_model(), {4300, 1097, 404, 1361, 3585}},
                          OperationTestData{VPUEM4PiecewiseShvTest::gelu_model(),
                                            {21845, 5079, 1509, 6439, 18116}},
                          OperationTestData{VPUEM4PiecewiseShvTest::hswish_model(), {7891, 1916, 635, 2404, 6560}},
                          OperationTestData{VPUEM4PiecewiseShvTest::log_model(), {4300, 1097, 404, 1361, 3585}},
                          OperationTestData{VPUEM4PiecewiseShvTest::mul_model(), {4108, 1039, 388, 1287, 3426}},
                          OperationTestData{VPUEM4PiecewiseShvTest::swish_model(), {7119, 1773, 639, 2205, 5931}},
                          OperationTestData{VPUEM4PiecewiseShvTest::tanh_model(), {4300, 1097, 404, 1361, 3585}}));
}  // namespace VPUNN_unit_tests