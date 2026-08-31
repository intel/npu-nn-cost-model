// Copyright © 2026 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#include "http_client/http_client_test.h"

namespace VPUNN_unit_tests {
using namespace VPUNN;

class SHAVEHTTPClient : public HTTPClientTest<SHAVEWorkload> {
protected:
    // Prepare SHAVEWorkload
    SHAVEWorkload wl{
        "test_operation",  // name
        VPUDevice::VPU_2_7,  // device
        {   // inputs
            VPUTensor({1, 64}, DataType::INT8),  // input tensor 1
            VPUTensor({64, 128}, DataType::INT8) // input tensor 2
        },
        {   // outputs
            VPUTensor({1, 128}, DataType::INT8)  // output tensor
        },
        {   // call_params (explicit types to avoid std::variant<int,float,string,bool> ambiguity in C++17)
            SHAVEWorkload::Param{42},                          // int parameter
            SHAVEWorkload::Param{3.14f},                       // float parameter
            SHAVEWorkload::Param{std::string("param_value")},  // string parameter
            SHAVEWorkload::Param{true}                         // bool parameter
        },
        {   // extra_params (explicit Param for values to avoid const char* -> bool conversion)
            {"extra_int", SHAVEWorkload::Param{7}},
            {"extra_float", SHAVEWorkload::Param{2.718f}},
            {"extra_string", SHAVEWorkload::Param{std::string("extra_value")}},
            {"extra_bool", SHAVEWorkload::Param{false}}
        },
        ProfilingServiceBackend::SILICON,  // profiling_service_backend_hint
        "test_location"                    // loc_name
    };

    void SetUp() override {
        HTTPClientTest<SHAVEWorkload>::SetUp();  // Start mock server
    }
};

TEST_F(SHAVEHTTPClient, GetSHAVECostSuccess) {
    GetCostSuccess(wl);
}

TEST_F(SHAVEHTTPClient, GetSHAVECostProfilingServiceError) {
    GetCostProfilingServiceError(wl);
}

TEST_F(SHAVEHTTPClient, GetSHAVECostMalformedResponse) {
    GetCostMalformedResponse(wl);
}

TEST_F(SHAVEHTTPClient, GetSHAVECostInvalidJsonResponse) {
    GetCostInvalidJsonResponse(wl);
}

TEST_F(SHAVEHTTPClient, GetSHAVECostWithInvalidBackend) {
    GetCostWithInvalidBackend(wl);
}

TEST_F(SHAVEHTTPClient, GetSHAVECostMultipleLatencies) {
    GetCostMultipleLatencies(wl);
}

TEST_F(SHAVEHTTPClient, GetSHAVECostEmptyLatencies) {
    GetCostEmptyLatencies(wl);
}

/**
 * @brief Tests HttpCostProvider's toJson serialization.
 *
 * Verifies that toJson correctly serializes a SHAVEWorkload into the expected JSON format.
 * Tests indirectly through getCost by capturing the request payload.
 */
TEST_F(SHAVEHTTPClient, SHAVEWlAsJsonSerialization) {
    // Create HttpCostProvider instance
    HttpCostProvider cost_provider("localhost", srv_port);

    // Since toJson is a private method, we'll indirectly test it via getCost
    // Setup mock handler for /generate_workload endpoint to capture the request
    _mock_server.Post("/generate_workload", [&](const httplib::Request& req, httplib::Response& res) {
        // Parse the request JSON
        nlohmann::json request = nlohmann::json::parse(req.body);

        if (HandleStatusCheck(request, res)) return;

        // Validate the request structure
        EXPECT_EQ(request["params"]["backend"], "silicon");
        EXPECT_EQ(request["params"]["name"], "profiling_request");
        
        // Verify SHAVE workload uses the correct key
        EXPECT_TRUE(request.contains("shave_workload"));
        EXPECT_FALSE(request.contains("workload"));  // Should not use generic key
        EXPECT_FALSE(request.contains("dma_workload"));
        EXPECT_FALSE(request.contains("dpu_workload"));

        const auto& sw = request["shave_workload"];

        // Validate device and operation
        EXPECT_EQ(sw["device"], "VPUDevice.VPU_2_7");
        EXPECT_EQ(sw["operation"], "test_operation");

        // Validate input 0: VPUTensor({1, 64}, INT8) => width=1, height=64, channels=0, batch=0
        EXPECT_EQ(sw["input_0_width"], 1);
        EXPECT_EQ(sw["input_0_height"], 64);
        EXPECT_EQ(sw["input_0_channels"], 0);
        EXPECT_EQ(sw["input_0_batch"], 0);
        EXPECT_EQ(sw["input_0_datatype"], static_cast<int>(DataType::INT8));
        EXPECT_EQ(sw["input_0_layout"], static_cast<int>(Layout::ZXY));
        EXPECT_EQ(sw["input_0_sparsity_enabled"], false);

        // Validate input 1: VPUTensor({64, 128}, INT8) => width=64, height=128, channels=0, batch=0
        EXPECT_EQ(sw["input_1_width"], 64);
        EXPECT_EQ(sw["input_1_height"], 128);
        EXPECT_EQ(sw["input_1_channels"], 0);
        EXPECT_EQ(sw["input_1_batch"], 0);
        EXPECT_EQ(sw["input_1_datatype"], static_cast<int>(DataType::INT8));
        EXPECT_EQ(sw["input_1_layout"], static_cast<int>(Layout::ZXY));
        EXPECT_EQ(sw["input_1_sparsity_enabled"], false);

        // Validate output 0: VPUTensor({1, 128}, INT8) => width=1, height=128, channels=0, batch=0
        EXPECT_EQ(sw["output_0_width"], 1);
        EXPECT_EQ(sw["output_0_height"], 128);
        EXPECT_EQ(sw["output_0_channels"], 0);
        EXPECT_EQ(sw["output_0_batch"], 0);
        EXPECT_EQ(sw["output_0_datatype"], static_cast<int>(DataType::INT8));
        EXPECT_EQ(sw["output_0_layout"], static_cast<int>(Layout::ZXY));
        EXPECT_EQ(sw["output_0_sparsity_enabled"], false);

        // Validate call_params: {42, 3.14f, "param_value", true}
        EXPECT_EQ(sw["param_0"], std::to_string(42));
        EXPECT_EQ(sw["param_1"], std::to_string(3.14f));
        EXPECT_EQ(sw["param_2"], "param_value");
        EXPECT_EQ(sw["param_3"], "true");

        // Validate extra_params (map is ordered alphabetically)
        // {"extra_bool", false}, {"extra_float", 2.718f}, {"extra_int", 7}, {"extra_string", "extra_value"}
        EXPECT_EQ(sw["extra_param_0"], "extra_bool/false");
        EXPECT_EQ(sw["extra_param_1"], "extra_float/" + std::to_string(2.718f));
        EXPECT_EQ(sw["extra_param_2"], "extra_int/7");
        EXPECT_EQ(sw["extra_param_3"], "extra_string/extra_value");

        // Send a successful response
        nlohmann::json response;
        response["info"] = "success";
        response["latencies"] = std::vector<CyclesInterfaceType>{5678};
        res.set_content(response.dump(), "application/json");
        res.status = 200;
    });

    // Prepare info string
    std::string info;

    // Get cost, which will trigger toJson
    CyclesInterfaceType cycles = cost_provider.getCost(wl, info);

    // Validate cost
    EXPECT_EQ(cycles, 5678);
}

TEST_F(SHAVEHTTPClient, GetSHAVECostWithWarning) {
    GetCostWithWarning(wl);
}

TEST_F(SHAVEHTTPClient, GetSHAVECostWithTrace) {
    GetCostWithTrace(wl);
}

} // namespace VPUNN_unit_tests