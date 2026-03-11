// Copyright © 2026 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#include "http_client/http_client.h"

namespace VPUNN_unit_tests {
using namespace VPUNN;

class DMAHTTPClientNPU45x : public HTTPClientTest<DMANNWorkload_NPU40_50> {
protected:
    DMANNWorkload_NPU40_50 wl_40_50{
            VPUNN::VPUDevice::VPU_4_0,  // VPUDevice device;  ///< NPU device

            8192,  // int src_width;
            8192,  // int dst_width;

            0,  // int num_dim;
            {{{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}}},
            Num_DMA_Engine::Num_Engine_1,
            MemoryDirection::CMX2CMX  // MemoryDirection transfer_direction;
    };

    void SetUp() override {
        HTTPClientTest<DMANNWorkload_NPU40_50>::SetUp();  // Start mock server
        wl_40_50.profiling_service_backend_hint = ProfilingServiceBackend::SILICON;
    }
};

TEST_F(DMAHTTPClientNPU45x, GetDMACostSuccess) {
    GetCostSuccess(wl_40_50);
}

TEST_F(DMAHTTPClientNPU45x, GetDMACostProfilingServiceError) {
    GetCostProfilingServiceError(wl_40_50);
}

TEST_F(DMAHTTPClientNPU45x, GetDMACostMalformedResponse) {
    GetCostMalformedResponse(wl_40_50);
}

TEST_F(DMAHTTPClientNPU45x, GetDMACostInvalidJsonResponse) {
    GetCostInvalidJsonResponse(wl_40_50);
}

TEST_F(DMAHTTPClientNPU45x, GetDMACostWithInvalidBackend) {
    GetCostWithInvalidBackend(wl_40_50);
}

TEST_F(DMAHTTPClientNPU45x, GetDMACostMultipleLatencies) {
    GetCostMultipleLatencies(wl_40_50);
}

TEST_F(DMAHTTPClientNPU45x, GetDMACostEmptyLatencies) {
    GetCostEmptyLatencies(wl_40_50);
}

/**
 * @brief Tests HttpCostProvider's toJson serialization.
 *
 * Verifies that toJson correctly serializes a DMANNWorkload_NPU40_50 into the expected JSON format.
 * Tests indirectly through getCost by capturing the request payload.
 */
TEST_F(DMAHTTPClientNPU45x, DMAWlAsJsonSerialization) {
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
        
        // Verify DMA workload uses the correct key
        EXPECT_TRUE(request.contains("dma_workload"));
        EXPECT_FALSE(request.contains("workload"));  // Should not use generic key
        
        // Validate the serialized DMANNWorkload_NPU40_50 fields
        EXPECT_EQ(request["dma_workload"]["src_width"], 8192);
        EXPECT_EQ(request["dma_workload"]["num_dim"], 0);
        EXPECT_EQ(request["dma_workload"]["device"], "VPUDevice.VPU_4_0");
        EXPECT_EQ(request["dma_workload"]["num_engine"], "Num_DMA_Engine.Num_Engine_1");
        EXPECT_EQ(request["dma_workload"]["transfer_direction"], "MemoryDirection.CMX2CMX");

        // Send a successful response
        nlohmann::json response;
        response["info"] = "success";
        response["latencies"] = std::vector<CyclesInterfaceType>{9101};
        res.set_content(response.dump(), "application/json");
        res.status = 200;
    });

    // Prepare info string
    std::string info;

    // Get cost, which will trigger toJson
    CyclesInterfaceType cycles = cost_provider.getCost(wl_40_50, info);

    // Validate cost
    EXPECT_EQ(cycles, 9101);
}

TEST_F(DMAHTTPClientNPU45x, GetDMACostWithWarning) {
    GetCostWithWarning(wl_40_50);
}

TEST_F(DMAHTTPClientNPU45x, GetDMACostWithTrace) {
    GetCostWithTrace(wl_40_50);
}

} // namespace VPUNN_unit_tests
