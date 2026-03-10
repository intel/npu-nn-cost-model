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

class DMAHTTPClientVPU2x : public HTTPClientTest<DMANNWorkload_NPU27> {
protected:
    DMANNWorkload_NPU27 wl_27{
            VPUNN::VPUDevice::VPU_2_7,  // VPUDevice device;  ///< NPU device

            3,     // int num_planes;  ///< starts from 0. 1 plane = 0 as value?
            8192,  // int length;

            4096,  // int src_width;
            512,   // int dst_width;
            128,   // int src_stride;
            0,     // int dst_stride;
            128,   // int src_plane_stride;
            1024,  // int dst_plane_stride;

            MemoryDirection::DDR2DDR  // MemoryDirection transfer_direction
    };

    void SetUp() override {
        HTTPClientTest<DMANNWorkload_NPU27>::SetUp();  // Start mock server
        wl_27.profiling_service_backend_hint = ProfilingServiceBackend::SILICON;
    }
};

TEST_F(DMAHTTPClientVPU2x, GetDMACostSuccess) {
    GetCostSuccess(wl_27);
}

TEST_F(DMAHTTPClientVPU2x, GetDMACostProfilingServiceError) {
    GetCostProfilingServiceError(wl_27);
}

TEST_F(DMAHTTPClientVPU2x, GetDMACostMalformedResponse) {
    GetCostMalformedResponse(wl_27);
}

TEST_F(DMAHTTPClientVPU2x, GetDMACostInvalidJsonResponse) {
    GetCostInvalidJsonResponse(wl_27);
}

TEST_F(DMAHTTPClientVPU2x, GetDMACostWithInvalidBackend) {
    GetCostWithInvalidBackend(wl_27);
}

TEST_F(DMAHTTPClientVPU2x, GetDMACostMultipleLatencies) {
    GetCostMultipleLatencies(wl_27);
}

TEST_F(DMAHTTPClientVPU2x, GetDMACostEmptyLatencies) {
    GetCostEmptyLatencies(wl_27);
}

/**
 * @brief Tests HttpCostProvider's toJson serialization.
 *
 * Verifies that toJson correctly serializes a DMANNWorkload_NPU27 into the expected JSON format.
 * Tests indirectly through getCost by capturing the request payload.
 */
TEST_F(DMAHTTPClientVPU2x, DMAWlAsJsonSerialization) {
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
        
        // Validate the serialized DMANNWorkload_NPU27 fields
        EXPECT_EQ(request["dma_workload"]["num_planes"], 3);
        EXPECT_EQ(request["dma_workload"]["length"], 8192);
        EXPECT_EQ(request["dma_workload"]["device"], "VPUDevice.VPU_2_7");
        EXPECT_EQ(request["dma_workload"]["src_width"], 4096);
        EXPECT_EQ(request["dma_workload"]["dst_width"], 512);
        EXPECT_EQ(request["dma_workload"]["src_stride"], 128);
        EXPECT_EQ(request["dma_workload"]["dst_stride"], 0);
        EXPECT_EQ(request["dma_workload"]["src_plane_stride"], 128);
        EXPECT_EQ(request["dma_workload"]["dst_plane_stride"], 1024);
        EXPECT_EQ(request["dma_workload"]["transfer_direction"], "MemoryDirection.DDR2DDR");

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
    CyclesInterfaceType cycles = cost_provider.getCost(wl_27, info);

    // Validate cost
    EXPECT_EQ(cycles, 5678);
}

TEST_F(DMAHTTPClientVPU2x, GetDMACostWithWarning) {
    GetCostWithWarning(wl_27);
}

TEST_F(DMAHTTPClientVPU2x, GetDMACostWithTrace) {
    GetCostWithTrace(wl_27);
}

} // namespace VPUNN_unit_tests