// Copyright © 2026 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#include "http_client/http_client.h"
#include "core/utils.h"
#include <unordered_map>
#include <string>

namespace VPUNN_unit_tests {
using namespace VPUNN;

class DPUHTTPClient : public HTTPClientTest<DPUOperation> {
protected:
    // Prepare DPUOperation
    DPUOperation op;
    // Initialize op as needed for the test
    // For simplicity, assuming default constructor sets up a valid operation

    // Store original environment variable values (static - saved once for all tests)
    inline static std::unordered_map<std::string, std::string> original_env_vars;
    inline static const std::vector<std::string> env_var_names = {
        "ENABLE_VPUNN_PROFILING_SERVICE",
        "VPUNN_PROFILING_SERVICE_HOST",
        "VPUNN_PROFILING_SERVICE_PORT",
        "VPUNN_PROFILING_SERVICE_BACKEND",
        "VPUNN_HTTP_CLIENT_DEBUG"
    };

    static void SetUpTestSuite() {
        // Save original environment variable values once for all tests
        original_env_vars = get_env_vars(env_var_names);
    }
    
    void SetUp() override {
        HTTPClientTest<DPUOperation>::SetUp();  // Start mock server
        op.profiling_service_backend_hint = ProfilingServiceBackend::SILICON;
    }

    void TearDown() override {
        // Restore original environment variable values after each test
        // This ensures cleanup even if a test crashes or fails
        for (const auto& var_name : env_var_names) {
            const auto& original_value = original_env_vars[var_name];
            if (original_value.empty()) {
                unset_env_var(var_name);
            } else {
                set_env_var(var_name, original_value);
            }
        }
        HTTPClientTest<DPUOperation>::TearDown();
    }
};

/**
 * @brief Helper to temporarily redirect std::cout to another stream.
 * 
 * Thread-Safety Considerations:
 * - Modifies the global std::cout.rdbuf(), which is not thread-safe across parallel tests
 * - Safe for use within TEST_F(DPUHTTPClient, ...) because:
 *   * GTest guarantees that tests within the same fixture run sequentially
 *   * Tests in this fixture will never execute in parallel with each other
 * - Not safe if:
 *   * Used across different test fixtures that could run in parallel
 *   * Multiple threads within a single test modify std::cout simultaneously
 */
class CoutRedirect {
public:
    explicit CoutRedirect(std::ostream& new_stream)
        : old_buf(std::cout.rdbuf(new_stream.rdbuf())) {}

    ~CoutRedirect() {
        std::cout.rdbuf(old_buf);
    }

    CoutRedirect(const CoutRedirect&) = delete;
    CoutRedirect& operator=(const CoutRedirect&) = delete;

private:
    std::streambuf* old_buf;
};

TEST_F(DPUHTTPClient, GetDPUCostSuccess) {
    GetCostSuccess(op);
}

TEST_F(DPUHTTPClient, GetDPUCostProfilingServiceError) {
    GetCostProfilingServiceError(op);
}

TEST_F(DPUHTTPClient, GetDPUCostMalformedResponse) {
    GetCostMalformedResponse(op);
}

TEST_F(DPUHTTPClient, GetDPUCostInvalidJsonResponse) {
    GetCostInvalidJsonResponse(op);
}

TEST_F(DPUHTTPClient, GetDPUCostWithInvalidBackend) {
    GetCostWithInvalidBackend(op);
}

TEST_F(DPUHTTPClient, GetDPUCostMultipleLatencies) {
    GetCostMultipleLatencies(op);
}

TEST_F(DPUHTTPClient, GetDPUCostEmptyLatencies) {
    GetCostEmptyLatencies(op);
}

/**
 * @brief Tests HttpDPUCostProvider's toJson serialization.
 *
 * Verifies that toJson correctly serializes a DPUOperation into the expected JSON format.
 */
TEST_F(DPUHTTPClient, DpuOpAsJsonSerialization) {
    // Create HttpCostProvider instance
    HttpCostProvider cost_provider("localhost", srv_port);

    // Prepare DPUOperation with specific values
    DPUOperation local_op;
    local_op.output_write_tiles = 2;
    local_op.in_place_output_memory = true;
    // Initialize other fields as needed

    // Since toJson is a private method, we'll indirectly test it via getCost
    // Setup mock handler for /generate_workload endpoint to capture the request
    _mock_server.Post("/generate_workload", [&](const httplib::Request& req, httplib::Response& res) {
        // Parse the request JSON
        nlohmann::json request = nlohmann::json::parse(req.body);

        if (HandleStatusCheck(request, res)) return;

        // Validate the request structure
        EXPECT_EQ(request["params"]["backend"], "silicon");
        EXPECT_EQ(request["params"]["name"], "profiling_request");
        
        // Verify DPU workload uses the correct key
        EXPECT_TRUE(request.contains("dpu_workload"));
        EXPECT_FALSE(request.contains("workload"));  // Should not use generic key
        
        // Validate the serialized DPUOperation fields
        EXPECT_EQ(request["dpu_workload"]["output_write_tiles"], 2);
        EXPECT_EQ(request["dpu_workload"]["in_place_output"], 1);  // true serialized as 1

        // Send a successful response
        nlohmann::json response;
        response["info"] = "success";
        response["latencies"] = std::vector<CyclesInterfaceType>{1234};
        res.set_content(response.dump(), "application/json");
        res.status = 200;
    });

    // Prepare info string
    std::string info;

    // Get cost, which will trigger dpuop_as_json
    CyclesInterfaceType cycles = cost_provider.getCost(local_op, info);

    // Validate cost
    EXPECT_EQ(cycles, 1234);
}

TEST_F(DPUHTTPClient, GetDPUCostWithWarning) {
    GetCostWithWarning(op);
}

TEST_F(DPUHTTPClient, GetDPUCostWithTrace) {
    GetCostWithTrace(op);
}

/**
 * @brief Tests HttpCostProvider debug functionality using initFromEnvironment.
 *
 * Verifies that the VPUNN_HTTP_CLIENT_DEBUG environment variable is properly read
 * and enables debug output when using the initFromEnvironment() factory method.
 */
TEST_F(DPUHTTPClient, InitFromEnvironmentWithDebug) {
    // Set environment variables for initialization
    std::string port_str = std::to_string(srv_port);
    set_env_var("ENABLE_VPUNN_PROFILING_SERVICE", "TRUE");
    set_env_var("VPUNN_PROFILING_SERVICE_HOST", "localhost");
    set_env_var("VPUNN_PROFILING_SERVICE_PORT", port_str);
    set_env_var("VPUNN_PROFILING_SERVICE_BACKEND", "silicon");
    set_env_var("VPUNN_HTTP_CLIENT_DEBUG", "1");

    // Verify environment variables were actually set
    auto env_check = get_env_vars(env_var_names);
    ASSERT_EQ(env_check["ENABLE_VPUNN_PROFILING_SERVICE"], "TRUE");
    ASSERT_EQ(env_check["VPUNN_PROFILING_SERVICE_HOST"], "localhost");
    ASSERT_EQ(env_check["VPUNN_PROFILING_SERVICE_PORT"], port_str);
    ASSERT_EQ(env_check["VPUNN_PROFILING_SERVICE_BACKEND"], "silicon");
    ASSERT_EQ(env_check["VPUNN_HTTP_CLIENT_DEBUG"], "1");
    
    // Setup mock handler for /generate_workload endpoint
    _mock_server.Post("/generate_workload", [&](const httplib::Request& req, httplib::Response& res) {
        // Parse request
        nlohmann::json request = nlohmann::json::parse(req.body);
        if (HandleStatusCheck(request, res)) return;
        
        // Send a successful response
        nlohmann::json response;
        response["info"] = "success";
        response["latencies"] = std::vector<CyclesInterfaceType>{9999};
        res.set_content(response.dump(), "application/json");
        res.status = 200;
    });

    // Capture stdout to verify debug output
    // Note: CoutRedirect is safe here because GTest runs all TEST_F(DPUHTTPClient, ...) sequentially
    std::ostringstream oss;
    {
        CoutRedirect redirect(oss);
        // Create provider using initFromEnvironment (this should read VPUNN_HTTP_CLIENT_DEBUG)
        auto provider = HttpCostProvider::initFromEnvironment();
        ASSERT_NE(provider, nullptr);
    
        // Make a request to trigger debug output
        std::string info;
        CyclesInterfaceType cycles = provider->getCost(op, info);
        
        // Verify debug messages are present
        EXPECT_TRUE(oss.str().find("[DEBUG]") != std::string::npos) 
            << "Expected debug output but none was found. Captured output:\n" << oss.str();
        EXPECT_TRUE(oss.str().find("HTTPClient::sendJsonRequest") != std::string::npos)
            << "Expected HTTPClient debug messages. Captured output:\n" << oss.str();
        EXPECT_TRUE(oss.str().find("Request payload") != std::string::npos || 
                    oss.str().find("Response received") != std::string::npos)
            << "Expected request/response debug info. Captured output:\n" << oss.str();
        
            // Verify the cost was retrieved successfully
        EXPECT_EQ(cycles, 9999);
    }
}

/**
 * @brief Tests HttpCostProvider without debug when environment variable is not set.
 *
 * Verifies that no debug output is produced when VPUNN_HTTP_CLIENT_DEBUG is not set.
 */
TEST_F(DPUHTTPClient, InitFromEnvironmentWithoutDebug) {
    // Set environment variables for initialization (without debug)
    std::string port_str = std::to_string(srv_port);
    set_env_var("ENABLE_VPUNN_PROFILING_SERVICE", "TRUE");
    set_env_var("VPUNN_PROFILING_SERVICE_HOST", "localhost");
    set_env_var("VPUNN_PROFILING_SERVICE_PORT", port_str);
    set_env_var("VPUNN_PROFILING_SERVICE_BACKEND", "silicon");
    // Explicitly unset debug variable
    unset_env_var("VPUNN_HTTP_CLIENT_DEBUG");

    // Verify environment variables were set correctly
    auto env_check = get_env_vars(env_var_names);
    ASSERT_EQ(env_check["ENABLE_VPUNN_PROFILING_SERVICE"], "TRUE");
    ASSERT_EQ(env_check["VPUNN_PROFILING_SERVICE_HOST"], "localhost");
    ASSERT_EQ(env_check["VPUNN_PROFILING_SERVICE_PORT"], std::to_string(srv_port));
    ASSERT_EQ(env_check["VPUNN_PROFILING_SERVICE_BACKEND"], "silicon");
    // Verify debug variable is not set (returns empty string)
    ASSERT_EQ(env_check["VPUNN_HTTP_CLIENT_DEBUG"], "");

    // Setup mock handler for /generate_workload endpoint
    _mock_server.Post("/generate_workload", [&](const httplib::Request& req, httplib::Response& res) {
        // Parse request
        nlohmann::json request = nlohmann::json::parse(req.body);
        if (HandleStatusCheck(request, res)) return;
        
        // Send a successful response
        nlohmann::json response;
        response["info"] = "success";
        response["latencies"] = std::vector<CyclesInterfaceType>{8888};
        res.set_content(response.dump(), "application/json");
        res.status = 200;
    });

    // Capture stdout via CoutRedirect to verify NO debug output is produced
    // Note: CoutRedirect is safe here because GTest runs all TEST_F(DPUHTTPClient, ...) sequentially
    std::ostringstream oss;
    {
        CoutRedirect redirect(oss);
        // Create provider using initFromEnvironment (debug should be disabled)
        auto provider = HttpCostProvider::initFromEnvironment();
        ASSERT_NE(provider, nullptr);
    
        // Make a request
        std::string info;
        CyclesInterfaceType cycles = provider->getCost(op, info);
        
        // Verify NO debug messages are present
        EXPECT_TRUE(oss.str().find("[DEBUG]") == std::string::npos) 
            << "Expected NO debug output but found some. Captured output:\n" << oss.str();
        
        // Verify the cost was retrieved successfully
        EXPECT_EQ(cycles, 8888);
    }
}

} // namespace VPUNN_unit_tests