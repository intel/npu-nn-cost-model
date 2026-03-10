// Copyright © 2026 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#include "http_client.h"

namespace VPUNN_unit_tests {
using namespace VPUNN;
// Tests for HTTPClientTestBase fixture
/**
 * @brief Tests HTTPClient's sendJsonRequest method for a successful request.
 *
 * Verifies that sendJsonRequest correctly sends a JSON payload and parses the JSON response.
 */
TEST_F(HTTPClientTestBase, SendJsonRequestSuccess) {
    // Setup mock handler for /test_success endpoint
    _mock_server.Post("/test_success", [&](const httplib::Request& req, httplib::Response& res) {
        // Parse the incoming request
        nlohmann::json request = nlohmann::json::parse(req.body);
        // Create a successful response
        nlohmann::json response;
        response["status"] = "ok";
        response["data"] = 42;
        res.set_content(response.dump(), "application/json");
        res.status = 200;
    });

    // Create HTTPClient instance
    HTTPClient client("localhost", srv_port);

    // Prepare payload
    nlohmann::json payload;
    payload["query"] = "test";

    // Send request
    nlohmann::json response = client.sendJsonRequest(payload, "/test_success");

    // Validate response
    EXPECT_EQ(response["status"], "ok");
    EXPECT_EQ(response["data"], 42);
}

/**
 * @brief Tests HTTPClient's sendJsonRequest method when the server returns a non-200 status.
 *
 * Verifies that sendJsonRequest throws a runtime_error when the server responds with an error status.
 */
TEST_F(HTTPClientTestBase, SendJsonRequestServerError) {
    // Setup mock handler for /test_error endpoint
    _mock_server.Post("/test_error", [&](const httplib::Request& req, httplib::Response& res) {
        (void)req;
        res.set_content("Internal Server Error", "text/plain");
        res.status = 500;
    });

    // Create HTTPClient instance
    HTTPClient client("localhost", srv_port);

    // Prepare payload
    nlohmann::json payload;
    payload["action"] = "fail";

    // Send request and expect an exception
    EXPECT_THROW({ client.sendJsonRequest(payload, "/test_error"); }, std::runtime_error);
}

/**
 * @brief Tests HTTPClient's sendJsonRequest method when the server returns malformed JSON.
 *
 * Verifies that sendJsonRequest throws a runtime_error when the response body is not valid JSON.
 */
TEST_F(HTTPClientTestBase, SendJsonRequestMalformedJson) {
    // Setup mock handler for /test_malformed_json endpoint
    _mock_server.Post("/test_malformed_json", [&](const httplib::Request& req, httplib::Response& res) {
        (void)req;
        res.set_content("Not a JSON string", "application/json");
        res.status = 200;
    });

    // Create HTTPClient instance
    HTTPClient client("localhost", srv_port);

    // Prepare payload
    nlohmann::json payload;
    payload["data"] = "test";

    // Send request and expect an exception
    EXPECT_THROW({ client.sendJsonRequest(payload, "/test_malformed_json"); }, std::runtime_error);
}

/**
 * @brief Tests HTTPProfilingClient's is_available method when the profiling service is available.
 *
 * Simulates a successful response from the /generate_workload endpoint indicating availability.
 */
TEST_F(HTTPClientTestBase, IsAvailableTrue) {
    // Setup mock handler for /generate_workload endpoint indicating service is available
    _mock_server.Post("/generate_workload", [&](const httplib::Request& req, httplib::Response& res) {
        (void)req;
        nlohmann::json response;
        response["info"] = "status";
        response["profiling"] = "true";
        res.set_content(response.dump(), "application/json");
        res.status = 200;
    });

    // Create HTTPProfilingClient instance
    HTTPProfilingClient profiling_client("localhost", srv_port);

    // Check availability
    EXPECT_TRUE(profiling_client.is_available("silicon"));
}

/**
 * @brief Tests HTTPProfilingClient's is_available method when a specific backend is unavailable.
 *
 * Simulates the /generate_workload endpoint responding that the specified backend is unavailable.
 */
TEST_F(HTTPClientTestBase, IsAvailableSpecificBackendFalse) {
    // Setup mock handler for /generate_workload endpoint indicating a specific backend is unavailable
    _mock_server.Post("/generate_workload", [&](const httplib::Request& req, httplib::Response& res) {
        (void)req;
        nlohmann::json response;
        response["info"] = "status";
        response["profiling"] = "false";
        res.set_content(response.dump(), "application/json");
        res.status = 200;
    });

    // Create HTTPProfilingClient instance
    HTTPProfilingClient profiling_client("localhost", srv_port);

    // Check availability for "silicon"
    EXPECT_FALSE(profiling_client.is_available("silicon"));
}

/**
 * @brief Tests HTTPProfilingClient's is_available method when the server is down.
 *
 * Verifies that is_available returns false when the profiling service is unreachable.
 */
TEST_F(HTTPClientTestBase, IsAvailableServerDown) {
    // No handler setup to simulate server being down

    // Stop the server to simulate it being down
    _mock_server.stop();
    if (_mock_server_thread.joinable()) {
        _mock_server_thread.join();
    }

    // Create HTTPProfilingClient instance
    HTTPProfilingClient profiling_client("localhost", srv_port);

    // Check availability
    EXPECT_THROW({ profiling_client.is_available("silicon"); }, std::runtime_error);
}

/**
 * @brief Tests HTTPProfilingClient's handle_profiler_response with a successful response.
 *
 * Ensures that handle_profiler_response correctly parses a successful profiling response.
 */
TEST_F(HTTPClientTestBase, HandleProfilerResponseSuccess) {
    // Create HTTPProfilingClient instance
    HTTPProfilingClient profiling_client("localhost", srv_port);

    // Prepare a mock successful response
    nlohmann::json mock_response;
    mock_response["info"] = "success";
    mock_response["latencies"] = std::vector<CyclesInterfaceType>{1234};

    // Invoke handle_profiler_response
    ProfilerResponse response = profiling_client.handle_profiler_response(mock_response);

    // Validate the response
    EXPECT_TRUE(response.success);
    ASSERT_EQ(response.cost.size(), 1);
    EXPECT_EQ(response.cost[0], 1234);
    EXPECT_EQ(response.res_type, "success");
}

/**
 * @brief Tests HTTPProfilingClient's handle_profiler_response with a profiling error.
 *
 * Ensures that handle_profiler_response correctly parses a profiling error response.
 */
TEST_F(HTTPClientTestBase, HandleProfilerResponseProfilingError) {
    // Create HTTPProfilingClient instance
    HTTPProfilingClient profiling_client("localhost", srv_port);

    // Prepare a mock profiling error response
    nlohmann::json mock_response;
    mock_response["info"] = "profiling_error";
    mock_response["msg"] = "Invalid backend specified";
    mock_response["path"] = "/generate_workload";

    // Invoke handle_profiler_response
    ProfilerResponse response = profiling_client.handle_profiler_response(mock_response);

    // Validate the response
    EXPECT_FALSE(response.success);
    EXPECT_EQ(response.message, "Invalid backend specified");
    EXPECT_EQ(response.res_type, "profiling_error");
    EXPECT_EQ(response.path, "/generate_workload");
}

} // namespace VPUNN_unit_tests
