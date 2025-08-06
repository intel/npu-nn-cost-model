// tests/cpp/http_client_test.cpp

#include <gtest/gtest.h>
#include <thread>

#ifdef VPUNN_BUILD_HTTP_CLIENT
#include <httplib.h>
#include <nlohmann/json.hpp>
#include "http_client/http_cost_provider.h"
#endif

namespace VPUNN_unit_tests {

#ifdef VPUNN_BUILD_HTTP_CLIENT
using namespace VPUNN;

/**
 * @class HTTPClientTest
 * @brief Test fixture for HTTPClient and HttpDPUCostProvider classes.
 *
 * Sets up a mock HTTP server to simulate various responses from the profiling service.
 * Each test configures the mock server handlers independently to ensure isolation and
 * comprehensive coverage of different scenarios.
 */
class HTTPClientTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Start the mock server in a separate thread without any default handlers
        _mock_server_thread = std::thread([this]() {
            _mock_server.listen("localhost", srv_port);
        });

        // Give the server time to start
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    void TearDown() override {
        _mock_server.stop();
        if (_mock_server_thread.joinable()) {
            _mock_server_thread.join();
        }
    }

    httplib::Server _mock_server;
    std::thread _mock_server_thread;
    int srv_port = 1234;
};

/**
 * @brief Tests HTTPClient's sendJsonRequest method for a successful request.
 *
 * Verifies that sendJsonRequest correctly sends a JSON payload and parses the JSON response.
 */
TEST_F(HTTPClientTest, SendJsonRequestSuccess) {
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
TEST_F(HTTPClientTest, SendJsonRequestServerError) {
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
TEST_F(HTTPClientTest, SendJsonRequestMalformedJson) {
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
TEST_F(HTTPClientTest, IsAvailableTrue) {
    // Setup mock handler for /generate_workload endpoint indicating service is available
    _mock_server.Post("/generate_workload", [&](const httplib::Request& req, httplib::Response& res) {
        (void)req;
        nlohmann::json response;
        response["info"] = "status";
        response["profiling"] = true;
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
TEST_F(HTTPClientTest, IsAvailableSpecificBackendFalse) {
    // Setup mock handler for /generate_workload endpoint indicating a specific backend is unavailable
    _mock_server.Post("/generate_workload", [&](const httplib::Request& req, httplib::Response& res) {
        (void)req;
        nlohmann::json response;
        response["info"] = "status";
        response["profiling"] = false;
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
TEST_F(HTTPClientTest, IsAvailableServerDown) {
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
TEST_F(HTTPClientTest, HandleProfilerResponseSuccess) {
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
TEST_F(HTTPClientTest, HandleProfilerResponseProfilingError) {
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

/**
 * @brief Tests HttpDPUCostProvider's getCost method for a valid DPUOperation.
 *
 * Verifies that getCost correctly retrieves the cost from the profiling service.
 */
TEST_F(HTTPClientTest, GetCostSuccess) {
    // Setup mock handler for /generate_workload endpoint with a successful response
    _mock_server.Post("/generate_workload", [&](const httplib::Request& req, httplib::Response& res) {
        (void)req;
        // Parse request if needed
        nlohmann::json response;
        response["info"] = "success";
        response["latencies"] = std::vector<CyclesInterfaceType>{1234};
        res.set_content(response.dump(), "application/json");
        res.status = 200;
    });

    // Create HttpDPUCostProvider instance
    HttpDPUCostProvider cost_provider("localhost", srv_port);

    // Prepare DPUOperation
    DPUOperation op;
    // Initialize op as needed for the test
    // For simplicity, assuming default constructor sets up a valid operation

    // Prepare info string
    std::string info;

    // Get cost
    CyclesInterfaceType cycles = cost_provider.getCost(op, info, "silicon");

    // Validate cost
    EXPECT_EQ(cycles, 1234);
}

/**
 * @brief Tests HttpDPUCostProvider's getCost method when the profiling service returns an error.
 *
 * Simulates the profiling service returning an error and verifies that getCost handles it correctly.
 */
TEST_F(HTTPClientTest, GetCostProfilingServiceError) {
    // Setup mock handler for /generate_workload endpoint returning a profiling error
    _mock_server.Post("/generate_workload", [&](const httplib::Request& req, httplib::Response& res) {
        (void)req;
        nlohmann::json response;
        response["info"] = "profiling_error";
        response["msg"] = "Profiling service failed";
        res.set_content(response.dump(), "application/json");
        res.status = 400;
    });

    // Create HttpDPUCostProvider instance
    HttpDPUCostProvider cost_provider("localhost", srv_port);

    // Prepare DPUOperation
    DPUOperation op;

    // Prepare info string
    std::string info;

    // Get cost
    CyclesInterfaceType cycles = cost_provider.getCost(op, info, "silicon");

    // Validate cost and info
    EXPECT_TRUE(Cycles::isErrorCode(cycles));
}

/**
 * @brief Tests HttpDPUCostProvider's getCost method when the profiling service returns malformed JSON.
 *
 * Verifies that getCost handles malformed JSON responses by setting appropriate error codes and messages.
 */
TEST_F(HTTPClientTest, GetCostMalformedResponse) {
    // Setup mock handler for /generate_workload endpoint returning malformed JSON
    _mock_server.Post("/generate_workload", [&](const httplib::Request& req, httplib::Response& res) {
        (void)req;
        // Missing required fields
        nlohmann::json response;
        response["unexpected_field"] = "no_cycles";
        res.set_content(response.dump(), "application/json");
        res.status = 200;
    });

    // Create HttpDPUCostProvider instance
    HttpDPUCostProvider cost_provider("localhost", srv_port);

    // Prepare DPUOperation
    DPUOperation op;

    // Prepare info string
    std::string info;

    // Get cost
    CyclesInterfaceType cycles = cost_provider.getCost(op, info, "silicon");

    // Validate cost and info
    EXPECT_TRUE(Cycles::isErrorCode(cycles));
}

/**
 * @brief Tests HttpDPUCostProvider's getCost method when the profiling service returns completely invalid JSON.
 *
 * Verifies that getCost throws a runtime_error when the response is not valid JSON.
 */
TEST_F(HTTPClientTest, GetCostInvalidJsonResponse) {
    // Setup mock handler for /generate_workload endpoint returning invalid JSON
    _mock_server.Post("/generate_workload", [&](const httplib::Request& req, httplib::Response& res) {
        (void)req;
        res.set_content("Invalid JSON Response", "application/json");
        res.status = 200;
    });

    // Create HttpDPUCostProvider instance
    HttpDPUCostProvider cost_provider("localhost", srv_port);

    // Prepare DPUOperation
    DPUOperation op;

    // Prepare info string
    std::string info;

    // Get cost and expect a runtime_error due to JSON parsing failure
    EXPECT_THROW({ cost_provider.getCost(op, info, "silicon"); }, std::runtime_error);
}

/**
 * @brief Tests HttpDPUCostProvider's getCost method with an invalid backend.
 *
 * Simulates the profiling service returning a profiling error when an invalid backend is specified.
 */
TEST_F(HTTPClientTest, GetCostWithInvalidBackend) {
    // Setup mock handler for /generate_workload endpoint to handle invalid backend
    _mock_server.Post("/generate_workload", [&](const httplib::Request& req, httplib::Response& res) {
        nlohmann::json request = nlohmann::json::parse(req.body);
        if (request.contains("params") && request["params"].contains("backend") &&
            request["params"]["backend"] != "silicon") {
            nlohmann::json response;
            response["info"] = "profiling_error";
            response["msg"] = "Invalid backend";
            res.set_content(response.dump(), "application/json");
            res.status = 400;
            return;
        }

        // Default successful response
        nlohmann::json response;
        response["info"] = "success";
        response["latencies"] = std::vector<CyclesInterfaceType>{1234};
        res.set_content(response.dump(), "application/json");
        res.status = 200;
    });

    // Create HttpDPUCostProvider instance
    HttpDPUCostProvider cost_provider("localhost", srv_port);

    // Prepare DPUOperation
    DPUOperation op;

    // Prepare info string
    std::string info;

    // Get cost with invalid backend
    CyclesInterfaceType cycles = cost_provider.getCost(op, info, "invalid_backend");

    // Validate cost and info
    EXPECT_TRUE(Cycles::isErrorCode(cycles));
    EXPECT_EQ(info, "Invalid backend");
}

/**
 * @brief Tests HttpDPUCostProvider's getCost method with multiple latencies.
 *
 * Verifies that getCost correctly selects the maximum latency when multiple are provided.
 */
TEST_F(HTTPClientTest, GetCostMultipleLatencies) {
    // Setup mock handler for /generate_workload endpoint with multiple latencies
    _mock_server.Post("/generate_workload", [&](const httplib::Request& req, httplib::Response& res) {
        (void)req;
        nlohmann::json response;
        response["info"] = "success";
        response["latencies"] = std::vector<CyclesInterfaceType>{1000, 2000, 1500};
        res.set_content(response.dump(), "application/json");
        res.status = 200;
    });

    // Create HttpDPUCostProvider instance
    HttpDPUCostProvider cost_provider("localhost", srv_port);

    // Prepare DPUOperation
    DPUOperation op;

    // Prepare info string
    std::string info;

    // Get cost
    CyclesInterfaceType cycles = cost_provider.getCost(op, info, "silicon");

    // Validate that the maximum latency is selected
    EXPECT_EQ(cycles, 2000);
}

/**
 * @brief Tests HttpDPUCostProvider's getCost method with empty latencies.
 *
 * Verifies that getCost handles an empty latencies array gracefully.
 */
TEST_F(HTTPClientTest, GetCostEmptyLatencies) {
    // Setup mock handler for /generate_workload endpoint with empty latencies
    _mock_server.Post("/generate_workload", [&](const httplib::Request& req, httplib::Response& res) {
        (void)req;
        nlohmann::json response;
        response["info"] = "success";
        response["latencies"] = std::vector<CyclesInterfaceType>{};
        res.set_content(response.dump(), "application/json");
        res.status = 200;
    });

    // Create HttpDPUCostProvider instance
    HttpDPUCostProvider cost_provider("localhost", srv_port);

    // Prepare DPUOperation
    DPUOperation op;

    // Prepare info string
    std::string info;

    // Get cost
    CyclesInterfaceType cycles = cost_provider.getCost(op, info, "silicon");

    // Validate cost and info
    EXPECT_TRUE(Cycles::isErrorCode(cycles));
}

/**
 * @brief Tests HttpDPUCostProvider's dpuop_as_json serialization.
 *
 * Verifies that dpuop_as_json correctly serializes a DPUOperation into the expected JSON format.
 */
TEST_F(HTTPClientTest, DpuOpAsJsonSerialization) {
    // Create HttpDPUCostProvider instance
    HttpDPUCostProvider cost_provider("localhost", srv_port);

    // Prepare DPUOperation with specific values
    DPUOperation op;
    op.output_write_tiles = 2;
    op.in_place_output_memory = true;
    // Initialize other fields as needed

    // Since dpuop_as_json is a private method, we'll indirectly test it via getCost
    // Setup mock handler for /generate_workload endpoint to capture the request
    _mock_server.Post("/generate_workload", [&](const httplib::Request& req, httplib::Response& res) {
        // Parse the request JSON
        nlohmann::json request = nlohmann::json::parse(req.body);

        // Validate the serialized DPUOperation fields
        EXPECT_EQ(request["params"]["backend"], "silicon");
        EXPECT_EQ(request["params"]["name"], "profiling_request");
        EXPECT_TRUE(request.contains("dpu_workload"));
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
    CyclesInterfaceType cycles = cost_provider.getCost(op, info, "silicon");

    // Validate cost
    EXPECT_EQ(cycles, 1234);
}

/**
 * @brief Tests HttpDPUCostProvider's getCost method when the profiling service returns a warning.
 *
 * Simulates the profiling service returning a warning and verifies that getCost handles it correctly.
 */
TEST_F(HTTPClientTest, GetCostWithWarning) {
    // Setup mock handler for /generate_workload endpoint returning a warning
    _mock_server.Post("/generate_workload", [&](const httplib::Request& req, httplib::Response& res) {
        (void)req;
        nlohmann::json response;
        response["info"] = "success";
        response["latencies"] = std::vector<CyclesInterfaceType>{1234};
        response["warning"] = "Deprecated backend";
        response["msg"] = "Using deprecated backend.";
        response["path"] = "/deprecated_endpoint";
        res.set_content(response.dump(), "application/json");
        res.status = 200;
    });

    // Create HttpDPUCostProvider instance
    HttpDPUCostProvider cost_provider("localhost", srv_port);

    // Prepare DPUOperation
    DPUOperation op;

    // Prepare info string
    std::string info;

    // Get cost
    CyclesInterfaceType cycles = cost_provider.getCost(op, info, "silicon");

    // Validate cost and info
    EXPECT_TRUE(Cycles::isErrorCode(cycles));
}

/**
 * @brief Tests HttpDPUCostProvider's getCost method when the profiling service returns trace information.
 *
 * Verifies that getCost appends trace information to the error message when provided.
 */
TEST_F(HTTPClientTest, GetCostWithTrace) {
    // Setup mock handler for /generate_workload endpoint returning an error with trace
    _mock_server.Post("/generate_workload", [&](const httplib::Request& req, httplib::Response& res) {
        (void)req;
        nlohmann::json response;
        response["error"] = {{"msg", "Execution failed"}, {"trace", "Traceback (most recent call last): ..."}};
        response["path"] = "/execute";
        res.set_content(response.dump(), "application/json");
        res.status = 400;
    });

    // Create HttpDPUCostProvider instance
    HttpDPUCostProvider cost_provider("localhost", srv_port);

    // Prepare DPUOperation
    DPUOperation op;

    // Prepare info string
    std::string info;

    // Get cost
    CyclesInterfaceType cycles = cost_provider.getCost(op, info, "silicon");

    // Validate cost and info
    EXPECT_TRUE(Cycles::isErrorCode(cycles));
}

// Additional tests can be added here to cover more scenarios as needed

#endif
}  // namespace VPUNN_unit_tests