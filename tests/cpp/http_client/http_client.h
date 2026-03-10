// Copyright © 2026 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#include "http_client/http_cost_provider.h"
#include <httplib.h>
#include <nlohmann/json.hpp>

#include <gtest/gtest.h>
#include <thread>

namespace VPUNN_unit_tests {

using namespace VPUNN;

/**
 * @class HTTPClientTestBase
 * @brief Base test fixture for HTTPClient classes.
 *
 * Sets up a mock HTTP server to simulate various responses from the profiling service.
 * Each test configures the mock server handlers independently to ensure isolation and
 * comprehensive coverage of different scenarios.
 */
class HTTPClientTestBase : public ::testing::Test {
protected:
    void SetUp() override {
        // Start the mock server in a separate thread without any default handlers
        _mock_server_thread = std::thread([this]() {
            _mock_server.bind_to_port("localhost", srv_port);
            _mock_server.listen_after_bind();
        });

        // Wait for the server to start
        auto timeout = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while(!_mock_server.is_running()) {
            if (std::chrono::steady_clock::now() > timeout) {
                FAIL() << "Server failed to start";
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }

    void TearDown() override {
        // Prevent stopping the thread if it is still running
        if(_mock_server.is_running()) {
            _mock_server.stop();
        }
       
        if (_mock_server_thread.joinable()) {
            _mock_server_thread.join();
        }
    }

    httplib::Server _mock_server;
    std::thread _mock_server_thread;
    const int srv_port{1234};
};

/**
 * @class HTTPClientTest
 * @brief Template test fixture for HttpCostProvider classes with specific workload types.
 *
 * Inherits from HTTPClientTestBase and provides typed test methods for workload-specific testing.
 * This class contains a suite of tests for a generic workload type WlT, covering various scenarios,
 * each component that has a implementation in HttpCostProvider, should run these tests with it's specific
 * workload type.
 */
template <typename WlT>
class HTTPClientTest : public HTTPClientTestBase {
protected:
    /**
     * @brief Helper function to handle status check requests.
     * Returns true if this was a status request that was handled, false otherwise.
     * This is needed when multiple tests need to handle status checks, but some just need it as intermediate step
     */
    bool HandleStatusCheck(const nlohmann::json& request, httplib::Response& res) {
        if (request["params"].contains("status") && request["params"]["status"] == true) {
            nlohmann::json response;
            response["info"] = "status";
            response["profiling"] = "true";
            res.set_content(response.dump(), "application/json");
            res.status = 200;
            return true;
        }
        return false;
    }

    /**
     * @brief Tests HttpCostProvider's getCost method for a valid workload type.
     *
     * Verifies that getCost correctly retrieves the cost from the profiling service.
     */
    void GetCostSuccess(const WlT& wl) {
        // Setup mock handler for /generate_workload endpoint with a successful response
        _mock_server.Post("/generate_workload", [&](const httplib::Request& req, httplib::Response& res) {
            nlohmann::json request = nlohmann::json::parse(req.body);
            
            if (HandleStatusCheck(request, res)) return;
            
            nlohmann::json response;
            response["info"] = "success";
            response["latencies"] = std::vector<CyclesInterfaceType>{1234};
            res.set_content(response.dump(), "application/json");
            res.status = 200;
        });

        // Create HttpCostProvider instance
        HttpCostProvider cost_provider("localhost", srv_port);

        // Prepare info string
        std::string info;

        // Get cost
        CyclesInterfaceType cycles = cost_provider.getCost(wl, info);

        // Validate cost
        EXPECT_EQ(cycles, 1234);
    }

    /**
     * @brief Tests HttpCostProvider's getCost method when the profiling service returns an error.
     *
     * Simulates the profiling service returning an error and verifies that getCost handles it correctly.
     */
    void GetCostProfilingServiceError(const WlT& wl) {
        // Setup mock handler for /generate_workload endpoint returning a profiling error
        _mock_server.Post("/generate_workload", [&](const httplib::Request& req, httplib::Response& res) {
            nlohmann::json request = nlohmann::json::parse(req.body);
            
            if (HandleStatusCheck(request, res)) return;
            
            nlohmann::json response;
            response["info"] = "profiling_error";
            response["msg"] = "Profiling service failed";
            res.set_content(response.dump(), "application/json");
            res.status = 400;
        });

        // Create HttpCostProvider instance
        HttpCostProvider cost_provider("localhost", srv_port);

        // Prepare info string
        std::string info;

        // Get cost
        CyclesInterfaceType cycles = cost_provider.getCost(wl, info);

        // Validate cost and info
        EXPECT_TRUE(Cycles::isErrorCode(cycles));
    }

    /**
     * @brief Tests HttpCostProvider's getCost method when the profiling service returns malformed JSON.
     *
     * Verifies that getCost handles malformed JSON responses by setting appropriate error codes and messages.
     */
    void GetCostMalformedResponse(const WlT& wl) {
        // Setup mock handler for /generate_workload endpoint returning malformed JSON
        _mock_server.Post("/generate_workload", [&](const httplib::Request& req, httplib::Response& res) {
            nlohmann::json request = nlohmann::json::parse(req.body);
            
            if (HandleStatusCheck(request, res)) return;
            
            // Missing required fields
            nlohmann::json response;
            response["unexpected_field"] = "no_cycles";
            res.set_content(response.dump(), "application/json");
            res.status = 200;
        });

        // Create HttpCostProvider instance
        HttpCostProvider cost_provider("localhost", srv_port);

        // Prepare info string
        std::string info;

        // Get cost
        CyclesInterfaceType cycles = cost_provider.getCost(wl, info);

        // Validate cost and info
        EXPECT_TRUE(Cycles::isErrorCode(cycles));
    }

    /**
     * @brief Tests HttpCostProvider's getCost method when the profiling service returns completely invalid JSON.
     *
     * Verifies that getCost returns invalid value for cycles when JSON parsing fails.
     */
    void GetCostInvalidJsonResponse(const WlT& wl) {
        // Setup mock handler for /generate_workload endpoint returning invalid JSON
        _mock_server.Post("/generate_workload", [&](const httplib::Request& req, httplib::Response& res) {
            nlohmann::json request = nlohmann::json::parse(req.body);
            
            if (HandleStatusCheck(request, res)) return;
            
            res.set_content("Invalid JSON Response", "application/json");
            res.status = 200;
        });

        // Create HttpCostProvider instance
        HttpCostProvider cost_provider("localhost", srv_port);

        // Prepare info string
        std::string info;

        // Get cost and expect a error cycles due to JSON parsing failure
        EXPECT_EQ(cost_provider.getCost(wl, info), Cycles::ERROR_PROFILING_SERVICE);
        EXPECT_TRUE(info.find("Profiling service error") != std::string::npos);
    }

    /**
     * @brief Tests HttpCostProvider's getCost method with an invalid backend.
     *
     * Simulates the profiling service returning a profiling error when an invalid backend is specified.
     */
    void GetCostWithInvalidBackend(const WlT& wl) {
        // Setup mock handler for /generate_workload endpoint to handle invalid backend
        _mock_server.Post("/generate_workload", [&](const httplib::Request& req, httplib::Response& res) {
            nlohmann::json request = nlohmann::json::parse(req.body);
            
            if (HandleStatusCheck(request, res)) return;
            
            // getCost cannot set invalid backend directly, because ProfilingServiceBackend::__size gets mapped to SILICON
            // So to simulate invalid backend, server will return profiling_error with Invalid backend, no matter what backend is sent
            if (request.contains("params") && request["params"].contains("backend")) {
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

        // Create HttpCostProvider instance
        HttpCostProvider cost_provider("localhost", srv_port);

        // Prepare info string
        std::string info;

        // Get cost with invalid backend
        CyclesInterfaceType cycles = cost_provider.getCost(wl, info);

        // Validate cost and info
        EXPECT_TRUE(Cycles::isErrorCode(cycles));
        EXPECT_EQ(info, "Invalid backend");
    }

    /**
     * @brief Tests HttpCostProvider's getCost method with multiple latencies.
     *
     * Verifies that getCost correctly selects the maximum latency when multiple are provided.
     */
    void GetCostMultipleLatencies(const WlT& wl) {
        // Setup mock handler for /generate_workload endpoint with multiple latencies
        _mock_server.Post("/generate_workload", [&](const httplib::Request& req, httplib::Response& res) {
            nlohmann::json request = nlohmann::json::parse(req.body);
            
            if (HandleStatusCheck(request, res)) return;
            
            nlohmann::json response;
            response["info"] = "success";
            response["latencies"] = std::vector<CyclesInterfaceType>{1000, 2000, 1500};
            res.set_content(response.dump(), "application/json");
            res.status = 200;
        });

        // Create HttpCostProvider instance
        HttpCostProvider cost_provider("localhost", srv_port);

        // Prepare info string
        std::string info;

        // Get cost
        CyclesInterfaceType cycles = cost_provider.getCost(wl, info);

        // Validate that the maximum latency is selected
        EXPECT_EQ(cycles, 2000);
    }

    /**
     * @brief Tests HttpCostProvider's getCost method with empty latencies.
     *
     * Verifies that getCost handles an empty latencies array gracefully.
     */
    void GetCostEmptyLatencies(const WlT& wl) {
        // Setup mock handler for /generate_workload endpoint with empty latencies
        _mock_server.Post("/generate_workload", [&](const httplib::Request& req, httplib::Response& res) {
            nlohmann::json request = nlohmann::json::parse(req.body);
            
            if (HandleStatusCheck(request, res)) return;
            
            nlohmann::json response;
            response["info"] = "success";
            response["latencies"] = std::vector<CyclesInterfaceType>{};
            res.set_content(response.dump(), "application/json");
            res.status = 200;
        });

        // Create HttpCostProvider instance
        HttpCostProvider cost_provider("localhost", srv_port);

        // Prepare info string
        std::string info;

        // Get cost
        CyclesInterfaceType cycles = cost_provider.getCost(wl, info);

        // Validate cost and info
        EXPECT_TRUE(Cycles::isErrorCode(cycles));
    }

    /**
     * @brief Tests HttpCostProvider's getCost method when the profiling service returns a warning.
     *
     * Simulates the profiling service returning a warning and verifies that getCost handles it correctly.
     */
    void GetCostWithWarning(const WlT& wl) {
        // Setup mock handler for /generate_workload endpoint returning a warning
        _mock_server.Post("/generate_workload", [&](const httplib::Request& req, httplib::Response& res) {
            nlohmann::json request = nlohmann::json::parse(req.body);
            
            if (HandleStatusCheck(request, res)) return;
            
            nlohmann::json response;
            response["info"] = "success";
            response["latencies"] = std::vector<CyclesInterfaceType>{1234};
            response["warning"] = "Deprecated backend";
            response["msg"] = "Using deprecated backend.";
            response["path"] = "/deprecated_endpoint";
            res.set_content(response.dump(), "application/json");
            res.status = 200;
        });

        // Create HttpCostProvider instance
        HttpCostProvider cost_provider("localhost", srv_port);

        // Prepare info string
        std::string info;

        // Get cost
        CyclesInterfaceType cycles = cost_provider.getCost(wl, info);

        // Validate cost and info
        EXPECT_TRUE(Cycles::isErrorCode(cycles));
    }

    /**
     * @brief Tests HttpCostProvider's getCost method when the profiling service returns trace information.
     *
     * Verifies that getCost appends trace information to the error message when provided.
     */
    void GetCostWithTrace(const WlT& wl) {
        // Setup mock handler for /generate_workload endpoint returning an error with trace
        _mock_server.Post("/generate_workload", [&](const httplib::Request& req, httplib::Response& res) {
            nlohmann::json request = nlohmann::json::parse(req.body);
            
            if (HandleStatusCheck(request, res)) return;
            
            nlohmann::json response;
            response["error"] = {{"msg", "Execution failed"}, {"trace", "Traceback (most recent call last): ..."}};
            response["path"] = "/execute";
            res.set_content(response.dump(), "application/json");
            res.status = 400;
        });

        // Create HttpCostProvider instance
        HttpCostProvider cost_provider("localhost", srv_port);

        // Prepare info string
        std::string info;

        // Get cost
        CyclesInterfaceType cycles = cost_provider.getCost(wl, info);

        // Validate cost and info
        EXPECT_TRUE(Cycles::isErrorCode(cycles));
    }
};

}  // namespace VPUNN_unit_tests