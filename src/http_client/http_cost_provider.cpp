#include "http_client/http_cost_provider.h"
#include <iostream>

nlohmann::json VPUNN::HTTPClient::sendJsonRequest(const nlohmann::json& request, const std::string& path) {
    if (_debug) {
        std::cout << "[DEBUG] HTTPClient::sendJsonRequest - Sending request to path: " << path << std::endl;
        std::cout << "[DEBUG] HTTPClient::sendJsonRequest - Request payload: " << request.dump(2) << std::endl;
    }
    
    try {
        auto res = _client.Post(path, request.dump(), "application/json");
        if (res) {
            if (_debug) {
                std::cout << "[DEBUG] HTTPClient::sendJsonRequest - Response received, status: " << res->status << std::endl;
                std::cout << "[DEBUG] HTTPClient::sendJsonRequest - Response body: " << res->body << std::endl;
            }
            return nlohmann::json::parse(res->body);
        } else {
            if (_debug) {
                std::cout << "[DEBUG] HTTPClient::sendJsonRequest - Failed to receive response from server" << std::endl;
            }
            throw std::runtime_error("Failed to send request to server");
        }
    } catch (const nlohmann::json::parse_error& e) {
        if (_debug) {
            std::cout << "[DEBUG] HTTPClient::sendJsonRequest - JSON parse error: " << e.what() << std::endl;
        }
        throw std::runtime_error("Failed to parse JSON response: " + std::string(e.what()));
    } catch (const std::exception& e) {
        if (_debug) {
            std::cout << "[DEBUG] HTTPClient::sendJsonRequest - Exception: " << e.what() << std::endl;
        }
        throw std::runtime_error("Exception in http client: " + std::string(e.what()));
    } catch (...) {
        if (_debug) {
            std::cout << "[DEBUG] HTTPClient::sendJsonRequest - Unknown exception caught" << std::endl;
        }
        throw std::runtime_error("Unknown exception in http client");
    }
}

bool VPUNN::HTTPProfilingClient::is_available(const std::string& check_backend) {
    if (_debug) {
        std::cout << "[DEBUG] HTTPProfilingClient::is_available - Checking availability for backend: " 
                  << (check_backend.empty() ? "any" : check_backend) << std::endl;
    }
    
    nlohmann::json status_request_payload;

    status_request_payload["params"] = nlohmann::json::object();
    status_request_payload["params"]["status"] = true;
    status_request_payload["params"]["name"] = "profiling_request";

    auto res = sendJsonRequest(status_request_payload, "/generate_workload");

    if (res.contains("info")) {
        if (res["info"] == "status") {
            if (check_backend.empty()) {
                if (_debug) {
                    std::cout << "[DEBUG] HTTPProfilingClient::is_available - Service is available" << std::endl;
                }
                return true;
            } else {
                if (check_backend == "silicon") {
                    bool silicon_available = res["profiling"].get<std::string>() == "true";
                    if (_debug) {
                        std::cout << "[DEBUG] HTTPProfilingClient::is_available - Silicon backend available: " 
                                  << (silicon_available ? "true" : "false") << std::endl;
                    }
                    return silicon_available;
                } else {
                    if (_debug) {
                        std::cout << "[DEBUG] HTTPProfilingClient::is_available - Backend " << check_backend 
                                  << " is available" << std::endl;
                    }
                    return true;
                }
            }
        }
    }

    if (_debug) {
        std::cout << "[DEBUG] HTTPProfilingClient::is_available - Service is not available" << std::endl;
    }
    return false;
}

VPUNN::ProfilerResponse VPUNN::HTTPProfilingClient::handle_profiler_response(const nlohmann::json& response) {
    if (_debug) {
        std::cout << "[DEBUG] HTTPProfilingClient::handle_profiler_response - Processing response" << std::endl;
        std::cout << "[DEBUG] HTTPProfilingClient::handle_profiler_response - Response: " << response.dump(2) << std::endl;
    }
    
    ProfilerResponse profiler_response;
    try {
        if (response.contains("info")) {
            auto info_str = response["info"].get<std::string>();
            if (info_str == "success") {
                profiler_response.success = true;
                profiler_response.cost = response["latencies"].get<std::vector<CyclesInterfaceType>>();
                if (response.contains("path")) {
                    profiler_response.path = response["path"].get<std::string>();
                }
                profiler_response.res_type = "success";
                if (_debug) {
                    std::cout << "[DEBUG] HTTPProfilingClient::handle_profiler_response - Success response, cost size: " 
                              << profiler_response.cost.size() << std::endl;
                }
            }

            if (info_str == "generation_error") {
                profiler_response.success = false;
                profiler_response.message = response["msg"].get<std::string>();
                profiler_response.res_type = "generation_error";
                if (_debug) {
                    std::cout << "[DEBUG] HTTPProfilingClient::handle_profiler_response - Generation error: " 
                              << profiler_response.message << std::endl;
                }
            }

            if (info_str == "profiling_error") {
                profiler_response.success = false;
                profiler_response.message = response["msg"].get<std::string>();
                profiler_response.res_type = "profiling_error";
                if (_debug) {
                    std::cout << "[DEBUG] HTTPProfilingClient::handle_profiler_response - Profiling error: " 
                              << profiler_response.message << std::endl;
                }
                if (response.contains("path")) {
                    profiler_response.path = response["path"].get<std::string>();
                }
            }

            if (info_str == "compilation_error") {
                profiler_response.success = false;
                profiler_response.message = response["msg"].get<std::string>();
                profiler_response.res_type = "compilation_error";
                if (_debug) {
                    std::cout << "[DEBUG] HTTPProfilingClient::handle_profiler_response - Compilation error: " 
                              << profiler_response.message << std::endl;
                }
                if (response.contains("path")) {
                    profiler_response.path = response["path"].get<std::string>();
                }
            }
        }

        if (response.contains("warning")) {
            profiler_response.success = false;
            profiler_response.message = response["msg"].get<std::string>();
            profiler_response.res_type = response["warning"].get<std::string>();
            profiler_response.path = response["path"].get<std::string>();
            if (_debug) {
                std::cout << "[DEBUG] HTTPProfilingClient::handle_profiler_response - Warning: " 
                          << profiler_response.message << std::endl;
            }
        }

        if (response.contains("error")) {
            profiler_response.success = false;
            if (_debug) {
                std::cout << "[DEBUG] HTTPProfilingClient::handle_profiler_response - Error response detected" << std::endl;
            }

            if (response["error"].contains("msg")) {
                profiler_response.message = response["error"]["msg"].get<std::string>();
                profiler_response.res_type = "unknown";
            } else {
                profiler_response.res_type = response["error"].get<std::string>();
            }

            if (response.contains("path")) {
                profiler_response.path = response["path"].get<std::string>();
            }

            if (response.contains("msg")) {
                profiler_response.message = response["msg"].get<std::string>();
            }

            if (response.contains("trace")) {
                profiler_response.message += "\n" + response["trace"].get<std::string>();
            }
        }

    } catch (const nlohmann::json::exception& e) {
        throw std::runtime_error("Failed to parse JSON response: " + std::string(e.what()));
    }
    return profiler_response;
}

VPUNN::CyclesInterfaceType VPUNN::HttpDPUCostProvider::getCost(const DPUOperation& op, std::string& info,
                                                               const std::string& backend) {
    if (_debug) {
        std::cout << "[DEBUG] HttpDPUCostProvider::getCost - Getting cost for DPU operation" << std::endl;
        std::cout << "[DEBUG] HttpDPUCostProvider::getCost - Backend: " << backend << std::endl;
        std::cout << "[DEBUG] HttpDPUCostProvider::getCost - Workload UID: " << op.hash() << std::endl;
    }
    
    nlohmann::json payload;

    payload["params"] = nlohmann::json::object();
    payload["params"]["backend"] = backend;

    payload["params"]["name"] = "profiling_request";
    payload["params"]["timeout"] = -1;  // Need to wait for the profiling to finish

    payload["dpu_workload"] = nlohmann::json::object();
    payload["dpu_workload"] = dpuop_as_json(op);

    nlohmann::json response = _client.sendJsonRequest(payload, "/generate_workload");

    auto parsed_res = _client.handle_profiler_response(response);

    CyclesInterfaceType cycles = Cycles::ERROR_PROFILING_SERVICE;
    info = parsed_res.message;

    if (parsed_res.success) {
        if (parsed_res.cost.size() == 1) {
            cycles = parsed_res.cost[0];
            if (_debug) {
                std::cout << "[DEBUG] HttpDPUCostProvider::getCost - Single latency returned: " << cycles << std::endl;
            }
        } else if (parsed_res.cost.size() > 1) {
            // If multiple latencies are returned, take the maximum
            cycles = *std::max_element(parsed_res.cost.begin(), parsed_res.cost.end());
            if (_debug) {
                std::cout << "[DEBUG] HttpDPUCostProvider::getCost - Multiple latencies returned, max: " << cycles << std::endl;
            }
        }
    } else {
        if (_debug) {
            std::cout << "[DEBUG] HttpDPUCostProvider::getCost - Failed to get cost: " << info << std::endl;
        }
    }

    return cycles;
}

const nlohmann::json VPUNN::HttpDPUCostProvider::dpuop_as_json(const DPUOperation& op) {
    if (_debug) {
        std::cout << "[DEBUG] HttpDPUCostProvider::dpuop_as_json - Converting DPUOperation to JSON" << std::endl;
    }
    
    nlohmann::json json_op;

    // TODO: extend Serialization to support JSON format instead of this
    json_op["device"] = "VPUDevice." + mapToText<VPUDevice>().at(static_cast<int>(op.device));
    json_op["operation"] = "Operation." + mapToText<Operation>().at(static_cast<int>(op.operation));

    json_op["input_0_batch"] = op.input_0.batch;
    json_op["input_0_channels"] = op.input_0.channels;
    json_op["input_0_height"] = op.input_0.height;
    json_op["input_0_width"] = op.input_0.width;

    json_op["input_1_batch"] = op.input_1.batch;
    json_op["input_1_channels"] = op.input_1.channels;
    json_op["input_1_height"] = op.input_1.height;
    json_op["input_1_width"] = op.input_1.width;

    json_op["input_sparsity_enabled"] = static_cast<int>(op.input_0.sparsity_enabled);
    json_op["weight_sparsity_enabled"] = static_cast<int>(op.input_1.sparsity_enabled);
    json_op["input_sparsity_rate"] = op.input_0.sparsity;
    json_op["weight_sparsity_rate"] = op.input_1.sparsity;

    json_op["execution_order"] = "ExecutionMode." + mapToText<ExecutionMode>().at(static_cast<int>(op.execution_order));
    json_op["activation_function"] =
            "ActivationFunction." + mapToText<ActivationFunction>().at(static_cast<int>(op.activation_function));

    json_op["kernel_height"] = op.kernel.height;
    json_op["kernel_width"] = op.kernel.width;
    json_op["kernel_pad_bottom"] = op.kernel.pad_bottom;
    json_op["kernel_pad_top"] = op.kernel.pad_top;
    json_op["kernel_pad_left"] = op.kernel.pad_left;
    json_op["kernel_pad_right"] = op.kernel.pad_right;
    json_op["kernel_stride_height"] = op.kernel.stride_height;
    json_op["kernel_stride_width"] = op.kernel.stride_width;

    json_op["output_0_batch"] = op.output_0.batch;
    json_op["output_0_channels"] = op.output_0.channels;
    json_op["output_0_height"] = op.output_0.height;
    json_op["output_0_width"] = op.output_0.width;
    json_op["output_sparsity_enabled"] = static_cast<int>(op.output_0.sparsity_enabled);

    json_op["input_0_datatype"] = "DataType." + mapToText<DataType>().at(static_cast<int>(op.input_0.datatype));
    json_op["input_0_layout"] = "Layout." + mapToText<Layout>().at(static_cast<int>(op.input_0.layout));
    json_op["input_0_swizzling"] = "Swizzling." + mapToText<Swizzling>().at(static_cast<int>(op.input_0.swizzling));
    json_op["input_1_datatype"] = "DataType." + mapToText<DataType>().at(static_cast<int>(op.input_1.datatype));
    json_op["input_1_layout"] = "Layout." + mapToText<Layout>().at(static_cast<int>(op.input_1.layout));
    json_op["input_1_swizzling"] = "Swizzling." + mapToText<Swizzling>().at(static_cast<int>(op.input_1.swizzling));
    json_op["output_0_datatype"] = "DataType." + mapToText<DataType>().at(static_cast<int>(op.output_0.datatype));
    json_op["output_0_layout"] = "Layout." + mapToText<Layout>().at(static_cast<int>(op.output_0.layout));
    json_op["output_0_swizzling"] = "Swizzling." + mapToText<Swizzling>().at(static_cast<int>(op.output_0.swizzling));

    json_op["isi_strategy"] = "ISIStrategy." + mapToText<ISIStrategy>().at(static_cast<int>(op.isi_strategy));
    json_op["output_write_tiles"] = op.output_write_tiles;

    json_op["in_place_input1"] = static_cast<int>(op.weightless_operation);
    json_op["in_place_output"] = static_cast<int>(op.in_place_output_memory);
    json_op["superdense_output"] = static_cast<int>(op.superdense);
    json_op["input_autopad"] = static_cast<int>(op.input_autopad);
    json_op["output_autopad"] = static_cast<int>(op.output_autopad);
    json_op["workload_uid"] = op.hash();

    // Add all the halo info
    // input_0_halo fields (TBLRFB)
    json_op["input_0_halo_top"] = op.halo.input_0_halo.top;
    json_op["input_0_halo_bottom"] = op.halo.input_0_halo.bottom;
    json_op["input_0_halo_left"] = op.halo.input_0_halo.left;
    json_op["input_0_halo_right"] = op.halo.input_0_halo.right;
    json_op["input_0_halo_front"] = op.halo.input_0_halo.front;
    json_op["input_0_halo_back"] = op.halo.input_0_halo.back;

    // output_0_halo fields (TBLRFB)
    json_op["output_0_halo_top"] = op.halo.output_0_halo.top;
    json_op["output_0_halo_bottom"] = op.halo.output_0_halo.bottom;
    json_op["output_0_halo_left"] = op.halo.output_0_halo.left;
    json_op["output_0_halo_right"] = op.halo.output_0_halo.right;
    json_op["output_0_halo_front"] = op.halo.output_0_halo.front;
    json_op["output_0_halo_back"] = op.halo.output_0_halo.back;

    // output_0_halo_broadcast_cnt fields (TBLRFB)
    json_op["output_0_halo_broadcast_top"] = op.halo.output_0_halo_broadcast_cnt.top;
    json_op["output_0_halo_broadcast_bottom"] = op.halo.output_0_halo_broadcast_cnt.bottom;
    json_op["output_0_halo_broadcast_left"] = op.halo.output_0_halo_broadcast_cnt.left;
    json_op["output_0_halo_broadcast_right"] = op.halo.output_0_halo_broadcast_cnt.right;
    json_op["output_0_halo_broadcast_front"] = op.halo.output_0_halo_broadcast_cnt.front;
    json_op["output_0_halo_broadcast_back"] = op.halo.output_0_halo_broadcast_cnt.back;

    // output_0_inbound_halo fields (TBLRFB)
    json_op["output_0_halo_inbound_top"] = op.halo.output_0_inbound_halo.top;
    json_op["output_0_halo_inbound_bottom"] = op.halo.output_0_inbound_halo.bottom;
    json_op["output_0_halo_inbound_left"] = op.halo.output_0_inbound_halo.left;
    json_op["output_0_halo_inbound_right"] = op.halo.output_0_inbound_halo.right;
    json_op["output_0_halo_inbound_front"] = op.halo.output_0_inbound_halo.front;
    json_op["output_0_halo_inbound_back"] = op.halo.output_0_inbound_halo.back;

    // aspect to do: why op.mpe_engine is not here? is it condensed by execution_order?

    // reduce_minmax_op
    json_op["minmax"] = static_cast<int>(op.reduce_minmax_op);
    json_op["wcb_mode"] = static_cast<int>(0);

    return json_op;
}
