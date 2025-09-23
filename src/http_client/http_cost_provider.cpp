#include "http_client/http_cost_provider.h"

nlohmann::json VPUNN::HTTPClient::sendJsonRequest(const nlohmann::json& request, const std::string& path) {
    try {
        auto res = _client.Post(path, request.dump(), "application/json");
        if (res) {
            return nlohmann::json::parse(res->body);
        } else {
            throw std::runtime_error("Failed to send request to server");
        }
    } catch (const nlohmann::json::parse_error& e) {
        throw std::runtime_error("Failed to parse JSON response: " + std::string(e.what()));
    } catch (const std::exception& e) {
        throw std::runtime_error("Exception in http client: " + std::string(e.what()));
    } catch (...) {
        throw std::runtime_error("Unknown exception in http client");
    }
}

bool VPUNN::HTTPProfilingClient::is_available(const std::string& check_backend) {
    nlohmann::json status_request_payload;
    
    status_request_payload["params"] = nlohmann::json::object();
    status_request_payload["params"]["status"] = true;
    status_request_payload["params"]["name"] = "profiling_request";
    
    auto res = sendJsonRequest(status_request_payload, "/generate_workload");

    if (res.contains("info")) {
        if (res["info"] == "status") {
            if (check_backend.empty()) {
                return true;
            } else {
                if (check_backend == "silicon") {
                    return res["profiling"].get<bool>();
                } else {
                    return true;
                }
            }
        }
    }

    return false;
}

VPUNN::ProfilerResponse VPUNN::HTTPProfilingClient::handle_profiler_response(const nlohmann::json& response) {
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
            }

            if (info_str == "generation_error") {
                profiler_response.success = false;
                profiler_response.message = response["msg"].get<std::string>();
                profiler_response.res_type = "generation_error";
            }

            if (info_str == "profiling_error") {
                profiler_response.success = false;
                profiler_response.message = response["msg"].get<std::string>();
                profiler_response.res_type = "profiling_error";
                if (response.contains("path")) {
                    profiler_response.path = response["path"].get<std::string>();
                }
            }

            if (info_str == "compilation_error") {
                profiler_response.success = false;
                profiler_response.message = response["msg"].get<std::string>();
                profiler_response.res_type = "compilation_error";
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
        }

        if (response.contains("error")) {
            profiler_response.success = false;

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

VPUNN::CyclesInterfaceType VPUNN::HttpDPUCostProvider::getCost(const DPUOperation& op, std::string& info, const std::string& backend) {
    nlohmann::json payload;
    
    payload["params"] = nlohmann::json::object();
    payload["params"]["backend"] = backend;
    payload["params"]["name"] = "profiling_request";
    payload["params"]["timeout"] = -1; // Need to wait for the profiling to finish

    payload["dpu_workload"] = nlohmann::json::object();
    payload["dpu_workload"] = dpuop_as_json(op);

    nlohmann::json response = _client.sendJsonRequest(payload, "/generate_workload");

    auto parsed_res = _client.handle_profiler_response(response);

    CyclesInterfaceType cycles = Cycles::ERROR_PROFILING_SERVICE;
    info = parsed_res.message;

    if (parsed_res.success) {
        if (parsed_res.cost.size() == 1) {
            cycles = parsed_res.cost[0];
        } else if (parsed_res.cost.size() > 1) {
            // If multiple latencies are returned, take the maximum
            cycles = *std::max_element(parsed_res.cost.begin(), parsed_res.cost.end());
        }
    }

    return cycles;
}

const nlohmann::json VPUNN::HttpDPUCostProvider::dpuop_as_json(const DPUOperation& op) {
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
    json_op["activation_function"] = "ActivationFunction." + mapToText<ActivationFunction>().at(static_cast<int>(op.activation_function));
    
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
    json_op["workload_uid"] = op.hash();

    return json_op;
}
