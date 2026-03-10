# Online Profiling Procedure

This document describes the procedure for using online profiling with the VPUNN HTTP cost provider to collect real hardware performance data during network compilation.

## Overview

Online profiling allows you to collect actual hardware performance measurements (cycles) from a running profiling service during model compilation. These measurements can then be used to update the VPUNN cost model cache files.

## Prerequisites

Before starting, verify that the profiling service is online on the target board:
- Check the service status on the designated host (e.g., IR-NPU_RESERVED_1-S-053-C.ger.corp.intel.com) - there should be a CMD terminal stating that it waits for connections.
- A list of RVPs assigned to cost model is available [here](https://intel.sharepoint.com/sites/MVDVPU4/_layouts/Doc.aspx?sourcedoc={81C45915-75FD-4E89-8D50-8728B441DE64}&wd=target%28Miscellaneous.one%7C74EB186C-CCA8-4A81-89A5-1302B0147007%2F%29&wdsectionfileid={D7E152D1-3638-456E-BE7A-CC7932A78DC9}).

## Step 1: Compile VPUX with HTTP Client Support

To enable online profiling, you need to build VPUX with VPUNN HTTP client support.

### Required CMake Configuration

Add the following flag when generating the CMake configuration:

```bash
-DVPUNN_BUILD_HTTP_CLIENT=ON
```

### Troubleshooting

If the above option alone doesn't work (eg. none of the costs are coming from profiling_service / HTTPCostProvider is not being used), you may need to explicitly add the compile definition. Add the following line near the top of the VPUX root `CMakeLists.txt`:

```cmake
add_compile_definitions(VPUNN_BUILD_HTTP_CLIENT)
```

## Step 2: Configure Environment Variables

Before compiling a network, set the following environment variables to enable and configure the HTTP profiling service:

```bash
export VPUNN_PROFILING_SERVICE_PORT=5000
export VPUNN_PROFILING_SERVICE_EXPERIMENT_NAME=real_world_db
export VPUNN_PROFILING_SERVICE_BACKEND=silicon
export ENABLE_VPUNN_PROFILING_SERVICE=TRUE
export VPUNN_PROFILING_SERVICE_HOST=IR-NPU_RESERVED_1-S-053-C.ger.corp.intel.com
export ENABLE_VPUNN_DATA_SERIALIZATION=TRUE
```

### Environment Variable Descriptions

#### Required Variables

- **`ENABLE_VPUNN_PROFILING_SERVICE`**: Master switch to enable/disable the profiling service (TRUE/FALSE)
  - Set to `TRUE` to use online profiling
  - Set to `FALSE` or leave unset to use only the offline cost model

- **`VPUNN_PROFILING_SERVICE_HOST`**: Hostname or IP address of the profiling service
  - Example: `IR-NPU_RESERVED_1-S-053-C.ger.corp.intel.com`
  - **Important**: Always verify that the profiling service is online on the specified host before setting this variable
  - Test connectivity: `ping <hostname>` or `curl http://<hostname>:<port>/`

- **`VPUNN_PROFILING_SERVICE_PORT`**: The port where the profiling service is running
  - Default: `5000`
  - Ensure the port is accessible and not blocked by firewall rules

- **`VPUNN_PROFILING_SERVICE_BACKEND`**: Hardware backend type for profiling
  - Common values: `silicon`, `simulation`, `emulator`
  - `silicon` is typically used for real hardware measurements

- **`VPUNN_PROFILING_SERVICE_EXPERIMENT_NAME`**: Name identifier for the profiling experiment
  - Example: `real_world_db`, `model_optimization`, `perf_analysis_jan2026`
  - Helps organize and track different profiling sessions
  - Used for logging and data organization on the profiling service

- **`ENABLE_VPUNN_DATA_SERIALIZATION`**: Enable serialization of profiling data to CSV files
  - Set to `TRUE` to generate CSV output files for each compiled model
  - Set to `FALSE` if you only need the costs without CSV export
  - Required if you plan to update cache files with profiled data

#### Optional Variables

- **`VPUNN_HTTP_CLIENT_DEBUG`**: Enable detailed debug logging for HTTP communication
  - Set to `1`, `true`, `TRUE`, or `True` to enable debug output
  - When enabled, prints detailed information about:
    - HTTP requests and responses
    - JSON payload contents
    - Connection status
    - Error details and stack traces
  - Useful for troubleshooting connectivity or profiling service issues
  - Example: `export VPUNN_HTTP_CLIENT_DEBUG=1`

### Configuration Examples

#### Minimal Configuration (Production)

```bash
export ENABLE_VPUNN_PROFILING_SERVICE=TRUE
export VPUNN_PROFILING_SERVICE_HOST=IR-NPU_RESERVED_1-S-053-C.ger.corp.intel.com
export VPUNN_PROFILING_SERVICE_PORT=5000
export VPUNN_PROFILING_SERVICE_BACKEND=silicon
export VPUNN_PROFILING_SERVICE_EXPERIMENT_NAME=production_profiling
export ENABLE_VPUNN_DATA_SERIALIZATION=TRUE
```

#### Debug Configuration (Troubleshooting)

```bash
export ENABLE_VPUNN_PROFILING_SERVICE=TRUE
export VPUNN_PROFILING_SERVICE_HOST=IR-NPU_RESERVED_1-S-053-C.ger.corp.intel.com
export VPUNN_PROFILING_SERVICE_PORT=5000
export VPUNN_PROFILING_SERVICE_BACKEND=silicon
export VPUNN_PROFILING_SERVICE_EXPERIMENT_NAME=debug_session
export ENABLE_VPUNN_DATA_SERIALIZATION=TRUE
export VPUNN_HTTP_CLIENT_DEBUG=1  # Enable detailed HTTP debug logging
```

#### Disable Online Profiling

```bash
export ENABLE_VPUNN_PROFILING_SERVICE=FALSE
# or simply unset the variable
unset ENABLE_VPUNN_PROFILING_SERVICE
```

## Step 3: Compile Networks and Collect Data

With the environment variables set, compile your models as usual. If the service is working correctly, you will get a CSV file for each model compiled.

### CSV Output Format

Each generated CSV file contains profiling data with a `cost_source` column that indicates the origin of the cost estimate:

- **`nn_sim_runs`**: Cost was provided by the neural network cost model
- **`fixed_cache`**: Cost was provided by the existing cache
- **`profiling_service_silicon`**: Cost was provided by the online profiling service (newly profiled costs)
- **`dyn_cache`**: Cost cache of the neural network cost model.

The actual cycle count is stored in the `vpunn_cycles` column.

## Step 4: Update Cache Files

After collecting profiling data, you can update the cache binary files with the newly profiled costs.

### Filter Profiled Rows

Extract only the rows where `cost_source` is `profiling_service_silicon` from your CSV files. These represent the new hardware measurements that should be added to the cache.

### Using cache_app

The `cache_app` utility is used to update `.cachebin` files with the new profiling data.

#### Basic Usage

```bash
cache_app \
    --csv <path_to_csv_file> \
    --cache <path_to_output_cachebin> \
    --type DPU \
    --intf <interface_version>
```

#### Parameters

- `-c, --csv`: Path to the CSV file containing filtered profiling data (only `profiling_service_silicon` rows). Can specify multiple files with: `-c <path> -c <other_path> ...`
- `-a, --cache`: Path to resulting cache file. If already exists, new data will be appended!
- `-t, --type`: Type of cache to create [DPU, DMA, SHAVE]
- `-l, --cycles_tag`: Tag for cycles field in CSV (e.g., "vpunn_cycles")
- `-i, --intf`: DPU interface version. Available options: **4011**, **4111**, **5112**, **5113**, **5114**
- `-d, --dma_type`: DMA workload type [27, 40_50] (only for DMA cache type)
- `-f, --filter`: Filters for CSV parsing. Format: `<key>=<value>`
- `-o, --use_new_costs`: In case of cache hit, use new costs

#### Interface Version Reference

- **4011**: NPU 4.0 interface 1.1
- **4111**: NPU 4.1 interface 1.1
- **5112**: NPU 5.1 interface 1.2
- **5113**: NPU 5.1 interface 1.3
- **5114**: NPU 5.1 interface 1.4

**Important**: For NPU 6.0 and later, do **NOT** set the `--intf` parameter.

### Example Workflow

1. Filter the CSV to get only profiled entries:
   ```bash
   # Example using awk to filter CSV
   awk -F',' '$cost_source_column == "profiling_service_silicon"' input.csv > filtered_profiled.csv
   ```

2. Create/update DPU cache for NPU 4.0 (interface 4011):
   ```bash
   cache_app \
       --csv filtered_profiled.csv \
       --cache updated_cache.cachebin \
       --type DPU \
       --intf 4011 \
       --cycles_tag vpunn_cycles
   ```

3. Create/update DPU cache for NPU 5.1 (interface 5113):
   ```bash
   cache_app \
       --csv filtered_profiled.csv \
       --cache updated_cache.cachebin \
       --type DPU \
       --intf 5113 \
       --cycles_tag vpunn_cycles
   ```

4. Create/update cache for NPU 6.0+ (without interface):
   ```bash
   cache_app \
       --csv filtered_profiled.csv \
       --cache updated_cache.cachebin \
       --type DPU \
       --cycles_tag vpunn_cycles
   ```

5. Create/update cache with multiple CSV files:
   ```bash
   cache_app \
       --csv file1.csv \
       --csv file2.csv \
       --csv file3.csv \
       --cache updated_cache.cachebin \
       --type DPU \
       --intf 5113
   ```

6. Update existing cache with new costs (overwrite on collision):
   ```bash
   cache_app \
       --csv filtered_profiled.csv \
       --cache existing_cache.cachebin \
       --type DPU \
       --intf 5113 \
       --use_new_costs
   ```

## Verification

After updating the cache files, verify that:

1. The `.cachebin` file was generated successfully
2. The file size is reasonable (not empty or corrupted)
3. Test compilation with the new cache to ensure it's being used correctly

## Troubleshooting

### Service Connection Issues

#### Symptom: Cannot connect to profiling service

**Diagnostic Steps:**

1. **Verify network connectivity:**
   ```bash
   ping IR-NPU_RESERVED_1-S-053-C.ger.corp.intel.com
   ```

2. **Check if port is accessible:**
   ```bash
   # Using telnet
   telnet IR-NPU_RESERVED_1-S-053-C.ger.corp.intel.com 5000
   
   # Using netcat
   nc -zv IR-NPU_RESERVED_1-S-053-C.ger.corp.intel.com 5000
   
   # Using curl
   curl -v http://IR-NPU_RESERVED_1-S-053-C.ger.corp.intel.com:5000/
   ```

3. **Enable debug logging to see detailed connection information:**
   ```bash
   export VPUNN_HTTP_CLIENT_DEBUG=1
   ```
   Then recompile to see detailed HTTP request/response logs.

**Common Causes:**
- Profiling service is down or not running on the specified host
- Firewall blocking the connection
- Wrong hostname or port number
- Network routing issues
- VPN or proxy configuration required

**Solutions:**
- Contact the profiling service administrator to verify service status
- Check firewall rules: `sudo iptables -L -n | grep 5000`
- Verify you're on the correct network (VPN, internal network, etc.)
- Try alternative profiling service hosts if available

### HTTP Communication Errors

#### Symptom: "Failed to send request to server" or timeout errors

**Enable debug mode for detailed diagnostics:**
```bash
export VPUNN_HTTP_CLIENT_DEBUG=1
```

**Check for:**
- Connection timeout (service too slow or overloaded)
- HTTP error codes in debug output (4xx, 5xx)
- JSON parsing errors in response

**Solutions:**
- Increase timeout settings if available
- Check profiling service logs for backend errors
- Verify the service isn't overloaded (too many concurrent requests)
- Try a simple test workload first

### No CSV Files Generated

#### Symptom: Compilation completes but no CSV files are created

**Check:**

1. **Serialization is enabled:**
   ```bash
   echo $ENABLE_VPUNN_DATA_SERIALIZATION  # Should output: TRUE
   ```

2. **HTTP client was compiled correctly:**
   ```bash
   # Verify in CMake configure output
   grep -i "VPUNN_BUILD_HTTP_CLIENT" build/CMakeCache.txt
   ```

3. **Write permissions in output directory:**
   ```bash
   ls -ld $(pwd)  # Check if you can write to current directory
   ```

4. **Check compilation logs for HTTP client initialization:**
   - Enable debug mode: `export VPUNN_HTTP_CLIENT_DEBUG=1`
   - Look for HTTP client initialization messages

**Solutions:**
- Ensure `ENABLE_VPUNN_DATA_SERIALIZATION=TRUE` is set before compilation
- Rebuild with `-DVPUNN_BUILD_HTTP_CLIENT=ON`
- Verify output directory has write permissions
- Check disk space availability

### Empty or Missing profiling_service_silicon Entries

#### Symptom: CSV files generated but no rows with `cost_source=profiling_service_silicon`

**This means the profiling service is not being used. Check:**

1. **Service availability:**
   ```bash
   export VPUNN_HTTP_CLIENT_DEBUG=1
   ```
   Look for connection or availability check messages in the logs.

2. **Backend compatibility:**
   - Verify `VPUNN_PROFILING_SERVICE_BACKEND=silicon` is correct
   - Check if the workload type is supported by the profiling service
   - Some operations may not be supported by the profiling backend

3. **Service responding correctly:**
   - Check profiling service logs for errors
   - Verify the service returns valid JSON responses
   - Test with a simple known-good workload

**Debug using curl:**
```bash
# Test service availability
curl -X POST http://IR-NPU_RESERVED_1-S-053-C.ger.corp.intel.com:5000/generate_workload \
  -H "Content-Type: application/json" \
  -d '{
    "params": {
      "status": true,
      "name": "profiling_request"
    }
  }'
```

Expected response should include backend information and status.

**Solutions:**
- Verify the profiling service is actually running: contact service administrator
- Check that the specified backend (`silicon`) is available on the service
- Review workload compatibility with the profiling service
- Check service capacity and queue status

### JSON Parsing Errors

#### Symptom: "Failed to parse JSON response" errors

**Enable debug to see the raw response:**
```bash
export VPUNN_HTTP_CLIENT_DEBUG=1
```

**Common causes:**
- Service returning HTML error page instead of JSON
- Malformed JSON response from service
- Encoding issues in response

**Solutions:**
- Check the raw response in debug output
- Verify service is returning JSON (Content-Type: application/json)
- Test service endpoint manually with curl
- Check profiling service version compatibility

### Performance Issues

#### Symptom: Compilation is very slow when profiling service is enabled

**This is expected behavior** - real hardware profiling takes time. However, if it's excessively slow:

1. **Check service load:**
   - The profiling service may be processing many concurrent requests
   - Contact administrator for service status

2. **Network latency:**
   - High latency to profiling service host
   - Use `ping` to check latency
   - Consider using a closer profiling service instance if available

3. **Large models:**
   - Models with many operations will make more profiling requests
   - Consider profiling a subset of operations first

**Mitigation:**
- Use profiling service only for critical models
- Profile in batches during off-peak hours
- Use existing cache when possible (disable profiling service for cached workloads)

### Cache Update Issues

#### Symptom: cache_app fails or produces invalid cache files

**Check:**

1. **CSV format is correct:**
   ```bash
   head -n 5 filtered_profiled.csv  # Verify headers and data format
   ```

2. **Correct interface version:**
   - For NPU 4.x/5.x: use `--intf <version>`
   - For NPU 6.0+: **do NOT use** `--intf`

3. **Cost values are valid:**
   - Check that `vpunn_cycles` column contains numeric values
   - No negative or invalid cycle counts

**Debug cache_app:**
```bash
cache_app --help  # Verify all parameters
cache_app --csv test.csv --cache out.cachebin --type DPU --intf 5113 -v  # If verbose flag exists
```

### Verifying Setup

#### Quick verification script:

```bash
#!/bin/bash
echo "=== VPUNN HTTP Profiling Configuration Check ==="
echo ""
echo "1. Environment Variables:"
echo "   ENABLE_VPUNN_PROFILING_SERVICE: ${ENABLE_VPUNN_PROFILING_SERVICE:-NOT SET}"
echo "   VPUNN_PROFILING_SERVICE_HOST: ${VPUNN_PROFILING_SERVICE_HOST:-NOT SET}"
echo "   VPUNN_PROFILING_SERVICE_PORT: ${VPUNN_PROFILING_SERVICE_PORT:-NOT SET}"
echo "   VPUNN_PROFILING_SERVICE_BACKEND: ${VPUNN_PROFILING_SERVICE_BACKEND:-NOT SET}"
echo "   ENABLE_VPUNN_DATA_SERIALIZATION: ${ENABLE_VPUNN_DATA_SERIALIZATION:-NOT SET}"
echo "   VPUNN_HTTP_CLIENT_DEBUG: ${VPUNN_HTTP_CLIENT_DEBUG:-NOT SET}"
echo ""
echo "2. Network Connectivity:"
if command -v nc &> /dev/null; then
    nc -zv ${VPUNN_PROFILING_SERVICE_HOST} ${VPUNN_PROFILING_SERVICE_PORT} 2>&1
else
    echo "   netcat not available, skipping port check"
fi
echo ""
echo "3. Service Availability Test:"
if command -v curl &> /dev/null; then
    curl -s -X POST http://${VPUNN_PROFILING_SERVICE_HOST}:${VPUNN_PROFILING_SERVICE_PORT}/generate_workload \
      -H "Content-Type: application/json" \
      -d '{"params": {"status": true, "name": "profiling_request"}}' | head -c 200
    echo ""
else
    echo "   curl not available, skipping service check"
fi
```

Save as `check_profiling_setup.sh`, make executable (`chmod +x check_profiling_setup.sh`), and run to verify your configuration.

## Best Practices

- Always verify service availability before starting a profiling session
- Keep profiling data organized by experiment name and date
- Maintain backups of original cache files before updating
- Document the profiling conditions (hardware, configuration, etc.)
- Validate updated caches with test compilations before deploying to production
