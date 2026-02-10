#!/usr/bin/env python3
"""
Test script that parses a CSV file, runs the cost_model_cli for each workload,
and writes the results back to the CSV under the 'cli_result' column.

Usage:
    python test_csv.py <csv_filename> <base_dir> <cli_executable_path>
    
Example:
    python test_csv.py workloads.csv /path/to/project /path/to/project/build/apps/cost_model_cli/cost_model_cli
"""

import pandas as pd
import subprocess
import os
import sys
import re
from pathlib import Path

class CSVTestRunner:
    def __init__(self, csv_path, cli_executable_path, model_path=None, cache_path=None, base_dir=None):
        """
        Initialize the CSV test runner.
        
        Args:
            csv_path: Path to the CSV file containing test workloads
            cli_executable_path: Path to the cost_model_cli executable
            model_path: Path to the VPU model file
            cache_path: Path to cache directory (optional)
            base_dir: Base directory to search for models and other files (current working directory)
        """
        self.csv_path = csv_path
        self.cli_executable_path = cli_executable_path
        self.model_path = model_path
        self.cache_path = cache_path or "/tmp/vpunn_cache"
        
        # Set base directory (should be current working directory)
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()
        
        # Verify paths exist
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        if not os.path.exists(cli_executable_path):
            raise FileNotFoundError(f"CLI executable not found: {cli_executable_path}")
            
    def parse_device_name(self, device_str):
        """Convert CSV device format to CLI format."""
        device_mapping = {
            'VPUDevice.VPU_2_0': 'VPU_2_0',
            'VPUDevice.VPU_2_1': 'VPU_2_1',
            'VPUDevice.VPU_2_7': 'VPU_2_7',
            'VPUDevice.VPU_4_0': 'VPU_4_0',
            'VPUDevice.NPU_5_0': 'NPU_5_0',
            'VPUDevice.NPU_RESERVED': 'NPU_RESERVED',
            'VPUDevice.NPU_RESERVED_1': 'NPU_RESERVED_1'
        }
        return device_mapping.get(device_str, device_str.replace('VPUDevice.', ''))
    
    def parse_operation_name(self, operation_str):
        """Convert CSV operation format to CLI format."""
        operation_mapping = {
            'Operation.CONVOLUTION': 'CONVOLUTION',
            'Operation.DW_CONVOLUTION': 'DW_CONVOLUTION',
            'Operation.CM_CONVOLUTION': 'CM_CONVOLUTION',
            'Operation.MAXPOOL': 'MAXPOOL',
            'Operation.AVEPOOL': 'AVEPOOL',
            'Operation.ELTWISE': 'ELTWISE',
            'Operation.ELTWISE_MUL': 'ELTWISE_MUL'
        }
        return operation_mapping.get(operation_str, operation_str.replace('Operation.', ''))
    
    def parse_execution_mode(self, exec_mode_str):
        """Convert CSV execution mode format to CLI format."""
        exec_mode_mapping = {
            'ExecutionMode.VECTOR_FP16': 'VECTOR_FP16',
            'ExecutionMode.VECTOR': 'VECTOR',
            'ExecutionMode.CUBOID_16x16': 'CUBOID_16x16',
            'ExecutionMode.CUBOID_8x16': 'CUBOID_8x16',
            'ExecutionMode.CUBOID_4x16': 'CUBOID_4x16'
        }
        return exec_mode_mapping.get(exec_mode_str, exec_mode_str.replace('ExecutionMode.', ''))
    
    def parse_datatype(self, datatype_str):
        """Convert CSV datatype format to CLI format."""
        datatype_mapping = {
            'DataType.UINT8': 'UINT8',
            'DataType.INT8': 'INT8',
            'DataType.UINT16': 'UINT16',
            'DataType.INT16': 'INT16',
            'DataType.INT32': 'INT32',
            'DataType.FLOAT16': 'FLOAT16',
            'DataType.BFLOAT16': 'BFLOAT16',
            'DataType.FLOAT32': 'FLOAT32',
            'DataType.BF8': 'BF8',
            'DataType.HF8': 'HF8',
            'DataType.UINT4': 'UINT4',
            'DataType.INT4': 'INT4',
            'DataType.UINT2': 'UINT2',
            'DataType.INT2': 'INT2',
            'DataType.UINT1': 'UINT1',
            'DataType.INT1': 'INT1',
            'DataType.FLOAT4': 'FLOAT4'
        }
        return datatype_mapping.get(datatype_str, datatype_str.replace('DataType.', ''))
    
    def parse_layout(self, layout_str):
        """Convert CSV layout format to CLI format."""
        layout_mapping = {
            'Layout.ZXY': 'ZXY',
            'Layout.XYZ': 'XYZ',
            'Layout.XZY': 'XZY',
            'Layout.YXZ': 'YXZ',
            'Layout.YZX': 'YZX',
            'Layout.ZYX': 'ZYX'
        }
        return layout_mapping.get(layout_str, layout_str.replace('Layout.', ''))
    
    def parse_swizzling(self, swizzling_str):
        """Convert CSV swizzling format to CLI format."""
        swizzling_mapping = {
            'Swizzling.KEY_0': 'KEY_0',
            'Swizzling.KEY_1': 'KEY_1',
            'Swizzling.KEY_2': 'KEY_2',
            'Swizzling.KEY_3': 'KEY_3',
            'Swizzling.KEY_4': 'KEY_4',
            'Swizzling.KEY_5': 'KEY_5'
        }
        return swizzling_mapping.get(swizzling_str, swizzling_str.replace('Swizzling.', ''))
    
    def parse_activation_function(self, activation_str):
        """Convert CSV activation function format to CLI format."""
        activation_mapping = {
            'ActivationFunction.NONE': 'NONE',
            'ActivationFunction.RELU': 'RELU',
            'ActivationFunction.LRELU': 'LRELU',
            'ActivationFunction.ADD': 'ADD',
            'ActivationFunction.SUB': 'SUB',
            'ActivationFunction.MULT': 'MULT'
        }
        return activation_mapping.get(activation_str, activation_str.replace('ActivationFunction.', ''))
    
    def parse_isi_strategy(self, isi_str):
        """Convert CSV ISI strategy format to CLI format."""
        isi_mapping = {
            'ISIStrategy.Clustering': 'CLUSTERING',
            'ISIStrategy.SplitOverH': 'SPLIT_OVER_H',
            'ISIStrategy.SplitOverK': 'SPLIT_OVER_K',
            'ISIStrategy.SplitOverKH': 'SPLIT_OVER_KH'
        }
        return isi_mapping.get(isi_str, isi_str.replace('ISIStrategy.', ''))
    
    def build_cli_command(self, row, model_path):
        """Build CLI command from CSV row data."""
        cmd = [self.cli_executable_path]
        
        # Basic required parameters
        cmd.extend(['--device', self.parse_device_name(row['device'])])
        cmd.extend(['--hw_module', 'DPU'])  # All CSV entries are DPU workloads
        cmd.extend(['--model', model_path])
        
        # if self.cache_path:
        #     cmd.extend(['--cache_path', self.cache_path])
        
        # DPU-specific parameters
        cmd.extend(['--operation', self.parse_operation_name(row['operation'])])
        
        # Input/output dimensions
        cmd.extend(['--input-channels', str(int(row['input_0_channels']))])
        cmd.extend(['--output-channels', str(int(row['output_0_channels']))])
        cmd.extend(['--inbatch', str(int(row['input_0_batch']))])
        cmd.extend(['--inheight', str(int(row['input_0_height']))])
        cmd.extend(['--inwidth', str(int(row['input_0_width']))])
        cmd.extend(['--outbatch', str(int(row['output_0_batch']))])
        cmd.extend(['--outheight', str(int(row['output_0_height']))])
        cmd.extend(['--outwidth', str(int(row['output_0_width']))])
        

        # Kernel parameters
        cmd.extend(['--kernel-height', str(int(row['kernel_height']))])
        cmd.extend(['--kernel-width', str(int(row['kernel_width']))])
        
        # Stride parameters
        cmd.extend(['--stride-height', str(int(row['kernel_stride_height']))])
        cmd.extend(['--stride-width', str(int(row['kernel_stride_width']))])
        
        # Padding parameters
        cmd.extend(['--pad-top', str(int(row['kernel_pad_top']))])
        cmd.extend(['--pad-bottom', str(int(row['kernel_pad_bottom']))])
        cmd.extend(['--pad-left', str(int(row['kernel_pad_left']))])
        cmd.extend(['--pad-right', str(int(row['kernel_pad_right']))])
        
        # Execution mode
        cmd.extend(['--execution-mode', self.parse_execution_mode(row['execution_order'])])
        
        # Data types
        cmd.extend(['--input-datatype', self.parse_datatype(row['input_0_datatype'])])
        cmd.extend(['--output-datatype', self.parse_datatype(row['output_0_datatype'])])


        if pd.notna(row['in_place_input1']):
            cmd.extend(['--in-place-input1', str(int(row['in_place_input1']))])
        if pd.notna(row['in_place_output']):    
            cmd.extend(['--in-place-output', str(int(row['in_place_output']))])
        if pd.notna(row['superdense_output']):
            cmd.extend(['--superdense-memory', str(int(row['superdense_output']))])
        if pd.notna(row['input_autopad']):
            cmd.extend(['--input-autopad', str(int(row['input_autopad']))])
        if pd.notna(row['output_autopad']):
            cmd.extend(['--output-autopad', str(int(row['output_autopad']))])


        # Optional parameters
        if pd.notna(row['input_1_datatype']) and row['input_1_datatype'] != 'DataType.INVALID':
            cmd.extend(['--weight-datatype', self.parse_datatype(row['input_1_datatype'])])
        
        if pd.notna(row['activation_function']):
            cmd.extend(['--activation-function', self.parse_activation_function(row['activation_function'])])
        
        if pd.notna(row['isi_strategy']):
            cmd.extend(['--isi-strategy', self.parse_isi_strategy(row['isi_strategy'])])
        
        if pd.notna(row['output_write_tiles']) and row['output_write_tiles'] != 1:
            cmd.extend(['--output-write-tiles', str(int(row['output_write_tiles']))])
        
        # Layout parameters
        if pd.notna(row['input_0_layout']) and row['input_0_layout'] != 'Layout.INVALID':
            cmd.extend(['--input-layout', self.parse_layout(row['input_0_layout'])])
        
        if pd.notna(row['output_0_layout']) and row['output_0_layout'] != 'Layout.INVALID':
            cmd.extend(['--output-layout', self.parse_layout(row['output_0_layout'])])
        
        # Swizzling parameters
        if pd.notna(row['input_0_swizzling']) and row['input_0_swizzling'] != 'Swizzling.KEY_0':
            cmd.extend(['--input-swizzling-0', self.parse_swizzling(row['input_0_swizzling'])])
        
        if pd.notna(row['output_0_swizzling']) and row['output_0_swizzling'] != 'Swizzling.KEY_0':
            cmd.extend(['--output-swizzling', self.parse_swizzling(row['output_0_swizzling'])])
        
        # Sparsity parameters
        if pd.notna(row['input_sparsity_enabled']) and int(row['input_sparsity_enabled']) == 1:
            cmd.append('--input-sparsity-enabled')
            if pd.notna(row['input_sparsity_rate']) and float(row['input_sparsity_rate']) > 0:
                cmd.extend(['--input-sparsity-rate', str(float(row['input_sparsity_rate']))])
        
        if pd.notna(row['weight_sparsity_enabled']) and int(row['weight_sparsity_enabled']) == 1:
            cmd.append('--weight-sparsity-enabled')
            if pd.notna(row['weight_sparsity_rate']) and float(row['weight_sparsity_rate']) > 0:
                cmd.extend(['--weight-sparsity-rate', str(float(row['weight_sparsity_rate']))])
        
        if pd.notna(row['output_sparsity_enabled']) and int(row['output_sparsity_enabled']) == 1:
            cmd.append('--output-sparsity-enabled')
        
        # MPE engine
        if pd.notna(row['mpe_engine']):
            cmd.extend(['--mpe-engine', row['mpe_engine']])
        
        return cmd
    
    def extract_cycles_from_output(self, output):
        """Extract cycle count from CLI output."""
        try:
            # Look for the pattern "DPU execution cycles: XXXXX"
            match = re.search(r'DPU execution cycles:\s*(\d+)', output)
            if match:
                return int(match.group(1))
            
            # Alternative pattern if different output format
            match = re.search(r'cycles:\s*(\d+)', output, re.IGNORECASE)
            if match:
                return int(match.group(1))
                
            return None
        except Exception as e:
            print(f"Error extracting cycles: {e}")
            return None
    
    def run_cli_command(self, cmd, timeout=60):
        """Execute CLI command and return result."""
        try:
            # Convert all arguments to strings
            cmd_str = [str(arg) for arg in cmd]
            
            print(f"Running: {' '.join(cmd_str)}")
            
            result = subprocess.run(
                cmd_str,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            if result.returncode == 0:
                cycles = self.extract_cycles_from_output(result.stdout)
                return {
                    'success': True,
                    'cycles': cycles,
                    'stdout': result.stdout,
                    'stderr': result.stderr
                }
            else:
                return {
                    'success': False,
                    'cycles': None,
                    'stdout': result.stdout,
                    'stderr': result.stderr,
                    'returncode': result.returncode
                }
                
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'cycles': None,
                'error': 'Command timeout'
            }
        except Exception as e:
            return {
                'success': False,
                'cycles': None,
                'error': str(e)
            }
    
    def find_model_path(self, device_name):
        """Find appropriate model path based on device."""
        # Common model paths - adjust these based on your setup
        model_paths = {
            'VPU_2_0': 'vpu_2_0.vpunn',
            'VPU_2_1': 'vpu_2_1.vpunn', 
            'VPU_2_7': 'vpu_2_7.vpunn',
            'VPU_4_0': 'vpu_4_0.vpunn',
            'NPU_5_0': 'vpu_5_1.vpunn',
            'NPU_RESERVED': 'vpu_5_1.vpunn',
            'NPU_RESERVED_1': 'vpu_6_1.vpunn'
        }
            
        # Try to find model in standard locations relative to base directory
        model_dirs = [
            self.base_dir / "models"
        ]
        print(model_dirs)
        for model_dir in model_dirs:
            if device_name in model_paths:
                full_path = model_dir / model_paths[device_name]
                print(f"  Checking for model: {full_path}")
                if full_path.exists():
                    return str(full_path)
        
        # Default fallback - use any available model
        for model_dir in model_dirs:
            if model_dir.exists():
                for file in model_dir.iterdir():
                    if file.suffix == '.vpunn':
                        return str(file)
        
        raise FileNotFoundError(f"No suitable model file found for device {device_name}. Searched in: {[str(d) for d in model_dirs]}")
    
    def run_tests(self, output_csv_path=None, max_rows=None, start_row=0):
        """Run tests for all rows in CSV and write results."""
        # Load CSV
        df = pd.read_csv(self.csv_path)
        
        if max_rows:
            df = df.head(max_rows)
        
        total_rows = len(df)
        print(f"Processing {total_rows} workloads (starting from row {start_row})...")
        
        # Initialize results columns if they don't exist
        if 'cli_result' not in df.columns:
            df['cli_result'] = None
        if 'cli_error' not in df.columns:
            df['cli_error'] = None
        if 'cli_success' not in df.columns:
            df['cli_success'] = None
        
        # Process each row starting from start_row
        successful_count = 0
        failed_count = 0
        
        for idx, row in df.iterrows():
            # Skip rows before start_row
            if idx < start_row:
                continue
                
            # Progress reporting
            progress = idx + 1
            print(f"\nProcessing row {progress}/{total_rows} (Success: {successful_count}, Failed: {failed_count})")
            
            # Skip if already processed (has result or error)
            if pd.notna(df.at[idx, 'cli_success']):
                if df.at[idx, 'cli_success']:
                    successful_count += 1
                else:
                    failed_count += 1
                print(f"  Skipping (already processed)")
                continue
            
            try:
                # Get device and find appropriate model
                device_name = self.parse_device_name(row['device'])
                model_path = self.find_model_path(device_name)
                print(f"  Using model: {model_path}")
                
                # Build and run CLI command
                cmd = self.build_cli_command(row, model_path)
                result = self.run_cli_command(cmd)
                
                # Store results
                if result['success'] and result['cycles'] is not None:
                    df.at[idx, 'cli_result'] = result['cycles']
                    df.at[idx, 'cli_success'] = True
                    df.at[idx, 'cli_error'] = None
                    successful_count += 1
                    print(f"  Success: {result['cycles']} cycles")
                else:
                    df.at[idx, 'cli_result'] = None
                    df.at[idx, 'cli_success'] = False
                    
                    # Collect comprehensive error information
                    error_parts = []
                    if result.get('error'):
                        error_parts.append(f"Error: {result['error']}")
                    if result.get('stderr') and result['stderr'].strip():
                        error_parts.append(f"STDERR: {result['stderr'].strip()}")
                    if result.get('stdout') and result['stdout'].strip():
                        error_parts.append(f"STDOUT: {result['stdout'].strip()}")
                    if result.get('returncode') is not None:
                        error_parts.append(f"Return code: {result['returncode']}")
                    
                    error_msg = " | ".join(error_parts) if error_parts else "Unknown error"
                    df.at[idx, 'cli_error'] = error_msg
                    failed_count += 1
                    print(f"  Failed: {error_msg}")
                    
            except Exception as e:
                df.at[idx, 'cli_result'] = None
                df.at[idx, 'cli_success'] = False
                error_msg = f"Exception: {type(e).__name__}: {str(e)}"
                df.at[idx, 'cli_error'] = error_msg
                failed_count += 1
                print(f"  {error_msg}")
        
        # Save final results
        output_path = output_csv_path or self.csv_path
        df.to_csv(output_path, index=False)
        print(f"\nCompleted! Results saved to {output_path}")
        print(f"Summary: {successful_count} successful, {failed_count} failed out of {total_rows} total rows")
        
        return df


def main():
    """Main function to run the tests."""
    if len(sys.argv) != 4:
        print("Usage: python test_csv.py <csv_filename> <base_dir> <cli_executable_path>")
        print("Example: python test_csv.py workloads.csv /path/to/project /path/to/cost_model_cli")
        return 1
    
    csv_filename = sys.argv[1]
    base_dir_arg = sys.argv[2]
    cli_executable_path = sys.argv[3]
    
    # Setup paths
    current_dir = Path(__file__).parent
    csv_path = Path(csv_filename)
    
    # If CSV path is relative, make it relative to current directory
    if not csv_path.is_absolute():
        csv_path = current_dir / csv_path
    
    # Use provided base directory
    base_dir = Path(base_dir_arg).resolve()
    
    # Verify base directory exists
    if not base_dir.exists():
        print(f"Error: Base directory not found: {base_dir}")
        return 1
    
    # Use provided CLI executable path
    cli_executable = Path(cli_executable_path).resolve()
    
    # Verify CLI executable exists
    if not cli_executable.exists():
        print(f"Error: CLI executable not found: {cli_executable}")
        return 1
    
    # Check if paths exist
    if not csv_path.exists():
        print(f"Error: CSV file not found at {csv_path}")
        return 1
        
    if not cli_executable.exists():
        print(f"Error: CLI executable not found at {cli_executable}")
        print("Please build the project first or specify correct path with --cli-executable")
        return 1
    
    # Generate output filename: input_with_results.csv
    output_path = csv_path.parent / f"{csv_path.stem}_with_results.csv"
    
    try:
        # Create test runner
        runner = CSVTestRunner(
            csv_path=str(csv_path),
            cli_executable_path=str(cli_executable),
            model_path=None,
            cache_path="/tmp/vpunn_test_cache",
            base_dir=str(base_dir)
        )
        
        # Run tests (process all rows)
        results_df = runner.run_tests(
            output_csv_path=str(output_path),
            max_rows=None,
            start_row=0
        )
        
        print(f"\nTest completed! Results available in: {output_path}")
        
        return 0
        
    except Exception as e:
        print(f"Error running tests: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
