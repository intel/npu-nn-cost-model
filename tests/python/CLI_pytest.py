#!/usr/bin/env python3
"""
Pytest for running CLI_inference on a CSV and comparing vpunn_cycles and inference_cycles.
Fails if any differences are found, and prints the CSV line where a difference is spotted.
"""
import pytest
import pandas as pd
import numpy as np
import os
import subprocess
import sys
from pathlib import Path

def run_cli_inference(input_csv, output_csv=None, base_dir=None, cli_executable=None):
    """
    Runs CLI_inference.py on the input CSV and returns the path to the output CSV.
    If output_csv is not provided, a temporary file is created.
    
    Args:
        input_csv: Path to input CSV file
        output_csv: Path for output CSV file (optional, auto-generated if None)
        base_dir: Base directory for the project (defaults to project root)
        cli_executable: Path to cost_model_cli executable (auto-detected if None)
    """
    if output_csv is None:
        output_csv = str(Path(input_csv).with_name("cli_inference_output.csv"))
    
    # Find CLI_inference.py script
    script_path = Path(__file__).parent / "CLI_inference.py"
    if not script_path.exists():
        pytest.fail(f"CLI_inference.py not found at {script_path}")
    
    # Set default base directory to project root
    if base_dir is None:
        base_dir = str(Path(__file__).parent.parent.parent)  # Go up to project root
    
    # Auto-detect CLI executable if not provided
    if cli_executable is None:
        project_root = Path(base_dir)
        # Common build locations for the CLI executable
        cli_candidates = [
            project_root / "out" / "build" / "x64-Debug" / "apps" / "cost_model_cli" / "cost_model_cli",
            project_root / "out" / "build" / "x64-Debug" / "apps" / "cost_model_cli" / "cost_model_cli.exe",
            project_root / "build" / "apps" / "cost_model_cli" / "cost_model_cli",
            project_root / "build" / "apps" / "cost_model_cli" / "cost_model_cli.exe",
            project_root / "apps" / "cost_model_cli" / "cost_model_cli",
            project_root / "apps" / "cost_model_cli" / "cost_model_cli.exe",
            
        ]
        
        for candidate in cli_candidates:
            if candidate.exists():
                cli_executable = str(candidate)
                break
        
        if cli_executable is None:
            pytest.fail(f"Could not find cost_model_cli executable. Checked: {[str(c) for c in cli_candidates]}")
    
    # CLI_inference.py expects: <csv_filename> <base_dir> <cli_executable_path>
    cmd = [sys.executable, str(script_path), input_csv, base_dir, cli_executable]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        pytest.fail(f"CLI_inference failed: {result.stderr}\n{result.stdout}")
    
    # CLI_inference.py generates output with pattern: input_with_results.csv
    expected_output = str(Path(input_csv).parent / f"{Path(input_csv).stem}_with_results.csv")
    
    if not os.path.exists(expected_output):
        pytest.fail(f"CLI_inference did not produce expected output CSV: {expected_output}")
    
    return expected_output

def compare_cycles(csv_path):
    """
    Compares vpunn_cycles and inference_cycles in the given CSV.
    Fails if any differences are found, and prints the CSV line where a difference is spotted.
    """
    df = pd.read_csv(csv_path)
    if 'vpunn_cycles' not in df.columns or 'inference_cycles' not in df.columns:
        pytest.fail(f"CSV missing required columns. Columns found: {list(df.columns)}")
    differences = []
    for idx, row in df.iterrows():
        v = row['vpunn_cycles']
        i = row['inference_cycles']
        if pd.isna(v) and pd.isna(i):
            continue
        if pd.isna(v) or pd.isna(i):
            differences.append(idx)
            continue
        try:
            if not np.isclose(float(v), float(i), rtol=1e-10, atol=1e-10):
                differences.append(idx)
        except Exception:
            if str(v) != str(i):
                differences.append(idx)
    if differences:
        lines = [f"Difference at CSV line {d+2}: vpunn_cycles={df.iloc[d]['vpunn_cycles']}, inference_cycles={df.iloc[d]['inference_cycles']}" for d in differences]
        pytest.fail("\n".join(lines))

def test_cli_inference_cycles(tmp_path):
    """
    Main pytest: takes a CSV, runs CLI_inference, and compares cycles.
    Uses a CSV file in the same directory as this pytest file.
    """
    # Use CSV file from the same directory as this pytest file
    csv_filename = "test_CLI.csv"  # Change this to your actual CSV filename
    input_csv = str(Path(__file__).parent / csv_filename)
    
    # Check if the CSV file exists
    if not os.path.exists(input_csv):
        pytest.fail(f"CSV file not found: {input_csv}. Please place your CSV file in the same directory as this test.")
    
    # Optional: Customize these parameters if needed
    base_dir = None  # Will auto-detect project root
    cli_executable = None  # Will auto-detect cost_model_cli executable
    
    # Run CLI_inference and get the output CSV path
    output_csv = run_cli_inference(input_csv, base_dir=base_dir, cli_executable=cli_executable)
    
    # Compare the cycles columns
    compare_cycles(output_csv)
