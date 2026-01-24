#!/usr/bin/env python3
"""
KeOps JAX Test Runner
=====================
Main entry point for running the test suite.

Usage:
    python run_tests.py                  # Run all tests
    python run_tests.py api              # Run API tests only
    python run_tests.py correctness      # Run correctness tests only
    python run_tests.py edge             # Run edge case tests only
    python run_tests.py benchmark        # Run single-GPU benchmarks
    python run_tests.py benchmark-multi  # Run multi-GPU benchmarks
    python run_tests.py quick            # Run quick sanity check
    
Options:
    --no-pytorch    Skip PyTorch comparison tests
    --verbose       Verbose output
    --save          Save results to file
"""

import sys
import argparse
import subprocess
from pathlib import Path


def get_script_dir():
    """Get directory containing test scripts."""
    return Path(__file__).parent


def run_script(script_name: str, extra_args: list = None) -> int:
    """Run a test script and return exit code."""
    script_path = get_script_dir() / script_name
    
    if not script_path.exists():
        print(f"Error: Script not found: {script_path}")
        return 1
    
    cmd = [sys.executable, str(script_path)]
    if extra_args:
        cmd.extend(extra_args)
    
    return subprocess.call(cmd)


def main():
    parser = argparse.ArgumentParser(
        description="KeOps JAX Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Test Suites:
  api             Core API tests (Genred, LazyTensor, Vi/Vj/Pm)
  correctness     Cross-validation against PyTorch KeOps
  edge            Edge case tests (bugs found during development)
  benchmark       Single-GPU performance benchmarks
  benchmark-multi Multi-GPU scaling benchmarks
  quick           Quick sanity check (subset of api tests)
  all             Run all tests (default)

Examples:
  python run_tests.py                    # Run all tests
  python run_tests.py edge               # Run edge case tests
  python run_tests.py api correctness    # Run multiple suites
  python run_tests.py benchmark --save   # Run benchmarks and save results
"""
    )
    
    parser.add_argument(
        'suites', 
        nargs='*', 
        default=['all'],
        help='Test suites to run'
    )
    parser.add_argument(
        '--no-pytorch', 
        action='store_true',
        help='Skip PyTorch comparison tests'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )
    parser.add_argument(
        '--save', '-s',
        action='store_true',
        help='Save results to file'
    )
    
    args = parser.parse_args()
    
    # Map suite names to scripts
    suite_map = {
        'api': 'test_api.py',
        'correctness': 'test_correctness.py',
        'edge': 'test_edge_cases.py',
        'advanced': 'test_advanced.py',
        'benchmark': 'test_benchmark_single_gpu.py',
        'benchmark-multi': 'test_benchmark_multi_gpu.py',
    }
    
    # Handle 'all' and 'quick' specially
    if 'all' in args.suites:
        suites = ['edge', 'api', 'correctness', 'advanced']
    elif 'quick' in args.suites:
        suites = ['edge']  # Edge cases are quick and comprehensive
    else:
        suites = args.suites
    
    # Validate suites
    for suite in suites:
        if suite not in suite_map:
            print(f"Unknown suite: {suite}")
            print(f"Available suites: {', '.join(suite_map.keys())}")
            return 1
    
    # Print header
    print("=" * 70)
    print("KeOps JAX Test Suite".center(70))
    print("=" * 70)
    print(f"\nRunning suites: {', '.join(suites)}")
    print()
    
    # Run tests
    results = {}
    overall_exit = 0
    
    for suite in suites:
        script = suite_map[suite]
        print(f"\n{'─' * 70}")
        print(f"Running: {suite}")
        print(f"{'─' * 70}\n")
        
        exit_code = run_script(script)
        results[suite] = exit_code
        
        if exit_code != 0:
            overall_exit = 1
    
    # Print summary
    print("\n" + "=" * 70)
    print("Test Summary".center(70))
    print("=" * 70)
    
    for suite, exit_code in results.items():
        status = "✓ PASSED" if exit_code == 0 else "✗ FAILED"
        print(f"  {suite:<20} {status}")
    
    print()
    if overall_exit == 0:
        print("  All tests passed!")
    else:
        print("  Some tests failed.")
    print()
    
    return overall_exit


if __name__ == "__main__":
    sys.exit(main())
