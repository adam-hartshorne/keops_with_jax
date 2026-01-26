#!/usr/bin/env python3
"""
KeOps JAX Test Runner
=====================
Main entry point for running the test suite.
"""

# To install KeOps JAX:
# pip install -e . in pykeops/pykeops and pykeops/keopscore


import sys
import argparse
import subprocess
from pathlib import Path

# Try to import rich for better UI
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich import box

    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False


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


def print_summary_table(results):
    """Print a visually pleasing summary table."""
    if RICH_AVAILABLE:
        table = Table(title="Test Execution Summary", box=box.ROUNDED)
        table.add_column("Test Suite", style="cyan", no_wrap=True)
        table.add_column("Result", justify="center")
        table.add_column("Exit Code", justify="right")

        for suite, code in results.items():
            if code == 0:
                status = "[bold green]PASSED[/bold green]"
                code_style = "green"
            else:
                status = "[bold red]FAILED[/bold red]"
                code_style = "red"

            table.add_row(suite, status, f"[{code_style}]{code}[/{code_style}]")

        console.print()
        console.print(table)

        overall_success = all(c == 0 for c in results.values())
        if overall_success:
            console.print(Panel("[bold green]All tests passed successfully![/bold green]",
                                border_style="green", expand=False))
        else:
            console.print(Panel("[bold red]Some tests failed. Check output above.[/bold red]",
                                border_style="red", expand=False))
        console.print()

    else:
        # Fallback for systems without rich
        print("\n" + "=" * 60)
        print("TEST SUMMARY".center(60))
        print("=" * 60)

        for suite, code in results.items():
            if code == 0:
                status = "\033[92mPASSED\033[0m"  # Green
            else:
                status = "\033[91mFAILED\033[0m"  # Red
            print(f"  {suite:<25} {status}")

        print("-" * 60)
        if all(c == 0 for c in results.values()):
            print("\033[92m  All tests passed!\033[0m")
        else:
            print("\033[91m  Some tests failed.\033[0m")
        print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="KeOps JAX Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Test Suites:
  api             Core API tests (Genred, LazyTensor, Vi/Vj/Pm)
  correctness     Cross-validation against PyTorch KeOps
  edge            Edge case tests (bugs found during development)
  advanced        Advanced features (Reductions, Math, Batches)
  batched         Batched (3D tensor) gradient tests vs PyTorch
  benchmark       Single-GPU performance benchmarks
  benchmark-multi Multi-GPU scaling benchmarks
  quick           Quick sanity check (subset of api tests)
  all             Run all tests (default)
"""
    )

    parser.add_argument('suites', nargs='*', default=['all'], help='Test suites to run')
    parser.add_argument('--no-pytorch', action='store_true', help='Skip PyTorch comparison tests')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--save', '-s', action='store_true', help='Save results to file')

    args = parser.parse_args()

    # Map suite names to scripts
    suite_map = {
        'api': 'test_api.py',
        'correctness': 'test_correctness.py',
        'edge': 'test_edge_cases.py',
        'advanced': 'test_advanced.py',
        'batched': 'test_batched_gradients.py',
        'benchmark': 'test_benchmark_single_gpu.py',
        'benchmark-multi': 'test_benchmark_multi_gpu.py',
    }

    # Handle 'all' and 'quick'
    if 'all' in args.suites:
        suites = ['edge', 'api', 'correctness', 'advanced', 'batched']
    elif 'quick' in args.suites:
        suites = ['edge']
    else:
        suites = args.suites

    # Validate suites
    for suite in suites:
        if suite not in suite_map:
            print(f"Unknown suite: {suite}")
            return 1

    # Print Header
    if RICH_AVAILABLE:
        console.rule("[bold blue]KeOps JAX Test Suite[/bold blue]")
        console.print(f"Running suites: [cyan]{', '.join(suites)}[/cyan]\n")
    else:
        print("=" * 70)
        print("KeOps JAX Test Suite".center(70))
        print("=" * 70)
        print(f"Running: {', '.join(suites)}\n")

    # Run tests
    results = {}
    overall_exit = 0

    for suite in suites:
        script = suite_map[suite]

        if RICH_AVAILABLE:
            console.rule(f"[bold magenta]Running: {suite}[/bold magenta]")
        else:
            print(f"\n{'─' * 70}")
            print(f"Running: {suite}")
            print(f"{'─' * 70}\n")

        # Pass through arguments
        extra_args = []
        if args.verbose: extra_args.append('--verbose')
        if args.save: extra_args.append('--save')

        exit_code = run_script(script, extra_args)
        results[suite] = exit_code

        if exit_code != 0:
            overall_exit = 1

    # Print Summary
    print_summary_table(results)

    return overall_exit


if __name__ == "__main__":
    sys.exit(main())
