#!/usr/bin/env python3
"""
KeOps JAX Test Utilities
========================
Shared utilities for test formatting, color output, and table rendering.

Features:
- Rich library integration for beautiful output (falls back to ANSI if unavailable)
- ASCII table rendering
- Test result tracking
- Progress indicators
- Comparison helpers
"""

import sys
import os
import time
from typing import List, Tuple, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from contextlib import contextmanager
import numpy as np


# =============================================================================
# Float64 Mode Support
# =============================================================================

def is_float64_mode() -> bool:
    """Check if tests should run in float64 mode."""
    return os.environ.get('KEOPS_TEST_FLOAT64', '0') == '1'


def get_np_dtype():
    """Get the numpy dtype to use for tests (float32 or float64)."""
    return np.float64 if is_float64_mode() else np.float32


def get_dtype_str() -> str:
    """Get the dtype string for KeOps ('float32' or 'float64')."""
    return 'float64' if is_float64_mode() else 'float32'


def setup_jax_float64():
    """
    Configure JAX for float64 if in float64 mode.
    Call this at the start of test scripts.
    """
    if is_float64_mode():
        # Set environment variable before JAX import
        os.environ['JAX_ENABLE_X64'] = '1'
        try:
            import jax
            jax.config.update('jax_enable_x64', True)
        except ImportError:
            pass

# =============================================================================
# Try to import Rich for beautiful output
# =============================================================================

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
    from rich.text import Text
    from rich import box
    from rich.live import Live
    from rich.style import Style
    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    console = None


# =============================================================================
# ANSI Color Codes (fallback when Rich not available)
# =============================================================================

class Colors:
    """ANSI color codes for terminal output."""
    # Basic colors
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    GREY = '\033[90m'
    
    # Styles
    BOLD = '\033[1m'
    DIM = '\033[2m'
    UNDERLINE = '\033[4m'
    
    # Reset
    RESET = '\033[0m'
    
    # Background colors
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'
    BG_BLUE = '\033[44m'

    @staticmethod
    def disable():
        """Disable colors for non-terminal output."""
        for attr in ['RED', 'GREEN', 'YELLOW', 'BLUE', 'MAGENTA', 'CYAN', 'WHITE', 'GREY',
                     'BOLD', 'DIM', 'UNDERLINE', 'RESET', 'BG_RED', 'BG_GREEN', 'BG_YELLOW', 'BG_BLUE']:
            setattr(Colors, attr, '')


# Check if output is to a terminal
if not sys.stdout.isatty():
    Colors.disable()


# =============================================================================
# Status Enum
# =============================================================================

class Status(Enum):
    PASS = "pass"
    FAIL = "fail"
    SKIP = "skip"
    ERROR = "error"
    WARN = "warn"


def status_icon(status: Status) -> str:
    """Get colored status icon."""
    if RICH_AVAILABLE:
        icons = {
            Status.PASS: "[bold green]✓ PASS[/]",
            Status.FAIL: "[bold red]✗ FAIL[/]",
            Status.SKIP: "[bold yellow]○ SKIP[/]",
            Status.ERROR: "[bold red]💥 ERROR[/]",
            Status.WARN: "[bold yellow]⚠ WARN[/]",
        }
    else:
        icons = {
            Status.PASS: f"{Colors.GREEN}✓ PASS{Colors.RESET}",
            Status.FAIL: f"{Colors.RED}✗ FAIL{Colors.RESET}",
            Status.SKIP: f"{Colors.YELLOW}○ SKIP{Colors.RESET}",
            Status.ERROR: f"{Colors.RED}💥 ERROR{Colors.RESET}",
            Status.WARN: f"{Colors.YELLOW}⚠ WARN{Colors.RESET}",
        }
    return icons.get(status, "?")


def status_style(status: Status) -> str:
    """Get Rich style for status."""
    styles = {
        Status.PASS: "bold green",
        Status.FAIL: "bold red",
        Status.SKIP: "bold yellow",
        Status.ERROR: "bold red",
        Status.WARN: "bold yellow",
    }
    return styles.get(status, "white")


# =============================================================================
# Test Result Tracking
# =============================================================================

@dataclass
class TestResult:
    """Container for a single test result."""
    name: str
    status: Status
    duration_ms: float = 0.0
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    max_diff: Optional[float] = None


class TestSuite:
    """Collection of test results with summary reporting."""
    
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.results: List[TestResult] = []
        self.start_time = time.time()
        
    def add_result(self, result: TestResult):
        """Add a test result."""
        self.results.append(result)
        
    def get_summary(self) -> Dict[Status, int]:
        """Get count by status."""
        summary = {s: 0 for s in Status}
        for r in self.results:
            summary[r.status] += 1
        return summary
    
    def all_passed(self) -> bool:
        """Check if all tests passed (or were skipped)."""
        return all(r.status in (Status.PASS, Status.SKIP) for r in self.results)
    
    def print_summary(self):
        """Print a summary table of all results."""
        total_time = time.time() - self.start_time
        summary = self.get_summary()
        
        if RICH_AVAILABLE:
            self._print_summary_rich(total_time, summary)
        else:
            self._print_summary_ansi(total_time, summary)
    
    def _print_summary_rich(self, total_time: float, summary: Dict[Status, int]):
        """Print summary using Rich."""
        # Create table
        table = Table(
            title=f"[bold cyan]{self.name}[/]",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold magenta"
        )
        
        table.add_column("Status", justify="center", width=10)
        table.add_column("Test Name", justify="left", width=50)
        table.add_column("Time", justify="right", width=12)
        table.add_column("Details", justify="left", width=25)
        
        for r in self.results:
            status_text = Text()
            if r.status == Status.PASS:
                status_text.append("✓ PASS", style="bold green")
            elif r.status == Status.FAIL:
                status_text.append("✗ FAIL", style="bold red")
            elif r.status == Status.SKIP:
                status_text.append("○ SKIP", style="bold yellow")
            elif r.status == Status.ERROR:
                status_text.append("💥 ERR", style="bold red")
            else:
                status_text.append("⚠ WARN", style="bold yellow")
            
            time_str = f"{r.duration_ms:.1f}ms" if r.duration_ms > 0 else "-"
            details = r.message[:25] if r.message else ""
            if r.max_diff is not None:
                details = f"diff: {r.max_diff:.2e}"
            
            table.add_row(status_text, r.name[:50], time_str, details)
        
        console.print()
        console.print(table)
        
        # Summary line
        console.print()
        parts = []
        if summary[Status.PASS] > 0:
            parts.append(f"[bold green]{summary[Status.PASS]} passed[/]")
        if summary[Status.FAIL] > 0:
            parts.append(f"[bold red]{summary[Status.FAIL]} failed[/]")
        if summary[Status.SKIP] > 0:
            parts.append(f"[bold yellow]{summary[Status.SKIP]} skipped[/]")
        if summary[Status.ERROR] > 0:
            parts.append(f"[bold red]{summary[Status.ERROR]} errors[/]")
        
        console.print(f"  {' │ '.join(parts)}")
        console.print(f"  [dim]Total: {len(self.results)} tests in {total_time:.2f}s[/]")
        console.print()
        
        # Overall status
        if self.all_passed():
            console.print(Panel("[bold green]ALL TESTS PASSED[/]", style="green"))
        else:
            console.print(Panel("[bold red]SOME TESTS FAILED[/]", style="red"))
    
    def _print_summary_ansi(self, total_time: float, summary: Dict[Status, int]):
        """Print summary using ANSI codes."""
        print()
        print(f"{Colors.BOLD}{'═' * 70}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}{self.name.center(70)}{Colors.RESET}")
        print(f"{Colors.BOLD}{'═' * 70}{Colors.RESET}")
        print()
        
        # Print each result
        for r in self.results:
            icon = "✓" if r.status == Status.PASS else "✗" if r.status == Status.FAIL else "○"
            color = Colors.GREEN if r.status == Status.PASS else Colors.RED if r.status in (Status.FAIL, Status.ERROR) else Colors.YELLOW
            time_str = f"{r.duration_ms:.1f}ms" if r.duration_ms > 0 else "-"
            print(f"  {color}{icon}{Colors.RESET} {r.name[:50]:<50} {time_str:>10}")
        
        # Summary
        print()
        passed = summary[Status.PASS]
        failed = summary[Status.FAIL]
        
        status_line = []
        if passed > 0:
            status_line.append(f"{Colors.GREEN}{passed} passed{Colors.RESET}")
        if failed > 0:
            status_line.append(f"{Colors.RED}{failed} failed{Colors.RESET}")
        
        print(f"  {' | '.join(status_line)}")
        print(f"  Total: {len(self.results)} tests in {total_time:.2f}s")
        print()
        
        if self.all_passed():
            print(f"  {Colors.BG_GREEN}{Colors.WHITE} ALL TESTS PASSED {Colors.RESET}")
        else:
            print(f"  {Colors.BG_RED}{Colors.WHITE} SOME TESTS FAILED {Colors.RESET}")
        print()


# =============================================================================
# Print Helpers
# =============================================================================

def print_header(title: str, subtitle: str = ""):
    """Print a major section header."""
    if RICH_AVAILABLE:
        console.print()
        if subtitle:
            console.print(Panel(f"[bold]{title}[/]\n[dim]{subtitle}[/]", 
                               style="cyan", box=box.DOUBLE))
        else:
            console.print(Panel(f"[bold]{title}[/]", style="cyan", box=box.DOUBLE))
        console.print()
    else:
        print()
        print(f"{Colors.BOLD}{Colors.CYAN}{'═' * 70}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}{title.center(70)}{Colors.RESET}")
        if subtitle:
            print(f"{Colors.DIM}{subtitle.center(70)}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}{'═' * 70}{Colors.RESET}")
        print()


def print_subheader(title: str):
    """Print a subsection header."""
    if RICH_AVAILABLE:
        console.print()
        console.print(f"[bold cyan]▶ {title}[/]")
        console.print(f"[dim]{'─' * 60}[/]")
    else:
        print()
        print(f"{Colors.BOLD}{Colors.CYAN}▶ {title}{Colors.RESET}")
        print(f"{Colors.DIM}{'─' * 60}{Colors.RESET}")


def print_info(message: str):
    """Print info message."""
    if RICH_AVAILABLE:
        console.print(f"  [dim]ℹ {message}[/]")
    else:
        print(f"  {Colors.DIM}ℹ {message}{Colors.RESET}")


def print_success(message: str):
    """Print success message."""
    if RICH_AVAILABLE:
        console.print(f"  [bold green]✓ {message}[/]")
    else:
        print(f"  {Colors.GREEN}✓ {message}{Colors.RESET}")


def print_error(message: str):
    """Print error message."""
    if RICH_AVAILABLE:
        console.print(f"  [bold red]✗ {message}[/]")
    else:
        print(f"  {Colors.RED}✗ {message}{Colors.RESET}")


def print_warning(message: str):
    """Print warning message."""
    if RICH_AVAILABLE:
        console.print(f"  [bold yellow]⚠ {message}[/]")
    else:
        print(f"  {Colors.YELLOW}⚠ {message}{Colors.RESET}")


# =============================================================================
# Comparison Utilities
# =============================================================================

def compare_arrays(arr1, arr2, rtol: float = 1e-5, atol: float = 1e-6, 
                   squeeze: bool = True) -> Tuple[bool, float]:
    """
    Compare two arrays, return (match, max_diff).
    
    Args:
        arr1, arr2: Arrays to compare
        rtol: Relative tolerance
        atol: Absolute tolerance
        squeeze: If True, squeeze arrays before comparison (handles KeOps trailing dim)
    """
    a1 = np.asarray(arr1)
    a2 = np.asarray(arr2)
    
    if squeeze:
        a1 = np.squeeze(a1)
        a2 = np.squeeze(a2)
    
    if a1.shape != a2.shape:
        return False, float('inf')
    
    max_diff = float(np.max(np.abs(a1 - a2)))
    matches = np.allclose(a1, a2, rtol=rtol, atol=atol)
    return matches, max_diff


def format_comparison_result(matches: bool, max_diff: float) -> str:
    """Format comparison result with color."""
    if RICH_AVAILABLE:
        if matches:
            return f"[bold green]✓ Match[/] (diff: {max_diff:.2e})"
        else:
            return f"[bold red]✗ Mismatch[/] (diff: {max_diff:.2e})"
    else:
        if matches:
            return f"{Colors.GREEN}✓ Match{Colors.RESET} (diff: {max_diff:.2e})"
        else:
            return f"{Colors.RED}✗ Mismatch{Colors.RESET} (diff: {max_diff:.2e})"


# =============================================================================
# Benchmark Table
# =============================================================================

def print_benchmark_table(title: str, rows: List[Dict[str, Any]], 
                          columns: List[Tuple[str, str, int]]):
    """
    Print a benchmark results table.
    
    Args:
        title: Table title
        rows: List of dicts with column data
        columns: List of (key, header, width) tuples
    """
    if RICH_AVAILABLE:
        table = Table(title=f"[bold cyan]{title}[/]", box=box.ROUNDED)
        
        for key, header, width in columns:
            table.add_column(header, justify="right" if key.endswith("_ms") else "left", 
                           width=width)
        
        for row in rows:
            values = []
            for key, _, _ in columns:
                val = row.get(key, "")
                if isinstance(val, float):
                    if key.endswith("_ms"):
                        values.append(f"{val:.3f}")
                    elif key == "speedup":
                        if val > 1.1:
                            values.append(f"[green]↑{val:.2f}x[/]")
                        elif val < 0.9:
                            values.append(f"[red]↓{val:.2f}x[/]")
                        else:
                            values.append(f"[yellow]→{val:.2f}x[/]")
                    else:
                        values.append(f"{val:.4f}")
                else:
                    values.append(str(val))
            table.add_row(*values)
        
        console.print(table)
    else:
        # ANSI fallback - simple table
        print(f"\n{Colors.BOLD}{title}{Colors.RESET}")
        print("─" * 80)
        
        # Header
        header = " │ ".join(h.center(w) for _, h, w in columns)
        print(header)
        print("─" * 80)
        
        # Rows
        for row in rows:
            values = []
            for key, _, width in columns:
                val = row.get(key, "")
                if isinstance(val, float):
                    s = f"{val:.3f}" if key.endswith("_ms") else f"{val:.2f}"
                else:
                    s = str(val)
                values.append(s.center(width))
            print(" │ ".join(values))
        print()


# =============================================================================
# Test Runner Helper
# =============================================================================

@contextmanager
def check_context(name: str, suite: TestSuite):
    """Context manager for running a test with timing and error handling."""
    start_time = time.time()
    result = TestResult(name=name, status=Status.PASS)
    
    try:
        yield result
    except AssertionError as e:
        result.status = Status.FAIL
        result.message = str(e)[:100]
    except Exception as e:
        result.status = Status.ERROR
        result.message = f"{type(e).__name__}: {str(e)[:80]}"
    finally:
        result.duration_ms = (time.time() - start_time) * 1000
        suite.add_result(result)


def run_test(name: str, test_fn: Callable, suite: TestSuite, **kwargs) -> TestResult:
    """
    Run a test function and record the result.
    
    Args:
        name: Test name
        test_fn: Function to run (should return (passed, max_diff) or raise exception)
        suite: TestSuite to add result to
        **kwargs: Additional kwargs for test_fn
    
    Returns:
        TestResult
    """
    start_time = time.time()
    
    try:
        result = test_fn(**kwargs)
        
        if isinstance(result, tuple) and len(result) == 2:
            passed, max_diff = result
            status = Status.PASS if passed else Status.FAIL
            tr = TestResult(name=name, status=status, max_diff=max_diff)
        elif isinstance(result, bool):
            status = Status.PASS if result else Status.FAIL
            tr = TestResult(name=name, status=status)
        else:
            tr = TestResult(name=name, status=Status.PASS)
            
    except AssertionError as e:
        tr = TestResult(name=name, status=Status.FAIL, message=str(e)[:100])
    except Exception as e:
        tr = TestResult(name=name, status=Status.ERROR, 
                       message=f"{type(e).__name__}: {str(e)[:80]}")
    
    tr.duration_ms = (time.time() - start_time) * 1000
    suite.add_result(tr)
    
    # Print immediate feedback
    if RICH_AVAILABLE:
        icon = "✓" if tr.status == Status.PASS else "✗" if tr.status == Status.FAIL else "○"
        style = status_style(tr.status)
        detail = f" (diff: {tr.max_diff:.2e})" if tr.max_diff is not None else ""
        console.print(f"  [{style}]{icon}[/] {name}{detail}")
    else:
        icon = "✓" if tr.status == Status.PASS else "✗" if tr.status == Status.FAIL else "○"
        color = Colors.GREEN if tr.status == Status.PASS else Colors.RED if tr.status in (Status.FAIL, Status.ERROR) else Colors.YELLOW
        print(f"  {color}{icon}{Colors.RESET} {name}")
    
    return tr


# =============================================================================
# Environment Info
# =============================================================================

def print_environment_info():
    """Print information about the test environment."""
    import platform
    
    info = {
        "Python": platform.python_version(),
        "Platform": platform.platform(),
    }
    
    # JAX
    try:
        import jax
        info["JAX"] = jax.__version__
        info["JAX Devices"] = str(jax.devices())
    except ImportError:
        info["JAX"] = "Not installed"
    
    # PyTorch
    try:
        import torch
        info["PyTorch"] = torch.__version__
        info["CUDA Available"] = str(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["CUDA Device"] = torch.cuda.get_device_name(0)
    except ImportError:
        info["PyTorch"] = "Not installed"
    
    # NumPy
    try:
        import numpy as np
        info["NumPy"] = np.__version__
    except ImportError:
        pass
    
    if RICH_AVAILABLE:
        table = Table(title="[bold]Environment[/]", box=box.SIMPLE)
        table.add_column("Component", style="cyan")
        table.add_column("Version/Info", style="white")
        
        for key, value in info.items():
            table.add_row(key, value)
        
        console.print(table)
    else:
        print_subheader("Environment")
        for key, value in info.items():
            print(f"  {key}: {value}")
