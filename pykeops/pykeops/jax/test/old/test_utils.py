"""
KeOps JAX Test Utilities
========================
Shared utilities for test formatting, color output, and table rendering.
"""

import sys
import time
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum


# =============================================================================
# ANSI Color Codes
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
        Colors.RED = ''
        Colors.GREEN = ''
        Colors.YELLOW = ''
        Colors.BLUE = ''
        Colors.MAGENTA = ''
        Colors.CYAN = ''
        Colors.WHITE = ''
        Colors.BOLD = ''
        Colors.DIM = ''
        Colors.UNDERLINE = ''
        Colors.RESET = ''
        Colors.BG_RED = ''
        Colors.BG_GREEN = ''
        Colors.BG_YELLOW = ''
        Colors.BG_BLUE = ''


# Check if output is to a terminal
if not sys.stdout.isatty():
    Colors.disable()


# =============================================================================
# Status Indicators
# =============================================================================

class Status(Enum):
    PASS = "pass"
    FAIL = "fail"
    SKIP = "skip"
    ERROR = "error"
    WARN = "warn"


def status_icon(status: Status) -> str:
    """Get colored status icon."""
    icons = {
        Status.PASS: f"{Colors.GREEN}✓ PASS{Colors.RESET}",
        Status.FAIL: f"{Colors.RED}✗ FAIL{Colors.RESET}",
        Status.SKIP: f"{Colors.YELLOW}○ SKIP{Colors.RESET}",
        Status.ERROR: f"{Colors.RED}💥 ERROR{Colors.RESET}",
        Status.WARN: f"{Colors.YELLOW}⚠ WARN{Colors.RESET}",
    }
    return icons.get(status, "?")


def status_badge(status: Status) -> str:
    """Get colored status badge for tables."""
    badges = {
        Status.PASS: f"{Colors.BG_GREEN}{Colors.WHITE} PASS {Colors.RESET}",
        Status.FAIL: f"{Colors.BG_RED}{Colors.WHITE} FAIL {Colors.RESET}",
        Status.SKIP: f"{Colors.BG_YELLOW}{Colors.WHITE} SKIP {Colors.RESET}",
        Status.ERROR: f"{Colors.BG_RED}{Colors.WHITE} ERR  {Colors.RESET}",
        Status.WARN: f"{Colors.BG_YELLOW}{Colors.WHITE} WARN {Colors.RESET}",
    }
    return badges.get(status, "  ?  ")


# =============================================================================
# Speed Formatting
# =============================================================================

def color_speed(time_ms: float, reference_ms: Optional[float] = None) -> str:
    """Color-code a timing value based on speed or comparison."""
    if reference_ms is not None:
        ratio = time_ms / reference_ms
        if ratio < 0.9:  # Faster
            color = Colors.GREEN
        elif ratio < 1.1:  # Similar
            color = Colors.YELLOW
        else:  # Slower
            color = Colors.RED
        return f"{color}{time_ms:.3f}{Colors.RESET}"
    else:
        # Standalone - color by absolute time
        if time_ms < 1.0:
            color = Colors.GREEN
        elif time_ms < 10.0:
            color = Colors.YELLOW
        else:
            color = Colors.RED
        return f"{color}{time_ms:.3f}{Colors.RESET}"


def format_speedup(speedup: float) -> str:
    """Format speedup value with color."""
    if speedup > 1.1:
        color = Colors.GREEN
        arrow = "↑"
    elif speedup < 0.9:
        color = Colors.RED
        arrow = "↓"
    else:
        color = Colors.YELLOW
        arrow = "→"
    return f"{color}{arrow} {speedup:.2f}x{Colors.RESET}"


def format_efficiency(efficiency_pct: float) -> str:
    """Format scaling efficiency percentage."""
    if efficiency_pct >= 80:
        color = Colors.GREEN
    elif efficiency_pct >= 50:
        color = Colors.YELLOW
    else:
        color = Colors.RED
    return f"{color}{efficiency_pct:.0f}%{Colors.RESET}"


# =============================================================================
# ASCII Table Rendering
# =============================================================================

@dataclass
class TableColumn:
    """Definition for a table column."""
    header: str
    width: int
    align: str = 'left'  # 'left', 'right', 'center'


class ASCIITable:
    """Simple ASCII table renderer with colors."""
    
    # Box drawing characters
    CORNER_TL = '┌'
    CORNER_TR = '┐'
    CORNER_BL = '└'
    CORNER_BR = '┘'
    HORIZONTAL = '─'
    VERTICAL = '│'
    T_DOWN = '┬'
    T_UP = '┴'
    T_RIGHT = '├'
    T_LEFT = '┤'
    CROSS = '┼'
    
    # Double line variants for headers
    DOUBLE_HORIZONTAL = '═'
    
    def __init__(self, columns: List[TableColumn], title: Optional[str] = None):
        self.columns = columns
        self.title = title
        self.rows: List[List[str]] = []
        
    def add_row(self, values: List[Any]):
        """Add a row of values."""
        self.rows.append([str(v) for v in values])
        
    def add_separator(self):
        """Add a separator row."""
        self.rows.append(None)  # None indicates separator
        
    def _strip_ansi(self, text: str) -> str:
        """Remove ANSI codes for width calculation."""
        import re
        return re.sub(r'\033\[[0-9;]*m', '', text)
    
    def _pad(self, text: str, width: int, align: str) -> str:
        """Pad text to width, accounting for ANSI codes."""
        visible_len = len(self._strip_ansi(text))
        padding = width - visible_len
        if padding <= 0:
            return text
        if align == 'left':
            return text + ' ' * padding
        elif align == 'right':
            return ' ' * padding + text
        else:  # center
            left = padding // 2
            right = padding - left
            return ' ' * left + text + ' ' * right
    
    def _horizontal_line(self, left: str, mid: str, right: str) -> str:
        """Create a horizontal line."""
        parts = [left]
        for i, col in enumerate(self.columns):
            parts.append(self.HORIZONTAL * (col.width + 2))
            if i < len(self.columns) - 1:
                parts.append(mid)
        parts.append(right)
        return ''.join(parts)
    
    def render(self) -> str:
        """Render the table as a string."""
        lines = []
        
        # Title
        if self.title:
            total_width = sum(c.width + 3 for c in self.columns) + 1
            lines.append('')
            lines.append(f"{Colors.BOLD}{Colors.CYAN}{self.title.center(total_width)}{Colors.RESET}")
            lines.append('')
        
        # Top border
        lines.append(self._horizontal_line(self.CORNER_TL, self.T_DOWN, self.CORNER_TR))
        
        # Header row
        header_parts = [self.VERTICAL]
        for col in self.columns:
            header_parts.append(f" {Colors.BOLD}{self._pad(col.header, col.width, 'center')}{Colors.RESET} ")
            header_parts.append(self.VERTICAL)
        lines.append(''.join(header_parts))
        
        # Header separator
        lines.append(self._horizontal_line(self.T_RIGHT, self.CROSS, self.T_LEFT))
        
        # Data rows
        for row in self.rows:
            if row is None:
                # Separator
                lines.append(self._horizontal_line(self.T_RIGHT, self.CROSS, self.T_LEFT))
            else:
                row_parts = [self.VERTICAL]
                for i, (value, col) in enumerate(zip(row, self.columns)):
                    row_parts.append(f" {self._pad(value, col.width, col.align)} ")
                    row_parts.append(self.VERTICAL)
                lines.append(''.join(row_parts))
        
        # Bottom border
        lines.append(self._horizontal_line(self.CORNER_BL, self.T_UP, self.CORNER_BR))
        
        return '\n'.join(lines)
    
    def print(self):
        """Print the table."""
        print(self.render())


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
    details: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.details is None:
            self.details = {}


class TestSuite:
    """Collection of test results with summary reporting."""
    
    def __init__(self, name: str):
        self.name = name
        self.results: List[TestResult] = []
        self.start_time = time.time()
        
    def add_result(self, result: TestResult):
        """Add a test result."""
        self.results.append(result)
        
    def get_summary(self) -> Dict[str, int]:
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
        
        # Header
        print()
        print(f"{Colors.BOLD}{'═' * 70}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}{self.name.center(70)}{Colors.RESET}")
        print(f"{Colors.BOLD}{'═' * 70}{Colors.RESET}")
        print()
        
        # Results table
        table = ASCIITable([
            TableColumn("Status", 8, 'center'),
            TableColumn("Test Name", 45, 'left'),
            TableColumn("Time", 10, 'right'),
        ])
        
        for r in self.results:
            time_str = f"{r.duration_ms:.1f}ms" if r.duration_ms > 0 else "-"
            status_str = status_icon(r.status).split()[0]  # Just the icon
            table.add_row([status_str, r.name[:45], time_str])
            
        table.print()
        
        # Summary line
        print()
        passed = summary[Status.PASS]
        failed = summary[Status.FAIL]
        skipped = summary[Status.SKIP]
        errors = summary[Status.ERROR]
        total = len(self.results)
        
        status_line = []
        if passed > 0:
            status_line.append(f"{Colors.GREEN}{passed} passed{Colors.RESET}")
        if failed > 0:
            status_line.append(f"{Colors.RED}{failed} failed{Colors.RESET}")
        if skipped > 0:
            status_line.append(f"{Colors.YELLOW}{skipped} skipped{Colors.RESET}")
        if errors > 0:
            status_line.append(f"{Colors.RED}{errors} errors{Colors.RESET}")
            
        print(f"  {' | '.join(status_line)}")
        print(f"  Total: {total} tests in {total_time:.2f}s")
        print()
        
        # Overall status
        if self.all_passed():
            print(f"  {Colors.BG_GREEN}{Colors.WHITE} ALL TESTS PASSED {Colors.RESET}")
        else:
            print(f"  {Colors.BG_RED}{Colors.WHITE} SOME TESTS FAILED {Colors.RESET}")
        print()


# =============================================================================
# Progress Display
# =============================================================================

class ProgressBar:
    """Simple progress bar for benchmarks."""
    
    def __init__(self, total: int, description: str = "", width: int = 40):
        self.total = total
        self.current = 0
        self.description = description
        self.width = width
        
    def update(self, n: int = 1):
        """Update progress by n steps."""
        self.current += n
        self._display()
        
    def _display(self):
        """Display the progress bar."""
        pct = self.current / self.total
        filled = int(self.width * pct)
        bar = '█' * filled + '░' * (self.width - filled)
        print(f"\r  {self.description}: [{Colors.CYAN}{bar}{Colors.RESET}] "
              f"{self.current}/{self.total}", end='', flush=True)
        if self.current >= self.total:
            print()  # New line when done


class Spinner:
    """Simple spinner for operations of unknown duration."""
    
    CHARS = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
    
    def __init__(self, description: str = ""):
        self.description = description
        self.idx = 0
        
    def spin(self):
        """Display next spinner frame."""
        char = self.CHARS[self.idx % len(self.CHARS)]
        print(f"\r  {Colors.CYAN}{char}{Colors.RESET} {self.description}", end='', flush=True)
        self.idx += 1
        
    def done(self, message: str = "Done"):
        """Display completion."""
        print(f"\r  {Colors.GREEN}✓{Colors.RESET} {self.description}: {message}")


# =============================================================================
# Section Headers
# =============================================================================

def print_header(title: str, width: int = 70):
    """Print a major section header."""
    print()
    print(f"{Colors.BOLD}{Colors.CYAN}{'=' * width}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{title.center(width)}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'=' * width}{Colors.RESET}")
    print()


def print_subheader(title: str, width: int = 70):
    """Print a subsection header."""
    print()
    print(f"{Colors.BOLD}{title}{Colors.RESET}")
    print(f"{Colors.DIM}{'-' * min(len(title) + 10, width)}{Colors.RESET}")


def print_test_start(name: str):
    """Print test start indicator."""
    print(f"\n  {Colors.DIM}▶ {name}...{Colors.RESET}", end='', flush=True)


def print_test_end(status: Status, message: str = ""):
    """Print test end status."""
    icon = "✓" if status == Status.PASS else "✗" if status == Status.FAIL else "○"
    color = Colors.GREEN if status == Status.PASS else Colors.RED if status in (Status.FAIL, Status.ERROR) else Colors.YELLOW
    suffix = f" ({message})" if message else ""
    print(f"\r  {color}{icon}{Colors.RESET} {suffix}")


# =============================================================================
# Benchmark Result Display
# =============================================================================

def print_benchmark_comparison(
    name: str,
    jax_time_ms: float,
    torch_time_ms: float,
    jax_std_ms: float = 0,
    torch_std_ms: float = 0
):
    """Print a single benchmark comparison line."""
    speedup = torch_time_ms / jax_time_ms if jax_time_ms > 0 else 0
    
    jax_str = color_speed(jax_time_ms, torch_time_ms)
    torch_str = f"{torch_time_ms:.3f}"
    speedup_str = format_speedup(speedup)
    
    # Format with optional std deviation
    if jax_std_ms > 0:
        jax_str += f" ±{jax_std_ms:.2f}"
    if torch_std_ms > 0:
        torch_str += f" ±{torch_std_ms:.2f}"
    
    print(f"  {name:<30} JAX: {jax_str:>15}ms  PyTorch: {torch_str:>12}ms  {speedup_str}")


# =============================================================================
# Validation Helpers
# =============================================================================

def compare_arrays(arr1, arr2, rtol: float = 1e-5, atol: float = 1e-6) -> Tuple[bool, float]:
    """Compare two arrays, return (match, max_diff)."""
    import numpy as np
    a1 = np.asarray(arr1)
    a2 = np.asarray(arr2)
    
    if a1.shape != a2.shape:
        return False, float('inf')
    
    max_diff = float(np.max(np.abs(a1 - a2)))
    matches = np.allclose(a1, a2, rtol=rtol, atol=atol)
    return matches, max_diff


def format_comparison_result(matches: bool, max_diff: float) -> str:
    """Format comparison result with color."""
    if matches:
        return f"{Colors.GREEN}✓ Match{Colors.RESET} (max diff: {max_diff:.2e})"
    else:
        return f"{Colors.RED}✗ Mismatch{Colors.RESET} (max diff: {max_diff:.2e})"
