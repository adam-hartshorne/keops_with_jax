#!/usr/bin/env python3
"""
Inspect the most recently compiled KeOps launcher to verify the ranges fix
"""
import os
import glob
from datetime import datetime

cache_dir = os.path.expanduser("~/.cache/keops2.3/")

print("=" * 70)
print("INSPECTING COMPILED KEOPS LAUNCHER")
print("=" * 70)

if not os.path.exists(cache_dir):
    print(f"\n❌ Cache directory not found: {cache_dir}")
    print("   Have you run any KeOps code yet?")
    exit(1)

# Find all .cu files
cu_files = []
for root, dirs, files in os.walk(cache_dir):
    for file in files:
        if file.endswith('.cu'):
            cu_files.append(os.path.join(root, file))

if not cu_files:
    print(f"\n❌ No .cu files found in {cache_dir}")
    print("   Run a KeOps computation first, then run this script.")
    exit(1)

# Sort by modification time, get most recent
cu_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
most_recent = cu_files[0]

mtime = datetime.fromtimestamp(os.path.getmtime(most_recent))
print(f"\nMost recent .cu file:")
print(f"  Path: {most_recent}")
print(f"  Modified: {mtime}")

# Read the file
with open(most_recent, 'r') as f:
    content = f.read()

# Check if it's a ranges kernel
print("\n" + "=" * 70)
print("KERNEL TYPE")
print("=" * 70)

if "GpuConv1DOnDevice_ranges" in content:
    print("\n✅ This is a RANGES kernel (for batched operations)")
elif "GpuConv1DOnDevice" in content:
    print("\n✅ This is a SIMPLE kernel (for non-batched operations)")
else:
    print("\n❌ Cannot determine kernel type")

# Check for h_ranges code
print("\n" + "=" * 70)
print("RANGES INITIALIZATION CODE")
print("=" * 70)

if "h_ranges[2*b" in content:
    # Extract the relevant lines
    lines = content.split('\n')
    in_ranges_section = False
    context = []

    for i, line in enumerate(lines):
        if 'h_ranges[2*b' in line:
            # Get context: 3 lines before and 3 lines after
            start = max(0, i - 3)
            end = min(len(lines), i + 4)
            context = lines[start:end]
            break

    if context:
        print("\nFound h_ranges initialization:")
        print("-" * 70)
        for line in context:
            print(line)
        print("-" * 70)

        # Check if it has the fix
        ranges_code = '\n'.join(context)
        if 'h_ranges[2*b + 0] = b * ny' in ranges_code:
            print("\n✅ RANGES FIX IS PRESENT!")
            print("   Code correctly calculates: h_ranges[2*b + 0] = b * ny")
        elif 'h_ranges[2*b + 0] = 0' in ranges_code:
            print("\n❌ RANGES FIX IS MISSING!")
            print("   Code incorrectly uses: h_ranges[2*b + 0] = 0")
            print("   This will cause all batches to read Batch 0 data!")
        else:
            print("\n⚠️  Cannot determine if fix is present")
else:
    print("\n⚠️  No h_ranges initialization found")
    print("   This might be a non-batched kernel")

# Additional check: Look for the kernel launch
print("\n" + "=" * 70)
print("KERNEL LAUNCH")
print("=" * 70)

if "GpuConv1DOnDevice_ranges<<<" in content:
    print("\n✅ Found batched kernel launch")
    # Extract the line
    for line in content.split('\n'):
        if "GpuConv1DOnDevice_ranges<<<" in line:
            print(f"   {line.strip()}")
            break
elif "GpuConv1DOnDevice<<<" in content:
    print("\n✅ Found simple kernel launch")
    for line in content.split('\n'):
        if "GpuConv1DOnDevice<<<" in line and "GpuConv1DOnDevice_ranges<<<" not in line:
            print(f"   {line.strip()}")
            break

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

# Final verdict
has_ranges_kernel = "GpuConv1DOnDevice_ranges" in content
has_ranges_code = "h_ranges[2*b" in content
has_fix = "h_ranges[2*b + 0] = b * ny" in content

if has_ranges_kernel and has_fix:
    print("\n✅ Everything looks CORRECT!")
    print("   - Ranges kernel is being used")
    print("   - Ranges fix is present in compiled code")
    print("\n   If tests are still failing, the issue is elsewhere.")
elif has_ranges_kernel and not has_fix:
    print("\n❌ PROBLEM FOUND!")
    print("   - Ranges kernel is being used")
    print("   - But ranges fix is NOT in the compiled code")
    print("\n   Solution:")
    print("   1. Clear cache: rm -rf ~/.cache/keops2.3/")
    print("   2. Re-run your test")
elif not has_ranges_kernel:
    print("\n✅ Simple (non-batched) kernel")
    print("   - No ranges code needed")
    print("   - This is normal for 2D inputs")

print()