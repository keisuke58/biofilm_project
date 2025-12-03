#!/usr/bin/env python
"""
Memory profiling tool for biofilm project.

Analyzes memory usage patterns to identify memory leaks and optimization opportunities.

Usage:
    python tools/memory_profile.py --component solver
    python tools/memory_profile.py --component hierarchical --plot
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt

try:
    from memory_profiler import memory_usage, profile
    MEMORY_PROFILER_AVAILABLE = True
except ImportError:
    MEMORY_PROFILER_AVAILABLE = False
    print("Warning: memory_profiler not installed")
    print("Install with: pip install memory_profiler")

from src.config import CONFIG, get_theta_true
from src.solver_newton import BiofilmNewtonSolver
from src.hierarchical import hierarchical_case2


def profile_solver():
    """Profile solver memory usage."""
    print("Profiling: PDE Solver Memory Usage")
    print("=" * 60)

    def run_solver():
        theta = get_theta_true()
        config = CONFIG["M1"]
        phi_init = CONFIG["phi_init_M1"]

        solver = BiofilmNewtonSolver(
            phi_init=phi_init,
            use_numba=True,
            **config
        )

        # Run multiple times to check for leaks
        for _ in range(5):
            solver.run_deterministic(theta, show_progress=False)

        return solver

    if MEMORY_PROFILER_AVAILABLE:
        mem_usage = memory_usage(run_solver, interval=0.1, include_children=True)

        print(f"Baseline memory: {mem_usage[0]:.1f} MB")
        print(f"Peak memory:     {np.max(mem_usage):.1f} MB")
        print(f"Final memory:    {mem_usage[-1]:.1f} MB")
        print(f"Memory increase: {mem_usage[-1] - mem_usage[0]:.1f} MB")

        # Check for memory leak
        if mem_usage[-1] > mem_usage[0] + 10:  # 10 MB threshold
            print("⚠ Warning: Possible memory leak detected!")
        else:
            print("✓ No significant memory leak detected")

        return mem_usage
    else:
        print("Memory profiler not available")
        return None


def profile_hierarchical():
    """Profile hierarchical inference memory usage."""
    print("Profiling: Hierarchical Inference Memory Usage")
    print("=" * 60)

    def run_hierarchical():
        config = CONFIG.copy()
        config["N0"] = 200  # Smaller for profiling
        config["Ndata"] = 10
        return hierarchical_case2(config)

    if MEMORY_PROFILER_AVAILABLE:
        mem_usage = memory_usage(run_hierarchical, interval=0.2, include_children=True)

        print(f"Baseline memory: {mem_usage[0]:.1f} MB")
        print(f"Peak memory:     {np.max(mem_usage):.1f} MB")
        print(f"Final memory:    {mem_usage[-1]:.1f} MB")
        print(f"Memory increase: {mem_usage[-1] - mem_usage[0]:.1f} MB")

        return mem_usage
    else:
        print("Memory profiler not available")
        return None


def plot_memory_profile(mem_usage, title, output_file=None):
    """Plot memory usage over time."""
    if mem_usage is None:
        return

    plt.figure(figsize=(10, 5))
    time_points = np.arange(len(mem_usage)) * 0.1  # Assuming 0.1s intervals
    plt.plot(time_points, mem_usage, linewidth=2)
    plt.xlabel('Time (seconds)', fontsize=12)
    plt.ylabel('Memory Usage (MB)', fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150)
        print(f"✓ Plot saved to: {output_file}")

    plt.show()


def analyze_array_memory():
    """Analyze memory usage of key arrays."""
    print("\nArray Memory Analysis")
    print("=" * 60)

    config = CONFIG["M1"]

    # Estimate memory for key arrays
    maxtimestep = config["maxtimestep"]
    n_species = 4
    n_params = 14

    # State vectors (phi, psi for each species)
    state_size = maxtimestep * n_species * 8 / (1024**2)  # 8 bytes per float64

    # Jacobian matrix (sparse in practice, but estimate full)
    n_state = n_species * 2  # phi and psi for each species
    jacobian_size = n_state * n_state * 8 / (1024**2)

    # TMCMC samples
    N0 = CONFIG.get("N0", 500)
    n_stages = 15
    samples_size = N0 * n_params * n_stages * 8 / (1024**2)

    print(f"Estimated memory usage:")
    print(f"  State vectors:      {state_size:.2f} MB")
    print(f"  Jacobian matrix:    {jacobian_size:.2f} MB")
    print(f"  TMCMC samples:      {samples_size:.2f} MB")
    print(f"  Total (est):        {state_size + jacobian_size + samples_size:.2f} MB")

    print(f"\nMemory optimization tips:")
    print(f"  - Reduce maxtimestep if possible")
    print(f"  - Use sparse Jacobian storage")
    print(f"  - Store only final TMCMC samples")
    print(f"  - Use float32 instead of float64 where appropriate")


def main():
    parser = argparse.ArgumentParser(description="Profile memory usage")
    parser.add_argument("--component",
                       choices=["solver", "hierarchical", "analysis"],
                       default="solver",
                       help="Component to profile")
    parser.add_argument("--plot", action="store_true",
                       help="Generate memory plot")
    parser.add_argument("--output", default="memory_profile.png",
                       help="Output plot filename")

    args = parser.parse_args()

    if not MEMORY_PROFILER_AVAILABLE and args.component != "analysis":
        print("Error: memory_profiler is required for profiling")
        print("Install with: pip install memory_profiler")
        return 1

    mem_usage = None

    if args.component == "solver":
        mem_usage = profile_solver()
        if args.plot and mem_usage:
            plot_memory_profile(mem_usage, "PDE Solver Memory Usage", args.output)

    elif args.component == "hierarchical":
        mem_usage = profile_hierarchical()
        if args.plot and mem_usage:
            plot_memory_profile(mem_usage, "Hierarchical Inference Memory Usage", args.output)

    elif args.component == "analysis":
        analyze_array_memory()

    return 0


if __name__ == "__main__":
    sys.exit(main())
