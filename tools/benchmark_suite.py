#!/usr/bin/env python
"""
Comprehensive benchmark suite for biofilm project.

Tracks performance across versions and configurations to detect regressions.

Usage:
    python tools/benchmark_suite.py --output benchmarks.json
    python tools/benchmark_suite.py --compare baseline.json
"""
import argparse
import json
import time
from datetime import datetime
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.config import CONFIG, get_model_config, get_theta_true
from src.solver_newton import BiofilmNewtonSolver
from src.tsm import BiofilmTSM
from src.tmcmc import tmcmc


class BenchmarkSuite:
    """Comprehensive benchmark suite."""

    def __init__(self):
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "DEBUG": CONFIG.get("DEBUG", False),
                "N0": CONFIG.get("N0", 500),
            },
            "benchmarks": {}
        }

    def benchmark_solver(self, iterations=5):
        """Benchmark PDE solver."""
        print("Benchmarking: PDE Solver...")

        theta = get_theta_true()
        config = get_model_config("M1")
        phi_init = config.pop("phi_init", 0.02)

        solver = BiofilmNewtonSolver(
            phi_init=phi_init,
            use_numba=True,
            **config
        )

        # Warmup
        solver.run_deterministic(theta, show_progress=False)

        # Benchmark
        times = []
        for _ in range(iterations):
            t0 = time.time()
            solver.run_deterministic(theta, show_progress=False)
            times.append(time.time() - t0)

        self.results["benchmarks"]["solver"] = {
            "mean": float(np.mean(times)),
            "std": float(np.std(times)),
            "min": float(np.min(times)),
            "max": float(np.max(times)),
            "iterations": iterations
        }

        print(f"  Mean: {np.mean(times):.3f}s")

    def benchmark_tsm(self, iterations=3):
        """Benchmark TSM sensitivity computation."""
        print("Benchmarking: TSM...")

        theta = get_theta_true()
        config = get_model_config("M1")
        phi_init = config.pop("phi_init", 0.02)

        solver = BiofilmNewtonSolver(
            phi_init=phi_init,
            use_numba=True,
            **config
        )

        tsm = BiofilmTSM(
            solver,
            cov_rel=CONFIG["cov_rel"],
            active_theta_indices=config.get("theta_indices"),
            use_analytical=True
        )

        # Warmup
        tsm.solve_tsm(theta)

        # Benchmark
        times = []
        for _ in range(iterations):
            t0 = time.time()
            tsm.solve_tsm(theta)
            times.append(time.time() - t0)

        self.results["benchmarks"]["tsm"] = {
            "mean": float(np.mean(times)),
            "std": float(np.std(times)),
            "min": float(np.min(times)),
            "max": float(np.max(times)),
            "iterations": iterations
        }

        print(f"  Mean: {np.mean(times):.3f}s")

    def benchmark_tmcmc_small(self, iterations=3):
        """Benchmark TMCMC on small problem."""
        print("Benchmarking: TMCMC (small)...")

        def log_likelihood(theta):
            theta_true = np.array([1.0, 2.0])
            if theta.ndim == 1:
                diff = theta - theta_true
                return -0.5 * np.sum(diff**2)
            else:
                diff = theta - theta_true
                return -0.5 * np.sum(diff**2, axis=1)

        def log_prior(theta):
            if theta.ndim == 1:
                return 0.0 if np.all((-5 <= theta) & (theta <= 5)) else -np.inf
            else:
                valid = np.all((-5 <= theta) & (theta <= 5), axis=1)
                logp = np.full(len(theta), -np.inf)
                logp[valid] = 0.0
                return logp

        n_samples = 100
        theta_init = np.random.uniform(-5, 5, size=(n_samples, 2))

        times = []
        for _ in range(iterations):
            t0 = time.time()
            result = tmcmc(
                log_likelihood=log_likelihood,
                log_prior=log_prior,
                theta_init_samples=theta_init.copy(),
                n_stages=10,
                show_progress=False,
            )
            times.append(time.time() - t0)

        self.results["benchmarks"]["tmcmc_small"] = {
            "mean": float(np.mean(times)),
            "std": float(np.std(times)),
            "min": float(np.min(times)),
            "max": float(np.max(times)),
            "iterations": iterations,
            "n_samples": n_samples
        }

        print(f"  Mean: {np.mean(times):.3f}s")

    def benchmark_memory_usage(self):
        """Benchmark memory usage (requires memory_profiler)."""
        try:
            from memory_profiler import memory_usage

            print("Benchmarking: Memory Usage...")

            def run_solver():
                theta = get_theta_true()
                config = get_model_config("M1")
                phi_init = config.pop("phi_init", 0.02)
                solver = BiofilmNewtonSolver(
                    phi_init=phi_init,
                    use_numba=True,
                    **config
                )
                solver.run_deterministic(theta, show_progress=False)

            mem_usage = memory_usage(run_solver, interval=0.1)

            self.results["benchmarks"]["memory"] = {
                "peak_mb": float(np.max(mem_usage)),
                "mean_mb": float(np.mean(mem_usage)),
                "baseline_mb": float(mem_usage[0])
            }

            print(f"  Peak: {np.max(mem_usage):.1f} MB")

        except ImportError:
            print("  Skipped (memory_profiler not installed)")
            self.results["benchmarks"]["memory"] = None

    def save(self, filename):
        """Save benchmark results to JSON."""
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n✓ Benchmarks saved to: {filename}")

    def compare(self, baseline_file):
        """Compare current results with baseline."""
        with open(baseline_file) as f:
            baseline = json.load(f)

        print("\n" + "=" * 70)
        print("BENCHMARK COMPARISON")
        print("=" * 70)
        print(f"Baseline: {baseline['timestamp']}")
        print(f"Current:  {self.results['timestamp']}")
        print("-" * 70)

        for name in self.results["benchmarks"]:
            if name in baseline["benchmarks"] and name in self.results["benchmarks"]:
                base_bench = baseline["benchmarks"][name]
                curr_bench = self.results["benchmarks"][name]

                if base_bench is None or curr_bench is None:
                    continue

                base_mean = base_bench.get("mean")
                curr_mean = curr_bench.get("mean")

                if base_mean and curr_mean:
                    speedup = base_mean / curr_mean
                    percent_change = (curr_mean - base_mean) / base_mean * 100

                    status = "✓" if speedup >= 1.0 else "⚠"
                    print(f"{name:20s}: {base_mean:.3f}s → {curr_mean:.3f}s "
                          f"({speedup:.2f}x, {percent_change:+.1f}%) {status}")

        print("-" * 70)


def main():
    parser = argparse.ArgumentParser(description="Run benchmark suite")
    parser.add_argument("--output", default="benchmark_results.json",
                       help="Output JSON file")
    parser.add_argument("--compare", default=None,
                       help="Baseline JSON file to compare against")
    parser.add_argument("--quick", action="store_true",
                       help="Quick benchmark (fewer iterations)")

    args = parser.parse_args()

    suite = BenchmarkSuite()

    iters = 2 if args.quick else 5

    print("=" * 70)
    print("BIOFILM PROJECT BENCHMARK SUITE")
    print("=" * 70)
    print(f"Mode: {'Quick' if args.quick else 'Full'}")
    print()

    suite.benchmark_solver(iterations=iters)
    suite.benchmark_tsm(iterations=max(2, iters // 2))
    suite.benchmark_tmcmc_small(iterations=max(2, iters // 2))
    suite.benchmark_memory_usage()

    suite.save(args.output)

    if args.compare and Path(args.compare).exists():
        suite.compare(args.compare)

    print("\nBenchmark complete!")
    print(f"Run 'python {__file__} --compare {args.output}' to compare future runs")


if __name__ == "__main__":
    main()
