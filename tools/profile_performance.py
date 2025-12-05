#!/usr/bin/env python
"""
Performance profiling script for biofilm project.

This script profiles the performance of key components:
- BiofilmNewtonSolver: PDE solver
- BiofilmTSM: Time-separated mechanics
- TMCMC: Sampling algorithm
- Full hierarchical inference

Usage:
    python tools/profile_performance.py [--component all|solver|tsm|tmcmc|hierarchical]
    python tools/profile_performance.py --profile  # Run with cProfile
"""
import argparse
import time
import cProfile
import pstats
import io
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.config import CONFIG, get_model_config, get_theta_true
from src.solver_newton import BiofilmNewtonSolver
from src.tsm import BiofilmTSM
from src.tmcmc import tmcmc
from src.hierarchical import hierarchical_case2


def profile_solver(n_runs=5):
    """Profile BiofilmNewtonSolver performance"""
    print("\n" + "=" * 70)
    print("PROFILING: BiofilmNewtonSolver")
    print("=" * 70)

    theta = get_theta_true()
    config_M1 = get_model_config("M1")
    phi_init = config_M1.pop("phi_init", 0.02)

    solver = BiofilmNewtonSolver(
        phi_init=phi_init,
        use_numba=True,
        **config_M1
    )

    # Warmup (JIT compilation)
    print("Warming up (JIT compilation)...")
    solver.run_deterministic(theta, show_progress=False)

    # Benchmark
    print(f"Running {n_runs} iterations...")
    times = []
    for i in range(n_runs):
        t0 = time.time()
        t, g = solver.run_deterministic(theta, show_progress=False)
        elapsed = time.time() - t0
        times.append(elapsed)
        print(f"  Run {i+1}: {elapsed:.3f} s")

    mean_time = np.mean(times)
    std_time = np.std(times)

    print(f"\nResults:")
    print(f"  Mean time:   {mean_time:.3f} ± {std_time:.3f} s")
    print(f"  Timesteps:   {config_M1['maxtimestep']}")
    print(f"  dt:          {config_M1['dt']}")
    print(f"  Time/step:   {mean_time / config_M1['maxtimestep'] * 1000:.3f} ms")

    return {"component": "solver", "mean": mean_time, "std": std_time}


def profile_tsm(n_runs=5):
    """Profile BiofilmTSM performance"""
    print("\n" + "=" * 70)
    print("PROFILING: BiofilmTSM (Time-Separated Mechanics)")
    print("=" * 70)

    theta = get_theta_true()
    config_M1 = get_model_config("M1")
    phi_init = config_M1.pop("phi_init", 0.02)

    solver = BiofilmNewtonSolver(
        phi_init=phi_init,
        use_numba=True,
        **config_M1
    )

    active_theta_indices = config_M1.get("theta_indices")

    tsm = BiofilmTSM(
        solver,
        cov_rel=CONFIG["cov_rel"],
        active_theta_indices=active_theta_indices,
        use_analytical=True
    )

    # Warmup
    print("Warming up...")
    tsm.solve_tsm(theta)

    # Benchmark
    print(f"Running {n_runs} iterations...")
    times = []
    for i in range(n_runs):
        t0 = time.time()
        result = tsm.solve_tsm(theta)
        elapsed = time.time() - t0
        times.append(elapsed)
        print(f"  Run {i+1}: {elapsed:.3f} s")

    mean_time = np.mean(times)
    std_time = np.std(times)

    print(f"\nResults:")
    print(f"  Mean time:        {mean_time:.3f} ± {std_time:.3f} s")
    if active_theta_indices is not None:
        print(f"  Active params:    {len(active_theta_indices)}")
    print(f"  Analytical sens:  Yes")

    return {"component": "tsm", "mean": mean_time, "std": std_time}


def profile_tmcmc(n_runs=2):
    """Profile TMCMC sampling"""
    print("\n" + "=" * 70)
    print("PROFILING: TMCMC Sampling")
    print("=" * 70)

    # Simple 2D test problem
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

    n_samples = 200
    theta_init = np.random.uniform(-5, 5, size=(n_samples, 2))

    print(f"Running {n_runs} iterations (n_samples={n_samples})...")
    times = []
    for i in range(n_runs):
        t0 = time.time()
        result = tmcmc(
            log_likelihood=log_likelihood,
            log_prior=log_prior,
            theta_init_samples=theta_init.copy(),
            n_stages=10,
            show_progress=False,
        )
        elapsed = time.time() - t0
        times.append(elapsed)
        print(f"  Run {i+1}: {elapsed:.3f} s (converged: {result.converged}, stages: {len(result.beta_schedule)})")

    mean_time = np.mean(times)
    std_time = np.std(times)

    print(f"\nResults:")
    print(f"  Mean time:   {mean_time:.3f} ± {std_time:.3f} s")
    print(f"  N samples:   {n_samples}")

    return {"component": "tmcmc", "mean": mean_time, "std": std_time}


def profile_hierarchical(n_runs=1):
    """Profile full hierarchical inference"""
    print("\n" + "=" * 70)
    print("PROFILING: Full Hierarchical Inference (Case II)")
    print("=" * 70)

    # Use debug config for faster profiling
    config = CONFIG.copy()
    config["DEBUG"] = True
    config["N0"] = 200  # Smaller for profiling
    config["Ndata"] = 10

    print(f"Running {n_runs} iteration(s) (N0={config['N0']}, Ndata={config['Ndata']})...")
    times = []
    for i in range(n_runs):
        t0 = time.time()
        results = hierarchical_case2(config)
        elapsed = time.time() - t0
        times.append(elapsed)
        print(f"  Run {i+1}: {elapsed:.1f} s")
        print(f"    M1 converged: {results.tmcmc_M1.converged}")
        print(f"    M2 converged: {results.tmcmc_M2.converged}")
        print(f"    M3 converged: {results.tmcmc_M3.converged}")

    mean_time = np.mean(times)
    std_time = np.std(times) if n_runs > 1 else 0.0

    print(f"\nResults:")
    print(f"  Mean time:   {mean_time:.1f} ± {std_time:.1f} s")
    print(f"  RMSE:        {np.sqrt(np.mean((results.theta_final - get_theta_true())**2)):.4f}")

    return {"component": "hierarchical", "mean": mean_time, "std": std_time}


def main():
    parser = argparse.ArgumentParser(description="Profile biofilm project performance")
    parser.add_argument(
        "--component",
        choices=["all", "solver", "tsm", "tmcmc", "hierarchical"],
        default="all",
        help="Component to profile"
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Use cProfile for detailed profiling"
    )
    parser.add_argument(
        "--output",
        default="profile_stats.txt",
        help="Output file for cProfile stats"
    )

    args = parser.parse_args()

    results = []

    if args.profile:
        print("Running with cProfile...")
        profiler = cProfile.Profile()
        profiler.enable()

    # Run selected profiles
    if args.component in ["all", "solver"]:
        results.append(profile_solver(n_runs=5))

    if args.component in ["all", "tsm"]:
        results.append(profile_tsm(n_runs=5))

    if args.component in ["all", "tmcmc"]:
        results.append(profile_tmcmc(n_runs=2))

    if args.component in ["all", "hierarchical"]:
        results.append(profile_hierarchical(n_runs=1))

    if args.profile:
        profiler.disable()

        # Print stats
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
        ps.print_stats(30)  # Top 30 functions

        stats_str = s.getvalue()
        print("\n" + "=" * 70)
        print("DETAILED PROFILE (cProfile - Top 30 functions)")
        print("=" * 70)
        print(stats_str)

        # Save to file
        with open(args.output, 'w') as f:
            f.write(stats_str)
        print(f"\nDetailed profile saved to: {args.output}")

    # Summary
    print("\n" + "=" * 70)
    print("PERFORMANCE SUMMARY")
    print("=" * 70)
    for result in results:
        print(f"{result['component']:15s}: {result['mean']:8.3f} ± {result['std']:.3f} s")

    print("\nTips for optimization:")
    print("  - Solver: Already using Numba JIT for maximum speed")
    print("  - TSM: Analytical sensitivities are faster than finite differences")
    print("  - TMCMC: Consider parallel likelihood evaluations for large N")
    print("  - Full: Reduce N0 or Ndata for faster convergence testing")


if __name__ == "__main__":
    main()
