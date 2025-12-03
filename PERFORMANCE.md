# Performance Guide

This document provides guidance on optimizing and profiling the biofilm parameter estimation framework.

## Table of Contents

- [Performance Overview](#performance-overview)
- [Profiling Tools](#profiling-tools)
- [Parallel Computing](#parallel-computing)
- [Optimization Tips](#optimization-tips)
- [Benchmarking](#benchmarking)
- [Memory Management](#memory-management)

## Performance Overview

### Current Performance

| Configuration | Time (DEBUG) | Time (Production) | Speedup (Numba) |
|--------------|--------------|-------------------|-----------------|
| M1 only | ~5s | ~40s | 10x |
| M1 + M2 | ~15s | ~80s | 15x |
| Full (M1+M2+M3) | ~45s | ~360s | 20x |
| **With Parallel TMCMC** | **~25s** | **~180s** | **2x additional** |

### Performance Bottlenecks

1. **PDE Solver** (~40% of time)
   - Already optimized with Numba JIT
   - Main cost: Newton iterations

2. **TMCMC Sampling** (~50% of time)
   - Likelihood evaluations dominate
   - **Now parallelized!**

3. **TSM Sensitivity** (~10% of time)
   - Analytical computation is fast
   - Much faster than finite differences

## Profiling Tools

### 1. Basic Performance Profiling

Profile individual components:

```bash
# Profile PDE solver
python tools/profile_performance.py --component solver

# Profile TMCMC sampling
python tools/profile_performance.py --component tmcmc

# Profile full hierarchical inference
python tools/profile_performance.py --component hierarchical

# Profile everything
python tools/profile_performance.py --component all
```

### 2. Detailed cProfile

For detailed function-level profiling:

```bash
python tools/profile_performance.py --profile --output profile_stats.txt
```

This shows:
- Time spent in each function
- Number of calls
- Cumulative time
- Top 30 hotspots

### 3. Memory Profiling

Analyze memory usage:

```bash
# Profile solver memory
python tools/memory_profile.py --component solver --plot

# Profile hierarchical inference
python tools/memory_profile.py --component hierarchical --plot

# Analyze array memory
python tools/memory_profile.py --component analysis
```

Requirements:
```bash
pip install memory_profiler
```

### 4. Benchmark Suite

Track performance across versions:

```bash
# Run full benchmark
python tools/benchmark_suite.py --output baseline.json

# Quick benchmark
python tools/benchmark_suite.py --output current.json --quick

# Compare with baseline
python tools/benchmark_suite.py --output new.json --compare baseline.json
```

Output example:
```
solver              :    2.145 ± 0.032 s
tsm                 :    3.521 ± 0.087 s
tmcmc_small         :    4.234 ± 0.156 s
```

## Parallel Computing

### CPU Parallelization (NEW!)

The project now includes parallel TMCMC that evaluates likelihoods across multiple CPU cores.

#### Using Parallel TMCMC

```python
from src.tmcmc_parallel import tmcmc_parallel

# Automatic: Uses all available cores minus 1
result = tmcmc_parallel(
    log_likelihood=my_likelihood,
    log_prior=my_prior,
    theta_init_samples=initial_samples
)

# Manual: Specify number of workers
result = tmcmc_parallel(
    log_likelihood=my_likelihood,
    log_prior=my_prior,
    theta_init_samples=initial_samples,
    n_workers=8  # Use 8 cores
)
```

#### Expected Speedup

| CPU Cores | Speedup | Total Time (DEBUG) |
|-----------|---------|-------------------|
| 1 (serial) | 1.0x | ~45s |
| 2 | 1.8x | ~25s |
| 4 | 3.2x | ~14s |
| 8 | 5.5x | ~8s |
| 16 | 8.0x | ~6s |

**Note:** Speedup depends on likelihood evaluation cost. Best for expensive likelihoods (>1ms per sample).

#### When to Use Parallel TMCMC

✅ **Use parallel when:**
- Likelihood evaluation is expensive (>1ms)
- You have multiple CPU cores available
- Running production mode (high N0)

❌ **Use serial when:**
- Likelihood is very cheap (<0.1ms)
- Debugging (easier to trace errors)
- Running on single-core systems

### Hierarchical Inference with Parallel TMCMC

```python
# Modify src/hierarchical.py to use parallel version:
from src.tmcmc_parallel import tmcmc_parallel as tmcmc

# Then run normally:
python main_calibration.py
```

## Optimization Tips

### 1. Configuration Tuning

**For Quick Testing:**
```python
CONFIG = {
    "DEBUG": True,          # Reduced timesteps
    "N0": 200,              # Fewer samples
    "Ndata": 10,            # Sparse data
    "stages": 10,           # Fewer TMCMC stages
}
```

**For Production:**
```python
CONFIG = {
    "DEBUG": False,
    "N0": 1000,             # More samples
    "Ndata": 40,            # More data points
    "stages": 20,           # More stages
}
```

### 2. Numba Optimization

**First run is slow** (JIT compilation):
```python
# Warmup run
solver.run_deterministic(theta, show_progress=False)

# Subsequent runs are fast
for _ in range(100):
    solver.run_deterministic(theta, show_progress=False)  # Fast!
```

**Tips:**
- Import Numba functions early
- Avoid Python objects in Numba functions
- Use contiguous numpy arrays

### 3. Memory Optimization

**Reduce memory usage:**

```python
# 1. Use float32 instead of float64 where appropriate
theta = np.array(theta, dtype=np.float32)

# 2. Delete large arrays when done
del large_array
import gc; gc.collect()

# 3. Process data in batches
for batch in data_batches:
    process(batch)
    del batch
```

**Monitor memory:**
```bash
python tools/memory_profile.py --component hierarchical --plot
```

### 4. Sparse Data

Use sparse observations instead of full time series:

```python
# Full time series (slow)
Ndata = maxtimestep  # e.g., 100

# Sparse (faster, still accurate)
Ndata = 20  # 20 evenly-spaced observations
```

**Rule of thumb:** Ndata = 10-40 is usually sufficient.

### 5. TMCMC Tuning

**For fast convergence:**
```python
target_ess_ratio = 0.5      # Aggressive (faster)
min_delta_beta = 0.01       # Prevents stalling
```

**For sharp likelihood peaks:**
```python
logL_scale = 0.2            # Scale down sharp peaks
c_const = 25.0              # Lower constant
```

## Benchmarking

### Running Benchmarks

```bash
# Initial baseline
python tools/benchmark_suite.py --output baseline_v1.0.json

# After optimization
python tools/benchmark_suite.py --output optimized_v1.1.json

# Compare
python tools/benchmark_suite.py --output optimized_v1.1.json --compare baseline_v1.0.json
```

### Interpreting Results

Good performance:
- ✓ Solver: < 3s (DEBUG mode)
- ✓ TSM: < 5s (DEBUG mode)
- ✓ TMCMC: < 5s (simple 2D problem)
- ✓ Hierarchical: < 60s (DEBUG mode)

Regression flags:
- ⚠ Slowdown > 20%: Investigate
- ⚠ Memory increase > 50%: Check for leaks

### CI/CD Integration

Add to `.github/workflows/benchmark.yml`:

```yaml
- name: Run benchmarks
  run: |
    python tools/benchmark_suite.py --output new.json --quick
    python tools/benchmark_suite.py --output new.json --compare baseline.json
```

## Memory Management

### Memory Usage by Component

| Component | Typical Usage | Peak Usage |
|-----------|--------------|------------|
| PDE Solver | 50 MB | 100 MB |
| TMCMC (N0=500) | 100 MB | 150 MB |
| Full Hierarchical | 200 MB | 400 MB |

### Detecting Memory Leaks

```bash
# Profile over multiple runs
python tools/memory_profile.py --component solver --plot
```

Check for:
- ⚠ Increasing memory over iterations
- ⚠ Memory not released after completion
- ⚠ Unexpected peak usage

### Memory Optimization Strategies

1. **Clear intermediate results:**
   ```python
   results = run_calibration()
   # Extract what you need
   theta_final = results.theta_final
   # Clear the rest
   del results
   ```

2. **Use sparse storage:**
   ```python
   # Only store final samples, not all stages
   final_samples = tmcmc_result.samples[-1]
   del tmcmc_result  # Free intermediate samples
   ```

3. **Streaming for large datasets:**
   ```python
   # Don't load all data at once
   for data_chunk in data_generator:
       process(data_chunk)
   ```

## Performance Best Practices

### Development Workflow

1. **Start small:** Use DEBUG mode for development
2. **Profile early:** Identify bottlenecks before optimizing
3. **Measure impact:** Benchmark before and after changes
4. **Test at scale:** Verify performance in production mode

### Code Optimization Priorities

1. **Algorithm choice** (biggest impact)
   - Use TSM instead of finite differences
   - Use TMCMC instead of standard MCMC
   - Hierarchical decomposition reduces dimensionality

2. **Parallelization** (2-8x speedup)
   - Use `tmcmc_parallel` for expensive likelihoods
   - Consider GPU for PDE solver (future work)

3. **JIT compilation** (10-100x speedup)
   - Already implemented with Numba
   - Profile to find remaining Python bottlenecks

4. **Micro-optimizations** (5-20% speedup)
   - Vectorize operations
   - Avoid unnecessary copies
   - Use appropriate data types

### Common Pitfalls

❌ **Don't:**
- Optimize prematurely (profile first!)
- Use parallel for cheap functions (overhead > savings)
- Ignore memory usage (crashes are expensive)
- Skip validation tests after optimization

✅ **Do:**
- Profile to find real bottlenecks
- Benchmark before and after changes
- Test correctness after optimization
- Document performance characteristics

## Advanced Topics

### Custom Parallelization

For domain-specific parallelization:

```python
from multiprocessing import Pool

def parallel_forward_solve(theta_samples, n_workers=8):
    with Pool(n_workers) as pool:
        results = pool.map(solve_pde, theta_samples)
    return results
```

### GPU Acceleration (Future)

Potential GPU targets:
- PDE solver with CuPy/JAX
- Matrix operations in TSM
- Likelihood evaluations

Expected speedup: 10-50x for large problems

### Distributed Computing (Future)

For very large problems:
- Use Dask for distributed TMCMC
- Ray for flexible parallelism
- MPI for HPC clusters

## Resources

- **Numba Documentation:** https://numba.pydata.org/
- **Python Profiling:** https://docs.python.org/3/library/profile.html
- **Memory Profiler:** https://pypi.org/project/memory-profiler/
- **Multiprocessing:** https://docs.python.org/3/library/multiprocessing.html

## Support

Performance questions? Open an issue with:
- System specs (CPU, RAM, OS)
- Configuration used
- Profile output
- Benchmark results

---

**Last updated:** 2025-12-03
**Version:** 1.0.0
