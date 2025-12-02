# Recommended Improvements

Based on the comprehensive code review, here are the prioritized improvements with actionable steps.

## ✅ Already Completed

- [x] Comprehensive code review (see PROJECT_REVIEW.md)
- [x] Created requirements.txt
- [x] Created requirements-dev.txt
- [x] Created basic test structure (tests/)
- [x] Added initial unit tests for solver and TSM

## 🔴 High Priority (Critical for Production)

### 1. Fix Error Handling in hierarchical.py

**Issue:** Bare `except:` blocks catch all exceptions, making debugging difficult.

**Fix:**
```python
# Replace lines 208, 260, 311 in src/hierarchical.py
# OLD:
try:
    tsm_res = tsm_M1.solve_tsm(theta_full)
    # ...
except:
    return -1e20

# NEW:
try:
    tsm_res = tsm_M1.solve_tsm(theta_full)
    # ...
except (RuntimeError, np.linalg.LinAlgError, ValueError) as e:
    if CONFIG.get("verbose", False):
        print(f"[M1] TSM failed for theta={theta_M1[:2]}...: {type(e).__name__}")
    return -1e20
```

### 2. Add Logging

**Create `src/logger.py`:**
```python
import logging
import sys

def setup_logger(name="biofilm", level=logging.INFO):
    """Configure project-wide logger"""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger

# Usage in other files:
# from src.logger import setup_logger
# logger = setup_logger()
# logger.info("Starting calibration...")
```

### 3. Remove Obsolete Files

```bash
# Move setup2.py to archive
mkdir -p archive
git mv setup2.py archive/
git commit -m "Archive obsolete setup2.py"
```

### 4. Add Configuration Validation

**Add to `src/config.py`:**
```python
def validate_config(config):
    """Validate configuration dictionary"""
    required_keys = ["M1", "M2", "M3", "Ndata", "N0"]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")

    for model in ["M1", "M2", "M3"]:
        model_config = config[model]
        if model_config["dt"] <= 0:
            raise ValueError(f"{model}: dt must be positive")
        if model_config["maxtimestep"] <= 0:
            raise ValueError(f"{model}: maxtimestep must be positive")

    return True

# Add to CONFIG initialization
CONFIG = get_config(DEBUG)
validate_config(CONFIG)
```

## 🟡 Medium Priority (Quality of Life)

### 5. Add Comprehensive Docstrings

**Template:**
```python
def solve_tsm(self, theta: np.ndarray) -> TSMResult:
    """
    Solve Time-Separated Mechanics problem.

    Computes the mean trajectory (deterministic solution) and variance
    propagation using first-order Taylor expansion of the PDE solution
    with respect to parameter uncertainty.

    Parameters
    ----------
    theta : np.ndarray, shape (14,)
        Parameter vector in order:
        [a11, a12, a22, b1, b2,      # M1: species 1-2
         a33, a34, a44, b3, b4,      # M2: species 3-4
         a13, a14, a23, a24]         # M3: cross-interactions

    Returns
    -------
    TSMResult
        Dataclass containing:
        - t_array : time points (n_time,)
        - mu : mean trajectory (n_time, 10)
        - sigma2 : variance (n_time, 10)
        - x0 : deterministic state (same as mu)
        - x1 : sensitivity ∂g/∂θ (n_time, 10, n_active)

    Raises
    ------
    RuntimeError
        If Newton solver encounters NaN values or fails to converge
    LinAlgError
        If Jacobian is singular

    Notes
    -----
    Uses analytical sensitivity computation when Numba is available,
    which is faster and more accurate than numerical differentiation.

    The variance is computed via linear error propagation:
    σ²(t) = Σₖ (∂g/∂θₖ)² Var(θₖ)

    References
    ----------
    .. [1] Fritsch et al. (2025), "Hierarchical Bayesian Inference for
           Multi-Scale Biofilm Models"
    """
```

### 6. Add Type Hints

Install mypy:
```bash
pip install mypy
```

Add types to key functions:
```python
from typing import Tuple
import numpy.typing as npt

def run_deterministic(
    self,
    theta: npt.NDArray[np.float64],
    show_progress: bool = False
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """..."""
```

### 7. Improve README.md

Update README with:
- Installation instructions
- Quick start guide
- Example usage
- Citation information
- License

**Example addition:**
```markdown
## Installation

### Requirements
- Python 3.9+
- NumPy, SciPy, Numba (see requirements.txt)

### Setup
```bash
git clone <repository>
cd biofilm_project
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Running Tests
```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

## Quick Start

```python
from src.config import CONFIG, get_theta_true
from src.hierarchical import hierarchical_case2

# Run hierarchical Bayesian calibration
results = hierarchical_case2(CONFIG)

print(f"Estimated parameters: {results.theta_final}")
print(f"RMSE: {np.sqrt(((results.theta_final - get_theta_true())**2).mean())}")
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{biofilm2025,
  title={Hierarchical Bayesian Inference for Multi-Scale Biofilm Models},
  author={...},
  journal={...},
  year={2025}
}
```
```

## 🟢 Low Priority (Nice to Have)

### 8. Add Performance Profiling

**Create `scripts/profile.py`:**
```python
import cProfile
import pstats
from src.config import CONFIG
from src.hierarchical import hierarchical_case2

def profile_calibration():
    """Profile full calibration run"""
    profiler = cProfile.Profile()
    profiler.enable()

    results = hierarchical_case2(CONFIG)

    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)  # Top 20 functions

if __name__ == "__main__":
    profile_calibration()
```

### 9. Add Parallel TMCMC

**For future optimization:**
```python
from multiprocessing import Pool

def parallel_likelihood_eval(log_likelihood, theta_list, n_workers=4):
    """Evaluate likelihoods in parallel"""
    with Pool(n_workers) as pool:
        log_likes = pool.map(log_likelihood, theta_list)
    return np.array(log_likes)

# Use in tmcmc.py:
# logL = parallel_likelihood_eval(log_likelihood, theta_curr)
```

### 10. Add Visualization Tests

**Ensure plots are generated correctly:**
```python
def test_plot_posterior(tmp_path):
    """Test that posterior plots are created"""
    from src.viz import BayesianVisualizer
    import matplotlib.pyplot as plt

    viz = BayesianVisualizer(str(tmp_path))
    samples = [np.random.randn(100, 5) for _ in range(3)]
    theta_true = np.array([1.0, 2.0, 1.0, 0.1, 0.2])
    labels = ["a11", "a12", "a22", "b1", "b2"]

    viz.plot_posterior(samples, theta_true, labels, "test")

    # Check file was created
    assert (tmp_path / "posterior_test.png").exists()
    plt.close('all')
```

## Implementation Checklist

Copy this checklist and track your progress:

```
High Priority:
[ ] Fix bare except blocks in hierarchical.py
[ ] Add logging throughout the project
[ ] Remove/archive setup2.py
[ ] Add configuration validation
[ ] Add comprehensive docstrings to public APIs
[ ] Complete test coverage (aim for >80%)

Medium Priority:
[ ] Add type hints to main functions
[ ] Update README with installation & usage
[ ] Add CI/CD pipeline (GitHub Actions)
[ ] Create documentation with Sphinx

Low Priority:
[ ] Profile and optimize bottlenecks
[ ] Implement parallel TMCMC
[ ] Add visualization tests
[ ] Create example notebooks
```

## Testing Your Changes

After each improvement:
```bash
# 1. Run tests
pytest tests/ -v --cov=src

# 2. Check code quality
black src/ tests/
flake8 src/ tests/ --max-line-length=100

# 3. Type check (if you added types)
mypy src/

# 4. Run a short calibration to ensure nothing broke
python main_calibration.py
```

## Getting Help

If you encounter issues:
1. Check PROJECT_REVIEW.md for detailed explanations
2. Review test files for usage examples
3. Open an issue with:
   - What you're trying to do
   - Error message
   - Minimal code to reproduce
