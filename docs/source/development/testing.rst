=======
Testing
=======

This page describes the testing framework and practices for the Biofilm Multi-Scale Parameter Estimation project.

Overview
========

The project uses **pytest** for unit testing and **pytest-cov** for coverage analysis. Tests are located in the ``tests/`` directory and cover:

* Numerical solvers
* TSM sensitivity computation
* TMCMC sampling
* Hierarchical updating
* Utility functions

Running Tests
=============

All Tests
---------

Run the complete test suite:

.. code-block:: bash

   pytest tests/ -v

Expected output:

.. code-block:: text

   ============================= test session starts ==============================
   tests/test_solver.py::test_solver_convergence PASSED
   tests/test_tsm.py::test_tsm_sensitivity PASSED
   ...
   ========================= X passed in Y seconds ==============================

Specific Test File
------------------

Run tests from a specific file:

.. code-block:: bash

   pytest tests/test_solver.py -v

Specific Test Function
----------------------

Run a single test function:

.. code-block:: bash

   pytest tests/test_solver.py::test_newton_convergence -v

With Coverage
-------------

Generate coverage report:

.. code-block:: bash

   pytest tests/ --cov=src --cov-report=html --cov-report=term

View the HTML coverage report:

.. code-block:: bash

   # Linux/macOS
   open htmlcov/index.html

   # Windows
   start htmlcov/index.html

Test Organization
=================

Test Structure
--------------

.. code-block:: text

   tests/
   ├── test_solver.py          # Numerical solver tests
   ├── test_tsm.py             # TSM algorithm tests
   ├── test_tmcmc.py           # TMCMC tests (if exists)
   ├── test_hierarchical.py    # Hierarchical updating tests (if exists)
   ├── test_utils.py           # Utility function tests
   └── conftest.py             # Pytest configuration

Test Categories
---------------

1. **Unit Tests**: Test individual functions
2. **Integration Tests**: Test module interactions
3. **Regression Tests**: Ensure previous bugs don't reoccur
4. **Performance Tests**: Benchmark critical operations

Writing Tests
=============

Basic Test Example
------------------

.. code-block:: python

   # tests/test_example.py
   import pytest
   import numpy as np
   from src.utils import example_function

   def test_example_function():
       """Test that example_function works correctly."""
       # Arrange
       input_value = 5
       expected_output = 25

       # Act
       result = example_function(input_value)

       # Assert
       assert result == expected_output

Parametrized Tests
------------------

Test multiple input combinations:

.. code-block:: python

   import pytest

   @pytest.mark.parametrize("input_val,expected", [
       (0, 0),
       (1, 1),
       (2, 4),
       (3, 9),
   ])
   def test_square_function(input_val, expected):
       """Test square function with various inputs."""
       from src.utils import square
       assert square(input_val) == expected

Testing with Fixtures
---------------------

Reuse test data and setup:

.. code-block:: python

   import pytest
   import numpy as np

   @pytest.fixture
   def sample_config():
       """Provide a sample configuration for testing."""
       return {
           "dt": 1e-4,
           "maxtimestep": 10,
           "N0": 100,
       }

   @pytest.fixture
   def sample_parameters():
       """Provide sample parameters for testing."""
       return np.array([0.8, 2.0, 1.0, 0.1, 0.2])

   def test_solver_with_fixture(sample_config, sample_parameters):
       """Test solver using fixtures."""
       from src.solver_newton import solve_biofilm
       result = solve_biofilm(sample_parameters, sample_config)
       assert result.shape[0] == sample_config["maxtimestep"]

Testing Numerical Code
-----------------------

.. code-block:: python

   import numpy as np
   from numpy.testing import assert_allclose

   def test_numerical_accuracy():
       """Test numerical accuracy within tolerance."""
       from src.numerics import compute_gradient

       # Known analytical solution
       x = np.linspace(0, 1, 100)
       f = np.sin(2 * np.pi * x)
       expected_grad = 2 * np.pi * np.cos(2 * np.pi * x)

       # Numerical computation
       computed_grad = compute_gradient(f, x)

       # Assert with tolerance
       assert_allclose(computed_grad, expected_grad, rtol=1e-5, atol=1e-8)

Testing Exceptions
------------------

.. code-block:: python

   import pytest

   def test_invalid_input_raises_error():
       """Test that invalid input raises appropriate error."""
       from src.solver_newton import solve_biofilm

       invalid_params = np.array([])  # Empty array

       with pytest.raises(ValueError, match="Parameter array cannot be empty"):
           solve_biofilm(invalid_params, {})

Example Test Files
==================

test_solver.py
--------------

.. code-block:: python

   """Tests for numerical solvers."""
   import pytest
   import numpy as np
   from numpy.testing import assert_allclose
   from src.solver_newton import solve_biofilm
   from src.config import CONFIG

   @pytest.fixture
   def default_params():
       """Default parameter set for testing."""
       return np.array([0.8, 2.0, 1.0, 0.1, 0.2])

   def test_solver_runs_without_error(default_params):
       """Test that solver completes without errors."""
       config = CONFIG["M1"].copy()
       config["maxtimestep"] = 10  # Short run for testing

       result = solve_biofilm(default_params, config)
       assert result is not None

   def test_solver_output_shape(default_params):
       """Test that solver output has correct shape."""
       config = CONFIG["M1"].copy()
       config["maxtimestep"] = 10

       result = solve_biofilm(default_params, config)
       assert result.shape[0] == config["maxtimestep"]

   def test_conservation_law(default_params):
       """Test that volume fractions sum to 1."""
       config = CONFIG["M1"].copy()
       config["maxtimestep"] = 10

       result = solve_biofilm(default_params, config)

       # Sum of volume fractions should be ≤ 1
       total = np.sum(result, axis=-1)
       assert np.all(total <= 1.01)  # Allow small numerical error

   def test_convergence_tolerance():
       """Test that Newton solver converges within tolerance."""
       # This would test convergence criteria
       pass

test_tsm.py
-----------

.. code-block:: python

   """Tests for Time-Separated Mechanics."""
   import pytest
   import numpy as np
   from src.tsm import compute_tsm_sensitivity
   from src.config import CONFIG

   def test_sensitivity_shape():
       """Test that sensitivity matrix has correct shape."""
       theta = np.array([0.8, 2.0, 1.0, 0.1, 0.2])
       config = CONFIG["M1"].copy()
       config["maxtimestep"] = 10

       S = compute_tsm_sensitivity(theta, config)

       # Shape should be (n_outputs, n_parameters)
       assert S.shape[1] == len(theta)

   def test_sensitivity_finite_difference_agreement():
       """Test TSM sensitivity against finite differences."""
       theta = np.array([0.8, 2.0, 1.0, 0.1, 0.2])
       config = CONFIG["M1"].copy()
       config["maxtimestep"] = 5  # Very short for speed

       # TSM sensitivity
       S_tsm = compute_tsm_sensitivity(theta, config)

       # Finite difference approximation
       eps = 1e-5
       S_fd = np.zeros_like(S_tsm)

       for i in range(len(theta)):
           theta_plus = theta.copy()
           theta_plus[i] += eps

           theta_minus = theta.copy()
           theta_minus[i] -= eps

           f_plus = solve_biofilm(theta_plus, config)
           f_minus = solve_biofilm(theta_minus, config)

           S_fd[:, i] = (f_plus - f_minus).flatten() / (2 * eps)

       # Should agree within tolerance
       np.testing.assert_allclose(S_tsm, S_fd, rtol=1e-3)

Continuous Integration
======================

GitHub Actions
--------------

The project uses GitHub Actions for CI/CD. See ``.github/workflows/tests.yml``:

.. code-block:: yaml

   name: Tests

   on: [push, pull_request]

   jobs:
     test:
       runs-on: ubuntu-latest
       strategy:
         matrix:
           python-version: ['3.9', '3.10', '3.11', '3.12']

       steps:
       - uses: actions/checkout@v3
       - name: Set up Python
         uses: actions/setup-python@v4
         with:
           python-version: ${{ matrix.python-version }}
       - name: Install dependencies
         run: |
           pip install -r requirements.txt
           pip install -r requirements-dev.txt
       - name: Run tests
         run: pytest tests/ -v --cov=src

Test Coverage Goals
-------------------

Target coverage levels:

* **Overall**: > 80%
* **Core modules**: > 90%
* **Critical paths**: 100%

Quality Assurance
=================

Code Quality Tools
------------------

**Black** - Code formatting:

.. code-block:: bash

   black src/ tests/

**Flake8** - Linting:

.. code-block:: bash

   flake8 src/ tests/ --max-line-length=100

**MyPy** - Type checking (optional):

.. code-block:: bash

   mypy src/ --ignore-missing-imports

Pre-commit Hooks
----------------

Set up pre-commit hooks to run tests automatically:

.. code-block:: bash

   # Install pre-commit
   pip install pre-commit

   # Set up hooks
   pre-commit install

Performance Testing
===================

Benchmarking
------------

Use ``tools/profile_performance.py`` for benchmarking:

.. code-block:: bash

   python tools/profile_performance.py

Profile specific functions:

.. code-block:: python

   import cProfile
   import pstats

   profiler = cProfile.Profile()
   profiler.enable()

   # Run code to profile
   result = hierarchical_case2(CONFIG)

   profiler.disable()
   stats = pstats.Stats(profiler)
   stats.sort_stats('cumulative')
   stats.print_stats(20)  # Top 20 functions

Memory Profiling
----------------

Use ``memory_profiler`` for memory analysis:

.. code-block:: bash

   pip install memory_profiler

   # Profile a function
   python -m memory_profiler main_calibration.py

Regression Testing
==================

Ensure results remain consistent across versions:

.. code-block:: python

   def test_regression_M1_calibration():
       """Test that M1 calibration gives consistent results."""
       from src.hierarchical import hierarchical_case2
       from src.config import CONFIG

       # Set random seed for reproducibility
       np.random.seed(42)

       results = hierarchical_case2(CONFIG)

       # Known good values from previous version
       expected_theta = np.array([0.82, 1.98, 1.02, 0.09, 0.21])

       # Allow small tolerance for numerical differences
       np.testing.assert_allclose(
           results.theta_M1,
           expected_theta,
           rtol=0.1  # 10% relative tolerance
       )

Best Practices
==============

1. **Test First**: Write tests before implementing features (TDD)
2. **Small Tests**: Each test should verify one specific behavior
3. **Descriptive Names**: Use clear, descriptive test names
4. **Fast Tests**: Keep unit tests fast (< 1s each)
5. **Isolated Tests**: Tests should not depend on each other
6. **Mock External Dependencies**: Use mocking for external resources
7. **Document Tests**: Add docstrings explaining what is being tested
8. **Regular Runs**: Run tests frequently during development

Debugging Failed Tests
=======================

Verbose Output
--------------

.. code-block:: bash

   pytest tests/test_solver.py -vv

Print Debugging
---------------

.. code-block:: bash

   pytest tests/test_solver.py -s  # Don't capture stdout

Drop into Debugger
------------------

.. code-block:: bash

   pytest tests/test_solver.py --pdb  # Drop into pdb on failure

Specific Test with Print
-------------------------

.. code-block:: python

   def test_debug_example():
       """Test with debug output."""
       result = some_function()
       print(f"Debug: result = {result}")  # Will show with -s flag
       assert result > 0

Common Issues
=============

Numerical Precision
-------------------

Use appropriate tolerances for floating-point comparisons:

.. code-block:: python

   # Bad
   assert result == 0.1

   # Good
   assert abs(result - 0.1) < 1e-10

Random Number Seeds
-------------------

Set seeds for reproducible tests:

.. code-block:: python

   import numpy as np

   def test_with_random():
       np.random.seed(42)  # Reproducible
       result = function_using_random()
       assert result > 0

Test Timeouts
-------------

Add timeouts for long-running tests:

.. code-block:: python

   @pytest.mark.timeout(60)  # 60 seconds
   def test_long_running():
       expensive_operation()

Next Steps
==========

* Review :doc:`contributing` for development guidelines
* Check :doc:`changelog` for recent updates
* Explore :doc:`../api/core` for implementation details
