============
Contributing
============

Thank you for your interest in contributing to the Biofilm Multi-Scale Parameter Estimation project! This guide will help you get started.

Getting Started
===============

Fork and Clone
--------------

1. Fork the repository on GitHub
2. Clone your fork locally:

.. code-block:: bash

   git clone https://github.com/YOUR_USERNAME/biofilm_project.git
   cd biofilm_project

3. Add upstream remote:

.. code-block:: bash

   git remote add upstream https://github.com/ORIGINAL_OWNER/biofilm_project.git

Set Up Development Environment
-------------------------------

1. Create a virtual environment:

.. code-block:: bash

   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\\Scripts\\activate

2. Install dependencies:

.. code-block:: bash

   pip install -r requirements.txt
   pip install -r requirements-dev.txt

3. Install pre-commit hooks:

.. code-block:: bash

   pip install pre-commit
   pre-commit install

4. Verify setup:

.. code-block:: bash

   pytest tests/ -v
   black --check src/ tests/
   flake8 src/ tests/

Development Workflow
====================

Creating a Branch
-----------------

Create a feature branch for your changes:

.. code-block:: bash

   git checkout -b feature/your-feature-name

Branch naming conventions:

* ``feature/`` - New features
* ``bugfix/`` - Bug fixes
* ``docs/`` - Documentation updates
* ``refactor/`` - Code refactoring
* ``test/`` - Test additions/improvements

Making Changes
--------------

1. **Write code** following the style guide (see below)
2. **Add tests** for new functionality
3. **Update documentation** as needed
4. **Run tests** to ensure nothing breaks:

.. code-block:: bash

   pytest tests/ -v

5. **Format code** with Black:

.. code-block:: bash

   black src/ tests/

6. **Check linting** with Flake8:

.. code-block:: bash

   flake8 src/ tests/ --max-line-length=100

Committing Changes
------------------

Write clear, descriptive commit messages:

.. code-block:: bash

   git add .
   git commit -m "Add feature: brief description

   More detailed explanation of what changed and why.
   Reference any related issues: #123"

Commit message format:

* First line: Brief summary (50 chars or less)
* Blank line
* Detailed description (wrapped at 72 chars)
* Reference issues/PRs

Examples:

.. code-block:: text

   Good:
   - "Fix convergence issue in Newton solver for edge cases"
   - "Add sensitivity analysis tutorial to documentation"
   - "Refactor TMCMC to improve readability"

   Bad:
   - "fix bug"
   - "update code"
   - "changes"

Pushing Changes
---------------

.. code-block:: bash

   git push origin feature/your-feature-name

Creating a Pull Request
=======================

1. Go to GitHub and create a Pull Request (PR)
2. Fill out the PR template:

   * **Title**: Clear, descriptive summary
   * **Description**: What changes were made and why
   * **Related Issues**: Reference any issues addressed
   * **Testing**: Describe testing performed
   * **Checklist**: Complete the checklist

3. Wait for review feedback
4. Address review comments
5. Update PR as needed

Pull Request Template
----------------------

.. code-block:: markdown

   ## Description
   Brief description of changes

   ## Motivation and Context
   Why are these changes needed?

   ## Type of Change
   - [ ] Bug fix
   - [ ] New feature
   - [ ] Documentation update
   - [ ] Refactoring
   - [ ] Performance improvement

   ## How Has This Been Tested?
   Describe testing performed

   ## Checklist
   - [ ] Code follows style guide
   - [ ] Tests added/updated
   - [ ] Documentation updated
   - [ ] All tests pass
   - [ ] Black formatting applied
   - [ ] Flake8 linting passed

Code Style Guide
================

General Principles
------------------

* **PEP 8**: Follow PEP 8 style guide
* **Black**: Use Black for automatic formatting
* **Line Length**: 100 characters maximum
* **Type Hints**: Use type hints where appropriate
* **Docstrings**: Document all public functions/classes

Naming Conventions
------------------

.. code-block:: python

   # Variables and functions: snake_case
   my_variable = 10
   def compute_sensitivity(theta, config):
       pass

   # Classes: PascalCase
   class BiofilmSolver:
       pass

   # Constants: UPPER_SNAKE_CASE
   MAX_ITERATIONS = 100
   DEFAULT_TOLERANCE = 1e-8

   # Private members: leading underscore
   def _internal_helper():
       pass

Docstring Format
----------------

Use NumPy-style docstrings:

.. code-block:: python

   def solve_biofilm(theta, config):
       """
       Solve the biofilm PDE system with given parameters.

       Parameters
       ----------
       theta : np.ndarray
           Parameter vector of shape (n_params,)
       config : dict
           Configuration dictionary with keys:
           - 'dt': Time step size
           - 'maxtimestep': Number of timesteps
           - 'c_const': Likelihood scaling constant

       Returns
       -------
       solution : np.ndarray
           Solution array of shape (maxtimestep, n_species)

       Raises
       ------
       ValueError
           If theta has incorrect dimensions
       RuntimeError
           If Newton solver fails to converge

       Examples
       --------
       >>> theta = np.array([0.8, 2.0, 1.0, 0.1, 0.2])
       >>> config = {"dt": 1e-4, "maxtimestep": 100}
       >>> solution = solve_biofilm(theta, config)
       >>> print(solution.shape)
       (100, 2)

       Notes
       -----
       Uses implicit Euler time integration with Newton-Raphson
       for nonlinear systems.

       References
       ----------
       .. [1] Author et al., "Paper title", Journal, 2024.
       """
       pass

Type Hints
----------

Use type hints for function signatures:

.. code-block:: python

   from typing import Dict, List, Optional, Tuple
   import numpy as np

   def compute_likelihood(
       theta: np.ndarray,
       data: np.ndarray,
       config: Dict[str, float]
   ) -> float:
       """Compute log-likelihood."""
       pass

   def run_mcmc(
       n_samples: int,
       initial_state: Optional[np.ndarray] = None
   ) -> Tuple[np.ndarray, List[float]]:
       """Run MCMC sampling."""
       pass

Code Organization
-----------------

.. code-block:: python

   """
   Module docstring describing purpose.
   """

   # Standard library imports
   import os
   import sys
   from typing import Dict

   # Third-party imports
   import numpy as np
   import scipy.optimize
   from matplotlib import pyplot as plt

   # Local imports
   from src.config import CONFIG
   from src.utils import load_data

   # Constants
   DEFAULT_TOLERANCE = 1e-8
   MAX_ITERATIONS = 100

   # Functions and classes
   def my_function():
       pass

Comments
--------

.. code-block:: python

   # Good comments explain WHY, not WHAT
   # Bad:
   x = x + 1  # Increment x

   # Good:
   x = x + 1  # Account for zero-indexing offset

   # For complex logic, explain the approach:
   # Use bisection to find beta that gives target ESS
   # since ESS is monotonic in beta
   while abs(ess - target_ess) > tolerance:
       beta_mid = (beta_low + beta_high) / 2
       ess = compute_ess(beta_mid)
       if ess > target_ess:
           beta_low = beta_mid
       else:
           beta_high = beta_mid

Testing Guidelines
==================

Test Coverage
-------------

* **All new functions**: Must have unit tests
* **Bug fixes**: Add regression test
* **Critical paths**: 100% coverage
* **Overall target**: > 80% coverage

Test Structure
--------------

.. code-block:: python

   def test_function_name():
       """Test that function_name does X correctly."""
       # Arrange
       input_data = prepare_input()
       expected_output = compute_expected()

       # Act
       actual_output = function_name(input_data)

       # Assert
       assert actual_output == expected_output

Test Independence
-----------------

Tests must be independent and reproducible:

.. code-block:: python

   # Good: Independent test with fixed seed
   def test_random_function():
       np.random.seed(42)
       result = function_using_random()
       assert result > 0

   # Bad: Depends on global state
   global_state = None

   def test_modifies_global():
       global global_state
       global_state = 123
       assert global_state == 123

Documentation
=============

Documentation Types
-------------------

1. **Code comments**: Explain complex logic
2. **Docstrings**: Document all public APIs
3. **User guides**: High-level tutorials
4. **API reference**: Auto-generated from docstrings

Building Documentation
----------------------

.. code-block:: bash

   cd docs
   make html
   open build/html/index.html

Documentation Structure
-----------------------

.. code-block:: text

   docs/source/
   ├── index.rst               # Main page
   ├── user_guide/             # User-facing docs
   │   ├── installation.rst
   │   ├── quickstart.rst
   │   └── tutorials.rst
   ├── api/                    # API reference
   │   ├── core.rst
   │   └── solvers.rst
   └── development/            # Developer docs
       ├── contributing.rst
       └── testing.rst

Review Process
==============

What Reviewers Look For
------------------------

1. **Correctness**: Does the code work as intended?
2. **Tests**: Are there adequate tests?
3. **Style**: Does it follow style guide?
4. **Documentation**: Is it well documented?
5. **Performance**: Any performance concerns?
6. **Maintainability**: Is it readable and maintainable?

Responding to Reviews
----------------------

* **Be respectful**: Reviewers are helping improve the code
* **Ask questions**: If feedback is unclear, ask for clarification
* **Make changes**: Address all feedback or explain why not
* **Be patient**: Reviews take time

Example Response
----------------

.. code-block:: markdown

   > Reviewer: This function could be simplified using numpy.einsum

   Thanks for the suggestion! I've updated the code to use einsum,
   which is clearer and slightly faster. Updated in commit abc123.

   > Reviewer: Missing test for edge case where theta is empty

   Good catch! Added test_empty_theta_raises_error in commit def456.

Common Pitfalls
===============

Performance
-----------

* **Don't optimize prematurely**: Profile first
* **Use Numba for hot loops**: JIT compile critical sections
* **Vectorize with NumPy**: Avoid Python loops when possible

.. code-block:: python

   # Slow
   result = []
   for i in range(n):
       result.append(expensive_function(data[i]))

   # Fast
   result = expensive_function(data)  # Vectorized

Code Complexity
---------------

* **Keep functions small**: < 50 lines ideally
* **Single responsibility**: Each function does one thing
* **Avoid deep nesting**: Max 3 levels of indentation

.. code-block:: python

   # Too complex
   def complex_function(x):
       if x > 0:
           if x < 10:
               if x % 2 == 0:
                   return "even small positive"

   # Better
   def is_even_small_positive(x):
       return 0 < x < 10 and x % 2 == 0

Backward Compatibility
----------------------

* **Deprecate, don't remove**: Give users time to migrate
* **Document breaking changes**: In changelog and migration guide
* **Semantic versioning**: Follow semver (MAJOR.MINOR.PATCH)

Getting Help
============

* **GitHub Issues**: Ask questions, report bugs
* **Discussions**: General discussions and ideas
* **Email**: Contact maintainers directly
* **Documentation**: Check docs first

Recognition
===========

Contributors are recognized in:

* **CONTRIBUTORS.md**: List of all contributors
* **Release notes**: Notable contributions highlighted
* **Git history**: Your commits remain part of the project

License
=======

By contributing, you agree that your contributions will be licensed under the MIT License.

Thank You!
==========

Your contributions make this project better for everyone. We appreciate your time and effort!
