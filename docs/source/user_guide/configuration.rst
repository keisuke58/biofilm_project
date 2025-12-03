=============
Configuration
=============

This guide explains all configuration options available in the Biofilm Multi-Scale Parameter Estimation framework.

Hierarchical Calibration Overview
=================================

The calibration pipeline estimates **14 parameters** across the three forward models (``M1``, ``M2``, ``M3``) plus a validation-only pass. All unknown parameters use uninformative :math:`\text{Uniform}(0, 3)` priors and the aleatory uncertainty is represented with a **0.5% Coefficient of Variation (CoV)** (``cov_rel = 0.005``). Each stage fits against **20 sparse data points** (:math:`N_{\text{data}} = 20`).

Stage 1: Submodel :math:`\mathcal{M}^1` (Species 1 & 2)
---------------------------------------------------------

* **Goal:** Infer 5 parameters :math:`\theta^{(1)} = [a_{11}, a_{12}, a_{22}, b_1, b_2]` (species 1–2 self/cross growth and antibiotic sensitivity).
* **Fixed values:** :math:`\eta_1 = \eta_2 = 1.0`.
* **Initial conditions:** :math:`\phi_1 = \phi_2 = 0.2` (species 3 & 4 absent).
* **Environment:** Nutrients :math:`c^* = 100`; antibiotics :math:`\alpha^* = 100`.
* **Simulation:** :math:`N = 2500` time steps with :math:`\Delta t = 10^{-5}` s.
* **Outcome:** MAP estimates of :math:`\theta^{(1)}` are fixed for later stages.

Stage 2: Submodel :math:`\mathcal{M}^2` (Species 3 & 4)
---------------------------------------------------------

* **Goal:** Infer 5 parameters :math:`\theta^{(2)} = [a_{33}, a_{34}, a_{44}, b_3, b_4]` for species 3–4.
* **Fixed values:** :math:`\eta_3 = \eta_4 = 1.0`.
* **Initial conditions:** :math:`\phi_3 = \phi_4 = 0.2` (species 1 & 2 absent).
* **Environment:** Nutrients :math:`c^* = 100`; antibiotics reduced to :math:`\alpha^* = 10` to avoid non-informative zero concentrations.
* **Simulation:** :math:`N = 5000` time steps with :math:`\Delta t = 10^{-5}` s (longer run for slower growth).
* **Outcome:** MAP estimates of :math:`\theta^{(2)}` are fixed for Stage 3.

Stage 3: Full Model :math:`\mathcal{M}^3` (Interaction Learning)
-----------------------------------------------------------------

* **Goal:** Infer the remaining 4 cross-interaction parameters :math:`\theta^{(3)} = [a_{13}, a_{14}, a_{23}, a_{24}]`.
* **Constraints:** The 10 parameters from :math:`\theta^{(1)}` and :math:`\theta^{(2)}` remain fixed at their MAP values.
* **Initial conditions:** :math:`\phi_1 = \phi_2 = \phi_3 = \phi_4 = 0.02`.
* **Environment:** Nutrients :math:`c^* = 25`; antibiotics off (:math:`\alpha^* = 0`).
* **Simulation:** :math:`N = 750` time steps with :math:`\Delta t = 10^{-4}` s.

Stage 4: Validation Model :math:`\mathcal{M}_{val}^3`
------------------------------------------------------

* **Goal:** No additional inference; validates predictions under a dynamic antibiotic profile using the 14 MAP parameters from earlier stages.
* **Initial conditions:** :math:`\phi_1 = \phi_2 = \phi_3 = \phi_4 = 0.02`.
* **Environment:** Nutrients :math:`c^* = 25`; antibiotics switch from :math:`\alpha^* = 0` for :math:`t \le 500` to :math:`\alpha^* = 50` afterward.
* **Simulation:** :math:`N = 1500` time steps with :math:`\Delta t = 10^{-4}` s.

``M1`` and ``M2`` are strictly two-species submodels; ``M3`` activates all four species while keeping the earlier parameters fixed.

Configuration File
==================

All configuration is centralized in ``src/config.py``. This file contains:

* Debug/production mode settings
* Model parameters for M1, M2, M3
* TMCMC algorithm settings
* TSM parameters
* File paths and output directories

Global Settings
===============

DEBUG Mode
----------

Controls whether to run in fast debug mode or slower production mode:

.. code-block:: python

   DEBUG = True   # Fast mode (~45s, lower accuracy)
   DEBUG = False  # Production mode (~360s, higher accuracy)

**Impact:**

* **DEBUG=True**: Reduced timesteps, fewer TMCMC stages, sparse data
* **DEBUG=False**: Full resolution, more samples, better convergence

Model Configurations
====================

M1 Configuration (Species 1-2)
-------------------------------

.. code-block:: python

   CONFIG["M1"] = {
       "dt": 1e-4,              # Time step size
       "maxtimestep": 80,       # Maximum number of timesteps
       "c_const": 100.0,        # Likelihood scaling constant
       "alpha_const": 100.0,    # TSM scaling parameter
   }

Parameters:
   * ``dt``: Time step for numerical integration
   * ``maxtimestep``: Number of time steps to simulate
   * ``c_const``: Controls likelihood sharpness (higher = sharper peaks)
   * ``alpha_const``: TSM sensitivity scaling

M2 Configuration (Species 3-4)
-------------------------------

.. code-block:: python

   CONFIG["M2"] = {
       "dt": 1e-4,
       "maxtimestep": 100,
       "c_const": 100.0,
       "alpha_const": 10.0,     # Lower than M1
   }

M3 Configuration (All Species)
-------------------------------

.. code-block:: python

   CONFIG["M3"] = {
       "dt": 1e-4,
       "maxtimestep": 60,
       "c_const": 25.0,         # Lower for stability
       "alpha_const": 0.0,      # No TSM scaling
   }

TMCMC Settings
==============

Sampling Parameters
-------------------

.. code-block:: python

   CONFIG = {
       "N0": 500,              # Initial number of samples
       "stages": 15,           # Number of TMCMC stages
       "Ndata": 20,           # Number of sparse data points
   }

**N0 (Initial Samples)**
   * Higher values: Better posterior approximation, slower
   * Lower values: Faster execution, less accurate
   * Recommended: 500-1000 for production, 200-500 for debug

**stages**
   * More stages: Smoother tempering schedule, better convergence
   * Fewer stages: Faster but may miss modes
   * Recommended: 15-20 for production, 10-15 for debug

**Ndata**
   * Sparse data points for computational efficiency
   * Higher values: More information, slower
   * Recommended: 20-40

TSM Parameters
==============

Sensitivity and Uncertainty
----------------------------

.. code-block:: python

   CONFIG = {
       "cov_rel": 0.005,       # Relative parameter uncertainty (0.5%)
       "sigma_obs": 0.005,     # Observation noise standard deviation
   }

**cov_rel**
   * Controls parameter perturbation for sensitivity analysis
   * Smaller values: More accurate but may be numerically unstable
   * Larger values: More stable but less accurate
   * Recommended: 0.001-0.01

**sigma_obs**
   * Measurement noise standard deviation
   * Should match your actual observation uncertainty
   * Affects likelihood width

Parameter Priors
================

M1 Parameters (θ[0:5])
-----------------------

.. code-block:: python

   # a11, a12, a22, b1, b2
   prior_M1 = {
       "lower": [0.0, 0.0, 0.0, 0.0, 0.0],
       "upper": [3.0, 3.0, 3.0, 3.0, 3.0],
       "distribution": "uniform"
   }

True values: ``[0.8, 2.0, 1.0, 0.1, 0.2]``

M2 Parameters (θ[5:10])
-----------------------

.. code-block:: python

   # a33, a34, a44, b3, b4
   prior_M2 = {
       "lower": [0.0, 0.0, 0.0, 0.0, 0.0],
       "upper": [3.0, 3.0, 3.0, 3.0, 3.0],
       "distribution": "uniform"
   }

True values: ``[1.5, 1.0, 2.0, 0.3, 0.4]``

M3 Parameters (θ[10:14])
------------------------

.. code-block:: python

   # a13, a14, a23, a24
   prior_M3 = {
       "lower": [0.0, 0.0, 0.0, 0.0],
       "upper": [3.0, 3.0, 3.0, 3.0],
       "distribution": "uniform"
   }

True values: ``[2.0, 1.0, 2.0, 1.0]``

Physical Parameters
===================

Domain and Grid
---------------

.. code-block:: python

   # Spatial domain
   L = 1.0          # Domain length
   nx = 100         # Number of grid points
   dx = L / nx      # Grid spacing

Initial Conditions
------------------

.. code-block:: python

   # Volume fractions (φ_i)
   phi_init = [0.2, 0.2, 0.2, 0.2]  # Species 1-4

   # Porosities (ψ_i)
   psi_init = [0.5, 0.5, 0.5, 0.5]

Physical Constants
------------------

.. code-block:: python

   # Diffusion coefficients
   mu = [1.0, 1.0, 1.0, 1.0]    # Volume fraction diffusivity
   kappa = [1.0, 1.0, 1.0, 1.0] # Porosity diffusivity

Output Settings
===============

File Paths
----------

.. code-block:: python

   # Results directory
   RESULTS_DIR = "results"

   # Figure output
   FIG_FORMAT = "png"
   FIG_DPI = 300

   # Logging
   LOG_FILE = "results/biofilm_calibration.log"
   LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR

Visualization Options
---------------------

.. code-block:: python

   # Plot settings
   PLOT_CONFIG = {
       "figsize": (10, 6),
       "style": "seaborn-v0_8",
       "context": "paper",
       "font_scale": 1.2,
   }

Advanced Configuration
======================

Numerical Solver Tolerances
----------------------------

.. code-block:: python

   SOLVER_CONFIG = {
       "newton_tol": 1e-8,       # Newton solver tolerance
       "max_newton_iter": 50,    # Maximum Newton iterations
       "linear_solver": "direct" # or "iterative"
   }

Numba JIT Settings
------------------

.. code-block:: python

   NUMBA_CONFIG = {
       "parallel": True,         # Enable parallel execution
       "fastmath": True,         # Fast math optimizations
       "cache": True,            # Cache compiled functions
   }

Example Configurations
======================

Quick Testing
-------------

.. code-block:: python

   DEBUG = True
   CONFIG = {
       "N0": 200,
       "stages": 10,
       "Ndata": 10,
       "M1": {"maxtimestep": 40},
       "M2": {"maxtimestep": 50},
       "M3": {"maxtimestep": 30},
   }

Execution time: ~15-20 seconds

Production Run
--------------

.. code-block:: python

   DEBUG = False
   CONFIG = {
       "N0": 1000,
       "stages": 20,
       "Ndata": 40,
       "M1": {"maxtimestep": 80},
       "M2": {"maxtimestep": 100},
       "M3": {"maxtimestep": 60},
   }

Execution time: ~5-10 minutes

High Accuracy
-------------

.. code-block:: python

   DEBUG = False
   CONFIG = {
       "N0": 2000,
       "stages": 30,
       "Ndata": 100,
       "cov_rel": 0.001,
       "M1": {"maxtimestep": 150, "c_const": 200.0},
       "M2": {"maxtimestep": 200, "c_const": 200.0},
       "M3": {"maxtimestep": 120, "c_const": 50.0},
   }

Execution time: ~20-30 minutes

Troubleshooting
===============

Common Issues
-------------

**"Convergence failed"**
   * Increase ``N0`` (more samples)
   * Increase ``stages`` (smoother tempering)
   * Adjust ``c_const`` (try lower values)

**"ESS too low"**
   * Increase ``N0``
   * Check ``c_const`` (may be too high)
   * Review prior ranges

**"Numerical instability"**
   * Decrease ``dt``
   * Increase ``cov_rel``
   * Check initial conditions

**"Out of memory"**
   * Decrease ``N0``
   * Reduce ``maxtimestep``
   * Enable ``DEBUG`` mode

Next Steps
==========

* Try different configurations with :doc:`quickstart`
* Explore :doc:`tutorials` for advanced usage
* Understand parameter meanings in :doc:`scientific_background`
