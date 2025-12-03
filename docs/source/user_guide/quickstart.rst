===========
Quick Start
===========

This guide will help you get started with the Biofilm Multi-Scale Parameter Estimation framework quickly.

Your First Calibration
=======================

Step 1: Basic Calibration
--------------------------

Run a basic hierarchical Bayesian calibration in DEBUG mode (fast):

.. code-block:: bash

   python main_calibration.py

Expected Output
^^^^^^^^^^^^^^^

You should see output similar to:

.. code-block:: text

   ========================================================================
   Biofilm Case II: TSM + TMCMC + Hierarchical Bayesian Updating
   ========================================================================
   DEBUG : True
   Ndata : 20, N0 = 500

   [M1] Running TMCMC...
   Stage 0: β=0.0000, ESS=500.0
   Stage 1: β=0.0500, ESS=450.2
   ...
   RMSE: 0.1234
   Total time: 45.2 s
   Convergence: M1=True, M2=True, M3=True

Step 2: Generate Full Report
-----------------------------

For a complete analysis with publication-quality figures:

.. code-block:: bash

   python main_calibration_report.py

This will generate:

* Posterior distribution plots
* MCMC trace plots
* Corner plots
* Time-series predictions with uncertainty bands
* Complete PDF report (``results/bayesian_report.pdf``)

Step 3: Forward Simulation
---------------------------

To run a forward simulation with true parameters:

.. code-block:: bash

   python main_simulation.py

This generates ``forward_simulation.png`` showing the biofilm evolution.

Understanding the Output
=========================

Output Files
------------

After running ``main_calibration_report.py``, you'll find:

.. code-block:: text

   results/
   ├── M1_posterior.png           # M1 posterior histograms
   ├── M1_trace.png               # M1 MCMC traces
   ├── M1_beta.png                # M1 β schedule
   ├── case2_M1_corner.png        # M1 corner plot
   ├── case2_M1_timeseries.png    # M1 time-series
   ├── (similar for M2, M3)
   ├── case2_M3_validation.png    # M3 validation
   └── bayesian_report.pdf        # Complete report

Interpreting Results
--------------------

**Posterior Distributions**
   Shows the estimated parameter distributions after calibration.
   Narrow peaks indicate well-constrained parameters.

**Trace Plots**
   Shows MCMC sampling history. Good mixing indicates convergence.

**Corner Plots**
   Shows pairwise parameter correlations. Useful for identifying dependencies.

**Time-Series Plots**
   Shows model predictions with uncertainty bands compared to observations.

**β Schedule**
   Shows the TMCMC tempering schedule. Smooth progression indicates stable sampling.

Key Metrics
-----------

* **RMSE**: Root Mean Square Error between predictions and observations
* **ESS**: Effective Sample Size - higher is better (>200 recommended)
* **Convergence**: Boolean flag indicating successful calibration

Configuration Modes
===================

DEBUG Mode (Fast)
-----------------

Default mode for quick testing:

.. code-block:: python

   # In src/config.py
   DEBUG = True

   # Results in:
   # - Reduced timesteps
   # - Fewer TMCMC stages
   # - Faster execution (~45s)
   # - Lower accuracy

Production Mode (Accurate)
---------------------------

For publication-quality results:

.. code-block:: python

   # In src/config.py
   DEBUG = False

   # Results in:
   # - Full timesteps
   # - More TMCMC stages
   # - Slower execution (~360s)
   # - Higher accuracy

Common Workflows
================

Workflow 1: Quick Parameter Estimation
---------------------------------------

.. code-block:: bash

   # 1. Enable DEBUG mode
   # Edit src/config.py: DEBUG = True

   # 2. Run calibration
   python main_calibration.py

   # 3. Check convergence in output
   # Look for "Convergence: M1=True, M2=True, M3=True"

Workflow 2: Publication-Ready Analysis
---------------------------------------

.. code-block:: bash

   # 1. Disable DEBUG mode
   # Edit src/config.py: DEBUG = False

   # 2. Adjust TMCMC settings (optional)
   # Edit src/config.py: N0 = 1000, stages = 20

   # 3. Generate full report
   python main_calibration_report.py

   # 4. Review PDF report
   # Open results/bayesian_report.pdf

Workflow 3: Parameter Sensitivity Study
----------------------------------------

.. code-block:: bash

   # 1. Run baseline
   python main_calibration_report.py

   # 2. Modify parameters in src/config.py
   # Try different priors, likelihood scaling, etc.

   # 3. Re-run and compare
   python main_calibration_report.py

   # 4. Compare corner plots and posteriors

Next Steps
==========

Now that you've run your first calibration:

* Learn about :doc:`configuration` options
* Explore :doc:`tutorials` for advanced usage
* Understand the :doc:`scientific_background`
* Check the :doc:`../api/core` for programmatic usage
