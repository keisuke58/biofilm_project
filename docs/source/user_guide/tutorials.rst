=========
Tutorials
=========

This section provides detailed tutorials for using the Biofilm Multi-Scale Parameter Estimation framework.

Tutorial 1: Basic Parameter Calibration
========================================

This tutorial walks through a complete parameter calibration workflow.

Objective
---------

Calibrate the 14-dimensional parameter vector for a biofilm model using hierarchical Bayesian updating.

Prerequisites
-------------

* Installed framework (see :doc:`installation`)
* Basic understanding of Bayesian inference
* Familiarity with Python

Step-by-Step Guide
-------------------

**Step 1: Understand the Problem**

We have a biofilm model with 4 species and 14 parameters:

* M1 parameters (5): Species 1-2 interactions
* M2 parameters (5): Species 3-4 interactions
* M3 parameters (4): Cross-species interactions

**Step 2: Generate Synthetic Data**

.. code-block:: python

   from src.config import CONFIG, TRUE_PARAMS
   from src.solver_newton import solve_biofilm
   import numpy as np

   # Run forward simulation with true parameters
   theta_true = TRUE_PARAMS
   solution = solve_biofilm(theta_true, CONFIG["M3"])

   # Add observation noise
   sigma_obs = 0.005
   observations = solution + np.random.randn(*solution.shape) * sigma_obs

**Step 3: Configure Calibration**

Edit ``src/config.py``:

.. code-block:: python

   DEBUG = True  # Start with fast mode
   CONFIG = {
       "N0": 500,
       "stages": 15,
       "Ndata": 20,
   }

**Step 4: Run Hierarchical Calibration**

.. code-block:: bash

   python main_calibration.py

**Step 5: Analyze Results**

.. code-block:: python

   # After calibration
   from src.hierarchical import hierarchical_case2

   results = hierarchical_case2(CONFIG)

   # Extract posterior statistics
   theta_est = results.theta_final
   print(f"True:      {TRUE_PARAMS}")
   print(f"Estimated: {theta_est}")
   print(f"Error:     {np.abs(theta_est - TRUE_PARAMS)}")

Tutorial 2: Custom Prior Distributions
=======================================

Learn how to modify prior distributions for parameters.

Custom Uniform Priors
---------------------

.. code-block:: python

   # In your script
   import numpy as np

   # Define custom prior bounds
   prior_bounds_M1 = {
       "a11": (0.5, 1.5),  # Narrow prior around true value
       "a12": (1.5, 2.5),
       "a22": (0.5, 1.5),
       "b1": (0.0, 0.5),
       "b2": (0.0, 0.5),
   }

   # Convert to arrays
   lower_M1 = [prior_bounds_M1[key][0] for key in ["a11", "a12", "a22", "b1", "b2"]]
   upper_M1 = [prior_bounds_M1[key][1] for key in ["a11", "a12", "a22", "b1", "b2"]]

Informative Gaussian Priors
----------------------------

.. code-block:: python

   from scipy.stats import truncnorm

   def gaussian_prior(mean, std, lower, upper, size=1000):
       """Sample from truncated normal distribution."""
       a, b = (lower - mean) / std, (upper - mean) / std
       return truncnorm.rvs(a, b, loc=mean, scale=std, size=size)

   # Use for initial samples
   samples_M1 = np.array([
       gaussian_prior(0.8, 0.2, 0.0, 3.0, N0),  # a11
       gaussian_prior(2.0, 0.3, 0.0, 3.0, N0),  # a12
       # ... etc
   ]).T

Tutorial 3: Forward Uncertainty Propagation
============================================

Propagate parameter uncertainty through the model.

Using Posterior Samples
------------------------

.. code-block:: python

   from src.posterior_simulator_tsm import generate_posterior_predictions
   import matplotlib.pyplot as plt

   # Get posterior samples from M1 calibration
   samples_M1 = results.tmcmc_M1.samples[-1]

   # Select subset of samples
   n_pred = 100
   idx = np.random.choice(len(samples_M1), n_pred, replace=False)
   samples_subset = samples_M1[idx]

   # Generate predictions
   predictions = []
   for theta_sample in samples_subset:
       pred = solve_biofilm(theta_sample, CONFIG["M1"])
       predictions.append(pred)

   predictions = np.array(predictions)

   # Compute statistics
   mean_pred = np.mean(predictions, axis=0)
   std_pred = np.std(predictions, axis=0)
   lower_95 = np.percentile(predictions, 2.5, axis=0)
   upper_95 = np.percentile(predictions, 97.5, axis=0)

   # Plot uncertainty bands
   t = np.linspace(0, CONFIG["M1"]["maxtimestep"] * CONFIG["M1"]["dt"], len(mean_pred))
   plt.fill_between(t, lower_95[:, 0], upper_95[:, 0], alpha=0.3, label='95% CI')
   plt.plot(t, mean_pred[:, 0], label='Mean')
   plt.xlabel('Time')
   plt.ylabel('Species 1 Volume Fraction')
   plt.legend()
   plt.savefig('uncertainty_propagation.png')

Tutorial 4: Sensitivity Analysis with TSM
==========================================

Perform efficient sensitivity analysis using Time-Separated Mechanics.

Computing Sensitivities
------------------------

.. code-block:: python

   from src.tsm import compute_tsm_sensitivity

   # Reference parameters
   theta_ref = TRUE_PARAMS[:5]  # M1 parameters

   # Compute sensitivities
   S = compute_tsm_sensitivity(theta_ref, CONFIG["M1"])

   # S is the sensitivity matrix: ∂y/∂θ
   print(f"Sensitivity matrix shape: {S.shape}")
   # (n_outputs, n_parameters)

   # Most influential parameter at each time
   max_influence = np.argmax(np.abs(S), axis=1)
   print(f"Most influential parameters: {max_influence}")

Visualizing Sensitivities
--------------------------

.. code-block:: python

   import matplotlib.pyplot as plt

   # Plot sensitivity over time
   t = np.linspace(0, CONFIG["M1"]["maxtimestep"] * CONFIG["M1"]["dt"], S.shape[0])

   fig, axes = plt.subplots(5, 1, figsize=(10, 12))
   param_names = ['a11', 'a12', 'a22', 'b1', 'b2']

   for i, (ax, name) in enumerate(zip(axes, param_names)):
       ax.plot(t, S[:, i])
       ax.set_ylabel(f'∂y/∂{name}')
       ax.grid(True)

   axes[-1].set_xlabel('Time')
   plt.tight_layout()
   plt.savefig('sensitivity_analysis.png')

Tutorial 5: Custom TMCMC Configuration
=======================================

Fine-tune TMCMC settings for challenging problems.

Adaptive β Schedule
-------------------

.. code-block:: python

   from src.tmcmc import tmcmc_adaptive

   # Custom TMCMC configuration
   tmcmc_config = {
       "N0": 1000,              # More samples
       "target_ess": 0.95,      # High ESS target (conservative)
       "min_beta_step": 0.01,   # Minimum β increment
       "max_beta_step": 0.2,    # Maximum β increment
       "resample_threshold": 0.5, # Resample when ESS < 0.5*N0
   }

   # Run TMCMC with custom config
   results = tmcmc_adaptive(
       log_likelihood_fn,
       prior_samples,
       config=tmcmc_config
   )

Monitoring Convergence
----------------------

.. code-block:: python

   import matplotlib.pyplot as plt

   # Plot β schedule
   beta_values = results.beta_schedule
   ess_values = results.ess_history

   fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

   # β progression
   ax1.plot(beta_values, marker='o')
   ax1.set_xlabel('Stage')
   ax1.set_ylabel('β')
   ax1.set_title('Tempering Schedule')
   ax1.grid(True)

   # ESS evolution
   ax2.plot(ess_values, marker='o')
   ax2.axhline(y=0.5*tmcmc_config["N0"], color='r', linestyle='--', label='Threshold')
   ax2.set_xlabel('Stage')
   ax2.set_ylabel('ESS')
   ax2.set_title('Effective Sample Size')
   ax2.legend()
   ax2.grid(True)

   plt.tight_layout()
   plt.savefig('tmcmc_diagnostics.png')

Tutorial 6: Programmatic Usage
===============================

Use the framework programmatically in your own scripts.

Complete Example
----------------

.. code-block:: python

   import numpy as np
   from src.config import CONFIG
   from src.hierarchical import hierarchical_case2
   from src.viz_paper import plot_corner, plot_timeseries
   import matplotlib.pyplot as plt

   # Configure
   CONFIG["N0"] = 500
   CONFIG["stages"] = 15
   CONFIG["Ndata"] = 20

   # Run hierarchical calibration
   print("Running hierarchical Bayesian calibration...")
   results = hierarchical_case2(CONFIG)

   # Extract results
   theta_M1 = results.theta_M1
   theta_M2 = results.theta_M2
   theta_M3 = results.theta_final

   samples_M1 = results.tmcmc_M1.samples[-1]
   samples_M2 = results.tmcmc_M2.samples[-1]
   samples_M3 = results.tmcmc_M3.samples[-1]

   # Compute statistics
   print("\nM1 Parameter Estimates:")
   for i, val in enumerate(theta_M1):
       std = np.std(samples_M1[:, i])
       print(f"  θ[{i}] = {val:.4f} ± {std:.4f}")

   # Generate visualizations
   print("\nGenerating visualizations...")

   # Corner plot
   fig = plot_corner(samples_M1, labels=[f'θ{i}' for i in range(5)])
   plt.savefig('results/custom_corner_M1.png', dpi=300)
   plt.close()

   # Time series
   fig = plot_timeseries(results, "M1")
   plt.savefig('results/custom_timeseries_M1.png', dpi=300)
   plt.close()

   print("✓ Analysis complete!")

Tutorial 7: Validation and Model Checking
==========================================

Validate your calibration results.

Posterior Predictive Checks
----------------------------

.. code-block:: python

   from src.validation_m3 import validate_posterior_predictions

   # Generate posterior predictions
   n_samples = 200
   idx = np.random.choice(len(samples_M3), n_samples)
   posterior_samples = samples_M3[idx]

   predictions = []
   for theta in posterior_samples:
       pred = solve_biofilm(theta, CONFIG["M3"])
       predictions.append(pred)

   predictions = np.array(predictions)

   # Compare with observations
   residuals = predictions - observations[np.newaxis, ...]
   standardized_residuals = residuals / sigma_obs

   # Check if ~95% within ±2σ
   within_2sigma = np.abs(standardized_residuals) < 2
   coverage = np.mean(within_2sigma)
   print(f"Coverage within 2σ: {coverage*100:.1f}% (expect ~95%)")

   # Plot residuals
   plt.figure(figsize=(10, 6))
   plt.hist(standardized_residuals.flatten(), bins=50, density=True, alpha=0.7)
   x = np.linspace(-4, 4, 100)
   plt.plot(x, np.exp(-x**2/2)/np.sqrt(2*np.pi), 'r-', label='N(0,1)')
   plt.xlabel('Standardized Residuals')
   plt.ylabel('Density')
   plt.legend()
   plt.savefig('residual_analysis.png')

Cross-Validation
----------------

.. code-block:: python

   # Leave-one-out validation
   from sklearn.model_selection import KFold

   kf = KFold(n_splits=5, shuffle=True, random_state=42)
   errors = []

   for train_idx, test_idx in kf.split(observations):
       # Train on subset
       obs_train = observations[train_idx]

       # Calibrate (simplified)
       results_cv = hierarchical_case2(CONFIG, data=obs_train)

       # Predict on test set
       pred_test = solve_biofilm(results_cv.theta_final, CONFIG["M3"])[test_idx]
       obs_test = observations[test_idx]

       # Compute error
       error = np.mean((pred_test - obs_test)**2)
       errors.append(error)

   print(f"Cross-validation MSE: {np.mean(errors):.6f} ± {np.std(errors):.6f}")

Best Practices
==============

1. **Start with DEBUG Mode**
   Always test with ``DEBUG=True`` before production runs.

2. **Check Convergence**
   Monitor ESS, β schedule, and trace plots.

3. **Validate Results**
   Use posterior predictive checks and cross-validation.

4. **Document Configuration**
   Save your ``config.py`` with results for reproducibility.

5. **Use Version Control**
   Track changes to code and configurations.

6. **Profile Performance**
   Use ``tools/profile_performance.py`` for optimization.

Common Pitfalls
===============

1. **Overconfident Priors**
   Too narrow priors can exclude true values.

2. **High Likelihood Scaling**
   Very high ``c_const`` can cause numerical issues.

3. **Insufficient Samples**
   Low ``N0`` leads to poor posterior approximation.

4. **Ignoring Warnings**
   Convergence warnings should not be ignored.

5. **Not Validating**
   Always validate your results against held-out data.

Next Steps
==========

* Dive deeper into :doc:`scientific_background`
* Explore the :doc:`../api/core` for implementation details
* Check :doc:`../development/testing` for quality assurance
