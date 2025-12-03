=====================
Scientific Background
=====================

This page provides the scientific and mathematical background for the Biofilm Multi-Scale Parameter Estimation framework.

Overview
========

The framework implements a hierarchical Bayesian approach to calibrate multi-scale biofilm formation models. The methodology combines three key components:

1. **Physical Model**: Coupled PDEs for biofilm species dynamics
2. **Statistical Framework**: Hierarchical Bayesian inference
3. **Computational Methods**: TSM and TMCMC algorithms

Physical Model
==============

Governing Equations
-------------------

The biofilm formation is modeled by a system of coupled partial differential equations (PDEs) for volume fractions φᵢ and porosities ψᵢ of four species (i = 1, 2, 3, 4):

.. math::

   \frac{\partial \phi_i}{\partial t} = -\nabla \cdot (\mu_i \nabla \phi_i) + g_i(\phi, \psi, \theta)

   \frac{\partial \psi_i}{\partial t} = -\nabla \cdot (\kappa_i \nabla \psi_i) + h_i(\phi, \psi, \theta)

where:

* φᵢ: Volume fraction of species i
* ψᵢ: Porosity associated with species i
* μᵢ: Diffusion coefficient for volume fraction
* κᵢ: Diffusion coefficient for porosity
* gᵢ, hᵢ: Nonlinear source/sink terms (growth, interaction)
* θ: Parameter vector (14-dimensional)

Conservation Constraint
-----------------------

The volume fractions must satisfy:

.. math::

   \sum_{i=1}^{4} \phi_i + \phi_0 = 1

where φ₀ is the water/void fraction.

Growth and Interaction Terms
-----------------------------

The nonlinear terms model species growth and interactions:

.. math::

   g_i(\phi, \psi, \theta) = r_i \phi_i (1 - \sum_j \phi_j) - \sum_{j \neq i} a_{ij} \phi_i \phi_j

   h_i(\phi, \psi, \theta) = b_i \psi_i (1 - \psi_i) + \sum_j c_{ij} \psi_i \psi_j

where:

* rᵢ: Intrinsic growth rate of species i
* aᵢⱼ: Competition coefficient between species i and j
* bᵢ: Porosity self-interaction
* cᵢⱼ: Porosity cross-interaction

Multi-Scale Decomposition
--------------------------

The full 14-dimensional problem is decomposed into three hierarchical scales:

**M1 (Coarse Scale)**
   Species 1-2 only, parameters θ[0:5] = {a₁₁, a₁₂, a₂₂, b₁, b₂}

**M2 (Medium Scale)**
   Species 3-4 only, parameters θ[5:10] = {a₃₃, a₃₄, a₄₄, b₃, b₄}

**M3 (Fine Scale)**
   All species with cross-interactions, θ[10:14] = {a₁₃, a₁₄, a₂₃, a₂₄}

Parameter Vector
================

Full Parameter Description
---------------------------

The 14-dimensional parameter vector θ = [θ₀, θ₁, ..., θ₁₃]:

.. list-table::
   :header-rows: 1
   :widths: 10 20 40 30

   * - Index
     - Parameter
     - Physical Meaning
     - Prior Range
   * - 0
     - a₁₁
     - Species 1 self-competition
     - U(0, 3)
   * - 1
     - a₁₂
     - Species 1-2 competition
     - U(0, 3)
   * - 2
     - a₂₂
     - Species 2 self-competition
     - U(0, 3)
   * - 3
     - b₁
     - Species 1 porosity interaction
     - U(0, 3)
   * - 4
     - b₂
     - Species 2 porosity interaction
     - U(0, 3)
   * - 5
     - a₃₃
     - Species 3 self-competition
     - U(0, 3)
   * - 6
     - a₃₄
     - Species 3-4 competition
     - U(0, 3)
   * - 7
     - a₄₄
     - Species 4 self-competition
     - U(0, 3)
   * - 8
     - b₃
     - Species 3 porosity interaction
     - U(0, 3)
   * - 9
     - b₄
     - Species 4 porosity interaction
     - U(0, 3)
   * - 10
     - a₁₃
     - Species 1-3 cross-competition
     - U(0, 3)
   * - 11
     - a₁₄
     - Species 1-4 cross-competition
     - U(0, 3)
   * - 12
     - a₂₃
     - Species 2-3 cross-competition
     - U(0, 3)
   * - 13
     - a₂₄
     - Species 2-4 cross-competition
     - U(0, 3)

True Parameter Values
---------------------

For validation and testing, the ground truth parameters are:

.. code-block:: python

   θ_true = [
       0.8, 2.0, 1.0, 0.1, 0.2,  # M1: Species 1-2
       1.5, 1.0, 2.0, 0.3, 0.4,  # M2: Species 3-4
       2.0, 1.0, 2.0, 1.0        # M3: Cross-interactions
   ]

Bayesian Framework
==================

Bayes' Theorem
--------------

The posterior distribution is computed via Bayes' theorem:

.. math::

   p(\theta | \mathcal{D}) = \frac{p(\mathcal{D} | \theta) p(\theta)}{p(\mathcal{D})}

where:

* p(θ|D): Posterior distribution (what we want)
* p(D|θ): Likelihood function
* p(θ): Prior distribution
* p(D): Evidence (normalization constant)

Likelihood Function
-------------------

The likelihood assumes Gaussian observation errors:

.. math::

   p(\mathcal{D} | \theta) = \prod_{i=1}^{N_{\text{data}}} \mathcal{N}(y_i | f(\theta, t_i), \sigma^2)

where:

* yᵢ: Observed data at time tᵢ
* f(θ, tᵢ): Model prediction with parameters θ
* σ²: Observation noise variance

In practice, we use:

.. math::

   \log p(\mathcal{D} | \theta) = -\frac{c}{2} \sum_{i=1}^{N_{\text{data}}} \frac{(y_i - f(\theta, t_i))^2}{\sigma^2}

where c is a scaling constant (``c_const`` in config).

Prior Distributions
-------------------

Uniform priors are used for all parameters:

.. math::

   p(\theta_j) = \begin{cases}
   \frac{1}{b_j - a_j} & \text{if } a_j \leq \theta_j \leq b_j \\
   0 & \text{otherwise}
   \end{cases}

For all parameters: [aⱼ, bⱼ] = [0, 3].

Hierarchical Structure
----------------------

The hierarchical approach sequences inference across scales:

1. **Prior for M1**: p(θ[0:5]) ~ U(0, 3)⁵

2. **Posterior from M1**: p(θ[0:5] | D₁) ∝ p(D₁ | θ[0:5]) p(θ[0:5])

3. **Prior for M2**: p(θ[0:10]) = p(θ[0:5] | D₁) × U(0, 3)⁵

4. **Posterior from M2**: p(θ[0:10] | D₁, D₂) ∝ p(D₂ | θ[0:10]) p(θ[0:10])

5. **Prior for M3**: p(θ) = p(θ[0:10] | D₁, D₂) × U(0, 3)⁴

6. **Final Posterior**: p(θ | D₁, D₂, D₃) ∝ p(D₃ | θ) p(θ)

Computational Methods
=====================

Time-Separated Mechanics (TSM)
-------------------------------

TSM provides efficient uncertainty quantification by computing analytical sensitivities.

**Key Idea**: Approximate the likelihood using linearization:

.. math::

   f(\theta) \approx f(\theta_0) + S(\theta_0) (\theta - \theta_0)

where S is the sensitivity matrix:

.. math::

   S_{ij} = \frac{\partial f_i}{\partial \theta_j}\bigg|_{\theta_0}

**Advantages**:

* Analytical computation of sensitivities
* No need for expensive finite differences
* Accelerated by Numba JIT compilation
* 10-100x speedup over standard methods

**Implementation**: See ``src/tsm.py``

Transitional MCMC (TMCMC)
--------------------------

TMCMC samples from complex posterior distributions using tempering.

**Tempering Schedule**: Introduce intermediate distributions:

.. math::

   p_\beta(\theta) \propto p(\mathcal{D} | \theta)^\beta p(\theta)

where β ∈ [0, 1]:

* β = 0: Prior distribution (easy to sample)
* β = 1: Posterior distribution (target)

**Algorithm**:

1. Initialize from prior: θ⁽⁰⁾ ~ p(θ)
2. For each stage j = 1, ..., J:

   a. Choose βⱼ to maintain target ESS
   b. Compute weights: wᵢ = p(D|θᵢ)^(βⱼ - βⱼ₋₁)
   c. Resample based on weights
   d. Apply MCMC kernel

3. Return samples from p₁(θ) = p(θ|D)

**Effective Sample Size (ESS)**:

.. math::

   \text{ESS} = \frac{(\sum_i w_i)^2}{\sum_i w_i^2}

Target: ESS > 0.5 × N₀

**Implementation**: See ``src/tmcmc.py``

Numerical Solver
----------------

The PDEs are solved using a Newton-Raphson method:

1. **Spatial Discretization**: Finite differences on uniform grid
2. **Time Integration**: Implicit Euler method
3. **Nonlinear Solver**: Newton-Raphson iteration

**Newton Update**:

.. math::

   \mathbf{x}^{k+1} = \mathbf{x}^k - J(\mathbf{x}^k)^{-1} F(\mathbf{x}^k)

where:

* **x**: State vector (φ, ψ)
* F: Residual function
* J: Jacobian matrix

**Numba Acceleration**: Critical loops are JIT-compiled for performance.

**Implementation**: See ``src/solver_newton.py`` and ``src/numerics.py``

Convergence Criteria
====================

TMCMC Convergence
-----------------

* **ESS Threshold**: ESS > 0.5 × N₀ at each stage
* **β Progression**: Smooth increase from 0 to 1
* **Acceptance Rate**: 20-50% for MCMC kernel

Newton Solver Convergence
--------------------------

* **Residual Tolerance**: ||F(x)|| < 10⁻⁸
* **Maximum Iterations**: 50
* **Relative Change**: ||x^(k+1) - x^k|| / ||x^k|| < 10⁻⁶

Hierarchical Updating
---------------------

* **M1 Convergence**: ESS_M1 > threshold
* **M2 Convergence**: ESS_M2 > threshold
* **M3 Convergence**: ESS_M3 > threshold
* **Overall**: All three stages converge

Uncertainty Quantification
===========================

Sources of Uncertainty
----------------------

1. **Parameter Uncertainty**: Uncertain θ
2. **Model Uncertainty**: Simplified physics
3. **Observation Uncertainty**: Measurement noise σ
4. **Numerical Uncertainty**: Discretization errors

Posterior Statistics
--------------------

From posterior samples {θ⁽ⁱ⁾}ᵢ₌₁ᴺ:

**Mean Estimate**:

.. math::

   \bar{\theta} = \frac{1}{N} \sum_{i=1}^N \theta^{(i)}

**Covariance**:

.. math::

   \Sigma = \frac{1}{N-1} \sum_{i=1}^N (\theta^{(i)} - \bar{\theta})(\theta^{(i)} - \bar{\theta})^T

**Credible Intervals** (95%):

.. math::

   [\theta_{0.025}, \theta_{0.975}]

Predictive Uncertainty
----------------------

Forward propagation of parameter uncertainty:

.. math::

   p(y | \mathcal{D}) = \int p(y | \theta) p(\theta | \mathcal{D}) d\theta

Approximated by:

.. math::

   p(y | \mathcal{D}) \approx \frac{1}{N} \sum_{i=1}^N p(y | \theta^{(i)})

Performance Considerations
===========================

Computational Complexity
------------------------

* **Full Forward Solve**: O(N_t × N_x × N_iter)

  - N_t: Number of timesteps
  - N_x: Number of spatial points
  - N_iter: Newton iterations

* **TSM Sensitivity**: O(N_θ × N_t × N_x)

  - N_θ: Number of parameters

* **TMCMC**: O(N₀ × N_stages × N_MCMC)

  - N₀: Sample size
  - N_stages: TMCMC stages (~15-20)
  - N_MCMC: MCMC steps per stage

Optimization Strategies
-----------------------

1. **Numba JIT**: 10-100x speedup for numerical kernels
2. **Sparse Data**: Use N_data << N_t observations
3. **TSM**: Avoid expensive finite differences
4. **Parallel TMCMC**: Evaluate likelihoods in parallel
5. **Adaptive β**: Minimize number of stages

Typical Performance
-------------------

On a modern CPU (Intel i7):

* **DEBUG Mode**: ~45s (N₀=500, reduced timesteps)
* **Production Mode**: ~360s (N₀=1000, full resolution)
* **High Accuracy**: ~1800s (N₀=2000, extended stages)

References
==========

Key Papers
----------

1. **Biofilm Modeling**: Mathematical models for biofilm formation
2. **TSM Method**: Time-Separated Mechanics for UQ
3. **TMCMC**: Transitional Markov Chain Monte Carlo
4. **Hierarchical Bayesian**: Multi-scale parameter estimation

Software
--------

* **NumPy**: https://numpy.org
* **SciPy**: https://scipy.org
* **Numba**: https://numba.pydata.org
* **Matplotlib**: https://matplotlib.org

Further Reading
===============

* :doc:`tutorials` - Practical examples
* :doc:`configuration` - Tuning parameters
* :doc:`../api/inference` - Implementation details
* :doc:`../development/testing` - Validation methods
