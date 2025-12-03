.. Biofilm Multi-Scale Parameter Estimation documentation master file

========================================================
Biofilm Multi-Scale Parameter Estimation Documentation
========================================================

.. image:: https://img.shields.io/badge/python-3.9%20|%203.10%20|%203.11%20|%203.12-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python 3.9+

.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT
   :alt: License: MIT

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black
   :alt: Code style: black

Welcome to the documentation for **Biofilm Multi-Scale Parameter Estimation**, a sophisticated hierarchical Bayesian framework for calibrating multi-scale biofilm formation models using Time-Separated Mechanics (TSM) and Transitional Markov Chain Monte Carlo (TMCMC).

Overview
========

This project implements a sophisticated hierarchical Bayesian framework for calibrating multi-scale biofilm formation models. The methodology combines:

* **Time-Separated Mechanics (TSM)**: Efficient uncertainty quantification via analytical sensitivity analysis
* **Transitional MCMC (TMCMC)**: Robust sampling of complex posterior distributions
* **Hierarchical Updating**: Sequential parameter estimation across three model scales (M1 → M2 → M3)

Key Features
============

* **14-dimensional parameter space** efficiently explored via hierarchical decomposition
* **Analytical sensitivities** computed with Numba acceleration (10-100x speedup)
* **Sharp likelihood peaks** handled via adaptive likelihood scaling
* **Publication-quality figures** generated automatically

Multi-Scale Modeling
--------------------

* **M1**: Coarse model (species 1-2, dt=1e-4, 80 timesteps)
* **M2**: Medium model (species 3-4, dt=1e-4, 100 timesteps)
* **M3**: Fine model (cross-interactions, dt=1e-4, 60 timesteps)

Quick Start
===========

Installation
------------

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/yourusername/biofilm_project.git
   cd biofilm_project

   # Create virtual environment (recommended)
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\\Scripts\\activate

   # Install dependencies
   pip install -r requirements.txt

   # Verify installation
   python -c "import numpy, scipy, numba; print('✓ All dependencies installed')"

Basic Usage
-----------

Run hierarchical Bayesian calibration:

.. code-block:: bash

   # Run calibration (DEBUG mode: fast)
   python main_calibration.py

   # Generate complete analysis with figures and PDF report
   python main_calibration_report.py

   # Run forward simulation with true parameters
   python main_simulation.py

Documentation Contents
======================

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user_guide/installation
   user_guide/quickstart
   user_guide/configuration
   user_guide/tutorials
   user_guide/scientific_background

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/core
   api/solvers
   api/inference
   api/visualization
   api/utilities

.. toctree::
   :maxdepth: 1
   :caption: Development

   development/testing
   development/contributing
   development/changelog

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
