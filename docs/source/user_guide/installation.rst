============
Installation
============

This guide will help you install and set up the Biofilm Multi-Scale Parameter Estimation framework.

Requirements
============

System Requirements
-------------------

* Python 3.9 or higher (tested on 3.9, 3.10, 3.11, 3.12)
* At least 4GB of RAM (8GB recommended for large-scale simulations)
* Linux, macOS, or Windows operating system

Python Dependencies
-------------------

The following Python packages are required:

* **NumPy** (>=1.21.0): Numerical computing
* **SciPy** (>=1.7.0): Scientific computing and optimization
* **Numba** (>=0.56.0): JIT compilation for performance
* **Matplotlib** (>=3.5.0): Visualization
* **tqdm** (>=4.60.0): Progress tracking

Optional Dependencies
---------------------

For development and testing:

* **pytest** (>=7.0.0): Unit testing
* **pytest-cov** (>=3.0.0): Code coverage
* **black** (>=22.0.0): Code formatting
* **flake8** (>=4.0.0): Linting

Installation Steps
==================

Step 1: Clone the Repository
-----------------------------

.. code-block:: bash

   git clone https://github.com/yourusername/biofilm_project.git
   cd biofilm_project

Step 2: Create a Virtual Environment (Recommended)
---------------------------------------------------

Using venv (Python 3.3+):

.. code-block:: bash

   # Create virtual environment
   python -m venv .venv

   # Activate on Linux/macOS
   source .venv/bin/activate

   # Activate on Windows
   .venv\\Scripts\\activate

Using conda:

.. code-block:: bash

   # Create conda environment
   conda create -n biofilm python=3.11

   # Activate environment
   conda activate biofilm

Step 3: Install Dependencies
-----------------------------

Production dependencies:

.. code-block:: bash

   pip install -r requirements.txt

Development dependencies (optional):

.. code-block:: bash

   pip install -r requirements-dev.txt

Documentation dependencies (if building docs):

.. code-block:: bash

   pip install -r requirements-docs.txt

Step 4: Verify Installation
----------------------------

Run the following command to verify all dependencies are installed correctly:

.. code-block:: bash

   python -c "import numpy, scipy, numba, matplotlib; print('✓ All dependencies installed')"

You should see the message "✓ All dependencies installed".

Step 5: Run a Test Simulation (Optional)
-----------------------------------------

To verify the installation is working correctly, run a quick test:

.. code-block:: bash

   python main_simulation.py

This should generate a file ``forward_simulation.png`` in your current directory.

Troubleshooting
===============

Numba Installation Issues
--------------------------

If you encounter issues installing Numba, try:

.. code-block:: bash

   # For conda users
   conda install numba

   # For pip users with specific version
   pip install numba==0.56.0

On Windows, you may need to install the Microsoft C++ Build Tools.

Import Errors
-------------

If you see ``ModuleNotFoundError``, ensure:

1. Your virtual environment is activated
2. All dependencies are installed: ``pip list``
3. You're running Python from the correct environment

Memory Issues
-------------

For large-scale simulations, if you encounter memory errors:

1. Reduce the number of TMCMC samples (``N0`` in config)
2. Use DEBUG mode for faster testing
3. Close other applications to free up memory

Next Steps
==========

After successful installation:

* Read the :doc:`quickstart` guide to run your first calibration
* Explore the :doc:`configuration` options
* Check out the :doc:`tutorials` for detailed examples
