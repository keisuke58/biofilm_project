# Biofilm Project Documentation

This directory contains the Sphinx documentation for the Biofilm Multi-Scale Parameter Estimation project.

## Building the Documentation

### Prerequisites

Install documentation dependencies:

```bash
pip install -r requirements-docs.txt
```

### Build HTML Documentation

```bash
cd docs
make html
```

The generated HTML files will be in `build/html/`. Open `build/html/index.html` in your browser to view the documentation.

### Build PDF Documentation (Optional)

Requires LaTeX installation:

```bash
cd docs
make latexpdf
```

### Clean Build Files

```bash
cd docs
make clean
```

## Documentation Structure

```
docs/
├── source/                      # Documentation source files
│   ├── index.rst               # Main documentation page
│   ├── conf.py                 # Sphinx configuration
│   ├── user_guide/             # User guides and tutorials
│   │   ├── installation.rst
│   │   ├── quickstart.rst
│   │   ├── configuration.rst
│   │   ├── tutorials.rst
│   │   └── scientific_background.rst
│   ├── api/                    # API reference
│   │   ├── core.rst
│   │   ├── solvers.rst
│   │   ├── inference.rst
│   │   ├── visualization.rst
│   │   └── utilities.rst
│   └── development/            # Developer documentation
│       ├── testing.rst
│       ├── contributing.rst
│       └── changelog.rst
├── build/                      # Generated documentation (git-ignored)
├── Makefile                    # Build script
└── README.md                   # This file
```

## Documentation Formats

The documentation is written in reStructuredText (.rst) format and can be built into:

* **HTML**: For web viewing (default)
* **PDF**: For offline reading
* **ePub**: For e-readers
* **Man pages**: For Unix systems

## Contributing to Documentation

When adding new features or making changes:

1. Update relevant documentation pages
2. Add docstrings to new functions/classes
3. Rebuild documentation to check for errors
4. Verify all links and references work

## Online Documentation

The documentation may be hosted online at:

* Read the Docs: `https://biofilm-project.readthedocs.io` (if configured)
* GitHub Pages: `https://username.github.io/biofilm_project/` (if configured)

## Troubleshooting

**Import errors during build:**
- These are expected if numpy/scipy aren't in the build environment
- The documentation will still build successfully using mock imports

**Missing dependencies:**
```bash
pip install -r ../requirements-docs.txt
```

**Build errors:**
```bash
# Clean and rebuild
make clean
make html
```

## Additional Resources

* [Sphinx Documentation](https://www.sphinx-doc.org/)
* [reStructuredText Primer](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html)
* [Read the Docs Tutorial](https://docs.readthedocs.io/)
