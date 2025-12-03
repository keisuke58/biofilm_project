# Biofilm Project Examples

This directory contains interactive Jupyter notebooks demonstrating various aspects of the biofilm parameter estimation framework.

## 📚 Available Notebooks

### 1. Basic Calibration (`01_basic_calibration.ipynb`)
**Difficulty:** Beginner | **Time:** ~10 minutes

A simple end-to-end example of hierarchical Bayesian calibration:
- Generate synthetic data
- Run M1 → M2 → M3 calibration
- Visualize posterior distributions
- Compare with ground truth

**Learn:** Basic workflow, configuration, interpretation of results

---

### 2. Sensitivity Analysis (`02_sensitivity_analysis.ipynb`)
**Difficulty:** Intermediate | **Time:** ~15 minutes

Explore parameter sensitivity using Time-Separated Mechanics:
- Compute analytical sensitivities
- Visualize parameter influence over time
- Identify most influential parameters
- Compare with finite differences

**Learn:** TSM algorithm, sensitivity interpretation, parameter importance

---

### 3. Custom Prior Distributions (`03_custom_priors.ipynb`)
**Difficulty:** Intermediate | **Time:** ~15 minutes

How to use custom priors for Bayesian inference:
- Define informative Gaussian priors
- Use truncated distributions
- Implement custom prior samplers
- Compare uniform vs. informative priors

**Learn:** Prior specification, Bayesian reasoning, prior sensitivity

---

### 4. Uncertainty Quantification (`04_uncertainty_quantification.ipynb`)
**Difficulty:** Advanced | **Time:** ~20 minutes

Complete uncertainty quantification workflow:
- Parameter uncertainty (posterior distributions)
- Predictive uncertainty (forward propagation)
- Credible intervals and confidence bands
- Posterior predictive checks
- Model validation

**Learn:** UQ techniques, posterior analysis, validation methods

---

### 5. Advanced Visualization (`05_advanced_visualization.ipynb`)
**Difficulty:** Intermediate | **Time:** ~15 minutes

Create publication-quality figures:
- Corner plots for parameter correlations
- Time-series with uncertainty bands
- TMCMC diagnostics (β schedule, ESS)
- Custom matplotlib styling
- Export for publications

**Learn:** Visualization best practices, matplotlib customization, figure export

---

## 🚀 Getting Started

### Prerequisites

1. Install Jupyter and plotting dependencies:

```bash
pip install jupyter ipykernel ipywidgets
pip install -r requirements.txt
```

2. Launch Jupyter:

```bash
jupyter notebook
# or
jupyter lab
```

3. Open any notebook and run cells sequentially

### Quick Start

Start with `01_basic_calibration.ipynb` if you're new to the project. Each notebook is self-contained and can be run independently.

## 📊 Example Data

The `data/` directory contains synthetic and example datasets:
- `synthetic_biofilm.npz`: Generated from true parameters
- `noisy_observations.npz`: With added measurement noise
- `experimental_*.csv`: Example experimental data (if available)

## 🎓 Learning Path

**For New Users:**
1. Start with `01_basic_calibration.ipynb`
2. Try `05_advanced_visualization.ipynb` to explore results
3. Move to `02_sensitivity_analysis.ipynb` for deeper understanding

**For Researchers:**
1. Review `02_sensitivity_analysis.ipynb` for parameter importance
2. Study `04_uncertainty_quantification.ipynb` for UQ workflow
3. Customize `03_custom_priors.ipynb` for your application

**For Developers:**
- All notebooks show programmatic API usage
- Easy to adapt for your own workflows
- Code can be extracted and used in scripts

## 💡 Tips

- **Run sequentially**: Execute cells in order for best results
- **Restart kernel**: Use "Kernel → Restart & Run All" to ensure reproducibility
- **Modify parameters**: Experiment with different CONFIG values
- **Save results**: Figures are saved to `examples/figures/`
- **Random seeds**: Set `np.random.seed()` for reproducible random sampling

## 🔧 Troubleshooting

**Kernel crashes:**
- Reduce `N0` (number of samples) in CONFIG
- Enable `DEBUG = True` for faster execution
- Check available RAM (8GB recommended)

**Import errors:**
```bash
# Make sure you're in the project root
cd /path/to/biofilm_project
jupyter notebook
```

**Slow execution:**
- Use DEBUG mode initially
- Reduce `maxtimestep` for testing
- Check if Numba JIT is working (first run is slow, subsequent runs are fast)

## 📖 Further Reading

- **Documentation**: See `docs/` for comprehensive guides
- **API Reference**: Detailed function documentation
- **Scientific Background**: Mathematical formulation and algorithms
- **Testing**: See `tests/` for more usage examples

## 🤝 Contributing

Found an issue or have an idea for a new example? Please:
1. Open an issue describing the problem/idea
2. Submit a pull request with your notebook
3. Follow the existing notebook structure and style

## 📝 Notebook Structure

Each notebook follows this structure:
1. **Overview**: What you'll learn
2. **Setup**: Imports and configuration
3. **Theory** (optional): Brief mathematical background
4. **Implementation**: Step-by-step code with explanations
5. **Visualization**: Results and interpretation
6. **Exercises** (optional): Try-it-yourself tasks
7. **Next Steps**: What to explore next

## ⚡ Performance Notes

- First run with Numba is slow (JIT compilation)
- Subsequent runs are 10-100x faster
- Use DEBUG mode for quick iterations
- Production mode for final results
- See `tools/profile_performance.py` for benchmarking

## 📧 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/biofilm_project/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/biofilm_project/discussions)
- **Documentation**: [Full Documentation](../docs/build/html/index.html)

---

**Happy Learning!** 🎉
