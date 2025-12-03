# Automated Analysis Pipeline

Complete guide for running all computational analyses automatically on GitHub Actions.

## 📋 Table of Contents

- [Overview](#overview)
- [Automated Workflows](#automated-workflows)
- [Manual Triggers](#manual-triggers)
- [Scheduled Runs](#scheduled-runs)
- [Artifacts](#artifacts)
- [Usage Examples](#usage-examples)
- [Configuration](#configuration)

## 🎯 Overview

The project includes comprehensive GitHub Actions workflows that automatically execute:

1. ✅ **TMCMC Calibration** - Full hierarchical Bayesian calibration
2. ✅ **TSM Sensitivity Analysis** - Parameter sensitivity computation
3. ✅ **Forward Simulation** - Predictive simulations
4. ✅ **Predictive Uncertainty** - UQ with posterior samples
5. ✅ **Report Generation** - PDF reports with all results
6. ✅ **Notebook Conversion** - Convert to HTML/PDF
7. ✅ **Performance Profiling** - Speed and memory analysis
8. ✅ **Benchmarking** - Performance tracking
9. ✅ **PDF Generation** - LaTeX compilation

## 🚀 Automated Workflows

### Main Pipeline: `automated-analysis.yml`

**Comprehensive pipeline that runs everything automatically.**

**Triggers:**
- Push to `main`/`master` branch
- Pull requests
- Weekly schedule (Sunday 00:00 UTC)
- Manual dispatch

**Jobs:**
1. TMCMC Calibration (30 min)
2. TSM Sensitivity Analysis (20 min)
3. Forward Simulation & UQ (25 min)
4. Profiling & Benchmarking (20 min)
5. Notebook Conversion (30 min)
6. Report Generation (30 min)
7. Summary Generation

**Total Time:** ~2.5 hours (jobs run in parallel)

### Quick Analysis: `quick-analysis.yml`

**Fast DEBUG mode analysis for testing.**

**Triggers:**
- Manual only

**Options:**
- Calibration only (~5 min)
- Sensitivity only (~3 min)
- Simulation only (~2 min)
- Profiling only (~5 min)
- All (~15 min)

## 🎮 Manual Triggers

### Run Complete Pipeline

```bash
# Via GitHub UI:
Actions → "Complete Automated Analysis Pipeline" → Run workflow

# Choose mode:
- Debug (fast, ~30 min total)
- Production (slow, ~2.5 hours)

# Via GitHub CLI:
gh workflow run automated-analysis.yml -f run_mode=debug
gh workflow run automated-analysis.yml -f run_mode=production
```

### Run Quick Analysis

```bash
# Via GitHub UI:
Actions → "Quick Analysis" → Run workflow → Select type

# Via CLI:
gh workflow run quick-analysis.yml -f analysis_type=calibration
gh workflow run quick-analysis.yml -f analysis_type=sensitivity
gh workflow run quick-analysis.yml -f analysis_type=simulation
gh workflow run quick-analysis.yml -f analysis_type=profiling
gh workflow run quick-analysis.yml -f analysis_type=all
```

## 📅 Scheduled Runs

The main pipeline runs automatically:

**Weekly Schedule:**
- Every Sunday at 00:00 UTC
- Runs in DEBUG mode
- Results retained for 90 days

**Customize Schedule:**

Edit `.github/workflows/automated-analysis.yml`:

```yaml
schedule:
  - cron: '0 0 * * 0'  # Weekly on Sunday
  # - cron: '0 0 1 * *'  # Monthly on 1st
  # - cron: '0 0 * * 1-5'  # Weekdays
```

## 📦 Artifacts

All results are uploaded as artifacts:

### Artifact Structure

| Artifact Name | Contents | Retention |
|--------------|----------|-----------|
| `tmcmc-results` | Calibration outputs (.npz, .npy) | 30 days |
| `sensitivity-results` | TSM sensitivity (.npy, .png) | 30 days |
| `simulation-results` | Forward sims & UQ | 30 days |
| `profiling-results` | Performance metrics | 30 days |
| `converted-notebooks` | HTML/PDF notebooks | 30 days |
| `complete-analysis-results` | Everything combined | 90 days |

### Download Artifacts

**Via GitHub UI:**
1. Go to Actions tab
2. Select workflow run
3. Scroll to "Artifacts" section
4. Click to download

**Via GitHub CLI:**
```bash
# List artifacts
gh run list --workflow=automated-analysis.yml

# Download specific run
gh run download RUN_ID

# Download specific artifact
gh run download RUN_ID -n complete-analysis-results
```

**Via API:**
```bash
# Get artifact URL
curl -H "Authorization: token $GITHUB_TOKEN" \
  https://api.github.com/repos/USER/REPO/actions/runs/RUN_ID/artifacts

# Download artifact
curl -L -H "Authorization: token $GITHUB_TOKEN" \
  ARTIFACT_URL -o results.zip
```

## 💡 Usage Examples

### Example 1: Weekly Production Run

```yaml
# In automated-analysis.yml
schedule:
  - cron: '0 0 * * 0'  # Sunday 00:00

# Automatically runs in production mode
# Results archived for 90 days
```

### Example 2: Quick Test Before PR

```bash
# Test calibration locally first
python main_calibration.py

# Then trigger quick analysis
gh workflow run quick-analysis.yml -f analysis_type=calibration

# Check results
gh run watch
```

### Example 3: Monthly Benchmark Tracking

```bash
# Run full pipeline monthly
gh workflow run automated-analysis.yml -f run_mode=production

# Download benchmarks
gh run download --name profiling-results

# Compare with previous
python tools/benchmark_suite.py --compare baseline.json
```

### Example 4: Notebook Testing

```bash
# Run with notebook conversion
gh workflow run automated-analysis.yml -f skip_notebooks=false

# Download converted notebooks
gh run download --name converted-notebooks

# Open in browser
open results/notebooks/html/01_basic_calibration.html
```

## ⚙️ Configuration

### Workflow Environment Variables

Edit in `.github/workflows/automated-analysis.yml`:

```yaml
env:
  PYTHON_VERSION: '3.11'      # Python version
  RESULTS_DIR: 'results'      # Output directory
  DEBUG: 'true'               # Debug mode
  N0: '200'                   # TMCMC samples
  NDATA: '10'                 # Data points
```

### Job Timeouts

Adjust based on your needs:

```yaml
jobs:
  tmcmc-calibration:
    timeout-minutes: 30  # Increase for production
```

### Artifact Retention

```yaml
- uses: actions/upload-artifact@v4
  with:
    retention-days: 30  # 1-90 days
```

### Conditional Execution

Skip jobs based on conditions:

```yaml
jobs:
  notebook-conversion:
    if: ${{ !inputs.skip_notebooks }}
```

## 🔍 Monitoring

### Check Workflow Status

```bash
# List recent runs
gh run list --workflow=automated-analysis.yml

# Watch active run
gh run watch

# View run logs
gh run view RUN_ID --log
```

### View Job Summary

Each run generates a summary showing:
- Job status (✅ success, ❌ failed)
- Execution times
- Artifact links
- Error messages (if any)

**Access:** Actions tab → Run → Summary

### Email Notifications

Configure in GitHub:
1. Settings → Notifications
2. Enable "Actions" notifications
3. Choose: All workflows / Failed only

## 🐛 Troubleshooting

### Common Issues

**Job timeout:**
```yaml
# Increase timeout
timeout-minutes: 60  # default is 30
```

**Out of memory:**
```yaml
# Use DEBUG mode or reduce samples
env:
  DEBUG: 'true'
  N0: '100'
```

**Artifact too large:**
```yaml
# Compress before upload
- run: tar -czf results.tar.gz results/
- uses: actions/upload-artifact@v4
  with:
    path: results.tar.gz
```

**LaTeX errors:**
```yaml
# Install additional packages
- run: |
    sudo apt-get install texlive-latex-extra
    sudo apt-get install texlive-fonts-extra
```

### Debug Failed Workflows

1. **Check logs:** Actions → Run → Failed job → View logs
2. **Re-run with debug logging:**
   ```bash
   # Enable runner debug logs
   gh secret set ACTIONS_RUNNER_DEBUG --body true
   gh secret set ACTIONS_STEP_DEBUG --body true
   ```
3. **Test locally:**
   ```bash
   # Use act to test locally
   act -W .github/workflows/automated-analysis.yml
   ```

## 🎨 Customization

### Add Custom Analysis

Edit `automated-analysis.yml`:

```yaml
jobs:
  custom-analysis:
    name: "My Custom Analysis"
    runs-on: ubuntu-latest
    needs: tmcmc-calibration

    steps:
      - uses: actions/checkout@v4
      - run: python my_custom_analysis.py
      - uses: actions/upload-artifact@v4
        with:
          name: custom-results
          path: results/custom/
```

### Parallel Execution

Run independent jobs in parallel:

```yaml
jobs:
  job1:
    runs-on: ubuntu-latest
    # No 'needs' - runs immediately

  job2:
    runs-on: ubuntu-latest
    # No 'needs' - runs in parallel

  job3:
    runs-on: ubuntu-latest
    needs: [job1, job2]  # Waits for both
```

### Matrix Strategy

Test multiple configurations:

```yaml
jobs:
  test-configs:
    strategy:
      matrix:
        n0: [100, 200, 500]
        mode: [debug, production]

    runs-on: ubuntu-latest
    steps:
      - run: python main_calibration.py
        env:
          N0: ${{ matrix.n0 }}
          DEBUG: ${{ matrix.mode == 'debug' }}
```

## 📊 Performance

### Resource Usage

Typical workflow consumption:

| Job | CPU Time | Memory | Storage |
|-----|----------|--------|---------|
| TMCMC | 30 min | 2 GB | 100 MB |
| TSM | 20 min | 1 GB | 50 MB |
| Simulation | 25 min | 2 GB | 200 MB |
| Profiling | 20 min | 1 GB | 10 MB |
| Notebooks | 30 min | 2 GB | 500 MB |
| Reports | 30 min | 2 GB | 100 MB |

**Total:** ~2.5 hours, ~960 MB artifacts

### Optimization Tips

1. **Use cache:**
   ```yaml
   - uses: actions/setup-python@v4
     with:
       cache: 'pip'  # Cache dependencies
   ```

2. **Parallel jobs:**
   - Independent jobs run in parallel
   - Use `needs` only when necessary

3. **Artifact size:**
   - Compress large files
   - Clean up intermediate files
   - Use appropriate retention

4. **DEBUG mode:**
   - Use for testing
   - 10x faster than production
   - Acceptable accuracy for CI

## 🔐 Security

### Secrets

For private data or API keys:

```yaml
steps:
  - run: python script.py
    env:
      API_KEY: ${{ secrets.MY_API_KEY }}
```

Add secrets: Settings → Secrets → Actions → New secret

### Permissions

Default workflow permissions are safe:
- Read repository
- Write workflow artifacts
- No access to secrets unless explicitly used

## 📚 References

- **GitHub Actions Docs:** https://docs.github.com/actions
- **Workflow Syntax:** https://docs.github.com/actions/reference/workflow-syntax-for-github-actions
- **Artifacts:** https://docs.github.com/actions/guides/storing-workflow-data-as-artifacts

## 🆘 Support

Issues with automation?

1. Check workflow logs in Actions tab
2. Review [Troubleshooting](#troubleshooting) section
3. Open issue with workflow logs attached

---

**Last updated:** 2025-12-03
