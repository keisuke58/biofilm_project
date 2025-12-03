# Documentation Deployment Guide

This guide explains how to deploy documentation to GitHub Pages and Read the Docs, and how to set up automated PDF report generation.

## 📚 Table of Contents

- [GitHub Pages Setup](#github-pages-setup)
- [Read the Docs Setup](#read-the-docs-setup)
- [Automated PDF Reports](#automated-pdf-reports)
- [Troubleshooting](#troubleshooting)

## 🌐 GitHub Pages Setup

### Prerequisites

- Repository on GitHub
- Admin access to repository settings

### Step 1: Enable GitHub Pages

1. Go to your repository on GitHub
2. Click **Settings** → **Pages**
3. Under "Build and deployment":
   - Source: **GitHub Actions**
   - (This will use the workflow in `.github/workflows/deploy-docs.yml`)

### Step 2: Verify Workflow

The deployment workflow (`.github/workflows/deploy-docs.yml`) will:
- Trigger on push to `main` or `master` branch
- Build Sphinx documentation
- Deploy to GitHub Pages

### Step 3: Check Deployment

After pushing to main:
1. Go to **Actions** tab
2. Check "Deploy Documentation to GitHub Pages" workflow
3. Once complete, documentation will be at: `https://[username].github.io/[repo-name]/`

### Manual Trigger

You can also manually trigger deployment:
1. Go to **Actions** tab
2. Select "Deploy Documentation to GitHub Pages"
3. Click **Run workflow**

### Local Preview

Preview before deployment:

```bash
cd docs
make html
python -m http.server 8000 -d build/html
# Visit http://localhost:8000
```

## 📖 Read the Docs Setup

### Prerequisites

- Account on [readthedocs.org](https://readthedocs.org)
- Repository on GitHub (or GitLab/Bitbucket)

### Step 1: Import Project

1. Log in to Read the Docs
2. Click **Import a Project**
3. Select your repository from the list
4. Click **Import**

### Step 2: Configuration

The project uses `.readthedocs.yaml` for configuration:

```yaml
version: 2
build:
  os: ubuntu-22.04
  tools:
    python: "3.11"
sphinx:
  configuration: docs/source/conf.py
formats:
  - pdf
  - epub
```

### Step 3: Build Documentation

Read the Docs will automatically:
- Build on every commit to main
- Generate HTML, PDF, and ePub versions
- Host at: `https://[project-name].readthedocs.io`

### Step 4: Custom Domain (Optional)

1. Go to project **Admin** → **Domains**
2. Add custom domain
3. Configure DNS CNAME record
4. Enable HTTPS

### Advanced Settings

In Read the Docs project settings:

**Admin → Advanced Settings:**
- Default version: `latest`
- Default branch: `main`
- Privacy level: Public
- Show version warning: ✓

**Admin → Automation Rules:**
- Activate versions from branches: `main`, `stable`
- Activate versions from tags: `v*`

## 📄 Automated PDF Reports

### Overview

The GitHub Actions workflow (`.github/workflows/generate-report.yml`) automatically:
- Runs calibration
- Generates PDF report
- Uploads as artifact
- Attaches to releases

### Workflow Triggers

**Automatic:**
- On push to main (if code changes)
- On new git tags (creates release)

**Manual:**
```bash
# Via GitHub UI: Actions → Generate PDF Report → Run workflow
# Or via GitHub CLI:
gh workflow run generate-report.yml
```

### Downloading Reports

**From Workflow Runs:**
1. Go to **Actions** tab
2. Select workflow run
3. Download artifacts:
   - `biofilm-report-[hash].pdf`
   - `calibration-results-[hash]/`

**From Releases:**
- Attached to release when creating tags
- Download from **Releases** page

### Local Generation

Generate report locally:

```bash
python main_calibration_report.py
# Output: results/bayesian_report.pdf
```

### Configuration

Edit workflow file (`.github/workflows/generate-report.yml`):

```yaml
env:
  DEBUG_MODE: 'true'  # Fast mode for testing
  # DEBUG_MODE: 'false'  # Production mode
```

## 🔧 Advanced Configuration

### GitHub Pages with Custom Domain

1. Add `CNAME` file to `docs/build/html/`:
   ```
   docs.yourdomain.com
   ```

2. Configure DNS:
   ```
   CNAME docs yourusername.github.io
   ```

3. Enable HTTPS in repository settings

### Read the Docs Webhooks

For faster builds, set up webhooks:

1. **Settings** → **Integrations** → **Add integration**
2. Select **GitHub incoming webhook**
3. Follow instructions to add webhook

### Multi-Version Documentation

Support multiple versions:

**In `.readthedocs.yaml`:**
```yaml
python:
  version: 3.11

sphinx:
  configuration: docs/source/conf.py

versions:
  - latest
  - stable
```

**Build branches and tags:**
- `main` → `latest`
- `v*` tags → versioned docs
- `stable` branch → `stable`

### PDF Report on Schedule

Run calibration weekly:

Add to `.github/workflows/generate-report.yml`:
```yaml
on:
  schedule:
    - cron: '0 0 * * 0'  # Every Sunday at midnight
```

## 🎨 Customization

### Documentation Theme

Edit `docs/source/conf.py`:

```python
html_theme = 'sphinx_rtd_theme'

html_theme_options = {
    'logo_only': True,
    'display_version': True,
    'style_nav_header_background': '#2980B9',
}
```

### PDF Styling

LaTeX configuration in `conf.py`:

```python
latex_elements = {
    'papersize': 'letterpaper',
    'pointsize': '10pt',
    'preamble': r'''
    \usepackage{amsmath}
    ''',
}
```

### GitHub Pages Index

Create custom landing page at `docs/source/index.html` (optional).

## 📊 Monitoring

### GitHub Pages Status

Check deployment status:
```bash
# Via GitHub CLI
gh run list --workflow=deploy-docs.yml

# Or visit
# https://github.com/[user]/[repo]/actions
```

### Read the Docs Build Status

- Build status badge (add to README):
  ```markdown
  [![Documentation Status](https://readthedocs.org/projects/[project]/badge/?version=latest)](https://[project].readthedocs.io/en/latest/?badge=latest)
  ```

- Build logs: Project → **Builds**

### Report Generation Status

Monitor in Actions tab or via:
```bash
gh run list --workflow=generate-report.yml
```

## 🐛 Troubleshooting

### GitHub Pages Not Updating

**Problem:** Changes not reflected on site

**Solutions:**
1. Check Actions tab for errors
2. Clear browser cache
3. Wait 5-10 minutes for CDN
4. Verify workflow permissions in Settings

### Read the Docs Build Failing

**Problem:** Build fails with import errors

**Solutions:**
1. Check `.readthedocs.yaml` configuration
2. Verify all dependencies in `requirements-docs.txt`
3. Check build logs for specific errors
4. Test locally: `cd docs && make html`

**Common issues:**
```bash
# Missing dependencies
pip install -r requirements-docs.txt

# Sphinx warnings as errors
# In conf.py: fail_on_warning = False
```

### PDF Report Generation Timeout

**Problem:** Workflow times out (>30 min)

**Solutions:**
1. Enable DEBUG mode in workflow
2. Reduce N0 or Ndata in config
3. Increase timeout in workflow:
   ```yaml
   timeout-minutes: 60
   ```

### LaTeX Errors in PDF Generation

**Problem:** PDF build fails with LaTeX errors

**Solutions:**
```bash
# Install additional LaTeX packages locally
sudo apt-get install texlive-latex-extra

# Test PDF build
cd docs
make latexpdf
```

### Artifact Upload Fails

**Problem:** Report artifact not uploaded

**Solutions:**
1. Check `results/` directory exists
2. Verify PDF was generated
3. Check workflow file path: `results/bayesian_report.pdf`

## 📝 Best Practices

### Documentation

1. **Write good docstrings:** Auto-generated API docs need them
2. **Test locally:** Always `make html` before pushing
3. **Version your docs:** Use tags for releases
4. **Update regularly:** Keep docs in sync with code

### PDF Reports

1. **Use DEBUG mode** for testing workflows
2. **Archive old reports** to save space
3. **Set retention days** appropriately (30 days default)
4. **Include metadata** (git hash, timestamp) in reports

### Performance

1. **Cache dependencies** in workflows (already configured)
2. **Only build on relevant changes** (docs/, src/)
3. **Use workflow_dispatch** for manual control
4. **Monitor build times** and optimize if >10 min

## 🔗 Useful Links

- **GitHub Pages Docs:** https://docs.github.com/pages
- **Read the Docs:** https://docs.readthedocs.io/
- **Sphinx Documentation:** https://www.sphinx-doc.org/
- **GitHub Actions:** https://docs.github.com/actions

## 📧 Support

For deployment issues:
- GitHub Pages: Check GitHub Actions logs
- Read the Docs: Check build logs on RTD
- PDF Reports: Check workflow artifacts

---

**Last updated:** 2025-12-03
