# Quick Setup Instructions

This guide provides quick instructions for setting up GitHub Pages, Read the Docs, and automated PDF reports.

## 🚀 Quick Start (5 minutes)

### 1. Enable GitHub Pages

```bash
# Push to main branch (triggers workflow)
git push origin main
```

Then in GitHub:
1. Go to **Settings** → **Pages**
2. Source: **GitHub Actions**
3. Wait 2-3 minutes
4. Visit: `https://[username].github.io/[repo]/`

### 2. Setup Read the Docs

1. Visit [readthedocs.org](https://readthedocs.org)
2. Sign in with GitHub
3. Click **Import a Project**
4. Select your repository
5. Click **Build**
6. Visit: `https://[project].readthedocs.io`

### 3. Test PDF Report Generation

```bash
# Via GitHub UI:
# Actions → Generate PDF Report → Run workflow

# Or locally:
python main_calibration_report.py
```

## ✅ Verification Checklist

- [ ] GitHub Pages deployed successfully
- [ ] Read the Docs builds without errors
- [ ] PDF report generates and uploads
- [ ] Badges show correct status in README
- [ ] All links work correctly

## 📝 Next Steps

- Customize theme in `docs/source/conf.py`
- Add custom domain (optional)
- Set up version control for docs
- Configure scheduled report generation

## 🔗 URLs to Bookmark

- **GitHub Pages:** `https://[username].github.io/[repo]/`
- **Read the Docs:** `https://[project].readthedocs.io/`
- **PDF Download:** `https://[project].readthedocs.io/_/downloads/en/latest/pdf/`
- **GitHub Actions:** `https://github.com/[user]/[repo]/actions`

## ⚙️ Configuration Files

All configuration is in place:

```
.github/workflows/
  ├── deploy-docs.yml         # GitHub Pages deployment
  ├── generate-report.yml     # PDF report generation
  └── tests.yml               # CI/CD tests

.readthedocs.yaml              # Read the Docs config
docs/
  ├── source/conf.py           # Sphinx configuration
  └── DEPLOYMENT.md            # Full deployment guide
```

## 🆘 Need Help?

See detailed guide: [docs/DEPLOYMENT.md](DEPLOYMENT.md)

For issues:
- GitHub Pages: Check Actions tab for errors
- Read the Docs: Check build logs
- PDF Reports: Check workflow artifacts
