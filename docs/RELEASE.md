# Release Process

This document outlines the process for releasing new versions of `open_trading_algo`.

## 🚀 Automated Release (Recommended)

### Option 1: GitHub Release (Triggers PyPI Publishing)

1. **Create a GitHub Release**:
   - Go to [Releases](https://github.com/thephiltacular/open_trading_algo/releases)
   - Click "Create a new release"
   - Choose a tag: `v1.2.3` (following [semantic versioning](https://semver.org/))
   - Title: `Release v1.2.3`
   - Description: List of changes
   - Click "Publish release"

2. **Automated Actions**:
   - GitHub Actions will automatically:
     - Run tests on multiple Python versions
     - Build the package
     - Publish to PyPI
     - Update release notes

### Option 2: Manual Tag Push

1. **Update version in `pyproject.toml`**:
   ```toml
   version = "1.2.3"
   ```

2. **Commit and tag**:
   ```bash
   git add pyproject.toml
   git commit -m "chore: bump version to 1.2.3"
   git tag v1.2.3
   git push origin main
   git push origin v1.2.3
   ```

3. **GitHub Actions will automatically publish to PyPI**

## 🔧 Manual Release Process

If you need to release manually:

### Prerequisites

1. **PyPI API Token**:
   - Go to https://pypi.org/manage/account/token/
   - Create a new API token
   - Store it securely

2. **Configure Poetry**:
   ```bash
   poetry config pypi-token.pypi YOUR_PYPI_TOKEN
   ```

### Release Steps

```bash
# 1. Update version
echo "Current version:"
poetry version

# Bump version (patch, minor, major)
poetry version patch  # or minor/major

# 2. Run tests
poetry run pytest

# 3. Build package
poetry build

# 4. Test installation
pip install dist/open_trading_algo-*.tar.gz
python -c "import open_trading_algo; print('✅ Success')"

# 5. Publish
poetry publish

# 6. Create git tag
VERSION=$(poetry version -s)
git tag "v$VERSION"
git push origin "v$VERSION"
```

## 📋 Pre-Release Checklist

- [ ] All tests pass: `poetry run pytest`
- [ ] Code is formatted: `poetry run black . && poetry run isort .`
- [ ] Documentation is updated
- [ ] CHANGELOG.md is updated
- [ ] Version is bumped in `pyproject.toml`
- [ ] Dependencies are up to date: `poetry update`

## 🏷️ Version Numbering

Follow [Semantic Versioning](https://semver.org/):

- **MAJOR** version for incompatible API changes
- **MINOR** version for backwards-compatible functionality additions
- **PATCH** version for backwards-compatible bug fixes

Examples:
- `1.0.0` - Initial release
- `1.0.1` - Bug fix
- `1.1.0` - New feature
- `2.0.0` - Breaking change

## 🔒 Security

- Never commit API tokens or secrets
- Use GitHub repository secrets for automated publishing
- Test on Test PyPI first: `poetry publish -r testpypi`

## 📊 Post-Release

After release:
- [ ] Verify package appears on PyPI: https://pypi.org/project/open_trading_algo/
- [ ] Test installation: `pip install open_trading_algo`
- [ ] Update any documentation links
- [ ] Announce release in relevant channels

## 🐛 Hotfixes

For urgent bug fixes:

1. Create a hotfix branch from the release tag
2. Fix the issue
3. Bump patch version
4. Merge back to main
5. Create new release

## 📞 Support

If you encounter issues:
- Check GitHub Actions logs
- Verify PyPI token is valid
- Ensure version number is unique
- Check package builds locally first
