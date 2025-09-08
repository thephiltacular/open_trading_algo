# Branch Naming Conventions

## Standard Branch Naming Patterns

Use these naming conventions for all branches to maintain consistency:

### Feature Branches
```
feature/description-of-feature
feat/description-of-feature
```
**Examples:**
- `feature/add-rsi-indicator`
- `feat/implement-sentiment-analysis`
- `feature/multi-asset-portfolio-backtest`

### Bug Fix Branches
```
fix/description-of-fix
bug/description-of-fix
```
**Examples:**
- `fix/resolve-memory-leak`
- `bug/handle-api-rate-limits`
- `fix/correct-signal-calculation`

### Documentation Branches
```
docs/description-of-docs
```
**Examples:**
- `docs/update-api-reference`
- `docs/add-troubleshooting-guide`
- `docs/create-contributing-tutorial`

### Hotfix Branches
```
hotfix/description-of-hotfix
```
**Examples:**
- `hotfix/critical-security-patch`
- `hotfix/fix-production-crash`

### Refactoring Branches
```
refactor/description-of-refactor
```
**Examples:**
- `refactor/optimize-database-queries`
- `refactor/simplify-signal-generation`

## Branch Creation Examples

```bash
# Create a new feature branch
git checkout -b feature/add-new-indicator

# Create a bug fix branch
git checkout -b fix/resolve-api-timeout

# Create a documentation branch
git checkout -b docs/update-contributing-guide

# Create from main branch
git checkout main
git pull origin main
git checkout -b feature/your-feature-name
```

## Best Practices

1. **Use lowercase with hyphens**: `feature/add-new-indicator` not `feature/AddNewIndicator`
2. **Be descriptive**: `fix/resolve-memory-leak` not `fix/bug`
3. **Keep it short**: Aim for 3-5 words in the description
4. **Use imperative mood**: `add-rsi-indicator` not `adding-rsi-indicator`
5. **Include issue number**: `feature/add-rsi-indicator-#123` when applicable

## Automated Branch Validation

Consider setting up branch name validation in your CI/CD pipeline:

```yaml
# .github/workflows/validate-branch.yml
name: Validate Branch Name
on:
  create:
    branches: ['*']

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - name: Check branch name
        run: |
          if [[ ! "$GITHUB_REF_NAME" =~ ^(feature|feat|fix|bug|docs|hotfix|refactor)/.+$ ]]; then
            echo "Branch name must follow pattern: type/description"
            exit 1
          fi
```
