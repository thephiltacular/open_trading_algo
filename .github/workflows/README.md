# GitHub Actions Workflows

This directory contains automated workflows for the open_trading_algo project.

## Workflows Overview

### 🔍 Branch Validation Workflows

#### `validate-branch-name.yml`
**Triggers:** Branch creation, push to any branch
**Purpose:** Validates branch names against project conventions
- ✅ Allows protected branches (`main`, `develop`, `master`)
- ✅ Allows release branches (`release/*`)
- ✅ Allows hotfix branches (`hotfix/*`)
- ✅ Enforces naming patterns for feature branches
- ❌ Blocks invalid branch names with helpful error messages

#### `validate-pr-branch.yml`
**Triggers:** Pull request opened, synchronized, or reopened
**Purpose:** Validates branch names for pull requests
- ✅ Comments on PRs with validation results
- ✅ Provides suggestions for invalid branch names
- ✅ Includes step-by-step fix instructions
- ❌ Blocks PR merging if branch name is invalid

#### `branch-health-check.yml`
**Triggers:** Daily at 9 AM UTC, manual trigger
**Purpose:** Analyzes repository branch health
- 📊 Counts branches by type
- ⏰ Identifies stale branches (>30 days old)
- 🔍 Checks naming convention compliance
- 📋 Generates health reports as artifacts

## Reusable Actions

### `validate-branch` Action
**Location:** `.github/actions/validate-branch/action.yml`
**Purpose:** Reusable action for branch name validation

**Inputs:**
- `branch-name`: The branch name to validate (required)
- `allow-protected`: Allow protected branches (default: true)
- `allow-release`: Allow release branches (default: true)
- `allow-hotfix`: Allow hotfix branches (default: true)
- `strict-validation`: Enable length/character validation (default: true)

**Outputs:**
- `is-valid`: Whether the branch name is valid
- `branch-type`: The type of branch (feature, fix, etc.)
- `suggestion`: Suggested branch name if invalid

**Usage:**
```yaml
- name: Validate branch name
  uses: ./.github/actions/validate-branch
  id: validation
  with:
    branch-name: ${{ github.ref_name }}
    allow-protected: 'true'
    strict-validation: 'true'
```

## Branch Naming Conventions

The workflows enforce these naming patterns:

```
feature/description  - New features
feat/description     - New features (short)
fix/description      - Bug fixes
bug/description      - Bug fixes (alternative)
docs/description     - Documentation
refactor/description - Code refactoring
chore/description    - Maintenance tasks
test/description     - Testing related
ci/description       - CI/CD changes
perf/description     - Performance improvements
style/description    - Code style changes
hotfix/description   - Hotfixes
```

## Configuration

### Customizing Validation Rules

To modify validation rules, edit the reusable action:

1. **Change allowed branch types:** Modify the regex pattern in `action.yml`
2. **Adjust length limits:** Change the min/max description length checks
3. **Add new branch types:** Update the pattern and suggestion logic

### Disabling Workflows

To temporarily disable a workflow:
1. Add a comment at the top of the workflow file
2. Or rename the file (e.g., `validate-branch-name.yml.disabled`)

### Workflow Permissions

Ensure the repository has these GitHub Actions permissions:
- ✅ Read access to repository contents
- ✅ Write access for PR comments
- ✅ Read access to pull requests

## Troubleshooting

### Common Issues

**Workflow doesn't trigger:**
- Check branch name patterns in workflow triggers
- Verify the workflow file is in the correct location
- Ensure GitHub Actions is enabled for the repository

**Validation fails unexpectedly:**
- Check the branch name against the documented patterns
- Review the workflow logs for specific error messages
- Verify the reusable action inputs are correct

**PR comments not appearing:**
- Check GitHub Actions permissions for the repository
- Ensure the workflow has access to create issue comments
- Verify the PR trigger conditions are met

### Testing Workflows

To test workflows locally:
1. Use GitHub CLI: `gh workflow run <workflow-name>`
2. Push a test branch with a known valid/invalid name
3. Check the Actions tab for workflow execution

### Debugging

Enable debug logging by setting repository secrets:
- `ACTIONS_RUNNER_DEBUG=true`
- `ACTIONS_STEP_DEBUG=true`

## Contributing

When modifying workflows:
1. Test changes on a feature branch first
2. Update this README if adding new workflows
3. Ensure workflows follow security best practices
4. Use reusable actions for common functionality

## Security Considerations

- Workflows run with minimal required permissions
- No sensitive data is logged or exposed
- Branch validation prevents malicious branch names
- All actions use official or verified sources
