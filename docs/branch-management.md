# Repository Settings for Branch Management

## Required GitHub Settings Changes

### 1. Branch Protection Rules
**Location:** Repository Settings → Branches → Branch protection rules

For the 'main' branch (keep protected):
- ✅ Require pull request reviews before merging
- ✅ Require status checks to pass before merging
- ✅ Include administrators in restrictions
- ✅ Restrict pushes that create matching branches

### 2. Repository Permissions
**Location:** Repository Settings → Manage access

Ensure contributors have appropriate permissions:
- **Read access**: Can view and clone repository
- **Triage access**: Can manage issues and PRs
- **Write access**: Can push to non-protected branches (for maintainers)

### 3. Branch Creation Permissions
**Location:** Repository Settings → Branches

Allow branch creation for:
- ✅ Repository collaborators
- ✅ Users with write access
- ✅ Organization members (if applicable)

## Testing Branch Creation

To verify settings work correctly:

```bash
# Test as a contributor
git checkout -b test/feature-branch
git push origin test/feature-branch
# Should succeed without requiring PR first

# Test protected branch restrictions
git checkout main
git push origin main  # Should be blocked
# Should require PR approval
```

## Troubleshooting

**Issue:** Cannot create branches
**Solution:** Check repository permissions and branch protection rules

**Issue:** Cannot push to new branches
**Solution:** Verify write access and branch naming permissions

**Issue:** All branches are protected
**Solution:** Ensure branch protection rules specify exact branch names (not wildcards)
