#!/usr/bin/env python3
"""
Release Helper Script for open_trading_algo

This script helps with the release process by:
- Checking if everything is ready for release
- Bumping version numbers
- Creating release notes
- Providing release checklist
"""

import subprocess
import sys
import re
from pathlib import Path
from typing import Optional


def run_command(cmd: str, check: bool = True) -> tuple[bool, str, str]:
    """Run a shell command and return success status and output."""
    try:
        result = subprocess.run(cmd, shell=True, check=check, capture_output=True, text=True)
        return True, result.stdout, result.stderr
    except subprocess.CalledProcessError as e:
        return False, e.stdout, e.stderr


def get_current_version() -> str:
    """Get current version from pyproject.toml."""
    pyproject_path = Path("pyproject.toml")
    if not pyproject_path.exists():
        print("❌ pyproject.toml not found")
        sys.exit(1)

    with open(pyproject_path, "r") as f:
        content = f.read()

    match = re.search(r'version = "([^"]+)"', content)
    if match:
        return match.group(1)
    else:
        print("❌ Could not find version in pyproject.toml")
        sys.exit(1)


def bump_version(current_version: str, bump_type: str) -> str:
    """Bump version according to semantic versioning."""
    major, minor, patch = map(int, current_version.split("."))

    if bump_type == "major":
        return f"{major + 1}.0.0"
    elif bump_type == "minor":
        return f"{major}.{minor + 1}.0"
    elif bump_type == "patch":
        return f"{major}.{minor}.{patch + 1}"
    else:
        print(f"❌ Invalid bump type: {bump_type}")
        sys.exit(1)


def update_version(new_version: str) -> None:
    """Update version in pyproject.toml."""
    pyproject_path = Path("pyproject.toml")

    with open(pyproject_path, "r") as f:
        content = f.read()

    # Update version
    content = re.sub(r'version = "[^"]+"', f'version = "{new_version}"', content)

    with open(pyproject_path, "w") as f:
        f.write(content)

    print(f"✅ Updated version to {new_version}")


def check_release_readiness() -> bool:
    """Check if everything is ready for release."""
    checks = [
        ("Git status clean", lambda: run_command("git status --porcelain")[1] == ""),
        ("Tests pass", lambda: run_command("poetry run pytest")[0]),
        ("Code formatted", lambda: run_command("poetry run black --check .")[0]),
        ("Imports sorted", lambda: run_command("poetry run isort --check-only .")[0]),
        ("Pre-commit passes", lambda: run_command("poetry run pre-commit run --all-files")[0]),
    ]

    all_passed = True
    for check_name, check_func in checks:
        print(f"🔍 Checking {check_name}...", end=" ")
        try:
            passed = check_func()
            if passed:
                print("✅")
            else:
                print("❌")
                all_passed = False
        except Exception as e:
            print(f"❌ (Error: {e})")
            all_passed = False

    return all_passed


def create_release_notes(version: str) -> str:
    """Generate basic release notes."""
    notes = f"""# Release v{version}

## 🚀 New Features
- [Add new features here]

## 🐛 Bug Fixes
- [Add bug fixes here]

## 📚 Documentation
- [Add documentation updates here]

## 🔧 Maintenance
- [Add maintenance changes here]

## 📦 Installation
```bash
pip install open_trading_algo=={version}
```

---
**Full Changelog**: [View on GitHub](https://github.com/thephiltacular/open_trading_algo/compare/v{version}...main)
"""

    return notes


def main():
    if len(sys.argv) < 2:
        print("Usage: python release_helper.py <command>")
        print()
        print("Commands:")
        print("  check     - Check if ready for release")
        print("  bump <type> - Bump version (patch, minor, major)")
        print("  notes <version> - Generate release notes")
        print("  prepare <type> - Full release preparation")
        return

    command = sys.argv[1]

    if command == "check":
        print("🔍 Checking release readiness...")
        if check_release_readiness():
            print("✅ All checks passed! Ready for release.")
        else:
            print("❌ Some checks failed. Please fix before releasing.")
            sys.exit(1)

    elif command == "bump":
        if len(sys.argv) < 3:
            print("Usage: python release_helper.py bump <patch|minor|major>")
            sys.exit(1)

        bump_type = sys.argv[2]
        current_version = get_current_version()
        new_version = bump_version(current_version, bump_type)

        print(f"📦 Bumping version: {current_version} → {new_version}")
        update_version(new_version)

        # Commit version bump
        run_command("git add pyproject.toml")
        run_command(f'git commit -m "chore: bump version to {new_version}"')

        print("✅ Version bumped and committed")
        print(f"🚀 Ready to create release v{new_version}")

    elif command == "notes":
        if len(sys.argv) < 3:
            print("Usage: python release_helper.py notes <version>")
            sys.exit(1)

        version = sys.argv[2]
        notes = create_release_notes(version)
        print(notes)

    elif command == "prepare":
        if len(sys.argv) < 3:
            print("Usage: python release_helper.py prepare <patch|minor|major>")
            sys.exit(1)

        bump_type = sys.argv[2]

        print("🚀 Preparing release...")
        print("Step 1: Checking readiness...")
        if not check_release_readiness():
            print("❌ Release checks failed. Please fix issues first.")
            sys.exit(1)

        print("Step 2: Bumping version...")
        current_version = get_current_version()
        new_version = bump_version(current_version, bump_type)
        update_version(new_version)

        print("Step 3: Committing version bump...")
        run_command("git add pyproject.toml")
        run_command(f'git commit -m "chore: bump version to {new_version}"')

        print("Step 4: Creating git tag...")
        run_command(f"git tag v{new_version}")

        print("Step 5: Pushing changes...")
        run_command("git push origin main")
        run_command(f"git push origin v{new_version}")

        print("\n✅ Release preparation complete!")
        print(f"📦 Version: {new_version}")
        print("🚀 GitHub Actions will automatically publish to PyPI")
        print("📝 Create a GitHub release with the tag to trigger publishing")

    else:
        print(f"❌ Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
