#!/usr/bin/env python3
"""
Open Trading Algo - PyPI Publishing Script
This script helps publish the package to PyPI
"""

import subprocess
import sys
import os
from pathlib import Path


class Colors:
    RED = "\033[0;31m"
    GREEN = "\033[0;32m"
    YELLOW = "\033[1;33m"
    BLUE = "\033[0;34m"
    NC = "\033[0m"  # No Color


def log_info(message):
    print(f"{Colors.BLUE}ℹ️  {message}{Colors.NC}")


def log_success(message):
    print(f"{Colors.GREEN}✅ {message}{Colors.NC}")


def log_warning(message):
    print(f"{Colors.YELLOW}⚠️  {message}{Colors.NC}")


def log_error(message):
    print(f"{Colors.RED}❌ {message}{Colors.NC}")


def run_command(cmd, check=True):
    """Run a shell command and return the result."""
    try:
        result = subprocess.run(cmd, shell=True, check=check, capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.CalledProcessError as e:
        return False, e.stdout, e.stderr


def check_poetry():
    """Check if poetry is available."""
    success, _, _ = run_command("poetry --version")
    if not success:
        log_error("Poetry is not installed. Please install it first:")
        print("curl -sSL https://install.python-poetry.org | python3 -")
        sys.exit(1)


def build_package():
    """Build the package."""
    log_info("Building package...")
    success, stdout, stderr = run_command("poetry build")
    if success:
        log_success("Package built successfully")
        return True
    else:
        log_error("Failed to build package")
        print(stderr)
        return False


def check_contents():
    """Check package contents."""
    dist_dir = Path("dist")
    if dist_dir.exists():
        log_info("Package files:")
        for file in dist_dir.glob("*"):
            print(f"  {file.name}")

        # Check tar.gz contents
        tar_files = list(dist_dir.glob("*.tar.gz"))
        if tar_files:
            log_info("Package contents:")
            run_command(f"tar -tzf {tar_files[0]} | head -20")
    else:
        log_error("No dist directory found. Run build first.")


def test_install():
    """Test local installation."""
    dist_dir = Path("dist")
    if dist_dir.exists():
        tar_files = list(dist_dir.glob("open_trading_algo-*.tar.gz"))
        if tar_files:
            log_info("Testing local installation...")
            success, _, _ = run_command(f"pip install {tar_files[0]} --force-reinstall")
            if success:
                success, _, _ = run_command("python -c \"import open_trading_algo; print('✅ Import successful!')\"")
                if success:
                    log_success("Local installation test passed")
                    return True
                else:
                    log_error("Import test failed")
            else:
                log_error("Installation failed")
        else:
            log_error("No package file found")
    else:
        log_error("No dist directory found")
    return False


def publish_test():
    """Publish to Test PyPI."""
    log_warning("Publishing to Test PyPI...")
    success, stdout, stderr = run_command("poetry publish -r testpypi")
    if success:
        log_success("Published to Test PyPI")
        log_info("Test your package: pip install -i https://test.pypi.org/simple/ open_trading_algo")
        return True
    else:
        log_error("Failed to publish to Test PyPI")
        print(stderr)
        return False


def publish_prod():
    """Publish to production PyPI."""
    log_warning("Publishing to production PyPI...")
    response = input("Are you sure you want to publish to production PyPI? (y/N): ")
    if response.lower() in ["y", "yes"]:
        success, stdout, stderr = run_command("poetry publish")
        if success:
            log_success("Published to production PyPI")
            log_info("Your package is now live: https://pypi.org/project/open_trading_algo/")
            return True
        else:
            log_error("Failed to publish to production PyPI")
            print(stderr)
    else:
        log_info("Publication cancelled")
    return False


def main():
    print(f"{Colors.BLUE}🚀 Open Trading Algo - PyPI Publisher{Colors.NC}")
    print("=" * 40)

    if len(sys.argv) < 2:
        command = "help"
    else:
        command = sys.argv[1]

    if command == "build":
        check_poetry()
        build_package()

    elif command == "check":
        check_contents()

    elif command == "test":
        test_install()

    elif command == "testpypi":
        check_poetry()
        if build_package():
            publish_test()

    elif command == "publish":
        check_poetry()
        if build_package():
            publish_prod()

    elif command == "all":
        check_poetry()
        if build_package():
            check_contents()
            if test_install():
                publish_test()

    else:
        print("Usage: python publish.py {build|check|test|testpypi|publish|all}")
        print()
        print("Commands:")
        print("  build     - Build the package")
        print("  check     - Check package contents")
        print("  test      - Test local installation")
        print("  testpypi  - Build and publish to Test PyPI")
        print("  publish   - Build and publish to production PyPI")
        print("  all       - Build, test, and publish to Test PyPI")
        print()
        print("Examples:")
        print("  python publish.py build          # Just build")
        print("  python publish.py testpypi       # Publish to test")
        print("  python publish.py all            # Full workflow")


if __name__ == "__main__":
    main()
