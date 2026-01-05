#!/usr/bin/env python3
"""
Setup script to install UV if not already installed.
"""

import subprocess
import sys
from pathlib import Path


def check_uv_installed():
    """Check if UV is installed."""
    try:
        subprocess.run(["uv", "--version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def install_uv():
    """Install UV using pip."""
    print("Installing UV...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "uv"], check=True)
        print("UV installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing UV: {e}")
        return False


def setup_project():
    """Set up the project using UV."""
    if not check_uv_installed():
        if not install_uv():
            print("Failed to install UV. Please install it manually: pip install uv")
            sys.exit(1)

    print("Setting up project with UV...")
    try:
        # Create virtual environment if it doesn't exist
        venv_path = Path(".venv")
        if not venv_path.exists():
            print("Creating virtual environment...")
            subprocess.run(["uv", "venv"], check=True)

        # Install dependencies
        print("Installing dependencies...")
        subprocess.run(["uv", "pip", "install", "-e", "."], check=True)

        print("\nProject setup complete!")
        print("\nTo activate the virtual environment:")
        print("  - On Windows: .venv\\Scripts\\activate")
        print("  - On macOS/Linux: source .venv/bin/activate")
        print("\nOr use 'make install' to install dependencies with UV")

    except subprocess.CalledProcessError as e:
        print(f"Error setting up project: {e}")
        sys.exit(1)


if __name__ == "__main__":
    setup_project()
