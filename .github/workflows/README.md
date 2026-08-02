# GitHub Actions Workflow Documentation

## Overview

This folder contains GitHub Actions workflow files for automated Continuous Integration (CI).

The workflows automatically build, set up an isolated Anaconda/Miniconda Python environment, install project dependencies, and run test suites whenever code changes are pushed or pull requests are submitted.

---

## Workflows

### 1. `python-package-conda.yml` (Linux)

Designed for the standard Ubuntu runner.

**Key Workflow Components**

* **Triggers (`on:`)**: Executes automatically on `push` and `pull_request` events to the `main` branch.
* **Runner Environment (`runs-on:`)**: Executes inside a GitHub-hosted Ubuntu Linux container (`ubuntu-latest`).
* **Environment Setup (`setup-miniconda`)**: Uses `conda-incubator/setup-miniconda` to provision a clean Conda environment with Python 3.10.
* **Dependency Management**: Installs required packages via `conda install` and `pip install` steps. It also checks for `environment.yml` and model-specific `requirements.txt` files.
* **Automated Testing**: Runs `pytest` to ensure code stability and prevent regression bugs across commits.

---

### 2. `python-package-conda-windows.yml` (Windows 11 equivalent)

To run the workflow natively on a Windows 11 / Windows Server 2022 runner on GitHub Actions.

**Key Changes for Windows 11 Compatibility**

1. **`runs-on: windows-latest`**: Swaps the runner operating system to the latest Windows Server environment (equivalent to Windows 11 architecture).
2. **`shell: pwsh`**: Explicitly configures PowerShell Core as the default shell engine to handle Windows-native script execution and conditional checks reliably (`Test-Path`).

---

## Notes for this Repository

- Full installation of **Kimi K3** dependencies (`flash-attn`, etc.) is intentionally skipped in CI because they require specific CUDA versions and heavy compilation.
- A placeholder test is created automatically if no `tests/` directory exists yet.
- You can expand the test suite later under the `tests/` folder.
