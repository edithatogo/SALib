# Developer Setup

This guide provides a quick way to install all development dependencies and run the test suite.

## 1. Clone the repository

```bash
git clone https://github.com/SALib/SALib.git
cd SALib
```

## 2. Create the conda environment

Use `conda` or `mamba` to create the environment defined in `environment.yml`:

```bash
conda env create -f environment.yml  # or: mamba env create -f environment.yml
```

## 3. Activate the environment and install SALib

```bash
conda activate SALib
pip install -e .[dev]
```

## 4. Install hooks and run the tests

```bash
pre-commit install
pytest
```

Running the tests confirms that your environment was created correctly and that the library works as expected.
