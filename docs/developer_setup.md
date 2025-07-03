# Developer Setup

This guide provides the basic steps to set up a development environment for SALib.

1. **Clone the repository**

```bash
git clone https://github.com/SALib/SALib.git
cd SALib
```

2. **Create the conda environment**

Use `conda` (or `mamba`) to create the environment specified in `environment.yml`:

```bash
conda env create -f environment.yml  # works with mamba too
```

3. **Activate the environment and install SALib**

```bash
conda activate SALib
pip install -e .[dev]
```

4. **Install pre-commit hooks and run the tests**

```bash
pre-commit install
pytest
```
