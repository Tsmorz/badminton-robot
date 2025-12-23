# badminton-robot

[![Coverage Status](https://coveralls.io/repos/github/Tsmorz/badminton-robot/badge.svg?branch=main)](https://coveralls.io/github/Tsmorz/badminton-robot?branch=main)
![Docker Image CI](https://github.com/Tsmorz/badminton-robot/actions/workflows/ci.yml/badge.svg)

Simple README.md for a Python project template.

## Install

To install the library from PyPI:

```bash
uv pip install badminton-robot==latest
```
OR
```bash
uv add git+https://github.com/Tsmorz/badminton-robot.git@<specific-tag>  # needs credentials
```

## Development
0. Install [uv](https://docs.astral.sh/uv/getting-started/installation/) from Astral.
1. `git clone git@github.com:Tsmorz/badminton-robot.git`
2. `make init` to create the virtual environment and install dependencies
3. `make format` to format the code and check for errors
4. `make test` to run the test suite
5. `make clean` to delete the temporary files and directories


## Publishing
It's super easy to publish your own packages on PyPI. To build and publish this package run:

```bash
uv build
uv publish  # make sure your version in pyproject.toml is updated
```
The package can then be found at: https://pypi.org/project/badminton-robot

## Module Usage
```python
"""Basic docstring for my module."""

from loguru import logger

from badminton_robot import definitions

def main() -> None:
    """Run a simple demonstration."""
    logger.info("Hello World!")

if __name__ == "__main__":
    main()
```

## Program Usage
```bash
uv run python -m badminton_robot
```

## Structure
The following tree shows the important permanent files. Run `make tree` to update.
<!-- TREE-START -->
```
├── src
│   └── badminton_robot
│       ├── __init__.py
│       ├── __main__.py
│       ├── app.py
│       ├── definitions.py
│       └── utils.py
├── tests
│   ├── __init__.py
│   ├── conftest.py
│   ├── app_test.py
│   └── utils_test.py
├── .dockerignore
├── .gitignore
├── .pre-commit-config.yaml
├── .python-version
├── CONTRIBUTING.md
├── Dockerfile
├── LICENSE
├── Makefile
├── README.md
├── pyproject.toml
├── repo_tree.py
└── uv.lock
```
<!-- TREE-END -->
