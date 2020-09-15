# sc3s
**Single Cell Consensus Clustering, with Speed!**

Development version.

## Installation
Use the `environment.yml` to install the virtual environment. It contains all the dependencies required for `sc3s`, as well as the popular `scanpy` package for single cell analysis in Python.

Activate the environment. `cd` to the root of this repository, and run `pip install -e .`.

This creates a symlink to this the `__init__.py` file in `sc3s/`, allowing you to access the local files.

## Unit test
You need `pytest`, which is included in the `environment.yml` file.

Activate the environmet, `cd` to the root of this repo, and run `python3 -m pytest`.