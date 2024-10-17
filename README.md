# FYS-STK4155 Project 2


## Set up environment
The `environment.yml` file contains all packages necessary to build and work with this. Create and activate the `conda` environment using the following command:
```sh
conda env create --file environment.yml
conda activate ptwo-dev
```

To update an existing environment:
```sh
conda env update --name ptwo-dev --file environment.yml --prune
```

The dependencies can also be installed directly from `requirements.txt`:
```sh
python3 -m pip install -r requirements.txt
```

## Installation
To install this project, run the following command:
```sh
python3 -m pip install -e .
```