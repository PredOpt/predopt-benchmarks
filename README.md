## Installation instructions



Installation requires packages within a virtual environement. This way, scripts and libraries installed are isolated from others on your machine.
#### It is recommended to use `venv`
Recommneded python version `Python 3.7.3`.

Create the virtual environment by running the following
```bash
python3 -m venv benchmarking_env 
source benchmarking_env/bin/activate
```

#### Pip
Recommended to upgrade pip
```bash
pip install --upgrade pip
```

Then install all required packages with pip

```bash
pip install -r requirements.txt
```
#### Conda
Alternatively, you can use conda to create a virtual environement.

First, make sure that conda is installed. Refer to [this guide](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) to install it on your system.

After cloning this repository from github, create a new virtual environment that installs the neede packages locally, with the following command:

```bash
conda env create -n benchmarking_env --file environment.yml
```

Then to activate this environment, use
```bash
conda activate benchmarking_env
```
Or, on Windows
```
source activate benchmarking_env
```

You can install any additional library using conda or pip.

### Then to run the benchmarking experiments, navigate to the corresponding directory.
