## Installation instructions

### Conda

Installation requires packages within a virtual environement. This way, scripts and libraries installed are isolated from others on your machine.
### Alternatively, instead of using conda, you can use `venv`
```
python3 -m venv benchmarking_env 
source benchmarking_env/bin/activate
```

### Pip
Recommend to upgrade pip
```bash
pip install --upgrade pip
```

Then install all required packages with pip

```bash
pip install -r requirements.txt
```
### Conda

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
