# Benchmarking Predict-then-Optimize (PtO) Problems

## About

This repository provides a comprehensive framework for benchmarking Predict-then-Optimize (PtO) problems using Decision-Focused Learning (DFL) approaches. PtO problems involve making predictions that are used as input to downstream optimization tasks, where traditional two-stage methods often lead to suboptimal solutions. DFL addresses this by training machine learning models that directly optimize for the downstream decision-making objectives.


This repository contains the implementation for the paper (Accepted to Journal of Artificial Intelligence Research (JAIR)):

> Mandi, J., Kotary, J., Berden, S., Mulamba, M., Bucarey, V., Guns, T., & Fioretto, F. (2024). Decision-focused learning: Foundations, state of the art, benchmark and future opportunities. Journal of Artificial Intelligence Research, 80, 1623-1701. [DOI: 10.1613/jair.1.14123](https://doi.org/10.1613/jair.1.14123)

If you use this code in your research, please cite:
```bibtex
@article{mandi2024decision,
  title={Decision-focused learning: Foundations, state of the art, benchmark and future opportunities},
  author={Mandi, Jayanta and Kotary, James and Berden, Senne and Mulamba, Maxime and Bucarey, Victor and Guns, Tias and Fioretto, Ferdinando},
  journal={Journal of Artificial Intelligence Research},
  volume={80},
  pages={1623--1701},
  year={2024},
  doi={10.1613/jair.1.14123}
}
```



## Installation

### Prerequisites
- Python 3.7.3 (recommended)
- pip or conda package manager

### Option 1: Using venv (Recommended)

1. Create and activate a virtual environment:
```bash
python3 -m venv benchmarking_env
source benchmarking_env/bin/activate
```

2. Upgrade pip:
```bash
pip install --upgrade pip
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

### Option 2: Using Conda

1. Install Conda by following the [official installation guide](https://conda.io/projects/conda/en/latest/user-guide/install/index.html)

2. Create and activate the environment:
```bash
# Create environment
conda env create -n benchmarking_env --file environment.yml

# Activate on Linux/macOS
conda activate benchmarking_env

# Activate on Windows
source activate benchmarking_env
```

## Running Experiments

Navigate to the corresponding experiment directory to run specific benchmarks.

## Contributing

Feel free to open issues or submit pull requests if you find any problems or have suggestions for improvements.
