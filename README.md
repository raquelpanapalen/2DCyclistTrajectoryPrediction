# 2DCyclistTrajectoryPrediction

## Install
The requirements.txt file contains a list of all the necessary packages and their versions for this project.

```bash
pip install -r requirements.txt
git clone git@github.com:raquelpanapalen/2DCyclistTrajectoryPrediction.git
cd 2DCyclistTrajectoryPrediction
```

## Project structure

This repository contains scripts and notebooks for the project. Below is an overview of the directory structure and contents:
```
- scripts/
    - preprocessing/          # Scripts for data preprocessing
    - evaluation/             # Scripts for model evaluation
    - models/                 # Scripts for model architectures
    - trainers/               # Scripts for model training
    - visualization/          # Scripts for data visualization
    - utils.py                # Utility functions script
    - metrics.py              # Metrics computation script
    
- notebooks/                  # Jupyter notebooks
    - 
```

### Scripts

The `scripts` directory serves as the main folder for all the project's scripts. It is organized into subfolders based on their specific functionalities:

* `preprocessing/`: This folder contains scripts responsible for data preprocessing tasks. These scripts handle data extraction from Google Cloud Storage, data formatting, normalization, and cleaning.

* `evaluation/`: Here, you can find scripts dedicated to model evaluation. These scripts compute various evaluation metrics, and generate performance reports.

* `models/`: The models folder contains scripts related to the definition and implementation of model architectures. These scripts include the code for building and configuring different models used in the project.

* `trainers/`: In this folder, you will find scripts responsible for model training. These scripts handle the training process, including data loading, validation steps, and saving trained models.

* `visualization/`: This folder contains scripts for data visualization. These scripts generate plots, graphs, and other visual representations of the data and model outputs.

* `utils.py`: This script contains utility functions that are commonly used across different parts of the project. It includes helper functions, data manipulation tools, and other general-purpose utilities.

* `metrics.py`: This script provides functions for computing various metrics used for evaluating model performance.


### Notebooks

The `notebooks` directory contains Jupyter notebooks used for interactive data analysis, experimentation, and documentation purposes. These notebooks provide a user-friendly environment for exploring the data, visualizing results, and documenting the research process.




Feel free to navigate through the directories and explore the code and notebooks to gain a deeper understanding of the project. Happy exploring!