# wine-quality-classification-mlflow-dvc
Experiment DVC and MLflow pipeline for Dataset versioning and ML experiment tracking.


## Project Structure
```bash
wine-quality-classification-mlflow-dvc/
│
├── data/ # Directory for datasets
├── notebooks/ # Jupyter notebooks for exploration and analysis
│
├── src/ # Source code for the project
│ ├── init.py # Makes src a Python package
│ ├── data/ # Scripts to download or generate data
│ ├── features/ # Scripts to turn raw data into features
│ ├── models/ # Scripts to train models and make predictions
│ └── visualization/ # Scripts to create visualizations
│
├── requirements.txt # List of dependencies
├── environment.yml # Conda environment file
├── dvc.yaml # DVC pipeline configuration
├── mlflow_tracking.py # Script for MLflow tracking
└── README.md # Project documentation
```
