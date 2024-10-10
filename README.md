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

## Results
Link: [Dagshub](https://dagshub.com/chogerlate/wine-quality-classification-mlflow-dvc)
- Successfully manage data using DVC
![image](https://github.com/user-attachments/assets/ba682ba0-7cf7-47cf-a990-3e10174d9b7f)
- Experiment logging using MLflow hosted on Dagshub
![image](https://github.com/user-attachments/assets/1991c840-94fc-4c84-9f27-a994eb96ddea)
