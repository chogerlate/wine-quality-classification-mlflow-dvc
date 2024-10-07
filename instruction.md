# Building ML project with Data versioning and ML experiment tracking.
## Project Description

This project aims to build a machine learning model to predict the quality of wine using various physicochemical properties. We will implement data versioning and ML experiment tracking to ensure reproducibility and efficient management of our machine learning workflow.

### Key Components:

1. **Data Versioning**: 
   - We'll use DVC (Data Version Control) to version our dataset and track changes over time.
   - This allows us to reproduce experiments and collaborate effectively.

2. **ML Experiment Tracking**:
   - We'll utilize MLflow to track our experiments, including parameters, metrics, and model artifacts.
   - This enables us to compare different runs and easily reproduce our best results.

3. **Model Development**:
   - We'll build and evaluate various machine learning models to predict wine quality.
   - The challenge lies in handling the imbalanced nature of the dataset and extracting meaningful features.

4. **Performance Evaluation**:
   - We'll use appropriate metrics to evaluate our model's performance, considering the ordinal nature of the quality scores.

5. **Reproducibility**:
   - By combining DVC and MLflow, we ensure that our entire ML pipeline is reproducible and well-documented.

This project will demonstrate best practices in ML project structure, data management, and experiment tracking, providing a solid foundation for future ML projects.

## Dataset 
I use Wine Quality Dataset from Kaggle in this experiment.
- Link: https://www.kaggle.com/datasets/yasserh/wine-quality-dataset

### Description:
    This datasets is related to red variants of the Portuguese "Vinho Verde" wine.The dataset describes the amount of various chemicals present in wine and their effect on it's quality. The datasets can be viewed as classification or regression tasks. The classes are ordered and not balanced (e.g. there are much more normal wines than excellent or poor ones).Your task is to predict the quality of wine using the given data.

    A simple yet challenging project, to anticipate the quality of wine.
    The complexity arises due to the fact that the dataset has fewer samples, & is highly imbalanced.
    Can you overcome these obstacles & build a good predictive model to classify them?

    This data frame contains the following columns:

    Input variables (based on physicochemical tests):\
    1 - fixed acidity\
    2 - volatile acidity\
    3 - citric acid\
    4 - residual sugar\
    5 - chlorides\
    6 - free sulfur dioxide\
    7 - total sulfur dioxide\
    8 - density\
    9 - pH\
    10 - sulphates\
    11 - alcohol\
    Output variable (based on sensory data):\
    12 - quality (score between 0 and 10)

