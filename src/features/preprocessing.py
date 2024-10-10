import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_data(file_path: str) -> tuple:
    """
    Summary: Load and preprocess the wine quality dataset for machine learning classification.

    Args:
        file_path (str): The path to the CSV file containing the wine quality dataset.

    Returns:
        tuple: A tuple containing the training features (X_train), 
               training labels (y_train), testing features (X_test), 
               and testing labels (y_test).
    """
    # Load the dataset
    data = pd.read_csv(file_path)

    # Separate features and target variable
    X = data.drop(columns=['quality'])
    y = data['quality'] - 3

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Standardize the feature values
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, y_train, X_test, y_test
