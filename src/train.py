import mlflow
import mlflow.xgboost
from models.xgb_model import initialize_xgb_model
from features.preprocessing import preprocess_data
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import dagshub
from utils import load_config
from skl2onnx.common.data_types import FloatTensorType
import onnxmltools


def train_model(file_path: str, config: dict) -> None:
    """
    Summary: Train the XGBoost model using the preprocessed wine quality dataset and log the experiment with MLflow.

    Args:
        file_path (str): The path to the CSV file containing the wine quality dataset.
        config (dict): The configuration dictionary containing model parameters.

    Returns:
        None
    """
    # Load and preprocess the data
    X_train, y_train, X_test, y_test = preprocess_data(file_path)

    # Initialize the model with parameters from the config
    params = config['model']['parameters']
    model = initialize_xgb_model(params)
    # Train the model
    model.fit(X_train, y_train)
    # Log metrics
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')


    # Convert the trained XGBoost model to ONNX format
    initial_type = [('float_input', FloatTensorType([None, X_test.shape[1]]))]
    onnx_model = onnxmltools.convert_xgboost(model, initial_types=initial_type)

    # Start MLflow experiment
    with mlflow.start_run(run_name=str(pd.Timestamp.now())):

        # Log model parameters
        mlflow.log_params(params)

        # Log the model
        mlflow.xgboost.log_model(model, "model")
        mlflow.onnx.log_model(onnx_model, "model_onnx")
        
        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)

if __name__ == "__main__":
    config = load_config("configs/config.yml")
    dagshub.init(repo_owner='chogerlate', repo_name='wine-quality-classification-mlflow-dvc', mlflow=True)
    train_model("data/WineQT.csv", config)

