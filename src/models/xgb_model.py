import xgboost as xgb

def initialize_xgb_model(params: dict) -> xgb.XGBClassifier:
    """
    Summary: Initialize an XGBoost model with the given parameters.

    Args:
        params (dict): A dictionary containing the hyperparameters for the XGBoost model.

    Returns:
        xgb.XGBClassifier: An initialized XGBoost classifier model.
    """
    model = xgb.XGBClassifier(**params)
    return model
