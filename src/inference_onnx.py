# test onnx model inference on sample
import onnxruntime as ort
import numpy as np

def infer_onnx_model_on_sample(model_path: str, input_data: np.ndarray) -> np.ndarray:
    """
    Summary: Perform inference using an ONNX model on sample data.

    Args:
        model_path (str): The path to the ONNX model file.
        input_data (np.ndarray): The input data for the model in the form of a NumPy array.

    Returns:
        np.ndarray: The model's predictions as a NumPy array.
    
    Example:
        Usage:
        >>> predictions = infer_onnx_model_on_sample("xgb_model.onnx", x_test[:1].to_numpy())
        >>> print("Predictions on sample data:", predictions)    
        Results:
        >>> Predictions on sample data: [0]
    """
    # Create an inference session
    session = ort.InferenceSession(model_path)

    # Get the name of the input layer
    input_name = session.get_inputs()[0].name

    # Perform inference
    predictions = session.run(None, {input_name: input_data.astype(np.float32)})

    return predictions[0]

if __name__ == "__main__":
    pass