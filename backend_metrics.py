import argparse
import importlib.util
import pandas as pd
import numpy as np
import pickle
import sys
import os
import tensorflow as tf
import keras
import glob
""""
def load_model(model_path):

    Load Keras/TensorFlow model.
    Supports .h5, .keras, or SavedModel directories.

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    # Load the model (compile=False is usually safer if you don't need to train further)
    try:
        print(model_path)
        model = keras.models.load_model(model_path, compile=False)
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load Keras model: {e}")
"""  
def load_model(model_path):
    """
    Load Keras/TensorFlow model.
    
    If model_path is a file or a standard SavedModel directory, it loads it.
    If model_path is a directory of checkpoints, it loads the latest one (by modification time).
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")

    final_path = model_path

    # Check if the path is a directory
    if os.path.isdir(model_path):
        # EDGE CASE: A "SavedModel" is a directory containing 'saved_model.pb'.
        # If this file exists, we treat the directory as a single model.
        if "saved_model.pb" in os.listdir(model_path):
            print(f"Detected SavedModel directory: {model_path}")
        else:
            # It is a directory of checkpoints. Find the latest file.
            print(f"Detected checkpoint directory: {model_path}")
            
            # Look for common Keras extensions. Add others if you use specific formats.
            extensions = ['*.h5', '*.keras', '*.hdf5', '*.pb'] 
            files = []
            for ext in extensions:
                files.extend(glob.glob(os.path.join(model_path, ext)))

            if not files:
                raise FileNotFoundError(f"No valid model files ({extensions}) found in {model_path}")

            # Find the latest file based on modification time
            latest_model = max(files, key=os.path.getmtime)
            print(f"Loading latest model: {latest_model}")
            final_path = latest_model

    # Load the model
    try:
        model = keras.models.load_model(final_path, compile=False)
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load Keras model from '{final_path}': {e}")
    
def ensure_numpy_1d(y_data):
    """
    Helper to convert One-Hot Encoded or 2D column vectors into 1D arrays.
    Example: [[0, 1], [1, 0]] -> [1, 0]
    """
    # Convert from DataFrame/Series to Numpy if needed
    if hasattr(y_data, 'to_numpy'):
        y_data = y_data.to_numpy()
    
    y_data = np.array(y_data) # Ensure it's numpy
    
    # Case 1: One-Hot Encoded (N, Classes) where Classes > 1
    if y_data.ndim > 1 and y_data.shape[1] > 1:
        return np.argmax(y_data, axis=1)
    
    # Case 2: Column vector (N, 1)
    if y_data.ndim > 1 and y_data.shape[1] == 1:
        return y_data.ravel()
    
    # Case 3: Already 1D (N,)
    return y_data

def main():
    parser = argparse.ArgumentParser(description="Calculate Global Metrics for Streamlit (TensorFlow).")
    parser.add_argument("--model", required=True, help="Path to the trained model.")
    parser.add_argument("--pipeline", required=True, help="Path to the preprocessing pipeline.")
    parser.add_argument("--dataset", required=True, help="Path to the dataset.")
    parser.add_argument("--output_file", required=True, help="Where to save the temp results.")
    
    args = parser.parse_args()

    # 1. Load Pipeline
    spec = importlib.util.spec_from_file_location("pipeline_module", args.pipeline)
    if spec is None:
        raise FileNotFoundError(f"Could not find pipeline file at: {args.pipeline}")
    pipeline_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pipeline_module)
    
    if not hasattr(pipeline_module, 'run_pipeline'):
        raise AttributeError("The pipeline file must contain a 'run_pipeline' method.")
    
    # 2. Run Pipeline
    X_train, X_test, y_train, y_test = pipeline_module.run_pipeline(args)
    
    # --- FIX START: Sanitize y_test immediately ---
    # This converts (178, 2) -> (178,) so Pandas won't crash
    y_test_processed = ensure_numpy_1d(y_test)
    # --- FIX END ---

    # Prepare Input for TF
    if hasattr(X_test, 'to_numpy'):
        X_test_input = X_test.to_numpy().astype('float32')
    else:
        X_test_input = np.array(X_test).astype('float32')

    # 3. Predict
    model = load_model(args.model)
    raw_preds = model.predict(X_test_input, verbose=0)
    
    # 4. Process Predictions
    y_pred_labels = None
    task_type = "regression" 

    # Determine type based on output shape
    if raw_preds.shape[1] > 1:
        task_type = "classification"
        # Softmax -> Class Index
        y_pred_labels = np.argmax(raw_preds, axis=1)
    else:
        # Binary or Regression
        # Check unique values in the NOW PROCESSED y_test
        unique_targets = len(np.unique(y_test_processed))
        
        if unique_targets < 20: # Heuristic for classification
            task_type = "classification"
            y_pred_labels = (raw_preds > 0.5).astype(int).flatten()
        else:
            task_type = "regression"
            y_pred_labels = raw_preds.flatten()

    # 5. Save Results
    results = {
        "X_test": X_test,
        "y_test": y_test_processed, # Sending the 1D version
        "y_pred": y_pred_labels,
        "task_type": task_type
    }

    with open(args.output_file, "wb") as f:
        pickle.dump(results, f)

if __name__ == "__main__":
    main()