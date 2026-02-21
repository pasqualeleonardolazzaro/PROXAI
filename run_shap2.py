import argparse
import importlib.util
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import tensorflow as tf
import keras
import joblib
import pickle
import glob

def get_prediction_function(model):
    """
    Returns the appropriate prediction function for a given model.
    Prefers `predict_proba` for classifiers, otherwise falls back to `predict`.
    """
    if hasattr(model, 'predict_proba'):
        #print("Using 'predict_proba' for model explanation.") #print for debugging
        return model.predict_proba
    else:
        #print("Using 'predict' for model explanation.") #print for debugging
        return model.predict
    
def load_tensorflow_model(model_path):
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
            #print(f"Detected SavedModel directory: {model_path}")
            next
        else:
            # It is a directory of checkpoints. Find the latest file.
            #print(f"Detected checkpoint directory: {model_path}")
            
            # Look for common Keras extensions. Add others if you use specific formats.
            extensions = ['*.h5', '*.keras', '*.hdf5', '*.pb'] 
            files = []
            for ext in extensions:
                files.extend(glob.glob(os.path.join(model_path, ext)))

            if not files:
                raise FileNotFoundError(f"No valid model files ({extensions}) found in {model_path}")

            # Find the latest file based on modification time
            latest_model = max(files, key=os.path.getmtime)
            #print(f"Loading latest model: {latest_model}")
            final_path = latest_model

    # Load the model
    try:
        model = keras.models.load_model(final_path, compile=False)
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load Keras model from '{final_path}': {e}")


def load_model_and_detect_framework(model_path):
    """
    Loads a model from the given path and automatically detects its framework.
    """
    try:
        #model = tf.keras.models.load_model(model_path)
        model=load_tensorflow_model(model_path)
        #print("Successfully loaded a TensorFlow model.") #print for debugging
        return model, "tensorflow"
    except Exception:
        #print("Failed to load as a TensorFlow model. Trying scikit-learn loaders.")
        pass

    try:
        model = joblib.load(model_path)
        #print("Successfully loaded a scikit-learn model with joblib.")
        return model, "scikit-learn"
    except Exception:
        #print("Failed to load with joblib. Trying pickle.")
        pass

    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        #print("Successfully loaded a scikit-learn model with pickle.")
        return model, "scikit-learn"
    except Exception:
        raise ValueError(f"Could not load the model from {model_path}. "
                         "Unsupported model format or corrupted file.")


def main():
    parser = argparse.ArgumentParser(description="Explain a machine learning model using SHAP.")
    parser.add_argument("--model", required=True, help="Path to the pretrained model file.")
    parser.add_argument("--pipeline", required=True, help="Path to the Python file containing the preprocessing pipeline.")
    parser.add_argument("--dataset", required=True, help="Path or identifier for the dataset used for preprocessing.")
    
    args = parser.parse_args()

    #print("Starting model explanation process.") #print for debugging

    # Load and run the preprocessing pipeline
    #print(f"Loading preprocessing pipeline from: {args.pipeline}") #print for debugging
    spec = importlib.util.spec_from_file_location("pipeline_module", args.pipeline)
    if spec is None:
        raise FileNotFoundError(f"Could not find pipeline file at: {args.pipeline}")
    pipeline_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pipeline_module)
    if not hasattr(pipeline_module, 'run_pipeline'):
        raise AttributeError("The pipeline file must contain a 'run_pipeline' method.")
    
    #print("Running data preprocessing pipeline...") #print for debugging
    X_train, X_test, y_train, y_test = pipeline_module.run_pipeline(args)
    feature_names = list(X_train.columns)
    #print(f"Data preprocessing complete. Found {len(feature_names)} features.") #print for debugging

    X_train_np = X_train.to_numpy()
    X_test_np = X_test.to_numpy()

    # Load the model
    #print(f"Loading model from: {args.model}") #print for debugging
    model, framework = load_model_and_detect_framework(args.model)
    #print(f"Model framework detected: {framework}") #print for debugging

    if X_train_np.size == 0:
        raise ValueError("Training data (X_train_np) is empty. Cannot create SHAP background data.")
    
    #print("Summarizing background data for SHAP explainer.") #print for debugging
    background_data = shap.sample(X_train_np, 100)

    # Create a generic SHAP explainer
    #print("Creating SHAP explainer.") #print for debugging
    if framework == "scikit-learn":
            # Framework is scikit-learn. Using the model-agnostic KernelExplainer
            
            # Use a smaller background sample for performance
            background_data = shap.sample(X_train_np, 50) 
            original_prediction_function = get_prediction_function(model)
            def prediction_function_wrapper(X):
                # This will be a list of arrays for multi-output models
                predictions = original_prediction_function(X)

                # If it's a list (multi-output), stack the arrays horizontally.
                # Otherwise, the output is likely already a single NumPy array and can be returned as is.
                if isinstance(predictions, list):
                    return np.hstack(predictions)
                else:
                    return predictions
            
            # Pass the robust wrapper to the explainer
            explainer = shap.KernelExplainer(prediction_function_wrapper, background_data)
    else: 
        explainer = shap.Explainer(model, background_data)

    if X_test_np.size == 0:
        raise ValueError("Test data (X_test_np) is empty. Cannot calculate SHAP values.")
        
    #print("Calculating SHAP values for the test set.") #print for debugging
    shap_explanation = explainer(X_test_np)
    shap_values = shap_explanation.values
    #print("SHAP value calculation complete.") #print for debugging

    # Calculate global feature importance
    if isinstance(shap_values, list):
        #Model has multiple outputs (list format). Averaging SHAP values across all outputs for global importance
        absolute_shap_values = np.mean([np.abs(output_shap) for output_shap in shap_values], axis=0)
    else:
        # This handles both single-output (regression) and multi-output (binary classification)
        # where shap_values is a single array.
        #print("Model output is in array format.") #print for debugging
        absolute_shap_values = np.abs(shap_values)

    # Average over the samples (axis 0)
    feature_importance_scores = np.mean(absolute_shap_values, axis=0)
    
    # If the result is 2D it means we have scores for each class
    # We average them to get one global score per feature.
    #print(f"Intermediate scores shape: {feature_importance_scores.shape}") #print for debugging
    if feature_importance_scores.ndim == 2:
        feature_importance_scores = np.mean(feature_importance_scores, axis=1)

    #length check for safety 
    num_features = len(feature_names)
    if len(feature_importance_scores) != num_features:
        raise ValueError(
            f"Final score calculation failed. Number of features ({num_features}) "
            f"does not match the final number of scores ({len(feature_importance_scores)})."
        )

    # Create a DataFrame for easy sorting and display
    df_importance = pd.DataFrame({
        'Feature': feature_names,
        'Absolute Average Score': feature_importance_scores
    })
    df_importance_sorted = df_importance.sort_values(by='Absolute Average Score', ascending=False).head(5)

    # --- Save the plot ---
    #print("Generating and saving feature importance plot.")
    plt.figure(figsize=(8, 6))
    plt.bar(df_importance_sorted['Feature'], df_importance_sorted['Absolute Average Score'], color='skyblue')
    plt.title('Top 5 SHAP Feature Importances')
    plt.xlabel('Feature')
    plt.ylabel('Mean Absolute SHAP Value')
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    output_plot_path = os.path.join(os.getcwd(), "shap_top5.png")
    plt.savefig(output_plot_path)
    plt.close()
    #print(f"Plot saved to: {output_plot_path}")

    # --- Print result to stdout as JSON ---
    result = {
        "top_features": df_importance_sorted.to_dict(orient="records"),
        "plot_path": output_plot_path
    }
    print(json.dumps(result, indent=4))

if __name__ == "__main__":
    main()