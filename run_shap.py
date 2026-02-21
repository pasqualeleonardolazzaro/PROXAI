import argparse
import importlib.util
import tensorflow as tf
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json   
import sys    
import os 


# Function to extract Tensors/NumPy arrays from a dataset object
def extract_from_dataset(dataset):
    features_list = []
    labels_list = []
    for features, labels in dataset:
        features_list.append(features.numpy())
        labels_list.append(labels.numpy())
    return np.array(features_list), np.array(labels_list)

def main():
    parser = argparse.ArgumentParser(description="Explain a TensorFlow model using SHAP.")
    parser.add_argument("--model", required=True, help="Path to the TensorFlow pretrained model.")
    parser.add_argument("--pipeline", required=True, help="Path to the Python file containing the preprocessing pipeline.")
    parser.add_argument("--dataset", required=True, help="Path or identifier for the dataset used for preprocessing.")
    
    args = parser.parse_args()

    #  Load the preprocessing pipeline
    pipeline_path = args.pipeline
    spec = importlib.util.spec_from_file_location("pipeline_module", pipeline_path)
    if spec is None:
        raise FileNotFoundError(f"Could not find pipeline file at: {pipeline_path}")
    pipeline_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pipeline_module)

    if not hasattr(pipeline_module, 'run_pipeline'):
        raise AttributeError("The pipeline file must contain a 'run_pipeline' method.")


    X_train, X_test, y_train, y_test=pipeline_module.run_pipeline(args)
    feature_names=list(X_train.columns)
    #train_ds, test_ds = pipeline_module.run_pipeline(args)
    #train_ds, test_ds = pipeline_module.run_pipeline(args.dataset)
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))


    # Extract numpy arrays from the datasets

    X_train_np, y_train_np = extract_from_dataset(train_ds)
    X_test_np, y_test_np = extract_from_dataset(test_ds)


    # Load the TensorFlow model

    model = tf.keras.models.load_model(args.model)
    #model.summary()

    # Use SHAP to explain the model
    # Ensure background data has the same shape as model input
    if X_train_np.size == 0:
        raise ValueError("Training data (X_train_np) is empty. Cannot create SHAP background data.")
    
    background_data = shap.sample(X_train_np, 100) # Using 100 samples for background


    # Create the GradientExplainer instance.
    explainer = shap.GradientExplainer(model, background_data)


    # Calculate SHAP values for the test set.
    if X_test_np.size == 0:
        raise ValueError("Test data (X_test_np) is empty. Cannot calculate SHAP values.")
        
    shap_values = explainer.shap_values(X_test_np)


    # Handle single output vs multiple outputs for shap_values structure
    # GradientExplainer returns a list of arrays if the model has multiple outputs
    # or if it's a multi-class classification where each class gets its own SHAP values.
    # For a single-output regression model, it might return a single array.
    
    if isinstance(shap_values, list):
        # Assuming for now that all outputs have the same feature dimension.
        # We will iterate through each output to calculate feature importance.
        # For simplicity, if you only care about one output, you can select it.
        
        # Stack shap_values to get a consistent 3D array (samples, features, outputs)
        # This assumes all sub-arrays in shap_values have the same shape[1] (features)
        try:
            shap_values_stacked = np.stack(shap_values, axis=-1)
        except ValueError as e:
            print(f"Warning: Could not stack SHAP values directly. Likely due to inconsistent shapes across outputs: {e}")
            print("Processing each output's SHAP values separately for importance calculation.")
            shap_values_processed = shap_values # Keep as list for iteration
            num_outputs = len(shap_values_processed)
            # Find the number of features from the first output
            if num_outputs > 0 and shap_values_processed[0].ndim >= 2:
                num_features = shap_values_processed[0].shape[1]
            else:
                raise ValueError("Could not determine number of features from SHAP values.")

        else:
            shap_values_processed = shap_values_stacked
            num_outputs = shap_values_processed.shape[-1]
            num_features = shap_values_processed.shape[1]

    else:
        # Single output model, shap_values is a single numpy array (samples, features)
        shap_values_processed = shap_values
        num_outputs = 1
        num_features = shap_values_processed.shape[1]

    # Calculate the absolute average score for each feature and each output


    if num_outputs > 1 and isinstance(shap_values_processed, list):
        # Handle the case where shap_values is a list of arrays (multiple outputs)
        # and stacking failed or we decided to iterate
        for output_idx in range(num_outputs):
            current_output_shap_values = shap_values_processed[output_idx]
            # Ensure current_output_shap_values is 2D (samples, features)
            if current_output_shap_values.ndim == 1:
                # If 1D, it might be (samples,) for a single feature output, needs reshaping
                current_output_shap_values = current_output_shap_values.reshape(-1, 1)
            
            absolute_avg_scores_output = np.mean(np.abs(current_output_shap_values), axis=0)
            
            if not feature_names:
                feature_names = [f"Feature {i+1}" for i in range(len(absolute_avg_scores_output))]
            df_output = pd.DataFrame({'Feature': feature_names, 'Absolute Average Score': absolute_avg_scores_output})
            df_output_sorted = df_output.sort_values(by='Absolute Average Score', ascending=False).head(5)


    else:
        # Handle single output or successfully stacked multiple outputs
        # If single output, shap_values_processed is (samples, features)
        # If stacked, shap_values_processed is (samples, features, outputs)
        
        # If it's 2D, reshape it to 3D for consistent processing with 1 output
        if shap_values_processed.ndim == 2:
            absolute_avg_scores = np.mean(np.abs(shap_values_processed), axis=0)
            # This is now (features,) for a single output. We can wrap it.
            absolute_avg_scores = absolute_avg_scores.reshape(-1, 1) # Make it (features, 1)
        elif shap_values_processed.ndim == 3:
            absolute_avg_scores = np.mean(np.abs(shap_values_processed), axis=0)
            # This is now (features, outputs)
        else:
            raise ValueError(f"Unexpected SHAP values dimension: {shap_values_processed.ndim}")
            
        
        for output_idx in range(absolute_avg_scores.shape[1]):
            output_scores = absolute_avg_scores[:, output_idx]

            if not feature_names:
                feature_names = [f"Feature {i+1}" for i in range(len(output_scores))]
            df_output = pd.DataFrame({'Feature': feature_names, 'Absolute Average Score': output_scores})
            df_output_sorted = df_output.sort_values(by='Absolute Average Score', ascending=False).head(5)


    # --- ✅ Save the plot ---
    sorted_indices = np.argsort(output_scores)[::-1][:5]
    plt.figure(figsize=(8, 6))
    plt.bar(np.array(feature_names)[sorted_indices], output_scores[sorted_indices], color='skyblue')
    plt.title('Top 5 SHAP Feature Importances')
    plt.xlabel('Feature')
    plt.ylabel('Absolute Average SHAP Value')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    output_plot_path = os.path.join(os.getcwd(), "shap_top5.png")
    plt.savefig(output_plot_path)  # ✅ Save plot instead of showing it
    plt.close()

    # --- ✅ Print result to stdout as JSON ---
    result = {
        "top_features": df_output_sorted.to_dict(orient="records"),
        "plot_path": output_plot_path
    }
    print(json.dumps(result))

if __name__ == "__main__":
    main()