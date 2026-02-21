import sys
import json
import importlib.util
import types
import pandas as pd
import numpy as np
import tensorflow as tf
import os
from deel.influenciae.common import InfluenceModel
from deel.influenciae.trac_in import TracIn

def get_unreduced_loss(original_loss):
    NONE = tf.keras.losses.Reduction.NONE
    if hasattr(original_loss, 'get_config'):
        try:
            config = original_loss.get_config()
            config['reduction'] = NONE
            return original_loss.__class__.from_config(config)
        except Exception:
            pass 
    loss_name = ""
    if isinstance(original_loss, str):
        loss_name = original_loss.lower()
    elif hasattr(original_loss, '__name__'):
        loss_name = original_loss.__name__.lower()
        
    if loss_name in ['mse', 'mean_squared_error']:
        return tf.keras.losses.MeanSquaredError(reduction=NONE)
    if loss_name in ['mae', 'mean_absolute_error']:
        return tf.keras.losses.MeanAbsoluteError(reduction=NONE)
    if loss_name in ['categorical_crossentropy']:
        return tf.keras.losses.CategoricalCrossentropy(reduction=NONE)
    if loss_name in ['sparse_categorical_crossentropy']:
        return tf.keras.losses.SparseCategoricalCrossentropy(reduction=NONE)
    if loss_name in ['binary_crossentropy']:
        return tf.keras.losses.BinaryCrossentropy(reduction=NONE)
    raise ValueError(f"Could not convert loss '{original_loss}' to non-reduced format.")

def safe_serialize(obj):
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    if hasattr(obj, 'item'): return obj.item()
    return obj

def extract_point_data(idx, X_data, y_data, columns):
    # Features
    if hasattr(X_data, 'iloc'):
        row_vals = X_data.iloc[idx].values
        col_names = X_data.columns
    else:
        row_vals = X_data[idx]
        col_names = columns

    if col_names is not None and len(col_names) == len(row_vals):
        features_dict = dict(zip(col_names, row_vals))
    else:
        features_dict = {f"feat_{x}": float(v) for x, v in enumerate(row_vals)}

    # Label
    if hasattr(y_data, 'iloc'):
        label_raw = y_data.iloc[idx]
    else:
        label_raw = y_data[idx]
    
    if hasattr(label_raw, 'size') and label_raw.size > 1:
        label_val = label_raw.tolist()
    elif hasattr(label_raw, 'item'):
        label_val = label_raw.item()
    else:
        label_val = label_raw
        
    return features_dict, label_val

def load_model_list_from_checkpoints(checkpoint_dir, unreduced_loss_fn=None, start_layer=-1):
    """
    Load all saved model checkpoints and recreate the model_list.
    
    Args:
        checkpoint_dir: Directory containing the .h5 checkpoint files
        unreduced_loss_fn: The unreduced loss function for InfluenceModel
        start_layer: The start_layer parameter for InfluenceModel
    
    Returns:
        List of InfluenceModel objects
    """
    model_list = []
    
    
    # Get all checkpoint files and sort them
    checkpoint_files = sorted([f for f in os.listdir(checkpoint_dir) if f.endswith('.h5')])
    
    for checkpoint_file in checkpoint_files:
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
        
        # Load the model
        loaded_model = tf.keras.models.load_model(checkpoint_path)
        if unreduced_loss_fn is None:
            unreduced_loss_fn=get_unreduced_loss(loaded_model.loss)
        
        # Create InfluenceModel and append to list
        influence_model = InfluenceModel(loaded_model, start_layer=-1, 
                                        loss_function=unreduced_loss_fn)
        model_list.append(influence_model)
    
    return model_list,loaded_model

def main():
    try:
        input_data = sys.stdin.read()
        if not input_data: return 
        payload = json.loads(input_data)
    except Exception as e:
        print(json.dumps({"error": f"JSON Parse Error: {str(e)}"}))
        return

    model_path = payload.get("model_path")
    pipeline_path = payload.get("pipeline_path")
    args = types.SimpleNamespace(**payload)
    args.pipeline = pipeline_path 

    #  LOAD DATA
    spec = importlib.util.spec_from_file_location("pipeline_module", pipeline_path)
    if spec is None: raise FileNotFoundError(f"Pipeline not found: {pipeline_path}")
    pipeline_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pipeline_module)
    
    X_train, X_test, y_train, y_test = pipeline_module.run_pipeline(args)

    X_train_np = np.asarray(X_train).astype(np.float32)
    X_test_np = np.asarray(X_test).astype(np.float32)
    
    y_train_np = np.asarray(y_train)
    if y_train_np.ndim == 1: y_train_np = y_train_np.reshape(-1, 1)
    
    y_test_np = np.asarray(y_test)
    if y_test_np.ndim == 1: y_test_np = y_test_np.reshape(-1, 1)

    N_TRAIN = X_train_np.shape[0]
    
    debug_shapes = {
        "X_train": str(X_train_np.shape),
        "y_train": str(y_train_np.shape),
        "X_test": str(X_test_np.shape)
    }

    if N_TRAIN < 2:
        print(json.dumps({"status": "error", "error": "Training set has < 2 samples.", "debug_info": debug_shapes}))
        return

    # SETUP DATASETS
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train_np, y_train_np)).batch(32)
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test_np, y_test_np)).batch(1)

    # SETUP MODEL
    try:
        model_list,model =load_model_list_from_checkpoints(model_path)
    except Exception as e:
        print(json.dumps({"error": f"selected model must be a directory containing checkpoints"}))
        return

    try:
        #find the learning rate
        lr_var = model.optimizer.learning_rate
        
        if hasattr(lr_var, 'numpy'):
            lr = float(lr_var.numpy())
        else:
            # Fallback if it's a schedule object
            lr = float(lr_var(0))
            
    except Exception:
        lr = 0.001

    calculator = TracIn(
    model_list, lr 
    )

    # COMPUTE INFLUENCE
    k_all = N_TRAIN #all training points
    top_k_iterator = calculator.top_k(test_dataset, train_dataset, k=k_all)

    global_influence_sum = np.zeros(N_TRAIN, dtype=np.float32)
    test_samples_count = 0
    
    # Pre-compute a lookup map for Training Rows
    # to map the feature vectors returned by top_k back to their index faster.
    train_lookup = {}
    for i in range(N_TRAIN):
        row_bytes = X_train_np[i].tobytes()
        train_lookup[row_bytes] = i

    for row in top_k_iterator:
        _, batch_values, batch_items = row
        
        b_vals = batch_values.numpy() 
        b_items = batch_items.numpy()

        if b_vals.ndim == 1:
            b_vals = np.expand_dims(b_vals, 0)
            b_items = np.expand_dims(b_items, 0)

        # loop over all samples
        for i in range(b_vals.shape[0]):
            vals = b_vals[i]   # (N_TRAIN,)
            items = b_items[i] # (N_TRAIN, n_features)
            
            
            for j in range(len(vals)):
                # Get the feature vector from top_k results
                item_vec = items[j] 
                
                # Convert to bytes to find original index
                item_bytes = item_vec.tobytes()
                
                # Retrieve index
                if item_bytes in train_lookup:
                    idx = train_lookup[item_bytes]
                    global_influence_sum[idx] += vals[j]
                else:
                    # Fallback for floating point slight mismatches
                    diffs = np.sum(np.abs(X_train_np - item_vec), axis=1)
                    idx = np.argmin(diffs)
                    global_influence_sum[idx] += vals[j]

            test_samples_count += 1

    #  RESULTS
    if test_samples_count > 0:
        global_influence_scores = global_influence_sum / test_samples_count
    else:
        global_influence_scores = np.zeros(N_TRAIN)

    stats = {
        "mean_influence": float(np.mean(global_influence_scores)),
        "variance_influence": float(np.var(global_influence_scores)),
        "min_influence": float(np.min(global_influence_scores)),
        "max_influence": float(np.max(global_influence_scores)),
        "n_training_points": int(N_TRAIN),
        "n_test_samples_analyzed": int(test_samples_count)
    }

    K_display = 10 #modifica qui per le prove 10,50,100
    sorted_indices = np.argsort(global_influence_scores)
    
    top_indices = sorted_indices[-K_display:][::-1]
    bottom_indices = sorted_indices[:K_display]
    #bottom_indices = sorted_indices[-K_display:][::-1]
    #top_indices = sorted_indices[:K_display]

    train_columns = X_train.columns if hasattr(X_train, "columns") else None

    def format_points(indices):
        results = []
        for idx in indices:
            score = global_influence_scores[idx]
            features_dict, label_val = extract_point_data(idx, X_train, y_train, train_columns)
            features_dict = {k: safe_serialize(v) for k, v in features_dict.items()}
            results.append({
                "training_index": int(idx),
                "mean_influence_score": float(score),
                "features": features_dict,
                "label": safe_serialize(label_val)
            })
        return results

    top_influential_results = format_points(top_indices)
    potential_mislabeled_results = format_points(bottom_indices)

    response = {
        "status": "success",
        "debug_info": debug_shapes,
        "global_statistics": stats,
        "most_influential_samples": top_influential_results,
        "potential_mislabeled_samples": potential_mislabeled_results
    }

    print(json.dumps(response, default=str))

if __name__ == "__main__":
    main()