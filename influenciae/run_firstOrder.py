import sys
import json
import importlib.util
import types
import pandas as pd
import numpy as np
import tensorflow as tf
import glob
import os
import h5py
from deel.influenciae.common import InfluenceModel
from deel.influenciae.influence import FirstOrderInfluenceCalculator
#from deel.influenciae.common import LissaIHVP
from deel.influenciae.common import ExactIHVP


def get_unreduced_loss(original_loss):
    """
    Helper to convert a standard Keras loss (string or object) 
    into a loss object with reduction=tf.keras.losses.Reduction.NONE
    """
    NONE = tf.keras.losses.Reduction.NONE
    
    # generic Keras Loss object
    if hasattr(original_loss, 'get_config'):
        try:
            config = original_loss.get_config()
            config['reduction'] = NONE
            # Re-instantiate the class with the new config
            return original_loss.__class__.from_config(config)
        except Exception:
            pass # Fallback to string matching if config fails

    # it is a string (common in saved models) or failed above
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
    if loss_name in ['kld', 'kullback_leibler_divergence']:
        return tf.keras.losses.KLDivergence(reduction=NONE)

    # raise an error telling the user they must supply a compatible loss.
    raise ValueError(
        f"Could not automatically convert loss '{original_loss}' to a non-reduced format. "
        "Deel-influenciae requires reduction=tf.keras.losses.Reduction.NONE. "
        "Please re-compile your model with this specific reduction or use a standard Keras loss string."
    )

def safe_serialize(obj):
    """Helper to ensure numpy types are converted to Python native types for JSON"""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if hasattr(obj, 'item'):
        return obj.item()
    return obj


def load_model(model_path):
    """
    Load Keras/TensorFlow model
    Handles version mismatches by skipping optimizer loading,
    extracting the loss function from metadata to recompile.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")

    final_path = model_path

    # Check if the path is a directory
    if os.path.isdir(model_path):
        if "saved_model.pb" not in os.listdir(model_path):
            extensions = ['*.h5', '*.keras', '*.hdf5', '*.pb'] 
            files = []
            for ext in extensions:
                files.extend(glob.glob(os.path.join(model_path, ext)))

            if not files:
                raise FileNotFoundError(f"No valid model files ({extensions}) found in {model_path}")

            latest_model = max(files, key=os.path.getmtime)
            final_path = latest_model

    # Try loading normally
    try:
        model = tf.keras.models.load_model(final_path)
        return model
    except Exception:
        pass

    # Load without compilation to bypass optimizer errors
    try:
        model = tf.keras.models.load_model(final_path, compile=False)
    except Exception as e:
        raise RuntimeError(f"Failed to load Keras model from '{final_path}': {e}")

    #detect loss
    detected_loss = None
    
    if os.path.isfile(final_path) and (final_path.endswith('.h5') or final_path.endswith('.hdf5') or final_path.endswith('.keras')):
        try:
            with h5py.File(final_path, 'r') as f:
                if 'training_config' in f.attrs:
                    # training_config is a JSON string stored in attributes
                    train_conf_str = f.attrs.get('training_config')
                    if hasattr(train_conf_str, 'decode'):
                        train_conf_str = train_conf_str.decode('utf-8')
                    
                    train_conf = json.loads(train_conf_str)
                    
                    if 'loss' in train_conf:
                        loss_config = train_conf['loss']
                        # Attempt to deserialize using Keras utils
                        try:
                            detected_loss = tf.keras.losses.deserialize(loss_config)
                        except:
                            # use the config directly
                            detected_loss = loss_config
        except Exception:
            pass 
    
    # Fallback Heuristic if metadata extraction failed
    if detected_loss is None:
        # Guess based on last layer activation
        try:
            last_layer = model.layers[-1]
            act = getattr(last_layer, 'activation', None)
            act_name = getattr(act, '__name__', str(act)).lower()
            
            if 'sigmoid' in act_name:
                detected_loss = 'binary_crossentropy'
            elif 'softmax' in act_name:
                detected_loss = 'sparse_categorical_crossentropy'
            else:
                detected_loss = 'mse' # Regression default
        except:
            detected_loss = 'mse'

    # use a dummy optimizer because only the loss is needed for Influence calculation
    model.compile(optimizer='adam', loss=detected_loss, metrics=['accuracy'])
    
    return model

def main():

    #  READ INPUT

    try:
        input_data = sys.stdin.read()
        if not input_data:
            return # Silent exit or print empty json
        payload = json.loads(input_data)
    except Exception as e:
        print(json.dumps({"error": f"JSON Parse Error: {str(e)}"}))
        return

    model_path = payload.get("model_path")
    pipeline_path = payload.get("pipeline_path")
    sample_dict = payload.get("sample_to_predict")

    # 2. LOAD DATA

    args = types.SimpleNamespace(**payload)
    args.pipeline = pipeline_path 
    
    spec = importlib.util.spec_from_file_location("pipeline_module", pipeline_path)
    if spec is None:
        raise FileNotFoundError(f"Pipeline not found: {pipeline_path}")
    pipeline_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pipeline_module)
    
    # Run user pipeline
    X_train, X_test, y_train, y_test = pipeline_module.run_pipeline(args)


    # PREPARE DATA FOR TENSORFLOW

    
    if hasattr(X_train, "columns"):
        train_columns = X_train.columns
        X_train_np = X_train.values.astype(np.float32)
    else:
        # Fallback if X_train is already numpy or list
        X_train_np = np.asarray(X_train).astype(np.float32)
        train_columns = None 

    
    # handles different data structures
    y_train_np = np.asarray(y_train)

    # Ensure labels have shape (N, 1) for TensorFlow
    if y_train_np.ndim == 1:
        y_train_np = y_train_np.reshape(-1, 1)

    BATCH_SIZE = 32
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train_np, y_train_np)).batch(BATCH_SIZE)


    # LOAD MODEL

    #model = tf.keras.models.load_model(model_path)
    model=load_model(model_path)
    if not model.loss:
        raise ValueError("Model has no compiled loss function.")


    # PREPARE SAMPLE TO EXPLAIN

    sample_df = pd.DataFrame([sample_dict])
    if train_columns is not None:
        sample_df = sample_df[train_columns] 
    
    sample_np = sample_df.values.astype(np.float32)

    # prediction to explain
    predicted_label = model.predict(sample_np, verbose=0)
    
    # Create dataset with both X and Y
    sample_dataset = tf.data.Dataset.from_tensor_slices((sample_np, predicted_label)).batch(1)


    # SETUP INFLUENCIAE

    unreduced_loss_fn = get_unreduced_loss(model.loss)

    influence_model = InfluenceModel(
        model,
        start_layer=-1, 
        loss_function=unreduced_loss_fn 
    )

    
  # Calculator Choice
    n_params = np.sum([np.prod(v.shape) for v in model.trainable_variables])
    if n_params > 10000:
        from deel.influenciae.common import LissaIHVP
        ihvp_calculator = LissaIHVP(influence_model, train_dataset, n_opt_iters=10)
    else:
        ihvp_calculator = ExactIHVP(influence_model, train_dataset)

    calculator = FirstOrderInfluenceCalculator(
        influence_model,
        train_dataset,
        ihvp_calculator=ihvp_calculator
    )

    calculator = FirstOrderInfluenceCalculator(
        influence_model,
        train_dataset,
        ihvp_calculator=ihvp_calculator
    )


    #  COMPUTE TOP K

    K = 5
    top_k_results = calculator.top_k(sample_dataset, train_dataset, k=K)


    #  FORMAT RESULTS

    formatted_results = []
    

    for row in top_k_results.as_numpy_iterator():
        
        
        (query_sample, query_label), top_k_values, top_k_items = row
        
        vals = top_k_values.flatten()
        items_flat = top_k_items
        
        # Flatten structure if nested
        if len(items_flat.shape) > 2: 
             items_flat = items_flat[0]
        elif len(items_flat.shape) == 2 and items_flat.shape[0] == 1:
             items_flat = items_flat[0]

        top_points = []
        for i in range(len(vals)):
            score = float(vals[i])
            item_ref = items_flat[i] 
            
            original_idx = -1
            features_dict = {}
            label_val = None

            # if returns Indices
            if np.issubdtype(item_ref.dtype, np.integer):
                original_idx = int(item_ref)
                
                # Get Features from X_train using the index
                if hasattr(X_train, 'iloc'):
                    original_row = X_train.iloc[original_idx]
                    features_dict = original_row.to_dict()
                else:
                    # Fallback for numpy X_train
                    row_vals = X_train_np[original_idx]
                    if train_columns:
                        features_dict = dict(zip(train_columns, row_vals))
                    else:
                        features_dict = {f"feat_{x}": float(v) for x, v in enumerate(row_vals)}
                    
                # Get Label
                if hasattr(y_train, 'iloc'):
                    label_raw = y_train.iloc[original_idx]
                else:
                    label_raw = y_train[original_idx]
                
                # Clean Label
                if hasattr(label_raw, 'item'):
                    label_val = label_raw.item()
                elif hasattr(label_raw, 'tolist'):
                    l = label_raw.tolist()
                    label_val = l[0] if isinstance(l, list) and len(l) == 1 else l
                else:
                    label_val = label_raw

            # if returns Raw Feature Values (this is what happens with the current implementaion)
            else:
                original_idx = -1 # Index unknown
                
                # item_ref is the vector of values
                item_vec = item_ref.flatten()
                
                #  map the raw values to the saved column names
                if train_columns is not None and len(train_columns) == len(item_vec):
                    features_dict = dict(zip(train_columns, item_vec))
                else:
                    # Fallback
                    features_dict = {f"feat_{x}": float(v) for x, v in enumerate(item_vec)}
                
                try:
                    # Find matching row index
                    diffs = np.sum(np.abs(X_train_np - item_vec), axis=1)
                    match_idx = np.argmin(diffs)
                    
                    # Retrieve Label from y_train_np
                    found_label = y_train_np[match_idx]
                    
                    #
                    if found_label.size == 1:
                        # scalar 
                        label_val = found_label.item()
                    else:
                        # vector
                        label_val = found_label.tolist()
                        
                except Exception as e:
                    label_val = f"Error finding match: {str(e)}"

            # JSON 
            features_dict = {k: safe_serialize(v) for k, v in features_dict.items()}

            top_points.append({
                "training_index": original_idx,
                "influence_score": score,
                "features": features_dict,
                "label": safe_serialize(label_val)
            })
            
        formatted_results = top_points

    response = {
        "status": "success",
        "sample_analyzed": sample_dict,
        "prediction": safe_serialize(predicted_label),
        "top_influential_points": formatted_results
    }

    print(json.dumps(response, default=str))

if __name__ == "__main__":
    main()