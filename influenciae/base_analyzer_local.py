import sys
import json
import importlib.util
import types
import pandas as pd
import numpy as np
import tensorflow as tf
from abc import ABC, abstractmethod

def get_unreduced_loss(original_loss):
    NONE = tf.keras.losses.Reduction.NONE
    if hasattr(original_loss, 'get_config'):
        try:
            config = original_loss.get_config()
            config['reduction'] = NONE
            return original_loss.__class__.from_config(config)
        except Exception:
            pass 

    if isinstance(original_loss, dict) and 'class_name' in original_loss:
        try:
            class_name = original_loss['class_name']
            config = original_loss.get('config', {})
            if hasattr(config, 'to_dict'): config = config.to_dict()
            elif not isinstance(config, dict):
                try: config = dict(config)
                except: pass
            if isinstance(config, dict):
                config = config.copy()
                config['reduction'] = NONE
                if hasattr(tf.keras.losses, class_name):
                    return getattr(tf.keras.losses, class_name).from_config(config)
                return tf.keras.losses.deserialize({'class_name': class_name, 'config': config})
        except Exception: pass

    loss_name = ""
    if isinstance(original_loss, str): loss_name = original_loss.lower()
    elif hasattr(original_loss, '__name__'): loss_name = original_loss.__name__.lower()
    elif isinstance(original_loss, dict) and 'class_name' in original_loss: loss_name = original_loss['class_name'].lower()
        
    loss_name = loss_name.replace('_', '')
    if loss_name in ['mse', 'meansquarederror']: return tf.keras.losses.MeanSquaredError(reduction=NONE)
    if loss_name in ['mae', 'meanabsoluteerror']: return tf.keras.losses.MeanAbsoluteError(reduction=NONE)
    if loss_name in ['categoricalcrossentropy']: return tf.keras.losses.CategoricalCrossentropy(reduction=NONE)
    if loss_name in ['sparsecategoricalcrossentropy']: return tf.keras.losses.SparseCategoricalCrossentropy(reduction=NONE)
    if loss_name in['binarycrossentropy']: return tf.keras.losses.BinaryCrossentropy(reduction=NONE)
    if loss_name in ['kld', 'kullbackleiblerdivergence']: return tf.keras.losses.KLDivergence(reduction=NONE)

    raise ValueError(f"Could not convert loss '{original_loss}' to non-reduced format.")

def safe_serialize(obj):
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    if hasattr(obj, 'item'): return obj.item()
    return obj


class AbstractLocalInfluence(ABC):
    def __init__(self, k_display=5):
        self.k_display = k_display

    @abstractmethod
    def load_and_setup(self, model_path, train_dataset):
        """
        Must be implemented by subclasses.
        Returns:
            calculator: The configured influence calculator.
            predictive_model: The Keras model used to generate predictions.
        """
        pass

    def run(self):
        # READ INPUT
        try:
            input_data = sys.stdin.read()
            if not input_data: return 
            payload = json.loads(input_data)
        except Exception as e:
            print(json.dumps({"error": f"JSON Parse Error: {str(e)}"}))
            return

        model_path = payload.get("model_path")
        pipeline_path = payload.get("pipeline_path")
        sample_dict = payload.get("sample_to_predict")

        # LOAD DATA
        args = types.SimpleNamespace(**payload)
        args.pipeline = pipeline_path 
        
        spec = importlib.util.spec_from_file_location("pipeline_module", pipeline_path)
        if spec is None:
            raise FileNotFoundError(f"Pipeline not found: {pipeline_path}")
        pipeline_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pipeline_module)
        
        X_train, X_test, y_train, y_test = pipeline_module.run_pipeline(args)

        # PREPARE DATA
        if hasattr(X_train, "columns"):
            train_columns = X_train.columns
            X_train_np = X_train.values.astype(np.float32)
        else:
            X_train_np = np.asarray(X_train).astype(np.float32)
            train_columns = None 

        y_train_np = np.asarray(y_train)
        if y_train_np.ndim == 1:
            y_train_np = y_train_np.reshape(-1, 1)

        BATCH_SIZE = 32
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train_np, y_train_np)).batch(BATCH_SIZE)

        # LOAD MODEL & SETUP CALCULATOR (DELEGATED)
        try:
            calculator, predictive_model = self.load_and_setup(model_path, train_dataset)
        except Exception as e:
            print(json.dumps({"error": f"Failed to load method: {str(e)}"}))
            return

        # PREPARE SAMPLE TO EXPLAIN
        sample_df = pd.DataFrame([sample_dict])
        if train_columns is not None:
            sample_df = sample_df[train_columns] 
        
        sample_np = sample_df.values.astype(np.float32)
        predicted_label = predictive_model.predict(sample_np, verbose=0)
        sample_dataset = tf.data.Dataset.from_tensor_slices((sample_np, predicted_label)).batch(1)

        # COMPUTE TOP K
        top_k_results = calculator.top_k(sample_dataset, train_dataset, k=self.k_display)

        # FORMAT RESULTS
        formatted_results =[]
        for row in top_k_results.as_numpy_iterator():
            (query_sample, query_label), top_k_values, top_k_items = row
            
            vals = top_k_values.flatten()
            items_flat = top_k_items
            
            if len(items_flat.shape) > 2: 
                 items_flat = items_flat[0]
            elif len(items_flat.shape) == 2 and items_flat.shape[0] == 1:
                 items_flat = items_flat[0]

            top_points =[]
            for i in range(len(vals)):
                score = float(vals[i])
                item_ref = items_flat[i] 
                original_idx = -1
                features_dict = {}
                label_val = None

                # if returns Indices
                if np.issubdtype(item_ref.dtype, np.integer):
                    original_idx = int(item_ref)
                    if hasattr(X_train, 'iloc'):
                        features_dict = X_train.iloc[original_idx].to_dict()
                    else:
                        row_vals = X_train_np[original_idx]
                        if train_columns: features_dict = dict(zip(train_columns, row_vals))
                        else: features_dict = {f"feat_{x}": float(v) for x, v in enumerate(row_vals)}
                        
                    label_raw = y_train.iloc[original_idx] if hasattr(y_train, 'iloc') else y_train[original_idx]
                    if hasattr(label_raw, 'item'): label_val = label_raw.item()
                    elif hasattr(label_raw, 'tolist'):
                        l = label_raw.tolist()
                        label_val = l[0] if isinstance(l, list) and len(l) == 1 else l
                    else:
                        label_val = label_raw

                # if returns Raw Feature Values
                else:
                    item_vec = item_ref.flatten()
                    if train_columns is not None and len(train_columns) == len(item_vec):
                        features_dict = dict(zip(train_columns, item_vec))
                    else:
                        features_dict = {f"feat_{x}": float(v) for x, v in enumerate(item_vec)}
                    
                    try:
                        diffs = np.sum(np.abs(X_train_np - item_vec), axis=1)
                        match_idx = np.argmin(diffs)
                        found_label = y_train_np[match_idx]
                        label_val = found_label.item() if found_label.size == 1 else found_label.tolist()
                    except Exception as e:
                        label_val = f"Error finding match: {str(e)}"

                features_dict = {k: safe_serialize(v) for k, v in features_dict.items()}
                top_points.append({
                    "training_index": original_idx,
                    "influence_score": score,
                    "features": features_dict,
                    "label": safe_serialize(label_val)
                })
                
            formatted_results = top_points

        # RESPOND
        response = {
            "status": "success",
            "sample_analyzed": sample_dict,
            "prediction": safe_serialize(predicted_label),
            "top_influential_points": formatted_results
        }
        print(json.dumps(response, default=str))