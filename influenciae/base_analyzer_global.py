import sys
import json
import importlib.util
import types
import pandas as pd
import numpy as np
import tensorflow as tf
from abc import ABC, abstractmethod

def get_unreduced_loss(original_loss):
    """Converts a loss function to one with reduction=NONE."""
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
    if loss_name in['mse', 'meansquarederror']: return tf.keras.losses.MeanSquaredError(reduction=NONE)
    if loss_name in ['mae', 'meanabsoluteerror']: return tf.keras.losses.MeanAbsoluteError(reduction=NONE)
    if loss_name in ['categoricalcrossentropy']: return tf.keras.losses.CategoricalCrossentropy(reduction=NONE)
    if loss_name in ['sparsecategoricalcrossentropy']: return tf.keras.losses.SparseCategoricalCrossentropy(reduction=NONE)
    if loss_name in['binarycrossentropy']: return tf.keras.losses.BinaryCrossentropy(reduction=NONE)
        
    raise ValueError(f"Could not convert loss '{original_loss}' to non-reduced format.")

def safe_serialize(obj):
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    if hasattr(obj, 'item'): return obj.item()
    return obj

def extract_point_data(idx, X_data, y_data, columns):
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

    if hasattr(y_data, 'iloc'): label_raw = y_data.iloc[idx]
    else: label_raw = y_data[idx]
    
    if hasattr(label_raw, 'size') and label_raw.size > 1: label_val = label_raw.tolist()
    elif hasattr(label_raw, 'item'): label_val = label_raw.item()
    else: label_val = label_raw
        
    return features_dict, label_val

class AbstractGlobalInfluence(ABC):
    def __init__(self, k_display=10):
        self.k_display = k_display

    @abstractmethod
    def setup_calculator(self, model_path, train_dataset):
        """
        Must be implemented by subclasses. 
        Returns an influence calculator object equipped with a `.top_k()` method.
        """
        pass

    def _format_points(self, indices, scores, X_train, y_train, train_columns):
        results = []
        for idx in indices:
            score = scores[idx]
            features_dict, label_val = extract_point_data(idx, X_train, y_train, train_columns)
            features_dict = {k: safe_serialize(v) for k, v in features_dict.items()}
            results.append({
                "training_index": int(idx),
                "mean_influence_score": float(score),
                "features": features_dict,
                "label": safe_serialize(label_val)
            })
        return results

    def run(self):
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

        # LOAD DATA
        try:
            spec = importlib.util.spec_from_file_location("pipeline_module", pipeline_path)
            if spec is None: raise FileNotFoundError(f"Pipeline not found: {pipeline_path}")
            pipeline_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(pipeline_module)
            X_train, X_test, y_train, y_test = pipeline_module.run_pipeline(args)
        except Exception as e:
            print(json.dumps({"status": "error", "error": f"Pipeline Error: {str(e)}"}))
            return

        X_train_np = np.asarray(X_train).astype(np.float32)
        X_test_np = np.asarray(X_test).astype(np.float32)
        y_train_np = np.asarray(y_train)
        if y_train_np.ndim == 1: y_train_np = y_train_np.reshape(-1, 1)
        y_test_np = np.asarray(y_test)
        if y_test_np.ndim == 1: y_test_np = y_test_np.reshape(-1, 1)

        N_TRAIN = X_train_np.shape[0]
        debug_shapes = {"X_train": str(X_train_np.shape), "y_train": str(y_train_np.shape), "X_test": str(X_test_np.shape)}

        if N_TRAIN < 2:
            print(json.dumps({"status": "error", "error": "Training set has < 2 samples.", "debug_info": debug_shapes}))
            return

        # SETUP DATASETS
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train_np, y_train_np)).batch(32)
        test_dataset = tf.data.Dataset.from_tensor_slices((X_test_np, y_test_np)).batch(1)

        # DELEGATE CALCULATOR SETUP TO SUBCLASS
        try:
            calculator = self.setup_calculator(model_path, train_dataset)
        except Exception as e:
            print(json.dumps({"status": "error", "error": str(e), "debug_info": debug_shapes}))
            return

        # COMPUTE INFLUENCE
        top_k_iterator = calculator.top_k(test_dataset, train_dataset, k=N_TRAIN)
        global_influence_sum = np.zeros(N_TRAIN, dtype=np.float32)
        test_samples_count = 0
        
        train_lookup = {X_train_np[i].tobytes(): i for i in range(N_TRAIN)}

        for row in top_k_iterator:
            _, batch_values, batch_items = row
            b_vals = batch_values.numpy() 
            b_items = batch_items.numpy()

            if b_vals.ndim == 1:
                b_vals = np.expand_dims(b_vals, 0)
                b_items = np.expand_dims(b_items, 0)

            for i in range(b_vals.shape[0]):
                vals, items = b_vals[i], b_items[i]
                for j in range(len(vals)):
                    item_vec = items[j] 
                    item_bytes = item_vec.tobytes()
                    if item_bytes in train_lookup:
                        idx = train_lookup[item_bytes]
                        global_influence_sum[idx] += vals[j]
                    else:
                        diffs = np.sum(np.abs(X_train_np - item_vec), axis=1)
                        global_influence_sum[np.argmin(diffs)] += vals[j]

                test_samples_count += 1

        # RESULTS AND FORMATTING
        global_influence_scores = (global_influence_sum / test_samples_count) if test_samples_count > 0 else np.zeros(N_TRAIN)

        stats = {
            "mean_influence": float(np.mean(global_influence_scores)),
            "variance_influence": float(np.var(global_influence_scores)),
            "min_influence": float(np.min(global_influence_scores)),
            "max_influence": float(np.max(global_influence_scores)),
            "n_training_points": int(N_TRAIN),
            "n_test_samples_analyzed": int(test_samples_count)
        }

        sorted_indices = np.argsort(global_influence_scores)
        top_indices = sorted_indices[-self.k_display:][::-1]
        bottom_indices = sorted_indices[:self.k_display]

        train_columns = X_train.columns if hasattr(X_train, "columns") else None

        response = {
            "status": "success",
            "debug_info": debug_shapes,
            "global_statistics": stats,
            "most_influential_samples": self._format_points(top_indices, global_influence_scores, X_train, y_train, train_columns),
            "potential_mislabeled_samples": self._format_points(bottom_indices, global_influence_scores, X_train, y_train, train_columns)
        }

        print(json.dumps(response, default=str))