import os
import glob
import h5py
import json
import tensorflow as tf
from deel.influenciae.common import InfluenceModel
from deel.influenciae.trac_in import TracIn

from base_analyzer_global import AbstractGlobalInfluence, get_unreduced_loss


def robust_load_single_model(checkpoint_path):
    """
    Robustly loads a single Keras/TensorFlow model checkpoint.
    Handles version mismatches by skipping optimizer loading,
    extracting the loss function from metadata to recompile.
    """
    # 1. Try loading normally
    try:
        model = tf.keras.models.load_model(checkpoint_path)
        if model.loss:
            return model, get_unreduced_loss(model.loss)
    except Exception:
        pass

    # 2. Load without compilation to bypass optimizer errors
    try:
        model = tf.keras.models.load_model(checkpoint_path, compile=False)
    except Exception as e:
        raise RuntimeError(f"Failed to load Keras model from '{checkpoint_path}': {e}")

    detected_loss = None
    
    # 3. Detect loss via h5py metadata extraction
    if os.path.isfile(checkpoint_path) and checkpoint_path.endswith(('.h5', '.hdf5', '.keras')):
        try:
            with h5py.File(checkpoint_path, 'r') as f:
                if 'training_config' in f.attrs:
                    train_conf_str = f.attrs.get('training_config')
                    if hasattr(train_conf_str, 'decode'):
                        train_conf_str = train_conf_str.decode('utf-8')
                    train_conf = json.loads(train_conf_str)
                    
                    if 'loss' in train_conf:
                        loss_config = train_conf['loss']
                        try:
                            detected_loss = tf.keras.losses.deserialize(loss_config)
                        except:
                            detected_loss = loss_config
        except Exception:
            pass 
    
    # 4. Fallback Heuristic based on the last layer
    if detected_loss is None:
        try:
            act_name = getattr(getattr(model.layers[-1], 'activation', None), '__name__', '').lower()
            detected_loss = 'binary_crossentropy' if 'sigmoid' in act_name else ('sparse_categorical_crossentropy' if 'softmax' in act_name else 'mse')
        except:
            detected_loss = 'mse'

    # Recompile to attach the loss function
    model.compile(optimizer='adam', loss=detected_loss, metrics=['accuracy'])
    return model, get_unreduced_loss(model.loss)


def load_model_list_from_checkpoints(checkpoint_dir, unreduced_loss_fn=None, start_layer=-1):
    model_list =[]
    
    if not os.path.isdir(checkpoint_dir):
        raise NotADirectoryError(f"Selected model path is not a directory: {checkpoint_dir}")
        
    checkpoint_files = sorted([f for f in os.listdir(checkpoint_dir) if f.endswith('.h5')])
    
    if not checkpoint_files:
        raise FileNotFoundError(f"No .h5 checkpoint files found in {checkpoint_dir}")
        
    loaded_model = None
    
    for checkpoint_file in checkpoint_files:
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
        

        loaded_model, checkpoint_loss_fn = robust_load_single_model(checkpoint_path)
        

        if unreduced_loss_fn is None:
            unreduced_loss_fn = checkpoint_loss_fn
            

        influence_model = InfluenceModel(
            loaded_model, 
            start_layer=start_layer, 
            loss_function=unreduced_loss_fn
        )
        
        model_list.append(influence_model)
    
    return model_list, loaded_model

class TracInGlobalAnalyzer(AbstractGlobalInfluence):
    def setup_calculator(self, model_path, train_dataset):
        try:
            model_list, model = load_model_list_from_checkpoints(model_path)
        except Exception as e:
            raise ValueError(f"selected model must be a directory containing checkpoints. Error: {e}")

        try:
            lr_var = model.optimizer.learning_rate
            lr = float(lr_var.numpy()) if hasattr(lr_var, 'numpy') else float(lr_var(0))
        except Exception:
            lr = 0.001

        return TracIn(model_list, lr)

if __name__ == "__main__":
    analyzer = TracInGlobalAnalyzer(k_display=10) 
    analyzer.run()