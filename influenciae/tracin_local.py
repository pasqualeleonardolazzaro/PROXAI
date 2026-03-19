import os
import h5py
import json
import tensorflow as tf
from deel.influenciae.common import InfluenceModel
from deel.influenciae.trac_in import TracIn
from base_analyzer_local import AbstractLocalInfluence, get_unreduced_loss

def robust_load_single_model(checkpoint_path):
    try:
        model = tf.keras.models.load_model(checkpoint_path)
        if model.loss: return model, get_unreduced_loss(model.loss)
    except Exception: pass

    try: model = tf.keras.models.load_model(checkpoint_path, compile=False)
    except Exception as e: raise RuntimeError(f"Failed to load Keras model: {e}")

    detected_loss = None
    if os.path.isfile(checkpoint_path) and checkpoint_path.endswith(('.h5', '.hdf5', '.keras')):
        try:
            with h5py.File(checkpoint_path, 'r') as f:
                if 'training_config' in f.attrs:
                    train_conf_str = f.attrs.get('training_config')
                    if hasattr(train_conf_str, 'decode'): train_conf_str = train_conf_str.decode('utf-8')
                    train_conf = json.loads(train_conf_str)
                    if 'loss' in train_conf:
                        try: detected_loss = tf.keras.losses.deserialize(train_conf['loss'])
                        except: detected_loss = train_conf['loss']
        except Exception: pass 
    
    if detected_loss is None:
        try:
            act_name = getattr(getattr(model.layers[-1], 'activation', None), '__name__', '').lower()
            detected_loss = 'binary_crossentropy' if 'sigmoid' in act_name else ('sparse_categorical_crossentropy' if 'softmax' in act_name else 'mse')
        except: detected_loss = 'mse'

    model.compile(optimizer='adam', loss=detected_loss, metrics=['accuracy'])
    return model, get_unreduced_loss(model.loss)


def load_model_list_from_checkpoints(checkpoint_dir, unreduced_loss_fn=None, start_layer=-1):
    model_list =[]
    if not os.path.isdir(checkpoint_dir): raise NotADirectoryError(f"Must be a directory: {checkpoint_dir}")
        
    checkpoint_files = sorted([f for f in os.listdir(checkpoint_dir) if f.endswith('.h5')])
    if not checkpoint_files: raise FileNotFoundError(f"No .h5 files found in {checkpoint_dir}")
        
    loaded_model = None
    for checkpoint_file in checkpoint_files:
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
        
        # Proper unpacking
        loaded_model, checkpoint_loss_fn = robust_load_single_model(checkpoint_path)
        
        if unreduced_loss_fn is None:
            unreduced_loss_fn = checkpoint_loss_fn
            
        influence_model = InfluenceModel(loaded_model, start_layer=start_layer, loss_function=unreduced_loss_fn)
        model_list.append(influence_model)
    
    return model_list, loaded_model

class TracInLocalAnalyzer(AbstractLocalInfluence):
    def load_and_setup(self, model_path, train_dataset):
        model_list, model = load_model_list_from_checkpoints(model_path)

        try:
            lr_var = model.optimizer.learning_rate
            lr = float(lr_var.numpy()) if hasattr(lr_var, 'numpy') else float(lr_var(0))
        except Exception:
            lr = 0.001

        calculator = TracIn(model_list, lr)
        
        return calculator, model

if __name__ == "__main__":
    analyzer = TracInLocalAnalyzer(k_display=5)
    analyzer.run()