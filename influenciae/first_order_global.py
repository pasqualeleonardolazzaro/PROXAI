import os
import glob
import h5py
import json
import numpy as np
import tensorflow as tf
from deel.influenciae.common import InfluenceModel, ExactIHVP
from deel.influenciae.influence import FirstOrderInfluenceCalculator

from base_analyzer_global import AbstractGlobalInfluence, get_unreduced_loss

def load_model(model_path):
    """Load Keras/TensorFlow model specific to FirstOrder setup."""
    if not os.path.exists(model_path): raise FileNotFoundError(f"Model not found at {model_path}")
    final_path = model_path

    if os.path.isdir(model_path):
        if "saved_model.pb" not in os.listdir(model_path):
            files = []
            for ext in['*.h5', '*.keras', '*.hdf5', '*.pb']:
                files.extend(glob.glob(os.path.join(model_path, ext)))
            if not files: raise FileNotFoundError(f"No valid models found in {model_path}")
            final_path = max(files, key=os.path.getmtime)

    try: return tf.keras.models.load_model(final_path)
    except Exception: pass

    try: model = tf.keras.models.load_model(final_path, compile=False)
    except Exception as e: raise RuntimeError(f"Failed to load Keras model: {e}")

    detected_loss = None
    if os.path.isfile(final_path) and final_path.endswith(('.h5', '.hdf5', '.keras')):
        try:
            with h5py.File(final_path, 'r') as f:
                if 'training_config' in f.attrs:
                    train_conf_str = f.attrs.get('training_config')
                    if hasattr(train_conf_str, 'decode'): train_conf_str = train_conf_str.decode('utf-8')
                    train_conf = json.loads(train_conf_str)
                    if 'loss' in train_conf:
                        loss_config = train_conf['loss']
                        try: detected_loss = tf.keras.losses.deserialize(loss_config)
                        except: detected_loss = loss_config
        except Exception: pass 
    
    if detected_loss is None:
        try:
            act_name = getattr(getattr(model.layers[-1], 'activation', None), '__name__', '').lower()
            detected_loss = 'binary_crossentropy' if 'sigmoid' in act_name else ('sparse_categorical_crossentropy' if 'softmax' in act_name else 'mse')
        except: detected_loss = 'mse'

    model.compile(optimizer='adam', loss=detected_loss, metrics=['accuracy'])
    return model


class FirstOrderGlobalAnalyzer(AbstractGlobalInfluence):
    def setup_calculator(self, model_path, train_dataset):
        model = load_model(model_path)
        if not model.loss: raise ValueError("Model has no compiled loss function.")
        unreduced_loss_fn = get_unreduced_loss(model.loss)
        influence_model = InfluenceModel(model, start_layer=-1, loss_function=unreduced_loss_fn)

        n_params = np.sum([np.prod(v.shape) for v in model.trainable_variables])
        if n_params > 10000:
            from deel.influenciae.common import LissaIHVP
            ihvp_calculator = LissaIHVP(influence_model, train_dataset, n_opt_iters=10)
        else:
            ihvp_calculator = ExactIHVP(influence_model, train_dataset)

        return FirstOrderInfluenceCalculator(influence_model, train_dataset, ihvp_calculator=ihvp_calculator)

if __name__ == "__main__":
    analyzer = FirstOrderGlobalAnalyzer(k_display=10) # modify 10, 50, 100 here
    analyzer.run()