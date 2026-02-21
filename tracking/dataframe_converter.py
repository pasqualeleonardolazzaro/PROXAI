import pandas as pd
import numpy as np
import tensorflow as tf
from typing import Union


def convert_to_dataframe(data: Union[pd.DataFrame, np.ndarray, tf.data.Dataset]) -> pd.DataFrame:
    """
    Convert various data types to pandas DataFrame with standardized column names.
    
    Parameters:
    -----------
    data : pd.DataFrame, np.ndarray, or tf.data.Dataset
        Input data to be converted
    
    Returns:
    --------
    pd.DataFrame
        Converted pandas DataFrame with proper column names
    """
    
    # Case 1: Already a pandas DataFrame
    if isinstance(data, pd.DataFrame):
        df = data.copy()
        # Check if columns are unnamed (default RangeIndex or integers)
        if all(isinstance(col, int) for col in df.columns):
            df.columns = [f'feature_{i+1}' for i in range(len(df.columns))]
        return df
    
    # Case 2: NumPy array
    elif isinstance(data, np.ndarray):
        # Handle 1D arrays
        if data.ndim == 1:
            df = pd.DataFrame(data, columns=['feature_1'])
        # Handle 2D arrays
        elif data.ndim == 2:
            n_cols = data.shape[1]
            df = pd.DataFrame(data, columns=[f'feature_{i+1}' for i in range(n_cols)])
        else:
            raise ValueError(f"NumPy arrays with {data.ndim} dimensions are not supported. Only 1D and 2D arrays are allowed.")
        return df
    
    # Case 3: TensorFlow Dataset
    elif isinstance(data, tf.data.Dataset):
        # Convert TensorFlow dataset to list of samples
        samples = []
        for batch in data:
            # Handle different batch structures
            if isinstance(batch, tuple):
                # Typically (features, labels) - we'll use features
                batch_data = batch[0]
            else:
                batch_data = batch
            
            # Convert tensor to numpy
            if isinstance(batch_data, tf.Tensor):
                batch_np = batch_data.numpy()
            else:
                batch_np = batch_data
            
            # Handle batched data
            if batch_np.ndim > 1:
                samples.extend(batch_np)
            else:
                samples.append(batch_np)
        
        # Convert to numpy array then to DataFrame
        arr = np.array(samples)
        
        if arr.ndim == 1:
            df = pd.DataFrame(arr, columns=['feature_1'])
        else:
            n_cols = arr.shape[1]
            df = pd.DataFrame(arr, columns=[f'feature_{i+1}' for i in range(n_cols)])
        
        return df
    
    else:
        raise TypeError(f"Unsupported data type: {type(data)}. Supported types are: pd.DataFrame, np.ndarray, tf.data.Dataset")