import pandas as pd
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf


#def run_pipeline(args, tracker)
def run_pipeline(args,tracker=None) -> None:

    input_path = args.dataset

    df = pd.read_csv(input_path, header=0)

    if hasattr(args, 'frac') and args.frac != 0.0:
        df = df.sample(frac=args.frac)

    if(tracker is not None):
        # Subscribe datafram
        df = tracker.subscribe(df)

    df.rename(columns={"PassengerId": "ID"}, inplace=True)

    # Remove the Name, Ticket and Cabin, which might not be useful
    df = df.drop(columns=["Name", "Ticket", "Cabin"])

    # Drop Nan Value in Embarked
    df = df.dropna(subset=["Embarked"])

    # Deal with Missing Age value. The way I did here is adding one more column justifying whether the age is missing or not.
    df['MissAge'] = df['Age'].isna().astype(int)
    df.fillna({'Age':0}, inplace=True)


    sex_trans = LabelEncoder()
    df["Sex"] = sex_trans.fit_transform(df["Sex"])

    Emb_trans = LabelEncoder()
    df["Embarked"] = sex_trans.fit_transform(df["Embarked"])

    
    y = df["Survived"]
    df = df.drop(columns=["Survived"])

    # The Step here is to make sure ID will not be the feature that affects the training process. 
    # Thus, turning that to a really small value.
    IDs = df["ID"].values.reshape(-1, 1).astype(np.float32)
    IDs = IDs  / 1e7

    df = df.drop(columns=["ID"])

    #convert to numpy
    df = df.values

    #change data type
    df = df.astype(np.float32)
    df = np.hstack((df, IDs))

    y = to_categorical(y.values, num_classes=2)

    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=42)

    df = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))


    return(df,test_ds)