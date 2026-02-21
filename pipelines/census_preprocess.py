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

    # Assign names to columns
    names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship',
                'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'label']

    df.columns = names

    #columns for the apply
    columns = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country',
            'label']
    df[columns] = df[columns].applymap(str.strip)
    df = df.replace('?', 0)

    #replace
    df = df.replace({'sex': {'Male': 1, 'Female': 0}, 'label': {'<=50K': 0, '>50K': 1}})

    df = df.drop(['fnlwgt'], axis=1)

    #rename
    df.rename(columns={'hours-per-week': 'hw'}, inplace=True)



    workclass_trans = LabelEncoder()
    df["workclass"] = workclass_trans.fit_transform(df["workclass"].values.astype(str))

    education_trans = LabelEncoder()
    df["education"] = education_trans.fit_transform(df["education"].values.astype(str))

    marital_status_trans = LabelEncoder()
    df["marital-status"] = marital_status_trans.fit_transform(df["marital-status"].values.astype(str))


    occupation_trans = LabelEncoder()
    df["occupation"] = occupation_trans.fit_transform(df["occupation"].values.astype(str))

    relationship_trans = LabelEncoder()
    df["relationship"] = relationship_trans.fit_transform(df["relationship"].values.astype(str))

    race_trans = LabelEncoder()
    df["race"] = race_trans.fit_transform(df["race"])

    native_country_trans = LabelEncoder()
    df["native-country"] = native_country_trans.fit_transform(df["native-country"].values.astype(str))


    y = df["label"]
    df = df.drop(columns=["label"])
    

    y = to_categorical(y.values, num_classes=2)

    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=42)


    return(X_train, X_test, y_train, y_test)