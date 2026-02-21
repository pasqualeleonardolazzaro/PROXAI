import pandas as pd
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf


def run_pipeline(args, tracker=None) -> None:

    # Load the dataset from the specified input path
    input_path = args.dataset

    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(input_path, header=0)

    # If a fraction is specified, sample the DataFrame to the specified fraction
    if hasattr(args, 'frac') and args.frac != 0.0:
        df = df.sample(frac=args.frac)

    # If a tracker is provided, subscribe the DataFrame to the tracker
    if(tracker is not None):
        # Subscribe the DataFrame to the tracker
        df = tracker.subscribe(df)
    tracker.analyze_changes(df)

    # Rename the 'PassengerId' column to 'ID'
    df.rename(columns={"PassengerId": "ID"}, inplace=True)
    # Remove the 'Name', 'Ticket', and 'Cabin' columns, which might not be useful
    df = df.drop(columns=["Name", "Ticket", "Cabin"])
    tracker.analyze_changes(df)

    # Drop rows with missing values in the 'Embarked' column
    df = df.dropna(subset=["Embarked"])
    # Create a new column 'MissAge' to indicate whether the 'Age' is missing
    df['MissAge'] = df['Age'].isna().astype(int)
    # Fill missing 'Age' values with 0
    df.fillna({'Age':0}, inplace=True)
    tracker.analyze_changes(df)

    # Create a LabelEncoder to transform categorical variables
    sex_trans = LabelEncoder()
    # Transform the 'Sex' column using the LabelEncoder
    df["Sex"] = sex_trans.fit_transform(df["Sex"])
    # Transform the 'Embarked' column using the LabelEncoder
    Emb_trans = LabelEncoder()
    df["Embarked"] = sex_trans.fit_transform(df["Embarked"])
    tracker.analyze_changes(df)

    # Split the data into features and target
    y = df["Survived"]
    # Drop the 'Survived' column from the features
    df = df.drop(columns=["Survived"])
    tracker.analyze_changes(df)

    # Convert the 'ID' column to float32
    df['ID'] = df['ID'].astype(np.float32)
    # Divide all values in the 'ID' column by 1e7
    df['ID'] = df['ID'] / 1e7
    # Move the 'ID' column to the last position
    df = df[[col for col in df.columns if col != 'ID'] + ['ID']]
    tracker.analyze_changes(df)

    # One-hot encode the target variable
    y = to_categorical(y.values, num_classes=2)
    tracker.analyze_changes(df)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=42)
    tracker.analyze_changes(df)

    return(X_train, X_test, y_train, y_test)