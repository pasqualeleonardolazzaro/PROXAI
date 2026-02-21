import sys

import argparse
import pandas as pd
from graph.logger import CustomLogger
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


def stratified_sample(df, frac):
    """
    Perform stratified sampling on a DataFrame.
    """
    if frac > 0.0 and frac < 1.0:
        # Infer categorical columns for stratification
        stratify_columns = df.select_dtypes(include=['object']).columns.tolist()

        # Check if any class in stratify columns has fewer than 2 members
        for col in stratify_columns:
            value_counts = df[col].value_counts()
            if value_counts.min() >= 2:
                # Perform stratified sampling for this column
                stratified_df = df.groupby(col, group_keys=False).apply(lambda x: x.sample(frac=frac))
                return stratified_df.reset_index(drop=True)

        # If no suitable stratification column is found, fall back to random sampling
        sampled_df = df.sample(frac=frac).reset_index(drop=True)
    else:
        sampled_df = df
    return sampled_df


def run_pipeline(args, tracker) -> None:
    input_path = args.dataset

    df = pd.read_csv(input_path)

    if args.frac != 0.0:
        df = df.sample(frac=args.frac)

    # Subscribe dataframe
    df = tracker.subscribe(df)

    # Drop rows with missing values
    df = df.dropna()

    # Separate features and target variable
    df = df.iloc[:, :-1]



    # Impute missing values in the numerical columns (assuming columns 1 and 2 are numerical)
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    df.iloc[:, 1:3] = imputer.fit_transform(df.iloc[:, 1:3])

    # Apply OneHotEncoder to the first column (assuming it is categorical)
    ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
    df = pd.DataFrame(ct.fit_transform(df))

    # Ensure column names are maintained or regenerated after transformation
    df.columns = [f'feature_{i}' for i in range(df.shape[1])]