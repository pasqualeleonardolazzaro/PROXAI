import pandas as pd

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
    
    # Subscribe dataframe
    df = tracker.subscribe(df)
    tracker.analyze_changes(df)

    # Separate features and target variable
    df = df.iloc[:, :-1]
    tracker.analyze_changes(df)

    # Impute missing values in the numerical column
    df['Age'].fillna(df['Age'].mean(), inplace=True)
    tracker.analyze_changes(df)

    print("Finished")