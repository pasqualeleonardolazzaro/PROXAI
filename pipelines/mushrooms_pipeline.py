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

    input_path = 'C:/Users/paleo/PycharmProjects/DPDS_LLM/datasets/mushrooms.csv'

    df = pd.read_csv(input_path, sep=';', index_col=False)

    if args.frac != 0.0:
        df = df.sample(frac=args.frac)

    # Subscribe dataframe
    df = tracker.subscribe(df)

    df = df.drop(["does-bruise-or-bleed", "gill-attachment", "gill-color", "stem-root", "stem-surface",
                  "stem-color", "veil-type", "veil-color", "has-ring", "ring-type", "spore-print-color", "habitat",
                  "gill-spacing"], axis=1)





    # Assign 1 if class is 'e', 0 otherwise
    df['class'] = df['class'].replace({'e': 1, 'p': 0})


    df = df.dropna()


    df = df.replace({'cap-color': {'n': 1, 'b': 2, 'g': 3,
                                   'r': 3, 'p': 4, 'u': 5, 'e': 6, 'w': 7, 'y': 8,
                                   'l': 9, 'o': 10, 'k': 11},
                     'cap-shape': {'b': 1, 'c': 2, 'x': 3, 'f': 4, 's': 5, 'p': 6, 'o': 7},
                     'cap-surface': {'i': 1, 'g': 2, 'y': 3, 's': 4, 'h': 5, 'l': 6, 'k': 7, 't': 8,
                                     'w': 9, 'e': 10, },
                     'season': {'s': 1, 'u': 2, 'a': 3, 'w': 4}})

