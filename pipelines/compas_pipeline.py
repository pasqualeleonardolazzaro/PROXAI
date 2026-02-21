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

    df = pd.read_csv(input_path, header=0)

    if args.frac != 0.0:
        df = df.sample(frac=args.frac)    
    
    # Subscribe dataframe
    df = tracker.subscribe(df)



    columns = ['age', 'c_charge_degree', 'race', 'sex', 'priors_count', 'days_b_screening_arrest', 'two_year_recid', 'c_jail_in', 'c_jail_out']
    df = df.drop(df.columns.difference(columns), axis=1)



    df = df.dropna()


    df['race'] = [0 if r != 'Caucasian' else 1 for r in df['race']]


    
    df = df.rename({'two_year_recid': 'label'}, axis=1)

	# Reverse label for consistency with function defs: 1 means no recid (good), 0 means recid (bad)
    df['label'] = [0 if l == 1 else 1 for l in df['label']]



    df['jailtime'] = (pd.to_datetime(df.c_jail_out) - pd.to_datetime(df.c_jail_in)).dt.days



    df = df.drop(['c_jail_in', 'c_jail_out'], axis=1)
	


	# M: misconduct, F: felony
    df['c_charge_degree'] = [0 if s == 'M' else 1 for s in df['c_charge_degree']]


