import argparse
import pandas as pd
import numpy as np


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
    frac = args.frac

    if frac > 0.0 and frac < 1.0:
        df = df.sample(frac=frac)
    elif frac > 1.0:
        df = pd.concat([df] * int(frac), ignore_index=True)

    # Subscribe dataframe
    df = tracker.subscribe(df)


    df = df.replace({'checking': {'A11': 'check_low', 'A12': 'check_mid', 'A13': 'check_high',
                                  'A14': 'check_none'},
                     'credit_history': {'A30': 'debt_none', 'A31': 'debt_noneBank',
                                        'A32': 'debt_onSchedule', 'A33': 'debt_delay',
                                        'A34': 'debt_critical'},
                     'purpose': {'A40': 'pur_newCar', 'A41': 'pur_usedCar',
                                 'A42': 'pur_furniture', 'A43': 'pur_tv',
                                 'A44': 'pur_appliance', 'A45': 'pur_repairs',
                                 'A46': 'pur_education', 'A47': 'pur_vacation',
                                 'A48': 'pur_retraining', 'A49': 'pur_business',
                                 'A410': 'pur_other'},
                     'savings': {'A61': 'sav_small', 'A62': 'sav_medium', 'A63': 'sav_large',
                                 'A64': 'sav_xlarge', 'A65': 'sav_none'},
                     'employment': {'A71': 'emp_unemployed', 'A72': 'emp_lessOne',
                                    'A73': 'emp_lessFour', 'A74': 'emp_lessSeven',
                                    'A75': 'emp_moreSeven'},
                     'other_debtors': {'A101': 'debtor_none', 'A102': 'debtor_coApp',
                                       'A103': 'debtor_guarantor'},
                     'property': {'A121': 'prop_realEstate', 'A122': 'prop_agreement',
                                  'A123': 'prop_car', 'A124': 'prop_none'},
                     'other_inst': {'A141': 'oi_bank', 'A142': 'oi_stores', 'A143': 'oi_none'},
                     'housing': {'A151': 'hous_rent', 'A152': 'hous_own', 'A153': 'hous_free'},
                     'job': {'A171': 'job_unskilledNR', 'A172': 'job_unskilledR',
                             'A173': 'job_skilled', 'A174': 'job_highSkill'},
                     'phone': {'A191': 0, 'A192': 1},
                     'foreigner': {'A201': 1, 'A202': 0},
                     'label': {2: 0}})




    status_mapping = {
        'A91': 'divorced',
        'A92': 'divorced',
        'A93': 'single',
        'A95': 'single'
    }

    df['status'] = df['personal_status'].map(status_mapping).fillna('married')

    # Translate gender values
    df['personal_status'] = np.where(df.personal_status == 'A92', 0, np.where(df.personal_status == 'A95', 0, 1))

    df = df.drop(['personal_status'], axis=1)



    columns = ['checking', 'credit_history', 'purpose', 'savings', 'employment', 'other_debtors', 'property', 'other_inst', 'housing', 'job']

    for i, col in enumerate(columns):
        dummies = pd.get_dummies(df[col])
        df_dummies = dummies.add_prefix(col + '_')
        df = df.join(df_dummies)
        df = df.drop([col], axis=1)

