import pandas as pd




def run_pipeline(args, tracker) -> None:

    input_path = args.dataset

    df = pd.read_csv(input_path, header=0)

    if args.frac != 0.0:
        df = df.sample(frac=args.frac)

    # Subscribe dataframe
    df = tracker.subscribe(df)


    cols = ['Name', 'Ticket', 'Cabin']
    df = df.drop(cols, axis = 1)
    
    

    df = df.dropna()
    
    

    cols = ['Pclass', 'Sex', 'Embarked']
    tracker.dataframe_tracking = False #to have the missing link for now
    for i,col in enumerate(cols):
            
        dummies = pd.get_dummies(df[col])
        df_dummies = dummies.add_prefix(col + '_')
        df = df.join(df_dummies)
        
        df = df.drop([col], axis=1)
