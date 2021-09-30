from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.utils.validation import indexable


files = [
    ('data/lsvm-credit-input.csv', 'data/lsvm-credit-check.csv', 'credit'),
    ('data/lsvm-ion-input.csv', 'data/lsvm-ion-check.csv', 'ion'),
    ('data/lsvm-simple-input.csv', 'data/lsvm-simple-check.csv', 'simple'),
]
for i in range(0, len(files)):
    df_a = pd.read_csv(files[i][0], header=None)
    df_b = pd.read_csv(files[i][1], header=None)
    df_a.insert(len(df_a.columns), 'check', df_b[0])

    train = df_a.sample(frac=0.8,random_state=200)
    test = df_a.drop(train.index)

    train_check = train[['check']]
    test_check = test[['check']]
    train_input = train.drop('check', axis= 1)
    test_input = test.drop('check', axis = 1)

    train_check.to_csv(f'data/{files[i][2]}_check_train.csv', index = False, header=False)
    test_check.to_csv(f'data/{files[i][2]}_check_test.csv', index = False, header=False)

    train_input.to_csv(f'data/{files[i][2]}_input_train.csv', index = False, header=False)
    test_input.to_csv(f'data/{files[i][2]}_input_test.csv', index = False, header=False)
