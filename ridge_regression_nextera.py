import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge

#read csv into pandas dataframe
df = pd.read_csv("pitcherData.csv")

#drop NaN values
df = df.dropna()

#features
x_cols = ['ba', 'babip', 'slg', 'woba', 'xwoba', 'xba', 'launch_speed', 'spin_rate', 'velocity']

#output (earned run average)
y_cols = ['next_era']

X = df[x_cols].values
y = df[y_cols].values

#use 5-fold cross validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

alphas = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]

#loop through different alpha values
for alpha in alphas:
    model = Ridge(alpha=alpha)

    #keep track of error and R2 scores
    mse_list = []
    r2_list = []

    #cross validation
    for train_index, test_index in kfold.split(X):
        X_train = X[train_index]
        X_test = X[test_index]
        y_train = y[train_index]
        y_test = y[test_index]

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        mse_list.append(metrics.mean_squared_error(y_test, y_pred))
        r2_list.append(metrics.r2_score(y_test, y_pred))
    
    print("Alpha: ", alpha)
    print("Average Error: ", np.mean(mse_list))
    print("Average R2: ", np.mean(r2_list)) 

