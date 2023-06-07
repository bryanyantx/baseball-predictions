import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import KFold, GridSearchCV
from xgboost import XGBRegressor

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

model = XGBRegressor(random_state=42)

#hyperparameters to test
param_grid = {
    'eta': [0.001, 0.01, 0.1, 0.3, 0.5],
    'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'min_child_weight': [1, 2, 4, 6, 8],
    'n_estimators': [50, 100, 150, 200, 250]
}

grid = GridSearchCV(model, param_grid=param_grid, cv=kfold, scoring='neg_mean_squared_error')
grid.fit(X, y)

#get best model
best_model = grid.best_estimator_

mse_list = []
r2_list = []

for train_index, test_index in kfold.split(X):
    X_train = X[train_index]
    X_test = X[test_index]
    y_train = y[train_index]
    y_test = y[test_index]

    best_model.fit(X_train, y_train)

    y_pred = best_model.predict(X_test)
    mse_list.append(metrics.mean_squared_error(y_test, y_pred))
    r2_list.append(metrics.r2_score(y_test, y_pred))

print("Average Error: ", np.mean(mse_list))
print("Average R2: ", np.mean(r2_list)) 