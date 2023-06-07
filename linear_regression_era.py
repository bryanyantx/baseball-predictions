import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#read csv into pandas dataframe
df = pd.read_csv("pitcherData.csv")


#features to be selected
x_cols = ['ba', 'babip', 'slg', 'woba', 'xwoba', 'xba', 'launch_speed', 'spin_rate', 'velocity']

#output (earned run average)
y_cols = ['era']

X = df[x_cols].values
y = df[y_cols].values

#split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

#calculate error and R2
mse = metrics.mean_squared_error(y_test, y_pred)
r2 = metrics.r2_score(y_test, y_pred)

print("MSE Error: ", mse)
print("R2: ", r2)