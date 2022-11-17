import lightgbm
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

X, y = make_regression()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = lightgbm.LGBMRegressor(max_depth=1, learning_rate=0.0001, nrounds=10)
model = model.fit(X_train, y_train, eval_set=[(X_test, y_test)])

print(model.booster_.num_trees())

