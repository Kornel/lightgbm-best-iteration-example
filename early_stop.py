import lightgbm
from lightgbm.callback import early_stopping
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

X, y = make_regression()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

N = 10
param_names = ["num_iteration", 
               "n_iter", 
               "num_tree",
               "num_trees",
               "num_round", 
               "num_rounds", 
               "nrounds", 
               "num_boost_round", 
               "n_estimators", 
               "max_iter", 
               "num_iterations"]

for param_name in param_names:
    callbacks = [lightgbm.callback.early_stopping(stopping_rounds=5, verbose=False)]
    kwargs = {param_name: N}
    model = lightgbm.LGBMRegressor(max_depth=1, learning_rate=0.0001, **kwargs)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], callbacks=callbacks, verbose=False)

    print(f"For {param_name}={N} best iteration is {model.best_iteration_}, num trees {model.booster_.num_trees()}")

