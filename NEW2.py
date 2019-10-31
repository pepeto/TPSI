import xgboost as xgb
from sklearn.datasets import load_boston

boston = load_boston()

# XGBoost API example
params = {'tree_method': 'gpu_hist', 'max_depth': 5, 'learning_rate': 0.1}
dtrain = xgb.DMatrix(boston.data, boston.target)
xgb.train(params, dtrain, evals=[(dtrain, "train")])

# sklearn API example
gbm = xgb.XGBRegressor(silent=True, n_estimators=1000, tree_method='gpu_hist', n_jobs=6)
gbm.fit(boston.data, boston.target, eval_set=[(boston.data, boston.target)])
