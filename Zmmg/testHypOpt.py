from sklearn.datasets import load_breast_cancer

# Classifiers
from lightgbm import LGBMClassifier

# Parallel gridsearch
from paragrid import paragrid

# spaces
space_gpdt = {'max_depth': [2, 20, 5],
              'learning_rate': [0.01, 0.1, 5],
              'n_estimators': [2, 50, 5]}
# Classification
breast_cancer = load_breast_cancer()
A, b = breast_cancer.data, breast_cancer.target
lgbm_cls = LGBMClassifier()

params = paragrid(model=lgbm_cls, space=space_gpdt, X=A, y=b)

params, results = params.gridsearch()
best_params = params.score()
