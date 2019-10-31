import numpy as np
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from hyperopt import tpe, hp, fmin
from sklearn.metrics import mean_squared_error
import xgboost as XGB
from sklearn.model_selection import cross_val_score

iris = datasets.load_iris()

x = iris.data
y = iris.target
#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y,  random_state=0)


def objective_func(args):
    if args['model'] == KNeighborsClassifier:
        n_neighbors = args['param']['n_neighbors']
        algorithm = args['param']['algorithm']
        leaf_size = args['param']['leaf_size']
        metric = args['param']['metric']
        clf = KNeighborsClassifier(n_neighbors=n_neighbors,
                                   algorithm=algorithm,
                                   leaf_size=leaf_size,
                                   metric=metric,
                                   )
        print("K")
    elif args['model'] == SVC:
        C = args['param']['C']
        kernel = args['param']['kernel']
        degree = args['param']['degree']
        gamma = args['param']['gamma']
        clf = SVC(C=C, kernel=kernel, degree=degree, gamma=gamma)
        print("S")

    elif args['model'] == XGB:
        n_estimators = args['param']['n_estimators']
        max_depth = args['param']['max_depth']
        min_child_weight = args['param']['min_child_weight']
        subsample = args['param']['subsample']
        learning_rate = args['param']['learning_rate']
        gammax = args['param']['gammax']
        colsample_bytree = args['param']['colsample_bytree']
        clf = XGB.XGBClassifier(n_estimators=n_estimators, max_depth=max_depth,
                               min_child_weight=min_child_weight, subsample=subsample,
                               learning_rate=learning_rate, gamma=gammax,
                               colsample_bytree=colsample_bytree,
                               objective='reg:squarederror', n_jobs=4)
        print("X")

    clf.fit(x_train, y_train)

    y_pred_train = clf.predict(x_train)
    loss = mean_squared_error(y_train, y_pred_train)
    print("Test Score:", clf.score(x_test, y_test))
    print("Train Score:", clf.score(x_train, y_train))
    print("\n=================")
    loss = 1 - cross_val_score(clf, x_train, y_train, cv=4).mean()
    return loss


space = hp.choice('classifier', [
    {'model': KNeighborsClassifier,
     'param': {'n_neighbors': hp.choice('n_neighbors', range(3, 11)),
               'algorithm': hp.choice('algorithm', ['ball_tree', 'kd_tree']),
               'leaf_size': hp.choice('leaf_size', range(1, 50)),
               'metric': hp.choice('metric', ["euclidean", "manhattan",
                                              "chebyshev", "minkowski"
                                              ])}
     },
    {'model': SVC,
     'param': {'C': hp.lognormal('C', 0, 1),
               'kernel': hp.choice('kernel', ['linear', 'poly', 'rbf', 'sigmoid']),
               'degree': hp.choice('degree', range(1, 15)),
               'gamma': hp.uniform('gamma', 0.001, 10000)}
     },
    {'model': XGB,
     'param': {'max_depth': hp.choice('max_depth', range(10, 30)),
               'min_child_weight': hp.quniform('min_child', 1, 20, 1),
               'subsample': hp.uniform('subsample', 0.8, 1),
               'n_estimators': hp.choice('n_estimators', range(1000, 10000, 100)),
               'learning_rate': hp.quniform('learning_rate', 0.025, 0.5, 0.025),
               'gammax': hp.quniform('gammax', 0.5, 1, 0.05),
               'colsample_bytree': hp.quniform('colsample_bytree', 0.5, 1, 0.05)}
     }
])

#best_classifier = fmin(objective_func, space, algo=tpe.suggest, max_evals=10)
best_classifier = fmin(objective_func, space, algo=tpe.suggest, max_evals=20)

print(best_classifier)

'''
hp.choice(label, options): label is a string input which refers to the hyperparameter, and options will contain a list, 
one element will be returned from the list for that particular label.

hp.uniform(label, low, high): Again the label will contain the string referring to hyperparameter and returns a value 
uniformly between low and high. And when optimizing, this variable is constrained to a two-sided interval.

hp.lognormal(label, mu, sigma): Returns a value drawn according to exp(normal(mu, sigma)) so that the logarithm of the 
return value is normally distributed. When optimizing, this variable is constrained to be positive.

100%|██████████| 100/100 [00:02<00:00, 49.12it/s, best loss: 0.0]
{'C': 0.5047755014365481, 'classifier': 1, 'degree': 1, 'gamma': 2443.4285952618266, 'kernel': 1}

100%|██████████| 1000/1000 [01:11<00:00, 13.95it/s, best loss: 0.0]
{'C': 0.626327892827402, 'classifier': 1, 'degree': 8, 'gamma': 3040.435294528072, 'kernel': 2}

https://blog.goodaudience.com/on-using-hyperopt-advanced-machine-learning-a2dde2ccece7?gi=d41c1eca6f1c
'''
