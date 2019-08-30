# from sklearn.datasets import load_iris
# from sklearn.model_selection import train_test_split
# import numpy as np

from ensemble import voting     # Levanta la funcion voting del archivo ensemple.py
from knn import knn
import time
from sklearn import datasets                            # https://scikit-learn.org/stable/datasets/index.html
from sklearn.model_selection import train_test_split
from LR import LR
from SVC import SVC1
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from tpot import TPOTClassifier

iris = datasets.load_iris()     # Levanta el dataset iris que está incluido en el sklearn
print(iris.feature_names)       # 'sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'
X = iris.data[:, :2]            # Pone en X todos los elementos (:) de las columnas 0 y 1 (:2)
y = iris.target                 # Pone en y la columna target (es convención poner el vector clase en minúscula


# Stratify - se hace el mismo split pero especificando stratify=y
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)


# Hace un split del set 75% - training 25% - test sin estratificación.
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=7)

# split the training en 75% training - 25% validation sin estratificación.
# X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X_train, y_train, test_size=0.25, random_state=7)

# ----------------------------------------------------------------------------------------------------------------

# KNN de 3 y 5 con 10 fold cross-validation
# con estratificación ya generada líneas atrás
knn_3_clf = KNeighborsClassifier(n_neighbors=3)
knn_5_clf = KNeighborsClassifier(n_neighbors=5)
knn_3_scores = cross_val_score(knn_3_clf, X_train, y_train, cv=10)
knn_5_scores = cross_val_score(knn_5_clf, X_train, y_train, cv=10)

print("\n\nknn_3 mean scores: % 5.2f" % knn_3_scores.mean(), "knn_3 std: % 5.2f" % knn_3_scores.std())
print("knn_5 mean scores: % 5.2f" % knn_5_scores.mean(), " knn_5 std: % 5.2f" % knn_5_scores.std())

# ----------------------------- Predictors ------------------------------
LR(X_train, y_train, X_test, y_test)

SVC1(X_train, y_train, X_test, y_test)

knn(X_train, y_train, X_test, y_test)

voting(X_train, y_train, X_test, y_test)

''' --------------------------- AutoML ----------------------------------
automl = autosklearn.classification.AutoSklearnClassifier()
automl.fit(X_train, y_train)
y_hat = automl.predict(X_test)
print("Accuracy score", sklearn.metrics.accuracy_score(y_test, y_hat))
'''
# ------------------------------- TPOT -----------------------------------
start_time = time.time()
tpot = TPOTClassifier(generations=3, population_size=30, verbosity=2)
tpot.fit(X_train, y_train)
print(tpot.score(X_train, y_train))
print("\n* * Elapsed time for TPOT: % 5.2f" % (time.time()-start_time), "segundos\n")
tpot.export('tpot_pipeline.py') # Exporta resultado a un archivo .py
