# from sklearn.datasets import load_iris
# from sklearn.model_selection import train_test_split
# import numpy as np

from ensemble import voting     # Levanta la funcion voting del archivo ensemple.py

from sklearn import datasets                            # https://scikit-learn.org/stable/datasets/index.html
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# load the classifying models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from tpot import TPOTClassifier

iris = datasets.load_iris()     # Levanta el dataset iris que está incluido en el sklearn
print(iris.feature_names)       # 'sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'
X = iris.data[:, :2]            # Pone en X todos los elementos (:) de las columnas 0 y 1 (:2)
y = iris.target                 # Pone en y la columna target (es convención poner el vector clase en minúscula


# Hace un split del set 75% - training 25% - test sin estratificación.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=7)

# split the training en 75% training - 25% validation sin estratificación.
X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X_train, y_train, test_size=0.25, random_state=7)

# Instancia objeto "Support Vector Classifier" con kernel linear y seed random=7 (para shuffle de data)
# Más sobre Kernels: https://www.youtube.com/watch?v=OmTu0fqUsQk
# https://chrisalbon.com/machine_learning/support_vector_machines/svc_parameters_using_rbf_kernel/
svc_clf = SVC(kernel='linear', random_state=7)  # kernel ‘linear’, ‘poly’, ‘rbf’ (default), ‘sigmoid’, ‘precomputed’
svc_clf.fit(X_train_2, y_train_2)               # Genera el modelo sobre los datos de training
svc_pred = svc_clf.predict(X_test_2)            # Predicción sobre set de test

# Impresión de resultados % 5.2f" % formatea el float en 5 posiciones entero y dos decimales.
print("Accuracy of SVC (75-25): % 5.2f" % (accuracy_score(y_test_2, svc_pred)))   # del SVC score de real sobre predicho
# Accuracy calculado sobre los datos originales.
print("Accuracy of SVC on original Test Set: % 5.2f" % accuracy_score(y_test, svc_clf.predict(X_test)))


# Insta un objeto "Logistic Regression" y genera el modelo en una sola línea.
lr_clf = LogisticRegression(multi_class='auto', solver='liblinear', random_state=7).fit(X_train_2, y_train_2)
lr_pred = lr_clf.predict(X_test_2)    # Predicción sobre set de test
# Impresión de resultados
print("Accuracy of LR (75-25): % 5.2f" % accuracy_score(y_test_2, lr_pred))     # del LR score de real sobre predicho


# Cross Validation - se le pasa el clasificador instanciado, el set de datos y el número de pedazos (4)
svc_scores = cross_val_score(svc_clf, X_train, y_train, cv=4)

print("\nAverage SVC scores (4 fold): % 5.2f" % svc_scores.mean())     # imprime el promedio de las 4 corridas
print("Standard Deviation of SVC scores: % 5.2f" % svc_scores.std())   # desviación estándar

lr_scores = cross_val_score(lr_clf, X_train, y_train, cv=4)     # lo mismo pero con Logistic Regression
print("\nAverage LR scores (4 fold): % 5.2f" % lr_scores.mean())
print("Standard Deviation of LR scores: % 5.2f" % lr_scores.std())

# Stratify - se hace el mismo split pero especificando stratify=y
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)

svc_scores = cross_val_score(svc_clf, X_train, y_train, cv=4)   # Se corre nuevamente el 4 fold cross val
print("\nAverage SVC scores (stratified): % 5.2f" % svc_scores.mean())
print("Standard Deviation of SVC scores: % 5.2f" % svc_scores.std())
print("Score on Final Test Set (stratified): % 5.2f" % accuracy_score(y_test, svc_clf.predict(X_test)))

'''
The preceding code is equivalent to:
from sklearn.model_selection import cross_val_score, StratifiedKFold
skf = StratifiedKFold(n_splits = 4)
svc_scores = cross_val_score(svc_clf, X_train, y_train, cv = skf)
'''

# KNN de 3 y 5 con 10 fold cross-validation
# con estratificación ya generada líneas atrás
knn_3_clf = KNeighborsClassifier(n_neighbors=3)
knn_5_clf = KNeighborsClassifier(n_neighbors=5)
knn_3_scores = cross_val_score(knn_3_clf, X_train, y_train, cv=10)
knn_5_scores = cross_val_score(knn_5_clf, X_train, y_train, cv=10)

print("\n\nknn_3 mean scores: % 5.2f" % knn_3_scores.mean(), "knn_3 std: % 5.2f" % knn_3_scores.std())
print("knn_5 mean scores: % 5.2f" % knn_5_scores.mean(), " knn_5 std: % 5.2f" % knn_5_scores.std())

# Listar resultados de K entre 3 y 9 para encontrar el mejor.
results = []
for N in range(3, 9, 1):                                            # for x=3 to 8 step 1 (por defecto step = 1)
    knn_clf = KNeighborsClassifier(n_neighbors=N)                   # Crea el clasificador con un N determinado
    # Mete N (cantidad de vecinos para el cálculo) en results y el promedio del 10 fold cross validation
    results.append((N, cross_val_score(knn_clf, X_train, y_train, cv=10).mean()))
# Imprime sorteado en orden descendente (reverse=true) la matriz resultados según columna score (key=lambda x: x[1])
# Más sobre sorted: https://realpython.com/python-sort/
print('\n', sorted(results, key=lambda x: x[1], reverse=True))

'''
automl = autosklearn.classification.AutoSklearnClassifier()
automl.fit(X_train, y_train)
y_hat = automl.predict(X_test)
print("Accuracy score", sklearn.metrics.accuracy_score(y_test, y_hat))
'''

'''
tpot = TPOTClassifier(generations=3, population_size=50, verbosity=2)
tpot.fit(X_train, y_train)
print(tpot.score(X_train, y_train))
# tpot.export('tpot_iris_pipeline.py')
'''

voting(X_train, y_train, X_test, y_test)