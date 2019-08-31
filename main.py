from ensemble import voting     # Levanta la funcion voting del archivo ensemple.py
from knn import knn
import time
from sklearn import datasets                            # https://scikit-learn.org/stable/datasets/index.html
from sklearn.model_selection import train_test_split
from LR import LR
from SVC import SVC1
from tpot import TPOTClassifier

iris = datasets.load_iris()     # Levanta el dataset iris que está incluido en el sklearn
print(iris.feature_names)       # 'sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'
X = iris.data[:, :2]            # Pone en X todos los elementos (:) solo de las columnas 0 y 1 (:2)
y = iris.target                 # Pone en y la columna target (es convención poner el vector clase en minúscula

# Parte el set original en train y test con Stratify, shuffle por defecto
# parametros doc en https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,  random_state=0)

# Hace un split del set 75% - training 25% - test sin estratificación.
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=7)

# ----------------------------- Predictors ------------------------------
start_time = time.time()

LR(X_train, y_train, X_test, y_test)

SVC1(X_train, y_train, X_test, y_test)

knn(X_train, y_train, X_test, y_test)

voting(X_train, y_train, X_test, y_test)

tm = (time.time() - start_time)    # Le damos al TPOT tm minutes máximo para correr y resultados commparables
print("\n* * Elapsed TOTAL time for TPOT: % 5.2f" % tm, "minutes\n")

# ------------------------------- TPOT -----------------------------------
start_time = time.time()
tpot = TPOTClassifier(max_time_mins=1, generations=3, population_size=30, verbosity=2) # poner tm
tpot.fit(X_train, y_train)
print(tpot.score(X_train, y_train))
print("\n* * Elapsed time for TPOT: % 5.2f" % (time.time()-start_time), "segundos\n")
tpot.export('tpot_pipeline.py') # Exporta resultado a un archivo .py
