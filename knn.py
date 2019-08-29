'''
Este estimador utiliza el algoritmo K neightbor
'''

import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

# Listar resultados de K entre 3 y 9 para encontrar el mejor.


def knn(X_train, y_train, X_test, y_test):
    start_time = time.time()
    results = []

    for N in range(3, 9, 1):                                            # for N=3 to 8 step 1 (por defecto step = 1)
        knn_clf = KNeighborsClassifier(n_neighbors=N)                   # Crea el clasificador con un N determinado
        # Mete N (cantidad de vecinos para el cálculo) en results y el promedio del 10 fold cross validation
        # en el vector resultado.
        results.append((N, cross_val_score(knn_clf, X_train, y_train, cv=10).mean()))

    # Imprime sorted orden descendente (reverse=true) la matriz resultados según columna score (key=lambda x: x[1])
    # y el tiempo que tardó el loop
    # Más sobre sorted: https://realpython.com/python-sort/
    print("\n", sorted(results, key=lambda x: x[1], reverse=True),
          "\n* * Elapsed time for KNN: % 5.2f" % (time.time()-start_time), "segundos\n")

'''
    # Informa accuracy sobre el set de test.
    # pone en N el ganador que es el elemento 0,0 de los results ordenados de mayor a menor (reverse=true).
    N = sorted(results, key=lambda x: x[1], reverse=True)[0][0]
    knn_clf = KNeighborsClassifier(n_neighbors=N)
    knn_clf.fit(X_train, y_train)
    print("Accuracy score: ", accuracy_score(y_test, knn_clf.predict(X_test)))
'''