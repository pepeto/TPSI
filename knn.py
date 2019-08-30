'''
Este estimador utiliza el algoritmo K neightbor
'''

import time
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV


# Listar resultados de K entre 3 y 9 para encontrar el mejor.


def knn(X_train, y_train, X_test, y_test):
    start_time = time.time()


    # Documentacion grid_search https://scikit-learn.org/0.17/modules/generated/sklearn.grid_search.GridSearchCV.html
    # param_grid = [{'n_neighbors': [3, 4, 5, 6, 7, 8, 9]}]
    param_grid = [{'n_neighbors': range(3, int(len(X_train)/2))}]

    # grid_search recibe: clasificador, parametros a variar, crossVal, scoring
    grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=10, iid=False) # scoring='neg_mean_squared_error',
    grid_search.fit(X_train, y_train)

    print("BEST Estimator: ", grid_search.best_estimator_, "\nScore:  ", grid_search.best_score_,
          " Params: ", grid_search.best_params_,"\n",
          "* Elapsed time for KNN: % 5.2f" % (time.time()-start_time))


    start_time = time.time()

    # Plot
    plt.scatter(range(3, int(len(X_train) / 2)), grid_search.cv_results_["mean_test_score"], c=("blue"), alpha=0.5)
    plt.title('Accuracy vs. N')
    plt.xlabel('N')
    plt.ylabel('Accuracy')
    plt.show()

    results = []

    for N in range(3, int(len(X_train)/2), 1): # for N=3 to 8 step 1 (por defecto step = 1)
        knn_clf = KNeighborsClassifier(n_neighbors=N)   # Crea el clasificador con un N determinado
        # Mete N (cantidad de vecinos para el cálculo) en results y el promedio del 10 fold cross validation
        # en el vector resultado.
        results.append((N, cross_val_score(knn_clf, X_train, y_train, cv=10).mean()))

    # Imprime sorted orden descendente (reverse=true) la matriz resultados según columna score (key=lambda x: x[1])
    # y el tiempo que tardó el loop
    # Más sobre sorted: https://realpython.com/python-sort/
    print("\n", sorted(results, key=lambda x: x[1], reverse=True),
          "\n* * Elapsed time for KNN: % 5.2f" % (time.time()-start_time), "segundos\n")

''' Otra manera de ver los resultados iterando sobre un zip de vectores 
    cvres = grid_search.cv_results_
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        #print(np.sqrt(-mean_score), params)
        print(mean_score, params)
'''

'''
    from sklearn.metrics import accuracy_score
    # Informa accuracy sobre el set de test.
    # pone en N el ganador que es el elemento 0,0 de los results ordenados de mayor a menor (reverse=true).
    N = sorted(results, key=lambda x: x[1], reverse=True)[0][0]
    knn_clf = KNeighborsClassifier(n_neighbors=N)
    knn_clf.fit(X_train, y_train)
    print("Accuracy score: ", accuracy_score(y_test, knn_clf.predict(X_test)))
'''

''' 
Documentacion del gridsearch 
https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
'''