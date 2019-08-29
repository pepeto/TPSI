'''
Este approach se basa en usar clasificadores diferentes y que el resultado sea
mediante una votación de cada uno de ellos. Para que esto funcione bien, los
clasificadores deben ser lo más disimiles posible, es decir no colineales, para que
no tengan errores en los mismos lugares aunque es difícil ya que todos se entrenan en el
mismo set.

En este caso se usa Random forest, Logistic regression y SVM

'''

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

import time

def voting(X_train, y_train, X_test, y_test):
    start_time = time.time()
    # Instancian cada uno de los clasificadores/estimadores con sus respectivos hiper parámetros
    log_clf = LogisticRegression(solver='lbfgs', multi_class='auto')
    rnd_clf = RandomForestClassifier(n_estimators=100)
    # SVC con probability=True hace cross validation y habilita poder usar hard en voting
    svm_clf = SVC(gamma='scale', probability=True)

    # Instancia un VotingClassifier que tiene como parámetros los objetos anteriormente instanciados (clasificadores)
    voting_clf = VotingClassifier(
        estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)],
        voting='soft'   # opción hard cambia calculo de probs (soft necesita SVC con probability=True)
    )

    # Entrena el clasificador de igual forma que si fuese uno solo mediante el método fit
    voting_clf.fit(X_train, y_train)

    # Corre independientemente cada clasificador para comparar resultados
    for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print(clf.__class__.__name__, ": %5.2f" % accuracy_score(y_test, y_pred))

    print("\n* * Elapsed time for Ensemble: % 5.2f" % (time.time()-start_time), "segundos\n")