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


def voting(X_train, y_train, X_test, y_test):
    log_clf = LogisticRegression(solver='lbfgs', multi_class='auto')
    rnd_clf = RandomForestClassifier(n_estimators=100)
    svm_clf = SVC(gamma='scale')

    voting_clf = VotingClassifier(
        estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)],
        voting='hard'
    )

    voting_clf.fit(X_train, y_train)

    from sklearn.metrics import accuracy_score
    for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print(clf.__class__.__name__, accuracy_score(y_test, y_pred))
