import time
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


def bagging(X_train, y_train, X_test, y_test):
    start_time = time.time()

    bag_clf = BaggingClassifier(DecisionTreeClassifier(),
                                n_estimators=500,max_samples=100, bootstrap=True, n_jobs=-1)
    bag_clf.fit(X_train, y_train)
    y_pred = bag_clf.predict(X_test)

    print("\n* * Elapsed time for Bagging: % 5.2f" % (time.time() - start_time), "segundos\n")

    y_pred = bag_clf.oob_decision_function_
    print("Accuracy score: ", accuracy_score(y_test, y_pred),"\n", bag_clf.oob_decision_function_)

