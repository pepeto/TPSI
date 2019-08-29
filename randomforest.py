import time
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier


def random_forest(X_train, y_train, X_test, y_test):
    start_time = time.time()
    rnd_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, n_jobs=-1)
    rnd_clf.fit(X_train, y_train)
    y_pred_rf = rnd_clf.predict(X_test)

    bag_clf = BaggingClassifier(DecisionTreeClassifier(splitter="random", max_leaf_nodes=16),
                                n_estimators=500, max_samples=1.0, bootstrap=True, n_jobs=-1)

    bag_clf.fit(X_train, y_train)
    y_pred = bag_clf.predict(X_test)

    print("\n* * Elapsed time for Random Forest Classifier: % 5.2f" % (time.time() - start_time), "segundos\n")

    y_pred = bag_clf.oob_decision_function_
    print("Accuracy score: ", accuracy_score(y_test, y_pred),"\n", bag_clf.oob_decision_function_)


