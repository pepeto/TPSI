import time
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

def SVC1(X_train, y_train, X_test, y_test):

    # Instancia objeto "Support Vector Classifier" con kernel linear y seed random=7 (para shuffle de data)
    # Más sobre Kernels: https://www.youtube.com/watch?v=OmTu0fqUsQk
    # https://chrisalbon.com/machine_learning/support_vector_machines/svc_parameters_using_rbf_kernel/
    svc_clf = SVC(kernel='linear', random_state=7)  # kernel ‘linear’, ‘poly’, ‘rbf’ (default), ‘sigmoid’, ‘precomputed’
    svc_clf.fit(X_train, y_train)               # Genera el modelo sobre los datos de training
    svc_pred = svc_clf.predict(X_test)            # Predicción sobre set de test

    # Impresión de resultados % 5.2f" % formatea el float en 5 posiciones entero y dos decimales.
    print("Accuracy of SVC (75-25): % 5.2f" % (accuracy_score(y_test, svc_pred))) # del SVC score de real sobre predicho
    # Accuracy calculado sobre los datos originales.
    print("Accuracy of SVC on original Test Set: % 5.2f" % accuracy_score(y_test, svc_clf.predict(X_test)))

    start_time = time.time()

    # Cross Validation - se le pasa el clasificador instanciado, el set de datos y el número de pedazos (4)
    svc_scores = cross_val_score(svc_clf, X_train, y_train, cv=4)

    print("\nAverage SVC scores (4 fold): % 5.2f" % svc_scores.mean())  # imprime el promedio de las 4 corridas
    print("Standard Deviation of SVC scores: % 5.2f" % svc_scores.std())  # desviación estándar

    print("\n* * Elapsed time for Ensemble: % 5.2f" % (time.time() - start_time), "segundos\n")

'''
class sklearn.svm.LinearSVC(penalty=’l2’, loss=’squared_hinge’, dual=True, tol=0.0001, C=1.0, multi_class=’ovr’, 
fit_intercept=True, intercept_scaling=1, class_weight=None, verbose=0, random_state=None, max_iter=1000)

Linear Support Vector Classification.
Similar to SVC with parameter kernel=’linear’, but implemented in terms of liblinear rather than libsvm, so it
has more flexibility in the choice of penalties and loss functions and should scale better to large numbers of
samples. This class supports both dense and sparse input and the multiclass support is handled according to a 
one-vs-therest scheme.

Parameters

penalty [string, ‘l1’ or ‘l2’ (default=’l2’)] Specifies the norm used in the penalization. The ‘l2’
penalty is the standard used in SVC. The ‘l1’ leads to coef_ vectors that are sparse.

loss [string, ‘hinge’ or ‘squared_hinge’ (default=’squared_hinge’)] Specifies the loss function.
‘hinge’ is the standard SVM loss (used e.g. by the SVC class) while ‘squared_hinge’ is the
square of the hinge loss.

dual [bool, (default=True)] Select the algorithm to either solve the dual or primal optimization
problem. Prefer dual=False when n_samples > n_features.

tol [float, optional (default=1e-4)] Tolerance for stopping criteria.

C [float, optional (default=1.0)] Penalty parameter C of the error term.

multi_class [string, ‘ovr’ or ‘crammer_singer’ (default=’ovr’)] Determines the multi-class
strategy if y contains more than two classes. "ovr" trains n_classes one-vs-rest classifiers,
while "crammer_singer" optimizes a joint objective over all classes. While
crammer_singer is interesting from a theoretical perspective as it is consistent, it is seldom
used in practice as it rarely leads to better accuracy and is more expensive to compute.
If "crammer_singer" is chosen, the options loss, penalty and dual will be ignored.

fit_intercept [boolean, optional (default=True)] Whether to calculate the intercept for this
model. If set to false, no intercept will be used in calculations (i.e. data is expected tobe already centered).
intercept_scaling [float, optional (default=1)] When self.fit_intercept is True, instance vector
x becomes [x, self.intercept_scaling], i.e. a “synthetic” feature with constant
value equals to intercept_scaling is appended to the instance vector. The intercept becomes
intercept_scaling * synthetic feature weight Note! the synthetic feature weight is subject to
l1/l2 regularization as all other features. To lessen the effect of regularization on synthetic
feature weight (and therefore on the intercept) intercept_scaling has to be increased.

class_weight [{dict, ‘balanced’}, optional] Set the parameter C of class i to class_weight[i]*C for SVC. If not given, 
all classes are supposed to have weight one. The “balanced” mode uses the values of y to automatically adjust weights 
inversely proportional to class frequencies in the input data as n_samples / (n_classes *np.bincount(y))

verbose [int, (default=0)] Enable verbose output. Note that this setting takes advantage of a perprocess
runtime setting in liblinear that, if enabled, may not work properly in a multithreaded
context.

random_state [int, RandomState instance or None, optional (default=None)] The seed of the
pseudo random number generator to use when shuffling the data for the dual coordinate
descent (if dual=True). When dual=False the underlying implementation of
LinearSVC is not random and random_state has no effect on the results. If int, random_
state is the seed used by the random number generator; If RandomState instance, random_
state is the random number generator; If None, the random number generator is the
RandomState instance used by np.random.

max_iter [int, (default=1000)] The maximum number of iterations to be run.

Attributes

coef_ [array, shape = [n_features] if n_classes == 2 else [n_classes, n_features]] Weights assigned
to the features (coefficients in the primal problem). This is only available in the case
of a linear kernel.
coef_ is a readonly property derived from raw_coef_ that follows the internal memory
layout of liblinear.

intercept_ [array, shape = [1] if n_classes == 2 else [n_classes]] Constants in decision function.
'''