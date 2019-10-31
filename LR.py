import time
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

def LR(X_train, y_train, X_test, y_test):

    # Insta un objeto "Logistic Regression" y genera el modelo en una sola línea.
    lr_clf = LogisticRegression(multi_class='auto', solver='liblinear', random_state=7)

    start_time = time.time()

    lr_scores = cross_val_score(lr_clf, X_train, y_train, cv=4)  # con Cross Validation
    print("\nLR (4 fold) Std Dev: %5.2f - Mean Acc: %5.2f - Elap: %5.2f" %
          (lr_scores.std(), lr_scores.mean(), (time.time() - start_time)))

'''
class sklearn.linear_model.LogisticRegression(penalty=’l2’, dual=False, tol=0.0001, C=1.0, fit_intercept=True, 
intercept_scaling=1, class_weight=None, random_state=None, solver=’warn’, max_iter=100, multi_class=’warn’, 
verbose=0, warm_start=False, n_jobs=None, l1_ratio=None)

Parameters
penalty [str, ‘l1’, ‘l2’, ‘elasticnet’ or ‘none’, optional (default=’l2’)] Used to specify the norm
used in the penalization. The ‘newton-cg’, ‘sag’ and ‘lbfgs’ solvers support only l2 penalties.
‘elasticnet’ is only supported by the ‘saga’ solver. If ‘none’ (not supported by the liblinear solver), 
no regularization is applied. New in version 0.19: l1 penalty with SAGA solver (allowing ‘multinomial’ + L1)

dual [bool, optional (default=False)] Dual or primal formulation. Dual formulation is only
implemented for l2 penalty with liblinear solver. Prefer dual=False when n_samples >
n_features.

tol [float, optional (default=1e-4)] Tolerance for stopping criteria.

C [float, optional (default=1.0)] Inverse of regularization strength; must be a positive float. Like
in support vector machines, smaller values specify stronger regularization.

fit_intercept [bool, optional (default=True)] Specifies if a constant (a.k.a. bias or intercept)
should be added to the decision function.

intercept_scaling [float, optional (default=1)] Useful only when the solver ‘liblinear’ is used
and self.fit_intercept is set to True. In this case, x becomes [x, self.intercept_scaling],
i.e. a “synthetic” feature with constant value equal to intercept_scaling is appended
to the instance vector. The intercept becomes intercept_scaling *
synthetic_feature_weight.
Note! the synthetic feature weight is subject to l1/l2 regularization as all other features. To
lessen the effect of regularization on synthetic feature weight (and therefore on the intercept)
intercept_scaling has to be increased.

class_weight [dict or ‘balanced’, optional (default=None)] Weights associated with classes in
the form {class_label: weight}. If not given, all classes are supposed to have
weight one.

Logistic Regression (aka logit, MaxEnt) classifier.
In the multiclass case, the training algorithm uses the one-vs-rest (OvR) scheme if the ‘multi_class’ option is
set to ‘ovr’, and uses the cross-entropy loss if the ‘multi_class’ option is set to ‘multinomial’. (Currently the
‘multinomial’ option is supported only by the ‘lbfgs’, ‘sag’, ‘saga’ and ‘newton-cg’ solvers.)
This class implements regularized logistic regression using the ‘liblinear’ library, ‘newton-cg’, ‘sag’, ‘saga’ and
‘lbfgs’ solvers. Note that regularization is applied by default. It can handle both dense and sparse input. Use
C-ordered arrays or CSR matrices containing 64-bit floats for optimal performance; any other input format will
be converted (and copied).
The ‘newton-cg’, ‘sag’, and ‘lbfgs’ solvers support only L2 regularization with primal formulation, or no regularization.
The ‘liblinear’ solver supports both L1 and L2 regularization, with a dual formulation only for the
L2 penalty. The Elastic-Net regularization is only supported by the ‘saga’ solver.

'''