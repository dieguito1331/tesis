# -*- coding: utf-8 -*-

from collections import defaultdict

import pandas as pd
from sklearn.metrics import mean_absolute_error

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier
from Perceptron import BinaryMLP


def encode(train, c, t):
    default_ratio = (train[t] > 0).mean()
    general_var = (train[t] > 0).var()
    ct = pd.crosstab(train[c], train[t] > 0)
    ct.columns = [0, 1]
    ct["n"] = ct.sum(axis=1)
    ct["p"] = ct[1] / ct.n
    temp = train.groupby(c)[t].var().fillna(0)
    temp.name = "local_var"
    ct = pd.concat([ct, temp], axis=1)
    ct["pond"] = (ct.n * ct.local_var) / (general_var + ct.n * ct.local_var)
    ct["final"] = ct.pond * ct.p + (1 - ct.pond) * default_ratio
    return defaultdict(lambda: default_ratio, ct.final.to_dict())


def train_es(learner, X_train, y_train, X_test, y_test, esr=20):
    if type(learner) in [XGBClassifier, BinaryMLP]:
        learner.fit(X_train, y_train,
                    eval_set=[(X_test, y_test)],
                    early_stopping_rounds=esr,
                    verbose=True)
        n_estimators = learner.best_iteration
        probs = learner.predict_proba(X_test, ntree_limit=n_estimators)[:, -1]
    elif type(learner) in [RandomForestClassifier, ExtraTreesClassifier,
                           AdaBoostClassifier, BaggingClassifier]:
        max_n_estimators = learner.n_estimators
        learner.set_params(warm_start = True)
        learner.set_params(n_estimators = 50)
        old_res = 1
        new_res = 0.99999999
        while (new_res < old_res) and \
              (learner.n_estimators + esr <= max_n_estimators):
            learner.set_params(n_estimators = learner.n_estimators + esr)
            old_res = new_res
            learner.fit(X_train, y_train)
            probs = learner.predict_proba(X_test)[:, -1]
            new_res = mean_absolute_error(y_test.values, probs >= 0.5)
            print("resultado con {0} estimadores: {1}".format(learner.n_estimators,
                                                              new_res))
        n_estimators = learner.n_estimators
        if (new_res > old_res):
            n_estimators -= esr
    else:
        learner.fit(X_train, y_train)
        if hasattr(learner, "predict_proba"):
            probs = learner.predict_proba(X_test)[:, -1]
        else:
            probs = learner.decision_function(X_test)
        n_estimators = 1
    return n_estimators, probs
